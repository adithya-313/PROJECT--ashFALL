# File: tests/integration/test_account_stream_end_to_end.py
# Purpose: End-to-end integration test proving outbox relay, Kafka ingestion, and streaming account updater.
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Integration test connecting real PostgreSQL (outbox), Redpanda (Kafka upi.payments topic),
and Redis (feature cache). Proves that recorded cross-bank payment sides are relayed to Kafka,
consumed by run_payment_updater, and stored with strict per-bank and per-account isolation.
Kafka gives no per-account ordering here; state is correct in any order because every write is idempotent
(see the shuffle test).

WHY: Verifies full-stack integration across transactional database storage, distributed messaging,
and sliding-window Redis cache with idempotency guarantees under duplicate event delivery.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
import redis

from api.services.outbox_relay import relay_one_batch
from api.v1.upi_outbox import UpiPaymentSide, record_upi_side
from core.config import config
from features.account_keys import namespace_pattern
from features.account_reader import empty_snapshot, read_account_snapshot
from features.account_stream import run_payment_updater
from showcase.kafka_tools import create_topic, delete_topic, make_producer, make_topic_name

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


def host_side_redis_url() -> str:
    """
    Resolve the Redis connection string for tests running on the host machine.

    The .env Redis address uses the Docker service name ('redis'), which only resolves
    inside Docker containers. On the host runner, this swaps 'redis' for 'localhost'.
    """
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


def _dump_account_keys(client: redis.Redis, run_id: str) -> dict[bytes, Any]:
    """Helper dumping all Redis keys, types, and values under the run namespace."""
    dump: dict[bytes, Any] = {}
    pattern = f"acctf:{run_id}:*"
    for k in sorted(client.scan_iter(match=pattern)):
        k_type = client.type(k)
        if k_type == b"zset":
            dump[k] = (k_type, client.zrange(k, 0, -1, withscores=True))
        elif k_type == b"hash":
            dump[k] = (k_type, client.hgetall(k))
        else:
            dump[k] = (k_type, client.get(k))
    return dump


async def test_account_stream_end_to_end(isolated_schema: tuple[Any, str]) -> None:
    """
    Records two sides of a cross-bank payment, relays them to a throwaway Kafka topic,
    runs the streaming updater, and proves bank-level isolation and replay idempotency.
    """
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex[:16]
    payment_id = f"{run_id}-000001"

    # 1. Record debit and credit sides via real atomic outbox writer
    debit_side = UpiPaymentSide(
        payment_id=payment_id,
        bank_id="bank_0",
        side="debit",
        account_id="acc_debit_0",
        counterparty_bank_id="bank_1",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_1",
        amount_paise=150000,
        event_time=now,
        status="success",
        counterparty_vpa="user1@bank1",
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint="fp_debit_101",
        sender_os_generation=2,
    )
    credit_side = UpiPaymentSide(
        payment_id=payment_id,
        bank_id="bank_1",
        side="credit",
        account_id="acc_credit_1",
        counterparty_bank_id="bank_0",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_1",
        amount_paise=150000,
        event_time=now,
        status="success",
        counterparty_vpa="user0@bank0",
        payee_type="person",
        channel="upi_app",
    )

    await record_upi_side(connection, debit_side)
    await record_upi_side(connection, credit_side)

    # 2. Relay outbox records to throwaway Kafka topic
    topic = make_topic_name(prefix="acctf")
    await create_topic(topic)
    producer = await make_producer()
    redis_client = redis.Redis.from_url(host_side_redis_url())

    try:
        # Relay all pending outbox records to Kafka
        while True:
            res = await relay_one_batch(connection, producer, batch_size=10, topic=topic)
            if res.waiting == 0:
                break

        # 3. Stream from Kafka into Redis account state
        processed = await run_payment_updater(
            redis_client,
            group_id=f"grp-{run_id}",
            idle_timeout_seconds=2.0,
            topic=topic,
        )
        assert processed == 2

        # 4. Verify per-bank and per-account isolation in Redis
        query_time = datetime.now(timezone.utc)
        # (a) Debit side stored exclusively under bank_0 for acc_debit_0
        snap_debit = read_account_snapshot(redis_client, run_id, "bank_0", "acc_debit_0", query_time, query_time)
        assert snap_debit.has_history is True
        assert len(snap_debit.side_records) == 1
        assert snap_debit.side_records[0].side == "debit"
        assert snap_debit.side_records[0].account_id == "acc_debit_0"
        assert snap_debit.side_records[0].sender_device_fingerprint == "fp_debit_101"

        snap_debit_other = read_account_snapshot(
            redis_client, run_id, "bank_0", "acc_credit_1", query_time, query_time
        )
        assert snap_debit_other == empty_snapshot()

        # (b) Credit side stored exclusively under bank_1 for acc_credit_1
        snap_credit = read_account_snapshot(redis_client, run_id, "bank_1", "acc_credit_1", query_time, query_time)
        assert snap_credit.has_history is True
        assert len(snap_credit.side_records) == 1
        assert snap_credit.side_records[0].side == "credit"
        assert snap_credit.side_records[0].account_id == "acc_credit_1"

        snap_credit_other = read_account_snapshot(
            redis_client, run_id, "bank_1", "acc_debit_0", query_time, query_time
        )
        assert snap_credit_other == empty_snapshot()

        # 5. Replay idempotency: re-publishing a message leaves Redis state identical
        dump_before = _dump_account_keys(redis_client, run_id)
        assert len(dump_before) > 0

        # Re-publish the debit event directly
        row = await connection.fetchrow("SELECT event_id, payload FROM outbox WHERE payload->>'side' = 'debit' LIMIT 1")
        assert row is not None
        key = str(row["event_id"]).encode("utf-8")
        await producer.send_and_wait(topic, value=row["payload"].encode("utf-8"), key=key)

        # Run consumer again to process the re-published event
        replayed = await run_payment_updater(
            redis_client,
            group_id=f"grp-{run_id}-replay",
            idle_timeout_seconds=2.0,
            topic=topic,
        )
        assert replayed >= 1

        dump_after = _dump_account_keys(redis_client, run_id)
        assert dump_before == dump_after

    finally:
        await producer.stop()
        await delete_topic(topic)
        # Clean up Redis test namespace
        for pat in (namespace_pattern(run_id, "bank_0"), namespace_pattern(run_id, "bank_1")):
            for k in redis_client.scan_iter(match=pat):
                redis_client.delete(k)
        redis_client.close()

