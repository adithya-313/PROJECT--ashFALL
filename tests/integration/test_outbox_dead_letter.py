# File: tests/integration/test_outbox_dead_letter.py
# Purpose: Proves the relay's dead-letter handling against a real database (each test has its own throwaway schema):
#          a poison message is parked after the attempt limit WITHOUT blocking the messages behind it, an outage never
#          parks anything, and a parked row stays in the outbox.
#
# The Kafka producer is a small fake that fails on purpose, so no Kafka is needed here.
from typing import Any

import asyncpg
import pytest
from aiokafka.errors import KafkaConnectionError, MessageSizeTooLargeError
from prometheus_client import REGISTRY

from api.routers.scoring import TransactionRequest
from api.services.outbox import StoredTransaction, record_transaction_with_outbox
from api.services.outbox_relay import fetch_unprocessed_batch, relay_one_batch

pytestmark = pytest.mark.anyio

MAX_ATTEMPTS = 3
BATCH_SIZE = 10


class FakeProducer:
    """Records what is sent, and can be told to reject chosen messages or to behave as if the broker is down."""

    def __init__(self) -> None:
        self.sent_event_ids: list[str] = []
        self.poison_event_ids: set[str] = set()
        self.broker_down = False

    async def send_and_wait(self, topic: str, value: bytes, key: bytes) -> None:
        """Pretend to send: fail for an outage or a poison event, otherwise remember the event id."""
        event_id = key.decode("utf-8")
        if self.broker_down:
            raise KafkaConnectionError("broker is down")
        if event_id in self.poison_event_ids:
            raise MessageSizeTooLargeError("message is too large")
        self.sent_event_ids.append(event_id)


def make_request(index: int) -> TransactionRequest:
    """A normal request with a key, so every row has its own event id."""
    return TransactionRequest(
        amount=10.0 + index, account_id="acct-dl", merchant_id="merchant-dl", device_fingerprint=f"device-{index}",
        geo_coordinates="1.0,2.0", idempotency_key=f"dead-letter-test-{index}",
    )


async def write_rows(conn: asyncpg.Connection, count: int) -> list[StoredTransaction]:
    """Write `count` transactions with the real writer."""
    return [await record_transaction_with_outbox(conn, make_request(i), 0.1, "test-model") for i in range(count)]


async def outbox_event_ids(conn: asyncpg.Connection) -> list[str]:
    """All outbox event ids in relay order."""
    rows = await conn.fetch("SELECT event_id FROM outbox ORDER BY created_at, id")
    return [str(row["event_id"]) for row in rows]


async def run_rounds(conn: asyncpg.Connection, producer: Any, rounds: int) -> list[Any]:
    """Run the real relay for a fixed number of rounds and return each round's result."""
    return [await relay_one_batch(conn, producer, BATCH_SIZE, max_attempts=MAX_ATTEMPTS) for _ in range(rounds)]


async def test_poison_message_is_parked_and_does_not_block_the_rest(isolated_schema):
    """One poison message among five is parked after the limit, and the other four are still delivered in order."""
    conn, _ = isolated_schema
    await write_rows(conn, 5)
    event_ids = await outbox_event_ids(conn)
    producer = FakeProducer()
    producer.poison_event_ids = {event_ids[1]}
    counter_before = REGISTRY.get_sample_value("outbox_dead_lettered_total") or 0.0

    await run_rounds(conn, producer, MAX_ATTEMPTS + 2)

    # The metric rose by exactly the number of parked rows.
    assert REGISTRY.get_sample_value("outbox_dead_lettered_total") == counter_before + 1

    # The four good messages arrived, in the same order as in the outbox.
    assert producer.sent_event_ids == [event_ids[0], event_ids[2], event_ids[3], event_ids[4]]

    # The poison message sits in the dead-letter table with its details.
    parked = await conn.fetch("SELECT * FROM outbox_dead_letter")
    assert len(parked) == 1
    assert str(parked[0]["event_id"]) == event_ids[1]
    assert parked[0]["attempt_count"] == MAX_ATTEMPTS
    assert parked[0]["error_type"] == "MessageSizeTooLargeError"
    assert parked[0]["error_message"]
    assert parked[0]["first_failed_at"] is not None and parked[0]["dead_lettered_at"] is not None

    # In the outbox: the poison row is marked parked (not processed), the other four are processed.
    rows = await conn.fetch("SELECT event_id, processed_at, dead_lettered_at FROM outbox ORDER BY id")
    assert rows[1]["dead_lettered_at"] is not None and rows[1]["processed_at"] is None
    for index in (0, 2, 3, 4):
        assert rows[index]["processed_at"] is not None and rows[index]["dead_lettered_at"] is None


async def test_below_the_limit_nothing_is_parked_and_order_is_kept(isolated_schema):
    """While a poison message is still being retried, it is not parked and the rows behind it wait."""
    conn, _ = isolated_schema
    await write_rows(conn, 5)
    event_ids = await outbox_event_ids(conn)
    producer = FakeProducer()
    producer.poison_event_ids = {event_ids[1]}

    results = await run_rounds(conn, producer, MAX_ATTEMPTS - 1)

    assert producer.sent_event_ids == [event_ids[0]]  # only the row before the poison one went out
    assert await conn.fetchval("SELECT count(*) FROM outbox_dead_letter") == 0
    assert await conn.fetchval("SELECT attempt_count FROM outbox WHERE event_id = $1::uuid", event_ids[1]) == 2
    assert results[-1].poison_pending is True and results[-1].dead_lettered == 0


async def test_an_outage_never_parks_anything(isolated_schema):
    """Many rounds with the broker down count no attempts and park nothing; everything arrives once it is back."""
    conn, _ = isolated_schema
    await write_rows(conn, 5)
    event_ids = await outbox_event_ids(conn)
    producer = FakeProducer()
    producer.broker_down = True

    results = await run_rounds(conn, producer, 20)

    assert all(result.poison_pending is False and result.error is not None for result in results)
    assert await conn.fetchval("SELECT count(*) FROM outbox_dead_letter") == 0
    assert await conn.fetchval("SELECT sum(attempt_count) FROM outbox") == 0

    # The broker comes back: one round delivers everything, in order.
    producer.broker_down = False
    await run_rounds(conn, producer, 1)
    assert producer.sent_event_ids == event_ids


async def test_parked_row_is_not_fetched_again_and_replay_still_works(isolated_schema):
    """A parked row is skipped by the relay, stays in the outbox, and its idempotency key still finds the original."""
    conn, _ = isolated_schema
    stored = await write_rows(conn, 3)
    event_ids = await outbox_event_ids(conn)
    producer = FakeProducer()
    producer.poison_event_ids = {event_ids[0]}

    await run_rounds(conn, producer, MAX_ATTEMPTS + 2)
    sent_before = list(producer.sent_event_ids)
    await run_rounds(conn, producer, 3)

    # Nothing was sent again and the relay no longer sees any waiting row.
    assert producer.sent_event_ids == sent_before == [event_ids[1], event_ids[2]]
    async with conn.transaction():
        assert await fetch_unprocessed_batch(conn, BATCH_SIZE) == []

    # The outbox row still exists, so repeating the first request returns the original transaction.
    assert await conn.fetchval("SELECT count(*) FROM outbox") == 3
    replay = await record_transaction_with_outbox(conn, make_request(0), 0.1, "test-model")
    assert replay.was_replay is True
    assert replay.transaction_id == stored[0].transaction_id
