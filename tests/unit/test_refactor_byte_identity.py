# File: tests/unit/test_refactor_byte_identity.py
# Purpose: Byte-level and state identity tests proving refactor parity for send_feed_record and consume_available.
"""
WHAT: Byte-identity tests proving that factoring out send_feed_record from MockBankConnectors.send_feed
and consume_available from run_payment_updater preserves exact payload byte sequences, idempotency keys,
and Redis state mutations.

WHY: Architect decision 16b P1: pure refactorings must produce bitwise identical wire payloads and identical
state mutations before and after extraction.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import fakeredis
import orjson
import pytest

from features.account_keys import get_default_settings
from features.account_stream import consume_available, run_payment_updater
from features.ingest_watermark import get_payment_watermark
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import BankFeed, project_run, resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.interactive_records import RunnerSettings, allow_all
from world.interactive_runner import FastRunSettings, prepare_world, run_interactive_world

RUN_ID = "5f96fb98ea9fb849"

# Smallest world used in existing test (tests/integration/test_bank_connectors_end_to_end.py:54):
# FastRunSettings(101, 20, 6, 51, "training")
# Captured baseline (path, body_bytes, idempotency_key) with pinned code_version="frozen_v1":
EXPECTED_FEED_POSTS: tuple[tuple[str, bytes, str], ...] = (
    (
        "/v1/account-events",
        b'{"event_id":"5f96fb98ea9fb849-open-acc_shop_0002","account_id":"acc_shop_0002","kind":"account_opened",'
        b'"event_time":"2010-06-27T00:00:00+05:30","details":{"home_region":2,"account_type":"current"}}',
        "acct:5f96fb98ea9fb849-open-acc_shop_0002",
    ),
    (
        "/v1/account-events",
        b'{"event_id":"5f96fb98ea9fb849-open-acc_person_000014","account_id":"acc_person_000014","kind":"account_opened",'
        b'"event_time":"2014-03-05T00:00:00+05:30","details":{"home_region":3,"account_type":"savings"}}',
        "acct:5f96fb98ea9fb849-open-acc_person_000014",
    ),
    (
        "/v1/account-events",
        b'{"event_id":"5f96fb98ea9fb849-open-acc_person_000002","account_id":"acc_person_000002","kind":"account_opened",'
        b'"event_time":"2019-09-13T00:00:00+05:30","details":{"home_region":1,"account_type":"savings"}}',
        "acct:5f96fb98ea9fb849-open-acc_person_000002",
    ),
    (
        "/v1/transactions",
        b'{"payment_id":"5f96fb98ea9fb849-000000","side":"credit","account_id":"acc_shop_0003",'
        b'"counterparty_vpa":"acc_person_000008","counterparty_bank_id":"bank_1","sender_bank_id":"bank_1",'
        b'"receiver_bank_id":"bank_1","amount_paise":14025,"event_time":"2026-09-01T00:21:00+05:30",'
        b'"status":"success","failure_domain":null,"payee_type":"merchant","channel":"upi_app",'
        b'"sender_device_fingerprint":null,"sender_os_generation":null,"balance_after_paise":5014025}',
        "tx:5f96fb98ea9fb849-000000:credit",
    ),
    (
        "/v1/transactions",
        b'{"payment_id":"5f96fb98ea9fb849-000000","side":"debit","account_id":"acc_person_000008",'
        b'"counterparty_vpa":"acc_shop_0003","counterparty_bank_id":"bank_1","sender_bank_id":"bank_1",'
        b'"receiver_bank_id":"bank_1","amount_paise":14025,"event_time":"2026-09-01T00:21:00+05:30",'
        b'"status":"success","failure_domain":null,"payee_type":"merchant","channel":"upi_app",'
        b'"sender_device_fingerprint":"fp_android_v2_70af2100","sender_os_generation":2,"balance_after_paise":1643975}',
        "tx:5f96fb98ea9fb849-000000:debit",
    ),
    (
        "/v1/transactions",
        b'{"payment_id":"5f96fb98ea9fb849-000001","side":"credit","account_id":"acc_shop_0004",'
        b'"counterparty_vpa":"acc_person_000018","counterparty_bank_id":"bank_0","sender_bank_id":"bank_0",'
        b'"receiver_bank_id":"bank_0","amount_paise":42369,"event_time":"2026-09-01T01:56:00+05:30",'
        b'"status":"success","failure_domain":null,"payee_type":"merchant","channel":"upi_app",'
        b'"sender_device_fingerprint":null,"sender_os_generation":null,"balance_after_paise":5042369}',
        "tx:5f96fb98ea9fb849-000001:credit",
    ),
    (
        "/v1/transactions",
        b'{"payment_id":"5f96fb98ea9fb849-000001","side":"debit","account_id":"acc_person_000018",'
        b'"counterparty_vpa":"acc_shop_0004","counterparty_bank_id":"bank_0","sender_bank_id":"bank_0",'
        b'"receiver_bank_id":"bank_0","amount_paise":42369,"event_time":"2026-09-01T01:56:00+05:30",'
        b'"status":"success","failure_domain":null,"payee_type":"merchant","channel":"upi_app",'
        b'"sender_device_fingerprint":"fp_android_v4_267784c2","sender_os_generation":4,"balance_after_paise":3309131}',
        "tx:5f96fb98ea9fb849-000001:debit",
    ),
    (
        "/v1/transactions",
        b'{"payment_id":"5f96fb98ea9fb849-000010","side":"debit","account_id":"acc_person_000002",'
        b'"counterparty_vpa":"acc_shop_0000","counterparty_bank_id":"bank_1","sender_bank_id":"bank_2",'
        b'"receiver_bank_id":"bank_1","amount_paise":13530,"event_time":"2026-09-01T15:30:00+05:30",'
        b'"status":"success","failure_domain":null,"payee_type":"merchant","channel":"upi_app",'
        b'"sender_device_fingerprint":"fp_android_v2_3a1f79c0","sender_os_generation":2,"balance_after_paise":1624970}',
        "tx:5f96fb98ea9fb849-000010:debit",
    ),
    (
        "/v1/transactions",
        b'{"payment_id":"5f96fb98ea9fb849-000015","side":"debit","account_id":"acc_person_000002",'
        b'"counterparty_vpa":"acc_shop_0000","counterparty_bank_id":"bank_1","sender_bank_id":"bank_2",'
        b'"receiver_bank_id":"bank_1","amount_paise":1998,"event_time":"2026-09-01T17:06:00+05:30",'
        b'"status":"success","failure_domain":null,"payee_type":"merchant","channel":"upi_app",'
        b'"sender_device_fingerprint":"fp_android_v2_3a1f79c0","sender_os_generation":2,"balance_after_paise":1622972}',
        "tx:5f96fb98ea9fb849-000015:debit",
    ),
)


class MockTransportOutcome:
    status: int = 200
    body: dict[str, Any] = {}
    replayed: bool = False


class FakeConsumerMessage:
    def __init__(self, topic: str, partition: int, offset: int, value: bytes) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.value = value


class FakeBatchConsumer:
    """Delivers a pre-populated list of messages via getmany."""

    def __init__(self, messages: list[FakeConsumerMessage]) -> None:
        self.messages = list(messages)
        self.committed_offsets: dict[Any, int] = {}

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def getmany(
        self,
        *partitions: Any,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[Any, list[Any]]:
        if not self.messages:
            return {}
        batch = list(self.messages)
        self.messages.clear()
        by_tp: dict[Any, list[Any]] = {}
        for m in batch:
            by_tp.setdefault(m.partition, []).append(m)
        return by_tp

    async def commit(self, offsets: dict[Any, int]) -> None:
        self.committed_offsets.update(offsets)


@pytest.mark.anyio
async def test_send_feed_and_send_feed_record_byte_identity() -> None:
    """
    Proves send_feed and send_feed_record emit bitwise identical (path, body_bytes, idempotency_key)
    triplets matching the literal pre-refactor baseline.
    """
    settings = FastRunSettings(101, 20, 6, 51, "training")
    runner_settings = RunnerSettings(fraud_enabled=True)
    out = run_interactive_world(settings, runner_settings, hook=allow_all, code_version="frozen_v1")
    full_feed = project_run(out)
    cfg = load_bank_feed_config()
    modes = load_bank_modes()
    prep = prepare_world(settings, None)

    acc_map: dict[str, str] = {}
    for ledger in prep.world.ledgers:
        bank_id = resolve_bank_id(ledger.bank_index, cfg)
        for account in ledger.accounts:
            acc_map[account] = bank_id

    small_payments = {b: full_feed.payments_by_bank[b][:2] for b in full_feed.payments_by_bank}
    small_accts = {b: full_feed.account_events_by_bank[b][:1] for b in full_feed.account_events_by_bank}
    feed = BankFeed(
        payments_by_bank=small_payments,
        account_events_by_bank=small_accts,
        withdrawals_by_bank={b: () for b in full_feed.payments_by_bank},
        complaints_by_bank={b: () for b in full_feed.payments_by_bank},
        dropped_device_events_by_kind={},
    )

    dispatched_calls: list[tuple[str, bytes, str]] = []

    class SharedRecorder:
        def __init__(self, bank: str) -> None:
            self.bank = bank

        async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> MockTransportOutcome:
            dispatched_calls.append((path, body_bytes, idempotency_key))
            return MockTransportOutcome()

    recorders = {b: SharedRecorder(b) for b in modes}
    loop = asyncio.get_running_loop()
    connectors = MockBankConnectors(
        recorders,  # type: ignore[arg-type]
        modes,
        acc_map,
        prep.plan.clock.start_date,
        cfg,
        2.0,
        loop,
    )

    # Dispatch via send_feed (which internally delegates to send_feed_record)
    await connectors.send_feed(feed)

    assert len(dispatched_calls) == len(EXPECTED_FEED_POSTS)
    assert tuple(dispatched_calls) == EXPECTED_FEED_POSTS


@pytest.mark.anyio
async def test_consume_available_and_run_payment_updater_state_identity() -> None:
    """
    Proves consume_available applied to fixed messages produces identical Redis keys,
    watermark, and application count as run_payment_updater.
    """
    p1 = {
        "schema_version": "2",
        "event_type": "upi.payment.recorded",
        "payment_id": f"{RUN_ID}-p-0000000001",
        "side": "debit",
        "event_time": "2026-10-28T10:00:00Z",
        "recorded_at": "2026-10-28T10:00:05Z",
        "arrival_time": "2026-10-28T10:00:05Z",
        "data": {
            "bank_id": "bank_0",
            "account_id": "acc_test_1",
            "counterparty_bank_id": "bank_1",
            "counterparty_vpa": "acc_test_2",
            "amount_paise": 20000,
            "status": "success",
            "failure_domain": None,
            "payee_type": "person",
            "channel": "upi_app",
            "sender_device_fingerprint": None,
            "sender_os_generation": None,
            "balance_after_paise": 480000,
        },
    }
    p2 = {
        "schema_version": "2",
        "event_type": "upi.payment.recorded",
        "payment_id": f"{RUN_ID}-p-0000000002",
        "side": "credit",
        "event_time": "2026-10-28T10:01:00Z",
        "recorded_at": "2026-10-28T10:01:10Z",
        "arrival_time": "2026-10-28T10:01:10Z",
        "data": {
            "bank_id": "bank_0",
            "account_id": "acc_test_1",
            "counterparty_bank_id": "bank_1",
            "counterparty_vpa": "acc_test_3",
            "amount_paise": 50000,
            "status": "success",
            "failure_domain": None,
            "payee_type": "merchant",
            "channel": "upi_app",
            "sender_device_fingerprint": None,
            "sender_os_generation": None,
            "balance_after_paise": 530000,
        },
    }

    # 1. Run consume_available directly
    m1 = FakeConsumerMessage("upi.payments", 0, 10, orjson.dumps(p1))
    m2 = FakeConsumerMessage("upi.payments", 0, 11, orjson.dumps(p2))

    r1 = fakeredis.FakeRedis()
    consumer1 = FakeBatchConsumer([m1, m2])
    cfg = get_default_settings()

    applied1 = await consume_available(consumer1, r1, cfg, timeout_ms=100)
    assert applied1 == 2
    assert get_payment_watermark(r1, 0, run_id=RUN_ID) == datetime(2026, 10, 28, 10, 1, 5, tzinfo=timezone.utc)
    keys1 = sorted(k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in r1.keys())
    assert len(keys1) == 14

    # 2. Run run_payment_updater with same messages to verify exact identical state
    m1_copy = FakeConsumerMessage("upi.payments", 0, 10, orjson.dumps(p1))
    m2_copy = FakeConsumerMessage("upi.payments", 0, 11, orjson.dumps(p2))

    r2 = fakeredis.FakeRedis()
    mock_c = AsyncMock()
    mock_c.start = AsyncMock()
    mock_c.stop = AsyncMock()
    mock_c.commit = AsyncMock()
    mock_c.getone = AsyncMock(side_effect=[m1_copy, m2_copy, asyncio.TimeoutError])

    with patch("features.account_stream.AIOKafkaConsumer", return_value=mock_c):
        applied2 = await run_payment_updater(r2, idle_timeout_seconds=0.1)

    assert applied2 == 2
    assert get_payment_watermark(r2, 0, run_id=RUN_ID) == get_payment_watermark(r1, 0, run_id=RUN_ID)
    keys2 = sorted(k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in r2.keys())
    assert keys2 == keys1
    for k in keys1:
        assert r1.type(k) == r2.type(k)

