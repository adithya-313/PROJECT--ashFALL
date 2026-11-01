# File: tests/unit/test_upi_payload_arrival.py
# Purpose: Level 16a Round 3a - unit tests for UPI payload arrival time and world watermark routing.
import logging
import uuid
from datetime import datetime
from typing import Any

import fakeredis
import orjson
import pytest

from api.v1.upi_outbox import UpiPaymentSide, record_upi_side
from core.event_clock import IST, ManualClock
from features.account_keys import (
    get_default_settings,
    watermark_key_for_payment_partition,
)
from features.account_records import InvalidAccountEventError, parse_upi_event, parse_upi_payload
from features.account_stream import record_watermark_for_side

pytestmark = pytest.mark.anyio

FIXED_TRANSACTION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
FIXED_CREATED_AT = datetime(2026, 9, 5, 12, 0, 0, tzinfo=IST)
CLOCK_ARRIVAL_TIME = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)

FIXED_SIDE = UpiPaymentSide(
    payment_id="0123456789abcdef-pmt-001",
    bank_id="bank_0",
    side="debit",
    account_id="acc_001",
    counterparty_bank_id="bank_1",
    sender_bank_id="bank_0",
    receiver_bank_id="bank_1",
    amount_paise=100000,
    event_time=datetime(2026, 9, 5, 12, 0, 0, tzinfo=IST),
    status="success",
    counterparty_vpa="user@bank1",
    failure_domain=None,
    payee_type="person",
    channel="mobile_app",
    sender_device_fingerprint="fp_123",
    sender_os_generation=14,
    balance_after_paise=900000,
)

# Golden string captured on the unchanged code before Commit 2 edits (GUARD 3).
EXPECTED_GOLDEN_PAYLOAD = (
    '{"event_type":"upi.payment.recorded","schema_version":"2",'
    '"event_id":"d8b8242a-7213-5f19-bdf4-8e8a3e800dfa",'
    '"transaction_id":"11111111-1111-1111-1111-111111111111",'
    '"payment_id":"0123456789abcdef-pmt-001","side":"debit",'
    '"event_time":"2026-09-05T12:00:00+05:30","recorded_at":"2026-09-05T12:00:00+05:30",'
    '"source":"world","data":{"bank_id":"bank_0","account_id":"acc_001",'
    '"counterparty_vpa":"user@bank1","counterparty_bank_id":"bank_1",'
    '"sender_bank_id":"bank_0","receiver_bank_id":"bank_1","amount_paise":100000,'
    '"status":"success","failure_domain":null,"payee_type":"person",'
    '"channel":"mobile_app","sender_device_fingerprint":"fp_123",'
    '"sender_os_generation":14,"balance_after_paise":900000}}'
)


class FakeTransactionContext:
    async def __aenter__(self) -> "FakeTransactionContext":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class FakeOutboxConnection:
    def __init__(self) -> None:
        self.captured_payload: str | None = None

    def transaction(self) -> FakeTransactionContext:
        return FakeTransactionContext()

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
        return {
            "transaction_id": FIXED_TRANSACTION_ID,
            "created_at": FIXED_CREATED_AT,
        }

    async def execute(self, query: str, *args: Any) -> None:
        if "INSERT INTO outbox" in query:
            # 4th argument is $4 (0-indexed 3)
            self.captured_payload = args[3]


async def test_golden_byte_equality_when_clock_is_none() -> None:
    """GUARD 3: when clock=None, serialized payload is byte-for-byte identical to baseline with NO arrival_time."""
    fake_conn = FakeOutboxConnection()
    await record_upi_side(fake_conn, FIXED_SIDE, ingestion_clock=None)  # type: ignore[arg-type]
    assert fake_conn.captured_payload is not None
    assert fake_conn.captured_payload == EXPECTED_GOLDEN_PAYLOAD
    parsed = orjson.loads(fake_conn.captured_payload)
    assert "arrival_time" not in parsed
    # Verify that legitimate null fields (e.g. failure_domain) are preserved (F1)
    assert parsed["data"]["failure_domain"] is None


async def test_world_clock_payload_contains_arrival_time() -> None:
    """When a Clock is injected, serialized payload contains top-level arrival_time equal to the clock."""
    fake_conn = FakeOutboxConnection()
    clock = ManualClock(CLOCK_ARRIVAL_TIME)
    await record_upi_side(fake_conn, FIXED_SIDE, ingestion_clock=clock)  # type: ignore[arg-type]
    assert fake_conn.captured_payload is not None
    parsed = orjson.loads(fake_conn.captured_payload)
    assert "arrival_time" in parsed
    assert parsed["arrival_time"] == CLOCK_ARRIVAL_TIME.isoformat()
    assert parsed["schema_version"] == "2"


async def test_parse_upi_payload_with_valid_arrival() -> None:
    """When arrival_time is present, record is marked world-clock and arrival_time equals it."""
    payload = orjson.loads(EXPECTED_GOLDEN_PAYLOAD)
    payload["arrival_time"] = CLOCK_ARRIVAL_TIME.isoformat()

    record = parse_upi_payload(payload)
    assert record.arrival_from_ingestion_clock is True
    assert record.arrival_time == CLOCK_ARRIVAL_TIME
    assert record.payment_id == "0123456789abcdef-pmt-001"
    assert record.run_id == "0123456789abcdef"


async def test_parse_upi_payload_unparseable_arrival_guard2() -> None:
    """GUARD 2: unparseable arrival_time raises InvalidAccountEventError; never falls back to recorded_at."""
    payload = orjson.loads(EXPECTED_GOLDEN_PAYLOAD)
    payload["arrival_time"] = "not-a-valid-iso-timestamp"

    with pytest.raises(InvalidAccountEventError, match="Unparseable ISO arrival_time"):
        parse_upi_payload(payload)


async def test_parse_upi_payload_naive_arrival_guard2() -> None:
    """GUARD 2: naive arrival_time raises InvalidAccountEventError; timezone required."""
    payload = orjson.loads(EXPECTED_GOLDEN_PAYLOAD)
    payload["arrival_time"] = "2026-09-05T14:00:00"

    with pytest.raises(InvalidAccountEventError, match="Naive arrival_time"):
        parse_upi_payload(payload)


async def test_parse_upi_payload_bad_run_prefix_guard2() -> None:
    """GUARD 2: bad payment_id run prefix raises InvalidAccountEventError even when arrival_time is present."""
    payload = orjson.loads(EXPECTED_GOLDEN_PAYLOAD)
    payload["payment_id"] = "badprefix_pmt_001"
    payload["arrival_time"] = CLOCK_ARRIVAL_TIME.isoformat()

    with pytest.raises(InvalidAccountEventError, match="Invalid run ID in payment_id"):
        parse_upi_payload(payload)


async def test_parse_upi_payload_arrival_earlier_than_event_time_guard4(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """GUARD 4: arrival_time < event_time logs warning with both values; does NOT reject."""
    payload = orjson.loads(EXPECTED_GOLDEN_PAYLOAD)
    # event_time in payload is 2026-09-05 12:00 IST
    earlier_arrival = datetime(2026, 9, 5, 11, 0, 0, tzinfo=IST)
    payload["arrival_time"] = earlier_arrival.isoformat()

    with caplog.at_level(logging.WARNING):
        record = parse_upi_payload(payload)

    assert record.arrival_from_ingestion_clock is True
    assert record.arrival_time == earlier_arrival
    assert any(
        "Payment arrival_time earlier than event_time" in r.message
        and earlier_arrival.isoformat() in r.message
        and "2026-09-05T12:00:00+05:30" in r.message
        for r in caplog.records
    )


async def test_consumer_watermark_routing_world_vs_real() -> None:
    """
    Consumer watermark routing:
    - world record (arrival_from_ingestion_clock=True) writes to run-scoped key and NOT shared key.
    - real record (arrival_from_ingestion_clock=False) writes to shared key only and NOT run-scoped key.
    """
    redis_client = fakeredis.FakeRedis()
    cfg = get_default_settings()
    partition = 2
    run_id = "0123456789abcdef"

    raw_bytes = orjson.dumps(orjson.loads(EXPECTED_GOLDEN_PAYLOAD))

    # 1. Real-clock record (legacy / clock=None)
    real_record = parse_upi_event(raw_bytes)
    assert real_record.arrival_from_ingestion_clock is False

    record_watermark_for_side(redis_client, partition, real_record, cfg)

    shared_key = watermark_key_for_payment_partition(partition=partition, run_id=None)
    run_key = watermark_key_for_payment_partition(partition=partition, run_id=run_id)

    assert redis_client.exists(shared_key) == 1
    assert redis_client.exists(run_key) == 0

    # 2. World-clock record (injected clock)
    payload_world = orjson.loads(EXPECTED_GOLDEN_PAYLOAD)
    payload_world["arrival_time"] = CLOCK_ARRIVAL_TIME.isoformat()
    world_bytes = orjson.dumps(payload_world)

    world_record = parse_upi_event(world_bytes)
    assert world_record.arrival_from_ingestion_clock is True

    # Use a fresh partition to isolate test verification
    world_partition = 3
    world_shared_key = watermark_key_for_payment_partition(partition=world_partition, run_id=None)
    world_run_key = watermark_key_for_payment_partition(partition=world_partition, run_id=run_id)

    record_watermark_for_side(redis_client, world_partition, world_record, cfg)

    assert redis_client.exists(world_run_key) == 1
    assert redis_client.exists(world_shared_key) == 0
