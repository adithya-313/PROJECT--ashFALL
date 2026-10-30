# File: tests/unit/test_ingest_watermark.py
# Purpose: Unit tests for Level 15c ingestion watermark recording and bank-level aggregation.
# Issue: Level 15c watermark and feature log (plan Task 1, D1, S1, S2).
"""
WHAT: Unit tests covering per-source ingestion watermarks for Kafka upi.payments and Postgres pollers.

WHY: Verifies strict monotonic writes via Redis Lua script, complete 5-source aggregate calculation,
and watermark updates across streaming consumers and database pollers.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import fakeredis
import orjson
import pytest

from features.account_keys import (
    payment_watermark_key,
    payments_partitions_key,
    poller_watermark_key,
)
from features.account_labels import apply_complaints_since, apply_feedback_since
from features.account_stream import (
    apply_account_events_since,
    apply_withdrawals_since,
    run_payment_updater,
)
from features.ingest_watermark import (
    POLLER_SOURCES,
    get_bank_watermark,
    get_payment_watermark,
    get_poller_watermark,
    record_payment_watermark,
    record_poller_watermark,
    to_unix_seconds,
    to_watermark_seconds,
)

pytestmark = pytest.mark.anyio

RUN_ID = "66db9aa439b092ee"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


class FakeDbConnection:
    """Mock database connection for table pollers."""

    def __init__(self, rows: list[dict[str, Any]], now_dt: datetime | None = None) -> None:
        self.rows = rows
        self.now_dt = now_dt or NOW_UTC

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        return self.rows

    async def fetchval(self, query: str, *args: Any) -> Any:
        return self.now_dt


class FakeKafkaMessage:
    """Mock Kafka consumer message."""

    def __init__(self, topic: str, partition: int, offset: int, value: bytes, timestamp: int = 0) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.value = value
        self.timestamp = timestamp


def test_to_unix_seconds_rejects_naive_datetime() -> None:
    """F10: Verifies that to_unix_seconds raises ValueError if datetime is naive."""
    naive_dt = datetime(2026, 10, 28, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        to_unix_seconds(naive_dt)

    with pytest.raises(ValueError, match="timezone-aware"):
        to_watermark_seconds(naive_dt)

    aware_dt = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    assert to_unix_seconds(aware_dt) == int(aware_dt.timestamp())
    assert to_watermark_seconds(aware_dt) == int(aware_dt.timestamp())
    assert to_watermark_seconds(12345) == 12345


def test_record_and_get_payment_watermark(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that payment watermark is stored, registered in partitions set, and retrieved."""
    assert get_payment_watermark(redis_client, 0) is None

    record_payment_watermark(redis_client, 0, NOW_UTC)
    wm = get_payment_watermark(redis_client, 0)
    assert wm is not None
    assert wm == NOW_UTC
    assert wm.tzinfo is not None

    raw = redis_client.get(payment_watermark_key(0))
    assert raw == str(int(NOW_UTC.timestamp())).encode("utf-8")

    # Verify partition is in the payments partitions set
    members = redis_client.smembers(payments_partitions_key())
    assert isinstance(members, set)
    assert b"0" in members or "0" in members or 0 in members


def test_record_and_get_poller_watermark(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies poller watermark operations for all 4 Postgres poller sources."""
    for src in POLLER_SOURCES:
        assert get_poller_watermark(redis_client, "bank_0", src) is None
        record_poller_watermark(redis_client, "bank_0", src, NOW_UTC)
        wm = get_poller_watermark(redis_client, "bank_0", src)
        assert wm is not None
        assert wm == NOW_UTC
        assert wm.tzinfo is not None

        raw = redis_client.get(poller_watermark_key("bank_0", src))
        assert raw == str(int(NOW_UTC.timestamp())).encode("utf-8")

    with pytest.raises(ValueError, match="Unknown poller source"):
        record_poller_watermark(redis_client, "bank_0", "unknown_source", NOW_UTC)

    with pytest.raises(ValueError, match="Unknown poller source"):
        get_poller_watermark(redis_client, "bank_0", "unknown_source")


def test_watermark_monotonic_write_guard(redis_client: fakeredis.FakeRedis) -> None:
    """
    SABOTAGE S2 TARGET:
    Verifies that writing an older timestamp does NOT move the watermark backwards.
    """
    t_100 = datetime.fromtimestamp(100, tz=timezone.utc)
    t_50 = datetime.fromtimestamp(50, tz=timezone.utc)
    t_150 = datetime.fromtimestamp(150, tz=timezone.utc)

    # Initial write: 100
    record_poller_watermark(redis_client, "bank_0", "events", t_100)
    assert get_poller_watermark(redis_client, "bank_0", "events") == t_100

    # Monotonic guard check: write older timestamp 50. Stored must remain 100!
    record_poller_watermark(redis_client, "bank_0", "events", t_50)
    assert get_poller_watermark(redis_client, "bank_0", "events") == t_100

    # Write newer timestamp 150: stored advances to 150
    record_poller_watermark(redis_client, "bank_0", "events", t_150)
    assert get_poller_watermark(redis_client, "bank_0", "events") == t_150


def test_get_bank_watermark_sources_and_min(redis_client: fakeredis.FakeRedis) -> None:
    """
    SABOTAGE S1 TARGET:
    Verifies that get_bank_watermark aggregates across all 5 sources and uses MIN across Kafka partitions.
    """
    # Partition 0: 100, Partition 1: 200
    record_payment_watermark(redis_client, 0, datetime.fromtimestamp(100, tz=timezone.utc))
    record_payment_watermark(redis_client, 1, datetime.fromtimestamp(200, tz=timezone.utc))

    # All 4 Postgres pollers set to 300
    for src in POLLER_SOURCES:
        record_poller_watermark(redis_client, "bank_0", src, datetime.fromtimestamp(300, tz=timezone.utc))

    # Partitions in set are {0, 1}. The Kafka watermark must be MIN(100, 200) = 100.
    # Overall bank watermark must be MIN(100, 300, 300, 300, 300) = 100.
    result = get_bank_watermark(redis_client, "bank_0")
    assert result == datetime.fromtimestamp(100, tz=timezone.utc)


def test_get_bank_watermark_missing_source_returns_none(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that if any source has no recorded watermark, get_bank_watermark returns None."""
    # No partitions recorded
    assert get_bank_watermark(redis_client, "bank_0") is None

    # Kafka partition recorded, but no pollers
    record_payment_watermark(redis_client, 0, NOW_UTC)
    assert get_bank_watermark(redis_client, "bank_0") is None

    # Add 3 pollers, leaving 1 missing
    record_poller_watermark(redis_client, "bank_0", "events", NOW_UTC)
    record_poller_watermark(redis_client, "bank_0", "withdrawals", NOW_UTC)
    record_poller_watermark(redis_client, "bank_0", "feedback", NOW_UTC)
    assert get_bank_watermark(redis_client, "bank_0") is None

    # Add the last poller: now all 5 sources are present
    record_poller_watermark(redis_client, "bank_0", "complaints", NOW_UTC)
    assert get_bank_watermark(redis_client, "bank_0") == NOW_UTC


def test_partitions_set_drives_bank_watermark(redis_client: fakeredis.FakeRedis) -> None:
    """
    R3 & SABOTAGE S8 TARGET:
    Verifies that get_bank_watermark is driven dynamically by the Redis partitions set:
    - empty set gives None
    - two partitions recorded with different times give MIN
    - one partition listed in set with its watermark missing gives None.
    """
    # 1. Empty set gives None
    assert get_bank_watermark(redis_client, "bank_0") is None

    # Setup 4 pollers
    for src in POLLER_SOURCES:
        record_poller_watermark(redis_client, "bank_0", src, datetime.fromtimestamp(500, tz=timezone.utc))

    # 2. Record two partitions with different times: partition 0 -> 200, partition 1 -> 150
    record_payment_watermark(redis_client, 0, datetime.fromtimestamp(200, tz=timezone.utc))
    record_payment_watermark(redis_client, 1, datetime.fromtimestamp(150, tz=timezone.utc))

    # Bank watermark must be MIN(150, 200, 500...) = 150
    assert get_bank_watermark(redis_client, "bank_0") == datetime.fromtimestamp(150, tz=timezone.utc)

    # 3. Add partition 2 to set but DO NOT record its watermark key
    redis_client.sadd(payments_partitions_key(), 2)
    # Now partition 2 has no watermark -> get_bank_watermark must return None
    assert get_bank_watermark(redis_client, "bank_0") is None


async def test_apply_account_events_watermark_update(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_account_events_since derives watermark from DB poll start time minus commit grace."""
    t_db = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "bank_id": "bank_0",
            "event_id": f"{RUN_ID}-dev-001",
            "account_id": "acc_1",
            "kind": "account_opened",
            "event_time": datetime(2026, 10, 28, 9, 0, 0, tzinfo=timezone.utc),
            "created_at": datetime(2026, 10, 28, 9, 0, 0, tzinfo=timezone.utc),
            "details": {},
        },
    ]
    conn = FakeDbConnection(rows, now_dt=t_db)
    await apply_account_events_since(conn, redis_client, bank_id="bank_0")
    expected_wm = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "events") == expected_wm

    # Idle poll (0 rows) also advances watermark to DB poll start time - commit_grace (5s)
    t_db_idle = datetime(2026, 10, 28, 13, 0, 0, tzinfo=timezone.utc)
    empty_conn = FakeDbConnection([], now_dt=t_db_idle)
    await apply_account_events_since(empty_conn, redis_client, bank_id="bank_0")
    expected_idle = datetime(2026, 10, 28, 12, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "events") == expected_idle


async def test_apply_withdrawals_watermark_update(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_withdrawals_since derives watermark from DB poll start time minus commit grace."""
    t_db = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "bank_id": "bank_0",
            "withdrawal_id": f"{RUN_ID}-w-001",
            "account_id": "acc_1",
            "channel": "atm",
            "amount_paise": 50000,
            "status": "success",
            "failure_domain": None,
            "event_time": datetime(2026, 10, 28, 9, 0, 0, tzinfo=timezone.utc),
            "balance_after_paise": 100000,
            "recorded_at": datetime(2026, 10, 28, 9, 0, 0, tzinfo=timezone.utc),
        }
    ]
    conn = FakeDbConnection(rows, now_dt=t_db)
    await apply_withdrawals_since(conn, redis_client, bank_id="bank_0")
    expected_wm = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "withdrawals") == expected_wm

    empty_conn = FakeDbConnection([], now_dt=datetime(2026, 10, 28, 13, 0, 0, tzinfo=timezone.utc))
    await apply_withdrawals_since(empty_conn, redis_client, bank_id="bank_0")
    expected_idle = datetime(2026, 10, 28, 12, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "withdrawals") == expected_idle


async def test_apply_feedback_watermark_update(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_feedback_since derives watermark from DB poll start time minus commit grace."""
    t_db = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "bank_id": "bank_0",
            "feedback_id": f"{RUN_ID}-fb-001",
            "payment_id": f"{RUN_ID}-p-001",
            "outcome": "not_fraud",
            "event_time": datetime(2026, 10, 28, 9, 0, 0, tzinfo=timezone.utc),
            "created_at": datetime(2026, 10, 28, 9, 0, 0, tzinfo=timezone.utc),
        }
    ]
    conn = FakeDbConnection(rows, now_dt=t_db)
    await apply_feedback_since(conn, redis_client, bank_id="bank_0")
    expected_wm = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "feedback") == expected_wm

    empty_conn = FakeDbConnection([], now_dt=datetime(2026, 10, 28, 13, 0, 0, tzinfo=timezone.utc))
    await apply_feedback_since(empty_conn, redis_client, bank_id="bank_0")
    expected_idle = datetime(2026, 10, 28, 12, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "feedback") == expected_idle


async def test_apply_complaints_watermark_update(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_complaints_since derives watermark from DB poll start time minus commit grace."""
    t_db = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "bank_id": "bank_0",
            "complaint_id": f"{RUN_ID}-cmp-001",
            "complainant_account": "acc_v",
            "reported_account": "acc_m",
            "reported_bank_id": "bank_0",
            "payment_ids": [f"{RUN_ID}-p-001"],
            "claimed_type": "fraud",
            "received_at": datetime(2026, 10, 28, 9, 0, 0, tzinfo=timezone.utc),
            "created_at": datetime(2026, 10, 28, 9, 0, 0, tzinfo=timezone.utc),
        }
    ]
    conn = FakeDbConnection(rows, now_dt=t_db)
    await apply_complaints_since(conn, redis_client, bank_id="bank_0")
    expected_wm = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "complaints") == expected_wm

    empty_conn = FakeDbConnection([], now_dt=datetime(2026, 10, 28, 13, 0, 0, tzinfo=timezone.utc))
    await apply_complaints_since(empty_conn, redis_client, bank_id="bank_0")
    expected_idle = datetime(2026, 10, 28, 12, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "complaints") == expected_idle


async def test_idle_partition_does_not_advance_watermark(redis_client: fakeredis.FakeRedis) -> None:
    """
    R3 & SABOTAGE S7 TARGET:
    Verifies that when the updater goes idle, the watermark remains pinned to the latest applied message
    recorded_at - commit_grace, and does NOT advance on idle timeout.
    """
    msg_payload = {
        "schema_version": "2",
        "event_type": "upi.payment.recorded",
        "payment_id": f"{RUN_ID}-p-0000000001",
        "side": "credit",
        "event_time": "2026-10-28T10:00:00Z",
        "recorded_at": "2026-10-28T10:00:05Z",
        "data": {
            "bank_id": "bank_0",
            "account_id": "acc_01",
            "counterparty_bank_id": "bank_1",
            "counterparty_vpa": "payer@bank1",
            "amount_paise": 10000,
            "status": "success",
            "failure_domain": None,
            "payee_type": "person",
            "channel": "upi_app",
            "sender_device_fingerprint": None,
            "sender_os_generation": None,
            "balance_after_paise": 50000,
        },
    }
    raw_val = orjson.dumps(msg_payload)
    msg = FakeKafkaMessage("upi.payments", 0, 0, raw_val)

    mock_consumer = AsyncMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()
    # First call returns message, second call times out (goes idle)
    mock_consumer.getone = AsyncMock(side_effect=[msg, asyncio.TimeoutError])

    with patch("features.account_stream.AIOKafkaConsumer", return_value=mock_consumer):
        processed = await run_payment_updater(
            redis_client,
            idle_timeout_seconds=0.1,
        )

    assert processed == 1
    # Watermark must be recorded_at (10:00:05) - commit_grace (5s) = 10:00:00
    expected_wm = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    assert get_payment_watermark(redis_client, 0) == expected_wm
