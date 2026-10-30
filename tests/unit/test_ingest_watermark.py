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
import pytest
from aiokafka import TopicPartition

from features.account_keys import (
    payment_watermark_key,
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
)

pytestmark = pytest.mark.anyio

RUN_ID = "66db9aa439b092ee"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


class FakeDbConnection:
    """Mock database connection for table pollers."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        return self.rows


class FakeKafkaMessage:
    """Mock Kafka consumer message."""

    def __init__(self, topic: str, partition: int, offset: int, value: bytes, timestamp: int = 0) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.value = value
        self.timestamp = timestamp


def test_record_and_get_payment_watermark(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that payment watermark is stored and retrieved as a timezone-aware UTC datetime."""
    assert get_payment_watermark(redis_client, 0) is None

    record_payment_watermark(redis_client, 0, NOW_UTC)
    wm = get_payment_watermark(redis_client, 0)
    assert wm is not None
    assert wm == NOW_UTC
    assert wm.tzinfo is not None

    raw = redis_client.get(payment_watermark_key(0))
    assert raw == str(int(NOW_UTC.timestamp())).encode("utf-8")


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

    # Assigned partitions [0, 1]. The Kafka watermark must be MIN(100, 200) = 100.
    # Overall bank watermark must be MIN(100, 300, 300, 300, 300) = 100.
    # If S1 replaces MIN with MAX across partitions, payments becomes 200 and this assertion fails!
    result = get_bank_watermark(redis_client, "bank_0", assigned_partitions=[0, 1])
    assert result == datetime.fromtimestamp(100, tz=timezone.utc)


def test_get_bank_watermark_missing_source_returns_none(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that if any source has no recorded watermark, get_bank_watermark returns None."""
    # No assigned partitions
    assert get_bank_watermark(redis_client, "bank_0", assigned_partitions=None) is None
    assert get_bank_watermark(redis_client, "bank_0", assigned_partitions=[]) is None

    # Kafka partition recorded, but no pollers
    record_payment_watermark(redis_client, 0, NOW_UTC)
    assert get_bank_watermark(redis_client, "bank_0", assigned_partitions=[0]) is None

    # Add 3 pollers, leaving 1 missing
    record_poller_watermark(redis_client, "bank_0", "events", NOW_UTC)
    record_poller_watermark(redis_client, "bank_0", "withdrawals", NOW_UTC)
    record_poller_watermark(redis_client, "bank_0", "feedback", NOW_UTC)
    assert get_bank_watermark(redis_client, "bank_0", assigned_partitions=[0]) is None

    # Add the last poller: now all 5 sources are present
    record_poller_watermark(redis_client, "bank_0", "complaints", NOW_UTC)
    assert get_bank_watermark(redis_client, "bank_0", assigned_partitions=[0]) == NOW_UTC


async def test_apply_account_events_watermark_update(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_account_events_since updates watermark on batches and on idle."""
    t1 = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 28, 10, 5, 0, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "bank_id": "bank_0",
            "event_id": f"{RUN_ID}-dev-001",
            "account_id": "acc_1",
            "kind": "account_opened",
            "event_time": t1,
            "created_at": t1,
            "details": {},
        },
        {
            "id": 2,
            "bank_id": "bank_0",
            "event_id": f"{RUN_ID}-dev-002",
            "account_id": "acc_1",
            "kind": "sim_swap",
            "event_time": t2,
            "created_at": t2,
            "details": {},
        },
    ]
    conn = FakeDbConnection(rows)
    await apply_account_events_since(conn, redis_client, bank_id="bank_0")
    assert get_poller_watermark(redis_client, "bank_0", "events") == t2

    # Idle poll (0 rows) advances watermark to now_utc - commit_grace (5s)
    empty_conn = FakeDbConnection([])
    idle_now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    await apply_account_events_since(empty_conn, redis_client, bank_id="bank_0", now_utc=idle_now)
    expected_idle = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "events") == expected_idle


async def test_apply_withdrawals_watermark_update(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_withdrawals_since updates watermark on batches and on idle."""
    t1 = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 28, 10, 8, 0, tzinfo=timezone.utc)
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
            "event_time": t1,
            "balance_after_paise": 100000,
            "recorded_at": t2,
        }
    ]
    conn = FakeDbConnection(rows)
    await apply_withdrawals_since(conn, redis_client, bank_id="bank_0")
    assert get_poller_watermark(redis_client, "bank_0", "withdrawals") == t2

    empty_conn = FakeDbConnection([])
    idle_now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    await apply_withdrawals_since(empty_conn, redis_client, bank_id="bank_0", now_utc=idle_now)
    expected_idle = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "withdrawals") == expected_idle


async def test_apply_feedback_watermark_update(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_feedback_since updates watermark on batches and on idle."""
    t1 = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 28, 10, 15, 0, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "bank_id": "bank_0",
            "feedback_id": f"{RUN_ID}-fb-001",
            "payment_id": f"{RUN_ID}-p-001",
            "outcome": "not_fraud",
            "event_time": t1,
            "created_at": t2,
        }
    ]
    conn = FakeDbConnection(rows)
    await apply_feedback_since(conn, redis_client, bank_id="bank_0")
    assert get_poller_watermark(redis_client, "bank_0", "feedback") == t2

    empty_conn = FakeDbConnection([])
    idle_now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    await apply_feedback_since(empty_conn, redis_client, bank_id="bank_0", now_utc=idle_now)
    expected_idle = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "feedback") == expected_idle


async def test_apply_complaints_watermark_update(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_complaints_since updates watermark on batches and on idle."""
    t1 = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 28, 10, 20, 0, tzinfo=timezone.utc)
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
            "received_at": t1,
            "created_at": t2,
        }
    ]
    conn = FakeDbConnection(rows)
    await apply_complaints_since(conn, redis_client, bank_id="bank_0")
    assert get_poller_watermark(redis_client, "bank_0", "complaints") == t2

    empty_conn = FakeDbConnection([])
    idle_now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    await apply_complaints_since(empty_conn, redis_client, bank_id="bank_0", now_utc=idle_now)
    expected_idle = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_poller_watermark(redis_client, "bank_0", "complaints") == expected_idle


async def test_run_payment_updater_advances_watermark_on_idle(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that run_payment_updater advances partition watermark when idle at end of partition."""
    tp = TopicPartition("upi.payments", 0)
    mock_consumer = AsyncMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()
    mock_consumer.getone = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_consumer.assignment = lambda: {tp}
    mock_consumer.end_offsets = AsyncMock(return_value={tp: 42})
    mock_consumer.position = AsyncMock(return_value=42)

    idle_now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    with patch("features.account_stream.AIOKafkaConsumer", return_value=mock_consumer):
        processed = await run_payment_updater(
            redis_client,
            idle_timeout_seconds=0.1,
            now_utc=idle_now,
        )

    assert processed == 0
    expected_idle = datetime(2026, 10, 28, 11, 59, 55, tzinfo=timezone.utc)
    assert get_payment_watermark(redis_client, 0) == expected_idle
