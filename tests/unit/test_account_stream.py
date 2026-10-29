# File: tests/unit/test_account_stream.py
# Purpose: Unit tests for streaming payment updater and account event polling adapter.
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Unit tests covering the Kafka stream consumer adapter (run_payment_updater)
and the Postgres table poller adapter (apply_account_events_since).

WHY: Verifies that malformed Kafka messages fail fast and re-raise without swallowing errors,
and that account events are polled in monotonic ID order and checkpoint the latest ID.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import fakeredis
import pytest

from features.account_reader import read_account_snapshot
from features.account_records import InvalidAccountEventError
from features.account_stream import apply_account_events_since, run_payment_updater

pytestmark = pytest.mark.anyio

RUN_ID = "66db9aa439b092ee"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


class FakeConsumerMessage:
    """Represents a single message yielded by a Kafka consumer."""

    def __init__(self, topic: str, partition: int, offset: int, value: bytes) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.value = value


class FakeAccountEventConnection:
    """In-memory mock database connection simulating bank_account_events table reads."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.queries.append((query, args))
        bank_id, after_id, limit = args
        matching = [
            r for r in self.rows
            if r["bank_id"] == bank_id and r["id"] > after_id
        ]
        matching.sort(key=lambda r: r["id"])
        return matching[:limit]


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


async def test_run_payment_updater_invalid_message_reraises(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that an invalid message encountered by run_payment_updater is re-raised."""
    bad_msg = FakeConsumerMessage("upi.payments", 0, 100, b"not valid json")

    mock_consumer = AsyncMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()
    mock_consumer.getone = AsyncMock(return_value=bad_msg)

    with patch("features.account_stream.AIOKafkaConsumer", return_value=mock_consumer):
        with pytest.raises(InvalidAccountEventError, match="Failed to decode JSON payload"):
            await run_payment_updater(redis_client, idle_timeout_seconds=1.0)

    # Consumer must be stopped cleanly in finally block
    mock_consumer.stop.assert_awaited_once()


async def test_apply_account_events_since_applies_in_id_order_and_returns_last_id(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """Verifies that apply_account_events_since processes rows in id order and returns the latest id."""
    rows = [
        {
            "id": 10,
            "bank_id": "bank_0",
            "event_id": f"{RUN_ID}-open-acc_01",
            "account_id": "acc_01",
            "kind": "account_opened",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
            "details": {},
        },
        {
            "id": 12,
            "bank_id": "bank_0",
            "event_id": f"{RUN_ID}-dev-000001",
            "account_id": "acc_01",
            "kind": "device_binding_change",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
            "details": {"platform": "ios"},
        },
        {
            "id": 11,
            "bank_id": "bank_0",
            "event_id": f"{RUN_ID}-dev-000002",
            "account_id": "acc_01",
            "kind": "sim_swap",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
            "details": {},
        },
    ]

    conn = FakeAccountEventConnection(rows)

    # Poll with after_id=5 -> should process all 3 in id order (10, 11, 12)
    last_id = await apply_account_events_since(conn, redis_client, bank_id="bank_0", after_id=5)
    assert last_id == 12

    snapshot = read_account_snapshot(redis_client, RUN_ID, "bank_0", "acc_01", NOW_UTC, NOW_UTC)
    assert len(snapshot.account_events) == 3
    event_ids = [e.event_id for e in snapshot.account_events]
    assert f"{RUN_ID}-open-acc_01" in event_ids
    assert f"{RUN_ID}-dev-000001" in event_ids
    assert f"{RUN_ID}-dev-000002" in event_ids


async def test_apply_account_events_since_empty_returns_after_id(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """Verifies that polling when no new rows exist returns after_id unchanged."""
    conn = FakeAccountEventConnection([])
    last_id = await apply_account_events_since(conn, redis_client, bank_id="bank_0", after_id=42)
    assert last_id == 42

