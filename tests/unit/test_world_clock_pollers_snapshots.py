# File: tests/unit/test_world_clock_pollers_snapshots.py
# Purpose: Level 16a Round 3b - unit tests for events and withdrawals pollers
# with ingestion clock and run-scoped watermarks.
from datetime import datetime, timezone
from typing import Any

import fakeredis
import pytest

from core.event_clock import IST, ManualClock
from features.account_reader import read_account_snapshot
from features.account_stream import apply_account_events_since, apply_withdrawals_since
from features.account_summaries import read_lifetime

pytestmark = pytest.mark.anyio

RUN_ID = "0123456789abcdef"
BANK_0 = "bank_0"
ACCOUNT_ID = "acc_clock_001"


class FakeDbConnection:
    """A minimal fake database connection tracking query calls and returning stubbed rows."""

    def __init__(
        self,
        fetch_rows: list[dict[str, Any]] | None = None,
        now_val: datetime | None = None,
    ) -> None:
        self.fetch_rows: list[dict[str, Any]] = fetch_rows if fetch_rows is not None else []
        self.now_val: datetime = (
            now_val if now_val is not None else datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
        )
        self.fetchval_calls: list[str] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append(query)
        if "SELECT now()" in query:
            return self.now_val
        raise ValueError(f"Unexpected fetchval query: {query}")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((query, args))
        return self.fetch_rows


async def test_manual_clock_and_run_id_writes_run_scoped_key_and_no_db_now() -> None:
    """
    ManualClock + run_id writes the run key ('acctwm:{run_id}:{bank_id}:{source}'),
    not the shared key, and never calls fetchval('SELECT now()').
    """
    redis_client = fakeredis.FakeRedis()
    conn_events = FakeDbConnection()
    clock = ManualClock(datetime(2026, 9, 5, 12, 0, 0, tzinfo=IST))

    # 1. Events poller
    await apply_account_events_since(
        conn_events,
        redis_client,
        bank_id=BANK_0,
        clock=clock,
        run_id=RUN_ID,
    )
    assert len(conn_events.fetchval_calls) == 0
    assert redis_client.get(f"acctwm:{RUN_ID}:{BANK_0}:events") is not None
    assert redis_client.get(f"acctwm:{BANK_0}:events") is None

    # 2. Withdrawals poller
    conn_wth = FakeDbConnection()
    await apply_withdrawals_since(
        conn_wth,
        redis_client,
        bank_id=BANK_0,
        clock=clock,
        run_id=RUN_ID,
    )
    assert len(conn_wth.fetchval_calls) == 0
    assert redis_client.get(f"acctwm:{RUN_ID}:{BANK_0}:withdrawals") is not None
    assert redis_client.get(f"acctwm:{BANK_0}:withdrawals") is None


async def test_clock_without_run_id_raises() -> None:
    """Providing clock without run_id raises ValueError."""
    redis_client = fakeredis.FakeRedis()
    conn = FakeDbConnection()
    clock = ManualClock(datetime(2026, 9, 5, 12, 0, 0, tzinfo=IST))

    with pytest.raises(ValueError, match="When clock is provided, run_id must also be provided"):
        await apply_account_events_since(conn, redis_client, bank_id=BANK_0, clock=clock, run_id=None)

    with pytest.raises(ValueError, match="When clock is provided, run_id must also be provided"):
        await apply_withdrawals_since(conn, redis_client, bank_id=BANK_0, clock=clock, run_id=None)


async def test_run_id_without_clock_raises() -> None:
    """Providing run_id without clock raises ValueError."""
    redis_client = fakeredis.FakeRedis()
    conn = FakeDbConnection()

    with pytest.raises(ValueError, match="run_id cannot be provided when clock is None"):
        await apply_account_events_since(conn, redis_client, bank_id=BANK_0, clock=None, run_id=RUN_ID)

    with pytest.raises(ValueError, match="run_id cannot be provided when clock is None"):
        await apply_withdrawals_since(conn, redis_client, bank_id=BANK_0, clock=None, run_id=RUN_ID)


async def test_clock_none_calls_db_now_and_writes_shared_key() -> None:
    """When clock is None, calls fetchval('SELECT now()') and writes legacy shared key."""
    redis_client = fakeredis.FakeRedis()
    conn_events = FakeDbConnection()

    await apply_account_events_since(
        conn_events,
        redis_client,
        bank_id=BANK_0,
        clock=None,
        run_id=None,
    )
    assert any("SELECT now()" in q for q in conn_events.fetchval_calls)
    assert redis_client.get(f"acctwm:{BANK_0}:events") is not None
    assert redis_client.get(f"acctwm:{RUN_ID}:{BANK_0}:events") is None

    conn_wth = FakeDbConnection()
    await apply_withdrawals_since(
        conn_wth,
        redis_client,
        bank_id=BANK_0,
        clock=None,
        run_id=None,
    )
    assert any("SELECT now()" in q for q in conn_wth.fetchval_calls)
    assert redis_client.get(f"acctwm:{BANK_0}:withdrawals") is not None
    assert redis_client.get(f"acctwm:{RUN_ID}:{BANK_0}:withdrawals") is None


async def test_poller_arrival_time_selected_and_visible_to_decision() -> None:
    """
    A row with arrival_time 2026-09-05 14:00 IST and created_at 2026-11-01
    makes the event/withdrawal visible to a 2026-09-06 decision.
    Proves that the SELECT query carries arrival_time end to end.
    """
    redis_client = fakeredis.FakeRedis()
    clock = ManualClock(datetime(2026, 9, 5, 15, 0, 0, tzinfo=IST))

    event_row = {
        "id": 1,
        "bank_id": BANK_0,
        "event_id": f"{RUN_ID}-ev-clock-01",
        "account_id": ACCOUNT_ID,
        "kind": "device_binding_change",
        "event_time": datetime(2026, 9, 5, 12, 0, 0, tzinfo=IST),
        "details": '{"fingerprint": "fp_world_clock_test"}',
        "created_at": datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc),
        "arrival_time": datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST),
    }

    withdrawal_row = {
        "id": 2,
        "bank_id": BANK_0,
        "withdrawal_id": f"{RUN_ID}-wth-clock-01",
        "account_id": ACCOUNT_ID,
        "channel": "atm",
        "amount_paise": 75000,
        "status": "success",
        "failure_domain": None,
        "event_time": datetime(2026, 9, 5, 12, 0, 0, tzinfo=IST),
        "balance_after_paise": 500000,
        "recorded_at": datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc),
        "arrival_time": datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST),
    }

    conn_events = FakeDbConnection(fetch_rows=[event_row])
    await apply_account_events_since(
        conn_events,
        redis_client,
        bank_id=BANK_0,
        clock=clock,
        run_id=RUN_ID,
    )
    # Check that SELECT includes arrival_time
    assert any("arrival_time" in q for q, _ in conn_events.fetch_calls)

    conn_wth = FakeDbConnection(fetch_rows=[withdrawal_row])
    await apply_withdrawals_since(
        conn_wth,
        redis_client,
        bank_id=BANK_0,
        clock=clock,
        run_id=RUN_ID,
    )
    # Check that SELECT includes arrival_time
    assert any("arrival_time" in q for q, _ in conn_wth.fetch_calls)

    # Decision at 2026-09-06 (future) sees both because arrival_time was 2026-09-05 14:00
    future_decision_time = datetime(2026, 9, 6, 12, 0, 0, tzinfo=IST)
    future_cutoff = datetime(2026, 9, 6, 12, 0, 0, tzinfo=IST)

    summary_future = read_lifetime(
        redis_client, RUN_ID, BANK_0, ACCOUNT_ID, future_decision_time, future_cutoff
    )
    assert "fp_world_clock_test" in summary_future.fingerprints

    snap_future = read_account_snapshot(
        redis_client, RUN_ID, BANK_0, ACCOUNT_ID, future_decision_time, future_cutoff
    )
    assert any(w.withdrawal_id == f"{RUN_ID}-wth-clock-01" for w in snap_future.withdrawals)

    # Decision at 2026-09-05 13:59 (before arrival 14:00) sees neither
    early_decision_time = datetime(2026, 9, 5, 13, 59, 0, tzinfo=IST)
    early_cutoff = datetime(2026, 9, 5, 13, 59, 0, tzinfo=IST)

    summary_early = read_lifetime(
        redis_client, RUN_ID, BANK_0, ACCOUNT_ID, early_decision_time, early_cutoff
    )
    assert "fp_world_clock_test" not in summary_early.fingerprints

    snap_early = read_account_snapshot(
        redis_client, RUN_ID, BANK_0, ACCOUNT_ID, early_decision_time, early_cutoff
    )
    assert not any(w.withdrawal_id == f"{RUN_ID}-wth-clock-01" for w in snap_early.withdrawals)
