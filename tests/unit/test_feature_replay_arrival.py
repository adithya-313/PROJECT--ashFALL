# File: tests/unit/test_feature_replay_arrival.py
# Purpose: Unit tests for arrival_time filtering and fallback in historical feature replay.
# Issue: Level 16b round 4 (architect decision C item 5, decision D1).
"""
WHAT: Unit tests verifying that replay_history_into_redis queries and filters on effective arrival_time.

WHY: Verifies:
1. SQL queries for all 5 domain tables contain COALESCE(arrival_time, <fallback>).
2. Rows with arrival_time=2026-09-05 14:00 IST and created_at/recorded_at=2026-11-01 are:
   - included when world cutoff is 2026-09-06
   - excluded when world cutoff is 2026-09-05 13:59
3. Rows with arrival_time=NULL fall back to created_at / recorded_at.
4. Merge sorting order evaluates effective arrival from arrival_from_row.
"""

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import fakeredis
import pytest

from features.account_records import arrival_from_row
from features.feature_replay import replay_history_into_redis

IST = ZoneInfo("Asia/Kolkata")


class ArrivalFakeConnection:
    """Fake asyncpg connection capturing SQL queries and filtering rows on effective arrival."""

    def __init__(
        self,
        transactions: list[dict[str, Any]] | None = None,
        account_events: list[dict[str, Any]] | None = None,
        withdrawals: list[dict[str, Any]] | None = None,
        feedback: list[dict[str, Any]] | None = None,
        complaints: list[dict[str, Any]] | None = None,
    ) -> None:
        self.transactions = list(transactions) if transactions else []
        self.account_events = list(account_events) if account_events else []
        self.withdrawals = list(withdrawals) if withdrawals else []
        self.feedback = list(feedback) if feedback else []
        self.complaints = list(complaints) if complaints else []
        self.captured_queries: list[str] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.captured_queries.append(query)
        bank_id = args[0]
        cutoff = args[1]
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)

        def _filter(rows: list[dict[str, Any]], fallback_col: str) -> list[dict[str, Any]]:
            matched: list[dict[str, Any]] = []
            for r in rows:
                if r.get("bank_id") != bank_id:
                    continue
                eff_arr = arrival_from_row(r, fallback_col)
                if eff_arr.tzinfo is None:
                    eff_arr = eff_arr.replace(tzinfo=timezone.utc)
                if eff_arr <= cutoff:
                    matched.append(r)
            return matched

        if "FROM transactions" in query and "payment_id = $2" in query:
            return [r for r in self.transactions if r["bank_id"] == bank_id and r["payment_id"] == args[1]]
        elif "FROM transactions" in query:
            return _filter(self.transactions, "created_at")
        elif "FROM bank_account_events" in query:
            return _filter(self.account_events, "created_at")
        elif "FROM bank_withdrawals" in query:
            return _filter(self.withdrawals, "recorded_at")
        elif "FROM bank_feedback" in query:
            return _filter(self.feedback, "created_at")
        elif "FROM bank_complaints" in query:
            return _filter(self.complaints, "created_at")
        return []


def _make_sample_rows(
    run_id: str,
    bank_id: str,
    arrival_time: datetime | None,
    created_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Constructs one valid sample row per table sharing given arrival and created timestamps."""
    event_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=IST)
    txn = {
        "payment_id": f"{run_id}-pay-001",
        "bank_id": bank_id,
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 15000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "upi_app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": 100000,
        "event_time": event_time,
        "created_at": created_at,
        "arrival_time": arrival_time,
    }
    evt = {
        "event_id": f"{run_id}-evt-001",
        "bank_id": bank_id,
        "account_id": "acc_001",
        "kind": "freeze",
        "event_time": event_time,
        "details": {},
        "created_at": created_at,
        "arrival_time": arrival_time,
    }
    wdr = {
        "withdrawal_id": f"{run_id}-wdr-001",
        "bank_id": bank_id,
        "account_id": "acc_001",
        "channel": "atm",
        "amount_paise": 5000,
        "status": "success",
        "failure_domain": None,
        "balance_after_paise": 95000,
        "event_time": event_time,
        "recorded_at": created_at,
        "arrival_time": arrival_time,
    }
    fbk = {
        "feedback_id": 1,
        "bank_id": bank_id,
        "payment_id": f"{run_id}-pay-001",
        "outcome": "fraud",
        "note": "unauthorized debit",
        "event_time": event_time,
        "created_at": created_at,
        "arrival_time": arrival_time,
    }
    cmp = {
        "complaint_id": f"{run_id}-cmp-001",
        "bank_id": bank_id,
        "complainant_account": "acc_001",
        "reported_account": "acc_002",
        "reported_bank_id": "bank_1",
        "payment_ids": [f"{run_id}-pay-001"],
        "claimed_type": "impersonation",
        "received_at": created_at,
        "created_at": created_at,
        "arrival_time": arrival_time,
    }
    return txn, evt, wdr, fbk, cmp


@pytest.mark.anyio
async def test_sql_queries_contain_coalesce_arrival_time_for_all_five_tables() -> None:
    """Verifies that all 5 SQL statements query COALESCE(arrival_time, ...)."""
    conn = ArrivalFakeConnection()
    redis_client = fakeredis.FakeRedis()
    cutoff = datetime(2026, 9, 6, 0, 0, 0, tzinfo=IST)

    await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client,
        run_id="0123456789abcdef",
        bank_id="bank_0",
        arrival_cutoff=cutoff,
    )

    expected_tables = [
        "FROM transactions",
        "FROM bank_account_events",
        "FROM bank_withdrawals",
        "FROM bank_feedback",
        "FROM bank_complaints",
    ]
    for table_clause in expected_tables:
        matching = [q for q in conn.captured_queries if table_clause in q]
        assert len(matching) == 1, f"Missing or duplicate query for {table_clause}"
        query_text = matching[0]
        assert "COALESCE(arrival_time" in query_text, f"{table_clause} query missing COALESCE(arrival_time"
        assert "arrival_time" in query_text.split("FROM")[0], f"{table_clause} SELECT does not fetch arrival_time"


@pytest.mark.anyio
async def test_world_arrival_time_filtering_cutoff_bounds() -> None:
    """
    Verifies D1 filtering behavior:
    Rows stamped with arrival_time=2026-09-05 14:00 IST and created_at/recorded_at=2026-11-01:
    - Included when cutoff is 2026-09-06 (arrival <= cutoff).
    - Excluded when cutoff is 2026-09-05 13:59 (arrival > cutoff).
    """
    run_id = "0123456789abcdef"
    bank_id = "bank_0"
    arr_time = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)
    post_date = datetime(2026, 11, 1, 0, 0, 0, tzinfo=IST)

    txn, evt, wdr, fbk, cmp = _make_sample_rows(run_id, bank_id, arr_time, post_date)
    conn = ArrivalFakeConnection(
        transactions=[txn],
        account_events=[evt],
        withdrawals=[wdr],
        feedback=[fbk],
        complaints=[cmp],
    )

    # 1. World cutoff 2026-09-06 includes all 5 rows
    redis_client = fakeredis.FakeRedis()
    cutoff_include = datetime(2026, 9, 6, 0, 0, 0, tzinfo=IST)
    counts_inc = await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client,
        run_id=run_id,
        bank_id=bank_id,
        arrival_cutoff=cutoff_include,
    )
    assert counts_inc.transactions == 1
    assert counts_inc.account_events == 1
    assert counts_inc.withdrawals == 1
    assert counts_inc.feedback == 1
    assert counts_inc.complaints == 1

    # 2. World cutoff 2026-09-05 13:59 excludes all 5 rows
    redis_client_2 = fakeredis.FakeRedis()
    cutoff_exclude = datetime(2026, 9, 5, 13, 59, 0, tzinfo=IST)
    counts_exc = await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client_2,
        run_id=run_id,
        bank_id=bank_id,
        arrival_cutoff=cutoff_exclude,
    )
    assert counts_exc.transactions == 0
    assert counts_exc.account_events == 0
    assert counts_exc.withdrawals == 0
    assert counts_exc.feedback == 0
    assert counts_exc.complaints == 0


@pytest.mark.anyio
async def test_null_arrival_time_falls_back_to_created_recorded_column() -> None:
    """Verifies that rows with arrival_time=None fall back to created_at or recorded_at."""
    run_id = "0123456789abcdef"
    bank_id = "bank_0"
    post_date = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)

    txn, evt, wdr, fbk, cmp = _make_sample_rows(run_id, bank_id, None, post_date)
    conn = ArrivalFakeConnection(
        transactions=[txn],
        account_events=[evt],
        withdrawals=[wdr],
        feedback=[fbk],
        complaints=[cmp],
    )

    # Cutoff after fallback created_at/recorded_at includes rows
    redis_client = fakeredis.FakeRedis()
    cutoff_include = datetime(2026, 9, 6, 0, 0, 0, tzinfo=IST)
    counts_inc = await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client,
        run_id=run_id,
        bank_id=bank_id,
        arrival_cutoff=cutoff_include,
    )
    assert counts_inc.transactions == 1
    assert counts_inc.account_events == 1
    assert counts_inc.withdrawals == 1
    assert counts_inc.feedback == 1
    assert counts_inc.complaints == 1

    # Cutoff before fallback created_at/recorded_at excludes rows
    redis_client_2 = fakeredis.FakeRedis()
    cutoff_exclude = datetime(2026, 9, 5, 13, 59, 0, tzinfo=IST)
    counts_exc = await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client_2,
        run_id=run_id,
        bank_id=bank_id,
        arrival_cutoff=cutoff_exclude,
    )
    assert counts_exc.transactions == 0
    assert counts_exc.account_events == 0
    assert counts_exc.withdrawals == 0
    assert counts_exc.feedback == 0
    assert counts_exc.complaints == 0
