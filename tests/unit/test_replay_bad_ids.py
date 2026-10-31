# File: tests/unit/test_replay_bad_ids.py
# Purpose: Unit tests proving unparseable IDs raise ValueError with table and row ID, while other runs are skipped.
# Issue: Level 15c build round (F1: remove silent fallbacks around extract_run_id).
"""
WHAT: Unit tests verifying that malformed or unparseable record identifiers encountered during
feature replay and snapshot account selection raise a plain ValueError containing the database
table name and the invalid row ID, while records belonging to distinct valid runs are skipped
strictly through explicit equality comparison.

WHY: Satisfies Section 22 code honesty and multi-run isolation rules:
1. No silent fallbacks: unparseable identifiers indicate corrupted storage state or schema mismatch,
   which must fail loudly rather than being swallowed with except ValueError: continue/pass.
2. Strict isolation: rows from other valid simulation runs must be skipped explicitly without
   raising exceptions.
"""

from datetime import datetime, timezone
from typing import Any

import fakeredis
import pytest

from features.account_snapshots import select_all_accounts, select_daily_snapshot_accounts
from features.feature_replay import replay_history_into_redis

pytestmark = pytest.mark.anyio


class MockPgConnection:
    """Mock asyncpg connection returning pre-configured row sets per SQL query."""

    def __init__(
        self,
        transactions: list[dict[str, Any]] | None = None,
        account_events: list[dict[str, Any]] | None = None,
        withdrawals: list[dict[str, Any]] | None = None,
        feedback: list[dict[str, Any]] | None = None,
        complaints: list[dict[str, Any]] | None = None,
    ) -> None:
        self.transactions = transactions or []
        self.account_events = account_events or []
        self.withdrawals = withdrawals or []
        self.feedback = feedback or []
        self.complaints = complaints or []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """Dispatch query to matching in-memory table rows."""
        if "FROM transactions" in query:
            if "payment_id = $2" in query:
                pid = args[1]
                return [r for r in self.transactions if r.get("payment_id") == pid]
            return list(self.transactions)
        elif "FROM bank_account_events" in query:
            return list(self.account_events)
        elif "FROM bank_withdrawals" in query:
            return list(self.withdrawals)
        elif "FROM bank_feedback" in query:
            return list(self.feedback)
        elif "FROM bank_complaints" in query:
            return list(self.complaints)
        return []


async def test_replay_unparseable_transactions_id_raises() -> None:
    """Verifies that an unparseable payment_id in transactions raises ValueError naming transactions."""
    bad_id = "malformed_payment_id_without_hex_prefix"
    row = {
        "payment_id": bad_id,
        "bank_id": "bank_0",
        "account_id": "acc_01",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "vpa_01",
        "amount_paise": 1000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "upi_app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": datetime(2026, 10, 28, 10, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 10, 1, tzinfo=timezone.utc),
    }
    conn = MockPgConnection(transactions=[row])
    redis_client = fakeredis.FakeRedis()

    with pytest.raises(ValueError, match="transactions") as exc_info:
        await replay_history_into_redis(
            pg_connection=conn,
            scratch_redis=redis_client,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            arrival_cutoff=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
        )
    assert bad_id in str(exc_info.value)


async def test_replay_unparseable_events_id_raises() -> None:
    """Verifies that an unparseable event_id in bank_account_events raises ValueError naming bank_account_events."""
    bad_id = "bad_event_id"
    row = {
        "event_id": bad_id,
        "bank_id": "bank_0",
        "account_id": "acc_01",
        "kind": "account_opened",
        "event_time": datetime(2026, 10, 28, 10, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 10, 1, tzinfo=timezone.utc),
        "details": {},
    }
    conn = MockPgConnection(account_events=[row])
    redis_client = fakeredis.FakeRedis()

    with pytest.raises(ValueError, match="bank_account_events") as exc_info:
        await replay_history_into_redis(
            pg_connection=conn,
            scratch_redis=redis_client,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            arrival_cutoff=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
        )
    assert bad_id in str(exc_info.value)


async def test_replay_unparseable_withdrawals_id_raises() -> None:
    """Verifies that an unparseable withdrawal_id in bank_withdrawals raises ValueError naming bank_withdrawals."""
    bad_id = "bad_withdrawal_id"
    row = {
        "withdrawal_id": bad_id,
        "bank_id": "bank_0",
        "account_id": "acc_01",
        "channel": "atm",
        "amount_paise": 50000,
        "status": "success",
        "failure_domain": None,
        "event_time": datetime(2026, 10, 28, 10, 0, tzinfo=timezone.utc),
        "balance_after_paise": 100000,
        "recorded_at": datetime(2026, 10, 28, 10, 1, tzinfo=timezone.utc),
    }
    conn = MockPgConnection(withdrawals=[row])
    redis_client = fakeredis.FakeRedis()

    with pytest.raises(ValueError, match="bank_withdrawals") as exc_info:
        await replay_history_into_redis(
            pg_connection=conn,
            scratch_redis=redis_client,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            arrival_cutoff=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
        )
    assert bad_id in str(exc_info.value)


async def test_replay_unparseable_feedback_id_raises() -> None:
    """Verifies that an unparseable payment_id in bank_feedback raises ValueError naming bank_feedback."""
    bad_id = "bad_feedback_payment_id"
    row = {
        "feedback_id": "0123456789abcdef-fb-01",
        "bank_id": "bank_0",
        "payment_id": bad_id,
        "outcome": "confirmed_fraud",
        "event_time": datetime(2026, 10, 28, 10, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 10, 1, tzinfo=timezone.utc),
    }
    conn = MockPgConnection(feedback=[row])
    redis_client = fakeredis.FakeRedis()

    with pytest.raises(ValueError, match="bank_feedback") as exc_info:
        await replay_history_into_redis(
            pg_connection=conn,
            scratch_redis=redis_client,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            arrival_cutoff=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
        )
    assert bad_id in str(exc_info.value)


async def test_replay_unparseable_complaints_id_raises() -> None:
    """Verifies that an unparseable complaint_id in bank_complaints raises ValueError naming bank_complaints."""
    bad_id = "bad_complaint_id"
    row = {
        "complaint_id": bad_id,
        "bank_id": "bank_0",
        "complainant_account": "acc_01",
        "reported_account": "acc_02",
        "reported_bank_id": "bank_0",
        "payment_ids": [],
        "claimed_type": "fraud",
        "received_at": datetime(2026, 10, 28, 10, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 10, 1, tzinfo=timezone.utc),
    }
    conn = MockPgConnection(complaints=[row])
    redis_client = fakeredis.FakeRedis()

    with pytest.raises(ValueError, match="bank_complaints") as exc_info:
        await replay_history_into_redis(
            pg_connection=conn,
            scratch_redis=redis_client,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            arrival_cutoff=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
        )
    assert bad_id in str(exc_info.value)


async def test_replay_skips_other_run_rows_by_explicit_comparison() -> None:
    """Verifies that rows belonging to other valid runs are skipped explicitly and not applied."""
    target_run = "0123456789abcdef"
    other_run = "fedcba9876543210"

    row_other = {
        "payment_id": f"{other_run}-pay-001",
        "bank_id": "bank_0",
        "account_id": "acc_other",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "vpa_other",
        "amount_paise": 1000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "upi_app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": datetime(2026, 10, 28, 10, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 10, 1, tzinfo=timezone.utc),
    }
    row_target = {
        "payment_id": f"{target_run}-pay-002",
        "bank_id": "bank_0",
        "account_id": "acc_target",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "vpa_target",
        "amount_paise": 2000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "upi_app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": datetime(2026, 10, 28, 10, 5, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 10, 6, tzinfo=timezone.utc),
    }

    conn = MockPgConnection(transactions=[row_other, row_target])
    redis_client = fakeredis.FakeRedis()

    counts = await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client,
        run_id=target_run,
        bank_id="bank_0",
        arrival_cutoff=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
    )
    # Only the target run's transaction was applied
    assert counts.transactions == 1


async def test_snapshots_select_daily_unparseable_id_raises() -> None:
    """Verifies that unparseable IDs in select_daily_snapshot_accounts raise ValueError."""
    bad_id = "malformed_event_id_xyz"
    row = {
        "account_id": "acc_01",
        "event_id": bad_id,
    }
    conn = MockPgConnection(account_events=[row])

    t_start = datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 10, 29, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="bank_account_events") as exc_info:
        await select_daily_snapshot_accounts(
            pg=conn,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            day_start=t_start,
            day_end=t_end,
        )
    assert bad_id in str(exc_info.value)


async def test_snapshots_select_daily_skips_other_run_rows() -> None:
    """Verifies that select_daily_snapshot_accounts skips rows belonging to other runs."""
    target_run = "0123456789abcdef"
    other_run = "fedcba9876543210"

    event_other = {"account_id": "acc_other", "event_id": f"{other_run}-evt-01"}
    event_target = {"account_id": "acc_target", "event_id": f"{target_run}-evt-02"}
    conn = MockPgConnection(account_events=[event_other, event_target])

    t_start = datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 10, 29, 0, 0, tzinfo=timezone.utc)

    accounts = await select_daily_snapshot_accounts(
        pg=conn,
        run_id=target_run,
        bank_id="bank_0",
        day_start=t_start,
        day_end=t_end,
    )
    assert accounts == ["acc_target"]


async def test_snapshots_select_all_unparseable_id_raises() -> None:
    """Verifies that unparseable IDs in select_all_accounts raise ValueError with table and row ID."""
    bad_id = "bad_event_id_no_dash"
    row = {"account_id": "acc_01", "event_id": bad_id}
    conn = MockPgConnection(account_events=[row])

    cutoff = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="bank_account_events") as exc_info:
        await select_all_accounts(
            pg=conn,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            arrival_cutoff=cutoff,
        )
    assert bad_id in str(exc_info.value)


async def test_snapshots_select_all_skips_other_run_rows() -> None:
    """Verifies that select_all_accounts skips rows belonging to other runs."""
    target_run = "0123456789abcdef"
    other_run = "fedcba9876543210"

    event_other = {"account_id": "acc_other", "event_id": f"{other_run}-evt-01"}
    event_target = {"account_id": "acc_target", "event_id": f"{target_run}-evt-02"}
    conn = MockPgConnection(account_events=[event_other, event_target])

    cutoff = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
    accounts = await select_all_accounts(
        pg=conn,
        run_id=target_run,
        bank_id="bank_0",
        arrival_cutoff=cutoff,
    )
    assert accounts == ["acc_target"]
