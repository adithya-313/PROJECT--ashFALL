# File: tests/unit/test_account_withdrawals_state.py
# Purpose: Unit tests for cash withdrawals account state, point-in-time filtering, balances, and overlap polling.
# Issue: Level 15b cash-out state and long-history summaries (plan Task 1, G2, R7).
"""
WHAT: Tests the withdrawal ingestion, point-in-time snapshotting, balance parsing,
and overlap polling mechanisms for bank cash withdrawals.

WHY: Verifies that cash-out events (ATM and branch withdrawals, G2) are properly captured
in Redis account state, obey the Section 22 future-leak rules, maintain replay idempotency,
and protect against transaction commit races in relational polling.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import orjson
import pytest

from features.account_keys import load_account_features_settings
from features.account_reader import read_account_snapshot
from features.account_records import (
    InvalidAccountEventError,
    WithdrawalRecord,
    parse_upi_event,
    withdrawal_from_row,
)
from features.account_stream import apply_withdrawals_since
from features.account_updater import apply_withdrawal

pytestmark = pytest.mark.anyio

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
ACC_1 = "acc_person_0001"
BASE_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


class FakeWithdrawalsConnection:
    """In-memory mock database connection simulating bank_withdrawals table reads."""

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


def _make_withdrawal(
    withdrawal_id: str,
    event_time: datetime,
    arrival_time: datetime,
    amount_paise: int = 50000,
    status: str = "success",
    failure_domain: str | None = None,
    balance_after_paise: int | None = 250000,
    channel: str = "atm",
) -> WithdrawalRecord:
    """Helper constructing a valid WithdrawalRecord."""
    return WithdrawalRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        withdrawal_id=withdrawal_id,
        channel=channel,
        amount_paise=amount_paise,
        status=status,
        failure_domain=failure_domain,
        balance_after_paise=balance_after_paise,
        event_time=event_time,
        arrival_time=arrival_time,
    )


def _dump_all_keys(client: fakeredis.FakeRedis, pattern: str) -> dict[bytes, Any]:
    """Helper dumping sorted keys and structures for byte-identical comparison."""
    dump: dict[bytes, Any] = {}
    for k in sorted(client.scan_iter(match=pattern)):
        k_type = client.type(k)
        if k_type == b"zset":
            dump[k] = (k_type, client.zrange(k, 0, -1, withscores=True))
        elif k_type == b"hash":
            raw_hash: Any = client.hgetall(k)
            dump[k] = (k_type, sorted(raw_hash.items()))
        else:
            dump[k] = (k_type, client.get(k))
    return dump


def test_withdrawal_stored_twice_gives_identical_state(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that applying the exact same withdrawal multiple times leaves Redis state byte-identical."""
    settings = load_account_features_settings()
    w_rec = _make_withdrawal(f"{RUN_ID}-w-000001", BASE_TIME, BASE_TIME)

    res1 = apply_withdrawal(redis_client, w_rec, settings)
    assert res1 == "stored"
    dump1 = _dump_all_keys(redis_client, f"acctf:{RUN_ID}:*")

    res2 = apply_withdrawal(redis_client, w_rec, settings)
    assert res2 == "stored"
    dump2 = _dump_all_keys(redis_client, f"acctf:{RUN_ID}:*")

    assert dump1 == dump2


def test_snapshot_includes_withdrawal_only_when_event_and_arrival_in_bounds(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """Verifies that withdrawals are admitted into snapshot only when event_time <= decision and arrival <= cutoff."""
    settings = load_account_features_settings()
    decision_time = BASE_TIME
    arrival_cutoff = BASE_TIME

    # w1: valid, in-bounds -> included
    w1 = _make_withdrawal(f"{RUN_ID}-w-000001", decision_time - timedelta(hours=2), arrival_cutoff - timedelta(hours=1))

    # w2: future event_time > decision_time -> excluded
    w2 = _make_withdrawal(
        f"{RUN_ID}-w-000002", decision_time + timedelta(minutes=5), arrival_cutoff - timedelta(hours=1)
    )

    # w3: future storage arrival_time > arrival_cutoff -> excluded
    w3 = _make_withdrawal(
        f"{RUN_ID}-w-000003", decision_time - timedelta(hours=1), arrival_cutoff + timedelta(minutes=10)
    )

    # w4: event_time older than retention window -> excluded
    w4 = _make_withdrawal(
        f"{RUN_ID}-w-000004",
        decision_time - timedelta(seconds=settings.retention_seconds + 3600),
        arrival_cutoff - timedelta(hours=1),
    )

    for w in (w1, w2, w3, w4):
        apply_withdrawal(redis_client, w, settings)

    snap = read_account_snapshot(redis_client, RUN_ID, BANK_0, ACC_1, decision_time, arrival_cutoff, settings)
    assert len(snap.withdrawals) == 1
    assert snap.withdrawals[0].withdrawal_id == f"{RUN_ID}-w-000001"
    assert snap.withdrawals[0].balance_after_paise == 250000


def test_failed_withdrawal_keeps_failure_domain_and_balance_none(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """Verifies that failed withdrawals preserve failure_domain and have None balance."""
    settings = load_account_features_settings()
    w_failed = _make_withdrawal(
        f"{RUN_ID}-w-failed-001",
        BASE_TIME,
        BASE_TIME,
        status="failed",
        failure_domain="atm_hardware",
        balance_after_paise=None,
    )

    apply_withdrawal(redis_client, w_failed, settings)
    snap = read_account_snapshot(redis_client, RUN_ID, BANK_0, ACC_1, BASE_TIME, BASE_TIME, settings)

    assert len(snap.withdrawals) == 1
    rec = snap.withdrawals[0]
    assert rec.status == "failed"
    assert rec.failure_domain == "atm_hardware"
    assert rec.balance_after_paise is None


def test_balance_after_paise_parses_and_rejects_negative_values() -> None:
    """Verifies that balance_after_paise parses valid integer amounts and rejects negative or malformed values."""
    envelope: dict[str, Any] = {
        "event_type": "upi.payment.recorded",
        "schema_version": "2",
        "payment_id": f"{RUN_ID}-000042",
        "side": "debit",
        "event_time": BASE_TIME.isoformat(),
        "recorded_at": BASE_TIME.isoformat(),
        "data": {
            "bank_id": BANK_0,
            "account_id": ACC_1,
            "counterparty_bank_id": "bank_1",
            "amount_paise": 10000,
            "status": "success",
        },
    }

    # 1. Absent balance -> parsed as None
    rec_absent = parse_upi_event(orjson.dumps(envelope))
    assert rec_absent.balance_after_paise is None

    # 2. Valid positive and zero balance
    base_data: dict[str, Any] = dict(envelope["data"])
    data_pos = dict(base_data)
    data_pos["balance_after_paise"] = 500000
    env_pos = dict(envelope)
    env_pos["data"] = data_pos
    rec_pos = parse_upi_event(orjson.dumps(env_pos))
    assert rec_pos.balance_after_paise == 500000

    data_zero = dict(base_data)
    data_zero["balance_after_paise"] = 0
    env_zero = dict(envelope)
    env_zero["data"] = data_zero
    rec_zero = parse_upi_event(orjson.dumps(env_zero))
    assert rec_zero.balance_after_paise == 0

    # 3. Negative balance -> InvalidAccountEventError
    data_neg = dict(base_data)
    data_neg["balance_after_paise"] = -100
    env_neg = dict(envelope)
    env_neg["data"] = data_neg
    with pytest.raises(InvalidAccountEventError, match="Invalid 'balance_after_paise'"):
        parse_upi_event(orjson.dumps(env_neg))

    # 4. Non-int or boolean -> InvalidAccountEventError
    data_str: dict[str, Any] = dict(base_data)
    data_str["balance_after_paise"] = "50000"
    env_str = dict(envelope)
    env_str["data"] = data_str
    with pytest.raises(InvalidAccountEventError, match="Invalid 'balance_after_paise'"):
        parse_upi_event(orjson.dumps(env_str))

    data_bool: dict[str, Any] = dict(base_data)
    data_bool["balance_after_paise"] = True
    env_bool = dict(envelope)
    env_bool["data"] = data_bool
    with pytest.raises(InvalidAccountEventError, match="Invalid 'balance_after_paise'"):
        parse_upi_event(orjson.dumps(env_bool))

    # 5. withdrawal_from_row validates balance_after_paise similarly
    valid_row = {
        "withdrawal_id": f"{RUN_ID}-w-001",
        "bank_id": BANK_0,
        "account_id": ACC_1,
        "channel": "atm",
        "amount_paise": 20000,
        "status": "success",
        "balance_after_paise": 80000,
        "event_time": BASE_TIME,
        "recorded_at": BASE_TIME,
    }
    w_parsed = withdrawal_from_row(valid_row)
    assert w_parsed.balance_after_paise == 80000

    bad_row_neg = dict(valid_row, balance_after_paise=-50)
    with pytest.raises(ValueError, match="Invalid balance_after_paise"):
        withdrawal_from_row(bad_row_neg)


async def test_overlap_poller_catches_row_whose_commit_came_late(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """Verifies that apply_withdrawals_since re-reads the overlap and catches late-committing lower-ID rows."""
    settings = load_account_features_settings()

    # Initial state: poller previously read row id 10
    prev_after_id = 10

    # Two rows now visible: row 12 and row 8 (which committed after id 10 due to concurrent transactions)
    rows = [
        {
            "id": 8,
            "bank_id": BANK_0,
            "withdrawal_id": f"{RUN_ID}-w-000008",
            "account_id": ACC_1,
            "channel": "atm",
            "amount_paise": 25000,
            "status": "success",
            "balance_after_paise": 100000,
            "event_time": BASE_TIME - timedelta(minutes=10),
            "recorded_at": BASE_TIME,
        },
        {
            "id": 12,
            "bank_id": BANK_0,
            "withdrawal_id": f"{RUN_ID}-w-000012",
            "account_id": ACC_1,
            "channel": "branch",
            "amount_paise": 75000,
            "status": "success",
            "balance_after_paise": 25000,
            "event_time": BASE_TIME,
            "recorded_at": BASE_TIME,
        },
    ]

    conn = FakeWithdrawalsConnection(rows)

    # Poller with overlap re-reads from start_id = max(0, 10 - 200) = 0
    # Both row 8 and row 12 must be read and applied
    new_after_id = await apply_withdrawals_since(conn, redis_client, settings, bank_id=BANK_0, after_id=prev_after_id)
    assert new_after_id == 12

    snap = read_account_snapshot(redis_client, RUN_ID, BANK_0, ACC_1, BASE_TIME, BASE_TIME, settings)
    assert len(snap.withdrawals) == 2
    w_ids = [w.withdrawal_id for w in snap.withdrawals]
    assert f"{RUN_ID}-w-000008" in w_ids
    assert f"{RUN_ID}-w-000012" in w_ids
