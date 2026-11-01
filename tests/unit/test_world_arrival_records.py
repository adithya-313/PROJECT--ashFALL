# File: tests/unit/test_world_arrival_records.py
# Purpose: Level 16a Round 3a - unit tests for world arrival timestamp parsing, visibility, and fallbacks.
from datetime import datetime, timezone
from typing import Any

import fakeredis
import pytest

from core.event_clock import IST
from features.account_keys import get_default_settings
from features.account_reader import read_account_snapshot
from features.account_records import (
    account_event_from_row,
    arrival_from_row,
    payment_side_from_transaction_row,
    withdrawal_from_row,
)
from features.account_summaries import (
    read_account_type,
    read_home_region,
    read_lifetime,
)
from features.account_updater import (
    apply_account_event,
    apply_payment_side,
    apply_withdrawal,
)

pytestmark = pytest.mark.anyio

RUN_ID = "0123456789abcdef"
BANK_ID = "bank_0"
ACCOUNT_ID = "acc_world_01"

# Domain event occurred at 13:30, ingested via world-clock at 14:00, database audit clock is 2026-11-01 (real now)
EVENT_TIME = datetime(2026, 9, 5, 13, 30, 0, tzinfo=IST)
ARRIVAL_TIME = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)
REAL_NOW = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)

# Decisions
DECISION_TIME_FUTURE = datetime(2026, 9, 6, 12, 0, 0, tzinfo=IST)
CUTOFF_FUTURE = datetime(2026, 9, 6, 12, 0, 0, tzinfo=IST)

DECISION_TIME_EARLY = datetime(2026, 9, 5, 13, 59, 0, tzinfo=IST)
CUTOFF_EARLY = datetime(2026, 9, 5, 13, 59, 0, tzinfo=IST)


def test_payment_side_visibility_via_arrival_time() -> None:
    """
    Verifies that a payment side parsed from a transactions row with arrival_time:
    - has arrival_time equal to world clock arrival_time, NOT created_at (real now).
    - is visible to the decision path reader (read_account_snapshot) at 2026-09-06.
    - is INVISIBLE to a decision at 2026-09-05 13:59 (before arrival_time 14:00).
    """
    redis_client = fakeredis.FakeRedis()
    cfg = get_default_settings()

    row = {
        "payment_id": f"{RUN_ID}-pmt-vis",
        "bank_id": BANK_ID,
        "side": "debit",
        "account_id": ACCOUNT_ID,
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "dest@bank1",
        "amount_paise": 150000,
        "event_time": EVENT_TIME,
        "created_at": REAL_NOW,
        "arrival_time": ARRIVAL_TIME,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "upi_app",
        "sender_device_fingerprint": "fp_tx_01",
        "sender_os_generation": 2,
        "balance_after_paise": 850000,
    }

    rec = payment_side_from_transaction_row(row)
    assert rec.arrival_time == ARRIVAL_TIME
    assert rec.arrival_from_ingestion_clock is True

    apply_payment_side(redis_client, rec, cfg)

    # Visible at future decision
    snap_future = read_account_snapshot(
        redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_FUTURE, CUTOFF_FUTURE
    )
    assert any(r.payment_id == rec.payment_id for r in snap_future.side_records)

    # Invisible at early decision (13:59 is before arrival 14:00)
    snap_early = read_account_snapshot(
        redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_EARLY, CUTOFF_EARLY
    )
    assert not any(r.payment_id == rec.payment_id for r in snap_early.side_records)


def test_withdrawal_visibility_via_arrival_time() -> None:
    """
    Verifies that a cash withdrawal parsed from bank_withdrawals row with arrival_time:
    - has arrival_time equal to world clock arrival_time, NOT recorded_at (real now).
    - is visible to read_account_snapshot at 2026-09-06.
    - is INVISIBLE to a decision at 2026-09-05 13:59.
    """
    redis_client = fakeredis.FakeRedis()
    cfg = get_default_settings()

    row = {
        "withdrawal_id": f"{RUN_ID}-wth-vis",
        "bank_id": BANK_ID,
        "account_id": ACCOUNT_ID,
        "channel": "atm",
        "amount_paise": 50000,
        "status": "success",
        "failure_domain": None,
        "event_time": EVENT_TIME,
        "recorded_at": REAL_NOW,
        "arrival_time": ARRIVAL_TIME,
        "balance_after_paise": 800000,
    }

    rec_wth = withdrawal_from_row(row)
    assert rec_wth.arrival_time == ARRIVAL_TIME

    apply_withdrawal(redis_client, rec_wth, cfg)

    # Visible at future decision
    snap_future = read_account_snapshot(
        redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_FUTURE, CUTOFF_FUTURE
    )
    assert any(w.withdrawal_id == rec_wth.withdrawal_id for w in snap_future.withdrawals)

    # Invisible at early decision
    snap_early = read_account_snapshot(
        redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_EARLY, CUTOFF_EARLY
    )
    assert not any(w.withdrawal_id == rec_wth.withdrawal_id for w in snap_early.withdrawals)


def test_fingerprint_from_account_event_visibility() -> None:
    """
    Verifies that a device_binding_change account event with arrival_time:
    - indexes the device fingerprint with available_from = arrival_time.
    - is visible in read_lifetime_summary at 2026-09-06.
    - is INVISIBLE to a decision at 2026-09-05 13:59.
    """
    redis_client = fakeredis.FakeRedis()
    cfg = get_default_settings()

    row = {
        "event_id": f"{RUN_ID}-ev-dev-01",
        "bank_id": BANK_ID,
        "account_id": ACCOUNT_ID,
        "kind": "device_binding_change",
        "event_time": EVENT_TIME,
        "created_at": REAL_NOW,
        "arrival_time": ARRIVAL_TIME,
        "details": {"fingerprint": "fp_dev_binding_01"},
    }

    rec_dev = account_event_from_row(row)
    assert rec_dev.arrival_time == ARRIVAL_TIME

    apply_account_event(redis_client, rec_dev, cfg)

    # Visible at future decision
    summary_future = read_lifetime(
        redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_FUTURE, CUTOFF_FUTURE
    )
    assert "fp_dev_binding_01" in summary_future.fingerprints

    # Invisible at early decision
    summary_early = read_lifetime(
        redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_EARLY, CUTOFF_EARLY
    )
    assert "fp_dev_binding_01" not in summary_early.fingerprints


def test_account_opening_visibility() -> None:
    """
    Verifies that an account_opened event with arrival_time:
    - opened_at, home_region, and account_type are visible at 2026-09-06.
    - each of them is INVISIBLE to a decision at 2026-09-05 13:59.
    """
    redis_client = fakeredis.FakeRedis()
    cfg = get_default_settings()

    row = {
        "event_id": f"{RUN_ID}-ev-open-01",
        "bank_id": BANK_ID,
        "account_id": ACCOUNT_ID,
        "kind": "account_opened",
        "event_time": EVENT_TIME,
        "created_at": REAL_NOW,
        "arrival_time": ARRIVAL_TIME,
        "details": {"home_region": "KA", "account_type": "savings"},
    }

    rec_open = account_event_from_row(row)
    assert rec_open.arrival_time == ARRIVAL_TIME

    apply_account_event(redis_client, rec_open, cfg)

    # Visible at future decision
    summary_future = read_lifetime(
        redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_FUTURE, CUTOFF_FUTURE
    )
    assert summary_future.opened_at is not None
    assert (
        read_home_region(redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_FUTURE, CUTOFF_FUTURE)
        == "KA"
    )
    assert (
        read_account_type(redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_FUTURE, CUTOFF_FUTURE)
        == "savings"
    )

    # Invisible at early decision
    summary_early = read_lifetime(
        redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_EARLY, CUTOFF_EARLY
    )
    assert summary_early.opened_at is None
    assert (
        read_home_region(redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_EARLY, CUTOFF_EARLY)
        is None
    )
    assert (
        read_account_type(redis_client, RUN_ID, BANK_ID, ACCOUNT_ID, DECISION_TIME_EARLY, CUTOFF_EARLY)
        is None
    )


def test_null_arrival_time_falls_back() -> None:
    """
    Verifies fallback when arrival_time is NULL:
    - account event falls back to created_at.
    - withdrawal falls back to recorded_at.
    """
    # 1. Event fallback
    event_row: dict[str, Any] = {
        "event_id": f"{RUN_ID}-ev-fallback",
        "bank_id": BANK_ID,
        "account_id": ACCOUNT_ID,
        "kind": "device_binding_change",
        "event_time": EVENT_TIME,
        "created_at": REAL_NOW,
        "arrival_time": None,
        "details": {},
    }
    ev_rec = account_event_from_row(event_row)
    assert ev_rec.arrival_time == REAL_NOW

    # 2. Withdrawal fallback
    wth_row: dict[str, Any] = {
        "withdrawal_id": f"{RUN_ID}-wth-fallback",
        "bank_id": BANK_ID,
        "account_id": ACCOUNT_ID,
        "channel": "atm",
        "amount_paise": 20000,
        "status": "success",
        "failure_domain": None,
        "event_time": EVENT_TIME,
        "recorded_at": REAL_NOW,
        "arrival_time": None,
    }
    wth_rec = withdrawal_from_row(wth_row)
    assert wth_rec.arrival_time == REAL_NOW

    # 3. Direct arrival_from_row test
    assert arrival_from_row({"arrival_time": None, "created_at": REAL_NOW}, "created_at") == REAL_NOW
    assert (
        arrival_from_row({"arrival_time": ARRIVAL_TIME, "created_at": REAL_NOW}, "created_at")
        == ARRIVAL_TIME
    )
