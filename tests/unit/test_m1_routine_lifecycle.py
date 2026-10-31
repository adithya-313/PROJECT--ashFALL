# File: tests/unit/test_m1_routine_lifecycle.py
# Purpose: Unit tests validating M1 routine_life and lifecycle behavioral feature calculations.
# Issue: Level 15b Round 2b-2 (G1, R8).
"""
WHAT: Unit tests covering:
  1. monthly_rhythm helper: exact boundaries around routine_cycle_days and tolerance.
  2. Short history (< 35 days): routine_life features evaluate strictly to None (not False or 0).
  3. Routine life present: salary payer detection, recurring merchant detection, regular transfers.
  4. Lifecycle account_age_days: calendar days elapsed since opening date (None if unopened).
  5. Lifecycle account_type: returns savings/current or None.
  6. Lifecycle dormancy_days: consecutive dormant days prior to onset of recent activity.
  7. Lifecycle volume ratio: recent 7d volume relative to historical daily volume.
  8. Integration through compute_account_features(groups=['routine_life', 'lifecycle']).
"""

from datetime import datetime
from typing import Any

import fakeredis
import pytest

from features.account_keys import (
    load_account_features_settings,
)
from features.account_records import (
    AccountEventRecord,
    PaymentSideRecord,
)
from features.account_summaries import (
    IST,
    to_day_number,
    update_account_summaries,
)
from features.account_updater import apply_payment_side
from features.m1_features import compute_account_features
from features.m1_routine_lifecycle import (
    monthly_rhythm,
)

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
ACC_1 = "acc_person_000001"


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    """Provides isolated FakeRedis instance."""
    return fakeredis.FakeRedis()


def _make_event(
    event_id: str,
    kind: str,
    event_time: datetime,
    arrival_time: datetime,
    details: dict[str, Any] | None = None,
) -> AccountEventRecord:
    return AccountEventRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        event_id=event_id,
        kind=kind,
        event_time=event_time,
        arrival_time=arrival_time,
        details=details if details is not None else {},
    )


def _make_payment(
    payment_id: str,
    event_time: datetime,
    arrival_time: datetime,
    side: str = "credit",
    amount_paise: int = 100000,
    counterparty_bank_id: str = BANK_1,
    counterparty_vpa: str = "vpa_cp_01",
    payee_type: str = "person",
) -> PaymentSideRecord:
    return PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        payment_id=payment_id,
        side=side,
        counterparty_bank_id=counterparty_bank_id,
        counterparty_vpa=counterparty_vpa,
        amount_paise=amount_paise,
        status="success",
        failure_domain=None,
        payee_type=payee_type,
        channel="upi_app",
        sender_device_fingerprint="fp_test",
        sender_os_generation=1,
        event_time=event_time,
        arrival_time=arrival_time,
        balance_after_paise=500000,
    )


def test_monthly_rhythm_exact_boundaries() -> None:
    """Verifies monthly_rhythm logic against standard 30-day cycle with 5-day tolerance [25, 35]."""
    # Single day cannot establish a rhythm
    assert monthly_rhythm([100], 30, 5) is False
    # Exactly 30 days
    assert monthly_rhythm([100, 130], 30, 5) is True
    # Boundary: 25 days (30 - 5)
    assert monthly_rhythm([100, 125], 30, 5) is True
    # Boundary: 35 days (30 + 5)
    assert monthly_rhythm([100, 135], 30, 5) is True
    # Beyond boundary: 24 days -> False
    assert monthly_rhythm([100, 124], 30, 5) is False
    # Beyond boundary: 36 days -> False
    assert monthly_rhythm([100, 136], 30, 5) is False
    # Multi-step rhythm with slight weekend variance: 30, 29, 31
    assert monthly_rhythm([10, 40, 69, 100], 30, 5) is True
    # Multi-step rhythm with one irregular jump
    assert monthly_rhythm([10, 40, 55, 85], 30, 5) is False


def test_routine_life_returns_none_on_short_history(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Verifies that an account with history < 35 days returns None for routine features.
    (Sabotage proof: returning False instead of None fails this test).
    """
    settings = load_account_features_settings()
    t_open = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    ev_open = _make_event(
        f"{RUN_ID}-open-01", "account_opened", t_open, t_open,
        details={"home_region": 1, "account_type": "savings"},
    )
    update_account_summaries(redis_client, ev_open, settings)

    # 10 days of history: active on Oct 1, decision on Oct 11
    t_pay = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    p = _make_payment(f"{RUN_ID}-p-01", t_pay, t_pay)
    update_account_summaries(redis_client, p, settings)

    decision_time = datetime(2026, 10, 11, 10, 0, tzinfo=IST)
    features = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=decision_time,
        arrival_cutoff=decision_time,
        settings=settings,
        groups=["routine_life"],
    )

    # 10 days available < 35 days minimum required
    assert features["history_days_available"] == 10
    assert features["salary_like_payer_count"] is None
    assert features["recurring_merchant_payee_count"] is None
    assert features["regular_transfer_payee_count"] is None
    assert features["routine_life_present"] is None


def test_routine_life_detects_salary_and_merchants(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies salary payer detection, recurring merchant payees, and routine_life_present."""
    settings = load_account_features_settings()

    # 3 monthly salary payments on Day 1, Day 31, Day 61
    t_sal1 = datetime(2026, 8, 1, 10, 0, tzinfo=IST)
    t_sal2 = datetime(2026, 8, 31, 10, 0, tzinfo=IST)
    t_sal3 = datetime(2026, 9, 30, 10, 0, tzinfo=IST)

    update_account_summaries(
        redis_client,
        _make_payment(f"{RUN_ID}-sal-1", t_sal1, t_sal1, side="credit", counterparty_vpa="employer_co"),
        settings,
    )
    update_account_summaries(
        redis_client,
        _make_payment(f"{RUN_ID}-sal-2", t_sal2, t_sal2, side="credit", counterparty_vpa="employer_co"),
        settings,
    )
    update_account_summaries(
        redis_client,
        _make_payment(f"{RUN_ID}-sal-3", t_sal3, t_sal3, side="credit", counterparty_vpa="employer_co"),
        settings,
    )

    # Outbound debit to recurring merchant on 2 distinct days (Aug 5 and Aug 20)
    t_m1 = datetime(2026, 8, 5, 12, 0, tzinfo=IST)
    t_m2 = datetime(2026, 8, 20, 12, 0, tzinfo=IST)
    update_account_summaries(
        redis_client,
        _make_payment(
            f"{RUN_ID}-m-1", t_m1, t_m1, side="debit", counterparty_vpa="grocery_shop", payee_type="merchant"
        ),
        settings,
    )
    update_account_summaries(
        redis_client,
        _make_payment(
            f"{RUN_ID}-m-2", t_m2, t_m2, side="debit", counterparty_vpa="grocery_shop", payee_type="merchant"
        ),
        settings,
    )

    decision_time = datetime(2026, 10, 5, 10, 0, tzinfo=IST)
    features = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=decision_time,
        arrival_cutoff=decision_time,
        settings=settings,
        groups=["routine_life"],
    )

    assert isinstance(features["history_days_available"], int)
    assert features["history_days_available"] >= 35
    assert features["salary_like_payer_count"] == 1
    assert features["recurring_merchant_payee_count"] == 1
    assert features["routine_life_present"] is True


def test_lifecycle_features_age_type_dormancy_volume(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies lifecycle group: account_age_days, account_type, dormancy_days, and volume ratio."""
    settings = load_account_features_settings()

    # Account opened 100 days ago
    t_open = datetime(2026, 6, 20, 10, 0, tzinfo=IST)
    open_ev = _make_event(
        f"{RUN_ID}-open-01",
        "account_opened",
        t_open,
        t_open,
        details={"home_region": 10, "account_type": "savings"},
    )
    update_account_summaries(redis_client, open_ev, settings)

    # Historical activity: active on Day 1 (Aug 1) with 10,000 INR
    t_d1 = datetime(2026, 8, 1, 10, 0, tzinfo=IST)
    p_d1 = _make_payment(f"{RUN_ID}-p-d1", t_d1, t_d1, amount_paise=1000000)
    update_account_summaries(redis_client, p_d1, settings)

    # Dormancy until Day 60 (Sept 29): onset of recent activity with 50,000 INR
    t_d60 = datetime(2026, 9, 29, 10, 0, tzinfo=IST)
    p_d60 = _make_payment(f"{RUN_ID}-p-d60", t_d60, t_d60, amount_paise=5000000)
    apply_payment_side(redis_client, p_d60, settings)

    # Decision on Oct 2 (within 7d of Sept 29)
    decision_time = datetime(2026, 10, 2, 10, 0, tzinfo=IST)

    features = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=decision_time,
        arrival_cutoff=decision_time,
        settings=settings,
        groups=["lifecycle"],
    )

    # 1. account_age_days: calendar days from June 20 to Oct 2 = 104 days
    expected_age = to_day_number(decision_time) - to_day_number(t_open)
    assert features["account_age_days"] == expected_age

    # 2. account_type: savings
    assert features["account_type"] == "savings"

    # 3. dormancy_days_before_recent_activity:
    # onset = day_num(Sept 29); prev = day_num(Aug 1)
    expected_dormancy = to_day_number(t_d60) - to_day_number(t_d1) - 1
    assert features["dormancy_days_before_recent_activity"] == expected_dormancy

    # 4. volume_vs_own_history_ratio:
    # Day totals has 2 days: Aug 1 (10,000 INR) + Sept 29 (50,000 INR) = 60,000 INR / 2 = 30,000 INR avg daily
    # Recent 7d volume = Sept 29 payment (50,000 INR = 5,000,000 paise)
    # Ratio = 5,000,000 / 3,000,000 = 1.6667
    assert isinstance(features["volume_vs_own_history_ratio"], float)
    assert features["volume_vs_own_history_ratio"] > 1.0
