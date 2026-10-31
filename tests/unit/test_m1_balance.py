# File: tests/unit/test_m1_balance.py
# Purpose: Unit tests for Level 15b M1 balance-pattern features.
# Issue: Level 15b M1 features (plan Task 1, G2).
"""
WHAT: Tests pure calculation functions for M1 balance-pattern features:
current_balance_paise, min_balance_paise_7d, and near_zero_cycles_7d,
as well as full feature orchestration across inflow, outflow, and balance_pattern groups.

WHY: Verifies correct cash-out detection, strict point-in-time state reading,
and guarantees that compute_account_features returns exactly the canonical names
without drift.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import pytest

from features.account_keys import AccountFeaturesSettings
from features.account_records import PaymentSideRecord, WithdrawalRecord
from features.account_updater import apply_payment_side
from features.feature_set import load_feature_set_m1
from features.m1_balance import (
    compute_current_balance_paise,
    compute_min_balance_paise_7d,
    compute_near_zero_cycles_7d,
)
from features.m1_features import compute_account_features

IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=IST)
RUN_ID = "0123456789abcdef"
BANK_0 = "bank_0"
ACC_1 = "acc_001"


def _make_payment_side(
    payment_id: str,
    side: str,
    amount_paise: int,
    event_time: datetime,
    balance_after_paise: int | None = None,
    status: str = "success",
    counterparty_bank_id: str = "bank_remote",
    counterparty_vpa: str | None = "acc_counterparty",
) -> PaymentSideRecord:
    return PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        payment_id=f"{RUN_ID}-{payment_id}",
        side=side,
        counterparty_bank_id=counterparty_bank_id,
        counterparty_vpa=counterparty_vpa,
        amount_paise=amount_paise,
        status=status,
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        balance_after_paise=balance_after_paise,
        event_time=event_time,
        arrival_time=event_time + timedelta(seconds=1),
    )


def _make_withdrawal(
    withdrawal_id: str,
    amount_paise: int,
    event_time: datetime,
    balance_after_paise: int | None = None,
    channel: str = "atm",
    status: str = "success",
) -> WithdrawalRecord:
    return WithdrawalRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        withdrawal_id=f"{RUN_ID}-{withdrawal_id}",
        channel=channel,
        amount_paise=amount_paise,
        status=status,
        failure_domain=None,
        balance_after_paise=balance_after_paise,
        event_time=event_time,
        arrival_time=event_time + timedelta(seconds=1),
    )


# -----------------------------------------------------------------------------
# Balance-Pattern Feature Tests
# -----------------------------------------------------------------------------


def test_current_balance_paise_normal_and_empty() -> None:
    """Verifies current_balance_paise returns the balance from the latest record."""
    decision_time = T0

    # Payment at T0 - 2 hours with balance ₹10,000 (1000000 paise)
    p1 = _make_payment_side("p1", "credit", 500000, decision_time - timedelta(hours=2), balance_after_paise=1000000)

    # Withdrawal at T0 - 30 minutes with balance ₹8,000 (800000 paise) -> LATEST
    w1 = _make_withdrawal("w1", 200000, decision_time - timedelta(minutes=30), balance_after_paise=800000)

    # Payment at T0 - 10 minutes with NO balance field
    p2 = _make_payment_side("p2", "debit", 100000, decision_time - timedelta(minutes=10), balance_after_paise=None)

    # Arithmetic: latest record carrying a balance is w1 with 800,000 paise.
    balance = compute_current_balance_paise([p1, p2], [w1], decision_time)
    assert balance == 800000

    # Future record beyond decision_time must be ignored
    p_future = _make_payment_side(
        "p_fut", "debit", 100000, decision_time + timedelta(minutes=5), balance_after_paise=200000
    )
    balance_with_future = compute_current_balance_paise([p1, p2, p_future], [w1], decision_time)
    assert balance_with_future == 800000

    # Empty case returns None ('no data when no record carries a balance')
    assert compute_current_balance_paise([], [], decision_time) is None
    assert compute_current_balance_paise([p2], [], decision_time) is None


def test_min_balance_paise_7d_normal_and_empty() -> None:
    """Verifies min_balance_paise_7d returns lowest balance in 7 days."""
    decision_time = T0

    # Balances: ₹5,000 (500000 paise), ₹1,200 (120000 paise), ₹2,500 (250000 paise)
    p1 = _make_payment_side("p1", "credit", 500000, decision_time - timedelta(days=2), balance_after_paise=500000)
    w1 = _make_withdrawal("w1", 380000, decision_time - timedelta(days=3), balance_after_paise=120000)
    p2 = _make_payment_side("p2", "credit", 130000, decision_time - timedelta(days=1), balance_after_paise=250000)

    # Record older than 7 days (8 days ago) with balance ₹50 (5000 paise): must be EXCLUDED
    p_old = _make_payment_side("p_old", "debit", 50000, decision_time - timedelta(days=8), balance_after_paise=5000)

    # Arithmetic: within (T0 - 7d, T0], balances are 500k, 120k, 250k. Lowest is 120k.
    min_bal = compute_min_balance_paise_7d([p1, p2, p_old], [w1], decision_time)
    assert min_bal == 120000

    # Empty case returns None
    assert compute_min_balance_paise_7d([], [], decision_time) is None
    assert compute_min_balance_paise_7d([p_old], [], decision_time) is None


def test_near_zero_cycles_7d_architect_case() -> None:
    """
    Specific prompt case:
    credit → debit leaving ₹50 → credit → debit leaving ₹5,000 → 1 cycle.
    near_zero_balance_max_paise = 10000 (₹100).
    """
    decision_time = T0
    max_near_zero = 10000  # ₹100

    # 1. Credit 1 at T0 - 4 days
    c1 = _make_payment_side("c1", "credit", 100000, decision_time - timedelta(days=4), balance_after_paise=100000)

    # 2. Debit 1 at T0 - 3 days leaving ₹50 (5000 paise <= 10000 paise) -> CYCLE 1
    d1 = _make_payment_side("d1", "debit", 95000, decision_time - timedelta(days=3), balance_after_paise=5000)

    # 3. Credit 2 at T0 - 2 days -> RESETS CYCLE
    c2 = _make_payment_side("c2", "credit", 500000, decision_time - timedelta(days=2), balance_after_paise=505000)

    # 4. Debit 2 at T0 - 1 day leaving ₹5,000 (500000 paise > 10000 paise) -> NOT NEAR ZERO
    d2 = _make_payment_side("d2", "debit", 5000, decision_time - timedelta(days=1), balance_after_paise=500000)

    # Arithmetic: 1 cycle (c1 followed by d1 with balance 5000 <= 10000; c2 followed by d2 with balance 500000 > 10000)
    cycles = compute_near_zero_cycles_7d([c1, d1, c2, d2], [], decision_time, max_near_zero)
    assert cycles == 1


def test_near_zero_cycles_7d_multiple_debits_in_same_cycle_counted_once() -> None:
    """Multiple near-zero balances following a single credit count as only 1 cycle."""
    decision_time = T0
    max_near_zero = 10000  # ₹100

    c1 = _make_payment_side("c1", "credit", 100000, decision_time - timedelta(days=3), balance_after_paise=100000)
    d1 = _make_payment_side("d1", "debit", 95000, decision_time - timedelta(days=2), balance_after_paise=5000)
    w1 = _make_withdrawal("w1", 4000, decision_time - timedelta(days=1), balance_after_paise=1000)

    # Arithmetic: c1 followed by d1 (5000 <= 10000) and w1 (1000 <= 10000) -> 1 cycle
    cycles = compute_near_zero_cycles_7d([c1, d1], [w1], decision_time, max_near_zero)
    assert cycles == 1


def test_near_zero_cycles_7d_no_balances_returns_none() -> None:
    """When no balances exist in 7d, returns None ('no data when no balances')."""
    decision_time = T0
    p1 = _make_payment_side("p1", "credit", 100000, decision_time - timedelta(days=1), balance_after_paise=None)
    assert compute_near_zero_cycles_7d([p1], [], decision_time, 10000) is None
    assert compute_near_zero_cycles_7d([], [], decision_time, 10000) is None


def test_near_zero_cycles_7d_balances_exist_but_no_cycle_returns_zero() -> None:
    """When balances exist but no near-zero cycle occurs, returns 0."""
    decision_time = T0
    c1 = _make_payment_side("c1", "credit", 500000, decision_time - timedelta(days=2), balance_after_paise=500000)
    d1 = _make_payment_side("d1", "debit", 100000, decision_time - timedelta(days=1), balance_after_paise=400000)

    # 400000 > 10000 paise -> 0 cycles
    cycles = compute_near_zero_cycles_7d([c1, d1], [], decision_time, 10000)
    assert cycles == 0


# -----------------------------------------------------------------------------
# Full Vector Orchestration & Schema Drift Tests
# -----------------------------------------------------------------------------


def test_compute_account_features_returns_exact_inflow_outflow_balance_names() -> None:
    """
    Verifies that compute_account_features for ('inflow', 'outflow', 'balance_pattern')
    returns exactly the canonical names from feature_set_m1.yaml without any drift.
    """
    client = fakeredis.FakeRedis()
    settings = AccountFeaturesSettings(
        windows_seconds=(3600, 86400, 604800),
        retention_seconds=604800,
        allowed_lateness_seconds=86400,
        key_expiry_margin_seconds=86400,
        key_prefix="acctf_full",
        run_id_hex_length=16,
        account_event_batch_size=500,
        account_event_overlap_rows=200,
    )

    decision_time = T0
    arrival_cutoff = T0 + timedelta(seconds=10)

    # Write a credit transaction
    rec = _make_payment_side("p1", "credit", 100000, decision_time - timedelta(hours=2), balance_after_paise=100000)
    apply_payment_side(client, rec, settings)

    features = compute_account_features(
        client=client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=settings,
        groups=("inflow", "outflow", "balance_pattern"),
    )

    feature_set = load_feature_set_m1()
    expected_names: set[str] = set()
    for g in ("inflow", "outflow", "balance_pattern"):
        expected_names.update(f.name for f in feature_set.features_by_group(g))

    assert set(features.keys()) == expected_names
    # Total count = 7 (inflow) + 9 (outflow) + 3 (balance_pattern) = 19
    assert len(features) == 19


def test_schema_drift_check_raises_value_error(monkeypatch: Any) -> None:
    """Verifies that if feature computation returns mismatched keys, ValueError is raised."""
    client = fakeredis.FakeRedis()
    settings = AccountFeaturesSettings(
        windows_seconds=(3600, 86400, 604800),
        retention_seconds=604800,
        allowed_lateness_seconds=86400,
        key_expiry_margin_seconds=86400,
        key_prefix="acctf_drift",
        run_id_hex_length=16,
        account_event_batch_size=500,
        account_event_overlap_rows=200,
    )

    # Intentionally tamper with expected keys in schema loader to simulate drift
    import features.m1_features as mf

    def mock_load_feature_set() -> Any:
        fs = load_feature_set_m1()
        # Return an altered feature set with a fabricated extra feature name
        from features.feature_set import M1Feature, M1FeatureSet
        extra_feat = M1Feature(
            name="hallucinated_feature_x",
            group="inflow",
            definition="A fake feature",
            bank_visible_source_records="fake",
            window_or_horizon="1h",
            empty_means="zero",
            issue_ids=("G1",),
        )
        return M1FeatureSet(
            features=fs.features + (extra_feat,),
            removed=fs.removed,
            gap_notes=fs.gap_notes,
        )

    monkeypatch.setattr(mf, "load_feature_set_m1", mock_load_feature_set)

    with pytest.raises(ValueError, match="M1 feature drift detected"):
        compute_account_features(
            client=client,
            run_id=RUN_ID,
            bank_id=BANK_0,
            account_id=ACC_1,
            decision_time=T0,
            arrival_cutoff=T0,
            settings=settings,
            groups=("inflow",),
        )
