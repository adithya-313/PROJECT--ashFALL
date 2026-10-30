# File: tests/unit/test_m1_inflow_outflow.py
# Purpose: Unit tests for Level 15b M1 inflow and outflow features.
# Issue: Level 15b M1 features (plan Task 1, G1, G2, R8).
"""
WHAT: Tests pure calculation functions for M1 inflow and outflow features, including normal cases
with hand-calculated fixture literals, empty history fallbacks, strict lower-boundary window exclusion,
isolation across commercial banks, and rejection of failed transaction attempts.

WHY: Verifies Section 22 window semantics, zero future-leakage, and strict tenant isolation
before models consume velocity and cash-out signals.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import pytest

from features.account_keys import AccountFeaturesSettings
from features.account_records import PaymentSideRecord, WithdrawalRecord
from features.account_updater import apply_payment_side
from features.m1_features import compute_account_features
from features.m1_inflow_outflow import (
    compute_below_threshold_credit_count_7d,
    compute_cash_out_channels_7d,
    compute_distinct_payers,
    compute_first_time_payer_share_24h,
    compute_median_credit_to_debit_seconds_7d,
    compute_other_bank_outflow_share_7d,
    compute_pass_through_ratio,
    compute_small_credit_count,
    compute_top_receiver_share_7d,
)

IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=IST)
RUN_ID = "0123456789abcdef"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
ACC_1 = "acc_001"


def _make_payment_side(
    payment_id: str,
    side: str,
    amount_paise: int,
    event_time: datetime,
    counterparty_bank_id: str = "bank_remote",
    counterparty_vpa: str | None = "acc_counterparty",
    status: str = "success",
    payee_type: str = "person",
    bank_id: str = BANK_0,
    account_id: str = ACC_1,
) -> PaymentSideRecord:
    return PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=f"{RUN_ID}-{payment_id}",
        side=side,
        counterparty_bank_id=counterparty_bank_id,
        counterparty_vpa=counterparty_vpa,
        amount_paise=amount_paise,
        status=status,
        failure_domain="customer" if status == "failed" else None,
        payee_type=payee_type,
        channel="upi_app",
        sender_device_fingerprint="fp_001" if side == "debit" else None,
        sender_os_generation=14 if side == "debit" else None,
        event_time=event_time,
        arrival_time=event_time + timedelta(seconds=1),
    )


def _make_withdrawal(
    withdrawal_id: str,
    amount_paise: int,
    event_time: datetime,
    channel: str = "atm",
    status: str = "success",
    bank_id: str = BANK_0,
    account_id: str = ACC_1,
) -> WithdrawalRecord:
    return WithdrawalRecord(
        run_id=RUN_ID,
        bank_id=bank_id,
        account_id=account_id,
        withdrawal_id=f"{RUN_ID}-{withdrawal_id}",
        channel=channel,
        amount_paise=amount_paise,
        status=status,
        failure_domain="customer" if status == "failed" else None,
        balance_after_paise=50000,
        event_time=event_time,
        arrival_time=event_time + timedelta(seconds=1),
    )


# -----------------------------------------------------------------------------
# Inflow Feature Unit Tests
# -----------------------------------------------------------------------------


def test_distinct_payers_normal_and_boundary_and_failed() -> None:
    """
    Verifies distinct_payers counts unique counterparties on successful credits,
    excludes failed credits, and strictly excludes events on the lower boundary.
    """
    decision_time = T0

    # 3 distinct payers within 24h window: payer_a (two payments), payer_b, payer_c
    p1 = _make_payment_side("p1", "credit", 10000, decision_time - timedelta(hours=2), counterparty_vpa="payer_a")
    p2 = _make_payment_side("p2", "credit", 15000, decision_time - timedelta(hours=3), counterparty_vpa="payer_a")
    p3 = _make_payment_side("p3", "credit", 20000, decision_time - timedelta(hours=5), counterparty_vpa="payer_b")
    p4 = _make_payment_side("p4", "credit", 25000, decision_time - timedelta(hours=10), counterparty_vpa="payer_c")

    # Failed credit from payer_d: must NOT be counted
    p_fail = _make_payment_side(
        "p_fail", "credit", 30000, decision_time - timedelta(hours=4), counterparty_vpa="payer_d", status="failed"
    )

    # Debit from payer_e: must NOT be counted
    p_debit = _make_payment_side(
        "p_debit", "debit", 40000, decision_time - timedelta(hours=6), counterparty_vpa="payer_e"
    )

    # Boundary record at exact lower bound (decision_time - 24h): must be EXCLUDED (> not >=)
    p_boundary = _make_payment_side(
        "p_bound", "credit", 50000, decision_time - timedelta(seconds=86400), counterparty_vpa="payer_bound"
    )

    # Older record (26 hours ago): must be EXCLUDED
    p_older = _make_payment_side(
        "p_old", "credit", 60000, decision_time - timedelta(hours=26), counterparty_vpa="payer_old"
    )

    records = [p1, p2, p3, p4, p_fail, p_debit, p_boundary, p_older]

    # Arithmetic: distinct counterparties in (T0 - 24h, T0] are payer_a, payer_b, payer_c -> 3
    result_24h = compute_distinct_payers(records, decision_time, 86400)
    assert result_24h == 3

    # Arithmetic: within 1h (T0 - 1h, T0], no records exist -> 0
    result_1h = compute_distinct_payers(records, decision_time, 3600)
    assert result_1h == 0

    # Arithmetic: within 7d, includes p1, p2, p3, p4, p_boundary (is inside 7d), p_older -> 5 distinct
    # payer_a, payer_b, payer_c, payer_bound, payer_old -> 5
    result_7d = compute_distinct_payers(records, decision_time, 604800)
    assert result_7d == 5


def test_distinct_payers_empty() -> None:
    """Empty record list yields 0 for distinct_payers."""
    assert compute_distinct_payers([], T0, 3600) == 0
    assert compute_distinct_payers([], T0, 86400) == 0
    assert compute_distinct_payers([], T0, 604800) == 0


def test_first_time_payer_share_24h_normal_architect_case() -> None:
    """
    Specific prompt case:
    two payers in 24h, one of whom paid on a closed day 10 days earlier → 0.5.
    """
    decision_time = T0

    # Payers in 24h: payer_new and payer_recurrent
    p1 = _make_payment_side("p1", "credit", 10000, decision_time - timedelta(hours=2), counterparty_vpa="payer_new")
    p2 = _make_payment_side(
        "p2", "credit", 20000, decision_time - timedelta(hours=5), counterparty_vpa="payer_recurrent"
    )

    # payer_recurrent paid on a closed day 10 days earlier
    ten_days_ago_day = (decision_time - timedelta(days=10)).date().toordinal()
    closed_cp_days = {
        "bank_remote:payer_recurrent": [ten_days_ago_day]
    }

    # Arithmetic: total 24h payers = 2 (payer_new, payer_recurrent).
    # payer_recurrent is seen in closed_cp_days. payer_new is not.
    # first_time_payers = 1. Share = 1 / 2 = 0.5.
    share = compute_first_time_payer_share_24h([p1, p2], closed_cp_days, decision_time)
    assert share == 0.5


def test_first_time_payer_share_24h_seen_in_earlier_raw_record() -> None:
    """A payer seen in raw records earlier than 24h (e.g. 3 days ago) is not a first-time payer."""
    decision_time = T0

    p_recent = _make_payment_side(
        "p1", "credit", 10000, decision_time - timedelta(hours=4), counterparty_vpa="payer_old_raw"
    )
    p_earlier_raw = _make_payment_side(
        "p2", "credit", 10000, decision_time - timedelta(days=3), counterparty_vpa="payer_old_raw"
    )

    # Arithmetic: 1 payer in 24h, seen 3 days ago in raw records. First time count = 0. Share = 0 / 1 = 0.0.
    share = compute_first_time_payer_share_24h([p_recent, p_earlier_raw], {}, decision_time)
    assert share == 0.0


def test_first_time_payer_share_24h_empty() -> None:
    """No credits in 24h returns None ('no data')."""
    assert compute_first_time_payer_share_24h([], {}, T0) is None


def test_small_credit_count_normal_and_empty() -> None:
    """Verifies small_credit_count counts successful credits <= small_credit_max_paise."""
    decision_time = T0
    max_paise = 50000  # ₹500

    # Credits: ₹100 (10000 paise), ₹500 (50000 paise), ₹501 (50100 paise), ₹1000 (100000 paise)
    p1 = _make_payment_side("p1", "credit", 10000, decision_time - timedelta(hours=1))  # <= 50000: YES
    p2 = _make_payment_side("p2", "credit", 50000, decision_time - timedelta(hours=2))  # <= 50000: YES
    p3 = _make_payment_side("p3", "credit", 50100, decision_time - timedelta(hours=3))  # > 50000: NO
    p4 = _make_payment_side("p4", "credit", 100000, decision_time - timedelta(hours=4))  # > 50000: NO
    p_fail = _make_payment_side("p_fail", "credit", 10000, decision_time - timedelta(hours=1), status="failed")

    # Arithmetic: p1 (10000 <= 50000) and p2 (50000 <= 50000) count -> 2
    count = compute_small_credit_count([p1, p2, p3, p4, p_fail], decision_time, 86400, max_paise)
    assert count == 2

    # Empty case returns 0
    assert compute_small_credit_count([], decision_time, 86400, max_paise) == 0


def test_below_threshold_credit_count_7d_architect_case() -> None:
    """
    Specific prompt case:
    credits of ₹9,600 and ₹9,400 with threshold ₹10,000 and band 5% → only ₹9,600 counts
    (band is ₹9,500–₹9,999.99).
    """
    decision_time = T0
    thresholds = (1000000, 5000000, 10000000)  # ₹10k, ₹50k, ₹100k
    fraction = 0.05

    # ₹9,600 = 960000 paise (inside [950000, 1000000))
    p_9600 = _make_payment_side("p1", "credit", 960000, decision_time - timedelta(days=1))
    # ₹9,400 = 940000 paise (below 950000: OUTSIDE)
    p_9400 = _make_payment_side("p2", "credit", 940000, decision_time - timedelta(days=2))
    # ₹10,000 = 1000000 paise (exact threshold: OUTSIDE, must be strictly below)
    p_10000 = _make_payment_side("p3", "credit", 1000000, decision_time - timedelta(days=3))
    # ₹48,000 = 4800000 paise (inside [4750000, 5000000))
    p_48000 = _make_payment_side("p4", "credit", 4800000, decision_time - timedelta(days=4))

    # Arithmetic: p_9600 (960k in [950k, 1M)) and p_48000 (4.8M in [4.75M, 5M)) count -> 2
    count = compute_below_threshold_credit_count_7d(
        [p_9600, p_9400, p_10000, p_48000], decision_time, thresholds, fraction
    )
    assert count == 2

    # Verify only p_9600 and p_9400 alone: only p_9600 counts -> 1
    assert compute_below_threshold_credit_count_7d([p_9600, p_9400], decision_time, thresholds, fraction) == 1

    # Empty case returns 0
    assert compute_below_threshold_credit_count_7d([], decision_time, thresholds, fraction) == 0


# -----------------------------------------------------------------------------
# Outflow Feature Unit Tests
# -----------------------------------------------------------------------------


def test_pass_through_ratio_normal_and_empty() -> None:
    """Verifies pass_through_ratio computes (debits + withdrawals) / credits."""
    decision_time = T0

    # Credits: ₹1,000 + ₹2,000 = ₹3,000 (300000 paise)
    c1 = _make_payment_side("c1", "credit", 100000, decision_time - timedelta(hours=3))
    c2 = _make_payment_side("c2", "credit", 200000, decision_time - timedelta(hours=2))

    # Debits: ₹1,500 (150000 paise)
    d1 = _make_payment_side("d1", "debit", 150000, decision_time - timedelta(hours=1))

    # Withdrawals: ₹600 (60000 paise)
    w1 = _make_withdrawal("w1", 60000, decision_time - timedelta(minutes=30))

    # Arithmetic: total outflow = 150000 + 60000 = 210000. Credits = 300000.
    # Ratio = 210000 / 300000 = 0.70.
    ratio = compute_pass_through_ratio([c1, c2, d1], [w1], decision_time, 86400)
    assert ratio == 0.7

    # Empty credits with outflow returns None ('no data when no credits') - REQUIRED FOR SABOTAGE C
    ratio_no_credits = compute_pass_through_ratio([d1], [w1], decision_time, 86400)
    assert ratio_no_credits is None

    # Empty history returns None
    assert compute_pass_through_ratio([], [], decision_time, 86400) is None


def test_median_credit_to_debit_seconds_7d_even_pairs() -> None:
    """
    Specific prompt case:
    median_credit_to_debit_seconds_7d with an even number of pairs (mean of two middle values).
    """
    decision_time = T0

    # Credit 1 at T0 - 500s
    c1 = _make_payment_side("c1", "credit", 50000, decision_time - timedelta(seconds=500))
    # Debit 1 at T0 - 400s -> delay since c1 = 500 - 400 = 100s
    d1 = _make_payment_side("d1", "debit", 25000, decision_time - timedelta(seconds=400))

    # Credit 2 at T0 - 300s
    c2 = _make_payment_side("c2", "credit", 50000, decision_time - timedelta(seconds=300))
    # Withdrawal 1 at T0 - 100s -> delay since c2 = 300 - 100 = 200s
    w1 = _make_withdrawal("w1", 20000, decision_time - timedelta(seconds=100))

    # Even pairs count = 2. Delays = [100.0, 200.0].
    # Arithmetic: median is mean of 100.0 and 200.0 = (100.0 + 200.0) / 2 = 150.0s.
    median_val = compute_median_credit_to_debit_seconds_7d([c1, d1, c2], [w1], decision_time)
    assert median_val == 150.0


def test_median_credit_to_debit_seconds_7d_empty() -> None:
    """When no credits or no debits exist, returns None ('no data when no pairs')."""
    decision_time = T0
    assert compute_median_credit_to_debit_seconds_7d([], [], decision_time) is None

    # Only debits (no earlier credits) -> None
    d1 = _make_payment_side("d1", "debit", 25000, decision_time - timedelta(seconds=400))
    assert compute_median_credit_to_debit_seconds_7d([d1], [], decision_time) is None


def test_top_receiver_share_7d_normal_and_empty() -> None:
    """Verifies largest single receiver's share of successful debits."""
    decision_time = T0

    # Debits:
    # receiver_a: ₹300 (30000 paise) + ₹200 (20000 paise) = 50000 paise
    # receiver_b: ₹250 (25000 paise)
    # receiver_c: ₹250 (25000 paise)
    # Total debits = 50000 + 25000 + 25000 = 100000 paise
    d1 = _make_payment_side("d1", "debit", 30000, decision_time - timedelta(days=1), counterparty_vpa="receiver_a")
    d2 = _make_payment_side("d2", "debit", 20000, decision_time - timedelta(days=2), counterparty_vpa="receiver_a")
    d3 = _make_payment_side("d3", "debit", 25000, decision_time - timedelta(days=3), counterparty_vpa="receiver_b")
    d4 = _make_payment_side("d4", "debit", 25000, decision_time - timedelta(days=4), counterparty_vpa="receiver_c")

    # Arithmetic: max receiver is receiver_a with 50000 / 100000 = 0.50
    share = compute_top_receiver_share_7d([d1, d2, d3, d4], decision_time)
    assert share == 0.5

    # Empty debits returns None ('no data when no debits')
    assert compute_top_receiver_share_7d([], decision_time) is None


def test_other_bank_outflow_share_7d_normal_and_empty() -> None:
    """Verifies share of successful debits sent to a different bank."""
    decision_time = T0
    this_bank = BANK_0

    # Same-bank debit: ₹400 (40000 paise)
    d_same = _make_payment_side(
        "d_same", "debit", 40000, decision_time - timedelta(days=1), counterparty_bank_id=this_bank
    )
    # Other-bank debit: ₹600 (60000 paise)
    d_other = _make_payment_side(
        "d_other", "debit", 60000, decision_time - timedelta(days=2), counterparty_bank_id="bank_other"
    )

    # Arithmetic: other bank amount = 60000 / (40000 + 60000) = 60000 / 100000 = 0.60
    share = compute_other_bank_outflow_share_7d([d_same, d_other], this_bank, decision_time)
    assert share == 0.6

    # Empty debits returns None
    assert compute_other_bank_outflow_share_7d([], this_bank, decision_time) is None


def test_cash_out_channels_7d_and_failed_exclusion() -> None:
    """
    Specific prompt case:
    ATM and branch withdrawals land in their own features; a failed withdrawal counts in none.
    """
    decision_time = T0

    # Person debit: ₹1,000 (100000 paise)
    d_person = _make_payment_side(
        "d_p", "debit", 100000, decision_time - timedelta(days=1), payee_type="person"
    )
    # Merchant debit: ₹2,000 (200000 paise)
    d_merchant = _make_payment_side(
        "d_m", "debit", 200000, decision_time - timedelta(days=2), payee_type="merchant"
    )
    # ATM withdrawal: ₹500 (50000 paise)
    w_atm = _make_withdrawal("w_atm", 50000, decision_time - timedelta(days=3), channel="atm")
    # Branch withdrawal: ₹1,500 (150000 paise)
    w_branch = _make_withdrawal("w_br", 150000, decision_time - timedelta(days=4), channel="branch")
    # Failed withdrawal: ₹10,000 - MUST NOT BE COUNTED ANYWHERE
    w_failed = _make_withdrawal("w_fail", 1000000, decision_time - timedelta(days=1), channel="atm", status="failed")

    channels = compute_cash_out_channels_7d([d_person, d_merchant], [w_atm, w_branch, w_failed], decision_time)

    assert channels["outflow_to_person_paise_7d"] == 100000
    assert channels["outflow_to_merchant_paise_7d"] == 200000
    assert channels["cash_atm_paise_7d"] == 50000
    assert channels["cash_branch_paise_7d"] == 150000

    # Empty returns 0 for all channels
    empty_channels = compute_cash_out_channels_7d([], [], decision_time)
    assert empty_channels["outflow_to_person_paise_7d"] == 0
    assert empty_channels["outflow_to_merchant_paise_7d"] == 0
    assert empty_channels["cash_atm_paise_7d"] == 0
    assert empty_channels["cash_branch_paise_7d"] == 0


# -----------------------------------------------------------------------------
# Bank Isolation & Orchestration Integration Tests
# -----------------------------------------------------------------------------


def test_bank_isolation_on_compute_account_features(monkeypatch: Any) -> None:
    """
    Specific prompt case:
    the same account_id with records under bank_0 and bank_1 →
    bank_0's features use only bank_0's records (R8 original test:
    'a bank's features never use another bank's ledger').
    """
    client = fakeredis.FakeRedis()
    settings = AccountFeaturesSettings(
        windows_seconds=(3600, 86400, 604800),
        retention_seconds=604800,
        allowed_lateness_seconds=86400,
        key_expiry_margin_seconds=86400,
        key_prefix="acctf_iso",
        run_id_hex_length=16,
        account_event_batch_size=500,
        account_event_overlap_rows=200,
    )

    decision_time = T0
    arrival_cutoff = T0 + timedelta(seconds=10)

    # Write a ₹1,000 credit for ACC_1 under BANK_0 (30 minutes ago, inside 1h window)
    rec_bank0 = _make_payment_side("p_b0", "credit", 100000, decision_time - timedelta(minutes=30), bank_id=BANK_0)
    apply_payment_side(client, rec_bank0, settings)

    # Write a ₹5,000 credit for the SAME ACC_1 under BANK_1 (30 minutes ago, inside 1h window)
    rec_bank1 = _make_payment_side("p_b1", "credit", 500000, decision_time - timedelta(minutes=30), bank_id=BANK_1)
    apply_payment_side(client, rec_bank1, settings)

    # Read features for BANK_0: must observe only BANK_0's credit
    features_b0 = compute_account_features(
        client=client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=settings,
        groups=("inflow",),
    )
    assert features_b0["distinct_payers_1h"] == 1
    assert features_b0["small_credit_count_24h"] == 0

    # Read features for BANK_1: must observe only BANK_1's credit
    features_b1 = compute_account_features(
        client=client,
        run_id=RUN_ID,
        bank_id=BANK_1,
        account_id=ACC_1,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=settings,
        groups=("inflow",),
    )
    assert features_b1["distinct_payers_1h"] == 1


def test_compute_account_features_unknown_group_raises() -> None:
    """Verifies that requesting an unrecognized group raises ValueError."""
    client = fakeredis.FakeRedis()
    with pytest.raises(ValueError, match="Unknown feature group"):
        compute_account_features(
            client=client,
            run_id=RUN_ID,
            bank_id=BANK_0,
            account_id=ACC_1,
            decision_time=T0,
            arrival_cutoff=T0,
            groups=("unsupported_group",),
        )
