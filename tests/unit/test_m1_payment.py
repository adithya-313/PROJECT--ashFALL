# File: tests/unit/test_m1_payment.py
# Purpose: Unit tests for Level 15b M1 payment-level features (first payment, amount ratios, same-bank risk, hour).
# Issue: Level 15b M1 payment-level features (plan Task 1, R8, G1, R7).
"""
WHAT: Unit tests covering proposed payment feature extraction:
  1. first_payment_to_payee: True on new payee, False after a closed day of paying that payee.
  2. amount_vs_payer_median_ratio and amount_vs_payer_max_ratio: exact ratios against historical debits.
  3. payee_same_bank_risk: None for other-bank payee, 1.0 for confirmed fraud, 0.5 for complained, 0.0 for normal.
  4. hour_ist: Exact IST hour conversion (e.g. 2026-10-30T20:15:00+00:00 -> 01:45 IST next day -> hour 1).
  5. customer_failed_attempts_24h: Counts 'customer' failures only; 'sender_bank' technical failures are excluded.
  6. ProposedPayment input validation: Rejects empty IDs, non-positive amounts, naive datetimes, invalid payee types.
"""

from datetime import datetime, timedelta, timezone

import fakeredis
import pytest

from features.account_keys import (
    complained_accounts_key,
    labelled_accounts_key,
    load_account_features_settings,
)
from features.account_records import PaymentSideRecord
from features.account_summaries import update_account_summaries
from features.account_updater import apply_payment_side
from features.m1_payment import (
    ProposedPayment,
    compute_payer_only_payment_features,
    compute_payment_features,
)

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
IST = timezone(timedelta(hours=5, minutes=30))


def _make_payment_side(
    payment_id: str,
    account_id: str,
    side: str,
    counterparty_bank_id: str,
    counterparty_vpa: str,
    event_time: datetime,
    amount_paise: int = 100000,
    status: str = "success",
    failure_domain: str | None = None,
    bank_id: str = BANK_0,
) -> PaymentSideRecord:
    return PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=payment_id,
        side=side,
        counterparty_bank_id=counterparty_bank_id,
        counterparty_vpa=counterparty_vpa,
        amount_paise=amount_paise,
        status=status,
        failure_domain=failure_domain,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=event_time,
        arrival_time=event_time,
        balance_after_paise=1000000,
    )


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


def test_first_payment_to_payee_true_then_false(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that first_payment_to_payee is True initially, then False after previous payments to payee."""
    settings = load_account_features_settings()
    t_prop = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)

    # 1. Proposed payment to brand new payee: bank_1:acc_merchant_new
    prop_1 = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_1,
        payee_vpa="acc_merchant_new",
        payee_type="merchant",
        amount_paise=50000,
        event_time=t_prop,
    )
    feats_1 = compute_payment_features(redis_client, prop_1, t_prop, settings)
    assert feats_1["first_payment_to_payee"] is True

    # 2. Record a closed day debit to bank_1:acc_merchant_new in history (Day 10)
    t_closed = datetime(2026, 10, 11, 10, 0, 0, tzinfo=IST)
    p_closed = _make_payment_side(
        payment_id=f"{RUN_ID}-p-closed-01",
        account_id="acc_payer_01",
        side="debit",
        counterparty_bank_id=BANK_1,
        counterparty_vpa="acc_merchant_new",
        event_time=t_closed,
        amount_paise=25000,
    )
    update_account_summaries(redis_client, p_closed, settings)

    # 3. Evaluate same proposed payment: now False
    feats_2 = compute_payment_features(redis_client, prop_1, t_prop, settings)
    assert feats_2["first_payment_to_payee"] is False


def test_amount_vs_payer_median_and_max_ratios(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates amount ratios against payer's 7-day raw window successful debits.
    Hand-worked arithmetic:
      - Historical successful debits = [10000, 20000, 30000] paise.
      - Median = 20000 paise; Max = 30000 paise.
      - Proposed amount = 60000 paise.
      - amount_vs_payer_median_ratio = 60000 / 20000 == 3.0.
      - amount_vs_payer_max_ratio = 60000 / 30000 == 2.0.
    """
    settings = load_account_features_settings()
    t_now = datetime(2026, 10, 28, 14, 0, 0, tzinfo=IST)
    t_ev1 = t_now - timedelta(hours=5)
    t_ev2 = t_now - timedelta(hours=3)
    t_ev3 = t_now - timedelta(hours=1)

    # Apply 3 successful debits in 7-day raw window
    apply_payment_side(
        redis_client,
        _make_payment_side(f"{RUN_ID}-p1", "acc_payer_02", "debit", BANK_0, "CP1", t_ev1, amount_paise=10000),
        settings,
    )
    apply_payment_side(
        redis_client,
        _make_payment_side(f"{RUN_ID}-p2", "acc_payer_02", "debit", BANK_0, "CP2", t_ev2, amount_paise=20000),
        settings,
    )
    apply_payment_side(
        redis_client,
        _make_payment_side(f"{RUN_ID}-p3", "acc_payer_02", "debit", BANK_0, "CP3", t_ev3, amount_paise=30000),
        settings,
    )

    prop = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_02",
        payee_bank_id=BANK_0,
        payee_vpa="CP4",
        payee_type="person",
        amount_paise=60000,
        event_time=t_now,
    )
    feats = compute_payment_features(redis_client, prop, t_now, settings)

    # Hand-worked arithmetic assertion:
    # 60000 / 20000 == 3.0
    assert feats["amount_vs_payer_median_ratio"] == 3.0
    # 60000 / 30000 == 2.0
    assert feats["amount_vs_payer_max_ratio"] == 2.0


def test_payee_same_bank_risk(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates payee_same_bank_risk:
      - None when payee is at another bank.
      - 1.0 when payee is a confirmed fraud labelled account.
      - 0.5 when payee is a complained account.
      - 0.0 when payee is unflagged.
    """
    settings = load_account_features_settings()
    t_now = datetime(2026, 10, 28, 15, 0, 0, tzinfo=IST)
    score = int(t_now.timestamp())

    # 1. Other-bank payee returns None
    prop_other = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_03",
        payee_bank_id=BANK_1,  # Other bank
        payee_vpa="acc_other_payee",
        payee_type="person",
        amount_paise=10000,
        event_time=t_now,
    )
    feats_other = compute_payment_features(redis_client, prop_other, t_now, settings)
    assert feats_other["payee_same_bank_risk"] is None

    # 2. Same-bank payee unflagged returns 0.0
    prop_normal = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_03",
        payee_bank_id=BANK_0,  # Same bank
        payee_vpa="acc_clean_payee",
        payee_type="person",
        amount_paise=10000,
        event_time=t_now,
    )
    feats_normal = compute_payment_features(redis_client, prop_normal, t_now, settings)
    assert feats_normal["payee_same_bank_risk"] == 0.0

    # 3. Same-bank payee with registered complaint returns 0.5
    redis_client.zadd(complained_accounts_key(RUN_ID, BANK_0), {"acc_complained_payee": score})
    prop_cmp = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_03",
        payee_bank_id=BANK_0,
        payee_vpa="acc_complained_payee",
        payee_type="person",
        amount_paise=10000,
        event_time=t_now,
    )
    feats_cmp = compute_payment_features(redis_client, prop_cmp, t_now, settings)
    assert feats_cmp["payee_same_bank_risk"] == 0.5

    # 4. Same-bank payee with confirmed fraud label returns 1.0
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {"acc_mule_payee": score})
    prop_mule = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_03",
        payee_bank_id=BANK_0,
        payee_vpa="acc_mule_payee",
        payee_type="person",
        amount_paise=10000,
        event_time=t_now,
    )
    feats_mule = compute_payment_features(redis_client, prop_mule, t_now, settings)
    assert feats_mule["payee_same_bank_risk"] == 1.0


def test_hour_ist_conversion(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that hour_ist evaluates exact IST hour from UTC or aware event_time."""
    settings = load_account_features_settings()
    # 2026-10-30T20:15:00+00:00 UTC -> 2026-10-31 01:45:00 IST -> hour is 1
    t_utc = datetime(2026, 10, 30, 20, 15, 0, tzinfo=timezone.utc)
    prop = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_04",
        payee_bank_id=BANK_0,
        payee_vpa="CP1",
        payee_type="person",
        amount_paise=10000,
        event_time=t_utc,
    )
    feats = compute_payment_features(redis_client, prop, t_utc, settings)
    assert feats["hour_ist"] == 1


def test_customer_failed_attempts_24h_customer_only(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates Issue R7 constraint:
    customer_failed_attempts_24h counts 'customer' failures only.
    Technical failures (e.g. 'sender_bank', 'npci_central') are NOT counted against the customer.
    (Sabotage proof: counting 'sender_bank' failures fails this test).
    """
    settings = load_account_features_settings()
    t_now = datetime(2026, 10, 28, 16, 0, 0, tzinfo=IST)
    t_fail1 = t_now - timedelta(hours=2)
    t_fail2 = t_now - timedelta(hours=4)
    t_fail3 = t_now - timedelta(hours=6)

    # 1. Failed payment with failure_domain == 'customer' -> COUNTS
    apply_payment_side(
        redis_client,
        _make_payment_side(
            payment_id=f"{RUN_ID}-f1",
            account_id="acc_payer_05",
            side="debit",
            counterparty_bank_id=BANK_0,
            counterparty_vpa="CP1",
            event_time=t_fail1,
            status="failed",
            failure_domain="customer",
        ),
        settings,
    )

    # 2. Failed payment with failure_domain == 'sender_bank' (technical) -> DOES NOT COUNT
    apply_payment_side(
        redis_client,
        _make_payment_side(
            payment_id=f"{RUN_ID}-f2",
            account_id="acc_payer_05",
            side="debit",
            counterparty_bank_id=BANK_0,
            counterparty_vpa="CP1",
            event_time=t_fail2,
            status="failed",
            failure_domain="sender_bank",
        ),
        settings,
    )

    # 3. Another failed payment with failure_domain == 'customer' -> COUNTS
    apply_payment_side(
        redis_client,
        _make_payment_side(
            payment_id=f"{RUN_ID}-f3",
            account_id="acc_payer_05",
            side="debit",
            counterparty_bank_id=BANK_0,
            counterparty_vpa="CP1",
            event_time=t_fail3,
            status="failed",
            failure_domain="customer",
        ),
        settings,
    )

    prop = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_05",
        payee_bank_id=BANK_0,
        payee_vpa="CP1",
        payee_type="person",
        amount_paise=10000,
        event_time=t_now,
    )
    feats = compute_payment_features(redis_client, prop, t_now, settings)

    # Hand-worked expectation: exactly 2 customer failures; sender_bank technical failure excluded
    # count == 2 (not 3)
    assert feats["customer_failed_attempts_24h"] == 2


def test_invalid_proposed_payment_raises() -> None:
    """Verifies that malformed ProposedPayment instances fail fast with ValueError."""
    t_now = datetime(2026, 10, 28, 16, 0, 0, tzinfo=IST)

    # 1. Non-positive amount
    with pytest.raises(ValueError, match="positive integer"):
        ProposedPayment(RUN_ID, BANK_0, "payer", BANK_0, "payee", "person", 0, t_now)
    with pytest.raises(ValueError, match="positive integer"):
        ProposedPayment(RUN_ID, BANK_0, "payer", BANK_0, "payee", "person", -500, t_now)

    # 2. Naive event_time
    with pytest.raises(ValueError, match="timezone-aware"):
        ProposedPayment(RUN_ID, BANK_0, "payer", BANK_0, "payee", "person", 1000, datetime(2026, 10, 28, 12, 0, 0))

    # 3. Invalid payee_type
    with pytest.raises(ValueError, match="payee_type"):
        ProposedPayment(RUN_ID, BANK_0, "payer", BANK_0, "payee", "robot", 1000, t_now)

    # 4. Empty identifier
    with pytest.raises(ValueError, match="payer_account_id"):
        ProposedPayment(RUN_ID, BANK_0, "", BANK_0, "payee", "person", 1000, t_now)


def test_compute_payer_only_payment_features_full_proposal(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that compute_payer_only_payment_features calculates all payer-side features on full proposal."""
    proposal = {
        "run_id": RUN_ID,
        "bank_id": BANK_0,
        "payer_account_id": "acc_01",
        "payee_bank_id": BANK_0,
        "payee_vpa": "vpa_01",
        "amount_paise": 10000,
        "channel": "upi_app",
        "sender_device_fingerprint": "fp_01",
        "sender_os_generation": 14,
        "payee_type": "merchant",
    }
    feats = compute_payer_only_payment_features(redis_client, proposal, NOW_UTC)
    assert feats["amount_paise"] == 10000
    assert feats["payment_channel"] == "upi_app"
    assert feats["payer_device_fingerprint"] == "fp_01"
    assert feats["payer_os_generation"] == 14
    assert feats["round_amount"] is True
    assert feats["structured_amount"] is False
    assert feats["same_amount_count_24h"] == 0
    assert feats["payee_type"] == "merchant"
    assert feats["payee_registry_match"] is None


def test_compute_payer_only_payment_features_missing_optional_fields(redis_client: fakeredis.FakeRedis) -> None:
    """
    SABOTAGE S3 TARGET:
    Verifies that missing optional fields (payee_type, channel, fingerprint, os_gen) cleanly produce None
    without raising any exceptions.
    """
    proposal = {
        "run_id": RUN_ID,
        "bank_id": BANK_0,
        "payer_account_id": "acc_01",
        "payee_bank_id": BANK_0,
        "payee_vpa": "vpa_01",
        "amount_paise": 10000,
        "channel": None,
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "payee_type": None,
    }
    # Must NOT raise; missing optional fields produce None in feature dictionary
    feats = compute_payer_only_payment_features(redis_client, proposal, NOW_UTC)
    assert feats["payee_type"] is None
    assert feats["payment_channel"] is None
    assert feats["payer_device_fingerprint"] is None
    assert feats["payer_os_generation"] is None


def test_compute_payer_only_payment_features_payee_vpa_none(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that proposal with payee_vpa=None returns payee_registry_match=None without calling screener."""
    from features.registry_screening import FakeRegistryScreener

    screener = FakeRegistryScreener()
    proposal = {
        "run_id": RUN_ID,
        "bank_id": BANK_0,
        "payer_account_id": "acc_01",
        "payee_bank_id": BANK_0,
        "payee_vpa": None,
        "amount_paise": 10000,
        "payee_type": "person",
    }
    feats = compute_payer_only_payment_features(redis_client, proposal, NOW_UTC, screener=screener)
    assert feats["payee_registry_match"] is None

