# File: tests/unit/test_payee_feature_visibility.py
# Purpose: Unit tests for payee feature computation and cross-bank data isolation (Decisions D1-D6, Ground Rule 16).
"""
WHAT: Unit tests verifying compute_payee_vpa_payment_features under tenant boundary isolation rules.
WHY: Ground rule 16 and Level 16c architect decisions:
- same-bank payee evaluates risk from this bank's own labelled/complained accounts.
- cross-bank payee always returns None for payee_same_bank_risk, even if flagged in another bank's partition.
- first_payment_to_payee tracks payer-specific successful counterparty history only.
- payee_registry_match is None when no screener is supplied (national registry screener wired in Level 16d).
"""
from datetime import datetime, timezone

import fakeredis
import pytest

from features.account_keys import (
    complained_accounts_key,
    labelled_accounts_key,
    load_account_features_settings,
)
from features.account_records import PaymentSideRecord
from features.account_updater import apply_payment_side
from features.m1_payment import compute_payee_vpa_payment_features

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
NOW = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


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
    """Helper constructing PaymentSideRecord with all required fields."""
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
        payee_type="merchant",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        balance_after_paise=None,
        event_time=event_time,
        arrival_time=event_time,
    )


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    """Provides an isolated in-memory Redis client."""
    return fakeredis.FakeRedis()


def test_same_bank_payee_risk_levels(redis_client: fakeredis.FakeRedis) -> None:
    """
    Verifies payee_same_bank_risk evaluates correctly for same-bank counterparties:
    - 0.0 for clean unflagged accounts
    - 0.5 for accounts present in this bank's registered complaint set
    - 1.0 for accounts present in this bank's confirmed fraud labelled set
    """
    settings = load_account_features_settings()
    score = NOW.timestamp()

    # 1. Clean same-bank account -> 0.0
    clean_feats = compute_payee_vpa_payment_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_0,
        payee_vpa="acc_clean_01",
        event_time=NOW,
        arrival_cutoff=NOW,
        settings=settings,
    )
    assert clean_feats["payee_same_bank_risk"] == 0.0

    # 2. Complained account in this bank's key -> 0.5
    redis_client.zadd(complained_accounts_key(RUN_ID, BANK_0), {"acc_cmp_01": score})
    cmp_feats = compute_payee_vpa_payment_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_0,
        payee_vpa="acc_cmp_01",
        event_time=NOW,
        arrival_cutoff=NOW,
        settings=settings,
    )
    assert cmp_feats["payee_same_bank_risk"] == 0.5

    # 3. Labelled fraudulent account in this bank's key -> 1.0
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {"acc_fraud_01": score})
    fraud_feats = compute_payee_vpa_payment_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_0,
        payee_vpa="acc_fraud_01",
        event_time=NOW,
        arrival_cutoff=NOW,
        settings=settings,
    )
    assert fraud_feats["payee_same_bank_risk"] == 1.0


def test_cross_bank_payee_risk_is_none_ground_rule_16(redis_client: fakeredis.FakeRedis) -> None:
    """
    Ground Rule 16 (Tenant Isolation):
    payee_same_bank_risk is None when payee_bank_id != evaluating bank_id,
    EVEN IF the payee account is labelled as confirmed fraud or complained in ANOTHER bank's keys.
    """
    settings = load_account_features_settings()
    score = NOW.timestamp()

    # The payee account is explicitly flagged as fraud in bank_1's partition
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_1), {"acc_cross_mule": score})
    redis_client.zadd(complained_accounts_key(RUN_ID, BANK_1), {"acc_cross_mule": score})

    # Evaluated from bank_0's perspective for a cross-bank payment to bank_1
    cross_feats = compute_payee_vpa_payment_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_1,
        payee_vpa="acc_cross_mule",
        event_time=NOW,
        arrival_cutoff=NOW,
        settings=settings,
    )
    assert cross_feats["payee_same_bank_risk"] is None


def test_first_payment_to_payee_lifecycle_and_tenant_isolation(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """
    Verifies first_payment_to_payee semantics:
    1. Returns True for a new payee unseen in payer's counterparty records.
    2. A payment made to that payee by a customer at a DIFFERENT bank does NOT change it to False.
    3. Becomes False only after the payer's OWN earlier successful payment to that payee.
    """
    settings = load_account_features_settings()
    target_payee_vpa = "acc_shop_001"

    # 1. New payee -> first_payment_to_payee is True
    initial_feats = compute_payee_vpa_payment_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_1,
        payee_vpa=target_payee_vpa,
        event_time=NOW,
        arrival_cutoff=NOW,
        settings=settings,
    )
    assert initial_feats["first_payment_to_payee"] is True

    # 2. A payment to target_payee by a customer at bank_1 does NOT change bank_0 payer's feature
    other_customer_side = _make_payment_side(
        payment_id="pay_other_001",
        account_id="acc_other_customer",
        side="debit",
        counterparty_bank_id=BANK_1,
        counterparty_vpa=target_payee_vpa,
        event_time=NOW,
        amount_paise=100000,
        status="success",
        bank_id=BANK_1,
    )
    apply_payment_side(redis_client, other_customer_side, settings)

    after_other_feats = compute_payee_vpa_payment_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_1,
        payee_vpa=target_payee_vpa,
        event_time=NOW,
        arrival_cutoff=NOW,
        settings=settings,
    )
    assert after_other_feats["first_payment_to_payee"] is True

    # 3. Payer's own successful payment to target_payee makes subsequent first_payment_to_payee False
    own_payment_side = _make_payment_side(
        payment_id="pay_own_001",
        account_id="acc_payer_01",
        side="debit",
        counterparty_bank_id=BANK_1,
        counterparty_vpa=target_payee_vpa,
        event_time=NOW,
        amount_paise=150000,
        status="success",
        bank_id=BANK_0,
    )
    apply_payment_side(redis_client, own_payment_side, settings)

    later_time = NOW + datetime.resolution
    after_own_feats = compute_payee_vpa_payment_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_1,
        payee_vpa=target_payee_vpa,
        event_time=later_time,
        arrival_cutoff=later_time,
        settings=settings,
    )
    assert after_own_feats["first_payment_to_payee"] is False


def test_payee_registry_match_none_when_no_screener(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """
    Verifies payee_registry_match returns None when no RegistryScreener is supplied.
    Note: The national intelligence registry is wired in Level 16d.
    """
    settings = load_account_features_settings()
    feats = compute_payee_vpa_payment_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id="acc_payer_01",
        payee_bank_id=BANK_1,
        payee_vpa="acc_merchant_01",
        event_time=NOW,
        arrival_cutoff=NOW,
        settings=settings,
        screener=None,
    )
    assert feats["payee_registry_match"] is None
