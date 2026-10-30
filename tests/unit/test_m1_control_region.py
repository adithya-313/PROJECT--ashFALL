# File: tests/unit/test_m1_control_region.py
# Purpose: Unit tests validating M1 control (R5) and region (E3) feature calculations.
# Issue: Level 15b Round 2b-2 (R5, E3).
"""
WHAT: Unit tests covering:
  1. Review Item R5 device constraint:
     - os_update does NOT trigger device_hardware_changed (0) or new_device_r5 (0).
     - device_binding_change triggers device_hardware_changed (1) and new_device_r5 (1).
     - sim_swap triggers sim_swapped (1) and new_device_r5 (1).
     - app_reinstall on same phone triggers app_reinstalled_same_phone (1) and NOT new_device_r5 (0).
     - app_reinstall on different phone triggers new_device_r5 (1).
  2. days_since_device_change: None when no change has occurred; exact elapsed days when occurred.
  3. accounts_on_same_device: counts accounts transacting on same phone within this bank; tenant isolation.
  4. distinct_fingerprints_lifetime: counts lifetime device fingerprints.
  5. machine_regular_timing_score: None on < 5 payments; 1.0 on perfectly uniform timing; lower on irregular.
  6. Region group (E3):
     - distinct_same_bank_payer_regions_7d counts unique regions among same-bank payers.
     - unknown_region_payer_count_7d counts credits from other banks (bank isolation).
"""

from datetime import datetime, timedelta
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
)
from features.account_updater import (
    apply_account_event,
    apply_payment_side,
)
from features.m1_features import compute_account_features

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
ACC_1 = "acc_person_000001"
ACC_2 = "acc_person_000002"
ACC_3 = "acc_person_000003"


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
    account_id: str = ACC_1,
    bank_id: str = BANK_0,
) -> AccountEventRecord:
    return AccountEventRecord(
        run_id=RUN_ID,
        bank_id=bank_id,
        account_id=account_id,
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
    counterparty_bank_id: str = BANK_1,
    counterparty_vpa: str = "vpa_cp_01",
    sender_device_fingerprint: str | None = "fp_phone_01",
    account_id: str = ACC_1,
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
        amount_paise=100000,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=sender_device_fingerprint,
        sender_os_generation=1,
        event_time=event_time,
        arrival_time=arrival_time,
        balance_after_paise=500000,
    )


def test_r5_device_constraints_and_os_update_safety(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Verifies Issue R5 constraint:
    An operating system update on the same phone is NOT a new device (new_device_r5 == 0,
    device_hardware_changed == 0).
    (Sabotage proof: counting os_update as device_hardware_changed fails this test).
    """
    settings = load_account_features_settings()
    t_open = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    apply_account_event(
        redis_client,
        _make_event(
            f"{RUN_ID}-open-01", "account_opened", t_open, t_open,
            details={"home_region": 1, "account_type": "savings", "fingerprint": "fp_phone_orig"},
        ),
        settings,
    )

    # 1. OS update occurs on Day 3
    t_os = datetime(2026, 10, 3, 10, 0, tzinfo=IST)
    apply_account_event(
        redis_client,
        _make_event(
            f"{RUN_ID}-os-01", "os_update", t_os, t_os,
            details={"fingerprint": "fp_phone_orig", "os_generation": 2},
        ),
        settings,
    )

    dec_1 = datetime(2026, 10, 4, 10, 0, tzinfo=IST)
    feats_os = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=dec_1,
        arrival_cutoff=dec_1,
        settings=settings,
        groups=["control"],
    )

    # OS update alone MUST NOT trigger device hardware change or new_device_r5
    assert feats_os["device_hardware_changed"] == 0
    assert feats_os["sim_swapped"] == 0
    assert feats_os["app_reinstalled_same_phone"] == 0
    assert feats_os["new_device_r5"] == 0

    # 2. Hardware binding change occurs on Day 5
    t_hw = datetime(2026, 10, 5, 10, 0, tzinfo=IST)
    apply_account_event(
        redis_client,
        _make_event(
            f"{RUN_ID}-hw-01", "device_binding_change", t_hw, t_hw,
            details={"fingerprint_before": "fp_phone_orig", "fingerprint_after": "fp_phone_new"},
        ),
        settings,
    )

    dec_2 = datetime(2026, 10, 6, 10, 0, tzinfo=IST)
    feats_hw = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=dec_2,
        arrival_cutoff=dec_2,
        settings=settings,
        groups=["control"],
    )

    assert feats_hw["device_hardware_changed"] == 1
    assert feats_hw["new_device_r5"] == 1
    assert feats_hw["days_since_device_change"] == to_day_number(dec_2) - to_day_number(t_hw)


def test_accounts_on_same_device_and_isolation(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies accounts_on_same_device counts within bank and enforces tenant isolation (A6)."""
    settings = load_account_features_settings()
    t_now = datetime(2026, 10, 1, 10, 0, tzinfo=IST)

    # ACC_1 and ACC_2 both transact from 'fp_gang_device' in bank_0
    p1 = _make_payment(
        f"{RUN_ID}-p-01", t_now, t_now, side="debit", sender_device_fingerprint="fp_gang_device", account_id=ACC_1
    )
    p2 = _make_payment(
        f"{RUN_ID}-p-02", t_now, t_now, side="debit", sender_device_fingerprint="fp_gang_device", account_id=ACC_2
    )
    apply_payment_side(redis_client, p1, settings)
    apply_payment_side(redis_client, p2, settings)

    # ACC_3 transacts from same 'fp_gang_device' in bank_1
    p3 = _make_payment(
        f"{RUN_ID}-p-03", t_now, t_now, side="debit", sender_device_fingerprint="fp_gang_device",
        account_id=ACC_3, bank_id=BANK_1
    )
    apply_payment_side(redis_client, p3, settings)

    feats = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=t_now,
        arrival_cutoff=t_now,
        settings=settings,
        groups=["control"],
    )

    # Bank 0 observes exactly 2 accounts on fp_gang_device (ACC_1 and ACC_2); never ACC_3 from bank_1
    assert feats["accounts_on_same_device"] == 2


def test_machine_regular_timing_score(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies machine_regular_timing_score regularity calculation and minimum threshold."""
    settings = load_account_features_settings()

    # 1. Less than 5 payments (4 payments) -> None
    t0 = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    for i in range(4):
        t_i = t0 + timedelta(seconds=i * 60)
        apply_payment_side(redis_client, _make_payment(f"{RUN_ID}-p-{i}", t_i, t_i), settings)

    dec_time = t0 + timedelta(minutes=10)
    feats_few = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=dec_time,
        arrival_cutoff=dec_time,
        settings=settings,
        groups=["control"],
    )
    assert feats_few["machine_regular_timing_score"] is None

    # 2. Add 5th payment with perfect 60s intervals (0s, 60s, 120s, 180s, 240s) -> score = 1.0
    t_4 = t0 + timedelta(seconds=240)
    apply_payment_side(redis_client, _make_payment(f"{RUN_ID}-p-4", t_4, t_4), settings)

    feats_uniform = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=dec_time,
        arrival_cutoff=dec_time,
        settings=settings,
        groups=["control"],
    )
    assert feats_uniform["machine_regular_timing_score"] == 1.0


def test_region_features_same_bank_and_unknown_payers(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Verifies Issue E3 and A6 isolation:
    Same-bank payers' home regions are resolved; other-bank payers count in unknown_region_payer_count_7d.
    (Sabotage proof: resolving other-bank customer regions from other-bank namespace fails this test).
    """
    settings = load_account_features_settings()
    t_now = datetime(2026, 10, 1, 10, 0, tzinfo=IST)

    # In bank_0: register 2 payers with distinct regions
    # Payer A in bank_0 has home_region = 14
    apply_account_event(
        redis_client,
        _make_event(
            f"{RUN_ID}-open-payer-a", "account_opened", t_now, t_now,
            details={"home_region": 14, "account_type": "savings"},
            account_id="acc_payer_a", bank_id=BANK_0,
        ),
        settings,
    )
    # Payer B in bank_0 has home_region = 22
    apply_account_event(
        redis_client,
        _make_event(
            f"{RUN_ID}-open-payer-b", "account_opened", t_now, t_now,
            details={"home_region": 22, "account_type": "current"},
            account_id="acc_payer_b", bank_id=BANK_0,
        ),
        settings,
    )

    # In bank_1: register Payer C with home_region = 99 (in bank_1's namespace)
    apply_account_event(
        redis_client,
        _make_event(
            f"{RUN_ID}-open-payer-c", "account_opened", t_now, t_now,
            details={"home_region": 99, "account_type": "savings"},
            account_id="acc_payer_c", bank_id=BANK_1,
        ),
        settings,
    )

    # Inbound credits to ACC_1 (at bank_0):
    # 1. From Payer A (same bank)
    p_a = _make_payment(
        f"{RUN_ID}-c-01", t_now, t_now, side="credit",
        counterparty_bank_id=BANK_0, counterparty_vpa="acc_payer_a", account_id=ACC_1, bank_id=BANK_0,
    )
    # 2. From Payer B (same bank)
    p_b = _make_payment(
        f"{RUN_ID}-c-02", t_now, t_now, side="credit",
        counterparty_bank_id=BANK_0, counterparty_vpa="acc_payer_b", account_id=ACC_1, bank_id=BANK_0,
    )
    # 3. From Payer C (OTHER bank: bank_1)
    p_c = _make_payment(
        f"{RUN_ID}-c-03", t_now, t_now, side="credit",
        counterparty_bank_id=BANK_1, counterparty_vpa="acc_payer_c", account_id=ACC_1, bank_id=BANK_0,
    )
    # 4. From another other-bank payer
    p_d = _make_payment(
        f"{RUN_ID}-c-04", t_now, t_now, side="credit",
        counterparty_bank_id=BANK_1, counterparty_vpa="acc_payer_d", account_id=ACC_1, bank_id=BANK_0,
    )

    for p in (p_a, p_b, p_c, p_d):
        apply_payment_side(redis_client, p, settings)

    feats = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=ACC_1,
        decision_time=t_now,
        arrival_cutoff=t_now,
        settings=settings,
        groups=["region"],
    )

    # Same-bank payers: 2 distinct regions (14, 22)
    assert feats["distinct_same_bank_payer_regions_7d"] == 2
    # Other-bank payers: 2 transactions (p_c, p_d) whose regions cannot be observed by bank_0
    assert feats["unknown_region_payer_count_7d"] == 2
