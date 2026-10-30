# File: showcase/level_15_support_b.py
# Purpose: Level 15b verification check implementations: M1 behavioral feature engine,
#          bank ledger isolation, R5 OS update safety, and rented mule detection.

import uuid
from datetime import datetime, timedelta
from typing import Any

from core.event_clock import IST
from features.account_keys import (
    load_account_features_settings,
    namespace_pattern,
)
from features.account_records import (
    AccountEventRecord,
    PaymentSideRecord,
    WithdrawalRecord,
)
from features.account_updater import (
    apply_account_event,
    apply_payment_side,
    apply_withdrawal,
)
from features.m1_features import SUPPORTED_GROUPS, compute_account_features
from showcase.framework import CheckOutcome
from showcase.level_15 import _get_redis, host_side_redis_url


async def check_15b_1_three_profiles_normal(workspace: Any) -> CheckOutcome:
    """
    Check 15b-1 (normal, part b): a mule, a tea stall and a salaried person side by side.
    Feeds representative event histories for three canonical account types into the feature engine:
      - a salaried person: long history (> 60 days), routine_life_present is True, zero small credits.
      - a tea stall: >= 40 days history, mostly repeat same-bank local payers (first_time_payer_share_24h <= 0.3,
        distinct_same_bank_payer_regions_7d >= 1, pass_through_ratio_7d <= 0.3).
      - a mule: fast pass-through (pass_through_ratio_7d >= 0.9, first_time_payer_share_24h >= 0.8).
    Asserts each profile's separating signal and puts comparison rows into outcome evidence.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = load_account_features_settings()
    run_id = f"sc15b1_{uuid.uuid4().hex[:10]}"
    bank_id = "bank_0"
    dec_time = datetime(2026, 11, 4, 18, 0, tzinfo=IST)
    arrival_cutoff = dec_time

    try:
        # 1. Salaried person (65 days history, regular salary credits, recurring utility, routine_life_present True)
        sal_acc = f"acc_sal_{run_id[:8]}"
        open_time = dec_time - timedelta(days=70)
        apply_account_event(
            client,
            AccountEventRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=sal_acc,
                event_id=f"{run_id}-open-sal",
                kind="account_opened",
                event_time=open_time,
                arrival_time=open_time,
                details={"home_region": 1, "account_type": "savings", "fingerprint": "fp_sal_01"},
            ),
            settings,
        )
        for day_offset in (65, 35, 5):
            t_sal = dec_time - timedelta(days=day_offset)
            apply_payment_side(
                client,
                PaymentSideRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=sal_acc,
                    payment_id=f"{run_id}-sal-pmt-{day_offset}",
                    side="credit",
                    counterparty_bank_id="bank_employer",
                    counterparty_vpa="payroll@corporate",
                    amount_paise=7500000,
                    status="success",
                    failure_domain=None,
                    payee_type="merchant",
                    channel="neft",
                    sender_device_fingerprint=None,
                    sender_os_generation=None,
                    balance_after_paise=7500000,
                    event_time=t_sal,
                    arrival_time=t_sal,
                ),
                settings,
            )
        for day_offset in (34, 4):
            t_deb = dec_time - timedelta(days=day_offset)
            apply_payment_side(
                client,
                PaymentSideRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=sal_acc,
                    payment_id=f"{run_id}-sal-deb-{day_offset}",
                    side="debit",
                    counterparty_bank_id=bank_id,
                    counterparty_vpa="utility@merchant",
                    amount_paise=200000,
                    status="success",
                    failure_domain=None,
                    payee_type="merchant",
                    channel="upi_app",
                    sender_device_fingerprint="fp_sal_01",
                    sender_os_generation=1,
                    balance_after_paise=7300000,
                    event_time=t_deb,
                    arrival_time=t_deb,
                ),
                settings,
            )

        # 2. Tea stall (>= 40 days history, mostly repeat same-bank local payers with stored home_region)
        tea_acc = f"acc_tea_{run_id[:8]}"
        open_tea = dec_time - timedelta(days=60)
        apply_account_event(
            client,
            AccountEventRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=tea_acc,
                event_id=f"{run_id}-open-tea",
                kind="account_opened",
                event_time=open_tea,
                arrival_time=open_tea,
                details={"home_region": 1, "account_type": "current", "fingerprint": "fp_tea_qr"},
            ),
            settings,
        )
        local_payers = [f"{run_id}_payer_{i}" for i in range(5)]
        for i, p_acc in enumerate(local_payers):
            t_p_open = dec_time - timedelta(days=55)
            apply_account_event(
                client,
                AccountEventRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=p_acc,
                    event_id=f"{run_id}-open-p-{i}",
                    kind="account_opened",
                    event_time=t_p_open,
                    arrival_time=t_p_open,
                    details={"home_region": 1, "account_type": "savings", "fingerprint": f"fp_p_{i}"},
                ),
                settings,
            )
        # Closed-day credits from repeat local customers across the 40 days (days 45, 30, 15, 3)
        for d_off in (45, 30, 15, 3):
            t_hist = dec_time - timedelta(days=d_off)
            for i, p_acc in enumerate(local_payers):
                apply_payment_side(
                    client,
                    PaymentSideRecord(
                        run_id=run_id,
                        bank_id=bank_id,
                        account_id=tea_acc,
                        payment_id=f"{run_id}-tea-hist-{d_off}-{i}",
                        side="credit",
                        counterparty_bank_id=bank_id,
                        counterparty_vpa=p_acc,
                        amount_paise=2500,
                        status="success",
                        failure_domain=None,
                        payee_type="merchant",
                        channel="upi_app",
                        sender_device_fingerprint=f"fp_p_{i}",
                        sender_os_generation=1,
                        balance_after_paise=100000 + i * 2500,
                        event_time=t_hist,
                        arrival_time=t_hist,
                    ),
                    settings,
                )
        # In the last 1h rush: repeat customers pay
        for i in range(5):
            t_rush = dec_time - timedelta(minutes=45 - i * 8)
            apply_payment_side(
                client,
                PaymentSideRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=tea_acc,
                    payment_id=f"{run_id}-tea-rush-{i}",
                    side="credit",
                    counterparty_bank_id=bank_id,
                    counterparty_vpa=local_payers[i],
                    amount_paise=2500,
                    status="success",
                    failure_domain=None,
                    payee_type="merchant",
                    channel="upi_app",
                    sender_device_fingerprint=f"fp_p_{i}",
                    sender_os_generation=1,
                    balance_after_paise=150000 + i * 2500,
                    event_time=t_rush,
                    arrival_time=t_rush,
                ),
                settings,
            )
        # Additional credits in 24h from same repeat customers
        for j in range(15):
            p_idx = j % 5
            t_morn = dec_time - timedelta(hours=4, minutes=j * 10)
            apply_payment_side(
                client,
                PaymentSideRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=tea_acc,
                    payment_id=f"{run_id}-tea-morn-{j}",
                    side="credit",
                    counterparty_bank_id=bank_id,
                    counterparty_vpa=local_payers[p_idx],
                    amount_paise=3000,
                    status="success",
                    failure_domain=None,
                    payee_type="merchant",
                    channel="upi_app",
                    sender_device_fingerprint=f"fp_p_{p_idx}",
                    sender_os_generation=1,
                    balance_after_paise=165000 + j * 3000,
                    event_time=t_morn,
                    arrival_time=t_morn,
                ),
                settings,
            )
        # One first-time walk-in tourist in 24h
        t_walkin = dec_time - timedelta(hours=2)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=tea_acc,
                payment_id=f"{run_id}-tea-walkin",
                side="credit",
                counterparty_bank_id="bank_other",
                counterparty_vpa="walkin_tourist@upi",
                amount_paise=5000,
                status="success",
                failure_domain=None,
                payee_type="merchant",
                channel="upi_app",
                sender_device_fingerprint="fp_walkin",
                sender_os_generation=1,
                balance_after_paise=220000,
                event_time=t_walkin,
                arrival_time=t_walkin,
            ),
            settings,
        )

        # 3. Mule account (fast pass-through >= 0.9, first_time_payer_share_24h >= 0.8)
        mule_acc = f"acc_mule_{run_id[:8]}"
        open_mule = dec_time - timedelta(days=80)
        apply_account_event(
            client,
            AccountEventRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                event_id=f"{run_id}-open-mule",
                kind="account_opened",
                event_time=open_mule,
                arrival_time=open_mule,
                details={"home_region": 2, "account_type": "savings", "fingerprint": "fp_mule_orig"},
            ),
            settings,
        )
        t_mule_cred1 = dec_time - timedelta(hours=2)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-mule-scam-1",
                side="credit",
                counterparty_bank_id="bank_victim_1",
                counterparty_vpa="victim1@upi",
                amount_paise=5000000,  # ₹50,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_victim_1",
                sender_os_generation=1,
                balance_after_paise=5000000,
                event_time=t_mule_cred1,
                arrival_time=t_mule_cred1,
            ),
            settings,
        )
        t_mule_cred2 = dec_time - timedelta(hours=1, minutes=45)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-mule-scam-2",
                side="credit",
                counterparty_bank_id="bank_victim_2",
                counterparty_vpa="victim2@upi",
                amount_paise=5000000,  # ₹50,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_victim_2",
                sender_os_generation=1,
                balance_after_paise=10000000,
                event_time=t_mule_cred2,
                arrival_time=t_mule_cred2,
            ),
            settings,
        )
        t_mule_deb = t_mule_cred2 + timedelta(minutes=15)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-mule-deb-1",
                side="debit",
                counterparty_bank_id="bank_other",
                counterparty_vpa="layer2@upi",
                amount_paise=8500000,  # ₹85,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_mule_gang",
                sender_os_generation=1,
                balance_after_paise=1500000,
                event_time=t_mule_deb,
                arrival_time=t_mule_deb,
            ),
            settings,
        )
        t_mule_atm = t_mule_deb + timedelta(minutes=10)
        apply_withdrawal(
            client,
            WithdrawalRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                withdrawal_id=f"{run_id}-mule-atm-1",
                channel="atm",
                amount_paise=1000000,  # ₹10,000
                status="success",
                failure_domain=None,
                balance_after_paise=500000,
                event_time=t_mule_atm,
                arrival_time=t_mule_atm,
            ),
            settings,
        )

        all_groups = list(SUPPORTED_GROUPS)
        f_sal = compute_account_features(
            client, run_id, bank_id, sal_acc, dec_time, arrival_cutoff, settings, groups=all_groups
        )
        f_tea = compute_account_features(
            client, run_id, bank_id, tea_acc, dec_time, arrival_cutoff, settings, groups=all_groups
        )
        f_mule = compute_account_features(
            client, run_id, bank_id, mule_acc, dec_time, arrival_cutoff, settings, groups=all_groups
        )

        # Assert separating signals per FIX 2
        # Salaried: routine life present, zero small credits
        assert f_sal["routine_life_present"] is True
        assert f_sal["small_credit_count_24h"] == 0

        # Tea stall: mostly repeat payers (<= 0.3), same bank regions (>= 1), pass-through (<= 0.3)
        assert isinstance(f_tea["first_time_payer_share_24h"], (int, float))
        assert f_tea["first_time_payer_share_24h"] <= 0.3
        assert isinstance(f_tea["distinct_same_bank_payer_regions_7d"], int)
        assert f_tea["distinct_same_bank_payer_regions_7d"] >= 1
        assert isinstance(f_tea["pass_through_ratio_7d"], (int, float))
        assert f_tea["pass_through_ratio_7d"] <= 0.3
        assert isinstance(f_tea["small_credit_count_24h"], int) and f_tea["small_credit_count_24h"] >= 20
        assert isinstance(f_tea["distinct_payers_1h"], int) and f_tea["distinct_payers_1h"] >= 5

        # Mule: pass-through 7d >= 0.9 and first-time payer share 24h >= 0.8
        assert isinstance(f_mule["pass_through_ratio_7d"], (int, float))
        assert f_mule["pass_through_ratio_7d"] >= 0.9
        assert isinstance(f_mule["first_time_payer_share_24h"], (int, float))
        assert f_mule["first_time_payer_share_24h"] >= 0.8

        headers = ["M1 Feature", "Mule Account", "Tea Stall", "Salaried Person"]
        rows = [
            [
                "routine_life_present",
                f_mule["routine_life_present"],
                f_tea["routine_life_present"],
                f_sal["routine_life_present"],
            ],
            [
                "pass_through_ratio_7d",
                f_mule["pass_through_ratio_7d"],
                f_tea["pass_through_ratio_7d"],
                f_sal["pass_through_ratio_7d"],
            ],
            [
                "first_time_payer_share_24h",
                f_mule["first_time_payer_share_24h"],
                f_tea["first_time_payer_share_24h"],
                f_sal["first_time_payer_share_24h"],
            ],
            [
                "distinct_same_bank_payer_regions_7d",
                f_mule["distinct_same_bank_payer_regions_7d"],
                f_tea["distinct_same_bank_payer_regions_7d"],
                f_sal["distinct_same_bank_payer_regions_7d"],
            ],
            [
                "small_credit_count_24h",
                f_mule["small_credit_count_24h"],
                f_tea["small_credit_count_24h"],
                f_sal["small_credit_count_24h"],
            ],
            [
                "distinct_payers_1h",
                f_mule["distinct_payers_1h"],
                f_tea["distinct_payers_1h"],
                f_sal["distinct_payers_1h"],
            ],
            [
                "history_days_available",
                f_mule["history_days_available"],
                f_tea["history_days_available"],
                f_sal["history_days_available"],
            ],
        ]
        evidence_sentence = (
            "M1 behavioral profiles cleanly separate: salaried person by routine_life_present, "
            "tea stall by repeat payers (low first_time_payer_share_24h and same-bank regions), "
            "and mule by high pass_through_ratio_7d and first_time_payer_share_24h."
        )

        return CheckOutcome(
            key_number=(
                f"mule pass-through {f_mule['pass_through_ratio_7d']:.2f}, "
                f"tea repeat share {1 - float(f_tea['first_time_payer_share_24h']):.2f}, "
                f"salaried routine {f_sal['routine_life_present']}"
            ),
            evidence={
                "headers": headers,
                "rows": rows,
                "evidence_sentence": evidence_sentence,
                "salaried_routine_life_present": f_sal["routine_life_present"],
                "tea_stall_first_time_share": f_tea["first_time_payer_share_24h"],
                "tea_stall_same_bank_regions": f_tea["distinct_same_bank_payer_regions_7d"],
                "tea_stall_pass_through_7d": f_tea["pass_through_ratio_7d"],
                "mule_pass_through_7d": f_mule["pass_through_ratio_7d"],
                "mule_first_time_share_24h": f_mule["first_time_payer_share_24h"],
            },
        )
    finally:
        for k in client.scan_iter(match=namespace_pattern(run_id, bank_id)):
            client.delete(k)
        client.close()


async def check_15b_2_cross_bank_isolation_edge(workspace: Any) -> CheckOutcome:
    """
    Check 15b-2 (edge, part b): a cross-bank payment shows only each bank's own side.
    Sets up a cross-bank payment from bank_0 (sender debit) to bank_1 (receiver credit).
    Computes features from both banks' perspectives:
      - bank_0 sees only the debit: outflow group reflects it, inflow metrics are 0.
      - bank_1 sees only the credit: inflow group reflects it, outflow metrics are 0.
    Asserts neither bank's feature vector contains data from the other's internal ledger.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = load_account_features_settings()
    run_id = f"sc15b2_{uuid.uuid4().hex[:10]}"
    t_pmt = datetime(2026, 11, 4, 12, 0, tzinfo=IST)
    dec_time = t_pmt + timedelta(minutes=10)

    sender_acc = f"acc_send_{run_id[:8]}"
    recv_acc = f"acc_recv_{run_id[:8]}"

    try:
        # Bank 0: records debit side only
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id="bank_0",
                account_id=sender_acc,
                payment_id=f"{run_id}-pmt-xbank",
                side="debit",
                counterparty_bank_id="bank_1",
                counterparty_vpa="receiver@bank1",
                amount_paise=50000,  # ₹500
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_sender_0",
                sender_os_generation=1,
                balance_after_paise=150000,
                event_time=t_pmt,
                arrival_time=t_pmt,
            ),
            settings,
        )

        # Bank 1: records credit side only
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id="bank_1",
                account_id=recv_acc,
                payment_id=f"{run_id}-pmt-xbank",
                side="credit",
                counterparty_bank_id="bank_0",
                counterparty_vpa="sender@bank0",
                amount_paise=50000,  # ₹500
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_sender_0",
                sender_os_generation=1,
                balance_after_paise=250000,
                event_time=t_pmt,
                arrival_time=t_pmt,
            ),
            settings,
        )

        f_b0 = compute_account_features(
            client, run_id, "bank_0", sender_acc, dec_time, dec_time, settings, groups=["inflow", "outflow"]
        )
        f_b1 = compute_account_features(
            client, run_id, "bank_1", recv_acc, dec_time, dec_time, settings, groups=["inflow", "outflow"]
        )

        # Bank 0 sees only outbound debit: inflow features must be 0
        assert f_b0["small_credit_count_24h"] == 0
        assert f_b0["distinct_payers_1h"] == 0
        assert f_b0["other_bank_outflow_share_7d"] == 1.0  # Outflow was to bank_1

        # Bank 1 sees only inbound credit: outflow features must be 0 or empty
        assert f_b1["small_credit_count_24h"] == 1
        assert f_b1["distinct_payers_1h"] == 1
        assert f_b1["pass_through_ratio_24h"] == 0.0
        assert f_b1["other_bank_outflow_share_7d"] is None

        headers = ["Perspective", "Small Credits (24h)", "Distinct Payers (1h)", "Other Bank Outflow Share"]
        rows = [
            [
                "Bank 0 (Sender)",
                f_b0["small_credit_count_24h"],
                f_b0["distinct_payers_1h"],
                f_b0["other_bank_outflow_share_7d"],
            ],
            [
                "Bank 1 (Receiver)",
                f_b1["small_credit_count_24h"],
                f_b1["distinct_payers_1h"],
                f_b1["other_bank_outflow_share_7d"],
            ],
        ]

        return CheckOutcome(
            key_number=(
                f"b0 other-bank share {f_b0['other_bank_outflow_share_7d']:.1f} vs "
                f"b1 small credits {f_b1['small_credit_count_24h']}"
            ),
            evidence={
                "headers": headers,
                "rows": rows,
                "bank_0_small_credits": f_b0["small_credit_count_24h"],
                "bank_0_other_bank_outflow_share": f_b0["other_bank_outflow_share_7d"],
                "bank_1_small_credits": f_b1["small_credit_count_24h"],
                "bank_1_other_bank_outflow_share": f_b1["other_bank_outflow_share_7d"],
            },
        )
    finally:
        for b in ("bank_0", "bank_1"):
            for k in client.scan_iter(match=namespace_pattern(run_id, b)):
                client.delete(k)
        client.close()


async def check_15b_3_os_update_not_new_device_edge(workspace: Any) -> CheckOutcome:
    """
    Check 15b-3 (edge, part b): an OS update is not a new device.
    Proves Issue R5 constraint:
      - Day 1: Account transacts on phone (fingerprint fp_phone_orig, OS 14.0).
      - Day 5: OS update occurs on the same phone (fp_phone_orig, OS 15.0).
        Features: device_hardware_changed == 0, sim_swapped == 0, app_reinstalled_same_phone == 0,
                  new_device_r5 == 0, distinct_fingerprints == 1.
      - Day 7: Genuine hardware switch occurs (new fingerprint fp_phone_new).
        Features: device_hardware_changed == 1, new_device_r5 == 1, distinct_fingerprints == 2.
    Asserts all four R5 flags after the OS update, and subsequent detection on genuine hardware switch.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = load_account_features_settings()
    run_id = f"sc15b3_{uuid.uuid4().hex[:10]}"
    bank_id = "bank_0"
    acc_id = f"acc_dev_{run_id[:8]}"

    try:
        # Day 1: Account opened and active on original hardware
        t_day1 = datetime(2026, 11, 1, 10, 0, tzinfo=IST)
        apply_account_event(
            client,
            AccountEventRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=acc_id,
                event_id=f"{run_id}-open",
                kind="account_opened",
                event_time=t_day1,
                arrival_time=t_day1,
                details={"home_region": 1, "account_type": "savings", "fingerprint": "fp_phone_orig"},
            ),
            settings,
        )
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=acc_id,
                payment_id=f"{run_id}-pmt-d1",
                side="debit",
                counterparty_bank_id="bank_1",
                counterparty_vpa="shop@upi",
                amount_paise=10000,
                status="success",
                failure_domain=None,
                payee_type="merchant",
                channel="upi_app",
                sender_device_fingerprint="fp_phone_orig",
                sender_os_generation=1,
                balance_after_paise=90000,
                event_time=t_day1,
                arrival_time=t_day1,
            ),
            settings,
        )

        # Day 5: OS update on the same phone hardware
        t_day5 = datetime(2026, 11, 5, 10, 0, tzinfo=IST)
        apply_account_event(
            client,
            AccountEventRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=acc_id,
                event_id=f"{run_id}-os-up",
                kind="os_update",
                event_time=t_day5,
                arrival_time=t_day5,
                details={"fingerprint": "fp_phone_orig", "os_generation": 2},
            ),
            settings,
        )

        dec_os = t_day5 + timedelta(hours=1)
        feats_os = compute_account_features(
            client, run_id, bank_id, acc_id, dec_os, dec_os, settings, groups=["control"]
        )

        # OS update must NOT trigger any of the four R5 flags
        assert feats_os["device_hardware_changed"] == 0
        assert feats_os["sim_swapped"] == 0
        assert feats_os["app_reinstalled_same_phone"] == 0
        assert feats_os["new_device_r5"] == 0
        assert feats_os["distinct_fingerprints_lifetime"] == 1

        # Day 7: Genuine physical hardware switch (device binding change)
        t_day7 = datetime(2026, 11, 7, 10, 0, tzinfo=IST)
        apply_account_event(
            client,
            AccountEventRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=acc_id,
                event_id=f"{run_id}-hw-change",
                kind="device_binding_change",
                event_time=t_day7,
                arrival_time=t_day7,
                details={"fingerprint_before": "fp_phone_orig", "fingerprint_after": "fp_phone_new"},
            ),
            settings,
        )
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=acc_id,
                payment_id=f"{run_id}-pmt-d7",
                side="debit",
                counterparty_bank_id="bank_1",
                counterparty_vpa="shop@upi",
                amount_paise=15000,
                status="success",
                failure_domain=None,
                payee_type="merchant",
                channel="upi_app",
                sender_device_fingerprint="fp_phone_new",
                sender_os_generation=1,
                balance_after_paise=75000,
                event_time=t_day7,
                arrival_time=t_day7,
            ),
            settings,
        )

        dec_hw = t_day7 + timedelta(hours=1)
        feats_hw = compute_account_features(
            client, run_id, bank_id, acc_id, dec_hw, dec_hw, settings, groups=["control"]
        )

        # Genuine hardware switch MUST trigger hardware change, new_device_r5, and increment fingerprints
        assert feats_hw["device_hardware_changed"] == 1
        assert feats_hw["new_device_r5"] == 1
        assert feats_hw["distinct_fingerprints_lifetime"] == 2

        headers = [
            "Lifecycle Event",
            "device_hardware_changed",
            "sim_swapped",
            "app_reinstalled_same_phone",
            "new_device_r5",
            "distinct_fingerprints_lifetime",
        ]
        rows = [
            [
                "After Day 5 OS Update",
                feats_os["device_hardware_changed"],
                feats_os["sim_swapped"],
                feats_os["app_reinstalled_same_phone"],
                feats_os["new_device_r5"],
                feats_os["distinct_fingerprints_lifetime"],
            ],
            [
                "After Day 7 New Device",
                feats_hw["device_hardware_changed"],
                feats_hw["sim_swapped"],
                feats_hw["app_reinstalled_same_phone"],
                feats_hw["new_device_r5"],
                feats_hw["distinct_fingerprints_lifetime"],
            ],
        ]

        return CheckOutcome(
            key_number=(
                f"OS update: all 4 R5 flags 0, distinct fp {feats_os['distinct_fingerprints_lifetime']}; "
                f"HW switch: hw_changed {feats_hw['device_hardware_changed']}, "
                f"new_device_r5 {feats_hw['new_device_r5']}"
            ),
            evidence={
                "headers": headers,
                "rows": rows,
                "os_update_device_hardware_changed": feats_os["device_hardware_changed"],
                "os_update_sim_swapped": feats_os["sim_swapped"],
                "os_update_app_reinstalled_same_phone": feats_os["app_reinstalled_same_phone"],
                "os_update_new_device_r5": feats_os["new_device_r5"],
                "os_update_distinct_fingerprints": feats_os["distinct_fingerprints_lifetime"],
                "hw_switch_device_hardware_changed": feats_hw["device_hardware_changed"],
                "hw_switch_new_device_r5": feats_hw["new_device_r5"],
                "hw_switch_distinct_fingerprints": feats_hw["distinct_fingerprints_lifetime"],
            },
        )
    finally:
        for k in client.scan_iter(match=namespace_pattern(run_id, bank_id)):
            client.delete(k)
        client.close()


async def check_15b_4_rented_mule_vs_salaried_edge(workspace: Any) -> CheckOutcome:
    """
    Check 15b-4 (edge, part b): a rented mule compared with a genuine salaried person.
    Proves the plan's point: a rented mule keeps its owner's normal routine life.
    Both accounts have ordinary salaried owner rhythm (monthly salary credit, recurring merchant payments,
    at least 45 days history, no dormancy gap).
    In the last 7 days, mule activity is added to the rented mule (first-time payers, fast pass-through >= 0.9,
    outflow to another bank, cash ATM extraction).
    Asserts both have routine life, but only the rented mule exhibits anomalous recent velocity.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = load_account_features_settings()
    run_id = f"sc15b4_{uuid.uuid4().hex[:10]}"
    bank_id = "bank_0"
    dec_time = datetime(2026, 11, 4, 18, 0, tzinfo=IST)
    arrival_cutoff = dec_time

    mule_acc = f"acc_rented_{run_id[:8]}"
    sal_acc = f"acc_sal_{run_id[:8]}"

    try:
        # Both accounts: ordinary salaried owners, opened 70 days ago (>= 45 days history, no dormancy gap)
        for acc_id, fp in [(sal_acc, "fp_sal_01"), (mule_acc, "fp_mule_owner")]:
            open_t = dec_time - timedelta(days=70)
            apply_account_event(
                client,
                AccountEventRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=acc_id,
                    event_id=f"{run_id}-open-{acc_id}",
                    kind="account_opened",
                    event_time=open_t,
                    arrival_time=open_t,
                    details={"home_region": 1, "account_type": "savings", "fingerprint": fp},
                ),
                settings,
            )
            # Monthly salary credits on days 65, 35, 5 (30-day rhythm)
            for d in (65, 35, 5):
                t_sal = dec_time - timedelta(days=d)
                apply_payment_side(
                    client,
                    PaymentSideRecord(
                        run_id=run_id,
                        bank_id=bank_id,
                        account_id=acc_id,
                        payment_id=f"{run_id}-sal-{acc_id}-{d}",
                        side="credit",
                        counterparty_bank_id="bank_employer",
                        counterparty_vpa="payroll@corporate",
                        amount_paise=7500000,
                        status="success",
                        failure_domain=None,
                        payee_type="merchant",
                        channel="neft",
                        sender_device_fingerprint=None,
                        sender_os_generation=None,
                        balance_after_paise=7500000,
                        event_time=t_sal,
                        arrival_time=t_sal,
                    ),
                    settings,
                )
            # Recurring merchant debits on days 34 and 4 (2 distinct closed days)
            for d in (34, 4):
                t_deb = dec_time - timedelta(days=d)
                apply_payment_side(
                    client,
                    PaymentSideRecord(
                        run_id=run_id,
                        bank_id=bank_id,
                        account_id=acc_id,
                        payment_id=f"{run_id}-deb-{acc_id}-{d}",
                        side="debit",
                        counterparty_bank_id=bank_id,
                        counterparty_vpa="utility@merchant",
                        amount_paise=200000,
                        status="success",
                        failure_domain=None,
                        payee_type="merchant",
                        channel="upi_app",
                        sender_device_fingerprint=fp,
                        sender_os_generation=1,
                        balance_after_paise=7300000,
                        event_time=t_deb,
                        arrival_time=t_deb,
                    ),
                    settings,
                )

        # Rented mule: add recent 7-day mule activity on decision day
        # 1. Credits from first-time payers in 24h
        t_scam1 = dec_time - timedelta(hours=4)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-scam-1",
                side="credit",
                counterparty_bank_id="bank_victim_1",
                counterparty_vpa="victim_fresh_1@upi",
                amount_paise=2500000,  # ₹25,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_victim_1",
                sender_os_generation=1,
                balance_after_paise=9800000,
                event_time=t_scam1,
                arrival_time=t_scam1,
            ),
            settings,
        )
        t_scam2 = dec_time - timedelta(hours=3)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-scam-2",
                side="credit",
                counterparty_bank_id="bank_victim_2",
                counterparty_vpa="victim_fresh_2@upi",
                amount_paise=2500000,  # ₹25,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_victim_2",
                sender_os_generation=1,
                balance_after_paise=12300000,
                event_time=t_scam2,
                arrival_time=t_scam2,
            ),
            settings,
        )
        # 2. Fast transfer out to another bank (different bank_id) 1h later
        t_scam_out = dec_time - timedelta(hours=2)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-scam-out",
                side="debit",
                counterparty_bank_id="bank_other",
                counterparty_vpa="syndicate_collector@upi",
                amount_paise=10000000,  # ₹100,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_mule_gang",
                sender_os_generation=1,
                balance_after_paise=2300000,
                event_time=t_scam_out,
                arrival_time=t_scam_out,
            ),
            settings,
        )
        # 3. Cash ATM extraction 1h later
        t_scam_atm = dec_time - timedelta(hours=1)
        apply_withdrawal(
            client,
            WithdrawalRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                withdrawal_id=f"{run_id}-scam-atm",
                channel="atm",
                amount_paise=1500000,  # ₹15,000
                status="success",
                failure_domain=None,
                balance_after_paise=800000,
                event_time=t_scam_atm,
                arrival_time=t_scam_atm,
            ),
            settings,
        )

        all_groups = list(SUPPORTED_GROUPS)
        f_mule = compute_account_features(
            client, run_id, bank_id, mule_acc, dec_time, arrival_cutoff, settings, groups=all_groups
        )
        f_sal = compute_account_features(
            client, run_id, bank_id, sal_acc, dec_time, arrival_cutoff, settings, groups=all_groups
        )

        # Assert with literals per FIX 1:
        # BOTH accounts: routine_life_present is True and salary_like_payer_count >= 1
        assert f_sal["routine_life_present"] is True
        assert isinstance(f_sal["salary_like_payer_count"], int) and f_sal["salary_like_payer_count"] >= 1
        assert f_mule["routine_life_present"] is True
        assert isinstance(f_mule["salary_like_payer_count"], int) and f_mule["salary_like_payer_count"] >= 1

        # The rented mule ONLY: pass_through_ratio_7d >= 0.9, other_bank_outflow_share_7d > 0,
        # first_time_payer_share_24h >= 0.5
        assert isinstance(f_mule["pass_through_ratio_7d"], (int, float))
        assert f_mule["pass_through_ratio_7d"] >= 0.9
        assert isinstance(f_mule["other_bank_outflow_share_7d"], (int, float))
        assert f_mule["other_bank_outflow_share_7d"] > 0
        assert isinstance(f_mule["first_time_payer_share_24h"], (int, float))
        assert f_mule["first_time_payer_share_24h"] >= 0.5

        # The salaried person: pass_through_ratio_7d <= 0.5 or None
        assert f_sal["pass_through_ratio_7d"] is None or (
            isinstance(f_sal["pass_through_ratio_7d"], (int, float))
            and f_sal["pass_through_ratio_7d"] <= 0.5
        )

        evidence_sentence = (
            "Routine life is the same for both, so it cannot tell a rented mule from its owner; "
            "the recent pass-through, first-time payers and other-bank outflow can."
        )

        headers = ["Behavioral Feature", "Rented Mule", "Salaried Person"]
        rows = [
            ["routine_life_present", f_mule["routine_life_present"], f_sal["routine_life_present"]],
            ["salary_like_payer_count", f_mule["salary_like_payer_count"], f_sal["salary_like_payer_count"]],
            ["pass_through_ratio_7d", f_mule["pass_through_ratio_7d"], f_sal["pass_through_ratio_7d"]],
            [
                "other_bank_outflow_share_7d",
                f_mule["other_bank_outflow_share_7d"],
                f_sal["other_bank_outflow_share_7d"],
            ],
            ["first_time_payer_share_24h", f_mule["first_time_payer_share_24h"], f_sal["first_time_payer_share_24h"]],
            ["cash_atm_paise_7d", f_mule["cash_atm_paise_7d"], f_sal["cash_atm_paise_7d"]],
            ["history_days_available", f_mule["history_days_available"], f_sal["history_days_available"]],
        ]

        return CheckOutcome(
            key_number=(
                "routine life present on both; "
                f"mule pass-through {f_mule['pass_through_ratio_7d']:.2f} vs "
                f"salaried {f_sal['pass_through_ratio_7d']:.2f}"
            ),
            evidence={
                "headers": headers,
                "rows": rows,
                "evidence_sentence": evidence_sentence,
                "mule_routine_life_present": f_mule["routine_life_present"],
                "mule_salary_like_payers": f_mule["salary_like_payer_count"],
                "mule_pass_through_7d": f_mule["pass_through_ratio_7d"],
                "mule_other_bank_outflow_share_7d": f_mule["other_bank_outflow_share_7d"],
                "mule_first_time_payer_share_24h": f_mule["first_time_payer_share_24h"],
                "salaried_routine_life_present": f_sal["routine_life_present"],
                "salaried_salary_like_payers": f_sal["salary_like_payer_count"],
                "salaried_pass_through_7d": f_sal["pass_through_ratio_7d"],
            },
        )
    finally:
        for k in client.scan_iter(match=namespace_pattern(run_id, bank_id)):
            client.delete(k)
        client.close()
