# File: showcase/level_15_support_b.py
# Purpose: Level 15b verification check implementations: M1 behavioral feature engine,
#          bank ledger isolation, R5 OS update safety, and rented mule detection.

import uuid
from collections.abc import Sequence
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


def _format_table(title: str, headers: list[str], rows: Sequence[Sequence[Any]]) -> str:
    """Renders a clean, monospace-aligned plain-text table for showcase outputs."""
    str_rows = [[str(cell if cell is not None else "-") for cell in row] for row in rows]
    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if len(cell) > col_widths[i]:
                col_widths[i] = len(cell)

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [f"\n{title}:", sep, header_line, sep]
    for row in str_rows:
        line = "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


async def check_15b_1_three_profiles_normal(workspace: Any) -> CheckOutcome:
    """
    Check 15b-1 (normal, part b): a mule, a tea stall and a salaried person side by side.
    Feeds representative event histories for three canonical account types into the feature engine:
      - a mule: fast pass-through (ratio > 0.8), multi-channel cash out, rapid credit-to-debit.
      - a tea stall: high small-credit count, burst of distinct payers in peak hours, zero pass-through.
      - a salaried person: long history (> 60 days), zero small-credit bursts, low velocity.
    Prints an aligned feature comparison table and asserts defining bounds.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = load_account_features_settings()
    run_id = f"sc15b1_{uuid.uuid4().hex[:10]}"
    bank_id = "bank_0"
    dec_time = datetime(2026, 11, 4, 18, 0, tzinfo=IST)
    arrival_cutoff = dec_time

    try:
        # 1. Salaried person (65 days history, regular salary credits, 0 small credits)
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

        # 2. Tea stall (daily small credits, 8 distinct payers in 1h evening rush, zero pass-through)
        tea_acc = f"acc_tea_{run_id[:8]}"
        open_tea = dec_time - timedelta(days=90)
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
        # 8 distinct small credits in the last 1h rush
        for i in range(8):
            t_rush = dec_time - timedelta(minutes=50 - i * 5)
            apply_payment_side(
                client,
                PaymentSideRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=tea_acc,
                    payment_id=f"{run_id}-tea-rush-{i}",
                    side="credit",
                    counterparty_bank_id="bank_1",
                    counterparty_vpa=f"cust_rush_{i}@upi",
                    amount_paise=2500,  # ₹25
                    status="success",
                    failure_domain=None,
                    payee_type="merchant",
                    channel="upi_app",
                    sender_device_fingerprint=f"fp_cust_{i}",
                    sender_os_generation=1,
                    balance_after_paise=50000 + i * 2500,
                    event_time=t_rush,
                    arrival_time=t_rush,
                ),
                settings,
            )
        # 15 additional small credits earlier in the 24h window
        for i in range(15):
            t_morn = dec_time - timedelta(hours=5, minutes=i * 10)
            apply_payment_side(
                client,
                PaymentSideRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=tea_acc,
                    payment_id=f"{run_id}-tea-morn-{i}",
                    side="credit",
                    counterparty_bank_id="bank_1",
                    counterparty_vpa=f"cust_morn_{i}@upi",
                    amount_paise=3000,
                    status="success",
                    failure_domain=None,
                    payee_type="merchant",
                    channel="upi_app",
                    sender_device_fingerprint=f"fp_morn_{i}",
                    sender_os_generation=1,
                    balance_after_paise=20000 + i * 3000,
                    event_time=t_morn,
                    arrival_time=t_morn,
                ),
                settings,
            )

        # 3. Mule account (fast pass-through > 0.8, ATM + UPI cash-out, rapid credit-to-debit)
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
        t_mule_cred = dec_time - timedelta(hours=2)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-mule-scam-1",
                side="credit",
                counterparty_bank_id="bank_victim",
                counterparty_vpa="victim1@upi",
                amount_paise=5000000,  # ₹50,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_victim_1",
                sender_os_generation=1,
                balance_after_paise=5000000,
                event_time=t_mule_cred,
                arrival_time=t_mule_cred,
            ),
            settings,
        )
        t_mule_deb = t_mule_cred + timedelta(minutes=15)
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
                amount_paise=4200000,  # ₹42,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_mule_gang",
                sender_os_generation=1,
                balance_after_paise=800000,
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
                amount_paise=600000,  # ₹6,000
                status="success",
                failure_domain=None,
                balance_after_paise=200000,
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

        # Print aligned feature table
        headers = ["M1 Feature", "Mule Account", "Tea Stall", "Salaried Person"]
        rows = [
            [
                "history_days_available",
                f_mule["history_days_available"],
                f_tea["history_days_available"],
                f_sal["history_days_available"],
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
                "pass_through_ratio_24h",
                f_mule["pass_through_ratio_24h"],
                f_tea["pass_through_ratio_24h"],
                f_sal["pass_through_ratio_24h"],
            ],
            [
                "outflow_to_person_paise_7d",
                f_mule["outflow_to_person_paise_7d"],
                f_tea["outflow_to_person_paise_7d"],
                f_sal["outflow_to_person_paise_7d"],
            ],
            [
                "cash_atm_paise_7d",
                f_mule["cash_atm_paise_7d"],
                f_tea["cash_atm_paise_7d"],
                f_sal["cash_atm_paise_7d"],
            ],
            [
                "median_credit_to_debit_seconds_7d",
                f_mule["median_credit_to_debit_seconds_7d"],
                f_tea["median_credit_to_debit_seconds_7d"],
                f_sal["median_credit_to_debit_seconds_7d"],
            ],
            [
                "first_time_payer_share_24h",
                f_mule["first_time_payer_share_24h"],
                f_tea["first_time_payer_share_24h"],
                f_sal["first_time_payer_share_24h"],
            ],
        ]
        print(_format_table("M1 Profiles Side-by-Side", headers, rows))

        # Assert defining bounds
        # Mule: pass-through > 0.8, multi-channel cash out (> 0 on ATM and person), fast median delay (< 3600s)
        assert isinstance(f_mule["pass_through_ratio_24h"], (int, float))
        assert f_mule["pass_through_ratio_24h"] > 0.8
        assert isinstance(f_mule["cash_atm_paise_7d"], int) and f_mule["cash_atm_paise_7d"] > 0
        assert isinstance(f_mule["outflow_to_person_paise_7d"], int) and f_mule["outflow_to_person_paise_7d"] > 0
        assert isinstance(f_mule["median_credit_to_debit_seconds_7d"], (int, float))
        assert f_mule["median_credit_to_debit_seconds_7d"] < 3600

        # Tea stall: high small credits (> 15), high distinct payers in rush (>= 5), zero pass-through
        assert isinstance(f_tea["small_credit_count_24h"], int) and f_tea["small_credit_count_24h"] >= 20
        assert isinstance(f_tea["distinct_payers_1h"], int) and f_tea["distinct_payers_1h"] >= 5
        assert f_tea["pass_through_ratio_24h"] == 0.0

        # Salaried: history > 60 days, zero small credit bursts, low velocity
        assert isinstance(f_sal["history_days_available"], int) and f_sal["history_days_available"] > 60
        assert f_sal["small_credit_count_24h"] == 0
        assert f_sal["pass_through_ratio_24h"] is None

        return CheckOutcome(
            key_number="15b-1",
            evidence={
                "salaried_history_days": f_sal["history_days_available"],
                "tea_stall_small_credits": f_tea["small_credit_count_24h"],
                "tea_stall_distinct_1h": f_tea["distinct_payers_1h"],
                "mule_pass_through_ratio": f_mule["pass_through_ratio_24h"],
                "mule_credit_to_debit_seconds": f_mule["median_credit_to_debit_seconds_7d"],
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
        rows: Sequence[Sequence[Any]] = [
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
        print(_format_table("Cross-Bank Perspective Isolation", headers, rows))

        return CheckOutcome(
            key_number="15b-2",
            evidence={
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
        Features: device_hardware_changed == 0, new_device_r5 == 0, distinct_fingerprints == 1.
      - Day 7: Genuine hardware switch occurs (new fingerprint fp_phone_new).
        Features: device_hardware_changed == 1, new_device_r5 == 1, distinct_fingerprints == 2.
    Asserts both transitions.
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

        # OS update must NOT trigger device hardware change or new_device_r5 flag
        assert feats_os["device_hardware_changed"] == 0
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

        headers = ["Lifecycle Event", "device_hardware_changed", "new_device_r5", "distinct_fingerprints_lifetime"]
        rows = [
            [
                "After Day 5 OS Update",
                feats_os["device_hardware_changed"],
                feats_os["new_device_r5"],
                feats_os["distinct_fingerprints_lifetime"],
            ],
            [
                "After Day 7 New Device",
                feats_hw["device_hardware_changed"],
                feats_hw["new_device_r5"],
                feats_hw["distinct_fingerprints_lifetime"],
            ],
        ]
        print(_format_table("R5 OS Update vs Genuine Device Transition", headers, rows))

        return CheckOutcome(
            key_number="15b-3",
            evidence={
                "os_update_device_hardware_changed": feats_os["device_hardware_changed"],
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
    Compares a rented mule (dormant account reactivated suddenly, burst of credits from new payers
    followed within minutes by ATM cash extraction and cross-bank debits) against a genuine salaried person.
    Prints the comparison table and asserts dormancy reactivation, high pass-through, and fast credit-to-debit delay.
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
        # 1. Salaried person: long history, regular payroll, zero dormancy gap
        open_sal = dec_time - timedelta(days=70)
        apply_account_event(
            client,
            AccountEventRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=sal_acc,
                event_id=f"{run_id}-open-sal",
                kind="account_opened",
                event_time=open_sal,
                arrival_time=open_sal,
                details={"home_region": 1, "account_type": "savings", "fingerprint": "fp_sal_01"},
            ),
            settings,
        )
        for d in (65, 35, 5):
            t_sal = dec_time - timedelta(days=d)
            apply_payment_side(
                client,
                PaymentSideRecord(
                    run_id=run_id,
                    bank_id=bank_id,
                    account_id=sal_acc,
                    payment_id=f"{run_id}-sal-p-{d}",
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

        # 2. Rented mule: opened 80 days ago, initial activity 45 days ago, then prolonged dormancy.
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
        t_early_mule = dec_time - timedelta(days=45)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-mule-early",
                side="credit",
                counterparty_bank_id="bank_other",
                counterparty_vpa="dep@upi",
                amount_paise=100000,
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_mule_orig",
                sender_os_generation=1,
                balance_after_paise=100000,
                event_time=t_early_mule,
                arrival_time=t_early_mule,
            ),
            settings,
        )

        # Reactivation on day 62 (3 days ago, fully closed by decision time):
        t_react = dec_time - timedelta(days=3)
        apply_account_event(
            client,
            AccountEventRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                event_id=f"{run_id}-mule-react",
                kind="account_reactivated",
                event_time=t_react,
                arrival_time=t_react,
                details={"reactivated_by": "customer_login"},
            ),
            settings,
        )
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-mule-react-pmt",
                side="credit",
                counterparty_bank_id="bank_other",
                counterparty_vpa="react_payer@upi",
                amount_paise=500000,
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_mule_gang",
                sender_os_generation=1,
                balance_after_paise=600000,
                event_time=t_react,
                arrival_time=t_react,
            ),
            settings,
        )

        # Scam burst on decision day:
        t_scam1 = dec_time - timedelta(hours=2)
        apply_payment_side(
            client,
            PaymentSideRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                payment_id=f"{run_id}-scam-1",
                side="credit",
                counterparty_bank_id="bank_victim",
                counterparty_vpa="victim_fresh@upi",
                amount_paise=5000000,  # ₹50,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_victim",
                sender_os_generation=1,
                balance_after_paise=5600000,
                event_time=t_scam1,
                arrival_time=t_scam1,
            ),
            settings,
        )
        # Fast transfer out 15 min later
        t_scam_out = t_scam1 + timedelta(minutes=15)
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
                amount_paise=4200000,  # ₹42,000
                status="success",
                failure_domain=None,
                payee_type="person",
                channel="upi_app",
                sender_device_fingerprint="fp_mule_gang",
                sender_os_generation=1,
                balance_after_paise=1400000,
                event_time=t_scam_out,
                arrival_time=t_scam_out,
            ),
            settings,
        )
        # Fast ATM cash extraction 10 min later
        t_scam_atm = t_scam_out + timedelta(minutes=10)
        apply_withdrawal(
            client,
            WithdrawalRecord(
                run_id=run_id,
                bank_id=bank_id,
                account_id=mule_acc,
                withdrawal_id=f"{run_id}-scam-atm",
                channel="atm",
                amount_paise=600000,  # ₹6,000
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

        headers = ["Behavioral Feature", "Rented Mule", "Salaried Person"]
        rows = [
            [
                "dormancy_days_before_recent_activity",
                f_mule["dormancy_days_before_recent_activity"],
                f_sal["dormancy_days_before_recent_activity"],
            ],
            [
                "pass_through_ratio_24h",
                f_mule["pass_through_ratio_24h"],
                f_sal["pass_through_ratio_24h"],
            ],
            [
                "median_credit_to_debit_seconds_7d",
                f_mule["median_credit_to_debit_seconds_7d"],
                f_sal["median_credit_to_debit_seconds_7d"],
            ],
            [
                "first_time_payer_share_24h",
                f_mule["first_time_payer_share_24h"],
                f_sal["first_time_payer_share_24h"],
            ],
            [
                "cash_atm_paise_7d",
                f_mule["cash_atm_paise_7d"],
                f_sal["cash_atm_paise_7d"],
            ],
            [
                "history_days_available",
                f_mule["history_days_available"],
                f_sal["history_days_available"],
            ],
        ]
        print(_format_table("Rented Mule vs Salaried Person Comparison", headers, rows))

        # Assert defining bounds:
        # Mule: dormant reactivated (>= 30 days gap), high pass-through (> 0.8), fast credit-to-debit (< 3600s)
        assert isinstance(f_mule["dormancy_days_before_recent_activity"], int)
        assert f_mule["dormancy_days_before_recent_activity"] >= 30
        assert isinstance(f_mule["pass_through_ratio_24h"], (int, float))
        assert f_mule["pass_through_ratio_24h"] > 0.8
        assert isinstance(f_mule["median_credit_to_debit_seconds_7d"], (int, float))
        assert f_mule["median_credit_to_debit_seconds_7d"] < 3600

        # Salaried: long history (> 60), no pass-through
        assert isinstance(f_sal["history_days_available"], int) and f_sal["history_days_available"] > 60
        assert f_sal["pass_through_ratio_24h"] is None

        return CheckOutcome(
            key_number="15b-4",
            evidence={
                "mule_dormancy_days": f_mule["dormancy_days_before_recent_activity"],
                "mule_pass_through_ratio": f_mule["pass_through_ratio_24h"],
                "mule_credit_to_debit_seconds": f_mule["median_credit_to_debit_seconds_7d"],
                "salaried_history_days": f_sal["history_days_available"],
            },
        )
    finally:
        for k in client.scan_iter(match=namespace_pattern(run_id, bank_id)):
            client.delete(k)
        client.close()
