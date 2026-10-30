# File: features/m1_control_region.py
# Purpose: Level 15b M1 control (R5) and region (E3) feature calculation functions.
# Issue: Level 15b M1 features (plan Task 1, R5, E3).
"""
WHAT: Pure and state-backed functions computing M1 control and region behavioral features:
  - Control (R5): days_since_device_change, accounts_on_same_device, device_hardware_changed,
    sim_swapped, app_reinstalled_same_phone, new_device_r5, distinct_fingerprints_lifetime,
    machine_regular_timing_score.
  - Region (E3): distinct_same_bank_payer_regions_7d, unknown_region_payer_count_7d.

WHY: M1 models require device control signals and payer geodiversity while strictly honoring tenant
isolation (A6: a bank never reads another bank's customer data). Device novelty must conform to
review item R5 constraint (operating system updates reach many people and are NOT a new device).
"""

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from features.account_keys import (
    AccountFeaturesSettings,
    get_default_settings,
)
from features.account_reader import AccountSnapshot
from features.account_summaries import (
    read_accounts_on_device,
    read_device_changes,
    read_home_region,
    read_lifetime,
    to_day_number,
)


def compute_control_features(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    snapshot: AccountSnapshot,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> dict[str, Any]:
    """
    PURPOSE:
        Computes the M1 control group features adhering to Issue R5 design constraints.

    INPUTS:
        client: Redis client.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        snapshot: Point-in-time AccountSnapshot holding raw 7-day window records.
        decision_time: Evaluation decision timestamp.
        arrival_cutoff: Ingestion watermark storage cutoff.
        settings: Optional AccountFeaturesSettings.

    KEY LOGIC:
        - new_device_r5 conforms strictly to review item R5:
          "A 'new device' means a change of hardware, a changed or swapped SIM, or a fresh
          app install on a different phone. An operating system update on the same phone
          is NOT a new device, and a rule must not fire on an update alone."
        - accounts_on_same_device queries this bank's device index for distinct customer accounts.
        - distinct_fingerprints_lifetime queries visible device fingerprints on the account.
        - machine_regular_timing_score measures arrival regularity across >= 5 payments in 7d.
    """
    cfg = settings if settings is not None else get_default_settings()
    cutoff_7d = decision_time - timedelta(seconds=cfg.device_change_window_seconds)

    # 1. days_since_device_change: calendar days since latest device change
    changes = read_device_changes(client, run_id, bank_id, account_id, decision_time, arrival_cutoff, cfg)
    if not changes:
        days_since_device_change = None
    else:
        latest_change_score = changes[-1][1]
        change_dt = datetime.fromtimestamp(latest_change_score, tz=timezone.utc)
        days_since_device_change = max(0, to_day_number(decision_time) - to_day_number(change_dt))

    # 2. 7-day account events filtering
    recent_events = [ev for ev in snapshot.account_events if cutoff_7d < ev.event_time <= decision_time]

    device_hardware_changed = 0
    sim_swapped = 0
    app_reinstalled_same_phone = 0
    fresh_app_different_phone = 0

    for ev in recent_events:
        if ev.kind == "device_binding_change":
            device_hardware_changed = 1
        elif ev.kind == "sim_swap":
            sim_swapped = 1
        elif ev.kind == "app_reinstall":
            fp_before = ev.details.get("fingerprint_before")
            fp_after = ev.details.get("fingerprint_after")
            if fp_before and fp_after and fp_before != fp_after:
                fresh_app_different_phone = 1
            else:
                app_reinstalled_same_phone = 1

    # new_device_r5 composite: 1 on hardware change, sim swap, or fresh app on different phone; 0 on OS updates
    new_device_r5 = 1 if (device_hardware_changed or sim_swapped or fresh_app_different_phone) else 0

    # 3. accounts_on_same_device: count distinct accounts transacting on current device within this bank
    current_fp = None
    # Check debit payment sides in 7d (newest first)
    recent_debits = [
        r
        for r in snapshot.side_records
        if r.side == "debit" and cutoff_7d < r.event_time <= decision_time and r.sender_device_fingerprint
    ]
    if recent_debits:
        recent_debits.sort(key=lambda r: r.event_time, reverse=True)
        current_fp = recent_debits[0].sender_device_fingerprint
    else:
        # Check recent account events for fingerprint
        for ev in reversed(recent_events):
            fp = ev.details.get("fingerprint_after") or ev.details.get("fingerprint")
            if isinstance(fp, str) and fp.strip():
                current_fp = fp.strip()
                break

    if not current_fp:
        accounts_on_same_device = 0
    else:
        device_accs = read_accounts_on_device(
            client, run_id, bank_id, current_fp, decision_time, arrival_cutoff, cfg
        )
        accounts_on_same_device = len(device_accs)

    # 4. distinct_fingerprints_lifetime
    lifetime = read_lifetime(client, run_id, bank_id, account_id, decision_time, arrival_cutoff, cfg)
    distinct_fingerprints_lifetime = len(lifetime.fingerprints)

    # 5. machine_regular_timing_score: statistical regularity across >= machine_timing_min_payments
    recent_payments = [r for r in snapshot.side_records if cutoff_7d < r.event_time <= decision_time]
    recent_payments.sort(key=lambda r: r.event_time)

    if len(recent_payments) < cfg.machine_timing_min_payments:
        machine_regular_timing_score = None
    else:
        deltas = [
            (recent_payments[i + 1].event_time - recent_payments[i].event_time).total_seconds()
            for i in range(len(recent_payments) - 1)
        ]
        mean_delta = statistics.mean(deltas)
        stdev_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        if mean_delta == 0:
            machine_regular_timing_score = 1.0
        else:
            cv = stdev_delta / mean_delta
            machine_regular_timing_score = round(1.0 / (1.0 + cv), 4)

    return {
        "days_since_device_change": days_since_device_change,
        "accounts_on_same_device": accounts_on_same_device,
        "device_hardware_changed": device_hardware_changed,
        "sim_swapped": sim_swapped,
        "app_reinstalled_same_phone": app_reinstalled_same_phone,
        "new_device_r5": new_device_r5,
        "distinct_fingerprints_lifetime": distinct_fingerprints_lifetime,
        "machine_regular_timing_score": machine_regular_timing_score,
    }


def compute_region_features(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    snapshot: AccountSnapshot,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> dict[str, Any]:
    """
    PURPOSE:
        Computes the M1 region group features: distinct_same_bank_payer_regions_7d and
        unknown_region_payer_count_7d, strictly adhering to tenant bank isolation (A6).

    INPUTS:
        client: Redis client.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        snapshot: Point-in-time AccountSnapshot holding raw 7-day window records.
        decision_time: Scoring evaluation timestamp.
        arrival_cutoff: Ingestion watermark storage cutoff.
        settings: Optional AccountFeaturesSettings.

    KEY LOGIC:
        - A bank observes the home_region of counterparties ONLY if the counterparty belongs
          to this bank (counterparty_bank_id == bank_id).
        - Payer regions from other banks are unknown (A6 isolation rule); their transactions
          increment unknown_region_payer_count_7d.
    """
    cfg = settings if settings is not None else get_default_settings()
    cutoff_7d = decision_time - timedelta(seconds=604800)

    recent_credits = [
        r
        for r in snapshot.side_records
        if r.side == "credit" and cutoff_7d < r.event_time <= decision_time
    ]

    same_bank_regions: set[str] = set()
    unknown_payer_count = 0

    for rec in recent_credits:
        if rec.counterparty_bank_id == bank_id:
            # Same bank: resolve home region from this bank's namespace
            payer_account = rec.counterparty_vpa
            if payer_account:
                reg = read_home_region(
                    client, run_id, bank_id, payer_account, decision_time, arrival_cutoff, cfg
                )
                if reg is not None:
                    same_bank_regions.add(reg)
        else:
            # Other bank: region cannot be determined by this bank
            unknown_payer_count += 1

    return {
        "distinct_same_bank_payer_regions_7d": len(same_bank_regions),
        "unknown_region_payer_count_7d": unknown_payer_count,
    }
