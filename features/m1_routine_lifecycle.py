# File: features/m1_routine_lifecycle.py
# Purpose: Level 15b M1 routine_life and lifecycle behavioral feature calculation functions.
# Issue: Level 15b M1 features (plan Task 1, G1, R8).
"""
WHAT: Pure and state-backed functions computing M1 routine_life and lifecycle features:
  - Routine life: salary_like_payer_count, recurring_merchant_payee_count,
    regular_transfer_payee_count, routine_life_present, history_days_available.
  - Lifecycle: account_age_days, account_type, dormancy_days_before_recent_activity,
    volume_vs_own_history_ratio.

WHY: M1 models require long-horizon memory features to detect deviations from established customer
rhythms. Commercial banks observe product classifications (savings/current), calculate dormancy gaps,
and detect monthly salary/merchant patterns across closed-day summaries.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from features.account_keys import (
    AccountFeaturesSettings,
    active_days_key,
    get_default_settings,
)
from features.account_reader import AccountSnapshot
from features.account_summaries import (
    day_closed_threshold,
    history_days_available,
    read_account_type,
    read_counterparty_days,
    read_counterparty_payee_types,
    read_day_totals,
    read_lifetime,
    to_day_number,
)


def monthly_rhythm(
    days: Sequence[int],
    routine_cycle_days: int = 30,
    tolerance: int = 5,
) -> bool:
    """
    PURPOSE:
        Evaluates whether a sequence of active day numbers exhibits an established monthly rhythm.

    INPUTS:
        days: Sequence of integer day numbers (IST calendar ordinals).
        routine_cycle_days: Expected interval between occurrences (default: 30 days).
        tolerance: Allowed deviation in days around routine_cycle_days (default: 5 days).

    KEY LOGIC:
        - Requires at least 2 distinct active days to establish a recurring cycle.
        - Computes consecutive differences between sorted unique active days.
        - Returns True if and only if every consecutive gap satisfies:
          abs(delta - routine_cycle_days) <= tolerance.
    """
    unique_days = sorted(set(days))
    if len(unique_days) < 2:
        return False
    diffs = [unique_days[i + 1] - unique_days[i] for i in range(len(unique_days) - 1)]
    return all(abs(d - routine_cycle_days) <= tolerance for d in diffs)


def compute_routine_life_features(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> dict[str, Any]:
    """
    PURPOSE:
        Computes the M1 routine_life group features using closed-day counterparty summaries.

    INPUTS:
        client: Redis client instance.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        decision_time: Point-in-time scoring evaluation timestamp.
        arrival_cutoff: Ingestion watermark storage cutoff.
        settings: Optional AccountFeaturesSettings.

    KEY LOGIC:
        1. Determines history_days_available from closed active days.
        2. If history_days_available < routine_min_history_days (35d):
           all routine metrics return None ('not enough history').
        3. If history is sufficient:
           - salary_like_payer_count: inbound credit counterparties exhibiting monthly rhythm.
           - recurring_merchant_payee_count: outbound debit counterparties classified as 'merchant'
             with >= 2 distinct active days.
           - regular_transfer_payee_count: outbound debit counterparties classified as 'person'
             exhibiting monthly rhythm.
           - routine_life_present: True if any routine count > 0, else False.
    """
    cfg = settings if settings is not None else get_default_settings()
    history_days = history_days_available(
        client, run_id, bank_id, account_id, decision_time, arrival_cutoff, cfg
    )

    if history_days < cfg.routine_min_history_days:
        return {
            "salary_like_payer_count": None,
            "recurring_merchant_payee_count": None,
            "regular_transfer_payee_count": None,
            "routine_life_present": None,
            "history_days_available": history_days,
        }

    # Query closed credit counterparties for salary-like payers
    credit_cps = read_counterparty_days(
        client, run_id, bank_id, account_id, "credit", decision_time, arrival_cutoff, cfg
    )
    salary_count = 0
    for days in credit_cps.values():
        if monthly_rhythm(days, cfg.routine_cycle_days, cfg.routine_rhythm_tolerance_days):
            salary_count += 1

    # Query closed debit counterparties and verified payee types
    debit_cps = read_counterparty_days(
        client, run_id, bank_id, account_id, "debit", decision_time, arrival_cutoff, cfg
    )
    payee_types = read_counterparty_payee_types(client, run_id, bank_id, account_id, "debit", cfg)

    recurring_merchants = 0
    regular_transfers = 0
    for cp, days in debit_cps.items():
        ptype = payee_types.get(cp)
        if ptype == "merchant" and len(set(days)) >= 2:
            recurring_merchants += 1
        elif ptype == "person" and monthly_rhythm(
            days, cfg.routine_cycle_days, cfg.routine_rhythm_tolerance_days
        ):
            regular_transfers += 1

    routine_present = bool(salary_count > 0 or recurring_merchants > 0 or regular_transfers > 0)

    return {
        "salary_like_payer_count": salary_count,
        "recurring_merchant_payee_count": recurring_merchants,
        "regular_transfer_payee_count": regular_transfers,
        "routine_life_present": routine_present,
        "history_days_available": history_days,
    }


def compute_lifecycle_features(
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
        Computes the M1 lifecycle group features: account_age_days, account_type,
        dormancy_days_before_recent_activity, and volume_vs_own_history_ratio.

    INPUTS:
        client: Redis client.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        snapshot: Point-in-time AccountSnapshot holding raw 7-day window records.
        decision_time: Evaluation decision timestamp.
        arrival_cutoff: Ingestion watermark cutoff timestamp.
        settings: Optional AccountFeaturesSettings.
    """
    cfg = settings if settings is not None else get_default_settings()

    # 1. account_age_days: calendar days since opening date
    lifetime = read_lifetime(client, run_id, bank_id, account_id, decision_time, arrival_cutoff, cfg)
    if lifetime.opened_at is None:
        account_age_days = None
    else:
        opened_day = to_day_number(lifetime.opened_at)
        dec_day = to_day_number(decision_time)
        account_age_days = max(0, dec_day - opened_day)

    # 2. account_type: savings, current, or None
    account_type = read_account_type(client, run_id, bank_id, account_id, decision_time, arrival_cutoff, cfg)

    # 3. dormancy_days_before_recent_activity
    act_k = active_days_key(run_id, bank_id, account_id)
    raw_days = client.zrange(act_k, 0, -1)
    closed_active_days: list[int] = []
    if raw_days:
        for d_bytes in raw_days:
            d_val = int(d_bytes.decode("utf-8") if isinstance(d_bytes, bytes) else str(d_bytes))
            thresh = day_closed_threshold(d_val, cfg.allowed_lateness_seconds)
            if thresh <= decision_time and thresh <= arrival_cutoff:
                closed_active_days.append(d_val)
    closed_active_days.sort()

    dec_day = to_day_number(decision_time)
    recent_7d_start = dec_day - 7
    recent_active_days = [d for d in closed_active_days if recent_7d_start < d <= dec_day]

    if not recent_active_days:
        dormancy_days = None
    else:
        onset = min(recent_active_days)
        preceding_days = [d for d in closed_active_days if d < onset]
        if not preceding_days:
            dormancy_days = None
        else:
            prev_active = max(preceding_days)
            dormancy_days = max(0, onset - prev_active - 1)

    # 4. volume_vs_own_history_ratio: recent 7d volume to historical daily volume
    day_totals = read_day_totals(client, run_id, bank_id, account_id, decision_time, arrival_cutoff, cfg)
    if not day_totals:
        volume_ratio = None
    else:
        total_hist_paise = sum(
            t.credit_paise + t.debit_paise + t.withdrawal_paise for t in day_totals
        )
        avg_daily_paise = total_hist_paise / len(day_totals)

        # Recent 7d volume from raw window snapshot
        cutoff_7d = decision_time - timedelta(seconds=604800)
        recent_credits = sum(
            r.amount_paise
            for r in snapshot.side_records
            if r.side == "credit" and r.status == "success" and cutoff_7d < r.event_time <= decision_time
        )
        recent_debits = sum(
            r.amount_paise
            for r in snapshot.side_records
            if r.side == "debit" and r.status == "success" and cutoff_7d < r.event_time <= decision_time
        )
        recent_withdrawals = sum(
            w.amount_paise
            for w in snapshot.withdrawals
            if w.status == "success" and cutoff_7d < w.event_time <= decision_time
        )
        recent_7d_paise = recent_credits + recent_debits + recent_withdrawals

        if avg_daily_paise == 0:
            volume_ratio = 0.0 if recent_7d_paise == 0 else None
        else:
            volume_ratio = round(recent_7d_paise / avg_daily_paise, 4)

    return {
        "account_age_days": account_age_days,
        "account_type": account_type,
        "dormancy_days_before_recent_activity": dormancy_days,
        "volume_vs_own_history_ratio": volume_ratio,
    }
