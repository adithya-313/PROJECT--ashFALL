# File: features/m1_balance.py
# Purpose: Level 15b M1 balance-pattern feature computation functions.
# Issue: Level 15b M1 features (plan Task 1, G2).
"""
WHAT: Provides pure calculation functions for M1 balance-pattern features:
current_balance_paise, min_balance_paise_7d, and near_zero_cycles_7d.

WHY: Rapid balance depletion following credit arrival is a hallmark indicator of money mule
and cash-out accounts (G2). Computing balance patterns strictly within the point-in-time
snapshot window enforces future-leak safety and avoids undefined behaviors on missing data.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

from features.account_records import PaymentSideRecord, WithdrawalRecord


def compute_current_balance_paise(
    side_records: Sequence[PaymentSideRecord],
    withdrawals: Sequence[WithdrawalRecord],
    decision_time: datetime,
) -> int | None:
    """
    PURPOSE:
        Returns ledger balance in paise immediately following the latest record
        that carries a balance. Issue: G2.

    INPUTS:
        side_records: Payment side records from snapshot.
        withdrawals: Cash withdrawal records from snapshot.
        decision_time: Scoring evaluation timestamp.

    KEY LOGIC:
        1. Considers all payment side and withdrawal records where event_time <= decision_time
           and balance_after_paise is not None.
        2. If no record carries a balance, returns None ('no data when no record carries a balance').
        3. Identifies the record with the latest event_time (tie-broken by arrival_time).

    RETURN:
        int | None: Balance in integer paise, or None if no balance is available.
    """
    records_with_balance: list[tuple[datetime, datetime, int]] = []

    for r in side_records:
        if r.event_time <= decision_time and r.balance_after_paise is not None:
            records_with_balance.append((r.event_time, r.arrival_time, r.balance_after_paise))

    for w in withdrawals:
        if w.event_time <= decision_time and w.balance_after_paise is not None:
            records_with_balance.append((w.event_time, w.arrival_time, w.balance_after_paise))

    if not records_with_balance:
        return None

    # Latest record by event_time, then arrival_time
    latest = max(records_with_balance, key=lambda item: (item[0], item[1]))
    return latest[2]


def compute_min_balance_paise_7d(
    side_records: Sequence[PaymentSideRecord],
    withdrawals: Sequence[WithdrawalRecord],
    decision_time: datetime,
) -> int | None:
    """
    PURPOSE:
        Returns the lowest ledger balance seen across all records carrying a balance
        in the 7-day sliding window. Issue: G2.

    INPUTS:
        side_records: Payment side records from snapshot.
        withdrawals: Cash withdrawal records from snapshot.
        decision_time: Scoring evaluation timestamp.

    KEY LOGIC:
        1. Filters records into the half-open window: (decision_time - 7d, decision_time].
        2. Collects non-None balance_after_paise values from successful transactions and withdrawals.
        3. If no balances exist in 7d, returns None ('no data when none').

    RETURN:
        int | None: Lowest balance in paise, or None if no balances observed.
    """
    cutoff = decision_time - timedelta(seconds=604800)
    balances_7d: list[int] = []

    for r in side_records:
        if cutoff < r.event_time <= decision_time and r.status == "success" and r.balance_after_paise is not None:
            balances_7d.append(r.balance_after_paise)

    for w in withdrawals:
        if cutoff < w.event_time <= decision_time and w.status == "success" and w.balance_after_paise is not None:
            balances_7d.append(w.balance_after_paise)

    if not balances_7d:
        return None

    return min(balances_7d)


def compute_near_zero_cycles_7d(
    side_records: Sequence[PaymentSideRecord],
    withdrawals: Sequence[WithdrawalRecord],
    decision_time: datetime,
    near_zero_balance_max_paise: int,
) -> int | None:
    """
    PURPOSE:
        Counts how many times a successful credit was followed, before the next credit,
        by an account balance <= near_zero_balance_max_paise in 7 days. Issue: G2.

    INPUTS:
        side_records: Payment side records from snapshot.
        withdrawals: Cash withdrawal records from snapshot.
        decision_time: Scoring evaluation timestamp.
        near_zero_balance_max_paise: Threshold in paise (₹100 = 10,000 paise).

    KEY LOGIC:
        1. Filters records into the 7-day window: (decision_time - 7d, decision_time].
        2. Combines payment sides and withdrawals and sorts them chronologically by (event_time, arrival_time).
        3. If no records in 7d carry a balance, returns None ('no data when no balances').
        4. Iterates through events in sequence:
           - On a successful credit, opens a credit cycle.
           - On any subsequent event before the next credit, if its balance <= near_zero_balance_max_paise,
             increments cycle count by 1 and marks the current cycle as triggered (at most once per credit).
           - The next successful credit resets the cycle state.

    RETURN:
        int | None: Number of completed near-zero cycles (>= 0), or None if no balances exist in 7d.
    """
    cutoff = decision_time - timedelta(seconds=604800)

    # Collect all successful events in 7d window
    # Structure: (event_time, arrival_time, is_credit, balance_after_paise)
    events: list[tuple[datetime, datetime, bool, int | None]] = []
    has_any_balance = False

    for r in side_records:
        if cutoff < r.event_time <= decision_time and r.status == "success":
            if r.balance_after_paise is not None:
                has_any_balance = True
            events.append((r.event_time, r.arrival_time, r.side == "credit", r.balance_after_paise))

    for w in withdrawals:
        if cutoff < w.event_time <= decision_time and w.status == "success":
            if w.balance_after_paise is not None:
                has_any_balance = True
            events.append((w.event_time, w.arrival_time, False, w.balance_after_paise))

    if not has_any_balance:
        return None

    # Sort strictly chronologically
    events.sort(key=lambda e: (e[0], e[1]))

    cycles = 0
    in_credit_cycle = False
    cycle_triggered = False

    for _, _, is_credit, bal in events:
        if is_credit:
            # A new credit begins a new potential near-zero cycle
            in_credit_cycle = True
            cycle_triggered = False
        else:
            # Non-credit event (debit or withdrawal) following a credit
            if in_credit_cycle and not cycle_triggered:
                if bal is not None and bal <= near_zero_balance_max_paise:
                    cycles += 1
                    cycle_triggered = True

    return cycles
