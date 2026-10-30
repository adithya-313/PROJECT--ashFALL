# File: features/m1_inflow_outflow.py
# Purpose: Level 15b M1 inflow and outflow feature computation functions.
# Issue: Level 15b M1 features (plan Task 1, G1, G2, R8).
"""
WHAT: Provides pure functions computing M1 inflow and outflow behavioral features
from point-in-time account snapshots, closed-day summaries, and cash withdrawals.

WHY: Fraud detection models require exact, reproducible velocity and structuring
signals over 1h, 24h, and 7d sliding windows. Computing features through pure functions
against already-filtered snapshot records guarantees Section 22 future-leak safety,
strict commercial bank tenant isolation, and deterministic replay.
"""

import statistics
from collections.abc import Sequence
from datetime import datetime, timedelta

from features.account_records import PaymentSideRecord, WithdrawalRecord


def filter_window(
    records: Sequence[PaymentSideRecord],
    decision_time: datetime,
    window_seconds: int,
) -> list[PaymentSideRecord]:
    """
    PURPOSE:
        Filters payment side records into the half-open window:
        (decision_time - window_seconds, decision_time].

    CRITICAL CONSTRAINTS:
        The lower boundary is strictly greater than (>) to match count_sides_in_window
        and Section 22 window semantics.
    """
    cutoff = decision_time - timedelta(seconds=window_seconds)
    return [r for r in records if cutoff < r.event_time <= decision_time]


def filter_withdrawals_window(
    withdrawals: Sequence[WithdrawalRecord],
    decision_time: datetime,
    window_seconds: int,
) -> list[WithdrawalRecord]:
    """
    PURPOSE:
        Filters withdrawal records into the half-open window:
        (decision_time - window_seconds, decision_time].

    CRITICAL CONSTRAINTS:
        The lower boundary is strictly greater than (>) to match count_sides_in_window.
    """
    cutoff = decision_time - timedelta(seconds=window_seconds)
    return [w for w in withdrawals if cutoff < w.event_time <= decision_time]


def get_counterparty(record: PaymentSideRecord) -> str:
    """
    PURPOSE:
        Resolves the opaque counterparty string for a payment side.
        Matches features/account_summaries.py counterparty encoding.
    """
    if record.counterparty_vpa:
        return f"{record.counterparty_bank_id}:{record.counterparty_vpa}"
    return record.counterparty_bank_id


# =============================================================================
# INFLOW FEATURES (Issue G1)
# =============================================================================


def compute_distinct_payers(
    side_records: Sequence[PaymentSideRecord],
    decision_time: datetime,
    window_seconds: int,
) -> int:
    """
    PURPOSE:
        Computes count of distinct sending counterparties on successful credit transactions
        within (decision_time - window_seconds, decision_time]. Issue: G1.

    INPUTS:
        side_records: Raw payment side records from the account snapshot.
        decision_time: Scoring evaluation timestamp.
        window_seconds: Window duration in seconds (3600 for 1h, 86400 for 24h, 604800 for 7d).

    KEY LOGIC:
        Only successful credit transactions are counted. Lower boundary is strictly greater than.

    RETURN:
        int: Count of distinct sending counterparties (0 if no credits).
    """
    in_win = filter_window(side_records, decision_time, window_seconds)
    credits = [r for r in in_win if r.side == "credit" and r.status == "success"]
    if not credits:
        return 0
    payers = {get_counterparty(r) for r in credits}
    return len(payers)


def compute_first_time_payer_share_24h(
    side_records: Sequence[PaymentSideRecord],
    closed_credit_counterparty_days: dict[str, list[int]],
    decision_time: datetime,
) -> float | None:
    """
    PURPOSE:
        Computes share of 24h credit payers never seen paying this account before
        the 24h window (closed counterparty days + earlier raw records). Issue: G1.

    INPUTS:
        side_records: Raw payment side records from the account snapshot.
        closed_credit_counterparty_days: Mapping of counterparty to closed calendar days
            from read_counterparty_days(..., direction='credit').
        decision_time: Scoring evaluation timestamp.

    KEY LOGIC:
        1. Identifies distinct payers across successful credits in (decision_time - 24h, decision_time].
        2. If no successful credits exist in 24h, returns None ('no data').
        3. A payer is considered 'seen before' if they appear in closed counterparty days OR
           in earlier successful raw credit records (event_time <= decision_time - 24h).
        4. Calculates: count(first_time_payers) / count(distinct_payers_24h).

    RETURN:
        float | None: Share from 0.0 to 1.0, or None if no credits in 24h.
    """
    win_24h = filter_window(side_records, decision_time, 86400)
    credits_24h = [r for r in win_24h if r.side == "credit" and r.status == "success"]
    if not credits_24h:
        return None

    distinct_24h_payers = {get_counterparty(r) for r in credits_24h}
    if not distinct_24h_payers:
        return None

    # Payers seen in earlier raw records (before the 24h window)
    cutoff_24h = decision_time - timedelta(seconds=86400)
    earlier_raw_payers = {
        get_counterparty(r)
        for r in side_records
        if r.side == "credit" and r.status == "success" and r.event_time <= cutoff_24h
    }

    # Payers seen in closed summary days
    closed_summary_payers = set(closed_credit_counterparty_days.keys())

    first_time_count = 0
    for payer in distinct_24h_payers:
        seen_before = (payer in earlier_raw_payers) or (payer in closed_summary_payers)
        if not seen_before:
            first_time_count += 1

    return float(first_time_count / len(distinct_24h_payers))


def compute_small_credit_count(
    side_records: Sequence[PaymentSideRecord],
    decision_time: datetime,
    window_seconds: int,
    small_credit_max_paise: int,
) -> int:
    """
    PURPOSE:
        Counts successful credits with amount <= small_credit_max_paise in window. Issue: G1.

    INPUTS:
        side_records: Raw payment side records from the account snapshot.
        decision_time: Scoring evaluation timestamp.
        window_seconds: Window duration in seconds.
        small_credit_max_paise: Upper limit in paise for small credits (from config).

    RETURN:
        int: Count of matching credits (0 if none).
    """
    in_win = filter_window(side_records, decision_time, window_seconds)
    return sum(
        1
        for r in in_win
        if r.side == "credit" and r.status == "success" and r.amount_paise <= small_credit_max_paise
    )


def compute_below_threshold_credit_count_7d(
    side_records: Sequence[PaymentSideRecord],
    decision_time: datetime,
    reporting_thresholds_paise: tuple[int, ...],
    below_threshold_band_fraction: float,
) -> int:
    """
    PURPOSE:
        Counts successful credits in 7d within below_threshold_band_fraction below any
        amount in reporting_thresholds_paise (structuring detection). Issue: G1.

    INPUTS:
        side_records: Raw payment side records from snapshot.
        decision_time: Scoring evaluation timestamp.
        reporting_thresholds_paise: Tuple of regulatory thresholds (e.g. ₹10k, ₹50k, ₹100k).
        below_threshold_band_fraction: Fraction (e.g. 0.05 for 5% band).

    KEY LOGIC:
        For threshold T, matching band is: T * (1 - band_fraction) <= amount_paise < T.
        Example: T=1,000,000 paise (₹10,000), fraction 0.05: band is [950,000, 1,000,000).

    RETURN:
        int: Count of matching credits (0 if none).
    """
    in_win = filter_window(side_records, decision_time, 604800)
    credits_7d = [r for r in in_win if r.side == "credit" and r.status == "success"]
    count = 0
    for r in credits_7d:
        for threshold in reporting_thresholds_paise:
            lower_bound = int(threshold * (1.0 - below_threshold_band_fraction))
            if lower_bound <= r.amount_paise < threshold:
                count += 1
                break  # Count each transaction at most once
    return count


# =============================================================================
# OUTFLOW FEATURES (Issues G1, G2, R8)
# =============================================================================


def compute_pass_through_ratio(
    side_records: Sequence[PaymentSideRecord],
    withdrawals: Sequence[WithdrawalRecord],
    decision_time: datetime,
    window_seconds: int,
) -> float | None:
    """
    PURPOSE:
        Computes (successful debits + successful withdrawals) amount / successful credits amount.
        Issues: G1, G2.

    INPUTS:
        side_records: Payment side records from snapshot.
        withdrawals: Cash withdrawal records from snapshot.
        decision_time: Scoring evaluation timestamp.
        window_seconds: 86400 (24h) or 604800 (7d).

    KEY LOGIC:
        If total successful credit amount in the window is 0 (or no successful credits exist),
        returns None ('no data when no credits').
        Outflow includes successful debits and successful cash withdrawals.

    RETURN:
        float | None: Ratio as float, or None if no credits.
    """
    win_sides = filter_window(side_records, decision_time, window_seconds)
    win_withdrawals = filter_withdrawals_window(withdrawals, decision_time, window_seconds)

    credit_amount = sum(r.amount_paise for r in win_sides if r.side == "credit" and r.status == "success")
    if credit_amount == 0:
        return None

    debit_amount = sum(r.amount_paise for r in win_sides if r.side == "debit" and r.status == "success")
    withdrawal_amount = sum(w.amount_paise for w in win_withdrawals if w.status == "success")

    total_outflow = debit_amount + withdrawal_amount
    return float(total_outflow / credit_amount)


def compute_median_credit_to_debit_seconds_7d(
    side_records: Sequence[PaymentSideRecord],
    withdrawals: Sequence[WithdrawalRecord],
    decision_time: datetime,
) -> float | None:
    """
    PURPOSE:
        For each successful debit or withdrawal in the last 7 days, measures seconds since
        the latest earlier successful credit; computes the median across all pairs.
        Issues: G1, G2.

    INPUTS:
        side_records: Payment side records from snapshot.
        withdrawals: Cash withdrawal records from snapshot.
        decision_time: Scoring evaluation timestamp.

    KEY LOGIC:
        1. Collects all successful credits in 7d window.
        2. Collects all successful debits and withdrawals in 7d window.
        3. For each outgoing event at t_out, finds the latest credit with event_time < t_out.
           If an outgoing event has no earlier credit, it does not form a pair.
        4. If no pairs exist, returns None ('no data when no pairs').
        5. For an even count of pairs, returns the mean of the two middle values (via statistics.median).

    RETURN:
        float | None: Median seconds as float, or None if no pairs.
    """
    win_sides = filter_window(side_records, decision_time, 604800)
    win_withdrawals = filter_withdrawals_window(withdrawals, decision_time, 604800)

    credits = [r for r in win_sides if r.side == "credit" and r.status == "success"]
    if not credits:
        return None

    outflow_times: list[datetime] = [
        r.event_time for r in win_sides if r.side == "debit" and r.status == "success"
    ]
    outflow_times.extend(
        w.event_time for w in win_withdrawals if w.status == "success"
    )

    if not outflow_times:
        return None

    delays: list[float] = []
    for t_out in outflow_times:
        earlier_credits = [c for c in credits if c.event_time < t_out]
        if earlier_credits:
            latest_credit = max(earlier_credits, key=lambda c: c.event_time)
            delay = (t_out - latest_credit.event_time).total_seconds()
            delays.append(delay)

    if not delays:
        return None

    return float(statistics.median(delays))


def compute_top_receiver_share_7d(
    side_records: Sequence[PaymentSideRecord],
    decision_time: datetime,
) -> float | None:
    """
    PURPOSE:
        Share of successful debit amount going to the largest single receiving counterparty
        in the last 7 days. Issue: G1.

    INPUTS:
        side_records: Payment side records from snapshot.
        decision_time: Scoring evaluation timestamp.

    KEY LOGIC:
        If no successful debits exist, returns None ('no data when no debits').
        Aggregates debited amount by counterparty identifier and computes the maximum share.

    RETURN:
        float | None: Share from 0.0 to 1.0, or None if no debits.
    """
    win_sides = filter_window(side_records, decision_time, 604800)
    debits = [r for r in win_sides if r.side == "debit" and r.status == "success"]
    if not debits:
        return None

    total_debit_amount = sum(r.amount_paise for r in debits)
    if total_debit_amount == 0:
        return None

    receiver_amounts: dict[str, int] = {}
    for r in debits:
        receiver = get_counterparty(r)
        receiver_amounts[receiver] = receiver_amounts.get(receiver, 0) + r.amount_paise

    max_receiver_amount = max(receiver_amounts.values())
    return float(max_receiver_amount / total_debit_amount)


def compute_other_bank_outflow_share_7d(
    side_records: Sequence[PaymentSideRecord],
    bank_id: str,
    decision_time: datetime,
) -> float | None:
    """
    PURPOSE:
        Share of successful debit amount whose counterparty_bank_id differs from this bank
        in the last 7 days. Issues: G1, R8.

    INPUTS:
        side_records: Payment side records from snapshot.
        bank_id: Tenant commercial bank identifier. The bank's own id comes from the snapshot's
            records' bank_id; never read another bank's keys (the reader is already scoped by bank_id).
        decision_time: Scoring evaluation timestamp.

    KEY LOGIC:
        If no successful debits exist, returns None ('no data when no debits').
        Calculates ratio of debit amounts where counterparty_bank_id != bank_id to total debit amount.

    RETURN:
        float | None: Share from 0.0 to 1.0, or None if no debits.
    """
    win_sides = filter_window(side_records, decision_time, 604800)
    debits = [r for r in win_sides if r.side == "debit" and r.status == "success"]
    if not debits:
        return None

    total_debit_amount = sum(r.amount_paise for r in debits)
    if total_debit_amount == 0:
        return None

    other_bank_amount = sum(
        r.amount_paise for r in debits if r.counterparty_bank_id != bank_id
    )
    return float(other_bank_amount / total_debit_amount)


def compute_cash_out_channels_7d(
    side_records: Sequence[PaymentSideRecord],
    withdrawals: Sequence[WithdrawalRecord],
    decision_time: datetime,
) -> dict[str, int]:
    """
    PURPOSE:
        Computes every cash-out channel the bank can see over 7 days (G2):
        - outflow_to_person_paise_7d (successful debits with payee_type == 'person')
        - outflow_to_merchant_paise_7d (successful debits with payee_type == 'merchant')
        - cash_atm_paise_7d (successful withdrawals with channel == 'atm')
        - cash_branch_paise_7d (successful withdrawals with channel == 'branch')

    INPUTS:
        side_records: Payment side records from snapshot.
        withdrawals: Cash withdrawal records from snapshot.
        decision_time: Scoring evaluation timestamp.

    RETURN:
        dict[str, int]: Amounts in paise for each channel (0 if none).
    """
    win_sides = filter_window(side_records, decision_time, 604800)
    win_withdrawals = filter_withdrawals_window(withdrawals, decision_time, 604800)

    person_paise = sum(
        r.amount_paise
        for r in win_sides
        if r.side == "debit" and r.status == "success" and r.payee_type == "person"
    )
    merchant_paise = sum(
        r.amount_paise
        for r in win_sides
        if r.side == "debit" and r.status == "success" and r.payee_type == "merchant"
    )
    atm_paise = sum(
        w.amount_paise
        for w in win_withdrawals
        if w.status == "success" and w.channel == "atm"
    )
    branch_paise = sum(
        w.amount_paise
        for w in win_withdrawals
        if w.status == "success" and w.channel == "branch"
    )

    return {
        "outflow_to_person_paise_7d": person_paise,
        "outflow_to_merchant_paise_7d": merchant_paise,
        "cash_atm_paise_7d": atm_paise,
        "cash_branch_paise_7d": branch_paise,
    }
