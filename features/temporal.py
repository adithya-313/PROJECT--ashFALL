# File: features/temporal.py
# Purpose: Pure-datetime temporal features (hour, weekday, account age, time since last transaction) for scoring.
#
# Why time matters for fraud: fraudsters and legitimate customers keep different schedules. Card-testing and
# account-takeover attacks cluster at night when the real owner is asleep and less likely to notice alerts, and some
# attacks favour weekends when fraud teams are thinly staffed. New accounts are riskier than old ones, and a burst
# of activity right after a long silence is suspicious. These functions turn timestamps into numbers that capture
# those patterns. They use only the standard library, so they are fast and have no dependencies.
from datetime import datetime

SECONDS_PER_DAY = 86400.0

# Returned by compute_time_since_last_txn when there is no earlier transaction. It is a "magic value" that can
# never be a real elapsed time (real gaps are zero or positive), so callers can tell the two cases apart.
NO_PRIOR_TXN_SENTINEL = -1.0


def compute_hour_of_day(timestamp: datetime) -> int:
    """
    Return the hour of the day (0-23) at which the transaction happened.

    Why this function exists: fraud is not spread evenly across the clock. Many account-takeover and card-testing
    attacks happen in the small hours (for example 01:00-05:00) when the real owner is asleep. Giving the model
    the hour lets it learn that a large purchase at 03:00 is riskier than the same purchase at 15:00.

    Args:
        timestamp: when the transaction arrived, as a naive UTC datetime (see the DATETIME CONTRACT in
            features/__init__.py).

    Returns:
        An integer from 0 (midnight hour) to 23 (the 11 pm hour), in UTC.

    Edge cases:
        The hour is in UTC, not the customer's local time. Midnight UTC is 0 and 23:59 UTC is 23.
    """
    return timestamp.hour


def compute_day_of_week(timestamp: datetime) -> int:
    """
    Return the day of the week as a number, 0 for Monday through 6 for Sunday.

    Why this function exists: fraud rates differ by day. Weekends and public holidays often see more fraud because
    review teams are smaller and fraudsters know it takes longer for anyone to react. Giving the model the weekday
    lets it learn these weekly rhythms.

    Args:
        timestamp: when the transaction arrived, as a naive UTC datetime.

    Returns:
        An integer where 0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday and
        6 = Sunday. This is Python's datetime.weekday() convention (not the isoweekday() 1-7 convention).

    Edge cases:
        None. Every valid datetime maps to exactly one weekday.
    """
    return timestamp.weekday()


def compute_days_since_account_open(first_seen_at: datetime, now: datetime) -> float:
    """
    Return how many days ago the account was first seen, as a float.

    Why this function exists: account age is one of the strongest fraud signals. Brand-new accounts are the
    favourite tool of money mules and fake-identity fraud, while accounts that have behaved normally for years are
    much less likely to be fraudulent. The model uses this number to weigh how much history backs the account.

    Args:
        first_seen_at: when we first saw any transaction for this account (stored in Redis as
            acct:first_seen:{account_id}).
        now: the time of the current transaction (the same request-arrival time used for every other feature).

    Returns:
        Days as a float, so 1.5 means one and a half days. Fractions matter because "opened 2 hours ago" and
        "opened 20 hours ago" are both under a day but very different in risk.

    Edge cases:
        If first_seen_at is later than now, we return 0.0 instead of a negative number or an error. A future
        first-seen time can only come from clock skew between servers or a bad stored value; it is not something the
        customer did. Raising an error would make the scoring request fail over a harmless glitch, and a negative
        age would confuse the model, so we treat it as "brand new" (0.0 days), which is the safe interpretation.
    """
    # Work out the gap in seconds first. total_seconds() keeps the fractional part, which we want for a float result.
    elapsed_seconds = (now - first_seen_at).total_seconds()

    # A negative gap means first_seen_at is in the future (clock skew or bad data). Clamp to zero rather than
    # raising, so a clock glitch cannot break scoring, and so the model never sees a nonsensical negative age.
    if elapsed_seconds < 0:
        return 0.0

    # Convert seconds into days (86,400 seconds per day) because the model works with account age in days.
    return elapsed_seconds / SECONDS_PER_DAY


def compute_time_since_last_txn(last_txn_at: datetime | None, now: datetime) -> float:
    """
    Return the number of seconds since this account's previous transaction, or -1.0 if there was none.

    Why this function exists: the gap between transactions reveals behaviour. Many transactions seconds apart
    suggest a script or a stolen card being tested, while a long quiet gap followed by activity can signal a
    dormant account being taken over. A gap of exactly zero (two transactions in the same instant) is also notable.

    Args:
        last_txn_at: the time of the previous transaction (stored in Redis as acct:last_txn_time:{account_id}), or
            None if the account has never transacted before.
        now: the time of the current transaction.

    Returns:
        Elapsed seconds as a float (zero or positive). Returns the sentinel -1.0 when last_txn_at is None. The
        caller MUST check for -1.0 before using the value in any calculation, because -1.0 is a signal ("no prior
        transaction"), not a real duration.

    Edge cases:
        - last_txn_at is None: returns -1.0. We do not return 0.0 because 0.0 is a real, meaningful answer ("two
          transactions at the same instant", a strong bot signal) and would be confused with "no history".
          We do not return None because the model feature columns are numeric floats, and a plain float sentinel
          keeps the return type simple and consistent. A negative value can never be a real elapsed time, so it
          cannot be mistaken for one.
        - last_txn_at equals now: returns 0.0, a genuine "same instant" gap.
    """
    # No previous transaction means this is the account's very first one. Return the sentinel so the caller can
    # detect "first transaction" and handle it, instead of silently treating it as a real gap.
    if last_txn_at is None:
        return NO_PRIOR_TXN_SENTINEL

    # Normal case: subtract the two times and convert to seconds (keeping fractions of a second).
    return (now - last_txn_at).total_seconds()
