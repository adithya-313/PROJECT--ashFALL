# File: features/temporal.py
# Purpose: Pure-datetime temporal features (hour, weekday, account age, time since last transaction) for scoring.
from datetime import datetime

SECONDS_PER_DAY = 86400.0
NO_PRIOR_TXN_SENTINEL = -1.0


def compute_hour_of_day(timestamp: datetime) -> int:
    """Return the hour of the day (0-23) for the timestamp."""
    return timestamp.hour


def compute_day_of_week(timestamp: datetime) -> int:
    """Return the weekday as 0 (Monday) through 6 (Sunday)."""
    return timestamp.weekday()


def compute_days_since_account_open(first_seen_at: datetime, now: datetime) -> float:
    """Return account age in days as a float, never negative."""
    elapsed_seconds = (now - first_seen_at).total_seconds()
    # A first-seen time in the future is clock skew or bad data; clamp to zero instead of going negative.
    if elapsed_seconds < 0:
        return 0.0
    return elapsed_seconds / SECONDS_PER_DAY


def compute_time_since_last_txn(last_txn_at: datetime | None, now: datetime) -> float:
    """Return seconds since the previous transaction, or -1.0 if there was none (caller must check)."""
    # None means this is the account's first transaction; return the sentinel so the caller can detect it.
    if last_txn_at is None:
        return NO_PRIOR_TXN_SENTINEL
    return (now - last_txn_at).total_seconds()
