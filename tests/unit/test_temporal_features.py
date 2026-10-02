# File: tests/unit/test_temporal_features.py
# Purpose: Unit tests for the temporal feature functions in features/temporal.py.
from datetime import datetime, timedelta

from features.temporal import (
    compute_day_of_week,
    compute_days_since_account_open,
    compute_hour_of_day,
    compute_time_since_last_txn,
)

NOW = datetime(2024, 1, 15, 12, 0, 0)  # a Monday


def test_hour_of_day_midnight():
    assert compute_hour_of_day(datetime(2024, 1, 15, 0, 0, 0)) == 0


def test_hour_of_day_noon():
    assert compute_hour_of_day(datetime(2024, 1, 15, 12, 0, 0)) == 12


def test_hour_of_day_11pm():
    assert compute_hour_of_day(datetime(2024, 1, 15, 23, 30, 0)) == 23


def test_day_of_week_monday():
    assert compute_day_of_week(datetime(2024, 1, 15)) == 0


def test_day_of_week_sunday():
    assert compute_day_of_week(datetime(2024, 1, 21)) == 6


def test_day_of_week_wednesday():
    assert compute_day_of_week(datetime(2024, 1, 17)) == 2  # Wednesday: Monday=0, so 2


def test_days_since_account_open_today():
    assert compute_days_since_account_open(NOW, NOW) == 0.0


def test_days_since_account_open_seven_days():
    assert compute_days_since_account_open(NOW - timedelta(days=7), NOW) == 7.0


def test_days_since_account_open_future_returns_zero():
    assert compute_days_since_account_open(NOW + timedelta(days=2), NOW) == 0.0


def test_time_since_last_txn_one_hour():
    assert compute_time_since_last_txn(NOW - timedelta(seconds=3600), NOW) == 3600.0


def test_time_since_last_txn_none_returns_sentinel():
    assert compute_time_since_last_txn(None, NOW) == -1.0


def test_time_since_last_txn_zero():
    assert compute_time_since_last_txn(NOW, NOW) == 0.0
