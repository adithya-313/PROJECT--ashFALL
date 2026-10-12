"""
WHAT: Unit tests for the simulated clock module (SimClock).
WHY: Proves accurate conversion from minute counters to calendar datetimes, weekdays, weekends,
     and month boundaries, while refusing invalid non-integer or negative inputs.
ISSUE IDs: R2 (calendar and simulated clock).
TERM: minute counter: An integer representing cumulative minutes elapsed since the world epoch at 00:00.
"""

from datetime import date, datetime

import pytest

from world.clock import WORLD_START_DATE, SimClock


def test_c1_minute_conversions_and_hour() -> None:
    """
    WHAT: Tests minute 0 starts on 2026-09-01 00:00 Tuesday, and validates minute rollover and hours.
    WHY: Verifies the fundamental mapping between simulation minutes, calendar time, and hours.
    """
    clock = SimClock(WORLD_START_DATE)
    dt0 = clock.to_datetime(0)
    assert dt0 == datetime(2026, 9, 1, 0, 0)
    assert clock.weekday(0) == 1  # Tuesday

    # Rollover to next day at 1440 minutes
    dt1440 = clock.to_datetime(1440)
    assert dt1440 == datetime(2026, 9, 2, 0, 0)
    assert clock.day_index(1440) == 1

    # End of first day at minute 1439 (23:59)
    assert clock.hour_of_day(1439) == 23
    # 90 minutes past midnight is 01:30 AM (hour 1)
    assert clock.hour_of_day(90) == 1


def test_c2_weekend_detection() -> None:
    """
    WHAT: Tests detection of weekends (Saturday, Sunday) versus weekdays (Friday, Monday).
    WHY: Ensures correct scheduling of weekend transaction uplift and behavior.
    """
    clock = SimClock(date(2026, 9, 1))  # Tuesday is day 0
    # Day 3: Friday 2026-09-04 -> weekday, not weekend
    assert clock.is_weekend(3 * 1440) is False
    # Day 4: Saturday 2026-09-05 -> weekend
    assert clock.is_weekend(4 * 1440) is True
    # Day 5: Sunday 2026-09-06 -> weekend
    assert clock.is_weekend(5 * 1440) is True
    # Day 6: Monday 2026-09-07 -> weekday, not weekend
    assert clock.is_weekend(6 * 1440) is False


def test_c3_real_calendar_month_lengths() -> None:
    """
    WHAT: Tests that September has 30 days and rolls smoothly into October 1 on day index 30.
    WHY: Proves the simulation uses authentic astronomical calendar months rather than fixed 28-day approximations.
    """
    clock = SimClock(WORLD_START_DATE)
    # Day index 29 is the 30th day of September (day_of_month 30)
    min_day_29 = 29 * 1440
    assert clock.day_index(min_day_29) == 29
    assert clock.day_of_month(min_day_29) == 30
    assert clock.to_datetime(min_day_29).month == 9

    # Day index 30 is the 1st day of October (day_of_month 1)
    min_day_30 = 30 * 1440
    assert clock.day_index(min_day_30) == 30
    assert clock.day_of_month(min_day_30) == 1
    assert clock.to_datetime(min_day_30).month == 10


def test_c4_refuse_negative_and_non_integer_minutes() -> None:
    """
    WHAT: Tests that negative or floating-point minutes are refused with clear error messages.
    WHY: Protects calendar calculations from corrupted, reversed, or fractional simulation clocks.
    """
    clock = SimClock(WORLD_START_DATE)
    with pytest.raises(ValueError, match="Minute counter must be a non-negative integer"):
        clock.to_datetime(-1)

    with pytest.raises(TypeError, match="Minute counter must be a non-negative integer"):
        clock.to_datetime(12.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Minute counter must be a non-negative integer"):
        clock.day_index(-10)

    with pytest.raises(TypeError, match="Minute counter must be a non-negative integer"):
        clock.hour_of_day("100")  # type: ignore[arg-type]
