"""
tests/unit/test_event_clock.py

Unit tests for core/event_clock.py.
Verifies ManualClock, ReplayClock scaling and validations, WallClock transport contract,
and exact conversions between world simulation minutes and timezone-aware IST datetimes under Ground Rule 30.
"""

from datetime import date, datetime, timedelta
from typing import Any, cast

import pytest

from core.event_clock import (
    IST,
    ManualClock,
    ReplayClock,
    WallClock,
    ist_to_world_minute,
    world_minute_to_ist,
)


class FakeWallSource:
    """Deterministic fake wall-clock source providing controllable monotonic seconds."""

    def __init__(self, start: float = 1000.0) -> None:
        self.current: float = start

    def __call__(self) -> float:
        return self.current

    def tick(self, seconds: float) -> None:
        self.current += seconds


def test_replay_clock_at_speed_1_and_speed_100() -> None:
    """
    Replay at speed 1 and speed 100 driven by a fake wall_source gives the expected simulated times.
    """
    sim_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)

    # Test speed 1.0
    wall_source_1 = FakeWallSource(start=100.0)
    clock_1 = ReplayClock(simulated_start=sim_start, speed=1.0, wall_source=wall_source_1)
    assert clock_1.now_ist() == sim_start

    wall_source_1.tick(60.0)  # 60 wall seconds = 60 simulated seconds
    assert clock_1.now_ist() == sim_start + timedelta(seconds=60)

    # Test speed 100.0
    wall_source_100 = FakeWallSource(start=500.0)
    clock_100 = ReplayClock(simulated_start=sim_start, speed=100.0, wall_source=wall_source_100)
    assert clock_100.now_ist() == sim_start

    wall_source_100.tick(12.0)  # 12 wall seconds * 100 = 1200 simulated seconds (20 minutes)
    assert clock_100.now_ist() == sim_start + timedelta(minutes=20)


def test_replay_clock_speed_zero_and_negative_are_rejected() -> None:
    """
    Speed 0 and negative speed are rejected with ValueError.
    """
    sim_start = datetime(2026, 9, 1, 0, 0, tzinfo=IST)
    with pytest.raises(ValueError, match="speed must be > 0"):
        ReplayClock(simulated_start=sim_start, speed=0.0)

    with pytest.raises(ValueError, match="speed must be > 0"):
        ReplayClock(simulated_start=sim_start, speed=-1.5)


def test_naive_datetime_is_rejected_in_replay_and_manual_clock() -> None:
    """
    A naive datetime is rejected across clock constructors and setters.
    """
    naive_dt = datetime(2026, 9, 1, 12, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        ReplayClock(simulated_start=naive_dt)

    with pytest.raises(ValueError, match="timezone-aware"):
        ManualClock(current_time=naive_dt)

    manual = ManualClock()
    with pytest.raises(ValueError, match="timezone-aware"):
        manual.set_time(naive_dt)


def test_naive_datetime_rejected_in_ist_to_world_minute() -> None:
    """
    ist_to_world_minute rejects naive datetimes with ValueError.
    """
    naive_dt = datetime(2026, 9, 1, 10, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        ist_to_world_minute(naive_dt, date(2026, 9, 1))


def test_world_minute_to_ist_at_zero() -> None:
    """
    world_minute_to_ist(0, date(2026,9,1)) == 2026-09-01T00:00+05:30.
    """
    expected = datetime(2026, 9, 1, 0, 0, tzinfo=IST)
    actual = world_minute_to_ist(0, date(2026, 9, 1))
    assert actual == expected
    assert actual.isoformat() == "2026-09-01T00:00:00+05:30"


def test_round_trip_conversion_holds() -> None:
    """
    A round-trip conversion between minute and datetime holds exactly.
    """
    start_date = date(2026, 9, 1)
    for minute in [0, 1, 45, 1440, 10000, 87840]:
        moment = world_minute_to_ist(minute, start_date)
        recovered_minute = ist_to_world_minute(moment, start_date)
        assert recovered_minute == minute


def test_world_minute_validations() -> None:
    """
    world_minute_to_ist rejects negative minutes, bools, and non-ints.
    """
    start_date = date(2026, 9, 1)
    with pytest.raises(ValueError, match="cannot be negative"):
        world_minute_to_ist(-1, start_date)

    with pytest.raises(ValueError, match="must be int"):
        world_minute_to_ist(True, start_date)  # bool

    with pytest.raises(ValueError, match="must be int"):
        world_minute_to_ist(False, start_date)  # bool

    with pytest.raises(ValueError, match="must be int"):
        world_minute_to_ist(cast(Any, "100"), start_date)  # str


def test_manual_clock_operations() -> None:
    """
    Verifies ManualClock set_time, advance, and now_ist functionality.
    """
    clock = ManualClock()
    assert clock.now_ist() == datetime(2026, 9, 1, 0, 0, tzinfo=IST)

    new_time = datetime(2026, 9, 5, 14, 30, tzinfo=IST)
    clock.set_time(new_time)
    assert clock.now_ist() == new_time

    clock.advance(timedelta(hours=2, minutes=15))
    assert clock.now_ist() == datetime(2026, 9, 5, 16, 45, tzinfo=IST)


def test_wall_clock_returns_aware_ist_datetime() -> None:
    """
    Verifies WallClock returns a timezone-aware datetime in IST.
    """
    clock = WallClock()
    now = clock.now_ist()
    assert now.tzinfo is not None
    assert now.tzname() in ("IST", "+0530")
