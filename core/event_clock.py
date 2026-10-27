"""
core/event_clock.py

Injectable event clock and IST conversion helpers for ashFALL and national systems mocks.
Provides Clock protocols, concrete ManualClock, ReplayClock, WallClock, and exact
conversions between integer world minutes and timezone-aware IST datetimes under Ground Rule 30.
"""

import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

IST: ZoneInfo = ZoneInfo("Asia/Kolkata")


class Clock(Protocol):
    """
    Protocol for time providers supplying timezone-aware Indian Standard Time (IST).
    """

    def now_ist(self) -> datetime:
        """
        Return the current clock moment as a timezone-aware datetime in IST (+05:30).
        """
        ...


class ManualClock:
    """
    Manual clock whose time is explicitly set or advanced, designed for deterministic unit testing.
    """

    def __init__(self, current_time: datetime | None = None) -> None:
        """
        Initialize the manual clock with an optional initial datetime.

        Args:
            current_time: Initial timezone-aware datetime, or default 2026-09-01 00:00 IST if None.

        Raises:
            ValueError: If current_time is naive (lacks tzinfo).
        """
        if current_time is None:
            self._current_time: datetime = datetime(2026, 9, 1, 0, 0, tzinfo=IST)
        else:
            if current_time.tzinfo is None:
                raise ValueError("ManualClock initial time must be timezone-aware")
            self._current_time = current_time.astimezone(IST)

    def set_time(self, new_time: datetime) -> None:
        """
        Set the clock to a specific timezone-aware datetime.

        Args:
            new_time: New timezone-aware datetime.

        Raises:
            ValueError: If new_time is naive.
        """
        if new_time.tzinfo is None:
            raise ValueError("new_time must be timezone-aware")
        self._current_time = new_time.astimezone(IST)

    def advance(self, delta: timedelta) -> None:
        """
        Advance the clock by a given timedelta.

        Args:
            delta: Timedelta offset to add to the current clock time.
        """
        self._current_time += delta

    def now_ist(self) -> datetime:
        """
        Return the current manual datetime in IST.
        """
        return self._current_time


class ReplayClock:
    """
    Replay clock that scales simulated time relative to an injected wall-clock source.
    Simulated time = simulated_start + (wall elapsed * speed).
    """

    def __init__(
        self,
        simulated_start: datetime,
        speed: float = 1.0,
        wall_source: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialize a replay clock.

        Args:
            simulated_start: Simulated starting moment (must be timezone-aware).
            speed: Simulation speed multiplier (> 0).
            wall_source: Callable returning elapsed wall seconds as float (default time.monotonic).

        Raises:
            ValueError: If speed <= 0 or simulated_start is naive.
        """
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")
        if simulated_start.tzinfo is None:
            raise ValueError("simulated_start must be timezone-aware")

        self._simulated_start: datetime = simulated_start.astimezone(IST)
        self._speed: float = float(speed)
        self._wall_source: Callable[[], float] = wall_source if wall_source is not None else time.monotonic
        self._initial_wall: float = self._wall_source()

    def now_ist(self) -> datetime:
        """
        Compute simulated time based on elapsed seconds from the injected wall source.
        """
        current_wall = self._wall_source()
        elapsed_wall = current_wall - self._initial_wall
        simulated_seconds = elapsed_wall * self._speed
        return self._simulated_start + timedelta(seconds=simulated_seconds)


class WallClock:
    """
    Real-time system wall clock in IST.
    Documented strictly for TRANSPORT ONLY per Ground Rule 30 (request signatures, cache expiry).
    Simulation models and registry logic must never consult WallClock directly.
    """

    def now_ist(self) -> datetime:
        """
        Return the real host system time in IST (+05:30).
        """
        return datetime.now(IST)


def world_minute_to_ist(minute: int, world_start: date) -> datetime:
    """
    Convert an integer world simulation minute into a timezone-aware IST datetime.

    Args:
        minute: Non-negative integer minute offset from midnight of world_start.
        world_start: Calendar date when simulation started (e.g. date(2026, 9, 1)).

    Returns:
        Timezone-aware datetime in Asia/Kolkata (+05:30).

    Raises:
        ValueError: If minute is not an integer, is a boolean, or is negative.
    """
    if isinstance(minute, bool) or not isinstance(minute, int):
        raise ValueError(f"minute must be int, got {type(minute).__name__}")
    if minute < 0:
        raise ValueError(f"minute cannot be negative, got {minute}")

    start_dt = datetime(world_start.year, world_start.month, world_start.day, 0, 0, tzinfo=IST)
    return start_dt + timedelta(minutes=minute)


def ist_to_world_minute(moment: datetime, world_start: date) -> int:
    """
    Convert a timezone-aware datetime into elapsed integer minutes from midnight of world_start.

    Args:
        moment: Timezone-aware datetime.
        world_start: Calendar date when simulation started.

    Returns:
        Elapsed integer minutes from start of world_start in IST.

    Raises:
        ValueError: If moment is naive (tzinfo is None).
    """
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")

    moment_ist = moment.astimezone(IST)
    start_dt = datetime(world_start.year, world_start.month, world_start.day, 0, 0, tzinfo=IST)
    elapsed = moment_ist - start_dt
    return int(elapsed.total_seconds() // 60)
