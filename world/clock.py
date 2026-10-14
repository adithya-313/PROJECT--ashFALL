"""
WHAT: Simulated clock converting minute counters into calendar dates, weekdays, and hours.
WHY: Translates continuous simulation time into realistic astronomical calendar events,
     enabling start-of-month and weekend rhythm modeling without fixed-month approximations.
ISSUE IDs: R2 (simulated calendar clock).
TERM: minute counter: The non-negative integer count of elapsed minutes since the simulation start.
TERM: day index: The zero-based count of completed 24-hour days elapsed since the simulation start.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

# A simulated calendar label, a structural choice, not a research number;
# the simulated festival is NOT pinned to the real 2026 Diwali date.
WORLD_START_DATE: date = date(2026, 9, 1)


def _validate_minute(minute: int) -> int:
    """
    WHAT: Validates that a minute counter is a non-negative integer.
    WHY: Protects calendar calculations from boolean, fractional, or negative values.

    Args:
        minute: Integer representing elapsed minutes since the start of the world simulation.

    Returns:
        The validated non-negative integer minute counter.

    Raises:
        TypeError: If minute is not an integer or is a boolean value.
        ValueError: If minute is negative.

    Example:
        >>> from world.clock import _validate_minute
        >>> _validate_minute(120)
        120
    """
    # Reject non-integer types or booleans that masquerade as integers in Python.
    if not isinstance(minute, int) or isinstance(minute, bool):
        raise TypeError(f"Minute counter must be a non-negative integer, got {type(minute).__name__}")
    # Ensure time does not run backwards before the start of the simulation.
    if minute < 0:
        raise ValueError(f"Minute counter must be a non-negative integer, got {minute}")
    return minute


@dataclass(frozen=True)
class SimClock:
    """
    WHAT: Calendar clock mapping simulation minutes to calendar datetimes and components.
    WHY: Evaluates temporal properties (hour of day, day of month, weekends) for rhythm scaling.
    """

    start_date: date

    def to_datetime(self, minute: int) -> datetime:
        """
        WHAT: Converts a minute counter into a concrete Python datetime.
        WHY: Provides full calendar date and time representation based on start_date.

        Args:
            minute: Non-negative integer minute offset from the world simulation start date.

        Returns:
            Concrete datetime object combining the simulation start date with elapsed days and minutes.

        Raises:
            TypeError: If minute is not an integer or is a boolean value.
            ValueError: If minute is negative.

        Example:
            >>> from datetime import date
            >>> from world.clock import SimClock
            >>> clock = SimClock(start_date=date(2026, 9, 1))
            >>> clock.to_datetime(60).isoformat()
            '2026-09-01T01:00:00'
        """
        # Validate the input minute counter to ensure it is a non-negative integer.
        valid_min = _validate_minute(minute)
        # Break down total minutes into elapsed full days and remaining minutes.
        days = valid_min // 1440
        rem_min = valid_min % 1440
        # Anchor the calendar at midnight of the simulation start date and offset forward.
        base_dt = datetime(self.start_date.year, self.start_date.month, self.start_date.day)
        return base_dt + timedelta(days=days, minutes=rem_min)

    def day_index(self, minute: int) -> int:
        """
        WHAT: Computes the 0-indexed day number in the world from a minute counter.
        WHY: Used for day-level scheduling, curves, and festival position tracking.

        Args:
            minute: Non-negative integer minute offset from the world simulation start date.

        Returns:
            0-indexed integer day count representing completed 24-hour days since simulation start.

        Raises:
            TypeError: If minute is not an integer or is a boolean value.
            ValueError: If minute is negative.

        Example:
            >>> from datetime import date
            >>> from world.clock import SimClock
            >>> clock = SimClock(start_date=date(2026, 9, 1))
            >>> clock.day_index(2880)
            2
        """
        valid_min = _validate_minute(minute)
        return valid_min // 1440

    def day_of_month(self, minute: int) -> int:
        """
        WHAT: Returns the calendar day of the month (1..31) for the given minute.
        WHY: Enables start-of-month payroll and rent billing rhythms on real calendar dates.

        Args:
            minute: Non-negative integer minute counter.

        Returns:
            Calendar day of the month as an integer between 1 and 31.

        Example:
            >>> from datetime import date
            >>> from world.clock import SimClock
            >>> clock = SimClock(start_date=date(2026, 9, 1))
            >>> clock.day_of_month(1440)
            2
        """
        return self.to_datetime(minute).day

    def weekday(self, minute: int) -> int:
        """
        WHAT: Returns the day of the week as an integer (Monday=0 to Sunday=6).
        WHY: Determines whether a given minute falls on a business day or weekend.

        Args:
            minute: Non-negative integer minute counter.

        Returns:
            Integer weekday index ranging from 0 (Monday) to 6 (Sunday).

        Example:
            >>> from datetime import date
            >>> from world.clock import SimClock
            >>> clock = SimClock(start_date=date(2026, 9, 1))
            >>> clock.weekday(0)
            1
        """
        return self.to_datetime(minute).weekday()

    def is_weekend(self, minute: int) -> bool:
        """
        WHAT: Returns True if the minute falls on a Saturday or Sunday, else False.
        WHY: Gates weekend activity multipliers against weekday baselines.

        Args:
            minute: Non-negative integer minute counter.

        Returns:
            True if the minute falls on a Saturday (5) or Sunday (6), False otherwise.

        Example:
            >>> from datetime import date
            >>> from world.clock import SimClock
            >>> clock = SimClock(start_date=date(2026, 9, 1))
            >>> clock.is_weekend(0)
            False
        """
        return self.weekday(minute) in (5, 6)

    def hour_of_day(self, minute: int) -> int:
        """
        WHAT: Returns the current 24-hour clock hour (0..23) for the given minute.
        WHY: Evaluates diurnal Gaussian activity curves by individual persona peak hours.

        Args:
            minute: Non-negative integer minute counter.

        Returns:
            Current clock hour as an integer between 0 and 23.

        Raises:
            TypeError: If minute is not an integer or is a boolean value.
            ValueError: If minute is negative.

        Example:
            >>> from datetime import date
            >>> from world.clock import SimClock
            >>> clock = SimClock(start_date=date(2026, 9, 1))
            >>> clock.hour_of_day(150)
            2
        """
        valid_min = _validate_minute(minute)
        return (valid_min % 1440) // 60
