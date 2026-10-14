"""
WHAT: Festival calendar generation and piecewise linear festival uplift multipliers.
WHY: Generates stochastic festival schedules (Diwali and regional secondary festivals)
     to model seasonal spending surges with linear rise and fade dynamics.
ISSUE IDs: R2 (festival calendar and multipliers).
TERM: festival calendar: The collection of active cultural festivals and their parameters for a simulated world.
TERM: uplift fraction: The proportional increase above baseline transaction activity at the peak of a festival.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.background_config import draw_whole_number
from world.config_reader import draw_in_range, get_parameter_entry


@dataclass(frozen=True)
class Festival:
    """
    WHAT: Represents a distinct festival event with peak timing, ramp durations, and peak uplift.
    WHY: Parameterizes the temporal shape of holiday transaction volume surges.
    """

    name: str
    peak_day: int
    rise_days: int
    fade_days: int
    uplift_fraction: float


def _validate_world_days(n_days: int, max_bound: int) -> None:
    """
    WHAT: Validates that n_days is an integer strictly greater than the maximum festival peak bound.
    WHY: Guarantees that the world duration provides enough room for the festival peak window.

    Args:
        n_days: Total duration of the world simulation in full 24-hour days.
        max_bound: Configured upper bound of festival peak day range.

    Returns:
        None.

    Raises:
        ValueError: If n_days is not an integer, is boolean, or is not strictly greater than max_bound.

    Example:
        >>> from world.festivals import _validate_world_days
        >>> _validate_world_days(60, 45)
    """
    if not isinstance(n_days, int) or isinstance(n_days, bool) or n_days <= max_bound:
        raise ValueError(
            f"n_days ({n_days}) must be an integer strictly greater than the upper bound "
            f"of festival_peak_day_in_world ({max_bound})"
        )


def _draw_secondary_festivals(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    n_days: int,
    range_name: str,
) -> list[Festival]:
    """
    WHAT: Draws the collection of secondary regional festivals.
    WHY: Models background cultural diversity across the simulated year outside Diwali.

    Args:
        ranges: Configuration dictionary containing background realism parameter specifications.
        rng: Pseudo-random number generator for reproducible sampling.
        n_days: Total duration of the simulation in full days.
        range_name: Target dataset split name containing parameter intervals (e.g. "training").

    Returns:
        List of secondary Festival instances sampled across the world calendar.

    Example:
        >>> import numpy as np
        >>> from world.background_config import read_background_realism
        >>> from world.festivals import _draw_secondary_festivals
        >>> rng = np.random.default_rng(seed=42)
        >>> cfg = read_background_realism()
        >>> sec = _draw_secondary_festivals(cfg, rng, 61, "training")
        >>> len(sec)
        1
    """
    # Look up parameter range configurations for secondary festival counts and timing.
    other_cnt_entry = get_parameter_entry(ranges, "other_festival_count")
    other_uplift_entry = get_parameter_entry(ranges, "other_festival_uplift_fraction")
    other_rf_entry = get_parameter_entry(ranges, "other_festival_rise_fade_days")

    # Sample how many secondary festivals occur in this world run.
    count = draw_whole_number(other_cnt_entry, rng, range_name=range_name)
    secondaries: list[Festival] = []
    # Generate each secondary festival with independent timing across the calendar.
    for i in range(1, count + 1):
        # A structural choice: secondary festival peak day is placed uniformly across the world's days.
        peak_day = int(rng.integers(0, n_days))
        # Sample identical rise and fade durations alongside peak uplift fraction.
        rf_days = draw_whole_number(other_rf_entry, rng, range_name=range_name)
        uplift = draw_in_range(other_uplift_entry, rng, range_name=range_name)
        secondaries.append(Festival(f"other_festival_{i}", peak_day, rf_days, rf_days, uplift))
    return secondaries


def draw_festival_calendar(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    n_days: int,
    range_name: str = "training",
) -> list[Festival]:
    """
    WHAT: Draws a stochastic festival calendar containing Diwali and secondary festivals.
    WHY: Generates reproducible world-specific holiday timings from pre-registered parameter intervals.

    Args:
        ranges: Configuration dictionary containing background realism parameter specifications.
        rng: Pseudo-random number generator for reproducible sampling.
        n_days: Total duration of the simulation in full days.
        range_name: Target dataset split name containing parameter intervals (default "training").

    Returns:
        List of Festival instances starting with Diwali and followed by sampled secondary festivals.

    Raises:
        ValueError: If range_name is missing or n_days does not exceed the maximum peak bound.

    Example:
        >>> import numpy as np
        >>> from world.background_config import read_background_realism
        >>> from world.festivals import draw_festival_calendar
        >>> rng = np.random.default_rng(seed=42)
        >>> cfg = read_background_realism()
        >>> cal = draw_festival_calendar(cfg, rng, 61, "training")
        >>> [f.name for f in cal]
        ['diwali', 'other_festival_1', 'other_festival_2']
    """
    # Look up the configuration range for the primary Diwali festival peak day.
    peak_entry = get_parameter_entry(ranges, "festival_peak_day_in_world")
    if range_name not in peak_entry:
        raise ValueError(f"Missing range split '{range_name}' for festival_peak_day_in_world")
    max_peak_bound = int(peak_entry[range_name][1])
    # Ensure world duration extends past the maximum allowable festival peak day.
    _validate_world_days(n_days, max_peak_bound)

    # Sample Diwali peak timing, rise duration, fade duration, and peak uplift.
    diwali_peak = draw_whole_number(peak_entry, rng, range_name=range_name)
    rise_entry = get_parameter_entry(ranges, "festival_rise_days")
    fade_entry = get_parameter_entry(ranges, "festival_fade_days")
    uplift_entry = get_parameter_entry(ranges, "festival_uplift_fraction")

    diwali_rise = draw_whole_number(rise_entry, rng, range_name=range_name)
    diwali_fade = draw_whole_number(fade_entry, rng, range_name=range_name)
    diwali_uplift = draw_in_range(uplift_entry, rng, range_name=range_name)
    diwali = Festival("diwali", diwali_peak, diwali_rise, diwali_fade, diwali_uplift)

    # Draw additional secondary festivals to complete the full calendar.
    others = _draw_secondary_festivals(ranges, rng, n_days, range_name)
    return [diwali] + others


def _calculate_single_bump(day_index: int, fest: Festival) -> float:
    """
    WHAT: Calculates the activity uplift fraction contributed by one festival on a specific day.
    WHY: Evaluates a piecewise linear triangle curve for transparent, predictable holiday bumps.

    Args:
        day_index: 0-indexed integer day number within the simulation calendar.
        fest: Festival instance defining peak timing, rise/fade windows, and uplift fraction.

    Returns:
        Float uplift fraction contributed by the festival on the specified day.

    Example:
        >>> from world.festivals import Festival, _calculate_single_bump
        >>> fest = Festival("sample", peak_day=10, rise_days=2, fade_days=2, uplift_fraction=0.5)
        >>> _calculate_single_bump(10, fest)
        0.5
        >>> _calculate_single_bump(9, fest)
        0.25
        >>> _calculate_single_bump(7, fest)
        0.0
    """
    # Determine the start and end boundaries of the festival influence window.
    start_rise = fest.peak_day - fest.rise_days
    end_fade = fest.peak_day + fest.fade_days
    # Return zero uplift if the day falls completely outside the festival window.
    if day_index < start_rise or day_index > end_fade:
        return 0.0

    # Straight lines are chosen for readability and transparency.
    # Interpolate linearly upward from zero to full peak uplift during rise days.
    if day_index <= fest.peak_day:
        if fest.rise_days <= 0:
            return fest.uplift_fraction
        return fest.uplift_fraction * (day_index - start_rise) / fest.rise_days

    # Interpolate linearly downward from full peak uplift to zero during fade days.
    if fest.fade_days <= 0:
        return 0.0
    return fest.uplift_fraction * (end_fade - day_index) / fest.fade_days


def festival_multiplier(day_index: int, festivals: list[Festival]) -> float:
    """
    WHAT: Computes the combined activity multiplier for a given day index across all festivals.
    WHY: Scales baseline daily transaction rates by the additive sum of overlapping festival bumps.

    Args:
        day_index: 0-indexed integer day number within the simulation calendar.
        festivals: List of active Festival instances evaluated across the world.

    Returns:
        Total multiplicative scaling factor (>= 1.0) adjusting baseline transaction volume.

    Example:
        >>> from world.festivals import Festival, festival_multiplier
        >>> fest = Festival("sample", peak_day=10, rise_days=2, fade_days=2, uplift_fraction=0.5)
        >>> festival_multiplier(10, [fest])
        1.5
        >>> festival_multiplier(0, [fest])
        1.0
    """
    # Sum the additive uplift contributions from all active festivals on this day.
    total_bump = 0.0
    for fest in festivals:
        total_bump += _calculate_single_bump(day_index, fest)
    return 1.0 + total_bump
