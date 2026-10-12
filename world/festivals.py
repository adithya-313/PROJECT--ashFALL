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
    """
    other_cnt_entry = get_parameter_entry(ranges, "other_festival_count")
    other_uplift_entry = get_parameter_entry(ranges, "other_festival_uplift_fraction")
    other_rf_entry = get_parameter_entry(ranges, "other_festival_rise_fade_days")

    count = draw_whole_number(other_cnt_entry, rng, range_name=range_name)
    secondaries: list[Festival] = []
    for i in range(1, count + 1):
        # A structural choice: secondary festival peak day is placed uniformly across the world's days.
        peak_day = int(rng.integers(0, n_days))
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
    """
    peak_entry = get_parameter_entry(ranges, "festival_peak_day_in_world")
    if range_name not in peak_entry:
        raise ValueError(f"Missing range split '{range_name}' for festival_peak_day_in_world")
    max_peak_bound = int(peak_entry[range_name][1])
    _validate_world_days(n_days, max_peak_bound)

    diwali_peak = draw_whole_number(peak_entry, rng, range_name=range_name)
    rise_entry = get_parameter_entry(ranges, "festival_rise_days")
    fade_entry = get_parameter_entry(ranges, "festival_fade_days")
    uplift_entry = get_parameter_entry(ranges, "festival_uplift_fraction")

    diwali_rise = draw_whole_number(rise_entry, rng, range_name=range_name)
    diwali_fade = draw_whole_number(fade_entry, rng, range_name=range_name)
    diwali_uplift = draw_in_range(uplift_entry, rng, range_name=range_name)
    diwali = Festival("diwali", diwali_peak, diwali_rise, diwali_fade, diwali_uplift)

    others = _draw_secondary_festivals(ranges, rng, n_days, range_name)
    return [diwali] + others


def _calculate_single_bump(day_index: int, fest: Festival) -> float:
    """
    WHAT: Calculates the activity uplift fraction contributed by one festival on a specific day.
    WHY: Evaluates a piecewise linear triangle curve for transparent, predictable holiday bumps.
    """
    start_rise = fest.peak_day - fest.rise_days
    end_fade = fest.peak_day + fest.fade_days
    if day_index < start_rise or day_index > end_fade:
        return 0.0

    # Straight lines are chosen for readability and transparency.
    if day_index <= fest.peak_day:
        if fest.rise_days <= 0:
            return fest.uplift_fraction
        return fest.uplift_fraction * (day_index - start_rise) / fest.rise_days

    if fest.fade_days <= 0:
        return 0.0
    return fest.uplift_fraction * (end_fade - day_index) / fest.fade_days


def festival_multiplier(day_index: int, festivals: list[Festival]) -> float:
    """
    WHAT: Computes the combined activity multiplier for a given day index across all festivals.
    WHY: Scales baseline daily transaction rates by the additive sum of overlapping festival bumps.
    """
    total_bump = 0.0
    for fest in festivals:
        total_bump += _calculate_single_bump(day_index, fest)
    return 1.0 + total_bump
