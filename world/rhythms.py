"""
WHAT: Diurnal, weekly, monthly, and festival payment activity rhythm multipliers.
WHY: Generates realistic human transaction waves (Diwali, start of month, weekends, diurnal peaks).
NOTE: Nothing in this module changes payment generation yet; all multipliers are standalone for now.
ISSUE IDs: R2 (always-on rhythms).
TERM: diurnal rhythm: Periodic fluctuations in human activity recurring over a 24-hour cycle.
TERM: circular distance: Shortest distance between two clock hours wrapping smoothly across midnight.
TERM: activity multiplier: A composite scaling factor adjusting base transaction probabilities.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.background_config import draw_whole_number
from world.clock import SimClock
from world.config_reader import draw_in_range, get_parameter_entry
from world.festivals import Festival, festival_multiplier

VALID_RANGES = ("training", "test", "stress")
PERSONA_KEYS = (
    "salaried_employee",
    "student",
    "retiree",
    "small_shopkeeper",
    "gig_worker",
)


@dataclass(frozen=True)
class RhythmSettings:
    """
    WHAT: Immutable configuration container holding drawn temporal rhythm parameters.
    WHY: Centralizes diurnal peak hours, weekend shifts, and monthly uplift constants.
    """

    month_start_uplift: float
    month_start_easing_days: int
    weekend_difference: float
    night_floor: float
    peak_width_hours: float
    peak_hours: dict[str, float]


def _draw_persona_peak_hours(
    ranges: dict[str, Any], rng: np.random.Generator, range_name: str
) -> dict[str, float]:
    """
    WHAT: Samples the daily peak activity hour for each demographic persona.
    WHY: Distinguishes lifestyle routines (e.g. retirees in morning vs students at night).
    """
    peaks: dict[str, float] = {}
    for persona in PERSONA_KEYS:
        entry = get_parameter_entry(ranges, f"peak_hour_{persona}")
        peaks[persona] = draw_in_range(entry, rng, range_name)
    return peaks


def draw_rhythm_settings(
    ranges: dict[str, Any], rng: np.random.Generator, range_name: str = "training"
) -> RhythmSettings:
    """
    WHAT: Samples all temporal rhythm settings from configured parameter ranges.
    WHY: Seeds deterministic world variation across monthly, weekly, and hourly habits.
    """
    if range_name not in VALID_RANGES:
        raise ValueError(f"Unknown range name '{range_name}'; expected one of {VALID_RANGES}")
    uplift = draw_in_range(get_parameter_entry(ranges, "month_start_uplift_fraction"), rng, range_name)
    easing = draw_whole_number(get_parameter_entry(ranges, "month_start_easing_days"), rng, range_name)
    weekend = draw_in_range(get_parameter_entry(ranges, "weekend_difference_fraction"), rng, range_name)
    floor = draw_in_range(get_parameter_entry(ranges, "night_activity_floor"), rng, range_name)
    width = draw_in_range(get_parameter_entry(ranges, "daily_peak_width_hours"), rng, range_name)
    peaks = _draw_persona_peak_hours(ranges, rng, range_name)
    return RhythmSettings(
        month_start_uplift=uplift,
        month_start_easing_days=easing,
        weekend_difference=weekend,
        night_floor=floor,
        peak_width_hours=width,
        peak_hours=peaks,
    )


def month_start_multiplier(day_of_month: int, settings: RhythmSettings) -> float:
    """
    WHAT: Computes the start-of-month salary and bill-paying activity multiplier.
    WHY: Models elevated spending at the beginning of calendar months fading to normal.
    """
    # Linear easing from peak uplift on day 1 to 0 on day 1 + easing_days.
    decay = max(0.0, 1.0 - (day_of_month - 1) / float(settings.month_start_easing_days))
    return 1.0 + settings.month_start_uplift * decay


def weekend_multiplier(is_weekend: bool, settings: RhythmSettings) -> float:
    """
    WHAT: Evaluates weekday versus weekend transactional volume difference factor.
    WHY: Applies commercial adjustments reflecting weekend leisure or business closures.
    """
    if is_weekend:
        return 1.0 + settings.weekend_difference
    return 1.0


def _raw_hour_value(hour: float, peak: float, floor: float, width: float) -> float:
    """
    WHAT: Computes unnormalized diurnal activity weight via circular Gaussian curve.
    WHY: Captures smooth falloff around peak hour with a baseline floor at night.
    """
    # Circular difference in hours wrapping across midnight (e.g., 23:00 to 01:00 is 2h).
    diff = abs(hour - peak)
    d = min(diff, 24.0 - diff)
    gaussian = np.exp(-0.5 * (d / width) ** 2)
    return float(floor + (1.0 - floor) * gaussian)


def hour_weight(hour: float, persona_name: str, settings: RhythmSettings) -> float:
    """
    WHAT: Calculates normalized diurnal weight for a persona with mean 1.0 over 24 hours.
    WHY: Concentrates transactions around persona peak hours without altering daily totals.
    """
    if persona_name not in settings.peak_hours:
        raise ValueError(f"Unknown persona '{persona_name}'; expected one of {list(settings.peak_hours.keys())}")
    peak = settings.peak_hours[persona_name]
    raw_target = _raw_hour_value(hour, peak, settings.night_floor, settings.peak_width_hours)
    # Average across all 24 integer hours to normalize daily integral to 1.0.
    mean_raw = sum(
        _raw_hour_value(float(h), peak, settings.night_floor, settings.peak_width_hours)
        for h in range(24)
    ) / 24.0
    return raw_target / mean_raw


def daily_multiplier(
    minute: int, clock: SimClock, settings: RhythmSettings, festivals: list[Festival]
) -> float:
    """
    WHAT: Combines festival, start-of-month, and weekend factors for a simulation minute.
    WHY: Produces macro calendar-level activity modulation factor.
    """
    f_mult = festival_multiplier(clock.day_index(minute), festivals)
    m_mult = month_start_multiplier(clock.day_of_month(minute), settings)
    w_mult = weekend_multiplier(clock.is_weekend(minute), settings)
    return f_mult * m_mult * w_mult


def activity_multiplier(
    minute: int,
    persona_name: str,
    clock: SimClock,
    settings: RhythmSettings,
    festivals: list[Festival],
) -> float:
    """
    WHAT: Calculates total transactional activity multiplier for a given persona and minute.
    WHY: Composes macro daily calendar factors with micro diurnal hourly weights.
    """
    daily = daily_multiplier(minute, clock, settings, festivals)
    hourly = hour_weight(clock.hour_of_day(minute), persona_name, settings)
    return daily * hourly


def daily_curve(
    clock: SimClock, n_days: int, settings: RhythmSettings, festivals: list[Festival]
) -> list[float]:
    """
    WHAT: Samples daily calendar multiplier values evaluated at midnight for each world day.
    WHY: Provides a continuous sequence of daily activity factors for plotting and analysis.
    """
    return [daily_multiplier(day * 1440, clock, settings, festivals) for day in range(n_days)]
