"""
WHAT: Festival uplift measurement for Level 11 showcase check 3.
It splits each day's payments into calendar-scaled and fixed-date parts, removes the weekend and month-start
effects with the engine's own multiplier functions, and compares the extra festival payments with the plan.

WHY: A plain inside-versus-outside average mixes three calendar effects and hides the festival effect on seed 101.
Only everyday and lookalike payments are scaled by the calendar; salary, rent, bill and episode payments are not.
Separating the parts lets the check test the festival rise itself with a four-standard-error bound.

TERM: calendar-scaled payment: a payment whose daily count the engine multiplies by daily_multiplier,
      e.g. an everyday shop purchase (purpose "everyday") or a small incoming transfer (purpose "lookalike_in").
TERM: fixed-date payment: a payment placed on a set day whatever the calendar,
      e.g. a salary on the 1st of the month (purpose "salary") or a wedding gift (purpose "episode_in").
TERM: other multiplier: the month-start factor times the weekend factor for one day, the festival factor left out,
      e.g. 1.2 x 1.1 = 1.32 on a Saturday that is also the 1st of the month.
TERM: base rate: calendar-scaled payments per day when every multiplier equals 1,
      e.g. 2,963 payments a day in a 4,000-person world.
TERM: standard error: the expected size of random wobble in a measured number,
      e.g. a Poisson count of 10,000 has a standard error of about 100.
"""

import math
from dataclasses import dataclass
from typing import Any

from world.festivals import festival_multiplier
from world.rhythms import month_start_multiplier, weekend_multiplier

# The two purposes whose counts the engine scales by daily_multiplier (see world/calendar_runner.py).
CALENDAR_SCALED_PURPOSES: tuple[str, ...] = ("everyday", "lookalike_in")


@dataclass
class DailySplit:
    """
    WHAT: Per-day payment counts split into total, fixed-date and calendar-scaled parts.
    WHY: Keeps the three lists together so evidence files and calculations read the same numbers.
    """

    total: list[int]
    fixed_date: list[int]
    calendar_scaled: list[int]


@dataclass
class FestivalUplift:
    """
    WHAT: Every intermediate number of the festival uplift test, plus the two test results.
    WHY: The evidence file and the key number must show exactly the values the pass decision used.
    """

    base: float
    sum_other_festival: float
    sum_other_normal: float
    count_normal: int
    observed: int
    expected_without: float
    excess: float
    planned_excess: float
    se_excess: float
    rise_is_real: bool
    size_matches_plan: bool


def split_daily_counts(events: list[Any], purposes: list[str], n_days: int) -> DailySplit:
    """
    WHAT: Counts payment events per day, separating calendar-scaled purposes from fixed-date purposes.
    It walks the events and the parallel purpose list together and adds each event to its day.
    Events past the last day (none in practice) are counted on the last day, matching check 3's daily view.

    WHY: The festival test must only use payments that the calendar multiplies.
    RunResult.event_purposes holds one purpose per event, and run_world refuses any purpose outside
    PAYMENT_PURPOSES, so the split covers every event with no gaps.

    Args:
        events: PaymentEvent records from a run, each with a minute field.
        purposes: Purpose strings aligned one-to-one with events.
        n_days: Number of days in the run.

    Returns:
        DailySplit with total, fixed-date and calendar-scaled counts per day.

    Example:
        >>> from types import SimpleNamespace as E
        >>> events = [E(minute=10), E(minute=1500), E(minute=1600)]
        >>> split = split_daily_counts(events, ["salary", "everyday", "bill"], 2)
        >>> split.total, split.fixed_date, split.calendar_scaled
        ([1, 2], [1, 1], [0, 1])
    """
    total = [0] * n_days
    fixed_date = [0] * n_days
    calendar_scaled = [0] * n_days
    for event, purpose in zip(events, purposes, strict=True):
        # 1440 minutes make one day; clamp so a late event still lands inside the list.
        day = min(event.minute // 1440, n_days - 1)
        total[day] += 1
        if purpose in CALENDAR_SCALED_PURPOSES:
            calendar_scaled[day] += 1
        else:
            fixed_date[day] += 1
    return DailySplit(total, fixed_date, calendar_scaled)


def calendar_parts(plan: Any, n_days: int) -> tuple[list[float], list[float]]:
    """
    WHAT: Returns each day's festival multiplier and its other (month-start x weekend) multiplier.
    It evaluates the engine's own functions at midnight of each day, as the calendar runner does.
    Multiplying the two lists day by day gives exactly world.rhythms.daily_multiplier.

    WHY: Using the engine's functions, not a copy of their formulas, keeps the test tied to the real plan.
    Separating the festival factor from the rest is what lets the check remove weekend and month-start effects.

    Args:
        plan: WorldPlan holding clock, rhythm_settings and festivals.
        n_days: Number of days to evaluate.

    Returns:
        Tuple of (other multipliers per day, festival multipliers per day).

    Example:
        >>> from world.world_plan import draw_world_plan, spawn_world_streams
        >>> plan = draw_world_plan(spawn_world_streams(101)[0], 61)
        >>> other, fest = calendar_parts(plan, 61)
        >>> len(other) == 61 and min(fest) == 1.0 and max(fest) > 1.0
        True
    """
    other: list[float] = []
    fest: list[float] = []
    for day in range(n_days):
        minute = day * 1440
        month_part = month_start_multiplier(plan.clock.day_of_month(minute), plan.rhythm_settings)
        weekend_part = weekend_multiplier(plan.clock.is_weekend(minute), plan.rhythm_settings)
        other.append(month_part * weekend_part)
        fest.append(festival_multiplier(day, plan.festivals))
    return other, fest


def measure_festival_uplift(
    scaled: list[int], other: list[float], fest: list[float], k_sigma: float
) -> FestivalUplift:
    """
    WHAT: Tests whether festival days carry the extra calendar-scaled payments that the plan sets.
    The base rate comes from non-festival days only: base = sum(s) / sum(other).
    On festival days it compares the observed count with base x other (no festival) and with the planned rise
    base x other x (fest - 1). The standard error adds the Poisson noise of the festival-day count to the
    noise carried in through the base: se = sqrt(observed + (S_f / S_n)^2 x C_n).

    WHY: Removing weekend and month-start effects isolates the festival factor; testing both that the rise
    is above zero and that its size matches the plan, each at k standard errors, shows the plan is applied.

    Args:
        scaled: Calendar-scaled payment counts per day.
        other: Month-start x weekend multipliers per day.
        fest: Festival multipliers per day (1.0 means no festival).
        k_sigma: Number of standard errors for both bounds (4.0 in the showcase).

    Returns:
        FestivalUplift with every intermediate value and both test results.

    Example:
        Worked by hand. Days: two normal days with other 1, 1 and counts 100, 100; two festival days with other 1, 2,
        fest 1.5, 1.5 and counts 150, 300. S_n = 2, C_n = 200, base = 100. S_f = 3, expected_without = 300,
        observed = 450, excess = 150, planned = 100 x 1 x 0.5 + 100 x 2 x 0.5 = 150.
        se = sqrt(450 + (3/2)^2 x 200) = sqrt(900) = 30. 150 > 4 x 30 = 120, and |150 - 150| = 0 <= 120.
        >>> r = measure_festival_uplift([100, 100, 150, 300], [1, 1, 1, 2], [1.0, 1.0, 1.5, 1.5], 4.0)
        >>> (r.base, r.expected_without, r.excess, r.planned_excess, r.se_excess, r.rise_is_real, r.size_matches_plan)
        (100.0, 300.0, 150.0, 150.0, 30.0, True, True)
    """
    normal_days = [d for d in range(len(fest)) if fest[d] == 1.0]
    festival_days = [d for d in range(len(fest)) if fest[d] > 1.0]
    sum_other_normal = sum(other[d] for d in normal_days)
    sum_other_festival = sum(other[d] for d in festival_days)
    count_normal = sum(scaled[d] for d in normal_days)
    # Payments per day when every multiplier is 1, learned from days without festivals.
    base = count_normal / sum_other_normal
    expected_without = base * sum_other_festival
    observed = sum(scaled[d] for d in festival_days)
    excess = observed - expected_without
    planned_excess = sum(base * other[d] * (fest[d] - 1.0) for d in festival_days)
    # Noise of the festival-day count plus the noise inherited from estimating base.
    ratio = sum_other_festival / sum_other_normal
    se_excess = math.sqrt(observed + ratio * ratio * count_normal)
    return FestivalUplift(
        base, sum_other_festival, sum_other_normal, count_normal, observed, expected_without, excess,
        planned_excess, se_excess, excess > k_sigma * se_excess,
        abs(excess - planned_excess) <= k_sigma * se_excess,
    )


def planned_signal_ratio(
    tiny_scaled: list[int], tiny_people: int, n_people: int, other: list[float], fest: list[float]
) -> float:
    """
    WHAT: Predicts planned_excess / sqrt(expected_without) for a world of n_people, before running it.
    It estimates base per person from the tiny world's non-festival days, scales it to n_people,
    and evaluates the planned rise and the no-festival expectation from the plan's multipliers only.

    WHY: The size rule must be decided from the plan, not from the festival-day results of the candidate world.
    Calendar-scaled payments are drawn per person, so base grows in proportion to the number of people.

    Args:
        tiny_scaled: Calendar-scaled counts per day from the tiny world.
        tiny_people: Number of people in the tiny world.
        n_people: Number of people in the candidate world.
        other: Month-start x weekend multipliers per day (same seed, so same plan).
        fest: Festival multipliers per day.

    Returns:
        The predicted ratio of planned excess to its Poisson noise.

    Example:
        Tiny base 100 per day for 10 people; candidate 40 people gives base 400.
        With festival days other [1, 2] and fest [1.5, 1.5]: planned 600, expected_without 1,200,
        ratio 600 / sqrt(1200) = 17.32.
        >>> round(planned_signal_ratio([100, 100, 0, 0], 10, 40, [1, 1, 1, 2], [1.0, 1.0, 1.5, 1.5]), 2)
        17.32
    """
    normal_days = [d for d in range(len(fest)) if fest[d] == 1.0]
    festival_days = [d for d in range(len(fest)) if fest[d] > 1.0]
    tiny_base = sum(tiny_scaled[d] for d in normal_days) / sum(other[d] for d in normal_days)
    base = tiny_base * n_people / tiny_people
    expected_without = sum(base * other[d] for d in festival_days)
    planned = sum(base * other[d] * (fest[d] - 1.0) for d in festival_days)
    return planned / math.sqrt(expected_without)


def choose_festival_people(
    tiny_scaled: list[int], tiny_people: int, candidates: list[int],
    other: list[float], fest: list[float], min_ratio: float,
) -> tuple[int, dict[int, float]]:
    """
    WHAT: Picks the smallest candidate population whose predicted signal ratio reaches min_ratio.
    It evaluates planned_signal_ratio for every candidate and returns the first that qualifies,
    together with all predicted ratios for the evidence file.

    WHY: Fixes the festival world size by a rule written in config before any festival-day result is seen,
    so the size cannot be tuned to make the check pass.

    Args:
        tiny_scaled: Calendar-scaled counts per day from the tiny world.
        tiny_people: Number of people in the tiny world.
        candidates: Candidate populations in increasing order.
        other: Month-start x weekend multipliers per day.
        fest: Festival multipliers per day.
        min_ratio: Smallest acceptable predicted ratio.

    Returns:
        Tuple of (chosen population, mapping of candidate to predicted ratio).

    Example:
        With the hand example of planned_signal_ratio, 10 people gives 8.66 and 40 people gives 17.32.
        >>> other, fest = [1, 1, 1, 2], [1.0, 1.0, 1.5, 1.5]
        >>> people, ratios = choose_festival_people([100, 100, 0, 0], 10, [10, 40], other, fest, 8.0)
        >>> people, round(ratios[10], 2)
        (10, 8.66)
    """
    ratios = {n: planned_signal_ratio(tiny_scaled, tiny_people, n, other, fest) for n in candidates}
    for n in candidates:
        if ratios[n] >= min_ratio:
            return n, ratios
    raise ValueError(f"No candidate population reaches ratio {min_ratio}: {ratios}")
