"""
WHAT: Unit tests for the Level 11 festival uplift calculation used by showcase check 3.
They feed small hand-made day lists whose answers can be worked out with pen and paper.

WHY: The festival check passes or fails on these formulas, so each formula needs a test that does not depend
on running a world. Hand-worked numbers make a wrong sign, a wrong sum or a loosened bound show up at once.

TERM: hand-made example: input lists short enough that every sum in the docstring can be checked by hand,
      e.g. four days with counts 100, 100, 150 and 300.
TERM: size rule: choosing the festival world's population from predicted numbers before running it,
      e.g. the smallest candidate whose predicted signal ratio is at least 8.
"""

from types import SimpleNamespace

import pytest

from showcase.level_11_festival import (
    choose_festival_people,
    measure_festival_uplift,
    planned_signal_ratio,
    split_daily_counts,
)


def test_measure_festival_uplift_hand_example() -> None:
    """
    WHAT: Checks every intermediate value of measure_festival_uplift on a four-day example worked by hand.
    WHY: Proves base, expected_without, excess, planned_excess and se use exactly the owner's formulas.

    Returns:
        None; fails with an assertion error if any value differs.

    Example:
        Normal days: other 1, 1, counts 100, 100 -> S_n = 2, C_n = 200, base = 100.
        Festival days: other 1, 2, fest 1.5, 1.5, counts 150, 300 -> S_f = 3, expected_without = 300,
        observed = 450, excess = 150, planned = 50 + 100 = 150, se = sqrt(450 + 2.25 x 200) = 30.
        Test 1: 150 > 120 passes. Test 2: |150 - 150| = 0 <= 120 passes.
    """
    result = measure_festival_uplift([100, 100, 150, 300], [1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.5, 1.5], 4.0)
    assert result.base == 100.0
    assert result.sum_other_normal == 2.0 and result.sum_other_festival == 3.0
    assert result.count_normal == 200 and result.observed == 450
    assert result.expected_without == 300.0
    assert result.excess == 150.0 and result.planned_excess == 150.0
    assert result.se_excess == pytest.approx(30.0)
    assert result.rise_is_real and result.size_matches_plan


def test_measure_festival_uplift_rejects_missing_or_wrong_size_rise() -> None:
    """
    WHAT: Checks that the test fails when festival days show no rise, and when the rise is far larger than planned.
    WHY: A check that cannot fail proves nothing; both bounds must reject data that contradicts the plan.

    Returns:
        None; fails with an assertion error if either bad case is accepted.

    Example:
        No rise: festival counts 100, 200 equal expected_without 300, so excess = 0, se = sqrt(300 + 450) = 27.4,
        and 0 > 109.5 fails. Too large: counts 300, 600 give excess 600 vs planned 150; se = sqrt(900 + 450) = 36.7,
        so 600 > 146.9 passes test 1 but |600 - 150| = 450 > 146.9 fails test 2.
    """
    flat = measure_festival_uplift([100, 100, 100, 200], [1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.5, 1.5], 4.0)
    assert flat.excess == 0.0 and not flat.rise_is_real
    big = measure_festival_uplift([100, 100, 300, 600], [1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.5, 1.5], 4.0)
    assert big.rise_is_real and not big.size_matches_plan


def test_split_daily_counts_by_purpose() -> None:
    """
    WHAT: Checks that only everyday and lookalike_in events count as calendar-scaled.
    WHY: Salary, rent, bill and episode payments are not scaled by the calendar and must not enter the festival test.

    Returns:
        None; fails with an assertion error if any event lands in the wrong part.

    Example:
        Day 0: salary, everyday. Day 1: lookalike_in, episode_in, rent.
        Totals [2, 3]; fixed-date [1, 2]; calendar-scaled [1, 1].
    """
    minutes = [5, 6, 1441, 1442, 1443]
    purposes = ["salary", "everyday", "lookalike_in", "episode_in", "rent"]
    events = [SimpleNamespace(minute=m) for m in minutes]
    split = split_daily_counts(events, purposes, 2)
    assert split.total == [2, 3]
    assert split.fixed_date == [1, 2]
    assert split.calendar_scaled == [1, 1]


def test_size_rule_picks_smallest_qualifying_population() -> None:
    """
    WHAT: Checks planned_signal_ratio and choose_festival_people on hand-worked numbers.
    WHY: The size rule must be computed from the plan alone and must pick the smallest qualifying candidate.

    Returns:
        None; fails with an assertion error if the ratio or the chosen size is wrong.

    Example:
        Tiny base = 200 / 2 = 100 for 10 people. For n people base = 10 n, expected_without = 30 n,
        planned = 15 n, ratio = 15 n / sqrt(30 n). n = 5 -> 6.12; n = 10 -> 8.66; n = 40 -> 17.32.
        With minimum 8 the rule picks 10.
    """
    other, fest = [1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.5, 1.5]
    assert planned_signal_ratio([100, 100, 0, 0], 10, 40, other, fest) == pytest.approx(17.3205, abs=1e-3)
    people, ratios = choose_festival_people([100, 100, 0, 0], 10, [5, 10, 40], other, fest, 8.0)
    assert people == 10
    assert ratios[5] == pytest.approx(6.1237, abs=1e-3)
    with pytest.raises(ValueError):
        choose_festival_people([100, 100, 0, 0], 10, [5], other, fest, 8.0)
