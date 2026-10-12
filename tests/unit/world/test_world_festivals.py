"""
WHAT: Unit tests for festival calendar generation and festival multipliers.
WHY: Proves Diwali and secondary festivals follow pre-registered windows without hardcoded days,
     exhibit proper linear triangular uplift curves, and respect worst-case multiplier bounds.
ISSUE IDs: R2 (festival calendar and multipliers).
TERM: festival multiplier: A multiplicative factor (>= 1.0) applied to transactions during holiday periods.
"""

from pathlib import Path

import numpy as np
import pytest

from world.background_config import read_background_realism
from world.config_reader import get_parameter_entry
from world.festivals import Festival, draw_festival_calendar, festival_multiplier

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "background_realism.yaml"


def test_f1_diwali_first_and_bounds_over_seeds() -> None:
    """
    WHAT: Asserts Diwali is first, secondary count matches config, and peak days stay within bounds.
    WHY: Verifies the festival calendar structure over 200 independent stochastic seeds.
    """
    ranges = read_background_realism(CONFIG_PATH)
    peak_bounds = get_parameter_entry(ranges, "festival_peak_day_in_world")["training"]
    other_bounds = get_parameter_entry(ranges, "other_festival_count")["training"]

    for seed in range(200):
        rng = np.random.default_rng(seed)
        calendar = draw_festival_calendar(ranges, rng, n_days=61, range_name="training")
        assert len(calendar) >= 1
        diwali = calendar[0]
        assert diwali.name == "diwali"
        assert peak_bounds[0] <= diwali.peak_day <= peak_bounds[1]

        # Secondary festival count check
        other_count = len(calendar) - 1
        assert other_bounds[0] <= other_count <= other_bounds[1]


def test_f2_no_hard_coded_day_diversity() -> None:
    """
    WHAT: Asserts that across 200 seeds, at least 10 distinct Diwali peak days occur.
    WHY: Proves festival timing varies genuinely per simulated world without hardcoded fixed days.
    """
    ranges = read_background_realism(CONFIG_PATH)
    peak_days = set()
    for seed in range(200):
        rng = np.random.default_rng(seed)
        cal = draw_festival_calendar(ranges, rng, n_days=61, range_name="training")
        peak_days.add(cal[0].peak_day)

    assert len(peak_days) >= 10, f"Expected at least 10 distinct peak days, found {len(peak_days)}"


def test_f3_triangular_bump_shape_and_overlap() -> None:
    """
    WHAT: Tests piecewise linear triangle bump shape, symmetry, monotonicity, and overlapping addition.
    WHY: Validates exact mathematical response of festival_multiplier against hand-built Festival fixtures.
    """
    f = Festival(name="test_fest", peak_day=20, rise_days=4, fade_days=4, uplift_fraction=0.2)
    fests = [f]

    # Before rise starts: multiplier is 1.0
    assert festival_multiplier(15, fests) == 1.0
    assert festival_multiplier(16, fests) == 1.0  # peak_day - rise_days = 16
    # Halfway up the rise (day 18): exactly 1 + uplift / 2 = 1.10
    assert abs(festival_multiplier(18, fests) - 1.10) < 1e-9
    # At the peak (day 20): exactly 1 + uplift = 1.20
    assert abs(festival_multiplier(20, fests) - 1.20) < 1e-9
    # Halfway down the fade (day 22): exactly 1 + uplift / 2 = 1.10
    assert abs(festival_multiplier(22, fests) - 1.10) < 1e-9
    # At and after fade (day 24 and 25): exactly 1.0
    assert festival_multiplier(24, fests) == 1.0
    assert festival_multiplier(25, fests) == 1.0

    # Monotonicity along rise and fade
    rise_vals = [festival_multiplier(d, fests) for d in range(16, 21)]
    assert all(x <= y for x, y in zip(rise_vals, rise_vals[1:]))
    fade_vals = [festival_multiplier(d, fests) for d in range(20, 25)]
    assert all(x >= y for x, y in zip(fade_vals, fade_vals[1:]))

    # Overlapping festivals add their bumps: f (0.2 at day 20) + g (0.1 at day 20) = 1.0 + 0.2 + 0.1 = 1.3
    g = Festival(name="overlap", peak_day=20, rise_days=2, fade_days=2, uplift_fraction=0.1)
    assert abs(festival_multiplier(20, [f, g]) - 1.30) < 1e-9


def test_f4_modest_peak_ceiling_over_seeds() -> None:
    """
    WHAT: Asserts that across 200 seeds, maximum festival multiplier never exceeds theoretical upper bound.
    WHY: Protects macroeconomic stability of simulation from unbounded exponential compounding.
    """
    ranges = read_background_realism(CONFIG_PATH)
    d_uplift_max = get_parameter_entry(ranges, "festival_uplift_fraction")["training"][1]
    other_cnt_max = get_parameter_entry(ranges, "other_festival_count")["training"][1]
    other_uplift_max = get_parameter_entry(ranges, "other_festival_uplift_fraction")["training"][1]
    theoretical_max = 1.0 + d_uplift_max + other_cnt_max * other_uplift_max

    for seed in range(200):
        rng = np.random.default_rng(seed)
        cal = draw_festival_calendar(ranges, rng, n_days=61, range_name="training")
        max_mult = max(festival_multiplier(d, cal) for d in range(61))
        assert max_mult <= theoretical_max + 1e-9


def test_f5_refusal_when_world_shorter_than_peak_bound() -> None:
    """
    WHAT: Tests refusal with a plain message if n_days is less than or equal to the peak day upper bound.
    WHY: Ensures worlds are configured with sufficient calendar length to contain the festival window.
    """
    ranges = read_background_realism(CONFIG_PATH)
    rng = np.random.default_rng(42)

    # Upper bound of training festival_peak_day_in_world is 50, so n_days=30 or 50 must be refused
    with pytest.raises(ValueError, match="strictly greater than the upper bound"):
        draw_festival_calendar(ranges, rng, n_days=30, range_name="training")

    with pytest.raises(ValueError, match="strictly greater than the upper bound"):
        draw_festival_calendar(ranges, rng, n_days=50, range_name="training")
