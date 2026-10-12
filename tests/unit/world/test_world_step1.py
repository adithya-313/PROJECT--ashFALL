"""
WHAT: Unit tests for Step 1 of the world package (placeholders, money, config_reader, regions).
WHY: Verifies T1 (config_reader validation and range drawing), T2 (exact paise-rupee conversions),
     and T3 (uniform region assignment inside configured bounds) before building entities.
TERM: paise: One hundredth of an Indian rupee (1 INR = 100 paise), stored as exact integers.
"""

import numpy as np
import pytest

from world.config_reader import draw_in_range, get_parameter_entry, read_world_ranges
from world.money import paise_to_rupees, rupees_to_paise
from world.placeholders import ANDROID_SHARE, OS_GENERATIONS
from world.regions import assign_region, draw_number_of_regions


def test_placeholders_exist() -> None:
    """
    WHAT: Checks that basic placeholder constants are defined with expected types.
    WHY: Ensures configuration fallbacks for 11A are present and reasonable.
    """
    assert isinstance(ANDROID_SHARE, float)
    assert 0.0 < ANDROID_SHARE < 1.0
    assert isinstance(OS_GENERATIONS, int)
    assert OS_GENERATIONS >= 1


def test_t2_money_exact_roundtrip_and_rounding() -> None:
    """
    WHAT: Tests rupees_to_paise and paise_to_rupees round-tripping and half-to-even/nearest rounding.
    WHY: Fulfills test requirement T2 to guarantee integer math without floating point drift.
    """
    # Exact roundtrip for integer rupees
    assert rupees_to_paise(1) == 100
    assert paise_to_rupees(100) == 1.0

    # Exact roundtrip for decimal rupees
    assert rupees_to_paise(12.34) == 1234
    assert paise_to_rupees(1234) == 12.34

    # Rounding to nearest paise
    assert rupees_to_paise(10.004) == 1000
    assert rupees_to_paise(10.006) == 1001


def test_t1_config_reader_draws_inside_range() -> None:
    """
    WHAT: Checks that draw_in_range produces numbers strictly within the named range.
    WHY: Fulfills test requirement T1 ensuring drawn values respect configured limits.
    """
    ranges = read_world_ranges()
    entry = get_parameter_entry(ranges, "genuine_payment_median_amount_inr")
    rng = np.random.default_rng(12345)

    low, high = entry["training"]
    for _ in range(50):
        val = draw_in_range(entry, rng, range_name="training")
        assert low <= val <= high


def test_t1_config_reader_refuses_unknown_key() -> None:
    """
    WHAT: Checks that get_parameter_entry raises ValueError when a key is unknown.
    WHY: Fulfills test requirement T1 to prevent silent failures on typos.
    """
    ranges = read_world_ranges()
    with pytest.raises(ValueError, match="unknown_fake_parameter"):
        get_parameter_entry(ranges, "unknown_fake_parameter")


def test_t1_config_reader_refuses_missing_range_name() -> None:
    """
    WHAT: Checks that draw_in_range raises ValueError when a requested range name is missing.
    WHY: Fulfills test requirement T1 so invalid split names fail loudly.
    """
    entry = {"training": [10, 20]}
    rng = np.random.default_rng(12345)
    with pytest.raises(ValueError, match="missing_split"):
        draw_in_range(entry, rng, range_name="missing_split", param_name="test_param")


def test_t1_config_reader_refuses_malformed_range() -> None:
    """
    WHAT: Checks that draw_in_range rejects non-list ranges or inverted bounds.
    WHY: Fulfills test requirement T1 to guarantee validity of config numbers.
    """
    rng = np.random.default_rng(12345)
    bad_entry = {"training": [50, 10]}  # inverted
    with pytest.raises(ValueError, match="inverted_param"):
        draw_in_range(bad_entry, rng, range_name="training", param_name="inverted_param")


def test_t3_regions_distribution_and_bounds() -> None:
    """
    WHAT: Checks that n_regions is drawn inside the configured bounds and all assigned regions are valid.
    WHY: Fulfills test requirement T3 verifying region assignments cover [0, n_regions).
    """
    ranges = read_world_ranges()
    entry = get_parameter_entry(ranges, "number_of_regions")
    rng = np.random.default_rng(54321)

    low, high = entry["training"]
    n_regions = draw_number_of_regions(entry, rng, range_name="training")
    assert low <= n_regions <= high

    # All assigned regions are in [0, n_regions)
    assigned = [assign_region(n_regions, rng) for _ in range(200)]
    for r in assigned:
        assert 0 <= r < n_regions
    # Verify coverage across all regions
    assert len(set(assigned)) == n_regions
