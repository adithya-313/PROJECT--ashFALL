"""
WHAT: Unit tests for Step 2 of the world package (amounts and the split rule).
WHY: Verifies T4 (heavy-tailed lognormal amounts, p99 ratio, spend-level partitioning)
     and T5 (UPI per-transaction limit splitting rule).
ISSUE IDs: E2 (amount distribution, UPI limits).
TERM: heavy-tailed distribution: A probability distribution where extreme high values occur
      more frequently than in a normal distribution (here, a lognormal distribution).
"""

import math

import numpy as np
import pytest

from world.amounts import (
    AmountProfile,
    create_amount_profile,
    draw_amount_paise,
    split_into_legal_payments,
)
from world.config_reader import draw_in_range, get_parameter_entry, read_world_ranges
from world.money import rupees_to_paise


def test_t4_amounts_heavy_tailed_properties() -> None:
    """
    WHAT: Draws 20,000 amounts and validates heavy-tailed lognormal properties (p99/median, mean > median, min >= 1).
    WHY: Fulfills test requirement T4 ensuring financial amounts match empirical economic shape.
    """
    rng = np.random.default_rng(42)
    median_paise = rupees_to_paise(300)  # 30,000 paise
    sigma = 1.2
    profile = AmountProfile(median_paise=median_paise, sigma=sigma)

    draws = np.array([draw_amount_paise(profile, rng) for _ in range(20000)], dtype=np.int64)

    # Every amount is at least 1 paise
    assert np.all(draws >= 1)

    measured_median = float(np.median(draws))
    measured_mean = float(np.mean(draws))
    # Heavy tail property: mean is strictly greater than median
    assert measured_mean > measured_median

    measured_p99 = float(np.percentile(draws, 99))
    ratio = measured_p99 / measured_median
    expected_ratio = math.exp(2.326 * sigma)

    # Within 25% of theoretical value
    relative_error = abs(ratio - expected_ratio) / expected_ratio
    assert relative_error < 0.25


def test_t4_spend_levels_give_different_medians() -> None:
    """
    WHAT: Tests that low, mid, and high spend levels draw different non-overlapping median segments.
    WHY: Fulfills test requirement T4 ensuring demographic tiers have distinct purchasing scales.
    """
    ranges = read_world_ranges()
    median_entry = get_parameter_entry(ranges, "genuine_payment_median_amount_inr")
    sigma_entry = get_parameter_entry(ranges, "genuine_payment_amount_spread_sigma")
    rng = np.random.default_rng(999)

    low_p = create_amount_profile("low", median_entry, sigma_entry, rng)
    mid_p = create_amount_profile("mid", median_entry, sigma_entry, rng)
    high_p = create_amount_profile("high", median_entry, sigma_entry, rng)

    assert low_p.median_paise < mid_p.median_paise < high_p.median_paise


def test_t5_split_rule_with_config_limit() -> None:
    """
    WHAT: Tests splitting an amount into parts where every part is at most the UPI limit.
    WHY: Fulfills test requirement T5 ensuring compliance with the built-in UPI per-transaction ceiling.
    """
    ranges = read_world_ranges()
    limit_entry = get_parameter_entry(ranges, "upi_per_transaction_limit_inr")
    rng = np.random.default_rng(101)

    # Draw limit from config training range (rupees converted to paise)
    limit_rupees = draw_in_range(limit_entry, rng, range_name="training")
    limit_paise = rupees_to_paise(limit_rupees)

    # Exactly at limit stays one part
    single_part = split_into_legal_payments(limit_paise, limit_paise)
    assert single_part == [limit_paise]

    # Limit + 1 paise becomes exactly two parts
    two_parts = split_into_legal_payments(limit_paise + 1, limit_paise)
    assert len(two_parts) == 2
    assert two_parts == [limit_paise, 1]

    # Large amount split
    large_amount = limit_paise * 3 + 45000
    parts = split_into_legal_payments(large_amount, limit_paise)
    assert sum(parts) == large_amount
    for p in parts:
        assert 1 <= p <= limit_paise


def test_split_rule_rejects_invalid_inputs() -> None:
    """
    WHAT: Checks that split_into_legal_payments refuses negative or zero amounts or limits.
    WHY: Defends against illegal financial states.
    """
    with pytest.raises(ValueError, match="amount"):
        split_into_legal_payments(0, 1000)
    with pytest.raises(ValueError, match="limit"):
        split_into_legal_payments(1000, 0)
