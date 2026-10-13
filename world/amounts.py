"""
WHAT: Payment amount modeling with heavy-tailed distributions and legal UPI transaction splitting.
WHY: Simulates realistic Indian digital payment sizes where most are small but a few are very large,
     and enforces the built-in banking rule that transactions above the regulatory ceiling are split.
ISSUE IDs: E2 (amount distribution, UPI limits).
TERM: heavy-tailed: A statistical pattern where extreme high values have a significant probability.
TERM: lognormal distribution: A distribution whose logarithm is normally distributed, creating a heavy right tail.
TERM: UPI limit: The regulatory maximum amount allowed for a single Unified Payments Interface transfer leg.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.config_reader import draw_in_range
from world.money import rupees_to_paise
from world.placeholders import LOOKALIKE_SMALL_AMOUNT_FRACTION


@dataclass
class AmountProfile:
    """
    WHAT: Holds statistical parameters for a person's transaction amount distribution.
    WHY: Parameterizes the lognormal generator so each individual has a consistent spending scale.
    """

    median_paise: int
    sigma: float


def get_spend_segment_bounds(spend_level: str, low: float, high: float) -> list[float]:
    """
    WHAT: Computes the lower and upper bounds for a spend level tier within an overall range.
    WHY: Splits the continuous range into three equal segments for low, mid, and high spenders.
    """
    delta = (high - low) / 3.0
    if spend_level == "low":
        return [low, low + delta]
    if spend_level == "mid":
        return [low + delta, low + 2.0 * delta]
    if spend_level == "high":
        return [low + 2.0 * delta, high]
    raise ValueError(f"Unknown spend level '{spend_level}'; expected 'low', 'mid', or 'high'")


def create_amount_profile(
    spend_level: str,
    median_entry: dict[str, Any],
    sigma_entry: dict[str, Any],
    rng: np.random.Generator,
    range_name: str = "training",
) -> AmountProfile:
    """
    WHAT: Constructs an AmountProfile for a given spend level ('low', 'mid', or 'high').
    WHY: Cuts the median range into three equal thirds so different demographic groups have distinct spending scales.
    """
    bounds = median_entry[range_name]
    sub_bounds = get_spend_segment_bounds(spend_level, float(bounds[0]), float(bounds[1]))
    segment_entry = {range_name: sub_bounds}

    # Draw the median in rupees for this segment and convert to integer paise.
    median_rupees = draw_in_range(segment_entry, rng, range_name=range_name)
    median_paise = rupees_to_paise(median_rupees)

    # Draw sigma from the configured spread sigma range.
    sigma = draw_in_range(sigma_entry, rng, range_name=range_name)
    return AmountProfile(median_paise=median_paise, sigma=sigma)


def draw_amount_paise(profile: AmountProfile, rng: np.random.Generator) -> int:
    """
    WHAT: Draws one payment amount in integer paise from a heavy-tailed lognormal distribution.
    WHY: Models genuine financial payments with many small transactions and a rare long tail of large orders.
    """
    # For a lognormal distribution, median = exp(mu). Setting mu = ln(median) centers it on profile.median_paise.
    mu = float(np.log(profile.median_paise))
    raw_value = float(rng.lognormal(mean=mu, sigma=profile.sigma))

    # Round to whole paise and ensure every payment is at least 1 paise.
    amount = int(round(raw_value))
    return max(1, amount)


def split_into_legal_payments(amount_paise: int, limit_paise: int) -> list[int]:
    """
    WHAT: Splits a transfer amount into legal payment legs, each at most limit_paise.
    WHY: Built-in banking rule: large amounts are split into multiple legal transactions to comply with
         regulatory caps. This is a built-in infrastructure rule, NOT an evasive gang tactic.
    """
    if amount_paise <= 0:
        raise ValueError(f"Transfer amount must be positive, got {amount_paise}")
    if limit_paise <= 0:
        raise ValueError(f"UPI per-transaction limit must be positive, got {limit_paise}")

    # If the payment already fits within the single-transaction limit, return it unchanged.
    if amount_paise <= limit_paise:
        return [amount_paise]

    # Divide into full limit chunks plus any non-zero remainder.
    parts: list[int] = []
    remaining = amount_paise
    while remaining > limit_paise:
        parts.append(limit_paise)
        remaining -= limit_paise

    if remaining > 0:
        parts.append(remaining)

    return parts


def lookalike_small_amount_paise(median_paise: int) -> int:
    """
    WHAT: Calculates the transfer size in paise for look-alike mule camouflage payments.
    WHY: Establishes a single authoritative location for this financial rule so the legacy
         11A routine runner and the calendar runner compute identical amounts and never drift apart.
    Args: median_paise: The recipient account's median transaction amount in paise.
    Returns: Calculated small payment value in integer paise, floored at a minimum of 1 paise.
    Example:
        # LOOKALIKE_SMALL_AMOUNT_FRACTION is 0.1; for median 500 paise, 0.1 * 500 = 50 paise
        # amt = lookalike_small_amount_paise(500)
        # assert amt == 50
    """
    # Scale recipient median transfer size by the configured fraction (0.1).
    scaled = LOOKALIKE_SMALL_AMOUNT_FRACTION * median_paise
    # Round to the nearest whole paise and enforce a minimum floor of 1 paise.
    return max(1, int(round(scaled)))
