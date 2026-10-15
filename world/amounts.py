"""
WHAT: Payment amount modeling with heavy-tailed distributions and legal UPI transaction splitting.
It generates realistic transaction values using lognormal statistical profiles tailored to demographic spend tiers.
It also partitions oversized transfers into compliant payment legs and standardizes mule camouflage calculations.

WHY: Simulates realistic Indian digital payment sizes where most are small but a few are very large.
Authentic transaction distributions are critical for training fraud detectors to recognize genuine spending variation.
Enforcing regulatory UPI transaction limits simulates real banking infrastructure behavior accurately.

ISSUE IDs: E2 (amount distribution, UPI limits).
TERM: heavy-tailed: A statistical pattern where extreme high values have a significant probability.
      For example, millions of small grocery payments occurring alongside rare expensive vehicle purchases.
TERM: lognormal distribution: A distribution whose logarithm is normally distributed, creating a heavy right tail.
      For example, personal wealth or transaction amounts where values are positive and right-skewed.
TERM: UPI limit: The regulatory maximum amount allowed for a single Unified Payments Interface transfer leg.
      For example, a 100,000-rupee cap per single transaction leg.
TERM: camouflage payment: Small routine payments received by an account to blend in with legitimate consumer traffic.
      For example, receiving 50 paise to mimic active everyday retail interactions.
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
    WHAT: Holds statistical parameters for an individual's transaction amount distribution.
    It encapsulates the baseline median payment amount in integer paise and the lognormal spread sigma.
    These values allow downstream functions to generate consistent payment amounts for a person.

    WHY: Parameterizes the lognormal generator so each individual has a consistent spending scale.
    Encapsulating spending parameters into a structured object prevents parameter mixing across agents.
    It provides a typed interface that simplifies passing spending characteristics to payment samplers.

    Args:
        median_paise: Central payment amount in integer paise characterizing the agent's typical spending.
        sigma: Shape parameter governing the dispersion and right-tail thickness of transaction amounts.

    Returns:
        Configured AmountProfile instance holding median paise and lognormal sigma.

    Example:
        # Create an amount profile with median 15,000 paise (Rs 150) and sigma 0.5:
        # prof = AmountProfile(median_paise=15000, sigma=0.5)
        # assert prof.median_paise == 15000
    """

    median_paise: int
    sigma: float


def get_spend_segment_bounds(spend_level: str, low: float, high: float) -> list[float]:
    """
    WHAT: Computes the lower and upper bounds for a spend level tier within an overall range.
    It divides the interval between low and high into three equal thirds for low, mid, and high spenders.
    It returns the specific two-element boundary list corresponding to the requested spend tier.

    WHY: Splits the continuous range into three equal segments for low, mid, and high spenders.
    Assigning personas to distinct spending segments creates realistic economic stratification.
    Deterministic mathematical segmentation avoids arbitrary hardcoded boundaries between socioeconomic tiers.

    Args:
        spend_level: String identifying the spending tier ('low', 'mid', or 'high').
        low: Minimum floating-point rupee bound of the complete empirical range.
        high: Maximum floating-point rupee bound of the complete empirical range.

    Returns:
        Two-element list containing the lower and upper bounds for the selected tier.

    Example:
        # Segmenting a 100 to 400 rupee range for 'low' tier gives [100.0, 200.0]:
        # bounds = get_spend_segment_bounds("low", 100.0, 400.0)
        # assert bounds == [100.0, 200.0]
    """
    # Calculate the width of one tier by dividing the total range into three equal segments.
    delta = (high - low) / 3.0
    # Map the requested spend tier to its corresponding third of the interval.
    if spend_level == "low":
        return [low, low + delta]
    if spend_level == "mid":
        return [low + delta, low + 2.0 * delta]
    if spend_level == "high":
        return [low + 2.0 * delta, high]
    # Raise an error if an unrecognized spend level string was supplied.
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
    It segments the configured median range into thirds and samples a median payment in rupees.
    The rupee median is converted to integer paise, and sigma is sampled from the spread configuration.

    WHY: Cuts the median range into three equal thirds so different demographic groups have distinct spending scales.
    Sampling both scale and spread from research intervals mirrors real-world population diversity.
    Centralizing profile creation ensures consistent statistical parameters across all simulated agents.

    Args:
        spend_level: Demographic spending tier string ('low', 'mid', or 'high').
        median_entry: Parameter configuration entry for the median transaction amount.
        sigma_entry: Parameter configuration entry for the lognormal spread sigma.
        rng: Initialized NumPy random generator for deterministic sampling.
        range_name: Name of the configuration split to sample from, defaulting to 'training'.

    Returns:
        AmountProfile containing median_paise and lognormal sigma.

    Example:
        # Build profile for 'mid' spend level using genuine payment parameter entries:
        # cfg = read_world_ranges()
        # med_entry = get_parameter_entry(cfg, "genuine_payment_median_amount_inr")
        # sig_entry = get_parameter_entry(cfg, "genuine_payment_amount_spread_sigma")
        # prof = create_amount_profile("mid", med_entry, sig_entry, np.random.default_rng(42))
        # assert prof.median_paise == 41609
    """
    bounds = median_entry[range_name]
    # Extract lower and upper bounds for the persona's designated spend segment.
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
    It computes the lognormal location parameter mu from the profile's median paise.
    The drawn value is rounded to whole paise and floored at a minimum of 1 paise.

    WHY: Models genuine financial payments with many small transactions and a rare long tail of large orders.
    Lognormal sampling accurately reproduces the empirical skew observed in real consumer payment switches.
    Enforcing a positive integer floor ensures every transaction transfers a legal, non-zero amount.

    Args:
        profile: AmountProfile containing median paise and lognormal spread sigma.
        rng: Initialized NumPy random generator for reproducible sampling.

    Returns:
        Positive integer payment amount in paise.

    Example:
        # Draw payment amount using profile with median 41609 paise and seed 42:
        # prof = AmountProfile(median_paise=41609, sigma=1.1633270638512314)
        # amt = draw_amount_paise(prof, np.random.default_rng(42))
        # assert amt == 99617
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
    If the amount is within the limit, it returns a single payment equal to the amount.
    Otherwise, it partitions the amount into full limit chunks followed by any remaining balance.

    WHY: Built-in banking rule: large amounts are split into multiple legal transactions to comply with
    regulatory caps. This is a built-in infrastructure rule, NOT an evasive gang tactic.
    Simulating these automatic payment splits ensures realistic multi-leg transaction streams.

    Args:
        amount_paise: Total transfer amount in integer paise to be partitioned.
        limit_paise: Regulatory maximum ceiling allowed per individual transaction leg.

    Returns:
        List of integer paise amounts each at or below limit_paise summing to amount_paise.

    Example:
        # Splitting a 250,000 paise payment with a 100,000 limit produces three legs:
        # legs = split_into_legal_payments(250000, 100000)
        # assert legs == [100000, 100000, 50000]
    """
    # Validate that transfer amount is strictly positive.
    if amount_paise <= 0:
        raise ValueError(f"Transfer amount must be positive, got {amount_paise}")
    # Validate that UPI limit ceiling is strictly positive.
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
    It scales the recipient account's median transaction amount by the configured look-alike fraction.
    The resulting value is rounded to whole paise and enforced to be at least 1 paise.

    WHY: Establishes a single authoritative location for this financial rule so the legacy
    11A routine runner and the calendar runner compute identical amounts and never drift apart.
    Camouflage payments must scale with account baseline spend to plausibly mimic routine activity.

    Args:
        median_paise: The recipient account's median transaction amount in paise.

    Returns:
        Calculated small payment value in integer paise, floored at a minimum of 1 paise.

    Example:
        # LOOKALIKE_SMALL_AMOUNT_FRACTION is 0.1; for median 5000 paise, 0.1 * 5000 = 500 paise:
        # amt = lookalike_small_amount_paise(5000)
        # assert amt == 500
    """
    # Scale recipient median transfer size by the configured fraction (0.1).
    scaled = LOOKALIKE_SMALL_AMOUNT_FRACTION * median_paise
    # Round to the nearest whole paise and enforce a minimum floor of 1 paise.
    return max(1, int(round(scaled)))

