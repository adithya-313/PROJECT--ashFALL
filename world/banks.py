"""
WHAT: Bank entity definition and ecosystem generation across three financial institutions.
It models the simulated municipal banking sector by creating public, private, and payments banks.
Each institution is configured with empirical market share proportions, KYC rigor, and hold delays.

WHY: Fraud vulnerability and response latency vary significantly across real-world banking institutions.
Mule recruiters exploit banks with weaker identity verification, while faster banks freeze funds quicker.
Explicitly modeling institutional diversity allows fraud detection algorithms to evaluate multi-bank risk.

ISSUE IDs: E4 (bank differences).
TERM: KYC: Know Your Customer identity verification processes required when opening bank accounts. For example, verifying a national ID card and proof of residence.
TERM: payments bank: A specialized, digital-focused banking institution with simplified account opening and deposit limits. For example, an app-based mobile wallet bank.
TERM: hold delay factor: Multiplier relative to the fastest bank indicating how slowly an institution freezes suspicious funds. For example, taking 4 times longer to enact a regulatory hold.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.config_reader import draw_in_range, get_parameter_entry


@dataclass
class Bank:
    """
    WHAT: Represents one of the three banking institutions operating in the simulated town.
    It encapsulates the bank's classification, customer reach, account verification quality, and hold delay.
    These institutional parameters govern account opening likelihoods and response timing for account holders.

    WHY: Encapsulates institution-level customer reach, verification strength, and hold responsiveness.
    Housing institutional properties in a dataclass enables clean references across people and shop models.
    It provides an immutable descriptor used during world building and subsequent fraud investigations.

    Args:
        bank_index: Unique integer index (0, 1, or 2) identifying the bank in the simulation.
        kind: Institutional classification label ('large_public', 'private', or 'payments_bank').
        customer_share: Fractional proportion of the town population banking with this institution.
        weak_kyc_share: Probability that an account opened at this bank has unverified or weak identity checks.
        hold_delay_factor: Relative latency factor scaling how slowly the bank executes administrative holds.

    Returns:
        Initialized Bank dataclass instance.

    Example:
        # Create a large public bank instance with 45% customer share:
        # bank = Bank(0, "large_public", 0.4472, 0.1270, 3.0519)
        # assert bank.bank_index == 0 and bank.kind == "large_public"
    """

    bank_index: int
    kind: str
    customer_share: float
    weak_kyc_share: float
    hold_delay_factor: float


def draw_customer_shares(
    ranges: dict[str, Any], rng: np.random.Generator, range_name: str
) -> tuple[float, float, float]:
    """
    WHAT: Draws customer share values for the three bank kinds and normalizes them to sum to 1.
    It extracts configured market share ranges for large public, private, and payments banks from the config.
    The raw draws are summed and normalized so their cumulative customer proportions total exactly 100%.

    WHY: Each institution draws from its empirical band, and proportions must sum to the whole town.
    Normalizing bounded samples prevents the aggregate population share from under- or over-allocating citizens.
    Using explicit random draws reflects market variability across different municipal scenarios.

    Args:
        ranges: Configuration dictionary containing empirical market share parameter ranges.
        rng: Initialized NumPy random generator for deterministic sampling.
        range_name: Configuration split identifier such as 'training' or 'test'.

    Returns:
        Three-element tuple of normalized customer share floats for public, private, and payments banks.

    Example:
        # Draw normalized customer shares using seed 42 in the training split:
        # cfg = read_world_ranges()
        # shares = draw_customer_shares(cfg, np.random.default_rng(42), "training")
        # assert tuple(round(s, 4) for s in shares) == (0.4743, 0.3174, 0.2083)
    """
    # Retrieve configuration entries for each of the three banking tiers.
    pub_e = get_parameter_entry(ranges, "bank_customer_share_large_public")
    priv_e = get_parameter_entry(ranges, "bank_customer_share_private")
    pay_e = get_parameter_entry(ranges, "bank_customer_share_payments_bank")

    # Sample raw customer share values uniformly from their respective configured ranges.
    raw_pub = draw_in_range(pub_e, rng, range_name=range_name)
    raw_priv = draw_in_range(priv_e, rng, range_name=range_name)
    raw_pay = draw_in_range(pay_e, rng, range_name=range_name)

    # Calculate total raw weight and normalize each share to sum to 1.0 across the town.
    total = raw_pub + raw_priv + raw_pay
    return (raw_pub / total, raw_priv / total, raw_pay / total)


def draw_kyc_and_hold(
    ranges: dict[str, Any], rng: np.random.Generator, range_name: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    WHAT: Draws sorted weak KYC rates and hold latency factors across the three bank types.
    It samples three KYC vulnerability rates and assigns them according to institutional profiles.
    It also samples hold latency ratios, establishing large public banks as slowest and private as fastest.

    WHY: DECISION: Payments bank is assumed to have weakest KYC; private bank fastest hold response.
    Empirical banking data indicates digital-first onboarding suffers from higher synthetic identity rates.
    Conversely, well-resourced private institutions implement automated fraud holds significantly faster.

    Args:
        ranges: Master configuration dictionary containing KYC and hold ratio boundaries.
        rng: Initialized NumPy random generator for reproducible draws.
        range_name: Name of the active configuration split ('training', 'test', etc.).

    Returns:
        Two-element tuple containing a triplet of weak KYC shares and a triplet of hold delay factors.

    Example:
        # Sample KYC and hold values using seed 42 in training split:
        # cfg = read_world_ranges()
        # kyc, holds = draw_kyc_and_hold(cfg, np.random.default_rng(42), "training")
        # assert tuple(round(k, 4) for k in kyc) == (0.1107, 0.0322, 0.1468)
    """
    # Extract KYC configuration entry and sample three values to order by vulnerability tier.
    kyc_entry = get_parameter_entry(ranges, "bank_weak_kyc_account_share")
    kyc_vals = sorted([draw_in_range(kyc_entry, rng, range_name=range_name) for _ in range(3)])
    # Ordering: private smallest, large_public middle, payments_bank largest.
    kyc_shares = (kyc_vals[1], kyc_vals[0], kyc_vals[2])

    # Sample maximum hold ratio to establish the baseline delay range across institutions.
    hold_entry = get_parameter_entry(ranges, "slowest_to_fastest_bank_hold_ratio")
    max_ratio = draw_in_range(hold_entry, rng, range_name=range_name)
    mid_hold = float(rng.uniform(1.0, max_ratio))
    # Ordering: large_public slowest (max), private fastest (1.0), payments_bank in between.
    holds = (max_ratio, 1.0, mid_hold)
    return (kyc_shares, holds)


def build_banks(
    ranges: dict[str, Any], rng: np.random.Generator, range_name: str = "training"
) -> list[Bank]:
    """
    WHAT: Constructs the three Bank instances: large_public (0), private (1), and payments_bank (2).
    It orchestrates sampling of customer shares, KYC verification rigor, and hold reaction latencies.
    The resulting Bank instances form the institutional foundation of the simulated municipal economy.

    WHY: Establishes institutional parameters that govern account allocation and security checks.
    Downstream entity builders for people and shops require concrete bank objects for account assignment.
    Constructing all three banks together ensures consistent cross-institutional calibration.

    Args:
        ranges: Configuration dictionary containing world generation parameter ranges.
        rng: Initialized NumPy random generator ensuring deterministic initialization.
        range_name: Parameter interval split to sample from, defaulting to 'training'.

    Returns:
        List of three fully initialized Bank objects representing the municipal banking sector.

    Example:
        # Build three banks for the town using seed 42:
        # cfg = read_world_ranges()
        # banks = build_banks(cfg, np.random.default_rng(42), "training")
        # assert len(banks) == 3 and banks[0].kind == "large_public"
    """
    # Sample customer market share proportions across the three banking types.
    c_pub, c_priv, c_pay = draw_customer_shares(ranges, rng, range_name)
    # Sample weak KYC account shares and regulatory hold response multipliers.
    (k_pub, k_priv, k_pay), (h_pub, h_priv, h_pay) = draw_kyc_and_hold(ranges, rng, range_name)

    # Instantiate Bank objects with their respective indices, kinds, and operational parameters.
    bank_pub = Bank(0, "large_public", c_pub, k_pub, h_pub)
    bank_priv = Bank(1, "private", c_priv, k_priv, h_priv)
    bank_pay = Bank(2, "payments_bank", c_pay, k_pay, h_pay)
    return [bank_pub, bank_priv, bank_pay]

