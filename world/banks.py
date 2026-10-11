"""
WHAT: Bank entity definition and ecosystem generation across three financial institutions.
WHY: Models the tripartite banking landscape (large public, private, and payments bank) with differing
     market shares, identity verification (KYC) rigour, and regulatory hold reaction latencies.
ISSUE IDs: E4 (bank differences).
TERM: KYC: Know Your Customer identity verification processes required when opening bank accounts.
TERM: hold delay factor: Multiplier relative to the fastest bank indicating how slowly a bank executes holds.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

from world.config_reader import draw_in_range, get_parameter_entry


@dataclass
class Bank:
    """
    WHAT: Represents one of the three banking institutions in the simulated town.
    WHY: Encapsulates institution-level customer reach, verification strength, and hold responsiveness.
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
    WHY: Each institution draws from its empirical band, and proportions must sum to the whole town.
    """
    pub_e = get_parameter_entry(ranges, "bank_customer_share_large_public")
    priv_e = get_parameter_entry(ranges, "bank_customer_share_private")
    pay_e = get_parameter_entry(ranges, "bank_customer_share_payments_bank")

    raw_pub = draw_in_range(pub_e, rng, range_name=range_name)
    raw_priv = draw_in_range(priv_e, rng, range_name=range_name)
    raw_pay = draw_in_range(pay_e, rng, range_name=range_name)

    total = raw_pub + raw_priv + raw_pay
    return (raw_pub / total, raw_priv / total, raw_pay / total)


def draw_kyc_and_hold(
    ranges: dict[str, Any], rng: np.random.Generator, range_name: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    WHAT: Draws sorted weak KYC rates and hold latency factors across the three bank types.
    WHY: DECISION: Payments bank is assumed to have weakest KYC; private bank fastest hold response.
    """
    kyc_entry = get_parameter_entry(ranges, "bank_weak_kyc_account_share")
    kyc_vals = sorted([draw_in_range(kyc_entry, rng, range_name=range_name) for _ in range(3)])
    # Ordering: private smallest, large_public middle, payments_bank largest.
    kyc_shares = (kyc_vals[1], kyc_vals[0], kyc_vals[2])

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
    WHY: Establishes institutional parameters that govern account allocation and security checks.
    """
    c_pub, c_priv, c_pay = draw_customer_shares(ranges, rng, range_name)
    (k_pub, k_priv, k_pay), (h_pub, h_priv, h_pay) = draw_kyc_and_hold(ranges, rng, range_name)

    bank_pub = Bank(0, "large_public", c_pub, k_pub, h_pub)
    bank_priv = Bank(1, "private", c_priv, k_priv, h_priv)
    bank_pay = Bank(2, "payments_bank", c_pay, k_pay, h_pay)
    return [bank_pub, bank_priv, bank_pay]
