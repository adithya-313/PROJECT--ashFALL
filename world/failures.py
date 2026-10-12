"""
WHAT: Payment failure domain constants, failure settings container, and PaymentEvent records.
WHY: Models attribution of payment failures across five realistic domains and encapsulates
     background probability settings for transaction declines (issues R7, E4, R2).
ISSUE IDs: R7 (failure domain classification), E4 (bank differences), R2 (background realism).
TERM: failure domain: The operational zone responsible for a payment rejection (central, banks, app, or user).
TERM: payment event: An immutable record capturing transaction details, status, diagnostic reason, and domain.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.config_reader import draw_in_range, get_parameter_entry

FAILURE_DOMAINS: list[str] = [
    "npci_central",
    "sender_bank",
    "receiver_bank",
    "app_or_network",
    "customer",
]

REASON_NPCI_INCIDENT: str = "npci_incident"
REASON_BANK_DOWNTIME: str = "bank_downtime"
REASON_WRONG_PIN: str = "wrong_pin"
REASON_NETWORK_DROP: str = "network_drop"
REASON_TECHNICAL_DECLINE: str = "technical_decline"
REASON_OK: str = "ok"

VALID_RANGES: tuple[str, ...] = ("training", "test", "stress")


@dataclass(frozen=True)
class PaymentEvent:
    """
    WHAT: Immutable transaction record documenting outcome, financial routing, and domain.
    WHY: Serves as the standardized event emitted for Level 14 feed ingestion and failure audit.
    """

    minute: int
    from_account: str
    to_account: str
    amount_paise: int
    from_bank_index: int
    to_bank_index: int
    status: str
    reason: str
    failure_domain: str | None = None


def _validate_event_args(
    minute: int,
    amount_paise: int,
    from_bank_index: int,
    to_bank_index: int,
    failure_domain: str | None,
) -> None:
    """
    WHAT: Validates minute, amount, bank indices, and failure domain boundaries.
    WHY: Centralizes parameter validation for PaymentEvent generation.
    """
    if not isinstance(minute, int) or isinstance(minute, bool) or minute < 0:
        raise ValueError(f"Minute must be a non-negative integer, got {minute!r}")
    if not isinstance(amount_paise, int) or isinstance(amount_paise, bool) or amount_paise <= 0:
        raise ValueError(f"Amount in paise must be a positive integer, got {amount_paise!r}")
    if not isinstance(from_bank_index, int) or isinstance(from_bank_index, bool) or from_bank_index < 0:
        raise ValueError(f"from_bank_index must be a non-negative integer, got {from_bank_index!r}")
    if not isinstance(to_bank_index, int) or isinstance(to_bank_index, bool) or to_bank_index < 0:
        raise ValueError(f"to_bank_index must be a non-negative integer, got {to_bank_index!r}")
    if failure_domain is not None and failure_domain not in FAILURE_DOMAINS:
        raise ValueError(f"Unknown failure_domain '{failure_domain}'; expected one of {FAILURE_DOMAINS}")


def make_event(
    minute: int, from_account: str, to_account: str, amount_paise: int,
    from_bank_index: int, to_bank_index: int, failure_domain: str | None, reason: str,
) -> PaymentEvent:
    """
    WHAT: Creates a validated PaymentEvent record with consistent status and reason tagging.
    WHY: Enforces that success/failure invariants and domain categorizations remain strictly valid.
    """
    _validate_event_args(minute, amount_paise, from_bank_index, to_bank_index, failure_domain)
    status = "success" if failure_domain is None else "failed"
    if status == "success" and reason != REASON_OK:
        raise ValueError(f"Successful payments must have reason 'ok', got '{reason}'")
    if status == "failed" and reason == REASON_OK:
        raise ValueError("Failed payments cannot have reason 'ok'")

    return PaymentEvent(
        minute=minute, from_account=from_account, to_account=to_account, amount_paise=amount_paise,
        from_bank_index=from_bank_index, to_bank_index=to_bank_index, status=status,
        reason=reason, failure_domain=failure_domain,
    )


@dataclass(frozen=True)
class FailureSettings:
    """
    WHAT: Container holding drawn background failure probabilities for a simulated world.
    WHY: Centralizes wrong PIN, network drop, and technical decline rates sampled from config.
    """

    wrong_pin_rate: float
    network_failure_rate: float
    technical_decline_rate: float


def draw_failure_settings(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    range_name: str = "training",
) -> FailureSettings:
    """
    WHAT: Samples failure background probability rates from configuration ranges.
    WHY: Supplies randomized per-world rates for PIN errors, network drops, and technical declines.
    """
    if range_name not in VALID_RANGES:
        raise ValueError(f"Unknown range name '{range_name}'; expected one of {VALID_RANGES}")

    pin_entry = get_parameter_entry(ranges, "customer_wrong_pin_rate")
    net_entry = get_parameter_entry(ranges, "app_network_failure_rate")
    tech_entry = get_parameter_entry(ranges, "technical_decline_rate")

    wrong_pin = draw_in_range(pin_entry, rng, range_name=range_name, param_name="customer_wrong_pin_rate")
    network = draw_in_range(net_entry, rng, range_name=range_name, param_name="app_network_failure_rate")
    technical = draw_in_range(tech_entry, rng, range_name=range_name, param_name="technical_decline_rate")

    return FailureSettings(
        wrong_pin_rate=wrong_pin,
        network_failure_rate=network,
        technical_decline_rate=technical,
    )
