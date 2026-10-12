"""
WHAT: Failure decision evaluation engine determining payment outcomes per incident schedules and background rates.
WHY: Simulates stochastic disruptions (NPCI incidents, bank downtime, background dropouts) with realistic
     attribution (issue R7).
ISSUE IDs: R7 (failure domain tagging), E4 (bank downtime differences), R2 (background realism rates).
TERM: active incident: An incident whose time window [start_minute, start_minute + duration_minutes)
      includes the query minute.
TERM: degradation: A temporary drop in transaction success probability without complete service blackout.
TERM: structural choice: A modeling split (e.g. 3-way technical decline allocation) chosen because empirical
      sources do not split it.

Nothing here calls make_payment or changes any ledger; this module is a pure failure evaluation model.
"""

import numpy as np

from world.failures import FailureSettings
from world.incidents import Incident


def _validate_decision_inputs(minute: int, from_bank_index: int, to_bank_index: int) -> None:
    """
    WHAT: Validates that minute and bank indexes are non-negative integer values.
    WHY: Defends against invalid temporal coordinates or non-existent bank identifiers.
    """
    if isinstance(minute, bool) or not isinstance(minute, int) or minute < 0:
        raise ValueError("minute must be a non-negative integer")
    if isinstance(from_bank_index, bool) or not isinstance(from_bank_index, int) or from_bank_index < 0:
        raise ValueError("from_bank_index must be a non-negative integer")
    if isinstance(to_bank_index, bool) or not isinstance(to_bank_index, int) or to_bank_index < 0:
        raise ValueError("to_bank_index must be a non-negative integer")


def active_incidents(minute: int, incidents: list[Incident]) -> list[Incident]:
    """
    WHAT: Filters a schedule to incidents active at the specified minute.
    WHY: Identifies all concurrent network or bank disruptions overlapping the payment timestamp.
    """
    active: list[Incident] = []
    for inc in incidents:
        # Window start is inclusive, end is exclusive.
        if inc.start_minute <= minute < inc.start_minute + inc.duration_minutes:
            active.append(inc)
    return active


def _eval_npci_incidents(active: list[Incident], rng: np.random.Generator) -> tuple[str, str] | None:
    """
    WHAT: Evaluates active NPCI incidents in order against uniform random draws.
    WHY: NPCI central switch failures take operational precedence over downstream bank outages.
    """
    for inc in active:
        if inc.kind == "npci_incident":
            u = float(rng.uniform(0.0, 1.0))
            if u >= inc.success_rate:
                return ("npci_central", "npci_incident")
    return None


def _eval_bank_downtimes(
    active: list[Incident], from_bank_index: int, to_bank_index: int, rng: np.random.Generator
) -> tuple[str, str] | None:
    """
    WHAT: Evaluates active bank downtimes affecting sender or receiver banks.
    WHY: Models bank-specific ledger/core-banking degradation with sender precedence if self-transfer.
    """
    for inc in active:
        if inc.kind == "bank_downtime":
            affected_domain: str | None = None
            if inc.bank_index == from_bank_index:
                # Same bank on both ends counts once as sender.
                affected_domain = "sender_bank"
            elif inc.bank_index == to_bank_index:
                affected_domain = "receiver_bank"
            if affected_domain is not None:
                u = float(rng.uniform(0.0, 1.0))
                if u >= inc.success_rate:
                    return (affected_domain, "bank_downtime")
    return None


def _eval_background_failures(settings: FailureSettings, rng: np.random.Generator) -> tuple[str, str] | None:
    """
    WHAT: Evaluates background failures (wrong PIN, network drop, technical decline) via a single uniform draw.
    WHY: Partitioning the unit interval ensures mutually exclusive background outcomes per payment.
    """
    u = float(rng.uniform(0.0, 1.0))
    p = settings.wrong_pin_rate
    n = settings.network_failure_rate
    t = settings.technical_decline_rate
    if u < p:
        return ("customer", "wrong_pin")
    if u < p + n:
        return ("app_or_network", "network_drop")
    if u < p + n + t:
        # DECISION: the three-way split of technical declines is a structural choice
        # (no source gives a split), not a research number.
        branch = int(rng.integers(0, 3))
        if branch == 0:
            return ("npci_central", "technical_decline")
        if branch == 1:
            return ("sender_bank", "technical_decline")
        return ("receiver_bank", "technical_decline")
    return None


def decide_failure(
    minute: int,
    from_bank_index: int,
    to_bank_index: int,
    incidents: list[Incident],
    settings: FailureSettings,
    rng: np.random.Generator,
) -> tuple[str, str] | None:
    """
    WHAT: Decides whether a payment attempt fails and returns (failure_domain, reason) or None.
    WHY: Provides the core failure attribution logic across central, bank, and background channels.
    """
    _validate_decision_inputs(minute, from_bank_index, to_bank_index)
    active = active_incidents(minute, incidents)
    npci_fail = _eval_npci_incidents(active, rng)
    if npci_fail is not None:
        return npci_fail
    bank_fail = _eval_bank_downtimes(active, from_bank_index, to_bank_index, rng)
    if bank_fail is not None:
        return bank_fail
    return _eval_background_failures(settings, rng)
