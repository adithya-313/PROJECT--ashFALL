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

    Args:
        minute: Integer representing elapsed minutes from simulation start.
        from_bank_index: Non-negative integer index of sending bank.
        to_bank_index: Non-negative integer index of receiving bank.

    Returns:
        None.

    Raises:
        ValueError: If minute, from_bank_index, or to_bank_index is boolean, non-integer, or negative.

    Example:
        >>> from world.failure_decision import _validate_decision_inputs
        >>> _validate_decision_inputs(60, 0, 1)
    """
    # Ensure minute coordinate is a non-negative integer.
    if isinstance(minute, bool) or not isinstance(minute, int) or minute < 0:
        raise ValueError("minute must be a non-negative integer")
    # Ensure sending bank index is a valid non-negative integer.
    if isinstance(from_bank_index, bool) or not isinstance(from_bank_index, int) or from_bank_index < 0:
        raise ValueError("from_bank_index must be a non-negative integer")
    # Ensure receiving bank index is a valid non-negative integer.
    if isinstance(to_bank_index, bool) or not isinstance(to_bank_index, int) or to_bank_index < 0:
        raise ValueError("to_bank_index must be a non-negative integer")


def active_incidents(minute: int, incidents: list[Incident]) -> list[Incident]:
    """
    WHAT: Filters a schedule to incidents active at the specified minute.
    WHY: Identifies all concurrent network or bank disruptions overlapping the payment timestamp.

    Args:
        minute: Non-negative integer minute coordinate being queried.
        incidents: List of scheduled Incident records to filter.

    Returns:
        List of Incident instances active at the specified minute timestamp.

    Example:
        >>> from world.incidents import Incident
        >>> from world.failure_decision import active_incidents
        >>> inc = Incident("npci_incident", 0, 60, None, 0.5)
        >>> len(active_incidents(30, [inc]))
        1
        >>> len(active_incidents(90, [inc]))
        0
    """
    active: list[Incident] = []
    # Filter the incident list to events whose time windows span the target minute.
    for inc in incidents:
        # Window start is inclusive, end is exclusive.
        if inc.start_minute <= minute < inc.start_minute + inc.duration_minutes:
            active.append(inc)
    return active


def _eval_npci_incidents(active: list[Incident], rng: np.random.Generator) -> tuple[str, str] | None:
    """
    WHAT: Evaluates active NPCI incidents in order against uniform random draws.
    WHY: NPCI central switch failures take operational precedence over downstream bank outages.

    Args:
        active: List of currently active Incident records.
        rng: Pseudo-random number generator for uniform draws.

    Returns:
        Tuple ("npci_central", "npci_incident") if a central failure occurs, or None if successful.

    Example:
        >>> import numpy as np
        >>> from world.incidents import Incident
        >>> from world.failure_decision import _eval_npci_incidents
        >>> rng = np.random.default_rng(seed=42)
        >>> inc = Incident("npci_incident", 0, 60, None, 0.0)
        >>> _eval_npci_incidents([inc], rng)
        ('npci_central', 'npci_incident')
    """
    # Check active disruptions for central NPCI switch failures.
    for inc in active:
        if inc.kind == "npci_incident":
            # Sample against degraded success rate to test if central routing drops payment.
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

    Args:
        active: List of currently active Incident records.
        from_bank_index: Sending bank index.
        to_bank_index: Receiving bank index.
        rng: Pseudo-random number generator for probability tests.

    Returns:
        Tuple (affected_domain, "bank_downtime") if bank downtime triggered failure, or None.

    Example:
        >>> import numpy as np
        >>> from world.incidents import Incident
        >>> from world.failure_decision import _eval_bank_downtimes
        >>> rng = np.random.default_rng(seed=42)
        >>> inc = Incident("bank_downtime", 0, 60, 1, 0.0)
        >>> _eval_bank_downtimes([inc], 1, 2, rng)
        ('sender_bank', 'bank_downtime')
    """
    # Check active disruptions for institution-specific downtime on participating banks.
    for inc in active:
        if inc.kind == "bank_downtime":
            affected_domain: str | None = None
            if inc.bank_index == from_bank_index:
                # Same bank on both ends counts once as sender.
                affected_domain = "sender_bank"
            elif inc.bank_index == to_bank_index:
                affected_domain = "receiver_bank"
            # Test against degraded bank success rate to determine if payment is rejected.
            if affected_domain is not None:
                u = float(rng.uniform(0.0, 1.0))
                if u >= inc.success_rate:
                    return (affected_domain, "bank_downtime")
    return None


def _eval_background_failures(settings: FailureSettings, rng: np.random.Generator) -> tuple[str, str] | None:
    """
    WHAT: Evaluates background failures (wrong PIN, network drop, technical decline) via a single uniform draw.
    WHY: Partitioning the unit interval ensures mutually exclusive background outcomes per payment.

    Args:
        settings: FailureSettings container with wrong PIN, network drop, and decline probabilities.
        rng: Pseudo-random number generator.

    Returns:
        Tuple (failure_domain, reason) describing the failure, or None for clean pass-through.

    Example:
        >>> import numpy as np
        >>> from world.failures import FailureSettings
        >>> from world.failure_decision import _eval_background_failures
        >>> rng = np.random.default_rng(seed=42)
        >>> s = FailureSettings(wrong_pin_rate=1.0, network_failure_rate=0.0, technical_decline_rate=0.0)
        >>> _eval_background_failures(s, rng)
        ('customer', 'wrong_pin')
    """
    # Draw a single uniform float to partition mutually exclusive background outcomes.
    u = float(rng.uniform(0.0, 1.0))
    p = settings.wrong_pin_rate
    n = settings.network_failure_rate
    t = settings.technical_decline_rate
    # Test whether draw falls into wrong PIN, network drop, or technical decline partition.
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

    Args:
        minute: Integer simulation minute timestamp.
        from_bank_index: Sending bank integer index.
        to_bank_index: Receiving bank integer index.
        incidents: List of scheduled Incident disruptions.
        settings: FailureSettings container holding background rates.
        rng: Pseudo-random number generator for stochastic evaluation.

    Returns:
        Tuple (failure_domain, reason) if transaction fails, or None if transaction succeeds.

    Raises:
        ValueError: If minute or bank indices fail validation constraints.

    Example:
        >>> import numpy as np
        >>> from world.failures import FailureSettings
        >>> from world.failure_decision import decide_failure
        >>> rng = np.random.default_rng(seed=42)
        >>> s = FailureSettings(wrong_pin_rate=0.0, network_failure_rate=0.0, technical_decline_rate=0.0)
        >>> decide_failure(0, 0, 1, [], s, rng) is None
        True
    """
    # Validate input coordinates and identify currently active disruptions.
    _validate_decision_inputs(minute, from_bank_index, to_bank_index)
    active = active_incidents(minute, incidents)
    # Evaluate failures in operational hierarchy: central NPCI first, then banks, then background.
    npci_fail = _eval_npci_incidents(active, rng)
    if npci_fail is not None:
        return npci_fail
    bank_fail = _eval_bank_downtimes(active, from_bank_index, to_bank_index, rng)
    if bank_fail is not None:
        return bank_fail
    return _eval_background_failures(settings, rng)
