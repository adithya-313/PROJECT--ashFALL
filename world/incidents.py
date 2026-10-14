"""
WHAT: Incident schedule modeling for NPCI central disruptions and individual bank downtimes.
WHY: Generates stochastic downtime intervals with partial success rates rather than total blackouts (issue R7).
ISSUE IDs: R7 (outages and downtime simulation), E4 (bank differences), R2 (background realism).
TERM: incident: A discrete temporal window during which payment routing experiences degraded success rates.
TERM: bank downtime: An institution-specific disruption impacting transactions sent from or received by that bank.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.background_config import draw_whole_number
from world.config_reader import draw_in_range, get_parameter_entry

VALID_RANGES: tuple[str, ...] = ("training", "test", "stress")


@dataclass(frozen=True)
class Incident:
    """
    WHAT: Immutable incident specification defining kind, timing, institution, and degraded rate.
    WHY: Encapsulates partial service degradation windows across the simulated timeline.
    """

    kind: str
    start_minute: int
    duration_minutes: int
    bank_index: int | None
    success_rate: float


def _validate_incident_schedule_args(n_days: int, n_banks: int, range_name: str) -> None:
    """
    WHAT: Validates world duration, bank count, and range split parameters.
    WHY: Rejects invalid or corrupt simulation arguments with plain descriptive messages.

    Args:
        n_days: Integer duration of the simulated world in full days (must be >= 2).
        n_banks: Integer count of participating banks (must be >= 1).
        range_name: Target dataset split name (one of "training", "test", "stress").

    Returns:
        None.

    Raises:
        ValueError: If n_days < 2, n_banks < 1, or range_name is unrecognized.

    Example:
        >>> from world.incidents import _validate_incident_schedule_args
        >>> _validate_incident_schedule_args(30, 2, "training")
    """
    # Verify world length spans at least two full calendar days.
    if not isinstance(n_days, int) or isinstance(n_days, bool) or n_days < 2:
        raise ValueError(f"n_days must be an integer >= 2, got {n_days!r}")
    # Ensure at least one participating bank exists in the simulation.
    if not isinstance(n_banks, int) or isinstance(n_banks, bool) or n_banks < 1:
        raise ValueError(f"n_banks must be an integer >= 1, got {n_banks!r}")
    # Validate range split environment against accepted options.
    if range_name not in VALID_RANGES:
        raise ValueError(f"Unknown range name '{range_name}'; expected one of {VALID_RANGES}")


def _incident_sort_key(inc: Incident) -> tuple[int, str, int]:
    """
    WHAT: Computes the canonical sorting key for an incident.
    WHY: Orders incidents chronologically by start minute, then kind, with None bank first.

    Args:
        inc: Incident dataclass instance whose sorting coordinates are extracted.

    Returns:
        Tuple of (start_minute, kind, bank_key) used for chronological ordering.

    Example:
        >>> from world.incidents import Incident, _incident_sort_key
        >>> inc = Incident("bank_downtime", 120, 30, 1, 0.8)
        >>> _incident_sort_key(inc)
        (120, 'bank_downtime', 1)
    """
    bank_key = -1 if inc.bank_index is None else inc.bank_index
    return (inc.start_minute, inc.kind, bank_key)


def _draw_npci_incidents(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    n_days: int,
    range_name: str,
) -> list[Incident]:
    """
    WHAT: Generates central NPCI disruption intervals scaled to world duration.
    WHY: Simulates rare nationwide payment routing degradation across all banks.

    Args:
        ranges: Configuration dictionary holding failure parameters.
        rng: Pseudo-random number generator for reproducible sampling.
        n_days: Total duration of the simulation in full days.
        range_name: Target dataset split name containing parameter intervals.

    Returns:
        List of Incident instances representing nationwide central switch disruptions.

    Example:
        >>> from core.config_validator import load_yaml
        >>> from world.incidents import _draw_npci_incidents
        >>> import numpy as np
        >>> rng = np.random.default_rng(seed=0)
        >>> cfg = load_yaml("config/failure_realism.yaml")
        >>> incs = _draw_npci_incidents(cfg, rng, 60, "training")
        >>> len(incs)
        4
    """
    # Look up parameter range configurations for central NPCI disruption events.
    count_entry = get_parameter_entry(ranges, "npci_incident_count_per_30_days")
    dur_entry = get_parameter_entry(ranges, "npci_incident_duration_minutes")
    rate_entry = get_parameter_entry(ranges, "npci_incident_success_rate")

    raw_count = draw_whole_number(count_entry, rng, range_name=range_name)
    # Python round() is used on purpose to scale monthly rate to world days.
    count = round(raw_count * n_days / 30)

    incidents: list[Incident] = []
    total_minutes = n_days * 1440
    # Sample each central incident window with random start minute and degraded success rate.
    for _ in range(count):
        duration = draw_whole_number(dur_entry, rng, range_name=range_name)
        rate = draw_in_range(rate_entry, rng, range_name=range_name)
        max_start = total_minutes - duration
        start = int(rng.integers(0, max_start + 1))
        # The window start is inclusive and the end (start + duration) is exclusive.
        incidents.append(Incident("npci_incident", start, duration, None, rate))
    return incidents


def _draw_bank_downtimes(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    n_days: int,
    n_banks: int,
    range_name: str,
) -> list[Incident]:
    """
    WHAT: Generates individual bank maintenance and outage windows.
    WHY: Simulates institution-specific core banking system downtimes.

    Args:
        ranges: Configuration dictionary holding failure realism parameters.
        rng: Pseudo-random number generator for reproducible sampling.
        n_days: Total duration of the simulation in full days.
        n_banks: Total number of active banks in the simulation.
        range_name: Target dataset split name containing parameter intervals.

    Returns:
        List of Incident instances representing institution-specific core banking downtimes.

    Example:
        >>> from core.config_validator import load_yaml
        >>> from world.incidents import _draw_bank_downtimes
        >>> import numpy as np
        >>> rng = np.random.default_rng(seed=42)
        >>> cfg = load_yaml("config/failure_realism.yaml")
        >>> incs = _draw_bank_downtimes(cfg, rng, 30, 2, "training")
        >>> len(incs)
        7
    """
    # Look up parameter range configurations for institution downtime windows.
    count_entry = get_parameter_entry(ranges, "bank_downtime_count_per_bank_per_30_days")
    dur_entry = get_parameter_entry(ranges, "bank_downtime_duration_minutes")
    rate_entry = get_parameter_entry(ranges, "bank_downtime_success_rate")

    incidents: list[Incident] = []
    total_minutes = n_days * 1440
    # Sample downtime intervals independently for each participating bank.
    for b_idx in range(n_banks):
        raw_count = draw_whole_number(count_entry, rng, range_name=range_name)
        # Python round() is used on purpose to scale monthly rate to world days.
        count = round(raw_count * n_days / 30)
        for _ in range(count):
            duration = draw_whole_number(dur_entry, rng, range_name=range_name)
            rate = draw_in_range(rate_entry, rng, range_name=range_name)
            max_start = total_minutes - duration
            start = int(rng.integers(0, max_start + 1))
            # The window start is inclusive and the end (start + duration) is exclusive.
            incidents.append(Incident("bank_downtime", start, duration, b_idx, rate))
    return incidents


def draw_incident_schedule(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    n_days: int,
    n_banks: int,
    range_name: str = "training",
) -> list[Incident]:
    """
    WHAT: Samples a chronological schedule of NPCI and bank downtime incidents.
    WHY: Coordinates system-wide and institution-level degradation without fixed calendar days.

    Args:
        ranges: Configuration dictionary holding failure realism parameters.
        rng: Pseudo-random number generator for reproducible sampling.
        n_days: Total duration of the simulation in full days.
        n_banks: Total number of active banks in the simulation.
        range_name: Target dataset split name (default "training").

    Returns:
        Chronologically sorted list of Incident instances across central switch and bank domains.

    Raises:
        ValueError: If n_days, n_banks, or range_name violate validation constraints.

    Example:
        >>> from core.config_validator import load_yaml
        >>> from world.incidents import draw_incident_schedule
        >>> import numpy as np
        >>> rng = np.random.default_rng(seed=42)
        >>> cfg = load_yaml("config/failure_realism.yaml")
        >>> sched = draw_incident_schedule(cfg, rng, 30, 2, "training")
        >>> len(sched)
        7
    """
    # Validate simulation arguments before sampling incident intervals.
    _validate_incident_schedule_args(n_days, n_banks, range_name)
    # Sample central NPCI disruptions and institution-specific bank downtimes.
    npci_incs = _draw_npci_incidents(ranges, rng, n_days, range_name)
    bank_incs = _draw_bank_downtimes(ranges, rng, n_days, n_banks, range_name)
    # Combine and sort all disruption windows chronologically.
    all_incs = npci_incs + bank_incs
    all_incs.sort(key=_incident_sort_key)
    return all_incs
