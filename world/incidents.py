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
    """
    if not isinstance(n_days, int) or isinstance(n_days, bool) or n_days < 2:
        raise ValueError(f"n_days must be an integer >= 2, got {n_days!r}")
    if not isinstance(n_banks, int) or isinstance(n_banks, bool) or n_banks < 1:
        raise ValueError(f"n_banks must be an integer >= 1, got {n_banks!r}")
    if range_name not in VALID_RANGES:
        raise ValueError(f"Unknown range name '{range_name}'; expected one of {VALID_RANGES}")


def _incident_sort_key(inc: Incident) -> tuple[int, str, int]:
    """
    WHAT: Computes the canonical sorting key for an incident.
    WHY: Orders incidents chronologically by start minute, then kind, with None bank first.
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
    """
    count_entry = get_parameter_entry(ranges, "npci_incident_count_per_30_days")
    dur_entry = get_parameter_entry(ranges, "npci_incident_duration_minutes")
    rate_entry = get_parameter_entry(ranges, "npci_incident_success_rate")

    raw_count = draw_whole_number(count_entry, rng, range_name=range_name)
    # Python round() is used on purpose to scale monthly rate to world days.
    count = round(raw_count * n_days / 30)

    incidents: list[Incident] = []
    total_minutes = n_days * 1440
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
    """
    count_entry = get_parameter_entry(ranges, "bank_downtime_count_per_bank_per_30_days")
    dur_entry = get_parameter_entry(ranges, "bank_downtime_duration_minutes")
    rate_entry = get_parameter_entry(ranges, "bank_downtime_success_rate")

    incidents: list[Incident] = []
    total_minutes = n_days * 1440
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
    """
    _validate_incident_schedule_args(n_days, n_banks, range_name)
    npci_incs = _draw_npci_incidents(ranges, rng, n_days, range_name)
    bank_incs = _draw_bank_downtimes(ranges, rng, n_days, n_banks, range_name)
    all_incs = npci_incs + bank_incs
    all_incs.sort(key=_incident_sort_key)
    return all_incs
