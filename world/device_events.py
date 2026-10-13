"""
WHAT: Device event specifications, event constructor validation, and device setting sampler.
WHY: Models hardware turnover, app reinstalls, and OS updates with R5 fingerprint preservation (issues R5, A1, R2).
ISSUE IDs: R5 (OS updates and fingerprint preservation), A1 (device realism), R2 (background realism).
TERM: device event: A discrete lifecycle occurrence (phone replacement, reinstall, or OS update) altering device state.
TERM: device settings: Drawn simulation parameters governing replacement intervals, festival sales, and reinstalls.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.background_config import draw_whole_number
from world.config_reader import draw_in_range, get_parameter_entry

DEVICE_EVENT_KINDS: list[str] = ["phone_change", "reinstall", "os_update"]
VALID_RANGES: tuple[str, ...] = ("training", "test", "stress")
VALID_PLATFORMS: tuple[str, ...] = ("android", "ios")


@dataclass(frozen=True)
class DeviceEvent:
    """
    WHAT: Immutable record documenting a device state change for a simulated individual.
    WHY: Provides event feed records for Level 14 and fraud rule validation.
    """

    person_id: str
    minute: int
    kind: str
    counts_as_new_device: bool
    platform: str
    os_generation_before: int
    os_generation_after: int
    fingerprint_before: str
    fingerprint_after: str


def _validate_device_event_args(
    minute: int,
    kind: str,
    platform: str,
    os_generation_before: int,
    os_generation_after: int,
    fingerprint_before: str,
    fingerprint_after: str,
) -> None:
    """
    WHAT: Validates all arguments passed to make_device_event.
    WHY: Enforces R5 invariants, platform validity, non-negative integer minutes, and event kinds.
    """
    if not isinstance(minute, int) or isinstance(minute, bool) or minute < 0:
        raise ValueError(f"Event minute must be a non-negative integer, got {minute!r}")
    if kind not in DEVICE_EVENT_KINDS:
        raise ValueError(f"Unknown device event kind '{kind}'; expected one of {DEVICE_EVENT_KINDS}")
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Unknown platform '{platform}'; expected one of {VALID_PLATFORMS}")
    if kind == "os_update":
        if fingerprint_before != fingerprint_after:
            raise ValueError(
                f"os_update must keep the same fingerprint; got '{fingerprint_before}' vs '{fingerprint_after}'"
            )
        if os_generation_after != os_generation_before + 1:
            raise ValueError(
                f"os_update generation after ({os_generation_after}) must equal before ({os_generation_before}) + 1"
            )
    elif kind in ("phone_change", "reinstall") and fingerprint_before == fingerprint_after:
        raise ValueError(f"{kind} event must change device fingerprint; got identical '{fingerprint_before}'")


def make_device_event(
    person_id: str,
    minute: int,
    kind: str,
    platform: str,
    os_generation_before: int,
    os_generation_after: int,
    fingerprint_before: str,
    fingerprint_after: str,
) -> DeviceEvent:
    """
    WHAT: Validates and constructs an immutable DeviceEvent instance.
    WHY: Sets counts_as_new_device flag and guarantees consistency with R5 fingerprint invariants.
    """
    _validate_device_event_args(
        minute, kind, platform, os_generation_before, os_generation_after, fingerprint_before, fingerprint_after
    )
    counts_new = kind in ("phone_change", "reinstall")
    return DeviceEvent(
        person_id=person_id,
        minute=minute,
        kind=kind,
        counts_as_new_device=counts_new,
        platform=platform,
        os_generation_before=os_generation_before,
        os_generation_after=os_generation_after,
        fingerprint_before=fingerprint_before,
        fingerprint_after=fingerprint_after,
    )


@dataclass(frozen=True)
class DeviceSettings:
    """
    WHAT: Container holding sampled device lifespan and turnover parameters for a world.
    WHY: Encapsulates replacement frequency, festival sale uplift, sale days, and reinstall rate.
    """

    replacement_months: float
    sale_uplift: float
    sale_days: int
    reinstall_per_year: float


def _get_entry(ranges: dict[str, Any], key: str) -> dict[str, Any]:
    """
    WHAT: Retrieves parameter dictionary block from raw or nested config dictionary.
    WHY: Supports both validated config files and lightweight hand-built test dictionaries.
    """
    if "parameters" in ranges:
        return get_parameter_entry(ranges, key)
    if key in ranges:
        return ranges[key]
    raise ValueError(f"Unknown config parameter key: '{key}'")


def draw_device_settings(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    range_name: str = "training",
) -> DeviceSettings:
    """
    WHAT: Samples device settings from configured bounds in deterministic sequence.
    WHY: Supplies empirical and assumed parameters controlling hardware churn.
    """
    if range_name not in VALID_RANGES:
        raise ValueError(f"Unknown range name '{range_name}' for key 'phone_replacement_months'")

    e_repl = _get_entry(ranges, "phone_replacement_months")
    e_uplift = _get_entry(ranges, "phone_change_festival_sale_uplift")
    e_days = _get_entry(ranges, "phone_change_festival_sale_days")
    e_reinst = _get_entry(ranges, "app_reinstall_per_person_per_year")

    replacement_months = draw_in_range(e_repl, rng, range_name=range_name, param_name="phone_replacement_months")
    sale_uplift = draw_in_range(e_uplift, rng, range_name=range_name, param_name="phone_change_festival_sale_uplift")
    sale_days = draw_whole_number(e_days, rng, range_name=range_name, param_name="phone_change_festival_sale_days")
    reinstall_per_year = draw_in_range(
        e_reinst, rng, range_name=range_name, param_name="app_reinstall_per_person_per_year"
    )

    return DeviceSettings(
        replacement_months=replacement_months,
        sale_uplift=sale_uplift,
        sale_days=sale_days,
        reinstall_per_year=reinstall_per_year,
    )
