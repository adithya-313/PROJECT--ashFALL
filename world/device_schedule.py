"""
WHAT: Generator for per-world device event schedules (phone changes, reinstalls, and OS updates).
WHY: Produces realistic hardware velocity and adoption curves without touching ledgers or
     calling make_payment (issues R5, A1, R2).
ISSUE IDs: R5 (OS updates and fingerprint preservation), A1 (device realism), R2 (background realism).
TERM: adoption curve: Temporal rollout trajectory modeled via beta(2,2) bell-shaped distribution.
TERM: device schedule: Chronologically sorted series of device events across a simulated timeline.
"""

from typing import Any

import numpy as np

from world.background_config import draw_whole_number
from world.config_reader import draw_in_range
from world.device_events import DeviceEvent, DeviceSettings, _get_entry, draw_device_settings, make_device_event
from world.festivals import Festival
from world.placeholders import OS_GENERATIONS

MINUTES_PER_DAY: int = 1440
DAYS_PER_MONTH: int = 30
DAYS_PER_YEAR: int = 365
VALID_RANGES: tuple[str, ...] = ("training", "test", "stress")
VALID_PLATFORMS: tuple[str | None, ...] = (None, "android", "ios")


def _validate_schedule_args(
    ranges: dict[str, Any],
    people: list[Any],
    festivals: list[Festival],
    n_days: int,
    range_name: str,
    wave_platform: str | None,
) -> None:
    """
    WHAT: Validates all inputs to draw_device_event_schedule.
    WHY: Rejects invalid horizons, empty populations, unsupported platforms, and unknown splits.
    """
    if not isinstance(n_days, int) or isinstance(n_days, bool) or n_days < 2:
        raise ValueError(f"n_days must be an integer >= 2, got {n_days!r}")
    if len(people) == 0:
        raise ValueError("people list cannot be empty")
    if len(festivals) == 0:
        raise ValueError("festivals list cannot be empty")
    if wave_platform not in VALID_PLATFORMS:
        raise ValueError(f"Unknown wave_platform '{wave_platform}'; expected one of {VALID_PLATFORMS}")
    if range_name not in VALID_RANGES:
        raise ValueError(f"Unknown range name '{range_name}'; expected one of {VALID_RANGES}")

    wave_len_entry = _get_entry(ranges, "os_wave_length_days")
    upper_bound = wave_len_entry[range_name][1]
    if n_days < upper_bound:
        raise ValueError(
            f"n_days ({n_days}) is smaller than upper bound of os_wave_length_days ({upper_bound}) "
            f"for range '{range_name}'"
        )


def _draw_unique_fingerprint(
    rng: np.random.Generator,
    platform: str,
    os_gen: int,
    seen_fingerprints: set[str],
) -> str:
    """
    WHAT: Draws a unique 8-character hex hardware token and creates a formatted fingerprint string.
    WHY: Enforces that newly assigned hardware fingerprints never collide across the simulation.
    """
    for _ in range(20):
        token = int(rng.integers(0, 0xFFFFFFFF, endpoint=True))
        candidate = f"fp_{platform}_v{os_gen}_{token:08x}"
        if candidate not in seen_fingerprints:
            seen_fingerprints.add(candidate)
            return candidate
    raise RuntimeError(f"Could not generate a unique device fingerprint after 20 attempts for platform '{platform}'")


def _schedule_single_wave(
    people: list[Any],
    start_day: int,
    length: int,
    share: float,
    wave_platform: str | None,
    scheduled_waves: dict[tuple[int, int], int],
    rng: np.random.Generator,
) -> None:
    """
    WHAT: Assigns adoption times across eligible people for one OS rollout wave.
    WHY: Implements beta(2,2) rollout bell-curve timing per person.
    """
    for i, person in enumerate(people):
        if wave_platform is not None and person.device.platform != wave_platform:
            continue
        if float(rng.random()) < share:
            # beta(2, 2) is a structural choice for a gradual bell-shaped adoption curve.
            offset = min(int(length * rng.beta(2.0, 2.0)), length - 1)
            update_day = start_day + offset
            update_minute = update_day * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY))
            if (i, update_day) not in scheduled_waves:
                scheduled_waves[(i, update_day)] = update_minute


def _schedule_os_waves(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    people: list[Any],
    n_days: int,
    range_name: str,
    wave_platform: str | None,
) -> dict[tuple[int, int], int]:
    """
    WHAT: Pre-computes per-person OS wave rollout timing across all configured waves.
    WHY: Models gradual adoption waves with beta(2,2) bell-shaped density across the wave window.
    """
    wave_cnt_entry = _get_entry(ranges, "os_wave_count_per_world")
    wave_len_entry = _get_entry(ranges, "os_wave_length_days")
    share_entry = _get_entry(ranges, "os_wave_final_share_of_eligible")

    wave_count = draw_whole_number(wave_cnt_entry, rng, range_name=range_name, param_name="os_wave_count_per_world")
    scheduled_waves: dict[tuple[int, int], int] = {}

    for _ in range(wave_count):
        length = draw_whole_number(wave_len_entry, rng, range_name=range_name, param_name="os_wave_length_days")
        share = draw_in_range(share_entry, rng, range_name=range_name, param_name="os_wave_final_share_of_eligible")
        start_day = int(rng.integers(0, n_days - length, endpoint=True))
        _schedule_single_wave(people, start_day, length, share, wave_platform, scheduled_waves, rng)
    return scheduled_waves


def _step_phone_change(
    person: Any,
    idx: int,
    day: int,
    current_os: list[int],
    current_fp: list[str],
    seen_fps: set[str],
    rng: np.random.Generator,
) -> DeviceEvent:
    """
    WHAT: Generates a phone change event with a new fingerprint and updated OS generation.
    WHY: Represents physical handset replacement to current newest OS generation.
    """
    plat = person.device.platform
    minute = day * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY))
    # A new phone comes with the newest OS (OS_GENERATIONS): structural choice.
    new_fp = _draw_unique_fingerprint(rng, plat, OS_GENERATIONS, seen_fps)
    ev = make_device_event(
        person.person_id, minute, "phone_change", plat, current_os[idx], OS_GENERATIONS, current_fp[idx], new_fp
    )
    current_fp[idx] = new_fp
    current_os[idx] = OS_GENERATIONS
    return ev


def _step_reinstall(
    person: Any,
    idx: int,
    day: int,
    current_os: list[int],
    current_fp: list[str],
    seen_fps: set[str],
    rng: np.random.Generator,
) -> DeviceEvent:
    """
    WHAT: Generates an app reinstall event altering fingerprint while keeping OS generation.
    WHY: Represents client application reset on the existing hardware unit.
    """
    plat = person.device.platform
    minute = day * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY))
    new_fp = _draw_unique_fingerprint(rng, plat, current_os[idx], seen_fps)
    ev = make_device_event(
        person.person_id, minute, "reinstall", plat, current_os[idx], current_os[idx], current_fp[idx], new_fp
    )
    current_fp[idx] = new_fp
    return ev


def _step_person_event(
    idx: int, person: Any, day: int, change_u: float, reinstall_u: float,
    daily_change_prob: float, reinstall_prob: float, scheduled_waves: dict[tuple[int, int], int],
    current_os: list[int], current_fp: list[str], seen_fps: set[str], rng: np.random.Generator,
) -> DeviceEvent | None:
    """
    WHAT: Evaluates phone change, reinstall, or OS update priority for one person on one day.
    WHY: Enforces priority order (phone_change > reinstall > os_update) with at most one event per day.
    """
    if change_u < daily_change_prob:
        return _step_phone_change(person, idx, day, current_os, current_fp, seen_fps, rng)
    if reinstall_u < reinstall_prob:
        return _step_reinstall(person, idx, day, current_os, current_fp, seen_fps, rng)
    if (idx, day) in scheduled_waves and current_os[idx] < OS_GENERATIONS:
        plat = person.device.platform
        minute = scheduled_waves[(idx, day)]
        ev = make_device_event(
            person.person_id, minute, "os_update", plat, current_os[idx],
            current_os[idx] + 1, current_fp[idx], current_fp[idx]
        )
        current_os[idx] += 1
        return ev
    return None


def _step_day_events(
    day: int,
    people: list[Any],
    change_prob: float,
    reinstall_prob: float,
    scheduled_waves: dict[tuple[int, int], int],
    current_os: list[int],
    current_fp: list[str],
    seen_fps: set[str],
    rng: np.random.Generator,
) -> list[DeviceEvent]:
    """
    WHAT: Samples and compiles events for all individuals on a single day.
    WHY: Vectorizes random draws for changes and reinstalls across the town.
    """
    changes = rng.random(len(people))
    reinstalls = rng.random(len(people))
    day_events: list[DeviceEvent] = []
    for i, person in enumerate(people):
        ev = _step_person_event(
            i, person, day, changes[i], reinstalls[i], change_prob, reinstall_prob,
            scheduled_waves, current_os, current_fp, seen_fps, rng
        )
        if ev is not None:
            day_events.append(ev)
    return day_events


def _simulate_daily_device_events(
    settings: DeviceSettings,
    people: list[Any],
    festivals: list[Festival],
    n_days: int,
    scheduled_waves: dict[tuple[int, int], int],
    rng: np.random.Generator,
) -> list[DeviceEvent]:
    """
    WHAT: Iterates over simulation days to generate phone replacements, reinstalls, and OS updates.
    WHY: Handles festival sale windows and maintains current OS and fingerprint state per citizen.
    """
    current_os = [p.device.os_generation for p in people]
    current_fp = [p.device.fingerprint for p in people]
    seen_fps: set[str] = set(current_fp)
    diwali = festivals[0]
    reinstall_prob = settings.reinstall_per_year / DAYS_PER_YEAR
    events: list[DeviceEvent] = []

    for day in range(n_days):
        in_sale = diwali.peak_day <= day < diwali.peak_day + settings.sale_days
        uplift = (1.0 + settings.sale_uplift) if in_sale else 1.0
        # 30 days per month is a structural choice.
        change_prob = (1.0 / (settings.replacement_months * DAYS_PER_MONTH)) * uplift
        events.extend(
            _step_day_events(
                day, people, change_prob, reinstall_prob, scheduled_waves, current_os, current_fp, seen_fps, rng
            )
        )
    return events


def _device_event_sort_key(ev: DeviceEvent) -> tuple[int, str, str]:
    """
    WHAT: Returns sort key for ordering device events by minute, person ID, and event kind.
    WHY: Provides deterministic chronological ordering without using lambdas.
    """
    return (ev.minute, ev.person_id, ev.kind)


def draw_device_event_schedule(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    people: list[Any],
    festivals: list[Festival],
    n_days: int,
    range_name: str = "training",
    wave_platform: str | None = None,
) -> list[DeviceEvent]:
    """
    WHAT: Generates full chronological schedule of device events across the simulated world timeline.
    WHY: Simulates phone replacements, app reinstalls, and OS adoption waves without altering state.
    """
    _validate_schedule_args(ranges, people, festivals, n_days, range_name, wave_platform)
    settings = draw_device_settings(ranges, rng, range_name=range_name)
    scheduled_waves = _schedule_os_waves(ranges, rng, people, n_days, range_name, wave_platform)
    events = _simulate_daily_device_events(settings, people, festivals, n_days, scheduled_waves, rng)
    return sorted(events, key=_device_event_sort_key)
