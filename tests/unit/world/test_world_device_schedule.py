"""
WHAT: Unit tests S1-S11 for per-world device event scheduling and R5 hardware preservation rules.
WHY: Verifies stochastic phone replacements, app reinstalls, and OS rollout waves (issues R5, A1, R2).
ISSUE IDs: R5 (OS updates and fingerprint preservation), A1 (device realism), R2 (background realism).
TERM: device schedule: The chronologically sorted series of phone replacements, reinstalls, and OS updates.
TERM: stand-in person: A lightweight mock entity pairing a person ID with a hardware device.
"""

import hashlib
import inspect
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from world.background_config import read_background_realism
from world.build_world import build_world
from world.device_events import DEVICE_EVENT_KINDS, DeviceEvent, make_device_event
from world.device_schedule import draw_device_event_schedule
from world.devices import Device
from world.festivals import Festival

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "device_realism.yaml"


@dataclass
class StandinPerson:
    """
    WHAT: Lightweight test citizen entity carrying a person identifier and mobile device profile.
    WHY: Supplies isolated agent fixtures without running full world generation.
    """

    person_id: str
    device: Device


def _make_standin_people(n: int, platform: str = "android", os_gen: int = 2) -> list[StandinPerson]:
    """
    WHAT: Generates n stand-in people with deterministic device IDs, platforms, and unique fingerprints.
    WHY: Provides predictable population fixtures for rate and distribution tests.
    """
    people: list[StandinPerson] = []
    for i in range(n):
        pid = f"person_{i:04d}"
        fp = f"fp_{platform}_v{os_gen}_{i:08x}"
        dev = Device(device_id=f"dev_{i:04d}", platform=platform, os_generation=os_gen, fingerprint=fp)
        people.append(StandinPerson(person_id=pid, device=dev))
    return people


def _make_people_generation_spread(n: int) -> list[StandinPerson]:
    """
    WHAT: Creates n people evenly distributed across OS generations 1 to 4.
    WHY: Supplies population for testing OS update eligibility and preservation invariants.
    """
    people: list[StandinPerson] = []
    quarter = n // 4
    for gen in (1, 2, 3, 4):
        for i in range(quarter):
            idx = (gen - 1) * quarter + i
            pid = f"person_{idx:05d}"
            fp = f"fp_android_v{gen}_{idx:08x}"
            dev = Device(device_id=f"dev_{idx:05d}", platform="android", os_generation=gen, fingerprint=fp)
            people.append(StandinPerson(person_id=pid, device=dev))
    return people


def _make_param_entry(val: float | int) -> dict[str, list[Any]]:
    """
    WHAT: Builds a 3-split range entry with identical boundaries.
    WHY: Helper for creating mock config parameters.
    """
    return {"training": [val, val], "test": [val, val], "stress": [val, val]}


def _make_test_ranges(
    replacement_months: float = 36.0,
    sale_uplift: float = 0.0,
    sale_days: int = 14,
    reinstall_per_year: float = 0.0,
    wave_count: int = 0,
    wave_length: int = 14,
    wave_share: float = 0.5,
) -> dict[str, Any]:
    """
    WHAT: Constructs a parameter ranges dictionary with customized fixed bounds.
    WHY: Supplies isolated config fixtures where specific parameters are pinned to exact values.
    """
    return {
        "parameters": {
            "phone_replacement_months": _make_param_entry(replacement_months),
            "phone_change_festival_sale_uplift": _make_param_entry(sale_uplift),
            "phone_change_festival_sale_days": _make_param_entry(sale_days),
            "app_reinstall_per_person_per_year": _make_param_entry(reinstall_per_year),
            "os_wave_count_per_world": _make_param_entry(wave_count),
            "os_wave_length_days": _make_param_entry(wave_length),
            "os_wave_final_share_of_eligible": _make_param_entry(wave_share),
        }
    }


def _assert_schedule_invariants(events: list[DeviceEvent], n_days: int) -> None:
    """
    WHAT: Asserts timing bounds, kinds, sorting, single event per day, and fingerprint uniqueness.
    WHY: Enforces core structural invariants on generated device event schedules.
    """
    seen_person_days: set[tuple[str, int]] = set()
    new_fingerprints: set[str] = set()
    for ev in events:
        assert 0 <= ev.minute < n_days * 1440
        assert ev.kind in DEVICE_EVENT_KINDS
        day = ev.minute // 1440
        assert (ev.person_id, day) not in seen_person_days
        seen_person_days.add((ev.person_id, day))
        if ev.kind in ("phone_change", "reinstall"):
            assert ev.fingerprint_after != ev.fingerprint_before
            assert ev.fingerprint_after not in new_fingerprints
            new_fingerprints.add(ev.fingerprint_after)
    for i in range(len(events) - 1):
        assert (events[i].minute, events[i].person_id, events[i].kind) <= (
            events[i + 1].minute,
            events[i + 1].person_id,
            events[i + 1].kind,
        )


def test_s1_shape_and_invariants() -> None:
    """
    WHAT: Tests shape, ordering, uniqueness, and day-isolation of events over 20 seeds.
    WHY: Verifies fundamental device event invariants against real world configurations.
    """
    config = read_background_realism(CONFIG_PATH)
    world = build_world(seed=5, n_people=300, n_shops=40)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        events = draw_device_event_schedule(config, rng, world.people, festivals, n_days=60)
        _assert_schedule_invariants(events, n_days=60)


def test_s2_phone_change_rate() -> None:
    """
    WHAT: Tests phone change rate matches 36-month replacement frequency within 25 percent.
    WHY: Confirms aggregate phone replacements align with empirical device lifecycle data.
    """
    ranges = _make_test_ranges(replacement_months=36.0, sale_uplift=0.0, reinstall_per_year=0.0, wave_count=0)
    people = _make_standin_people(3000)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    rng = np.random.default_rng(11)
    events = draw_device_event_schedule(ranges, rng, people, festivals, n_days=60)
    phone_changes = [e for e in events if e.kind == "phone_change"]
    expected = 3000 * 60 / (36 * 30)
    measured = len(phone_changes)
    assert abs(measured - expected) / expected <= 0.25


def test_s3_reinstall_rate() -> None:
    """
    WHAT: Tests app reinstall frequency matches annual rate within 25 percent margin.
    WHY: Validates authentic reinstall dynamics producing new hardware signatures.
    """
    ranges = _make_test_ranges(replacement_months=1000.0, reinstall_per_year=0.3, wave_count=0)
    people = _make_standin_people(3000)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    rng = np.random.default_rng(12)
    events = draw_device_event_schedule(ranges, rng, people, festivals, n_days=60)
    reinstalls = [e for e in events if e.kind == "reinstall"]
    expected = 3000 * 60 * 0.3 / 365
    measured = len(reinstalls)
    assert abs(measured - expected) / expected <= 0.25


def test_s4_festival_sale_uplift() -> None:
    """
    WHAT: Tests phone change rate during festival sale window is between 1.5x and 2.5x baseline.
    WHY: Verifies holiday commercial surges increase hardware turnover during Diwali sale windows.
    """
    ranges = _make_test_ranges(
        replacement_months=12.0, sale_uplift=1.0, sale_days=14, reinstall_per_year=0.0, wave_count=0
    )
    people = _make_standin_people(3000)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    rng = np.random.default_rng(13)
    events = draw_device_event_schedule(ranges, rng, people, festivals, n_days=60)
    inside_count = sum(1 for e in events if e.kind == "phone_change" and 20 <= e.minute // 1440 < 34)
    outside_count = sum(1 for e in events if e.kind == "phone_change" and not (20 <= e.minute // 1440 < 34))
    rate_inside = inside_count / 14.0
    rate_outside = outside_count / 46.0
    ratio = rate_inside / rate_outside
    assert 1.5 <= ratio <= 2.5


def test_s5_platform_independence() -> None:
    """
    WHAT: Tests platform parity in phone replacement and lack of platform in failure decisions.
    WHY: Confirms device platform does not bias replacement rates or payment rejections.
    """
    from world.failure_decision import decide_failure

    ranges = _make_test_ranges(replacement_months=12.0, reinstall_per_year=0.0, wave_count=0)
    people = _make_standin_people(2000, platform="android") + _make_standin_people(2000, platform="ios")
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    rng = np.random.default_rng(14)
    events = draw_device_event_schedule(ranges, rng, people, festivals, n_days=60)
    android_count = sum(1 for e in events if e.kind == "phone_change" and e.platform == "android")
    ios_count = sum(1 for e in events if e.kind == "phone_change" and e.platform == "ios")
    ratio = android_count / ios_count
    assert 0.75 <= ratio <= 1.25
    sig = inspect.signature(decide_failure)
    for p in sig.parameters.keys():
        assert "platform" not in p and "os" not in p and "device" not in p


def test_s6_r5_proof() -> None:
    """
    WHAT: Tests R5 invariant that OS updates preserve hardware fingerprints and increment generation.
    WHY: Proves legitimate OS rollout waves are not classified as suspicious device changes.
    """
    ranges = _make_test_ranges(
        replacement_months=1000.0, reinstall_per_year=0.0, wave_count=1, wave_length=30, wave_share=0.5
    )
    people = _make_people_generation_spread(4000)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    rng = np.random.default_rng(15)
    events = draw_device_event_schedule(ranges, rng, people, festivals, n_days=60)
    updates = [e for e in events if e.kind == "os_update"]
    for u in updates:
        assert not u.counts_as_new_device
        assert u.fingerprint_before == u.fingerprint_after
        assert u.os_generation_after == u.os_generation_before + 1
        assert u.os_generation_before < 4
    expected = 0.5 * 3000
    measured = len(updates)
    # Architect-approved margin change AFTER failure: count is binomial draw with SD ~27;
    # 8% is ~4.4 SDs (1 in 15000 fail); 5% was set without that calculation and was too tight.
    assert abs(measured - expected) / expected <= 0.08


def test_s7_adoption_curve() -> None:
    """
    WHAT: Tests bell-shaped adoption curve dynamics across 10-day thirds of an OS wave.
    WHY: Verifies beta(2,2) rollout generates classic gradual adoption peak and tail.
    """
    ranges = _make_test_ranges(
        replacement_months=1000.0, reinstall_per_year=0.0, wave_count=1, wave_length=30, wave_share=0.5
    )
    people = _make_people_generation_spread(4000)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    rng = np.random.default_rng(15)
    events = draw_device_event_schedule(ranges, rng, people, festivals, n_days=60)
    updates = [e for e in events if e.kind == "os_update"]
    min_day = min(u.minute for u in updates) // 1440
    t1 = sum(1 for u in updates if min_day <= u.minute // 1440 < min_day + 10)
    t2 = sum(1 for u in updates if min_day + 10 <= u.minute // 1440 < min_day + 20)
    t3 = sum(1 for u in updates if min_day + 20 <= u.minute // 1440 < min_day + 30)
    assert t2 > t1 and t2 > t3
    cumulatives = [t1, t1 + t2, t1 + t2 + t3]
    assert cumulatives[0] <= cumulatives[1] <= cumulatives[2]
    for u in updates:
        assert min_day * 1440 <= u.minute < (min_day + 30) * 1440


def test_s8_wave_platform_filter() -> None:
    """
    WHAT: Tests filtering OS wave participation by platform targeting.
    WHY: Verifies OS rollouts can selectively target Android or iOS ecosystems.
    """
    ranges = _make_test_ranges(
        replacement_months=1000.0, reinstall_per_year=0.0, wave_count=1, wave_length=30, wave_share=0.5
    )
    p_and = _make_standin_people(2000, platform="android", os_gen=1)
    p_ios = _make_standin_people(2000, platform="ios", os_gen=1)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    rng1 = np.random.default_rng(15)
    ev_ios = draw_device_event_schedule(ranges, rng1, p_and + p_ios, festivals, n_days=60, wave_platform="ios")
    updates_ios = [e for e in ev_ios if e.kind == "os_update"]
    assert len(updates_ios) > 0 and all(u.platform == "ios" for u in updates_ios)
    rng2 = np.random.default_rng(15)
    ev_all = draw_device_event_schedule(ranges, rng2, p_and + p_ios, festivals, n_days=60, wave_platform=None)
    updates_all = [e for e in ev_all if e.kind == "os_update"]
    platforms = {u.platform for u in updates_all}
    assert "android" in platforms and "ios" in platforms


def test_s9_no_hard_coded_time() -> None:
    """
    WHAT: Tests variability of wave start days across 100 seeds.
    WHY: Confirms absence of hard-coded event days or deterministic calendar offsets.
    """
    ranges = _make_test_ranges(
        replacement_months=1000.0, reinstall_per_year=0.0, wave_count=1, wave_length=14, wave_share=0.5
    )
    people = _make_standin_people(500, platform="android", os_gen=1)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    start_days: set[int] = set()
    for seed in range(100):
        rng = np.random.default_rng(seed)
        events = draw_device_event_schedule(ranges, rng, people, festivals, n_days=60)
        updates = [e for e in events if e.kind == "os_update"]
        if updates:
            start_days.add(min(u.minute // 1440 for u in updates))
    assert len(start_days) >= 10


def _check_s10_refusals(ranges: dict[str, Any], people: list[Any], festivals: list[Any]) -> None:
    """
    WHAT: Validates argument refusal error handling for invalid days, empty entities, and platforms.
    WHY: Proves robust input validation against malformed or out-of-range simulation parameters.
    """
    rng = np.random.default_rng(0)
    for bad_days in [1, 0, -3, 2.5, "60"]:
        with pytest.raises(ValueError):
            draw_device_event_schedule(ranges, rng, people, festivals, n_days=bad_days)  # type: ignore
    with pytest.raises(ValueError, match="people"):
        draw_device_event_schedule(ranges, rng, [], festivals, n_days=60)
    with pytest.raises(ValueError, match="festivals"):
        draw_device_event_schedule(ranges, rng, people, [], n_days=60)
    with pytest.raises(ValueError, match="wave_platform"):
        draw_device_event_schedule(ranges, rng, people, festivals, n_days=60, wave_platform="windows")
    with pytest.raises(ValueError, match="range name"):
        draw_device_event_schedule(ranges, rng, people, festivals, n_days=60, range_name="unknown")
    real_cfg = read_background_realism(CONFIG_PATH)
    with pytest.raises(ValueError, match="smaller than"):
        draw_device_event_schedule(real_cfg, rng, people, festivals, n_days=60, range_name="stress")


def _run_subprocess_hash(seed: int) -> str:
    """
    WHAT: Spawns fresh python subprocess to compute SHA-256 hash of schedule.
    WHY: Verifies process-independent determinism without memory contamination.
    """
    py_code = (
        "import hashlib, numpy as np; "
        "from world.device_schedule import draw_device_event_schedule; "
        "from world.festivals import Festival; "
        "from tests.unit.world.test_world_device_schedule import _make_standin_people, _make_test_ranges; "
        f"rng = np.random.default_rng({seed}); "
        "p = _make_standin_people(100); "
        "f = [Festival('diwali', 20, 5, 5, 0.5)]; "
        "r = _make_test_ranges(replacement_months=12.0, wave_count=1); "
        "ev = draw_device_event_schedule(r, rng, p, f, n_days=60); "
        "print(hashlib.sha256(repr(ev).encode('utf-8')).hexdigest())"
    )
    cmd = [sys.executable, "-B", "-c", py_code]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _check_s10_determinism(ranges: dict[str, Any], people: list[Any], festivals: list[Any]) -> None:
    """
    WHAT: Tests identical schedule generation for identical seeds and variation across different seeds.
    WHY: Guarantees deterministic reproducibility for auditability and verification.
    """
    rng1 = np.random.default_rng(7)
    ev1 = draw_device_event_schedule(ranges, rng1, people, festivals, n_days=60)
    rng2 = np.random.default_rng(7)
    ev2 = draw_device_event_schedule(ranges, rng2, people, festivals, n_days=60)
    assert ev1 == ev2

    h1 = hashlib.sha256(repr(ev1).encode("utf-8")).hexdigest()
    assert _run_subprocess_hash(7) == h1

    differs = False
    for s in range(1, 21):
        rng_diff = np.random.default_rng(s)
        if draw_device_event_schedule(ranges, rng_diff, people, festivals, n_days=60) != ev1:
            differs = True
            break
    assert differs


def test_s10_refusals_and_determinism() -> None:
    """
    WHAT: Executes refusal checks and determinism proofs across processes and seeds.
    WHY: Verifies parameter defense and cryptographic schedule reproducibility.
    """
    ranges = _make_test_ranges(replacement_months=12.0, wave_count=1)
    people = _make_standin_people(100)
    festivals = [Festival("diwali", peak_day=20, rise_days=5, fade_days=5, uplift_fraction=0.5)]
    _check_s10_refusals(ranges, people, festivals)
    _check_s10_determinism(ranges, people, festivals)


def test_s11_make_device_event_validation() -> None:
    """
    WHAT: Tests make_device_event argument validation, refusals, and counts_as_new_device tagging.
    WHY: Proves strict constructor invariants for all DeviceEvent instances.
    """
    with pytest.raises(ValueError, match="kind"):
        make_device_event("p1", 10, "unknown_kind", "android", 1, 2, "fp1", "fp1")
    for bad_min in [-1, -100, 1.5, "10", True]:
        with pytest.raises(ValueError, match="minute"):
            make_device_event("p1", bad_min, "phone_change", "android", 1, 4, "fp1", "fp2")  # type: ignore
    with pytest.raises(ValueError, match="fingerprint"):
        make_device_event("p1", 10, "os_update", "android", 1, 2, "fp1", "fp2")
    with pytest.raises(ValueError, match="generation"):
        make_device_event("p1", 10, "os_update", "android", 1, 3, "fp1", "fp1")
    with pytest.raises(ValueError, match="fingerprint"):
        make_device_event("p1", 10, "phone_change", "android", 1, 4, "fp1", "fp1")
    with pytest.raises(ValueError, match="platform"):
        make_device_event("p1", 10, "phone_change", "windows", 1, 4, "fp1", "fp2")

    ev_phone = make_device_event("p1", 10, "phone_change", "android", 1, 4, "fp1", "fp2")
    ev_reinst = make_device_event("p1", 10, "reinstall", "android", 1, 1, "fp1", "fp2")
    ev_os = make_device_event("p1", 10, "os_update", "android", 1, 2, "fp1", "fp1")
    assert ev_phone.counts_as_new_device is True
    assert ev_reinst.counts_as_new_device is True
    assert ev_os.counts_as_new_device is False
