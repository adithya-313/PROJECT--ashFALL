"""
WHAT: Integration and regression tests for device event execution in calendar simulations.
WHY: Proves device lifecycle events mutate personal devices in time order and stamp payment logs (issue R5).
TERM: device event: a discrete occurrence altering handset state such as phone replacement or OS update,
      e.g. upgrading an operating system generation while preserving hardware serial numbers.
TERM: device stamp: telemetry metadata recording sender hardware fingerprint and novelty per transaction,
      e.g. tagging a payment with the sender phone fingerprint and marking new devices.
TERM: golden hash: a cryptographic SHA-256 digest ensuring simulation outputs reproduce identically,
      e.g. verifying that a repeated simulation run produces the exact same payment ledger.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from world.device_events import make_device_event
from world.device_runtime import (
    NO_DEVICE_STAMP,
    DeviceStamp,
    apply_device_event,
    initial_known_fingerprints,
    stamp_sender_device,
)
from world.devices import Device
from world.world_plan import spawn_life_streams, spawn_world_streams

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVICE_CONFIG_PATH = REPO_ROOT / "config" / "device_realism.yaml"


@dataclass
class DummyPerson:
    """
    WHAT: Stand-in test person entity holding an identifier string and mobile device record.
    WHY: Isolates unit tests from full world generation dependencies.
    """

    person_id: str
    device: Device


def test_spawn_life_streams() -> None:
    """
    WHAT: Tests reproducibility and independence of auxiliary device and episode pseudo-random streams.
    WHY: Proves child streams 2 and 3 do not collide with each other or perturb the primary calendar streams.
    Example:
        dev_rng, ep_rng = spawn_life_streams(101)
        assert dev_rng.uniform() != ep_rng.uniform()
    """
    # 1. Test reproducibility across repeated calls with identical seed.
    dev1, ep1 = spawn_life_streams(101)
    dev2, ep2 = spawn_life_streams(101)
    d1, e1 = [float(dev1.uniform()) for _ in range(5)], [float(ep1.uniform()) for _ in range(5)]
    d2, e2 = [float(dev2.uniform()) for _ in range(5)], [float(ep2.uniform()) for _ in range(5)]
    assert d1 == d2 and e1 == e2
    # 2. Test independence between the two life streams.
    assert d1 != e1
    # 3. Test independence from primary plan and run simulation streams.
    plan_rng, run_rng = spawn_world_streams(101)
    p_draws = [float(plan_rng.uniform()) for _ in range(5)]
    r_draws = [float(run_rng.uniform()) for _ in range(5)]
    assert d1 != p_draws and d1 != r_draws
    assert e1 != p_draws and e1 != r_draws
    # 4. Prove spawn_world_streams matches children 0 and 1 of SeedSequence.spawn(4).
    s4 = np.random.SeedSequence(101).spawn(4)
    rng0 = np.random.default_rng(s4[0])
    rng1 = np.random.default_rng(s4[1])
    assert p_draws == [float(rng0.uniform()) for _ in range(5)]
    assert r_draws == [float(rng1.uniform()) for _ in range(5)]


def test_apply_device_event_precondition_mismatches() -> None:
    """
    WHAT: Tests argument and hardware state verification before mutating citizen device profiles.
    WHY: Proves mismatched person ID, fingerprint before, or OS version raises RuntimeError immediately.
    Example:
        # with pytest.raises(RuntimeError):
        #     apply_device_event(person, event)
    """
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_old")
    p1 = DummyPerson("p1", dev)
    # 1. Wrong person ID must raise RuntimeError.
    ev_bad_p = make_device_event("p2", 10, "phone_change", "android", 2, 4, "fp_old", "fp_new")
    with pytest.raises(RuntimeError, match="Person mismatch"):
        apply_device_event(p1, ev_bad_p)
    # 2. Fingerprint mismatch must raise RuntimeError.
    ev_bad_fp = make_device_event("p1", 10, "phone_change", "android", 2, 4, "fp_other", "fp_new")
    with pytest.raises(RuntimeError, match="FP mismatch"):
        apply_device_event(p1, ev_bad_fp)
    # 3. OS generation mismatch must raise RuntimeError.
    ev_bad_os = make_device_event("p1", 10, "phone_change", "android", 1, 4, "fp_old", "fp_new")
    with pytest.raises(RuntimeError, match="OS mismatch"):
        apply_device_event(p1, ev_bad_os)


def test_apply_device_event_os_update() -> None:
    """
    WHAT: Verifies that OS update events mutate OS generation without altering hardware serial numbers.
    WHY: Proves rule R5 hardware preservation where OS updates keep identical fingerprints and device IDs.
    Example:
        # apply_device_event(person, os_event)
        # assert person.device.fingerprint == "fp_init" and person.device.os_generation == 3
    """
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_init")
    person = DummyPerson("p1", dev)
    # 1. Construct valid OS update event keeping fingerprint.
    ev = make_device_event("p1", 50, "os_update", "android", 2, 3, "fp_init", "fp_init")
    apply_device_event(person, ev)
    # 2. Check that OS generation increments while fingerprint and device ID stay invariant.
    assert person.device.os_generation == 3
    assert person.device.fingerprint == "fp_init"
    assert person.device.device_id == "dev_p1"


def test_apply_device_event_phone_change() -> None:
    """
    WHAT: Asserts that phone change events update hardware fingerprint, OS generation, and device ID suffix.
    WHY: Represents physical handset replacement, advancing device identifier across sequential upgrades.
    Example:
        # apply_device_event(person, phone_ev)
        # assert person.device.device_id == "dev_p1_p1" and person.device.fingerprint == "fp_new"
    """
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_1")
    person = DummyPerson("p1", dev)
    # 1. Apply first phone replacement event.
    ev1 = make_device_event("p1", 60, "phone_change", "android", 2, 4, "fp_1", "fp_2")
    apply_device_event(person, ev1)
    assert person.device.device_id == "dev_p1_p1"
    assert person.device.fingerprint == "fp_2"
    # 2. Apply second phone replacement event to verify suffix incrementation.
    ev2 = make_device_event("p1", 120, "phone_change", "android", 4, 4, "fp_2", "fp_3")
    apply_device_event(person, ev2)
    assert person.device.device_id == "dev_p1_p2"
    assert person.device.fingerprint == "fp_3"


def test_apply_device_event_reinstall() -> None:
    """
    WHAT: Tests that app reinstall events cycle the client fingerprint while preserving physical handset state.
    WHY: Models application resets where hardware device ID and OS version remain completely unchanged.
    Example:
        # apply_device_event(person, reinstall_ev)
        # assert person.device.fingerprint == "fp_reinst" and person.device.os_generation == 2
    """
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_orig")
    person = DummyPerson("p1", dev)
    # 1. Construct valid reinstall event altering fingerprint only.
    ev = make_device_event("p1", 80, "reinstall", "android", 2, 2, "fp_orig", "fp_reinst")
    apply_device_event(person, ev)
    # 2. Check fingerprint changes while device ID and OS generation stay invariant.
    assert person.device.fingerprint == "fp_reinst"
    assert person.device.device_id == "dev_p1"
    assert person.device.os_generation == 2


def test_stamp_sender_device_novelty_and_shop() -> None:
    """
    WHAT: Tests device stamp generation, new device tagging, and shop sender sentinel behavior.
    WHY: Proves initial hardware is recognized, newly adopted fingerprints trigger one new stamp,
         and shops receive NO_DEVICE_STAMP sentinel.
    Example:
        # stamp = stamp_sender_device(person, known)
        # assert stamp.is_new_device is False
    """
    # 1. Commercial shop senders without devices receive NO_DEVICE_STAMP.
    assert stamp_sender_device(None, {}) == NO_DEVICE_STAMP
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_init")
    person = DummyPerson("p1", dev)
    known = initial_known_fingerprints([person])
    # 2. First payment on starting hardware is recognized as known.
    s1 = stamp_sender_device(person, known)
    assert s1 == DeviceStamp("fp_init", 2, False)
    # 3. After fingerprint change, first payment marks new device and second payment is recognized.
    person.device.fingerprint = "fp_new"
    s2 = stamp_sender_device(person, known)
    assert s2 == DeviceStamp("fp_new", 2, True)
    s3 = stamp_sender_device(person, known)
    assert s3 == DeviceStamp("fp_new", 2, False)

