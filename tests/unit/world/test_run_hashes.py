"""
WHAT: Unit tests verifying canonical hash recipes reproduce golden baselines and detect mutations.
WHY: Proves cryptographic hash recipes protect against simulation regressions across domain objects.
TERM: hash: A fixed-length 64-character hexadecimal SHA-256 digest calculated over serialized domain objects.
TERM: golden value: A pre-calculated baseline hash or count saved from an audited run to prevent regressions.
"""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from tests.unit.world.test_world_calendar_runner import _hash_event_list
from world.background_config import read_background_realism
from world.build_world import build_world
from world.calendar_runner import run_world
from world.device_runtime import DeviceStamp
from world.device_schedule import draw_device_event_schedule
from world.routine import run_one_month
from world.run_hashes import (
    hash_device_stamps,
    hash_ledger_entries,
    hash_payment_events,
    hash_people_devices,
)
from world.world_plan import draw_world_plan, spawn_life_streams, spawn_world_streams

# G1 ledger entries golden hash (set by 11A routine; recorded in 11B-3a cleanup, PR #48).
GOLDEN_G1_HASH: str = "1ede1d085a0a503bbe7873588ff25a0f6c5b83b49b8850b75a183be7afad16da"

# G2 payment events golden hash (set by 11B-3a calendar runner, PR #47).
GOLDEN_G2_HASH: str = "12921b27db24a71a00be469709e768408a8e1a52cd568fd5db4f4dcea9d9f8dd"

# G3 event sender device stamps golden hash (set by 11B-3b device runtime, PR #49).
GOLDEN_G3_STAMP_HASH: str = "d826b197411f9a2d80e77970ceafc3be071525c794ac32b06385376b22d76b46"

# G3 people device state golden hash (set by 11B-3b device runtime, PR #49).
GOLDEN_G3_PEOPLE_HASH: str = "606b19fe450db9a6367c36596364ae2000c5f26c7681911e452d2923c63dab97"


def _run_g3_simulation() -> tuple[Any, Any]:
    """
    WHAT: Builds world and executes 61-day calendar simulation with device events schedule.
    WHY: Supplies executed RunResult and populated World instance for G3 golden hash assertions.
    Returns: Tuple containing the simulated World container and calendar RunResult container.
    Example:
        # world, result = _run_g3_simulation()
        # assert len(world.people) == 60 and len(result.events) > 0
    """
    # 1. Construct world and spawn pseudo-random streams for calendar and device schedules.
    w = build_world(seed=101, n_people=60, n_shops=12)
    p_rng, r_rng = spawn_world_streams(101)
    plan = draw_world_plan(p_rng, n_days=61)
    dev_rng, _ = spawn_life_streams(101)
    # 2. Draw device events schedule from device realism configuration.
    bg = read_background_realism(Path("config/device_realism.yaml"))
    sched = draw_device_event_schedule(bg, dev_rng, w.people, plan.festivals, plan.n_days)
    # 3. Execute calendar simulation run applying device lifecycle transitions.
    res = run_world(w, plan, r_rng, device_events=sched)
    return w, res


def test_h1_hash_ledger_entries_reproduces_g1() -> None:
    """
    WHAT: Tests hash_ledger_entries reproduces G1 golden hash and detects single-field record modifications.
    WHY: Proves deterministic verification for double-entry bookkeeping journals against Level 11A baselines.
    Example:
        # test_h1_hash_ledger_entries_reproduces_g1()
        # assert hash_ledger_entries(world.ledgers) == GOLDEN_G1_HASH
    """
    # 1. Build standard 300-person, 40-shop world and execute one-month simulation routine.
    w = build_world(seed=7, n_people=300, n_shops=40)
    run_one_month(w, np.random.default_rng(11))
    # 2. Verify calculated ledger entries hash reproduces G1 golden baseline digest.
    h1 = hash_ledger_entries(w.ledgers)
    assert h1 == GOLDEN_G1_HASH
    # 3. Mutate a single transaction entry amount to verify hash sensitivity to content edits.
    orig = w.ledgers[0].entries[0].amount_paise
    w.ledgers[0].entries[0].amount_paise = orig + 1
    assert hash_ledger_entries(w.ledgers) != h1
    # 4. Output measured verification line for test execution auditing.
    print(f"MEASURED H1_G1: hash={h1[:16]}... diff_detected=True")


def test_h2_hash_payment_events_reproduces_g2_and_matches_runner() -> None:
    """
    WHAT: Tests hash_payment_events matches G2 golden hash, equals _hash_event_list, and senses mutations.
    WHY: Guarantees unified event serialization matches calendar runner baselines from Level 11B-3a.
    Example:
        # test_h2_hash_payment_events_reproduces_g2_and_matches_runner()
        # assert hash_payment_events(events) == GOLDEN_G2_HASH
    """
    # 1. Build world and run calendar simulation using standard golden G2 parameter setup.
    w = build_world(seed=101, n_people=60, n_shops=12)
    p_rng, r_rng = spawn_world_streams(101)
    res = run_world(w, draw_world_plan(p_rng, n_days=61), r_rng)
    # 2. Assert calculated event hash matches G2 baseline and legacy runner hash function.
    h2 = hash_payment_events(res.events)
    assert h2 == GOLDEN_G2_HASH
    assert h2 == _hash_event_list(res.events)
    # 3. Mutate one event status attribute to verify hash sensitivity to state divergence.
    ev_list = list(res.events)
    ev_list[0] = dataclasses.replace(ev_list[0], amount_paise=ev_list[0].amount_paise + 1)
    assert hash_payment_events(ev_list) != h2
    # 4. Print measured audit line confirming parity and sensitivity.
    print(f"MEASURED H2_G2: hash={h2[:16]}... matches_runner=True diff_detected=True")


def test_h3_hash_device_stamps_reproduces_g3() -> None:
    """
    WHAT: Asserts hash_device_stamps matches G3 stamp golden hash and senses single stamp edits.
    WHY: Verifies telemetry stamping determinism and novelty detection against Level 11B-3b baseline.
    Example:
        # test_h3_hash_device_stamps_reproduces_g3()
        # assert hash_device_stamps(res.event_sender_devices) == GOLDEN_G3_STAMP_HASH
    """
    # 1. Execute standard G3 simulation with active device events schedule.
    _, res = _run_g3_simulation()
    # 2. Calculate telemetry stamps hash and assert match with verified G3 baseline.
    h3 = hash_device_stamps(res.event_sender_devices)
    assert h3 == GOLDEN_G3_STAMP_HASH
    # 3. Mutate one stamp novelty flag to verify sensitivity to telemetry discrepancies.
    stamps = list(res.event_sender_devices)
    orig = stamps[0]
    stamps[0] = DeviceStamp(orig.fingerprint, orig.os_generation, not orig.is_new_device)
    assert hash_device_stamps(stamps) != h3
    # 4. Print measured verification output for audit logging.
    print(f"MEASURED H3_G3_STAMPS: hash={h3[:16]}... diff_detected=True")


def test_h4_hash_people_devices_reproduces_g3() -> None:
    """
    WHAT: Asserts hash_people_devices reproduces G3 people golden hash and catches device changes.
    WHY: Verifies population device assignments and OS state transitions against Level 11B-3b baseline.
    Example:
        # test_h4_hash_people_devices_reproduces_g3()
        # assert hash_people_devices(world.people) == GOLDEN_G3_PEOPLE_HASH
    """
    # 1. Execute standard G3 simulation run with device turnover schedule.
    w, _ = _run_g3_simulation()
    # 2. Compute people device state hash and assert match with verified G3 baseline.
    h4 = hash_people_devices(w.people)
    assert h4 == GOLDEN_G3_PEOPLE_HASH
    # 3. Mutate one citizen OS generation to verify digest sensitivity to device drift.
    w.people[0].device.os_generation += 1
    assert hash_people_devices(w.people) != h4
    # 4. Output measured verification line for audit records.
    print(f"MEASURED H4_G3_PEOPLE: hash={h4[:16]}... diff_detected=True")
