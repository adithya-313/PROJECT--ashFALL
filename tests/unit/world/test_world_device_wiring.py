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

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party numerical computation and test framework modules.
import numpy as np
import pytest

# Simulation engine imports for world construction, calendar runner, and device runtime.
from tests.unit.world.test_world_calendar_runner import _hash_event_list
from world.background_config import read_background_realism
from world.build_world import build_world
from world.calendar_runner import RunResult, run_world
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.device_events import make_device_event
from world.device_runtime import (
    NO_DEVICE_STAMP,
    DeviceStamp,
    apply_device_event,
    stamp_sender_device,
    start_device_runtime,
)
from world.device_schedule import draw_device_event_schedule
from world.devices import Device
from world.world_plan import WorldPlan, draw_world_plan, spawn_life_streams, spawn_world_streams

# Absolute path to repository root for configuration file resolution.
REPO_ROOT = Path(__file__).resolve().parents[3]
# Device realism configuration file path defining empirical device lifecycle transition rates.
DEVICE_CONFIG_PATH = REPO_ROOT / "config" / "device_realism.yaml"


# Stand-in lightweight dummy citizen data container for isolated unit tests.
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
    WHAT: Tests argument and hardware state verification before mutating citizen mobile device profiles.
    WHY: Proves mismatched person ID, fingerprint before, or OS version raises RuntimeError immediately,
         preventing simulation world state and device lifecycle drift.
    Example:
        # with pytest.raises(RuntimeError):
        #     apply_device_event(person, event, state)
    """
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_old")
    p1 = DummyPerson("p1", dev)
    state = start_device_runtime([p1])
    # 1. Wrong person ID must raise RuntimeError.
    ev_bad_p = make_device_event("p2", 10, "phone_change", "android", 2, 4, "fp_old", "fp_new")
    with pytest.raises(RuntimeError, match="Person mismatch"):
        apply_device_event(p1, ev_bad_p, state)
    # 2. Fingerprint mismatch must raise RuntimeError.
    ev_bad_fp = make_device_event("p1", 10, "phone_change", "android", 2, 4, "fp_other", "fp_new")
    with pytest.raises(RuntimeError, match="FP mismatch"):
        apply_device_event(p1, ev_bad_fp, state)
    # 3. OS generation mismatch must raise RuntimeError.
    ev_bad_os = make_device_event("p1", 10, "phone_change", "android", 1, 4, "fp_old", "fp_new")
    with pytest.raises(RuntimeError, match="OS mismatch"):
        apply_device_event(p1, ev_bad_os, state)


def test_apply_device_event_os_update() -> None:
    """
    WHAT: Verifies that OS update events mutate OS generation without altering hardware serial numbers.
    WHY: Proves rule R5 hardware preservation where OS updates keep identical fingerprints and device IDs.
    Example:
        # apply_device_event(person, os_event, state)
        # assert person.device.fingerprint == "fp_init" and person.device.os_generation == 3
    """
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_init")
    person = DummyPerson("p1", dev)
    state = start_device_runtime([person])
    # 1. Construct valid OS update event keeping fingerprint.
    ev = make_device_event("p1", 50, "os_update", "android", 2, 3, "fp_init", "fp_init")
    apply_device_event(person, ev, state)
    # 2. Check that OS generation increments while fingerprint and device ID stay invariant.
    assert person.device.os_generation == 3
    assert person.device.fingerprint == "fp_init"
    assert person.device.device_id == "dev_p1"


def test_apply_device_event_phone_change() -> None:
    """
    WHAT: Asserts that phone change events update hardware fingerprint, OS generation, and device ID suffix.
    WHY: Represents physical handset replacement, advancing device identifier across sequential upgrades.
    Example:
        # apply_device_event(person, phone_ev, state)
        # assert person.device.device_id == "dev_p1_p1" and person.device.fingerprint == "fp_new"
    """
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_1")
    person = DummyPerson("p1", dev)
    state = start_device_runtime([person])
    # 1. Apply first phone replacement event.
    ev1 = make_device_event("p1", 60, "phone_change", "android", 2, 4, "fp_1", "fp_2")
    apply_device_event(person, ev1, state)
    assert person.device.device_id == "dev_p1_p1"
    assert person.device.fingerprint == "fp_2"
    # 2. Apply second phone replacement event to verify suffix incrementation.
    ev2 = make_device_event("p1", 120, "phone_change", "android", 4, 4, "fp_2", "fp_3")
    apply_device_event(person, ev2, state)
    assert person.device.device_id == "dev_p1_p2"
    assert person.device.fingerprint == "fp_3"


def test_apply_device_event_reinstall() -> None:
    """
    WHAT: Tests that app reinstall events cycle the client fingerprint while preserving physical handset state.
    WHY: Models application resets where hardware device ID and OS version remain completely unchanged.
    Example:
        # apply_device_event(person, reinstall_ev, state)
        # assert person.device.fingerprint == "fp_reinst" and person.device.os_generation == 2
    """
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_orig")
    person = DummyPerson("p1", dev)
    state = start_device_runtime([person])
    # 1. Construct valid reinstall event altering fingerprint only.
    ev = make_device_event("p1", 80, "reinstall", "android", 2, 2, "fp_orig", "fp_reinst")
    apply_device_event(person, ev, state)
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
        # stamp = stamp_sender_device(person, state)
        # assert stamp.is_new_device is False
    """
    # 1. Commercial shop senders without devices receive NO_DEVICE_STAMP.
    empty_state = start_device_runtime([])
    assert stamp_sender_device(None, empty_state) == NO_DEVICE_STAMP
    dev = Device(device_id="dev_p1", platform="android", os_generation=2, fingerprint="fp_init")
    person = DummyPerson("p1", dev)
    state = start_device_runtime([person])
    # 2. First payment on starting hardware is recognized as known.
    s1 = stamp_sender_device(person, state)
    assert s1 == DeviceStamp("fp_init", 2, False)
    # 3. After fingerprint change, first payment marks new device and second payment is recognized.
    person.device.fingerprint = "fp_new"
    s2 = stamp_sender_device(person, state)
    assert s2 == DeviceStamp("fp_new", 2, True)
    s3 = stamp_sender_device(person, state)
    assert s3 == DeviceStamp("fp_new", 2, False)



def _run_sim_with_devices(seed: int = 101) -> tuple[Any, WorldPlan, list[Any], RunResult]:
    """
    WHAT: Constructs a test town world, life streams, and executes calendar simulation with devices.
    WHY: Extracts boilerplate simulation setup into a shared helper to maintain 30-line test limits.
    Args: seed: Integer pseudo-random master seed controlling world and schedule generation.
    Returns: Tuple of world container, plan instance, generated device schedule, and RunResult.
    Example:
        # world, plan, sched, res = _run_sim_with_devices(101)
        # assert len(res.events) > 0
    """
    # 1. Load device realism config and build town world with 60 citizens and 12 merchants.
    ranges = read_background_realism(DEVICE_CONFIG_PATH)
    w = build_world(seed=seed, n_people=60, n_shops=12)
    p_rng, r_rng = spawn_world_streams(seed)
    dev_rng, _ = spawn_life_streams(seed)
    # 2. Draw world plan and device lifecycle event schedule for 61 calendar days.
    plan = draw_world_plan(p_rng, n_days=61)
    sched = draw_device_event_schedule(ranges, dev_rng, w.people, plan.festivals, plan.n_days)
    # 3. Execute calendar simulation with device events and return run result.
    res = run_world(w, plan, r_rng, device_events=sched)
    return w, plan, sched, res


def test_w1_empty_schedule_reproduces_g2_hash_and_counts() -> None:
    """
    WHAT: Runs calendar simulation with empty device event sequence and compares hash and counts to G2.
    WHY: Proves backward compatibility and zero side effects when device lifecycle events
         are omitted from calendar runs.
    Example:
        # run_world(w, plan, r_rng, device_events=())
        # assert _hash_event_list(res.events) == GOLDEN_G2_HASH
    """
    # 1. Build world with seed 101, 60 people, 12 shops matching golden G2 configuration.
    w = build_world(seed=101, n_people=60, n_shops=12)
    p_rng, r_rng = spawn_world_streams(101)
    plan = draw_world_plan(p_rng, n_days=61)
    # 2. Run simulation passing empty device schedule tuple.
    res = run_world(w, plan, r_rng, device_events=())
    # 3. Assert exact reproduction of G2 golden hash and failure breakdown counts from Level 11B preflight.
    gold_h = "12921b27db24a71a00be469709e768408a8e1a52cd568fd5db4f4dcea9d9f8dd"
    gold_counts = {
        "attempts": 2915, "successes": 2865, "failures": 50, "domain_sender_bank": 22,
        "domain_npci_central": 8, "domain_app_or_network": 6, "domain_receiver_bank": 9, "domain_customer": 5,
    }
    assert _hash_event_list(res.events) == gold_h
    assert res.counts == gold_counts
    assert res.device_counts == {"events_applied": 0, "new_device_payments": 0}
    # 4. Print measured verification line for test audit tracking.
    print(f"MEASURED W1: hash={gold_h[:16]}... attempts={res.counts['attempts']} failures={res.counts['failures']}")


def test_w2_determinism_with_device_schedules() -> None:
    """
    WHAT: Asserts identical seeds produce identical device stamps, while different seeds diverge.
    WHY: Enforces strict simulation determinism and reproducibility across multi-stream pseudo-random execution.
    Example:
        # res1 = run_world(w1, plan1, r_rng1, device_events=ev1)
        # assert res1.event_sender_devices == res2.event_sender_devices
    """
    # 1. Execute run 1 with seed 101 including generated device lifecycle events.
    w1, _, _, res1 = _run_sim_with_devices(101)
    # 2. Execute run 2 with identical seed 101 to verify deterministic stamp replay.
    w2, _, _, res2 = _run_sim_with_devices(101)
    # 3. Execute run 3 with different seed 202 to verify schedule divergence.
    w3, _, _, res3 = _run_sim_with_devices(202)
    # 4. Assert identical event hashes and device stamp alignment between runs 1 and 2.
    assert _hash_event_list(res1.events) == _hash_event_list(res2.events)
    assert res1.event_sender_devices == res2.event_sender_devices
    # 5. Assert device stamps diverge under different random seeds.
    assert res1.event_sender_devices != res3.event_sender_devices
    print(f"MEASURED W2: stamps1={len(res1.event_sender_devices)} identical=True diff_seed_diverges=True")


def test_w3_completeness_all_events_applied_final_devices() -> None:
    """
    WHAT: Asserts all scheduled device events are applied and each person reflects their final event state.
    WHY: Proves calendar runner advances device event cursor completely without skipping or dropping items.
    Example:
        # world, plan, sched, res = _run_sim_with_devices(101)
        # assert res.device_counts["events_applied"] == len(sched)
    """
    # 1. Run calendar simulation with active device event schedule.
    w, _, sched, res = _run_sim_with_devices(101)
    assert res.device_counts["events_applied"] == len(sched)
    # 2. Group device events by citizen to locate the terminal event per individual.
    ev_by_person: dict[str, list[Any]] = {}
    for ev in sched:
        ev_by_person.setdefault(ev.person_id, []).append(ev)
    # 3. Verify terminal handset state equals the after-state of the last applied event.
    matched = 0
    for p in w.people:
        if p.person_id in ev_by_person:
            last_ev = ev_by_person[p.person_id][-1]
            assert p.device.fingerprint == last_ev.fingerprint_after
            assert p.device.os_generation == last_ev.os_generation_after
            matched += 1
    # 4. Output measured verification line for test completeness confirmation.
    n_app = res.device_counts["events_applied"]
    print(f"MEASURED W3: total_events={len(sched)} applied={n_app} people_with_events={matched}")


def test_w4_exact_new_device_payment_accounting() -> None:
    """
    WHAT: Asserts new_device_payments matches an independent tally of events followed by sender payments.
    WHY: Guarantees exact telemetry accounting for new hardware usage without trusting runner counters.
    Example:
        # world, plan, sched, res = _run_sim_with_devices(101)
        # assert res.device_counts["new_device_payments"] == expected_count
    """
    # 1. Run simulation and map bank account IDs to citizen person IDs.
    w, _, sched, res = _run_sim_with_devices(101)
    acc_map = {p.account_id: p.person_id for p in w.people}
    pmt_mins: dict[str, list[int]] = {}
    for ev in res.events:
        if ev.from_account in acc_map:
            pmt_mins.setdefault(acc_map[ev.from_account], []).append(ev.minute)
    # 2. Group fingerprint-altering events and independently count subsequent payment adoption.
    fp_events: dict[str, list[Any]] = {}
    for e in sched:
        if e.fingerprint_before != e.fingerprint_after:
            fp_events.setdefault(e.person_id, []).append(e)
    # 3. Count events whose citizen paid before any subsequent fingerprint change.
    expected = 0
    for pid, evs in fp_events.items():
        mins = pmt_mins.get(pid, [])
        for i, ev in enumerate(evs):
            nxt_m = evs[i + 1].minute if i + 1 < len(evs) else float("inf")
            expected += int(any(ev.minute <= m < nxt_m for m in mins))
    # 4. Assert exact match with calendar runner reported new device transaction tally.
    assert res.device_counts["new_device_payments"] == expected
    print(f"MEASURED W4: new_device_payments={res.device_counts['new_device_payments']} independent_count={expected}")



def test_w5_r5_os_update_keeps_fingerprint_increments_os() -> None:
    """
    WHAT: Asserts each OS update preserves device familiarity and increments the OS generation stamp.
    WHY: Directly verifies issue R5 compliance on full simulation datasets, ensuring OS updates are never new devices.
    Example:
        # for ev in os_updates:
        #     assert stamp.is_new_device is False and stamp.os_generation == ev.os_generation_after
    """
    # 1. Run calendar simulation with realistic device events.
    w, _, sched, res = _run_sim_with_devices(101)
    acc_map = {p.account_id: p.person_id for p in w.people}
    payments = list(zip(res.events, res.event_sender_devices))
    # 2. Filter schedule for OS update events and verify subsequent payment stamps.
    os_updates = [e for e in sched if e.kind == "os_update"]
    checked = 0
    for ev in os_updates:
        subs = [
            (pmt, s) for pmt, s in payments
            if acc_map.get(pmt.from_account) == ev.person_id and pmt.minute >= ev.minute
        ]
        # 3. Assert first payment after OS update is not flagged as new device and has updated OS.
        if subs:
            _, stamp = subs[0]
            assert stamp.is_new_device is False
            assert stamp.os_generation == ev.os_generation_after
            checked += 1
    # 4. Assert at least one OS update was evaluated and output measured summary.
    assert checked > 0
    print(f"MEASURED W5: total_os_updates={len(os_updates)} verified_payments={checked}")


def test_w6_shop_senders_receive_no_device_stamp() -> None:
    """
    WHAT: Asserts all transactions initiated from merchant store accounts carry the NO_DEVICE_STAMP sentinel.
    WHY: Proves commercial entity payments (such as salary disbursements) do not claim consumer handset telemetry.
    Example:
        # world, plan, sched, res = _run_sim_with_devices(101)
        # assert all(s == NO_DEVICE_STAMP for ev, s in zip(res.events, res.event_sender_devices) if is_shop(ev))
    """
    # 1. Run calendar simulation and isolate merchant store account IDs.
    w, _, _, res = _run_sim_with_devices(101)
    shop_accs = {s.account_id for s in w.shops}
    # 2. Filter all executed transactions where the debit origin is a shop account.
    shop_legs = [
        (ev, s) for ev, s in zip(res.events, res.event_sender_devices)
        if ev.from_account in shop_accs
    ]
    # 3. Verify every shop debit payment carries the empty NO_DEVICE_STAMP record.
    assert len(shop_legs) > 0
    for ev, s in shop_legs:
        assert s == NO_DEVICE_STAMP
    # 4. Output measured telemetry line confirming shop sentinel consistency.
    print(f"MEASURED W6: total_shop_payments={len(shop_legs)} all_no_device_stamp=True")


def test_w7_ledger_invariants_pass_with_devices() -> None:
    """
    WHAT: Evaluates double-entry accounting integrity checkers on all bank ledgers after running with devices.
    WHY: Proves device lifecycle state and metadata stamping do not corrupt monetary conservation or balances.
    Example:
        # world, plan, sched, res = _run_sim_with_devices(101)
        # assert check_money_conserved(world.ledgers, total_opening) == []
    """
    # 1. Build world and compute total opening balance across all bank accounts.
    ranges = read_background_realism(DEVICE_CONFIG_PATH)
    w = build_world(seed=101, n_people=60, n_shops=12)
    opening = sum(ldg.balance_of(acc) for ldg in w.ledgers for acc in ldg.accounts)
    # 2. Execute 61-day simulation with active device lifecycle events.
    p_rng, r_rng = spawn_world_streams(101)
    dev_rng, _ = spawn_life_streams(101)
    plan = draw_world_plan(p_rng, n_days=61)
    sched = draw_device_event_schedule(ranges, dev_rng, w.people, plan.festivals, plan.n_days)
    run_world(w, plan, r_rng, device_events=sched)
    # 3. Verify non-negative balances, entry sum equality, and total money conservation.
    assert check_no_negative_balances(w.ledgers) == []
    assert check_balances_equal_entry_sums(w.ledgers) == []
    assert check_money_conserved(w.ledgers, opening) == []
    # 4. Output measured accounting confirmation line for audit logs.
    print(f"MEASURED W7: opening_paise={opening} no_neg=True sums_match=True conserved=True")


def test_w8_injected_schedule_stamps_payments_in_order() -> None:
    """
    WHAT: Asserts hand-crafted OS update and phone change events produce exact chronological stamps on payments.
    WHY: Validates precision of payment-by-payment stamping across sequential handset lifecycle transitions.
    Example:
        # res = run_world(w, plan, r_rng, device_events=[os_ev, phone_ev])
        # assert stamps[0].is_new_device is False and stamps[2].is_new_device is True
    """
    # 1. Build world and retrieve person 0 device profile and original device identifier.
    w = build_world(seed=101, n_people=60, n_shops=12)
    p0 = w.people[0]
    orig_dev = p0.device.device_id
    fp0, os0, plat = p0.device.fingerprint, p0.device.os_generation, p0.device.platform
    # 2. Inject hand-built OS update at minute 1130 and phone replacement at minute 2000.
    ev1 = make_device_event(p0.person_id, 1130, "os_update", plat, os0, os0 + 1, fp0, fp0)
    ev2 = make_device_event(p0.person_id, 2000, "phone_change", plat, os0 + 1, os0 + 2, fp0, "fp_new_hw")
    p_rng, r_rng = spawn_world_streams(101)
    plan = draw_world_plan(p_rng, n_days=61)
    res = run_world(w, plan, r_rng, device_events=[ev1, ev2])
    # 3. Assert p0 final device identifier reflects phone replacement suffix.
    assert p0.device.device_id == f"{orig_dev}_p1"
    # 4. Collect chronological stamps on person 0 payments and assert expected transitions.
    stamps = [s for ev, s in zip(res.events, res.event_sender_devices) if ev.from_account == p0.account_id]
    assert stamps[0] == DeviceStamp(fp0, os0, False)
    assert stamps[1] == DeviceStamp(fp0, os0 + 1, False)
    assert stamps[2] == DeviceStamp("fp_new_hw", os0 + 2, True)
    assert stamps[3] == DeviceStamp("fp_new_hw", os0 + 2, False)
    # 5. Print measured verification line confirming orderly injected transitions.
    print(f"MEASURED W8: p0_payments={len(stamps)} s0={stamps[0]} s1={stamps[1]} s2={stamps[2]} s3={stamps[3]}")


def test_d1_person_dataclass_fields_clean_after_run() -> None:
    """
    WHAT: Asserts every Person entity retains only its declared dataclass fields after a full calendar run.
    WHY: Proves device event processing and phone replacements use runtime state without monkey-patching attributes.
         Prevents state leaks onto domain entities and ensures Person instances remain clean dataclasses.
    Example:
        # for p in world.people:
        #     assert set(vars(p)) == {f.name for f in dataclasses.fields(p)}
    """
    # 1. Execute full simulation run with drawn device event schedule.
    world, plan, sched, res = _run_sim_with_devices(seed=101)
    # 2. Extract declared field names from the Person dataclass definition.
    expected_fields = {f.name for f in dataclasses.fields(world.people[0])}
    # 3. Assert every citizen's instance attributes exactly match declared fields.
    for p in world.people:
        diff = set(vars(p)) - expected_fields
        assert set(vars(p)) == expected_fields, f"Person {p.person_id} has extraneous attributes: {diff}"
    # 4. Print measured line confirming no hidden monkey-patched attributes exist.
    print(f"MEASURED D1: people_count={len(world.people)} all_fields_clean=True")


def test_d2_two_phone_changes_and_reinstall_device_ids() -> None:
    """
    WHAT: Asserts two successive phone change events advance device_id suffix from _p1 to _p2, with reinstall between.
    WHY: Proves device runtime state tracks physical handset upgrades while app reinstalls preserve device ID.
         Enforces that device ID numbering relies on original ID and cumulative phone replacements in runtime state.
    Example:
        # apply_device_event(person, ev1, state); assert person.device.device_id == f"{orig}_p1"
        # apply_device_event(person, reinst, state); assert person.device.device_id == f"{orig}_p1"
        # apply_device_event(person, ev2, state); assert person.device.device_id == f"{orig}_p2"
    """
    # 1. Initialize test person with starting handset and allocate explicit runtime state.
    dev = Device(device_id="dev_orig", platform="android", os_generation=2, fingerprint="fp_0")
    person = DummyPerson("p1", dev)
    state = start_device_runtime([person])
    # 2. Apply first phone change event and verify suffix advances to _p1.
    ev1 = make_device_event("p1", 10, "phone_change", "android", 2, 3, "fp_0", "fp_1")
    apply_device_event(person, ev1, state)
    assert person.device.device_id == "dev_orig_p1"
    # 3. Apply intermediate app reinstall and verify hardware device identifier remains unchanged.
    ev_re = make_device_event("p1", 20, "reinstall", "android", 3, 3, "fp_1", "fp_re")
    apply_device_event(person, ev_re, state)
    assert person.device.device_id == "dev_orig_p1"
    # 4. Apply second phone change event and verify suffix advances to _p2.
    ev2 = make_device_event("p1", 30, "phone_change", "android", 3, 4, "fp_re", "fp_2")
    apply_device_event(person, ev2, state)
    assert person.device.device_id == "dev_orig_p2"
    # 5. Output measured line confirming sequential device identifiers.
    print(f"MEASURED D2: p1_id=dev_orig_p1 reinstall_id=dev_orig_p1 p2_id={person.device.device_id}")

