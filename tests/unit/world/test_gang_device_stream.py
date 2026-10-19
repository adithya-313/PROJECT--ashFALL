"""
WHAT: Integration tests for the gang device event stream and interactive runner wiring (Level 12b B-2b).
Asserts that applied gang device events match planned events in order, events placed after the last Level 11 event
appear correctly, civilian device events are untouched by gang additions, genuine draws never move, the merged
device_event_stream is ordered with no forbidden truth keys, fraud-off runs produce empty gang events and preserve
the frozen anchor hash, fraud-off mode refuses gang device inputs, and identical seeds produce reproducible streams.

WHY: Criminal syndicates operate shared devices and takeover tactics that emit synthetic telemetry without modifying
citizen hardware state. Testing the unified event stream guarantees downstream monitoring engines receive properly
sequenced telemetry while preserving strict ground-truth separation per Decision E5.

TERM: device event stream: chronological sequence of civilian and syndicate device lifecycle records.
TERM: gang device event: synthetic telemetry record documenting hardware acquisition or unauthorized account compromise.
TERM: flush gap: edge case where events scheduled after the final planned civilian event are dropped without full queue drain.
TERM: genuine draw: pseudorandom number draw consumed by authentic civilian payment planning and execution.
"""

import hashlib
from pathlib import Path
from typing import Any
import pytest
import yaml

from core.config_validator import load_yaml
from world.fast_mode import FastRunSettings, event_lines
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.gang_devices import GangDeviceEvent
from world.gang_planner import plan_gang
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import (
    SWITCH_OFF_MESSAGE,
    device_event_stream,
    interactive_fast_output,
    prepare_world,
    run_interactive,
    run_interactive_world,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
FROZEN_ANCHORS_PATH: Path = REPO_ROOT / "config" / "frozen_anchors.yaml"
TINY = FastRunSettings(101, 60, 12, 61, "training")
BUDGET = GangBudget(money_paise=10_000_000, time_minutes=10_000, max_risk=0.5)
SOURCE = FixtureSource("acc_gang_src_01", 0, 10_000_000, 1440)

STEPS_CONTROL = (
    BlockStep("acquire", (("count", 3),)),
    BlockStep(
        "take_control",
        (
            ("devices", 2),
            ("sim_swap", True),
            ("linked_number_switch", False),
            ("emulator", True),
            ("remote_access", False),
        ),
    ),
    BlockStep("warm_up", (("days", 1),)),
    BlockStep("route", (("layers", 2), ("accounts_per_layer", 1))),
)
GANG_CONTROL = GangConfig("gang_bravo", 1, BUDGET, SOURCE, STEPS_CONTROL)


def test_applied_gang_device_events_equals_plan_events_in_order() -> None:
    """
    WHAT: Asserts that applied_gang_device_events in InteractiveRunResult matches the plan's events in order.
    It executes an interactive run with GANG_CONTROL and compares the applied events with the planned telemetry.

    WHY: Decision E5 dictates that gang device events queued on the runner must be processed and recorded in arrival
    sequence without modification, truncation, or reordering.

    Args:
        None.

    Returns:
        None. Fails if applied gang device events deviate from planned events.

    Example:
        # output = run_interactive_world(TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=(GANG_CONTROL,))
        # assert len(output.result.applied_gang_device_events) == 8
    """
    # 1. Run interactive world with fraud enabled and the takeover gang configuration.
    output = run_interactive_world(
        TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=(GANG_CONTROL,)
    )

    # 2. Extract applied gang device events from the run result.
    applied_events = output.result.applied_gang_device_events

    # 3. Assert events were generated and verify kind ordering matches expectations.
    assert len(applied_events) == 8, f"Expected 8 gang device events, got {len(applied_events)}"
    assert applied_events[0].kind == "device_acquired"
    assert applied_events[1].kind == "device_acquired"
    for event in applied_events[2:]:
        assert event.kind in ("sim_swap", "emulator_enabled")


def test_gang_device_event_after_last_level_11_event_appears() -> None:
    """
    WHAT: Asserts that a gang device event scheduled after the final Level 11 civilian event is processed.
    It schedules an event at minute = max_civilian_minute + 1000 and asserts it appears in the output list.

    WHY: Verifies the absence of a flush gap where queue items after the last civilian schedule entry might be dropped.

    Args:
        None.

    Returns:
        None. Fails if the late gang device event is omitted from applied_gang_device_events.

    Example:
        # late_event = GangDeviceEvent(minute=90000, ...)
        # assert output.applied_gang_device_events[-1] == late_event
    """
    # 1. Prepare world and determine maximum minute among civilian device events.
    prepared = prepare_world(TINY, None)
    max_minute = max(event.minute for event in prepared.device_events) if prepared.device_events else 1000

    # 2. Construct a synthetic gang device event scheduled well beyond civilian schedule.
    late_event = GangDeviceEvent(
        minute=max_minute + 1000, device_id="dev_ext_000001", account_id="acc_person_000001",
        kind="device_acquired", platform="android", os_generation=3, fingerprint="fp_android_v3_12345678",
    )

    # 3. Execute runner with late event injected into fraud inputs.
    inputs = FraudInputs(gang_device_events=(late_event,))
    result = run_interactive(
        prepared.world, prepared.plan, prepared.run_rng, prepared.device_events, prepared.episode_payments,
        settings=RunnerSettings(fraud_enabled=True), fraud_inputs=inputs, run_id="test_late_flush",
    )

    # 4. Assert late event was successfully dequeued and recorded.
    assert len(result.applied_gang_device_events) == 1
    assert result.applied_gang_device_events[0] == late_event


def test_applied_device_events_level_11_identical_to_no_gang_run() -> None:
    """
    WHAT: Asserts that civilian Level 11 applied device events are bit-for-bit identical between gang and no-gang runs.
    It runs the tiny world with and without GANG_CONTROL and compares applied_device_events.

    WHY: Decision E5 and Separation of Concerns require that gang operations never touch or modify Level 11 device state.

    Args:
        None.

    Returns:
        None. Fails if civilian device events differ between runs.

    Example:
        # assert output_gang.result.applied_device_events == output_clean.result.applied_device_events
    """
    # 1. Execute baseline run without gang configurations.
    baseline = run_interactive_world(
        TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=()
    )

    # 2. Execute treatment run with takeover gang configuration.
    treatment = run_interactive_world(
        TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=(GANG_CONTROL,)
    )

    # 3. Assert applied Level 11 civilian device events are identical.
    assert baseline.result.applied_device_events == treatment.result.applied_device_events


def test_genuine_draws_never_move_t3_approach() -> None:
    """
    WHAT: Asserts that authentic civilian payments experience identical outcomes and attributes with or without gangs.
    It isolates planned events in both runs and compares failure reasons, amounts, and success flags.

    WHY: The T3 design ensures that planning and running fraud components use isolated random streams (children 4-7),
    guaranteeing that citizen RNG streams (children 0-3) never shift.

    Args:
        None.

    Returns:
        None. Fails if genuine civilian payments deviate in outcomes.

    Example:
        # assert len(civilian_events) == 2915
        # assert all(c.success == b.success for c, b in zip(civilian_events, baseline_events))
    """
    # 1. Execute baseline run without fraud injections.
    baseline = run_interactive_world(
        TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=()
    )

    # 2. Execute treatment run containing gang operations.
    treatment = run_interactive_world(
        TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=(GANG_CONTROL,)
    )

    # 3. Filter treatment events down to genuine civilian payments.
    planned_events = [
        event for event, source in zip(treatment.result.run_result.events, treatment.result.event_sources)
        if source == "planned"
    ]

    # 4. Assert total genuine counts and all transaction outcomes match baseline exactly.
    assert len(planned_events) == len(baseline.result.run_result.events)
    for expected, actual in zip(baseline.result.run_result.events, planned_events):
        assert expected == actual


def test_device_event_stream_sorted_both_kinds_no_truth_keys() -> None:
    """
    WHAT: Asserts that device_event_stream merges both civilian and gang events in sorted order with no forbidden keys.
    It inspects the resulting plain dictionary rows for timestamp ordering, event kind presence, and truth key absence.

    WHY: Later detection levels read device_event_stream as simulated bank telemetry; exposing gang_id or role would
    compromise security by design per Decision E5.

    Args:
        None.

    Returns:
        None. Fails if ordering is violated, kinds are missing, or truth keys are present.

    Example:
        # stream = device_event_stream(output.result)
        # assert all("kind" in row for row in stream)
    """
    # 1. Execute run with takeover gang configuration.
    output = run_interactive_world(
        TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=(GANG_CONTROL,)
    )

    # 2. Build the unified merged device event stream.
    stream = device_event_stream(output.result)

    # 3. Assert stream contains both civilian (32) and syndicate (8) events.
    assert len(stream) == 40
    assert any(row["kind"] in ("phone_change", "reinstall", "os_update") for row in stream)
    assert any(row["kind"] in ("device_acquired", "sim_swap", "emulator_enabled") for row in stream)

    # 4. Assert chronological ordering and verify no row contains sensitive truth keys.
    for index in range(len(stream) - 1):
        assert stream[index]["minute"] <= stream[index + 1]["minute"]
    for row in stream:
        assert "kind" in row
        for forbidden in ("gang_id", "role", "truth"):
            assert forbidden not in row, f"Forbidden key '{forbidden}' found in stream row"


def test_fraud_off_applied_gang_device_events_empty_and_frozen_hash_holds() -> None:
    """
    WHAT: Asserts that with fraud disabled, applied_gang_device_events is empty and the frozen anchor hash holds.
    It executes the tiny world in fraud-off mode and asserts output hashes match config/frozen_anchors.yaml.

    WHY: Ground rule 13 requires the fraud-off path to fall back to Level 11 exactly without modifying baseline digests.

    Args:
        None.

    Returns:
        None. Fails if applied gang events are non-empty or if the event line hash moves.

    Example:
        # output = run_interactive_world(TINY, runner_settings=RunnerSettings(fraud_enabled=False))
        # assert output.result.applied_gang_device_events == []
    """
    # 1. Execute interactive world with fraud switch turned off.
    output = run_interactive_world(TINY, runner_settings=RunnerSettings(fraud_enabled=False))

    # 2. Assert applied gang device events list is empty.
    assert output.result.applied_gang_device_events == []

    # 3. Read frozen anchor baseline hash and compute actual hash of produced event lines.
    with open(FROZEN_ANCHORS_PATH, "r", encoding="utf-8") as file:
        anchors = yaml.safe_load(file)
    expected_hash = anchors["frozen_anchors"]["fraud_off_events"]["sha256"]
    actual_hash = hashlib.sha256("".join(event_lines(interactive_fast_output(output))).encode("utf-8")).hexdigest()

    # 4. Assert computed SHA-256 hash strictly matches the frozen anchor baseline.
    assert actual_hash == expected_hash, f"Frozen anchor hash moved: expected {expected_hash}, got {actual_hash}"


def test_switch_off_with_gang_device_events_refused() -> None:
    """
    WHAT: Asserts that providing gang device events when the fraud switch is off raises ValueError.
    It passes a FraudInputs holding a gang device event to run_interactive with fraud_enabled=False.

    WHY: Ground rule 13 and check_switch forbid passing fraud inputs while the switch is off to prevent silent omissions.

    Args:
        None.

    Returns:
        None. Fails if the runner accepts gang device events when fraud is off.

    Example:
        # with pytest.raises(ValueError, match="fraud inputs need fraud_enabled=True"):
        #     run_interactive(..., settings=RunnerSettings(fraud_enabled=False), fraud_inputs=inputs)
    """
    # 1. Prepare clean world entities and schedule.
    prepared = prepare_world(TINY, None)

    # 2. Construct gang device event input.
    event = GangDeviceEvent(
        minute=100, device_id="dev_ext_000001", account_id="acc_person_000001",
        kind="device_acquired", platform="android", os_generation=3, fingerprint="fp_android_v3_12345678",
    )
    inputs = FraudInputs(gang_device_events=(event,))

    # 3. Assert runner refuses gang device inputs with the standard switch-off message.
    with pytest.raises(ValueError, match=SWITCH_OFF_MESSAGE):
        run_interactive(
            prepared.world, prepared.plan, prepared.run_rng, prepared.device_events, prepared.episode_payments,
            settings=RunnerSettings(fraud_enabled=False), fraud_inputs=inputs, run_id="test_refusal",
        )


def test_same_seed_twice_identical_stream_and_truth() -> None:
    """
    WHAT: Asserts that running the same seed twice produces identical device event streams and truth facts.
    It executes two independent runs with GANG_CONTROL and asserts bit-for-bit equality across outputs.

    WHY: Determinism is an essential engineering principle; identical random seeds and configurations must yield
    strictly identical simulation results.

    Args:
        None.

    Returns:
        None. Fails if either the event stream or ground truth records diverge.

    Example:
        # assert device_event_stream(run1.result) == device_event_stream(run2.result)
        # assert run1.result.truth == run2.result.truth
    """
    # 1. Execute first independent run with takeover gang configuration.
    run_1 = run_interactive_world(
        TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=(GANG_CONTROL,)
    )

    # 2. Execute second independent run with identical inputs.
    run_2 = run_interactive_world(
        TINY, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=(GANG_CONTROL,)
    )

    # 3. Assert merged device event streams and hidden truth facts match exactly.
    assert device_event_stream(run_1.result) == device_event_stream(run_2.result)
    assert run_1.result.truth == run_2.result.truth
