"""
Tests for building and writing label events from a finished run (world/label_builder.py).
Covers that true complaints name only successful victim payments, that a victim whose payments all failed never
complains, that the label file is repeatable per seed, that building labels leaves the run untouched, that a
fraud-off world gets only false complaints, and that a stress layer changes label delays through the run's own
layered world ranges.

TERM: label file: the JSON Lines file of label events written by write_label_events_file.
TERM: layered world ranges: the run's world_ranges after a stress layer such as complaint-backlog was applied.
"""

import copy
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from core.config_validator import load_yaml
from showcase.level_12_support_b import get_showcase_b_settings
from showcase.level_12_support_c import get_showcase_c_outage_run, load_showcase_c_fixture
from world.device_schedule import MINUTES_PER_DAY
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.label_builder import (
    build_label_events,
    collect_genuine_payments,
    collect_victim_losses,
    write_label_events_file,
)
from world.layers import apply_layers, read_layer
from world.run_identity import REPO_ROOT
from world.scam_config import resolve_remainder_scam_type
from world.scam_planner import SCAM_INVESTMENT, SCAM_JOB
from world.truth_records import SCAM_STEP_PAYOUT_BACK, SCAM_STEP_VICTIM_PAYMENT
from world.world_configs import read_world_configs

FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
BACKLOG_LAYER_PATH = REPO_ROOT / "scenarios" / "stress" / "complaint-backlog.yaml"
BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)


def run_with_gang(
    gang: GangConfig, settings: FastRunSettings | None = None, configs: Any = None
) -> InteractiveWorldOutput:
    """Runs the showcase world (or the given settings) with fraud on and one gang."""
    base_settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        settings if settings is not None else base_settings, RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,), configs=configs, code_version=code_version,
    )


def make_large_scam_gang() -> GangConfig:
    """Builds a scam gang with many victims so several scam types appear in one world."""
    steps = (
        BlockStep("acquire", (("count", 6),)),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 12))),
    )
    return GangConfig("g_large", 1, BUDGET, None, steps, 1440)


@pytest.fixture(scope="module")
def cashout_output() -> InteractiveWorldOutput:
    """Runs the Part C victim-to-cash-out fixture gang (showcase/fixtures/level_12/gang_victims_cashout.yaml)."""
    return run_with_gang(load_showcase_c_fixture())


@pytest.fixture(scope="module")
def large_output() -> InteractiveWorldOutput:
    """Runs a larger scam gang once, shared by the tests that need many victims."""
    return run_with_gang(make_large_scam_gang())


def with_every_victim_complaining(output: InteractiveWorldOutput) -> InteractiveWorldOutput:
    """
    Returns the output with the victim complaint share range set to exactly 1.0 for its split, in a copy of the
    effective configuration, so every victim loss complains and a test cannot pass for lack of complaints.
    """
    forced_config = copy.deepcopy(output.effective_config)
    share_entry = forced_config["world_configs"]["world_ranges"]["parameters"]["victim_complaint_share"]
    share_entry[forced_config["settings"]["range_name"]] = [1.0, 1.0]
    return dataclasses.replace(output, effective_config=forced_config)


def fraud_story_by_payment_id(output: InteractiveWorldOutput) -> dict[str, Any]:
    """Maps the payment id of every fraud-source event to its payment story."""
    result = output.result
    stories: dict[str, Any] = {}
    for position, source in enumerate(result.event_sources):
        if source == "fraud":
            input_index = result.event_fraud_input_index[position]
            assert input_index is not None
            stories[result.payment_ids[position]] = result.truth.payment_stories[input_index]
    return stories


def event_by_payment_id(output: InteractiveWorldOutput) -> dict[str, Any]:
    """Maps every payment id of the run to its event."""
    result = output.result
    return {result.payment_ids[i]: event for i, event in enumerate(result.run_result.events)}


def test_true_complaints_name_only_successful_victim_payments_of_the_complainant(
    cashout_output: InteractiveWorldOutput,
) -> None:
    """
    Checks fraud-id complaints name successful victim_payment events of the victim, and no payout_back id appears.
    The share is forced to 1.0 so every loss complains; the drawn share of this world gives no complaint at all.
    """
    labels = build_label_events(with_every_victim_complaining(cashout_output))
    stories = fraud_story_by_payment_id(cashout_output)
    events = event_by_payment_id(cashout_output)
    losses = collect_victim_losses(cashout_output)
    assert len(losses) > 0

    fraud_labels = [label for label in labels if label.payment_ids[0] in stories]
    for label in labels:
        for payment_id in label.payment_ids:
            if payment_id in stories:
                assert stories[payment_id].scam_step != SCAM_STEP_PAYOUT_BACK
    for label in fraud_labels:
        for payment_id in label.payment_ids:
            assert stories[payment_id].scam_step == SCAM_STEP_VICTIM_PAYMENT
            assert events[payment_id].status == "success"
            assert events[payment_id].from_account == label.complainant_account
            assert events[payment_id].to_account == label.reported_account

    # With the share at 1.0 each loss complains exactly once, about all of its successful victim payments.
    assert len(fraud_labels) == len(losses)
    assert {label.payment_ids for label in fraud_labels} == {loss.payment_ids for loss in losses}


def test_a_victim_whose_payments_all_failed_never_complains() -> None:
    """Checks the pinned outage run: its only victim payment failed, so there is no loss and no fraud complaint."""
    outage_output = get_showcase_c_outage_run()
    stories = fraud_story_by_payment_id(outage_output)
    victim_events = [
        event for position, event in enumerate(outage_output.result.run_result.events)
        if outage_output.result.event_sources[position] == "fraud"
        and stories[outage_output.result.payment_ids[position]].scam_step == SCAM_STEP_VICTIM_PAYMENT
    ]
    assert len(victim_events) > 0
    assert all(event.status != "success" for event in victim_events)

    assert collect_victim_losses(outage_output) == []
    # Even with every loss forced to complain, the victim whose payment failed produces no fraud complaint.
    labels = build_label_events(with_every_victim_complaining(outage_output))
    assert all(label.payment_ids[0] not in stories for label in labels)


def test_same_seed_gives_the_same_label_file_and_another_seed_a_different_one(
    cashout_output: InteractiveWorldOutput, tmp_path: Path
) -> None:
    """Checks two runs of one seed write identical label files and a run of another seed writes a different one."""
    gang = load_showcase_c_fixture()
    first_hash = write_label_events_file(build_label_events(cashout_output), tmp_path / "first.jsonl")
    second_output = run_with_gang(gang)
    second_hash = write_label_events_file(build_label_events(second_output), tmp_path / "second.jsonl")
    settings, _version = get_showcase_b_settings()
    other_output = run_with_gang(gang, dataclasses.replace(settings, seed=settings.seed + 1))
    other_hash = write_label_events_file(build_label_events(other_output), tmp_path / "other.jsonl")

    assert first_hash == second_hash
    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()
    assert first_hash != other_hash
    assert hashlib.sha256((tmp_path / "first.jsonl").read_bytes()).hexdigest() == first_hash


def test_building_labels_does_not_change_the_run(cashout_output: InteractiveWorldOutput) -> None:
    """Checks the events, payment ids and effective configuration are identical before and after building labels."""
    def fingerprint() -> tuple[str, list[str], str]:
        """Hashes the run's events and returns it with the payment ids and a text form of the configuration."""
        events_text = repr(cashout_output.result.run_result.events)
        return (
            hashlib.sha256(events_text.encode("utf-8")).hexdigest(),
            list(cashout_output.result.payment_ids),
            repr(cashout_output.effective_config),
        )

    before = fingerprint()
    build_label_events(cashout_output)
    build_label_events(cashout_output)
    assert fingerprint() == before


def test_fraud_off_world_gets_only_false_complaints() -> None:
    """
    Checks a fraud-off tiny world gives only false complaints: remainder claimed type, every id a planned event.

    The configured false rate is below 2 per 100,000, so this world would draw no complaint at all. The test
    raises the training range of the rate to 20,000 in a copy of the effective configuration so the check has
    complaints to look at; nothing else changes.
    """
    tiny = FastRunSettings(
        FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"]
    )
    output = run_interactive_world(tiny, RunnerSettings(fraud_enabled=False), code_version="test-version-labels")
    boosted_config = copy.deepcopy(output.effective_config)
    ranges_parameters = boosted_config["world_configs"]["world_ranges"]["parameters"]
    rate_entry = ranges_parameters["false_complaints_per_100000_genuine_payments"]
    rate_entry[boosted_config["settings"]["range_name"]] = [20_000.0, 20_000.0]
    boosted_output = dataclasses.replace(output, effective_config=boosted_config)

    labels = build_label_events(boosted_output)

    planned_ids = {
        output.result.payment_ids[position] for position, source in enumerate(output.result.event_sources)
        if source == "planned"
    }
    assert collect_victim_losses(output) == []
    assert len(labels) > 0
    for label in labels:
        assert label.claimed_type == resolve_remainder_scam_type()
        assert label.payment_ids[0] in planned_ids
    assert len(collect_genuine_payments(output)) > 0


def test_complaint_backlog_layer_moves_scam_complaint_delays_into_the_layered_bounds() -> None:
    """Checks investment and job complaints of a layered world arrive inside the layer's day ranges, not the base."""
    layer = read_layer(BACKLOG_LAYER_PATH)
    layered_configs, _changes = apply_layers(read_world_configs(), [layer])
    output = run_with_gang(make_large_scam_gang(), configs=layered_configs)
    split = output.effective_config["settings"]["range_name"]
    layer_ranges = layer.changes["world_ranges"]
    investment_low, investment_high = (
        round(value * MINUTES_PER_DAY) for value in layer_ranges["complaint_delay_investment_scam_days"][split]
    )
    job_low, job_high = (
        round(value * MINUTES_PER_DAY) for value in layer_ranges["complaint_delay_job_scam_days"][split]
    )
    bounds = {SCAM_INVESTMENT: (investment_low, investment_high), SCAM_JOB: (job_low, job_high)}

    labels = build_label_events(output)

    stories = fraud_story_by_payment_id(output)
    losses = {loss.payment_ids: loss for loss in collect_victim_losses(output)}
    checked = 0
    for label in labels:
        if label.payment_ids[0] not in stories or label.claimed_type not in bounds:
            continue
        loss = losses[label.payment_ids]
        low, high = bounds[label.claimed_type]
        assert low <= label.arrival_minute - loss.last_paid_minute <= high
        checked += 1
    assert checked > 0
