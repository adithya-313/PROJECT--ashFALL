"""
WHAT: Support utilities and execution caching for Level 12 Showcase Part E checks (21 through 24).
Loads the Part E fixtures and runs the complaints world and the bulk-confirmation world once each, caching the
results so every check that needs them shares one real run.

WHY: Part E shows labels: complaints about payments, and police crackdowns that confirm mules in bulk. Every key
number is measured from a real run of a fixture, and the fixtures keep the scenarios in files instead of in check
code. The bulk fixture values are a labelled what-if, copied into a copy of the run's effective configuration
because the real layer rules reject them (they sit outside the real test ranges on purpose).

TERM: label event: a complaint a bank receives about a payment, with no field saying whether it is true.
TERM: crackdown: one police action that confirms many mule accounts of one region at once.
TERM: fixture layer: showcase/fixtures/level_12/bulk_short_lag.yaml, a labelled non-real range set (issue 12-12).
TERM: cached run: an in-memory simulation result reused across check runners to avoid running it twice.
"""

import copy
import dataclasses
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from showcase.level_12_support_b import get_showcase_b_settings
from showcase.level_12_support_c import load_showcase_c_fixture
from world.device_schedule import MINUTES_PER_DAY
from world.gang_config import GangConfig, gang_config_from_plain
from world.interactive_loop import build_component_streams
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.label_builder import build_bulk_confirmations, build_label_events
from world.label_records import (
    BulkConfirmationResult,
    LabelEvent,
    LabelWorldSettings,
    draw_label_world_settings,
    load_labels_config,
)
from world.layers import read_layer
from world.run_identity import load_runner_config

FIXTURE_DIR: Path = Path(__file__).resolve().parent / "fixtures" / "level_12"
COMPLAINTS_FIXTURE_PATH: Path = FIXTURE_DIR / "gang_complaints.yaml"
BULK_LAYER_PATH: Path = FIXTURE_DIR / "bulk_short_lag.yaml"
CODE_VERSION_E: str = "showcase-12e"
DELAY_DAYS_NAME: str = "bulk_confirmation_delay_days"

_CACHED_COMPLAINTS: tuple[InteractiveWorldOutput, list[LabelEvent], LabelWorldSettings] | None = None
_CACHED_BULK: tuple[InteractiveWorldOutput, BulkConfirmationResult, tuple[int, int]] | None = None


def load_complaints_gang(path: Path = COMPLAINTS_FIXTURE_PATH) -> GangConfig:
    """
    WHAT: Loads the Part E complaints gang fixture and returns its validated GangConfig.
    WHY: Keeps the gang typology in a fixture file; it goes through the same validation as any gang config.
    Args:
        path: Filesystem Path of the fixture YAML file.
    Returns:
        The GangConfig of the fixture.
    Example:
        assert load_complaints_gang().gang_id == "gang_complaints"
    """
    return gang_config_from_plain(load_yaml(path))


def fraud_story_by_payment_id(output: InteractiveWorldOutput) -> dict[str, Any]:
    """
    WHAT: Maps the payment id of every fraud-source event of a run to its payment story.
    WHY: Truth (stories) is world side knowledge; the showcase may read it in memory to judge the labels.
    Args:
        output: The finished InteractiveWorldOutput.
    Returns:
        payment id -> PaymentStory for every fraud event.
    Raises:
        ValueError: If a fraud event has no valid story index.
    Example:
        stories = fraud_story_by_payment_id(output); assert all(hasattr(s, "scam_step") for s in stories.values())
    """
    result = output.result
    stories: dict[str, Any] = {}
    for position, source in enumerate(result.event_sources):
        if source != "fraud":
            continue
        input_index = result.event_fraud_input_index[position]
        if input_index is None or not 0 <= input_index < len(result.truth.payment_stories):
            raise ValueError(f"Fraud event {result.payment_ids[position]} has no valid payment story")
        stories[result.payment_ids[position]] = result.truth.payment_stories[input_index]
    return stories


def get_complaints_run() -> tuple[InteractiveWorldOutput, list[LabelEvent], LabelWorldSettings]:
    """
    WHAT: Runs the complaints gang on the seed-101 frozen-anchor world with fraud on, builds its label events with
    the world's own drawn complaint share (nothing forced), and also returns that drawn share's settings.
    WHY: Check 21 needs a real run, its complaints and the share the world drew for itself. The settings are drawn
    again from the same stream, in the same order the builder uses, so they are the values the builder drew.
    Args:
        None.
    Returns:
        The InteractiveWorldOutput, its label events and its LabelWorldSettings.
    Example:
        output, labels, settings = get_complaints_run(); assert 0.0 <= settings.victim_complaint_share <= 1.0
    """
    global _CACHED_COMPLAINTS
    if _CACHED_COMPLAINTS is not None:
        return _CACHED_COMPLAINTS
    settings, _ = get_showcase_b_settings()
    output = run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True), gang_configs=(load_complaints_gang(),),
        code_version=CODE_VERSION_E,
    )
    labels = build_label_events(output)
    stream_name = "label_world_settings"
    children = load_runner_config()["random_stream_children"]
    stream = build_component_streams(output.effective_config["settings"]["seed"], {stream_name: children[stream_name]})
    drawn = draw_label_world_settings(
        output.effective_config["world_configs"]["world_ranges"], output.effective_config["settings"]["range_name"],
        stream[stream_name], load_labels_config(),
    )
    _CACHED_COMPLAINTS = (output, labels, drawn)
    return _CACHED_COMPLAINTS


def get_bulk_run() -> tuple[InteractiveWorldOutput, BulkConfirmationResult, tuple[int, int]]:
    """
    WHAT: Runs the 12c scam fixture world, copies the training values of the bulk_short_lag fixture layer into a copy
    of its effective configuration, and builds its bulk confirmations. Also returns the fixture's lag bounds in whole
    minutes (the delay range in days times minutes per day, rounded as the planner rounds it).
    WHY: A 61-day world with the real ranges rarely has a crackdown, so the check uses the labelled fixture. The
    layer is read with read_layer but not applied with apply_layers, which rejects its values on purpose.
    Args:
        None.
    Returns:
        The output carrying the copied configuration, the BulkConfirmationResult and (low, high) lag minutes.
    Example:
        output, result, (low, high) = get_bulk_run(); assert low <= high
    """
    global _CACHED_BULK
    if _CACHED_BULK is not None:
        return _CACHED_BULK
    settings, code_version = get_showcase_b_settings()
    output = run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True), gang_configs=(load_showcase_c_fixture(),),
        code_version=code_version,
    )
    layer = read_layer(BULK_LAYER_PATH)
    new_config = copy.deepcopy(output.effective_config)
    split = new_config["settings"]["range_name"]
    parameters = new_config["world_configs"]["world_ranges"]["parameters"]
    for name, ranges in layer.changes["world_ranges"].items():
        parameters[name][split] = list(ranges[split])
    layered = dataclasses.replace(output, effective_config=new_config)
    low_days, high_days = parameters[DELAY_DAYS_NAME][split]
    lag_bounds = (int(round(float(low_days) * MINUTES_PER_DAY)), int(round(float(high_days) * MINUTES_PER_DAY)))
    _CACHED_BULK = (layered, build_bulk_confirmations(layered), lag_bounds)
    return _CACHED_BULK
