"""
WHAT: Unit tests for separating a world's split from its range column: the effective configuration is unchanged when no
split is given, the 12d family guard follows the split, a wrong split and range pairing fails before any world is built,
and planned worlds only ever use the training or stress column. No world is run.
Expected values are written in the tests or read straight from the config files.

WHY: Report Section 6 gives test worlds the training ranges, so range_name alone can no longer say whether held-out
tactics are allowed. A mistake here would either leak held-out tactics into training or move every existing run id.

TERM: split: which group a world belongs to (training, validation, test, stress); it decides the tactics allowed.
TERM: range column: which column of world_ranges.yaml a world draws its numbers from.
"""

import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from world.fast_mode import FastRunSettings
from world.gang_config import gang_config_from_plain
from world.interactive_records import RunnerSettings
from world.interactive_runner import (
    effective_config_for,
    family_guard_split,
    run_interactive_world,
)
from world.tactic_families import load_tactic_families, validate_gang_families
from world.world_configs import read_world_configs
from world.world_sampler import load_sampler_config, sample_block

REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS = FastRunSettings(101, 60, 12, 61, "training")


def _gang_with_family(family: str) -> Any:
    """Builds a small scam-funded gang that carries the given tactic family."""
    plain = {
        "gang_id": "split_guard_gang",
        "strength": 1,
        "budget": {"money_paise": 50000000, "time_minutes": 500000, "max_risk": 0.99},
        "start_minute": 1440,
        "family": family,
        "steps": [
            {"block": "acquire", "params": {"count": 4}},
            {"block": "scam", "params": {"receiving_mules": 2, "victims_per_receiving_mule": 8}},
            {"block": "route", "params": {"layers": 2, "accounts_per_layer": 2}},
        ],
    }
    return gang_config_from_plain(plain)


def test_effective_config_is_unchanged_without_a_world_split() -> None:
    """Leaving world_split out or passing None gives the same dictionary and no world_split key."""
    configs = read_world_configs()
    omitted = effective_config_for(SETTINGS, configs)
    explicit_none = effective_config_for(SETTINGS, configs, world_split=None)
    assert omitted == explicit_none
    assert "world_split" not in omitted
    with_split = effective_config_for(SETTINGS, configs, world_split="test")
    assert with_split["world_split"] == "test"
    del with_split["world_split"]
    assert with_split == omitted


def test_held_out_family_follows_the_split_not_the_range_column() -> None:
    """A held-out family passes for split 'test' on the training column, and is refused for split 'training'."""
    families = load_tactic_families()
    held_out = sorted(families["held_out"])[0]
    gang = _gang_with_family(held_out)
    validate_gang_families([gang], family_guard_split("test", "training"), families=families)
    with pytest.raises(ValueError, match="held-out family"):
        validate_gang_families([gang], family_guard_split("training", "training"), families=families)
    assert family_guard_split(None, "stress") == "stress"


def test_wrong_split_and_range_pairing_fails_before_any_world_is_built() -> None:
    """Split 'stress' on the training column, an unknown split, and 'validation' on stress are refused within 2 s."""
    configs = read_world_configs()
    started = time.perf_counter()
    with pytest.raises(ValueError, match="stress.*training range column"):
        run_interactive_world(SETTINGS, RunnerSettings(), configs=configs, world_split="stress")
    stress_settings = FastRunSettings(101, 60, 12, 61, "stress")
    with pytest.raises(ValueError, match="cannot use the stress range column"):
        run_interactive_world(stress_settings, RunnerSettings(), configs=configs, world_split="validation")
    with pytest.raises(ValueError, match="must be training, validation, test or stress"):
        run_interactive_world(SETTINGS, RunnerSettings(), configs=configs, world_split="holdout")
    assert time.perf_counter() - started < 2.0


@pytest.mark.parametrize(
    "world_split, range_name",
    [("training", "test"), ("validation", "test"), ("test", "test"), ("test", "stress"), ("stress", "test")],
)
def test_pairings_outside_the_section_6_table_are_refused(world_split: str, range_name: str) -> None:
    """Only the Section 6 pairs are accepted (stress on stress, the rest on training); any other names the pair."""
    settings = FastRunSettings(101, 60, 12, 61, range_name)
    started = time.perf_counter()
    with pytest.raises(ValueError, match=f"world_split '{world_split}' cannot use the {range_name} range column"):
        run_interactive_world(settings, RunnerSettings(), world_split=world_split)
    assert time.perf_counter() - started < 2.0


def test_every_planned_split_uses_the_training_or_stress_column() -> None:
    """One block of each split: validation and test read the training column, stress its own, never the test column."""
    cfg = load_sampler_config()
    with open(REPO_ROOT / "config" / "world_sampler.yaml", encoding="utf-8") as handle:
        written = yaml.safe_load(handle)["world_sampler"]["range_name_for_split"]["mapping"]
    expected = {"training": "training", "validation": "training", "test": "training", "stress": "stress"}
    assert written == expected
    ranges = read_world_configs().world_ranges
    for split in ("training", "validation", "test", "stress"):
        block = sample_block(split, 0, 3, cfg, ranges)
        assert {spec.range_name for spec in block} == {expected[split]}
        assert {spec.range_name for spec in block} <= {"training", "stress"}
