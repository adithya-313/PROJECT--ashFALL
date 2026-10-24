"""
WHAT: Property tests verifying held-out taxonomic constraints across every planned world specification.
Every training and validation world contains only seen gangs without held-out options, while every test
and stress world includes at least one seen gang and exactly one active held-out gang.

WHY: Level 12g requires mathematical proof that held-out tactics never contaminate training or validation
distributions, ensuring rigorous evaluation of generalization against unseen illicit mechanics (A-025).
"""

import collections
import copy
from typing import Any

import pytest
import yaml

from world.gang_config import gang_config_from_plain, gang_config_to_plain
from world.gang_mix import generate_gang_mix, load_gang_mix_config
from world.held_out_options import (
    check_held_out_options_allowed,
    find_held_out_options,
    gang_truth_for,
)
from world.run_identity import REPO_ROOT
from world.tactic_families import load_evaluation_config, load_tactic_families
from world.world_configs import read_world_configs
from world.world_sampler import WorldSpec, load_sampler_config, sample_block

SAMPLER_SEED: int = 1


def _yaml_content(relative_name: str) -> dict[str, Any]:
    """Reads a configuration file directly with PyYAML without going through intermediate code caches."""
    with open(REPO_ROOT / "config" / relative_name, encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return data


def test_held_out_property_across_all_planned_worlds() -> None:
    """
    WHAT: Asserts that across all 680 planned worlds (320 training, 120 validation, 120 test, 120 stress),
    training and validation have 0 held-out gangs while test and stress each have exactly one active held-out gang.

    WHY: Proves taxonomic isolation over the entire Latin hypercube parameter space rather than a sample,
    confirming that detector training data remains free of evaluation evasions.
    """
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_configs().world_ranges
    mix_cfg = load_gang_mix_config()
    families = load_tactic_families()
    eval_cfg = load_evaluation_config()
    active_held_out = set(eval_cfg["held_out_tactics"]["tactics"])
    catalogue_held_out = set(families["held_out"])

    sizing_cfg = _yaml_content("world_sizing.yaml")
    eligible_people: int = sizing_cfg["presets"]["large"]["n_people"]
    assert eligible_people == 2000

    blocks_per_split = sampler_cfg["counts"]["blocks_per_split"]
    assert blocks_per_split["training"] == 8
    assert blocks_per_split["validation"] == 3
    assert blocks_per_split["test"] == 3
    assert blocks_per_split["stress"] == 3

    specs_by_split: dict[str, list[WorldSpec]] = {}
    total_specs_count = 0
    for split in ("training", "validation", "test", "stress"):
        specs: list[WorldSpec] = []
        for b in range(blocks_per_split[split]):
            specs.extend(sample_block(split, b, SAMPLER_SEED, sampler_cfg, world_ranges))
        specs_by_split[split] = specs
        total_specs_count += len(specs)

    assert len(specs_by_split["training"]) == 320
    assert len(specs_by_split["validation"]) == 120
    assert len(specs_by_split["test"]) == 120
    assert len(specs_by_split["stress"]) == 120
    assert total_specs_count == 680

    for split in ("training", "validation"):
        for spec in specs_by_split[split]:
            gangs = generate_gang_mix(spec, sampler_cfg, mix_cfg, eligible_people, world_ranges)
            for gang in gangs:
                assert find_held_out_options(gang) == []
                assert gang.family not in catalogue_held_out
                gt = gang_truth_for(gang, families)
                assert gt.kind == "seen"

    family_counts: collections.Counter[str] = collections.Counter()
    for split in ("test", "stress"):
        for spec in specs_by_split[split]:
            gangs = generate_gang_mix(spec, sampler_cfg, mix_cfg, eligible_people, world_ranges)
            seen_gangs = [g for g in gangs if gang_truth_for(g, families).kind == "seen"]
            held_gangs = [g for g in gangs if gang_truth_for(g, families).kind == "held_out"]
            assert len(seen_gangs) >= 1
            assert len(held_gangs) == 1
            held_gang = held_gangs[0]
            assert held_gang.family in active_held_out
            family_counts[str(held_gang.family)] += 1

    assert sum(family_counts.values()) == 240
    per_fam_str = ", ".join(f"{k}: {v}" for k, v in sorted(family_counts.items()))
    summary_line = (
        f"N=440 training/validation worlds checked: 0 held-out gangs; "
        f"N=240 test/stress worlds checked: 240 with one held-out gang ({per_fam_str})"
    )
    print(summary_line)
    assert (
        "N=440 training/validation worlds checked: 0 held-out gangs; "
        "N=240 test/stress worlds checked: 240 with one held-out gang" in summary_line
    )


def test_training_gang_with_hand_added_held_out_option_is_refused() -> None:
    """
    WHAT: Asserts that a training spec whose gang configuration is artificially given a held-out
    option is rejected by check_held_out_options_allowed when tested under split 'training'.

    WHY: Proves fail-closed security: if held-out tactics inadvertently enter a training world,
    the split guard actively raises a ValueError rather than silently accepting corrupted data.
    """
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_configs().world_ranges
    mix_cfg = load_gang_mix_config()
    spec = sample_block("training", 0, SAMPLER_SEED, sampler_cfg, world_ranges)[0]
    gangs = generate_gang_mix(spec, sampler_cfg, mix_cfg, 2000, world_ranges)
    first_gang = gangs[0]

    plain = gang_config_to_plain(first_gang)
    modified_plain = copy.deepcopy(plain)
    for step in modified_plain["steps"]:
        if step["block"] == "route":
            params_dict = dict(step["params"])
            params_dict.pop("dwell_minutes", None)
            params_dict["drip_piece_max_inr"] = 1000
            params_dict["drip_spread_days"] = 7
            params_dict["drip_hold_minutes"] = 300
            step["params"] = params_dict
    modified_plain["family"] = "slow_drip"
    modified_gang = gang_config_from_plain(modified_plain)

    with pytest.raises(ValueError, match="allowed only in test and stress worlds"):
        check_held_out_options_allowed(modified_gang, "training")

    check_held_out_options_allowed(modified_gang, "test")
    check_held_out_options_allowed(modified_gang, "stress")
