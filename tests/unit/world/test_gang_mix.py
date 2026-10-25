"""
WHAT: Unit tests for the per-world gang mix: repeatable and seed-dependent draws, every drawn value inside its
centre-plus-spread bounds, the right number of gangs, training-only family tags that follow the committed rules, a loud
refusal when the world has too few people, the private stream (child 22), and one small world run with a generated mix.
Expected values are computed in the tests from the YAML files, never from the code under test.

WHY: The mix decides what fraud each sampled world holds. A silent shrink, a held-out tag on training behaviour or a
shared random stream would all make many-worlds results look sound while being wrong.

TERM: centre: the world-level value of a gang setting that each of its gangs draws around.
TERM: spread: the share of a parameter's range width a gang may stray from the centre.
"""

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from world.gang_config import GangConfig, gang_config_to_plain
from world.gang_mix import gang_mix_rng, generate_gang_mix, load_gang_mix_config
from world.sampled_world import run_sampled_world
from world.world_sampler import WorldSpec, load_sampler_config, sample_worlds

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_FOR_SAMPLER = 12


def _yaml(name: str) -> dict[str, Any]:
    """Reads a config file straight with yaml, so expected answers do not pass through the code under test."""
    with open(REPO_ROOT / "config" / name, encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return data


RANGES = _yaml("world_ranges.yaml")
SAMPLER = load_sampler_config()
MIX = load_gang_mix_config()
SMALL_PEOPLE = _yaml("world_sizing.yaml")["presets"]["small"]["n_people"]


def _mix_cfg() -> dict[str, Any]:
    """Returns a fresh deep copy of the gang mix config a test may change."""
    return copy.deepcopy(MIX)


def _spec(split: str = "training", index: int = 0, count: int = 1) -> WorldSpec:
    """Plans count worlds of a split and returns the one at index."""
    return sample_worlds(split, count, SEED_FOR_SAMPLER, SAMPLER, RANGES)[index]


def _range_for(name: str) -> tuple[float, float]:
    """Returns the training-column or own range of a gang-scope key parameter, read from the YAML files."""
    setting = SAMPLER["key_parameters"][name]
    if "range" in setting:
        low, high = setting["range"]
    else:
        low, high = RANGES["parameters"][setting["world_ranges_reference"]]["training"]
    return low, high


def _params(gang: GangConfig, block: str) -> dict[str, Any]:
    """Returns the parameters of the first step of a block in a gang."""
    for step in gang.steps:
        if step.block == block:
            return dict(step.params)
    raise AssertionError(f"gang {gang.gang_id} has no {block} step")


@pytest.fixture(scope="module")
def timed_world() -> tuple[Any, float, WorldSpec]:
    """Runs one small training world with a generated mix; returns the output, the seconds and the spec."""
    spec = _spec()
    started = time.perf_counter()
    output = run_sampled_world(spec, "small", None, SAMPLER, "test-gang-mix")
    return output, time.perf_counter() - started, spec


def test_same_spec_gives_the_same_mix_and_other_seeds_differ() -> None:
    """The same spec twice gives equal gangs; two specs with different seeds give different gangs."""
    first = _spec(count=2, index=0)
    second = _spec(count=2, index=1)
    assert generate_gang_mix(first, SAMPLER, MIX, SMALL_PEOPLE, RANGES) == generate_gang_mix(
        first, SAMPLER, MIX, SMALL_PEOPLE, RANGES
    )
    assert generate_gang_mix(first, SAMPLER, MIX, SMALL_PEOPLE, RANGES) != generate_gang_mix(
        second, SAMPLER, MIX, SMALL_PEOPLE, RANGES
    )


def test_drawn_values_stay_within_centre_plus_or_minus_the_spread() -> None:
    """Each gang's layers, dwell, devices and victims lie within centre +/- spread x width, clipped to the range."""
    spread = MIX["spread_share_of_range"]["value"]
    for index in range(6):
        spec = _spec(count=6, index=index)
        centres = dict(spec.key_values)
        for gang in generate_gang_mix(spec, SAMPLER, MIX, SMALL_PEOPLE, RANGES):
            acquire_count = _params(gang, "acquire")["count"]
            observed = {
                "route_layers": _params(gang, "route")["layers"],
                "dwell_minutes": _params(gang, "route")["dwell_minutes"],
                "devices_per_gang": _params(gang, "take_control")["devices"],
                "victims_per_receiving_mule": _params(gang, "scam")["victims_per_receiving_mule"],
            }
            for name, value in observed.items():
                low, high = _range_for(name)
                half = spread * (high - low)
                lowest = max(low, centres[name] - half)
                highest = min(high, centres[name] + half)
                floor_value = math.floor(lowest)
                if name == "devices_per_gang":
                    floor_value = min(floor_value, acquire_count)
                assert floor_value <= value <= math.ceil(highest), f"{gang.gang_id} {name}={value}"
            low_mules, high_mules = MIX["receiving_mules"]["range"]
            assert low_mules <= _params(gang, "scam")["receiving_mules"] <= high_mules


def test_gang_count_matches_the_spec_and_every_gang_is_typed() -> None:
    """The mix has exactly the spec's gang_count gangs, each a GangConfig with its own id."""
    for index in range(5):
        spec = _spec(count=5, index=index)
        gangs = generate_gang_mix(spec, SAMPLER, MIX, SMALL_PEOPLE, RANGES)
        assert len(gangs) == dict(spec.key_values)["gang_count"]
        assert all(isinstance(gang, GangConfig) for gang in gangs)
        assert [gang.gang_id for gang in gangs] == [f"gang_mix_{k}" for k in range(len(gangs))]


def test_families_follow_the_rules_training_never_held_out_test_and_stress_get_one_held_out() -> None:
    """Every family is training in training and validation specs; test and stress specs get one held-out gang."""
    families = _yaml("tactic_families.yaml")["tactic_families"]
    eval_cfg = _yaml("evaluation.yaml")
    active_held_out = set(eval_cfg["held_out_tactics"]["tactics"])
    training = set(families["training"])
    held_out = set(families["held_out"])

    for split in ("training", "validation"):
        for gang in generate_gang_mix(_spec(split), SAMPLER, MIX, SMALL_PEOPLE, RANGES):
            layers = _params(gang, "route")["layers"]
            expected = "layered_chain" if layers >= 2 else "fast_pass_through"
            assert gang.family == expected
            assert gang.family in training
            assert gang.family not in held_out

    for split in ("test", "stress"):
        gangs = generate_gang_mix(_spec(split), SAMPLER, MIX, SMALL_PEOPLE, RANGES)
        for gang in gangs[:-1]:
            layers = _params(gang, "route")["layers"]
            expected = "layered_chain" if layers >= 2 else "fast_pass_through"
            assert gang.family == expected
            assert gang.family in training
            assert gang.family not in held_out

        last_gang = gangs[-1]
        assert last_gang.family in active_held_out
        if last_gang.family == "slow_drip":
            route_p = _params(last_gang, "route")
            assert "drip_piece_max_inr" in route_p
            assert "drip_spread_days" in route_p
            assert "drip_hold_minutes" in route_p
            assert "dwell_minutes" not in route_p
        elif last_gang.family == "device_rotation":
            tc_p = _params(last_gang, "take_control")
            assert "rotate_every_days" in tc_p
            assert "rotations" in tc_p
            assert "devices" not in tc_p

    # A rule that names a held-out family must be refused.
    bad_rules = _mix_cfg()
    bad_rules["family_rules"]["rules"][0]["family"] = sorted(held_out)[0]
    with pytest.raises(ValueError, match="not a training family"):
        generate_gang_mix(_spec(), SAMPLER, bad_rules, SMALL_PEOPLE, RANGES)

    # held_out_gang.splits containing training is refused.
    training_in_splits = _mix_cfg()
    training_in_splits["held_out_gang"]["splits"] = ["training", "test", "stress"]
    with pytest.raises(ValueError, match="must be a subset of exempt splits"):
        generate_gang_mix(_spec(), SAMPLER, training_in_splits, SMALL_PEOPLE, RANGES)


def test_training_and_validation_mixes_are_byte_identical_to_commit_2d66bf3() -> None:
    """Training and validation gang mixes (plain form) match the SHA-256 computed at 2d66bf3."""
    training_spec = _spec("training")
    training_gangs = generate_gang_mix(training_spec, SAMPLER, MIX, SMALL_PEOPLE, RANGES)
    training_dumped = json.dumps([gang_config_to_plain(g) for g in training_gangs], sort_keys=True)
    assert hashlib.sha256(training_dumped.encode()).hexdigest() == (
        "7fd819d38b7113f1f3ef2338f0799c4a26ed3b23b3ebc61bc42015aa1c58691a"
    )

    val_spec = _spec("validation")
    val_gangs = generate_gang_mix(val_spec, SAMPLER, MIX, SMALL_PEOPLE, RANGES)
    val_dumped = json.dumps([gang_config_to_plain(g) for g in val_gangs], sort_keys=True)
    assert hashlib.sha256(val_dumped.encode()).hexdigest() == (
        "0f9ab9849af7aec85c47d2cc6e2907d2a2e376116e794c5ef7d64d6c4fdc0ed9"
    )


def test_test_and_stress_prefix_gangs_equal_commit_2d66bf3_and_last_is_held_out() -> None:
    """Test and stress specs keep gangs 0..n-2 identical to 2d66bf3 and place a held-out gang at index n-1."""
    eval_cfg = _yaml("evaluation.yaml")
    active_held_out = set(eval_cfg["held_out_tactics"]["tactics"])

    test_spec = _spec("test")
    test_gangs = generate_gang_mix(test_spec, SAMPLER, MIX, SMALL_PEOPLE, RANGES)
    test_prefix_dumped = json.dumps([gang_config_to_plain(g) for g in test_gangs[:-1]], sort_keys=True)
    assert hashlib.sha256(test_prefix_dumped.encode()).hexdigest() == (
        "8c4fb9ea9a69f59891b7432e7a41c9d55723e8391f3451152736498572cfe4fe"
    )
    assert test_gangs[-1].family in active_held_out

    stress_spec = _spec("stress")
    stress_gangs = generate_gang_mix(stress_spec, SAMPLER, MIX, SMALL_PEOPLE, RANGES)
    stress_prefix_dumped = json.dumps([gang_config_to_plain(g) for g in stress_gangs[:-1]], sort_keys=True)
    assert hashlib.sha256(stress_prefix_dumped.encode()).hexdigest() == (
        "1426a0110afe75395155258244a8c2e290aa75f2283e28f417053170881f4206"
    )
    assert stress_gangs[-1].family in active_held_out


def test_spec_with_gang_count_one_in_test_split_raises() -> None:
    """A test-split spec whose settings specify gang_count 1 raises ValueError because a seen gang must remain."""
    base_spec = _spec("test")
    new_key_values = tuple((k, 1 if k == "gang_count" else v) for k, v in base_spec.key_values)

    modified_spec = WorldSpec(
        world_id=base_spec.world_id,
        split="test",
        seed=base_spec.seed,
        range_name=base_spec.range_name,
        key_values=new_key_values,
    )
    with pytest.raises(ValueError, match="requires at least 2 gangs to include a held-out gang"):
        generate_gang_mix(modified_spec, SAMPLER, MIX, SMALL_PEOPLE, RANGES)


def test_too_few_people_is_refused_with_both_numbers() -> None:
    """With one eligible person the mix is refused, and the message names the people needed and the people there are."""
    with pytest.raises(ValueError, match=r"needs \d+ people .* only 1 eligible"):
        generate_gang_mix(_spec(), SAMPLER, MIX, 1, RANGES)


def test_the_mix_stream_is_child_22_of_the_seed() -> None:
    """gang_mix_rng equals a generator built here from child 22, the number read from the runner YAML."""
    child = _yaml("interactive_runner.yaml")["interactive_runner"]["random_stream_children"]["gang_mix"]
    assert child == 22
    seed = _spec().seed
    expected = np.random.default_rng(np.random.SeedSequence(seed).spawn(child + 1)[child])
    assert gang_mix_rng(seed).bit_generator.state == expected.bit_generator.state


def test_a_small_world_runs_with_a_generated_mix(timed_world: tuple[Any, float, WorldSpec]) -> None:
    """The world completes, and its truth has mule roles from at least gang_count different gangs."""
    output, seconds, spec = timed_world
    print(f"small world with generated mix: {seconds:.2f}s")
    mule_gangs = {role.gang_id for role in output.result.truth.roles if role.role == "mule"}
    assert len(mule_gangs) >= dict(spec.key_values)["gang_count"]
    assert output.effective_config["world_split"] == "training"
