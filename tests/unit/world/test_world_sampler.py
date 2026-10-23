"""
WHAT: Unit tests for the many-worlds sampler: Latin hypercube strata, value ranges per split, disjoint seeds,
refused configs and byte-identical manifests.
Every expected value comes from config/world_sampler.yaml, config/world_ranges.yaml or from construction.

WHY: The sampler decides which worlds a detector learns from and which it is scored on, so a slip in seeds or ranges
would leak evaluation worlds into training without any error.

TERM: stratum: one of n equal slices of [0, 1]; an n-point Latin hypercube puts exactly one point in each.
"""

import copy
import functools
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from world.world_sampler import (
    check_seeds_disjoint,
    latin_hypercube,
    load_sampler_config,
    read_manifest_provenance,
    read_world_manifest,
    sample_block,
    sample_worlds,
    validate_sampler_config,
    write_world_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("training", "validation", "test", "stress")


SAMPLER_PATH = REPO_ROOT / "config" / "world_sampler.yaml"
RANGES_PATH = REPO_ROOT / "config" / "world_ranges.yaml"


@functools.lru_cache(maxsize=None)
def _yaml_file(name: str) -> dict[str, Any]:
    """Reads a config file once per module with yaml, so expected answers do not pass through the code under test."""
    with open(REPO_ROOT / "config" / name, encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return data


def _cfg() -> dict[str, Any]:
    """Returns a fresh deep copy of the cached world_sampler block, so a test may change it freely."""
    block: dict[str, Any] = copy.deepcopy(_yaml_file("world_sampler.yaml")["world_sampler"])
    return block


def _ranges() -> dict[str, Any]:
    """Returns the cached world_ranges.yaml contents; tests only read it and never change it."""
    return _yaml_file("world_ranges.yaml")


def _range_of(parameter: str, range_name: str) -> list[float]:
    """Looks up the [low, high] pair of one parameter in one range column of world_ranges.yaml."""
    bounds: list[float] = _ranges()["parameters"][parameter][range_name]
    return bounds


def test_each_dimension_has_one_point_per_stratum() -> None:
    """With n=10 and 4 dimensions, every dimension must hit each tenth of [0, 1) exactly once."""
    n = 10
    points = latin_hypercube(n, 4, np.random.default_rng(5))
    assert len(points) == n
    for dim in range(4):
        strata_hit = sorted(int(point[dim] * n) for point in points)
        assert strata_hit == list(range(n))


def test_values_sit_inside_the_splits_own_range() -> None:
    """Sampled key values stay inside the range column the split reads; validation reads training."""
    cfg = _cfg()
    ranges = _ranges()
    mapping = {"training": "training", "validation": "training", "test": "training", "stress": "stress"}
    for split in SPLITS:
        for spec in sample_worlds(split, 30, 11, cfg, ranges):
            assert spec.range_name == mapping[split]
            for name, value in spec.key_values:
                low, high = _range_of(cfg["key_parameters"][name]["world_ranges_reference"], mapping[split])
                assert low <= value <= high, f"{spec.world_id} {name}={value} outside [{low}, {high}]"


def test_integer_parameter_is_a_whole_number() -> None:
    """number_of_regions is marked integer in the YAML, so every sampled value must be an int."""
    for spec in sample_worlds("test", 20, 3, _cfg(), _ranges()):
        values = dict(spec.key_values)
        assert isinstance(values["number_of_regions"], int)


def test_test_split_stays_inside_the_training_range() -> None:
    """Report Section 6: test worlds read the training ranges, so no test key value may leave the training range."""
    cfg = _cfg()
    for spec in sample_worlds("test", 50, 13, cfg, _ranges()):
        assert spec.range_name == "training"
        for name, value in spec.key_values:
            low, high = _range_of(cfg["key_parameters"][name]["world_ranges_reference"], "training")
            assert low <= value <= high


def test_any_split_mapped_to_the_test_column_is_refused() -> None:
    """Mapping a split to the world_ranges test column must be refused by name, whichever split it is."""
    cfg = _cfg()
    cfg["range_name_for_split"]["mapping"]["stress"] = "test"
    with pytest.raises(ValueError, match="test range column"):
        validate_sampler_config(cfg, _ranges())


def test_block_size_must_fit_in_every_seed_block() -> None:
    """A block_size below 1 or above a seed block's size must be refused."""
    cfg = _cfg()
    cfg["block_size"]["value"] = 0
    with pytest.raises(ValueError, match="block_size must be at least 1"):
        validate_sampler_config(cfg, _ranges())
    cfg["block_size"]["value"] = 10_000_000
    with pytest.raises(ValueError, match="larger than"):
        validate_sampler_config(cfg, _ranges())


def test_blocks_have_contiguous_disjoint_seeds_and_different_values() -> None:
    """Blocks 0 and 1 use back-to-back seed runs; their key values differ for one master seed."""
    cfg = _cfg()
    ranges = _ranges()
    size = cfg["block_size"]["value"]
    first = cfg["seed_blocks"]["validation"]["first_seed"]
    block0 = sample_block("validation", 0, 5, cfg, ranges)
    block1 = sample_block("validation", 1, 5, cfg, ranges)
    assert [spec.seed for spec in block0] == list(range(first, first + size))
    assert [spec.seed for spec in block1] == list(range(first + size, first + 2 * size))
    assert block0[3].world_id == "validation-b00-0003"
    assert {spec.key_values for spec in block0}.isdisjoint({spec.key_values for spec in block1})


def test_one_block_is_a_full_latin_hypercube() -> None:
    """In a block of 40 worlds, each continuous key parameter has exactly one value in each fortieth of its range."""
    cfg = _cfg()
    size = cfg["block_size"]["value"]
    block = sample_block("training", 0, 9, cfg, _ranges())
    for name in cfg["key_parameters"]:
        if cfg["key_parameters"][name]["integer"]:
            continue
        low, high = _range_of(cfg["key_parameters"][name]["world_ranges_reference"], "training")
        strata = sorted(int((dict(spec.key_values)[name] - low) / (high - low) * size) for spec in block)
        assert strata == list(range(size))


def test_block_past_the_seed_block_is_refused() -> None:
    """A block number whose last seed passes the split's last seed must be refused."""
    cfg = _cfg()
    block = cfg["seed_blocks"]["stress"]
    last_block = (block["last_seed"] - block["first_seed"] + 1) // cfg["block_size"]["value"]
    with pytest.raises(ValueError, match="past the split's last seed"):
        sample_block("stress", last_block, 1, cfg, _ranges())


def test_seeds_are_disjoint_and_never_reserved() -> None:
    """Seeds come from each split's block in the YAML, never repeat across splits, and never equal 101."""
    cfg = _cfg()
    ranges = _ranges()
    by_split = {split: sample_worlds(split, 40, 21, cfg, ranges) for split in SPLITS}
    reserved = cfg["reserved_seeds"]["seeds"]
    check_seeds_disjoint(by_split, reserved)
    seen: set[int] = set()
    for split, specs in by_split.items():
        first = cfg["seed_blocks"][split]["first_seed"]
        assert [spec.seed for spec in specs] == list(range(first, first + 40))
        for spec in specs:
            assert spec.seed not in seen
            assert spec.seed not in reserved
            seen.add(spec.seed)
    assert 101 not in seen


def test_check_seeds_disjoint_refuses_a_shared_seed() -> None:
    """A validation world built with a training seed must be refused, naming the seed."""
    cfg = _cfg()
    ranges = _ranges()
    training = sample_worlds("training", 3, 1, cfg, ranges)
    validation = sample_worlds("validation", 3, 1, cfg, ranges)
    leaked = type(validation[0])(
        validation[0].world_id, "validation", training[1].seed, validation[0].range_name, validation[0].key_values
    )
    with pytest.raises(ValueError, match=str(training[1].seed)):
        check_seeds_disjoint({"training": training, "validation": [leaked]}, cfg["reserved_seeds"]["seeds"])


def test_count_above_block_size_is_refused() -> None:
    """Asking for more worlds than the split's block holds must raise and name the block size."""
    cfg = _cfg()
    block = cfg["seed_blocks"]["stress"]
    size = block["last_seed"] - block["first_seed"] + 1
    with pytest.raises(ValueError, match="larger than"):
        sample_worlds("stress", size + 1, 1, cfg, _ranges())


def test_overlapping_blocks_are_refused() -> None:
    """Moving the test block's start into the validation block must fail with an overlap message."""
    cfg = _cfg()
    cfg["seed_blocks"]["test"]["first_seed"] = cfg["seed_blocks"]["validation"]["last_seed"]
    with pytest.raises(ValueError, match="overlap"):
        validate_sampler_config(cfg, _ranges())


def test_unknown_parameter_is_refused() -> None:
    """A key parameter that world_ranges.yaml does not know must fail with an unknown-parameter message."""
    cfg = _cfg()
    cfg["key_parameters"]["made_up"] = {"world_ranges_reference": "no_such_parameter", "integer": False}
    with pytest.raises(ValueError, match="unknown parameter"):
        validate_sampler_config(cfg, _ranges())


def test_validation_mapped_to_test_is_refused() -> None:
    """Mapping validation to the test range breaks amendment A-015 and must be refused by name."""
    cfg = _cfg()
    cfg["range_name_for_split"]["mapping"]["validation"] = "test"
    with pytest.raises(ValueError, match="validation must map to the training range"):
        validate_sampler_config(cfg, _ranges())


def test_reserved_seed_inside_a_block_is_refused() -> None:
    """Putting reserved seed 101 inside the training block must fail with a reserved-seed message."""
    cfg = _cfg()
    cfg["seed_blocks"]["training"]["first_seed"] = 100
    with pytest.raises(ValueError, match="reserved seed 101"):
        validate_sampler_config(cfg, _ranges())


def test_real_config_passes_validation() -> None:
    """The shipped YAML must satisfy every rule the validator enforces."""
    validate_sampler_config(load_sampler_config(), _ranges())


def test_same_sampler_seed_gives_identical_manifest_bytes(tmp_path: Path) -> None:
    """Two runs with one sampler seed must write byte-identical files; a different seed must differ."""
    cfg = _cfg()
    ranges = _ranges()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    other = tmp_path / "other.json"
    write_world_manifest(sample_worlds("validation", 12, 99, cfg, ranges), first, 99, SAMPLER_PATH, RANGES_PATH)
    write_world_manifest(sample_worlds("validation", 12, 99, cfg, ranges), second, 99, SAMPLER_PATH, RANGES_PATH)
    write_world_manifest(sample_worlds("validation", 12, 100, cfg, ranges), other, 100, SAMPLER_PATH, RANGES_PATH)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != other.read_bytes()


def test_manifest_round_trips_exactly(tmp_path: Path) -> None:
    """Reading a written manifest must give back equal WorldSpec records, int and float types included."""
    specs = sample_worlds("test", 15, 8, _cfg(), _ranges())
    path = tmp_path / "test_manifest.json"
    write_world_manifest(specs, path, 8, SAMPLER_PATH, RANGES_PATH)
    assert read_world_manifest(path) == specs


def test_manifest_provenance_matches_file_hashes(tmp_path: Path) -> None:
    """The manifest must carry the sampler seed and the SHA-256 of both config files, computed here with hashlib."""
    specs = sample_worlds("training", 5, 77, _cfg(), _ranges())
    path = tmp_path / "provenance.json"
    write_world_manifest(specs, path, 77, SAMPLER_PATH, RANGES_PATH)
    provenance = read_manifest_provenance(path)
    assert provenance["sampler_seed"] == 77
    assert provenance["sampler_config_sha256"] == hashlib.sha256(SAMPLER_PATH.read_bytes()).hexdigest()
    assert provenance["world_ranges_sha256"] == hashlib.sha256(RANGES_PATH.read_bytes()).hexdigest()


def test_validation_key_values_differ_from_training_for_one_master_seed() -> None:
    """With one master seed, training and validation must share no key-value tuple (shared-stream regression)."""
    cfg = _cfg()
    ranges = _ranges()
    training = {spec.key_values for spec in sample_worlds("training", 20, 42, cfg, ranges)}
    validation = {spec.key_values for spec in sample_worlds("validation", 20, 42, cfg, ranges)}
    assert training.isdisjoint(validation)


def test_integer_values_are_uniform() -> None:
    """With n a multiple of the integer count, every whole number in the training range appears exactly 10 times."""
    low, high = _range_of("number_of_regions", "training")
    how_many = int(high) - int(low) + 1
    specs = sample_worlds("training", 10 * how_many, 5, _cfg(), _ranges())
    counts: dict[int, int] = {}
    for spec in specs:
        value = int(dict(spec.key_values)["number_of_regions"])
        counts[value] = counts.get(value, 0) + 1
    expected = {number: 10 for number in range(int(low), int(high) + 1)}
    assert counts == expected
