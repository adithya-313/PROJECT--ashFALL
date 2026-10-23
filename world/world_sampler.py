"""
WHAT: Plans many worlds before any is run. Each planned world gets a seed from its split's seed block and values
for a few key parameters, spread evenly over the parameter's range with a Latin hypercube. The plan is written
as one JSON manifest per split.

WHY: A detector trained on one world only learns that world. Spreading key values and keeping the seed blocks of
training, validation, test and stress apart lets a later level measure how well it carries to unseen worlds.
This module only plans; running a world stays with the world builder, so the two can change independently.

ISSUE IDs: 12-14, 12-18.
TERM: Latin hypercube: a way to pick n points so that each parameter's range, cut into n equal strata, gets exactly
      one point per stratum. For example, 10 points over [0, 1] land one in each tenth.
TERM: seed block: a fixed run of integer seeds owned by one split, so no two splits can share a world.
TERM: manifest: the JSON file listing every planned world of one split.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLER_PATH = REPO_ROOT / "config" / "world_sampler.yaml"
EXPECTED_SPLITS = ("training", "validation", "test", "stress")


@dataclass(frozen=True)
class WorldSpec:
    """
    WHAT: One planned world: its id, split, seed, the range column it reads and its key parameter values.
    The key values are stored as a tuple of (name, value) pairs so the whole record can be frozen and compared.

    WHY: A frozen record cannot be changed after planning, so the manifest on disk and the worlds that
    are later run from it cannot drift apart.

    Args:
        world_id: Text such as "validation-0003".
        split: One of training, validation, test, stress.
        seed: Seed for the world builder.
        range_name: Column of world_ranges.yaml the values were mapped onto.
        key_values: (parameter name, value) pairs, whole numbers for integer parameters.
    """

    world_id: str
    split: str
    seed: int
    range_name: str
    key_values: tuple[tuple[str, float | int], ...]


def load_sampler_config(path: Path | str | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/world_sampler.yaml (or the given path) and returns the contents of its world_sampler block.
    The block holds the key parameters, seed blocks, reserved seeds and the split-to-range mapping.

    WHY: Keeping every number in YAML means a later amendment changes a file, not code.

    Args:
        path: Optional YAML path; defaults to config/world_sampler.yaml in the repository.

    Returns:
        The world_sampler dictionary.

    Raises:
        ValueError: If the file has no world_sampler block.
    """
    chosen = DEFAULT_SAMPLER_PATH if path is None else Path(path)
    with open(chosen, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "world_sampler" not in data:
        raise ValueError(f"{chosen} has no world_sampler block")
    block: dict[str, Any] = data["world_sampler"]
    return block


def _check_splits_and_mapping(cfg: dict[str, Any]) -> None:
    """Refuses unknown splits and any mapping that sends validation anywhere but training."""
    mapping = cfg["range_name_for_split"]["mapping"]
    for split in list(cfg["seed_blocks"]) + list(mapping):
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"unknown split '{split}' in the sampler config")
    for split in EXPECTED_SPLITS:
        if split not in cfg["seed_blocks"] or split not in mapping:
            raise ValueError(f"split '{split}' needs both a seed block and a range mapping")
    if mapping["validation"] != "training":
        raise ValueError("validation must map to the training range (amendment A-015)")


def _check_parameters(cfg: dict[str, Any], world_ranges: dict[str, Any]) -> None:
    """Refuses key parameters that world_ranges.yaml does not know or that lack a range for a mapped split."""
    parameters = world_ranges.get("parameters", {})
    mapping = cfg["range_name_for_split"]["mapping"]
    for name, setting in cfg["key_parameters"].items():
        reference = setting["world_ranges_reference"]
        if reference not in parameters:
            raise ValueError(f"unknown parameter '{reference}' for key parameter '{name}'")
        for range_name in set(mapping.values()):
            if range_name not in parameters[reference]:
                raise ValueError(f"parameter '{reference}' has no '{range_name}' range")


def _check_seed_blocks(cfg: dict[str, Any]) -> None:
    """Refuses blocks that run backwards, overlap each other, or contain a reserved seed."""
    spans: list[tuple[int, int, str]] = []
    for split, block in cfg["seed_blocks"].items():
        first = int(block["first_seed"])
        last = int(block["last_seed"])
        if first > last:
            raise ValueError(f"seed block '{split}' starts at {first}, after its end {last}")
        spans.append((first, last, split))
    spans.sort()
    for index in range(len(spans) - 1):
        if spans[index][1] >= spans[index + 1][0]:
            raise ValueError(f"seed blocks '{spans[index][2]}' and '{spans[index + 1][2]}' overlap")
    for reserved in cfg["reserved_seeds"]["seeds"]:
        for first, last, split in spans:
            if first <= reserved <= last:
                raise ValueError(f"reserved seed {reserved} sits inside the '{split}' seed block")


def validate_sampler_config(cfg: dict[str, Any], world_ranges: dict[str, Any]) -> None:
    """
    WHAT: Checks the sampler config against world_ranges.yaml and refuses it with a plain reason: unknown
    parameter, unknown split, validation not mapped to training, overlapping seed blocks, or a reserved seed
    inside a block.

    WHY: A bad block would silently let a validation world share a seed with a training world, and nothing
    downstream would notice; refusing early is cheaper than a leaked evaluation.

    Args:
        cfg: The world_sampler dictionary from load_sampler_config.
        world_ranges: The dictionary read from config/world_ranges.yaml.

    Raises:
        ValueError: With a message naming the problem.
    """
    _check_splits_and_mapping(cfg)
    _check_parameters(cfg, world_ranges)
    _check_seed_blocks(cfg)


def latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> list[list[float]]:
    """
    WHAT: Returns n points in [0, 1) with dims coordinates each. For every dimension the n equal strata are
    shuffled into a random order and each point gets one stratum plus a random offset inside it.

    WHY: Plain random draws can clump; this gives exactly one point per stratum in every dimension, so a small
    number of worlds still covers each parameter's whole range.

    Args:
        n: Number of points; at least 1.
        dims: Number of coordinates per point; at least 1.
        rng: Generator that supplies every random number.

    Returns:
        n lists of dims floats.

    Raises:
        ValueError: If n or dims is below 1.
    """
    if n < 1 or dims < 1:
        raise ValueError(f"latin_hypercube needs n and dims of at least 1, got n={n}, dims={dims}")
    points: list[list[float]] = []
    for _ in range(n):
        points.append([0.0] * dims)
    for dim in range(dims):
        strata = rng.permutation(n)
        jitter = rng.random(n)
        for index in range(n):
            points[index][dim] = (float(strata[index]) + float(jitter[index])) / n
    return points


def _map_to_range(unit: float, low: float, high: float, is_integer: bool) -> float | int:
    """Maps a value in [0, 1) onto [low, high]; integers are rounded and kept inside the range."""
    value = low + unit * (high - low)
    if not is_integer:
        return float(value)
    rounded = int(round(value))
    return max(int(low), min(int(high), rounded))


def _make_spec(
    split: str, index: int, seed: int, range_name: str, cfg: dict[str, Any],
    world_ranges: dict[str, Any], point: list[float],
) -> WorldSpec:
    """Builds one WorldSpec from an LHS point by mapping each coordinate onto its parameter's range."""
    values: list[tuple[str, float | int]] = []
    names = list(cfg["key_parameters"])
    for dim in range(len(names)):
        name = names[dim]
        setting = cfg["key_parameters"][name]
        entry = world_ranges["parameters"][setting["world_ranges_reference"]]
        low, high = entry[range_name]
        values.append((name, _map_to_range(point[dim], low, high, bool(setting["integer"]))))
    return WorldSpec(f"{split}-{index:04d}", split, seed, range_name, tuple(values))


def sample_worlds(
    split: str, count: int, sampler_seed: int, cfg: dict[str, Any], world_ranges: dict[str, Any]
) -> list[WorldSpec]:
    """
    WHAT: Plans count worlds for one split. Key values come from a Latin hypercube mapped onto the range
    column the split reads (validation reads training); world k gets seed block_start + k.

    WHY: One call per split with its own seed block keeps splits apart, and the same sampler_seed always gives the
    same plan, so a manifest can be rebuilt instead of trusted.

    Args:
        split: training, validation, test or stress.
        count: Number of worlds; must fit inside the split's seed block.
        sampler_seed: Seed for the hypercube's generator (not a world seed).
        cfg: The world_sampler dictionary.
        world_ranges: The dictionary read from config/world_ranges.yaml.

    Returns:
        count WorldSpec records in seed order.

    Raises:
        ValueError: If the split is unknown, count is below 1, or count exceeds the block.
    """
    if split not in cfg["seed_blocks"]:
        raise ValueError(f"unknown split '{split}'")
    block = cfg["seed_blocks"][split]
    first = int(block["first_seed"])
    block_size = int(block["last_seed"]) - first + 1
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if count > block_size:
        raise ValueError(f"count {count} is larger than the '{split}' seed block of {block_size} seeds")
    range_name = cfg["range_name_for_split"]["mapping"][split]
    rng = np.random.default_rng(sampler_seed)
    points = latin_hypercube(count, len(cfg["key_parameters"]), rng)
    specs: list[WorldSpec] = []
    for index in range(count):
        specs.append(_make_spec(split, index, first + index, range_name, cfg, world_ranges, points[index]))
    return specs


def check_seeds_disjoint(
    specs_by_split: dict[str, Sequence[WorldSpec]], reserved_seeds: Sequence[int]
) -> None:
    """
    WHAT: Raises if any seed appears in two splits, or if any planned seed is a reserved one.
    Every seed is recorded with the split that first used it.

    WHY: Seed blocks should make overlap impossible; this checks the finished plan anyway, because a leak between
    training and evaluation worlds is the failure that would stay hidden longest.

    Args:
        specs_by_split: Planned worlds grouped by split name.
        reserved_seeds: Seeds no sampled world may use, such as the Level 11 golden world.

    Raises:
        ValueError: Naming the seed and the splits involved.
    """
    owner: dict[int, str] = {}
    for split, specs in specs_by_split.items():
        for spec in specs:
            if spec.seed in reserved_seeds:
                raise ValueError(f"seed {spec.seed} in split '{split}' is a reserved seed")
            if spec.seed in owner and owner[spec.seed] != split:
                raise ValueError(f"seed {spec.seed} is used by both '{owner[spec.seed]}' and '{split}'")
            owner[spec.seed] = split


def write_world_manifest(specs: Sequence[WorldSpec], path: Path | str) -> None:
    """
    WHAT: Writes the planned worlds of one split to a JSON file with sorted keys, indented, with a final newline.
    Line endings are always "\\n".

    WHY: Sorted keys and fixed line endings make the same plan produce the same bytes on any machine, so two
    manifests can be compared as files.

    Args:
        specs: Planned worlds; all must belong to the same split.
        path: Destination file.

    Raises:
        ValueError: If specs is empty or mixes splits.
    """
    if not specs:
        raise ValueError("cannot write a manifest with no worlds")
    split = specs[0].split
    worlds: list[dict[str, Any]] = []
    for spec in specs:
        if spec.split != split:
            raise ValueError(f"manifest mixes splits '{split}' and '{spec.split}'")
        pairs = [[name, value] for name, value in spec.key_values]
        worlds.append({
            "world_id": spec.world_id, "split": spec.split, "seed": spec.seed,
            "range_name": spec.range_name, "key_values": pairs,
        })
    text = json.dumps({"split": split, "worlds": worlds}, sort_keys=True, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def read_world_manifest(path: Path | str) -> list[WorldSpec]:
    """
    WHAT: Reads a manifest written by write_world_manifest back into WorldSpec records.
    Integer and float values keep the type they were written with.

    WHY: Later levels run worlds from the file, not from the sampler, so the file must give back exactly what was
    planned.

    Args:
        path: Manifest file.

    Returns:
        WorldSpec records in file order.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    specs: list[WorldSpec] = []
    for world in data["worlds"]:
        pairs = tuple((str(name), value) for name, value in world["key_values"])
        specs.append(WorldSpec(world["world_id"], world["split"], int(world["seed"]), world["range_name"], pairs))
    return specs
