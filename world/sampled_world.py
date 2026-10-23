"""
WHAT: Runs one world planned by the many-worlds sampler. It checks a manifest against the current config files,
pins the world's key parameters in a private copy of the configs, and calls the interactive runner with fraud on.

WHY: The sampler only plans. A plan made from older config files must never run silently, and a run must read only
the values its plan chose, so the pinning and the staleness check live in one place between the plan and the runner.
A pinned world is a different world from the unpinned one with the same seed, not a copy with one value changed:
number_of_regions is drawn with rng.integers, and a pinned [v, v] range leaves that stream in a different state than
a normal range (measured in commit 1, Step 0 d'), so every later draw shifts.

ISSUE IDs: 12-14, 12-18.
TERM: pinning: setting the training, test and stress ranges of a parameter to [value, value], so the world's own draw
      can only return that value.
TERM: stale manifest: a plan whose recorded config hashes differ from the config files now on disk.
"""

import copy
import dataclasses
from pathlib import Path
from typing import Any

from core.config_validator import contains
from world.fast_mode import FastRunSettings
from world.gang_config import GangConfig
from world.gang_mix import generate_gang_mix, load_gang_mix_config
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.world_configs import WorldConfigs, read_world_configs
from world.world_sampler import (
    DEFAULT_SAMPLER_PATH,
    WorldSpec,
    file_sha256,
    read_manifest_provenance,
    read_world_manifest,
)
from world.world_sizing import resolve_settings

DEFAULT_RANGES_PATH = DEFAULT_SAMPLER_PATH.parent / "world_ranges.yaml"
RANGE_COLUMNS = ("training", "test", "stress")


def load_manifest_checked(
    path: Path | str,
    sampler_config_path: Path | str | None = None,
    world_ranges_path: Path | str | None = None,
) -> list[WorldSpec]:
    """
    WHAT: Reads a manifest and refuses it if the sampler config or world_ranges.yaml on disk no longer hashes to
    what the manifest recorded. Otherwise it returns the planned worlds.

    WHY: A plan made from older configs would run worlds whose values no longer sit inside today's ranges, and
    nothing downstream would say so.

    Args:
        path: Manifest file written by write_world_manifest.
        sampler_config_path: Sampler config to compare with; defaults to config/world_sampler.yaml.
        world_ranges_path: Ranges file to compare with; defaults to config/world_ranges.yaml.

    Returns:
        The WorldSpec records of the manifest.

    Raises:
        ValueError: Naming the manifest and the file whose hash differs.
    """
    sampler_path = DEFAULT_SAMPLER_PATH if sampler_config_path is None else Path(sampler_config_path)
    ranges_path = DEFAULT_RANGES_PATH if world_ranges_path is None else Path(world_ranges_path)
    provenance = read_manifest_provenance(path)
    if provenance["sampler_config_sha256"] != file_sha256(sampler_path):
        raise ValueError(f"manifest {path} is stale: {sampler_path} changed since the plan was made")
    if provenance["world_ranges_sha256"] != file_sha256(ranges_path):
        raise ValueError(f"manifest {path} is stale: {ranges_path} changed since the plan was made")
    return read_world_manifest(path)


def pinned_configs(
    spec: WorldSpec, sampler_cfg: dict[str, Any], base: WorldConfigs | None = None
) -> WorldConfigs:
    """
    WHAT: Returns a copy of the world configs in which every key parameter of the spec is pinned: its training, test
    and stress ranges all become [value, value]. The base configs are never changed.

    WHY: A world reads only its own range column. Pinning all three keeps training inside test inside stress true,
    which narrowing one column would break. A value outside the original range of the spec's column, or a name the
    sampler config does not list, means an edited plan or config and is refused.

    Args:
        spec: The planned world.
        sampler_cfg: The world_sampler dictionary from load_sampler_config.
        base: Configs to start from; defaults to read_world_configs().

    Returns:
        New WorldConfigs with the same background, failure and device dictionaries and pinned world ranges.

    Raises:
        ValueError: For an unknown key name, a value outside the original range, or a pinned entry that breaks the
            nesting rule.
    """
    start = read_world_configs() if base is None else base
    new_ranges = copy.deepcopy(start.world_ranges)
    for name, value in spec.key_values:
        if name not in sampler_cfg["key_parameters"]:
            raise ValueError(f"key parameter '{name}' of {spec.world_id} is not in the sampler config")
        setting = sampler_cfg["key_parameters"][name]
        # A gang-scope value is only the centre of a per-gang draw (world/gang_mix.py). Writing it into the ranges
        # would also pin ranges that other readers use, so gang-scope values never enter world_ranges.
        if setting["scope"] == "gang":
            continue
        reference = setting["world_ranges_reference"]
        entry = new_ranges["parameters"][reference]
        low, high = start.world_ranges["parameters"][reference][spec.range_name]
        if not low <= value <= high:
            raise ValueError(
                f"{spec.world_id}: {name}={value} is outside the original {spec.range_name} range [{low}, {high}]"
            )
        for column in RANGE_COLUMNS:
            entry[column] = [value, value]
        _check_nesting(reference, entry)
    return dataclasses.replace(start, world_ranges=new_ranges)


def _check_nesting(reference: str, entry: dict[str, Any]) -> None:
    """Raises if a pinned entry's training range is not inside test, or test not inside stress."""
    # core.config_validator.check_nested_ranges also demands min < max, which a pinned [v, v] never meets,
    # so only its nesting rule (contains) is applied here.
    if not contains(entry["test"], entry["training"]):
        raise ValueError(f"pinned '{reference}': training range is not inside the test range")
    if not contains(entry["stress"], entry["test"]):
        raise ValueError(f"pinned '{reference}': test range is not inside the stress range")


def fast_settings_for(spec: WorldSpec, preset_name: str) -> FastRunSettings:
    """
    WHAT: Builds the run settings for a spec: the preset's people, shops and days, but always the spec's own seed
    and range name.

    WHY: The preset's range name would send a validation world to training by habit and hide a wrong mapping; taking
    it from the spec keeps the split rule in one place.

    Args:
        spec: The planned world.
        preset_name: A preset of config/world_sizing.yaml, such as "tiny".

    Returns:
        FastRunSettings for run_interactive_world.

    Raises:
        ValueError: If the preset is unknown.
    """
    return resolve_settings(seed=spec.seed, preset=preset_name, range_name=spec.range_name)


def run_sampled_world(
    spec: WorldSpec,
    preset_name: str,
    gang_configs: tuple[GangConfig, ...] | None,
    sampler_cfg: dict[str, Any],
    code_version: str,
) -> InteractiveWorldOutput:
    """
    WHAT: Runs one planned world with fraud on, using pinned configs and the spec's seed and range name. When
    gang_configs is None the world's gang mix is generated from the spec (world/gang_mix.py); explicit gangs still work.

    WHY: A sampled world is a fraud-on world, so a run without gangs is refused rather than run as a quiet world.
    Generating the mix is what makes sampled worlds differ in their fraud and not only in their labels.

    Args:
        spec: The planned world.
        preset_name: A preset of config/world_sizing.yaml.
        gang_configs: The gangs to run (at least one), or None to generate the mix from the spec.
        sampler_cfg: The world_sampler dictionary.
        code_version: The code version to record in the run id.

    Returns:
        The InteractiveWorldOutput of the run.

    Raises:
        ValueError: If gang_configs is an empty tuple, or if pinning, the preset or the mix is refused.
    """
    if gang_configs is not None and not gang_configs:
        raise ValueError("a sampled world is a fraud-on world and needs at least one gang")
    settings = fast_settings_for(spec, preset_name)
    configs = pinned_configs(spec, sampler_cfg)
    gangs = gang_configs
    if gangs is None:
        # Eligible people are counted in the same world the run will build, so the capacity check is exact.
        world = prepare_world(settings, configs).world
        shop_accounts = {shop.account_id for shop in world.shops}
        eligible = len([person for person in world.people if person.account_id not in shop_accounts])
        gangs = generate_gang_mix(spec, sampler_cfg, load_gang_mix_config(), eligible, configs.world_ranges)
    return run_interactive_world(
        settings,
        RunnerSettings(fraud_enabled=True),
        configs=configs,
        gang_configs=gangs,
        code_version=code_version,
        world_split=spec.split,
    )
