"""
WHAT: Support helpers for Level 12 Showcase Part F checks (26 through 30): the example pattern files, the example
plugin gang and its two run ids, the coverage numbers of one sampled block, the training and validation manifests, and
one test-split world. No numpy is imported here; random generators come from world helpers.

WHY: Part F shows the Level 12f tools working on real inputs: a pattern file that validates and one that is refused,
a local plugin inside a gang, a block that covers its ranges, validation worlds kept apart from training worlds, and
a test world on the training ranges. The checks stay short because the work lives here.

TERM: stratum: one of n equal slices of a 0-1 scale; a Latin hypercube block puts one world in each slice.
TERM: manifest: the JSON file listing the planned worlds of one split, with the hashes of the configs it came from.
TERM: plugin_sha256: the hash of a plugin's file, written into a gang's plain form so the run id follows the code.
"""

from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.fast_mode import FastRunSettings
from world.gang_config import GangConfig, gang_config_from_plain
from world.gang_planner import GangPlan, plan_gang
from world.interactive_loop import build_component_streams
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import effective_config_for, prepare_world, run_interactive_world
from world.pattern_files import PatternFileError, PatternPreview, validate_pattern_file
from world.run_identity import REPO_ROOT, load_runner_config, make_run_id
from world.tactic_plugins import plugins_registered
from world.world_configs import read_world_configs
from world.world_sampler import (
    WorldSpec,
    key_parameter_range,
    load_sampler_config,
    sample_block,
)
from world.world_sizing import resolve_settings

BASIC_PATTERN = REPO_ROOT / "patterns" / "examples" / "scam_ring_basic.yaml"
OVER_BUDGET_PATTERN = REPO_ROOT / "patterns" / "examples" / "scam_ring_over_budget.yaml"
COOL_OFF_PLUGIN = REPO_ROOT / "patterns" / "plugins" / "cool_off.py"
CODE_VERSION_F = "showcase-12f"
PLUGIN_SEED = 1000000
MINUTES_PER_DAY = 1440
INTEGER_NOTE = (
    "integer settings are stratified on the 0-1 scale, but rounding to whole numbers can merge neighbouring strata, "
    "so distinct values can be fewer than the block size; this is expected"
)


def validate_example_patterns() -> tuple[PatternPreview, str]:
    """
    Validates the basic example pattern and the over-budget one. Returns the basic preview and the refusal message of
    the over-budget file; raises AssertionError unless the refusal is a PatternFileError starting with the file name
    and "over budget:".
    """
    preview = validate_pattern_file(BASIC_PATTERN)
    try:
        validate_pattern_file(OVER_BUDGET_PATTERN)
    except PatternFileError as refusal:
        message = str(refusal)
        prefix = f"{OVER_BUDGET_PATTERN.name}: over budget:"
        assert message.startswith(prefix), f"the over-budget file was refused for another reason: {message}"
        return preview, message
    raise AssertionError("the over-budget example was not refused")


def plugin_gang_plain(with_plugin: bool) -> dict[str, Any]:
    """Returns the plain gang of the plugin test: acquire, optionally cool_off, then scam and route."""
    steps: list[dict[str, Any]] = [{"block": "acquire", "params": {"count": 4}}]
    if with_plugin:
        steps.append({"block": "cool_off", "params": {}})
    steps.append({"block": "scam", "params": {"receiving_mules": 2, "victims_per_receiving_mule": 8}})
    steps.append({"block": "route", "params": {"layers": 2, "accounts_per_layer": 2}})
    return {
        "gang_id": "plugin_showcase_gang", "strength": 1,
        "budget": {"money_paise": 50000000, "time_minutes": 500000, "max_risk": 0.99},
        "start_minute": 1440, "steps": steps,
    }


def _plugin_run_id(settings: FastRunSettings, gang: GangConfig) -> str:
    """Builds the run id the interactive runner gives this gang in this world, from the same effective configuration."""
    config = effective_config_for(settings, None, (gang,), runner_settings=RunnerSettings(fraud_enabled=True))
    return make_run_id(settings.seed, config, CODE_VERSION_F, load_runner_config()["run_id_hex_length"])


def run_plugin_world() -> dict[str, Any]:
    """
    Registers the cool_off plugin for the duration of the call, plans a gang with acquire, cool_off, scam and route in
    the first training world (tiny preset), runs that world, and returns the days waited, the plugin hash recorded in
    the run's effective configuration, the run id, and the run id of the same world planned without the plugin step.
    """
    settings = resolve_settings(seed=PLUGIN_SEED, preset="tiny", range_name="training")
    with plugins_registered([COOL_OFF_PLUGIN]) as loaded:
        gang = gang_config_from_plain(plugin_gang_plain(True))
        streams = build_component_streams(settings.seed, {"gang_devices": 6, "gang_planning": 7})
        plan: GangPlan = plan_gang(
            gang, prepare_world(settings, None).world, streams["gang_planning"],
            reserved_accounts=set(), gang_devices_rng=streams["gang_devices"],
        )
        days = plan.costs[1].time_minutes // MINUTES_PER_DAY
        output: InteractiveWorldOutput = run_interactive_world(
            settings, RunnerSettings(fraud_enabled=True), gang_configs=(gang,), code_version=CODE_VERSION_F
        )
        recorded = output.effective_config["gang_configs"][0]["steps"][1]["plugin_sha256"]
        expected_run_id = _plugin_run_id(settings, gang)
        plugin_hash = loaded[0].sha256
    without_gang = gang_config_from_plain(plugin_gang_plain(False))
    return {
        "days_waited": days, "recorded_sha256": recorded, "file_sha256": plugin_hash,
        "run_id": output.result.run_id, "expected_run_id": expected_run_id,
        "run_id_without_plugin": _plugin_run_id(settings, without_gang),
    }


def strata_hit(values: list[float], low: float, high: float, block_size: int) -> int:
    """Counts the distinct strata (n equal slices of [low, high]) the values fall in; the top edge is in the last."""
    hit: set[int] = set()
    for value in values:
        stratum = int((value - low) / (high - low) * block_size)
        hit.add(min(stratum, block_size - 1))
    return len(hit)


def coverage_of_block() -> dict[str, Any]:
    """
    Plans training block 0 (sampler seed 1) and returns the scatter points of victim_complaint_share against
    share_reported_to_shared_database, the strata each continuous axis hits, and for every integer key parameter its
    number of distinct values and the number of whole numbers in its range. Plan only; no world is run.
    """
    cfg = load_sampler_config()
    ranges = read_world_configs().world_ranges
    block = sample_block("training", 0, 1, cfg, ranges)
    columns = [dict(spec.key_values) for spec in block]
    x_low, x_high = key_parameter_range(cfg["key_parameters"]["victim_complaint_share"], "training", ranges)
    y_low, y_high = key_parameter_range(cfg["key_parameters"]["share_reported_to_shared_database"], "training", ranges)
    xs = [float(column["victim_complaint_share"]) for column in columns]
    ys = [float(column["share_reported_to_shared_database"]) for column in columns]
    integers: dict[str, dict[str, int]] = {}
    for name, setting in cfg["key_parameters"].items():
        if setting["integer"]:
            low, high = key_parameter_range(setting, "training", ranges)
            integers[name] = {
                "distinct_values": len({column[name] for column in columns}),
                "integers_in_range": int(high) - int(low) + 1,
            }
    return {
        "block_size": len(block), "points": [[x, y] for x, y in zip(xs, ys)],
        "x_strata_hit": strata_hit(xs, x_low, x_high, len(block)),
        "y_strata_hit": strata_hit(ys, y_low, y_high, len(block)),
        "integer_settings": integers, "note": INTEGER_NOTE,
    }


def plan_split_blocks(split_for_validation: str, folder: Path) -> dict[str, Any]:
    """
    Plans training block 0 and block 0 of the given split as "validation" worlds, writes both manifests into folder,
    reads them back, and returns the specs read back with the manifests' provenance. Passing "training" as the second
    split reproduces the mistake the check must catch: validation worlds drawn from the training seed block.
    """
    from world.world_sampler import (
        read_manifest_provenance,
        read_world_manifest,
        write_world_manifest,
    )

    cfg = load_sampler_config()
    ranges = read_world_configs().world_ranges
    sampler_path = REPO_ROOT / "config" / "world_sampler.yaml"
    ranges_path = REPO_ROOT / "config" / "world_ranges.yaml"
    training = sample_block("training", 0, 1, cfg, ranges)
    validation = sample_block(split_for_validation, 0, 1, cfg, ranges)
    paths = {"training": folder / "training_manifest.json", "validation": folder / "validation_manifest.json"}
    write_world_manifest(training, paths["training"], 1, sampler_path, ranges_path)
    write_world_manifest(validation, paths["validation"], 1, sampler_path, ranges_path)
    return {
        "reserved": list(cfg["reserved_seeds"]["seeds"]),
        "training": read_world_manifest(paths["training"]),
        "validation": read_world_manifest(paths["validation"]),
        "training_provenance": read_manifest_provenance(paths["training"]),
        "validation_provenance": read_manifest_provenance(paths["validation"]),
    }


def seed_range(specs: list[WorldSpec]) -> str:
    """Returns the first and last seed of a list of planned worlds as text, such as "1000000 to 1000039"."""
    return f"{specs[0].seed} to {specs[-1].seed}"


def run_test_split_world() -> tuple[InteractiveWorldOutput, WorldSpec, int, list[str]]:
    """
    Plans the first world of test block 0, runs it with run_sampled_world (generated gang mix, small preset) and checks
    it with check_world and its labels. Returns the output, the spec, the number of generated gangs and the problems.
    """
    from world.label_builder import build_label_events
    from world.sampled_world import run_sampled_world
    from world.world_invariants import check_world

    cfg = load_sampler_config()
    spec = sample_block("test", 0, 1, cfg, read_world_configs().world_ranges)[0]
    output = run_sampled_world(spec, "small", None, cfg, CODE_VERSION_F)
    gang_ids = {role.gang_id for role in output.result.truth.roles}
    return output, spec, len(gang_ids), check_world(output, build_label_events(output))


def example_pattern_names() -> dict[str, Any]:
    """Returns the names of the two example pattern files, read straight from their YAML for the evidence."""
    return {
        "basic": load_yaml(BASIC_PATTERN)["pattern"]["name"],
        "over_budget": load_yaml(OVER_BUDGET_PATTERN)["pattern"]["name"],
    }
