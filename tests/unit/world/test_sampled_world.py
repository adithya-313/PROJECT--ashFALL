"""
WHAT: Unit tests for running a sampled world: pinned configs, refusals, the run really using its pins, the split
to range mapping, the stale-manifest check and run determinism. One tiny fraud-on world is run once and shared.
Expected values come from the spec or from the config files, never from the run being checked.

WHY: A pinned value that the world quietly ignores, or a stale plan that runs anyway, would make every later
many-worlds result meaningless while looking fine.

TERM: pin: a parameter whose training, test and stress ranges are all [value, value].
"""

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from core.config_validator import load_yaml
from world.fast_mode import event_lines
from world.gang_config import GangConfig, gang_config_from_plain
from world.interactive_loop import build_component_streams
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import effective_config_for, interactive_fast_output
from world.label_records import draw_label_world_settings, load_labels_config
from world.registry_reports import draw_bank_registry_report_share
from world.run_identity import load_runner_config
from world.sampled_world import (
    fast_settings_for,
    load_manifest_checked,
    pinned_configs,
    run_sampled_world,
)
from world.world_configs import WorldConfigs, read_world_configs
from world.world_sampler import (
    WorldSpec,
    load_sampler_config,
    sample_worlds,
    write_world_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLER_PATH = REPO_ROOT / "config" / "world_sampler.yaml"
RANGES_PATH = REPO_ROOT / "config" / "world_ranges.yaml"
GANG_PATH = REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_complaints.yaml"
PRESET = "tiny"
CODE_VERSION = "test-sampled-world"
SAMPLER_SEED = 6


def _gang() -> GangConfig:
    """Loads the complaints gang fixture the way showcase part E does, without importing from showcase."""
    return gang_config_from_plain(load_yaml(GANG_PATH))


def _values(spec: WorldSpec) -> dict[str, float | int]:
    """Returns the spec's key values as a name to value dictionary."""
    return dict(spec.key_values)


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    """Reads the sampler config once for the module; tests must not change it."""
    return load_sampler_config()


@pytest.fixture(scope="module")
def base() -> WorldConfigs:
    """Reads the real world configs once for the module; tests must not change them."""
    return read_world_configs()


@pytest.fixture(scope="module")
def spec(cfg: dict[str, Any], base: WorldConfigs) -> WorldSpec:
    """Plans one training world with the sampler."""
    return sample_worlds("training", 1, SAMPLER_SEED, cfg, base.world_ranges)[0]


@pytest.fixture(scope="module")
def timed_run(cfg: dict[str, Any], spec: WorldSpec) -> tuple[InteractiveWorldOutput, float]:
    """Runs the planned world once with the tiny preset and the complaints gang; returns the output and seconds."""
    started = time.perf_counter()
    output = run_sampled_world(spec, PRESET, (_gang(),), cfg, CODE_VERSION)
    return output, time.perf_counter() - started


def test_pinned_entries_are_flat_and_the_rest_is_untouched(
    spec: WorldSpec, cfg: dict[str, Any], base: WorldConfigs
) -> None:
    """Pinned entries are [v, v] in all three columns, every other entry equals the base, and the base is unchanged."""
    before = copy.deepcopy(base)
    pinned = pinned_configs(spec, cfg, base)
    pinned_references = set()
    for name, value in spec.key_values:
        setting = cfg["key_parameters"][name]
        if setting["scope"] == "gang":
            continue
        reference = setting["world_ranges_reference"]
        pinned_references.add(reference)
        entry = pinned.world_ranges["parameters"][reference]
        for column in ("training", "test", "stress"):
            assert entry[column] == [value, value]
    assert len(pinned_references) == 5
    for reference, entry in base.world_ranges["parameters"].items():
        if reference not in pinned_references:
            assert pinned.world_ranges["parameters"][reference] == entry
    # Gang scope is never pinned: the ranges that gang centres reference stay exactly as they were.
    for reference in ("hops_to_cash_out", "minutes_money_stays_in_mule_account", "victims_per_mule_typical"):
        assert pinned.world_ranges["parameters"][reference] == base.world_ranges["parameters"][reference]
    assert pinned.background_realism == base.background_realism
    assert pinned.failure_realism == base.failure_realism
    assert pinned.device_realism == base.device_realism
    assert base == before


def test_disk_read_parameters_are_not_key_parameters(cfg: dict[str, Any]) -> None:
    """Parameters whose readers load world_ranges.yaml from disk cannot be pinned, so they are not key parameters."""
    # See the note above key_parameters in config/world_sampler.yaml (amendment A-023).
    disk_read = (
        "share_of_accounts_sold_to_several_gangs",
        "mule_origin_share_business",
        "share_of_scams_investment_or_job",
    )
    for name in disk_read:
        assert name not in cfg["key_parameters"]
        for setting in cfg["key_parameters"].values():
            assert setting.get("world_ranges_reference") != name


def test_value_outside_original_range_is_refused(spec: WorldSpec, cfg: dict[str, Any], base: WorldConfigs) -> None:
    """A pinned value above the original range of the spec's column must be refused, naming the value."""
    name, _ = spec.key_values[0]
    reference = cfg["key_parameters"][name]["world_ranges_reference"]
    high = base.world_ranges["parameters"][reference][spec.range_name][1]
    outside = WorldSpec(spec.world_id, spec.split, spec.seed, spec.range_name, ((name, high + 1),))
    with pytest.raises(ValueError, match="outside the original"):
        pinned_configs(outside, cfg, base)


def test_unknown_key_name_is_refused(spec: WorldSpec, cfg: dict[str, Any], base: WorldConfigs) -> None:
    """A spec naming a parameter the sampler config does not list must be refused, naming it."""
    unknown = WorldSpec(spec.world_id, spec.split, spec.seed, spec.range_name, (("made_up_name", 1.0),))
    with pytest.raises(ValueError, match="made_up_name"):
        pinned_configs(unknown, cfg, base)


def test_empty_gang_list_and_unknown_preset_are_refused(spec: WorldSpec, cfg: dict[str, Any]) -> None:
    """A sampled world needs a gang, and an unknown preset must be refused before anything runs."""
    with pytest.raises(ValueError, match="at least one gang"):
        run_sampled_world(spec, PRESET, (), cfg, CODE_VERSION)
    with pytest.raises(ValueError, match="Unknown preset"):
        fast_settings_for(spec, "no_such_preset")


def test_run_used_the_pins(
    timed_run: tuple[InteractiveWorldOutput, float], spec: WorldSpec, cfg: dict[str, Any]
) -> None:
    """The run drew the pinned region count, label settings and registry share; expected values come from the spec."""
    output, _ = timed_run
    wanted = _values(spec)
    assert output.world.parameters_drawn["n_regions"] == wanted["number_of_regions"]

    ranges = output.effective_config["world_configs"]["world_ranges"]
    split = output.effective_config["settings"]["range_name"]
    seed = output.effective_config["settings"]["seed"]
    children = load_runner_config()["random_stream_children"]
    labels = load_labels_config()

    label_streams = build_component_streams(seed, {"label_world_settings": children["label_world_settings"]})
    settings = draw_label_world_settings(ranges, split, label_streams["label_world_settings"], labels)
    assert settings.victim_complaint_share == wanted["victim_complaint_share"]
    assert settings.false_complaint_rate_per_100000 == wanted["false_complaints_per_100000_genuine_payments"]
    assert settings.false_complaint_merchant_weight == wanted["false_complaint_merchant_weight"]

    registry_streams = build_component_streams(seed, {"registry_reports": children["registry_reports"]})
    share = draw_bank_registry_report_share(ranges, split, registry_streams["registry_reports"], labels)
    assert share == wanted["share_reported_to_shared_database"]


def test_effective_config_carries_the_specs_seed_and_range_name(
    timed_run: tuple[InteractiveWorldOutput, float], spec: WorldSpec
) -> None:
    """The run's recorded settings must hold the spec's seed and range name."""
    output, _ = timed_run
    assert output.effective_config["settings"]["seed"] == spec.seed
    assert output.effective_config["settings"]["range_name"] == spec.range_name


def test_validation_spec_gets_the_training_range_name(cfg: dict[str, Any], base: WorldConfigs) -> None:
    """A validation spec reads the training column; this only plans and does not run a world."""
    validation = sample_worlds("validation", 1, SAMPLER_SEED, cfg, base.world_ranges)[0]
    settings = fast_settings_for(validation, PRESET)
    assert settings.range_name == "training"
    assert settings.seed == validation.seed


def test_test_spec_reads_training_and_records_its_split(cfg: dict[str, Any], base: WorldConfigs) -> None:
    """A planned test spec gets range_name 'training', and the effective config records world_split 'test'."""
    test_spec = sample_worlds("test", 1, SAMPLER_SEED, cfg, base.world_ranges)[0]
    settings = fast_settings_for(test_spec, PRESET)
    assert settings.range_name == "training"
    # The same arguments run_sampled_world hands to run_interactive_world, without running a second world.
    recorded = effective_config_for(
        settings, pinned_configs(test_spec, cfg, base), (_gang(),),
        runner_settings=RunnerSettings(fraud_enabled=True), world_split=test_spec.split,
    )
    assert recorded["world_split"] == "test"
    assert recorded["settings"]["range_name"] == "training"


def test_stale_manifest_is_refused_and_fresh_one_accepted(
    tmp_path: Path, spec: WorldSpec
) -> None:
    """A manifest whose world_ranges hash was edited is refused, naming the file; an unedited copy loads."""
    good = tmp_path / "good.json"
    write_world_manifest([spec], good, SAMPLER_SEED, SAMPLER_PATH, RANGES_PATH)
    assert load_manifest_checked(good) == [spec]

    document = json.loads(good.read_text(encoding="utf-8"))
    document["world_ranges_sha256"] = "0" * 64
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="world_ranges.yaml"):
        load_manifest_checked(stale)


def test_same_spec_runs_to_the_same_run_id_and_events(
    timed_run: tuple[InteractiveWorldOutput, float], spec: WorldSpec, cfg: dict[str, Any]
) -> None:
    """Running the same spec twice gives the same run id and the same event-line hash (the only second run)."""
    first, first_seconds = timed_run
    started = time.perf_counter()
    second = run_sampled_world(spec, PRESET, (_gang(),), cfg, CODE_VERSION)
    second_seconds = time.perf_counter() - started
    print(f"first run {first_seconds:.2f}s, second run {second_seconds:.2f}s")
    assert first.result.run_id == second.result.run_id

    def events_hash(output: InteractiveWorldOutput) -> str:
        """Hashes the Level 11 event lines of a run so two runs can be compared with one value."""
        text = "\n".join(event_lines(interactive_fast_output(output)))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert events_hash(first) == events_hash(second)
