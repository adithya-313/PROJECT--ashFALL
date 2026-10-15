"""
WHAT: Unit test suite for configuration layer loading, validation, stacking, and effective-world reporting.
It tests that layers modify only targeted settings, that clashes resolve by merge order, and that nesting holds.
It verifies that invalid config names, missing metadata, and out-of-order bounds are refused with descriptive errors.

WHY: Verifies the Level 11C-3 layer subsystem ensures bitwise reproducibility when no layers are applied.
Testing stacking semantics and error boundaries protects simulation integrity against malformed layer inputs.
Validating manifest generation and sizing bounds ensures seamless integration with fast mode and CLI tooling.

TERM: layer: A YAML file specifying targeted parameter range overrides for a simulation run.
      For example, a file shifting the Diwali festival peak window bounds.
TERM: merge order: The sequential resolution rule where later layers override earlier layers on clashes.
      For example, layer two overriding a range set by layer one.
TERM: effective world: The composite configuration state produced after applying all layers in sequence.
      For example, the summary of changed ranges and remaining unchanged parameters.
TERM: nesting: The invariant requiring parameter ranges to satisfy training inside test inside stress.
      For example, requiring [30, 50] to sit entirely inside [25, 55].
"""

import json
from pathlib import Path

import numpy as np
import pytest

from world.fast_mode import FastRunSettings, run_fast_world, write_events_file, write_world_files
from world.layers import Layer, apply_layers, describe_effective_world, read_layer
from world.world_configs import CONFIG_NAMES, read_world_configs
from world.world_plan import draw_world_plan
from world.world_sizing import resolve_settings

EXPECTED_SHA256 = "24fc0e25009d79462d508f040a109620dd4308486cc78aee589393fe921f92fd"


def test_no_layers_gives_identical_events(tmp_path: Path) -> None:
    """
    WHAT: Runs fast-mode world simulation with configs=None and configs=read_world_configs() and checks SHA-256 digests.
    It exports payment events to temporary files and verifies both runs produce the canonical 11C-1 hash.
    Neither run modifies the underlying configuration or alters the deterministic event sequence.

    WHY: Proves the off switch functions correctly and that handing in loaded baseline configs causes zero drift.
    Preserving bit-for-bit event reproducibility guarantees backwards compatibility with existing benchmarks.
    It verifies that unlayered execution remains completely byte-identical across invocation pathways.

    Args:
        tmp_path: Pytest temporary directory fixture for isolating written event files.

    Returns:
        None upon asserting both event files match the golden 11C-1 SHA-256 digest.

    Example:
        >>> # Both None and read_world_configs() match EXPECTED_SHA256:
        >>> # assert h1 == EXPECTED_SHA256 and h2 == EXPECTED_SHA256
    """
    # 1. Initialize standard Level 11C-1 fast-mode settings.
    settings = FastRunSettings(101, 60, 12, 61, "training")

    # 2. Run simulation with default disk configs and verify output hash.
    out_default = run_fast_world(settings, configs=None)
    sha_default = write_events_file(out_default, tmp_path / "default_events.jsonl")
    assert sha_default == EXPECTED_SHA256

    # 3. Run simulation with explicitly loaded configs and verify output hash.
    out_loaded = run_fast_world(settings, configs=read_world_configs())
    sha_loaded = write_events_file(out_loaded, tmp_path / "loaded_events.jsonl")
    assert sha_loaded == EXPECTED_SHA256


def test_layer_changes_only_what_it_names(tmp_path: Path) -> None:
    """
    WHAT: Applies the late-festival layer and performs a deep comparison against baseline configurations.
    It verifies that only festival_peak_day_in_world training range is modified and that test/stress remain untouched.
    It asserts that all other 111 parameter settings across all four domain configs remain strictly equal.

    WHY: Proves configuration layers maintain strict isolation and do not cause collateral parameter mutations.
    Validating that unmentioned ranges stay identical guarantees operator changes have surgical precision.
    Deep equality assertions protect against accidental state contamination during layer application.

    Args:
        tmp_path: Pytest temporary directory fixture for writing layer YAML files.

    Returns:
        None upon verifying that untouched settings exactly match baseline values.

    Example:
        >>> # Targeted setting changes while all other settings match:
        >>> # assert layered.background_realism['parameters']['festival_peak_day_in_world']['training'] == [40, 45]
    """
    # 1. Write the late-festival overlay layer to a temporary YAML file.
    layer_file = tmp_path / "late.yaml"
    layer_file.write_text(
        "name: late-festival\n"
        "reason: Shift festival peak window.\n"
        "changes:\n"
        "  background_realism:\n"
        "    festival_peak_day_in_world:\n"
        "      training: [40, 45]\n",
        encoding="utf-8",
    )

    # 2. Load base configs, read layer, and apply overlay.
    base = read_world_configs()
    layer = read_layer(layer_file)
    layered, _ = apply_layers(base, [layer])

    # 3. Verify modified setting has updated training range but identical test and stress ranges.
    target_base = base.background_realism["parameters"]["festival_peak_day_in_world"]
    target_layered = layered.background_realism["parameters"]["festival_peak_day_in_world"]
    assert target_layered["training"] == [40, 45]
    assert target_layered["test"] == target_base["test"]
    assert target_layered["stress"] == target_base["stress"]

    # 4. Verify all other 16 background realism settings are identical to baseline.
    for name, setting in base.background_realism["parameters"].items():
        if name != "festival_peak_day_in_world":
            assert layered.background_realism["parameters"][name] == setting

    # 5. Verify the other three configuration domains are completely untouched.
    assert layered.world_ranges == base.world_ranges
    assert layered.failure_realism == base.failure_realism
    assert layered.device_realism == base.device_realism


def test_layer_with_base_values_changes_nothing(tmp_path: Path) -> None:
    """
    WHAT: Creates a layer setting festival_peak_day_in_world training to [30, 50] and runs simulation.
    It writes generated events to disk and asserts the resulting SHA-256 matches the golden 11C-1 baseline.
    Setting an existing parameter to its current baseline bounds creates zero functional alteration.

    WHY: Proves layer processing logic is semantically neutral when values match baseline values.
    Demonstrates that the layering mechanism introduces zero synthetic artifacts or timing disturbances.
    It confirms that applying identical bounds produces bit-for-bit identical golden outputs.

    Args:
        tmp_path: Pytest temporary directory fixture for writing layer and event files.

    Returns:
        None upon verifying the resulting events digest matches EXPECTED_SHA256.

    Example:
        >>> # Running with base values produces canonical hash:
        >>> # assert sha == EXPECTED_SHA256
    """
    # 1. Write layer specifying exact existing baseline training bounds [30, 50].
    layer_file = tmp_path / "same.yaml"
    layer_file.write_text(
        "name: same-festival\n"
        "reason: Keep festival peak window identical.\n"
        "changes:\n"
        "  background_realism:\n"
        "    festival_peak_day_in_world:\n"
        "      training: [30, 50]\n",
        encoding="utf-8",
    )

    # 2. Apply layer onto baseline configuration.
    base = read_world_configs()
    layer = read_layer(layer_file)
    layered, _ = apply_layers(base, [layer])

    # 3. Execute fast-mode simulation and verify cryptographic events digest.
    settings = FastRunSettings(101, 60, 12, 61, "training")
    output = run_fast_world(settings, configs=layered)
    sha = write_events_file(output, tmp_path / "same_events.jsonl")
    assert sha == EXPECTED_SHA256


def test_late_festival_layer_moves_the_peak() -> None:
    """
    WHAT: Generates world plans across seeds 1 to 5 with the late-festival layer and checks Diwali peak day.
    It draws festival calendars using the layered background configuration and asserts peak_day is between 40 and 45.
    Every seed independently confirms that festival peak timing respects the constrained bounds.

    WHY: Proves parameter changes in layered configs actually govern downstream stochastic world generators.
    Verifying that draws fall strictly within [40, 45] confirms the calendar generator consumes layered ranges.
    Testing multiple seeds guarantees the effect is robust across independent pseudo-random streams.

    Args:
        None: Test executes deterministic draws without external fixtures.

    Returns:
        None upon asserting peak_day is within [40, 45] for all seeds 1 through 5.

    Example:
        >>> # Diwali peak day is within [40, 45]:
        >>> # for seed in range(1, 6): assert 40 <= plan.festivals[0].peak_day <= 45
    """
    # 1. Create late-festival layer shifting festival peak training window to [40, 45].
    base = read_world_configs()
    changes = {"background_realism": {"festival_peak_day_in_world": {"training": [40, 45]}}}
    layer = Layer("late", "Shift diwali window", changes, "dummy_hash", "memory")
    layered, _ = apply_layers(base, [layer])

    # 2. Draw world plans for seeds 1 through 5 and verify Diwali peak day bounds.
    for seed in range(1, 6):
        rng = np.random.default_rng(seed)
        plan = draw_world_plan(
            rng, n_days=61, range_name="training", background_config=layered.background_realism
        )
        diwali = plan.festivals[0]
        assert 40 <= diwali.peak_day <= 45


def test_unknown_config_name_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Attempts to load a layer targeting a non-existent configuration name and asserts ValueError.
    It writes a layer specifying 'invalid_config_name' and verifies the reader rejects it by name.
    The exception message lists allowed configuration names to guide the operator.

    WHY: Prevents typos in configuration section keys from being silently accepted or ignored.
    Enforcing that config names match CONFIG_NAMES ensures all layer changes map to active domains.
    Failing early with actionable diagnostic messages prevents invalid runs from starting.

    Args:
        tmp_path: Pytest temporary directory fixture for writing malformed layer file.

    Returns:
        None upon catching ValueError identifying the unknown configuration name.

    Example:
        >>> # Rejects unknown config section:
        >>> # with pytest.raises(ValueError, match="unknown config name"): read_layer(p)
    """
    # 1. Write layer targeting an unknown configuration name.
    bad_layer = tmp_path / "bad_config.yaml"
    bad_layer.write_text(
        "name: bad-config\n"
        "reason: Test unknown config rejection.\n"
        "changes:\n"
        "  unknown_config_name:\n"
        "    some_setting:\n"
        "      training: [1, 2]\n",
        encoding="utf-8",
    )

    # 2. Assert read_layer raises ValueError naming the unknown config.
    with pytest.raises(ValueError, match="unknown config name 'unknown_config_name'"):
        read_layer(bad_layer)


def test_unknown_setting_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Attempts to load a layer targeting 'festival_peak_day_in_wrld' and asserts ValueError naming the setting.
    It writes a layer with the misspelled setting name under background_realism and checks exception text.
    The error message explicitly names the invalid setting to facilitate immediate debugging.

    WHY: Proves misspelled setting names are refused immediately rather than creating dead config entries.
    Catching unknown parameter names guarantees that every layer alteration affects a real simulation parameter.
    Explicit error naming protects developers from frustrating silent configuration typos.

    Args:
        tmp_path: Pytest temporary directory fixture for writing malformed layer file.

    Returns:
        None upon verifying ValueError explicitly names 'festival_peak_day_in_wrld'.

    Example:
        >>> # Misspelled setting names are rejected by name:
        >>> # with pytest.raises(ValueError, match="festival_peak_day_in_wrld"): read_layer(p)
    """
    # 1. Write layer referencing misspelled setting name 'festival_peak_day_in_wrld'.
    bad_setting_file = tmp_path / "bad_setting.yaml"
    bad_setting_file.write_text(
        "name: typo-layer\n"
        "reason: Test misspelled setting rejection.\n"
        "changes:\n"
        "  background_realism:\n"
        "    festival_peak_day_in_wrld:\n"
        "      training: [40, 45]\n",
        encoding="utf-8",
    )

    # 2. Assert read_layer raises ValueError explicitly naming the misspelled setting.
    with pytest.raises(ValueError, match="festival_peak_day_in_wrld"):
        read_layer(bad_setting_file)


def test_bad_range_name_or_value_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Tests layer validation against misspelled range names, inverted intervals, and non-numeric values.
    It verifies that 'trainng', inverted bounds [45, 40], and non-numeric bounds ['a', 3] all raise ValueError.
    Each invalid variation is written to a temporary file and inspected independently.

    WHY: Enforces strict structural typing and mathematical validity on configuration range bounds.
    Preventing invalid range names protects against unread overlays that never affect active ranges.
    Rejecting inverted and non-numeric bounds protects random sampling routines from runtime crashes.

    Args:
        tmp_path: Pytest temporary directory fixture for creating test YAML files.

    Returns:
        None upon asserting all three malformed variants raise ValueError.

    Example:
        >>> # Bad range name, inverted bounds, and strings are refused:
        >>> # with pytest.raises(ValueError): read_layer(p)
    """
    # 1. Test misspelled range name 'trainng'.
    f1 = tmp_path / "bad_range_name.yaml"
    f1.write_text(
        "name: l1\nreason: test\nchanges:\n"
        "  background_realism:\n    festival_peak_day_in_world:\n      trainng: [40, 45]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trainng"):
        read_layer(f1)

    # 2. Test inverted range [45, 40] where low > high.
    f2 = tmp_path / "inverted_range.yaml"
    f2.write_text(
        "name: l2\nreason: test\nchanges:\n"
        "  background_realism:\n    festival_peak_day_in_world:\n      training: [45, 40]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="low > high"):
        read_layer(f2)

    # 3. Test non-numeric range ['a', 3].
    f3 = tmp_path / "non_numeric.yaml"
    f3.write_text(
        "name: l3\nreason: test\nchanges:\n"
        "  background_realism:\n    festival_peak_day_in_world:\n      training: ['a', 3]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="numeric"):
        read_layer(f3)


def test_missing_name_or_reason_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Tests that layer files missing name or reason keys, or containing empty strings, are refused.
    It tests files missing 'name', missing 'reason', and containing empty whitespace strings for both.
    Every variant asserts read_layer raises a descriptive ValueError.

    WHY: Enforces mandatory audit metadata requirements across all configuration layer files.
    Requiring clear names and reasons guarantees that every layer's origin and rationale can be audited.
    Preventing anonymous or undocumented layers maintains long-term research maintainability.

    Args:
        tmp_path: Pytest temporary directory fixture for generating test layer files.

    Returns:
        None upon asserting missing or empty metadata fields raise ValueError.

    Example:
        >>> # Missing name or reason is rejected:
        >>> # with pytest.raises(ValueError): read_layer(p)
    """
    # 1. Test layer missing 'name' attribute.
    f1 = tmp_path / "no_name.yaml"
    f1.write_text(
        "reason: test\nchanges:\n  background_realism:\n    festival_peak_day_in_world:\n      training: [40, 45]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top-level keys"):
        read_layer(f1)

    # 2. Test layer missing 'reason' attribute.
    f2 = tmp_path / "no_reason.yaml"
    f2.write_text(
        "name: test\nchanges:\n  background_realism:\n    festival_peak_day_in_world:\n      training: [40, 45]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top-level keys"):
        read_layer(f2)

    # 3. Test layer with whitespace-only name.
    f3 = tmp_path / "empty_name.yaml"
    f3.write_text(
        "name: '   '\nreason: test\nchanges:\n"
        "  background_realism:\n    festival_peak_day_in_world:\n      training: [40, 45]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty strings"):
        read_layer(f3)


def test_later_layer_wins_and_both_steps_are_shown(tmp_path: Path) -> None:
    """
    WHAT: Stacks two layers modifying the same parameter range and checks final value and summary description.
    Layer one sets training to [35, 45]; layer two sets training to [40, 42].
    It verifies the layered config retains [40, 42] and that describe_effective_world shows both change steps.

    WHY: Validates merge-order semantics where later layers take precedence during parameter clashes.
    Verifying that both steps appear in the summary provides complete operational auditability.
    Operators can clearly observe intermediate overrides leading up to the final effective parameter state.

    Args:
        tmp_path: Pytest temporary directory fixture for creating layer files.

    Returns:
        None upon asserting the later layer wins and both transition lines appear in order.

    Example:
        >>> # Final value is second layer and summary displays both:
        >>> # assert layered_val == [40, 42] and 'from layer1' in summary and 'from layer2' in summary
    """
    # 1. Create two layers modifying the same festival peak training range.
    f1 = tmp_path / "l1.yaml"
    f1.write_text(
        "name: layer1\nreason: first\nchanges:\n"
        "  background_realism:\n    festival_peak_day_in_world:\n      training: [35, 45]\n",
        encoding="utf-8",
    )
    f2 = tmp_path / "l2.yaml"
    f2.write_text(
        "name: layer2\nreason: second\nchanges:\n"
        "  background_realism:\n    festival_peak_day_in_world:\n      training: [40, 42]\n",
        encoding="utf-8",
    )

    # 2. Read and apply both layers in sequence.
    base = read_world_configs()
    l1 = read_layer(f1)
    l2 = read_layer(f2)
    layered, changes = apply_layers(base, [l1, l2])

    # 3. Assert final configuration reflects the second layer's bounds.
    final_val = layered.background_realism["parameters"]["festival_peak_day_in_world"]["training"]
    assert final_val == [40, 42]

    # 4. Generate effective-world summary and assert both changes appear in chronological order.
    summary = describe_effective_world([l1, l2], changes, layered)
    assert "festival_peak_day_in_world (training): [30, 50] -> [35, 45]   from layer1" in summary
    assert "festival_peak_day_in_world (training): [35, 45] -> [40, 42]   from layer2" in summary


def test_broken_nesting_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Attempts to apply a layer setting training [30, 70] when stress ends at 60 and checks ValueError.
    It asserts that apply_layers refuses the invalid layer and that the error names the setting and the layer.
    The setting festival_peak_day_in_world and layer name broken-nesting must appear in the exception.

    WHY: Guarantees layers cannot construct configurations that violate core statistical validation rules.
    Enforcing the training inside test inside stress invariant prevents invalid parameter configurations.
    Explicitly identifying the offending setting and author layer enables fast remediation of errors.

    Args:
        tmp_path: Pytest temporary directory fixture for writing invalid nesting layer.

    Returns:
        None upon catching ValueError naming both the setting and the layer.

    Example:
        >>> # Refuses broken nesting:
        >>> # with pytest.raises(ValueError, match="festival_peak_day_in_world"): apply_layers(base, [l])
    """
    # 1. Write layer expanding training upper bound beyond baseline stress upper bound (60).
    bad_nesting_file = tmp_path / "bad_nesting.yaml"
    bad_nesting_file.write_text(
        "name: broken-nesting\n"
        "reason: Test nesting boundary violation.\n"
        "changes:\n"
        "  background_realism:\n"
        "    festival_peak_day_in_world:\n"
        "      training: [30, 70]\n",
        encoding="utf-8",
    )

    # 2. Load baseline configs, read layer, and assert apply_layers raises ValueError.
    base = read_world_configs()
    layer = read_layer(bad_nesting_file)
    with pytest.raises(ValueError) as exc_info:
        apply_layers(base, [layer])

    # 3. Verify exception message names both festival_peak_day_in_world and the layer name.
    err_msg = str(exc_info.value)
    assert "festival_peak_day_in_world" in err_msg
    assert "broken-nesting" in err_msg


def test_summary_counts_settings() -> None:
    """
    WHAT: Generates effective-world summary with no layers and verifies computed parameter counts.
    It dynamically sums parameter dictionary entries across all four baseline configs without hardcoding.
    It asserts the summary contains 'unchanged (<total> of <total> settings)' using the computed sum.

    WHY: Ensures summary parameter counts are dynamically calculated rather than hardcoded in code or tests.
    If new parameters are added to baseline configuration files in future tasks, the check remains valid.
    Verifying that all settings remain unchanged when no layers are used confirms zero modification.

    Args:
        None: Dynamically computes counts from baseline configuration files.

    Returns:
        None upon verifying the formatted unchanged count string appears in the summary.

    Example:
        >>> # Total count is computed dynamically from configs:
        >>> # assert f"unchanged ({total} of {total} settings)" in summary
    """
    # 1. Load baseline configs and dynamically compute total settings count across all four configs.
    configs = read_world_configs()
    total_settings = sum(len(getattr(configs, name)["parameters"]) for name in CONFIG_NAMES)

    # 2. Generate summary with no applied layers.
    summary = describe_effective_world([], [], configs)

    # 3. Assert the summary prints the exact computed unchanged settings count.
    expected_line = f"Everything else: unchanged ({total_settings} of {total_settings} settings)"
    assert expected_line in summary


def test_manifest_records_layers_only_when_used(tmp_path: Path) -> None:
    """
    WHAT: Runs fast-mode export without layers and with a layer, inspecting resulting manifest.json files.
    It asserts that the unlayered manifest contains no 'layers' key whatsoever.
    It asserts that the layered manifest contains a 'layers' list holding layer name and SHA-256 digest.

    WHY: Preserves strict byte-for-byte backwards compatibility for standard unlayered simulations.
    Adding the 'layers' key only when layers are applied prevents metadata schema drift on baseline runs.
    Recording name and SHA-256 for layered runs guarantees cryptographic auditability and provenance.

    Args:
        tmp_path: Pytest temporary directory fixture for writing simulation output directories.

    Returns:
        None upon asserting presence or absence of the 'layers' key in manifest metadata.

    Example:
        >>> # Manifest omits layers key when none are used, includes it when used:
        >>> # assert "layers" not in m_none and "layers" in m_layered
    """
    # 1. Run simulation without layers and verify manifest contains no 'layers' key.
    settings = FastRunSettings(101, 10, 4, 61, "training")
    out_dir_none = tmp_path / "run_nolayers"
    manifest_path_none = write_world_files(settings, out_dir_none)
    data_none = json.loads(manifest_path_none.read_text(encoding="utf-8"))
    assert "layers" not in data_none

    # 2. Create layer and run simulation with the layer applied.
    layer_file = tmp_path / "layer.yaml"
    layer_file.write_text(
        "name: audit-layer\nreason: audit\nchanges:\n"
        "  background_realism:\n    festival_peak_day_in_world:\n      training: [40, 45]\n",
        encoding="utf-8",
    )
    base = read_world_configs()
    layer = read_layer(layer_file)
    layered, _ = apply_layers(base, [layer])

    # 3. Verify layered manifest includes 'layers' key with layer name and SHA-256.
    out_dir_layered = tmp_path / "run_layered"
    manifest_path_layered = write_world_files(
        settings, out_dir_layered, configs=layered, layers=[layer]
    )
    data_layered = json.loads(manifest_path_layered.read_text(encoding="utf-8"))
    assert "layers" in data_layered
    assert data_layered["layers"] == [{"name": "audit-layer", "sha256": layer.sha256}]


def test_layered_festival_bound_is_used_for_the_day_check(tmp_path: Path) -> None:
    """
    WHAT: Tests horizon validation with and without a layer widening the festival peak training window to [30, 55].
    Without the layer, baseline upper bound is 50, so --days 55 is permitted by resolve_settings.
    With the layer, upper bound becomes 55, so --days 55 is refused with ValueError.

    WHY: Proves that CLI and sizing parameter validation consumes layered background configuration bounds.
    Ensuring duration exceeds festival peak upper bound prevents index errors in calendar simulations.
    Testing boundary behavior confirms that layered overrides protect sizing resolution before execution.

    Args:
        tmp_path: Pytest temporary directory fixture for writing wide-festival layer.

    Returns:
        None upon verifying --days 55 succeeds without the layer and raises ValueError with the layer.

    Example:
        >>> # Without layer days=55 passes, with layer days=55 is refused:
        >>> # assert s.n_days == 55
        >>> # with pytest.raises(ValueError): resolve_settings(..., n_days=55, background_config=layered.bg)
    """
    # 1. Verify --days 55 is allowed under baseline background realism (training peak upper bound is 50).
    base_settings = resolve_settings(seed=101, n_people=60, n_shops=12, n_days=55, range_name="training")
    assert base_settings.n_days == 55

    # 2. Create layer widening festival peak window to [30, 55] (widening test and stress to keep nesting).
    wide_layer_file = tmp_path / "wide_festival.yaml"
    wide_layer_file.write_text(
        "name: wide-festival\n"
        "reason: Test festival boundary horizon validation.\n"
        "changes:\n"
        "  background_realism:\n"
        "    festival_peak_day_in_world:\n"
        "      training: [30, 55]\n"
        "      test: [25, 58]\n"
        "      stress: [20, 65]\n",
        encoding="utf-8",
    )

    # 3. Read and apply wide layer onto baseline configuration.
    base = read_world_configs()
    layer = read_layer(wide_layer_file)
    layered, _ = apply_layers(base, [layer])

    # 4. Assert resolve_settings refuses --days 55 when using layered background config.
    with pytest.raises(ValueError, match="must be greater than 55"):
        resolve_settings(
            seed=101,
            n_people=60,
            n_shops=12,
            n_days=55,
            range_name="training",
            background_config=layered.background_realism,
        )
