"""
WHAT: Unit test suite verifying the stress-test scenario library, world cards, and grounding audit table.
It asserts that all stress layers are properly labelled, modify only targeted parameters, and execute correctly.
It verifies that world cards capture demographic and financial partitions, and that grounding tables account
for every parameter.

WHY: Enforces rigorous validation for Level 11D-1 stress-testing and operational card generation subsystems.
Proving that deferred scenarios produce zero event drift guarantees backwards compatibility.
Verifying grounding counts guarantees complete empirical provenance across all simulation domains.

TERM: stress test: An extreme scenario configuration testing system resilience against severe operational shocks,
      e.g. simulating a complete central switch outage or an extended bank downtime.
TERM: world card: A standardized summary document recording simulation environment settings, entity distributions,
      e.g. exporting demographic allocations and transaction failure counts to world_card.json.
TERM: grounding table: A structured catalog linking configuration parameter settings to empirical publications,
      e.g. recording the NPCI source article and publication date for technical decline rates.
"""

import json
from pathlib import Path

import numpy as np

from world.fast_mode import FastRunSettings, run_fast_world, write_events_file, write_world_files
from world.grounding import count_by_label, grounding_rows
from world.layers import apply_layers, describe_effective_world, read_layer
from world.world_card import build_world_card
from world.world_configs import CONFIG_NAMES, read_world_configs
from world.world_plan import draw_world_plan, spawn_life_streams, spawn_world_streams

EXPECTED_11C1_SHA256 = "24fc0e25009d79462d508f040a109620dd4308486cc78aee589393fe921f92fd"
STRESS_DIR = Path("scenarios/stress")


def test_every_stress_file_is_labelled() -> None:
    """
    WHAT: Asserts that all six stress-test YAML files parse via read_layer and define complete metadata blocks.
    It verifies that real_event, exaggeration, and takes_effect_from_level attributes are populated in every layer.
    It checks that deferred layers declare their respective activation levels (Level 13 for registry and 12 for backlog).

    WHY: Guarantees that no unlabelled or malformed stress tests exist within the scenarios repository.
    Enforcing explicit real-world event grounding and exaggeration ratios protects simulation credibility.

    Returns:
        None upon asserting all six files satisfy the metadata schema.

    Example:
        >>> test_every_stress_file_is_labelled()
    """
    # 1. Discover all scenario layer files in scenarios/stress/ and assert exactly six files exist.
    stress_files = sorted(STRESS_DIR.glob("*.yaml"))
    assert len(stress_files) == 6, f"Expected 6 stress scenario files, found {len(stress_files)}"

    # 2. Inspect each layer file and assert presence and validity of the stress_test block.
    expected_levels = {"registry-outage": 13, "complaint-backlog": 12}
    for file_path in stress_files:
        layer = read_layer(file_path)
        assert layer.stress_test is not None, f"Layer {layer.name} missing stress_test block"
        st = layer.stress_test
        assert isinstance(st["real_event"], str) and st["real_event"].strip(), f"Empty real_event in {layer.name}"
        assert isinstance(st["exaggeration"], str) and st["exaggeration"].strip(), f"Empty exaggeration in {layer.name}"
        expected_level = expected_levels.get(layer.name)
        assert st["takes_effect_from_level"] == expected_level, f"Wrong activation level in {layer.name}"


def test_stress_files_change_only_what_they_name() -> None:
    """
    WHAT: Applies each stress scenario independently and asserts that all unmentioned parameters remain byte-identical.
    It compares modified configurations against baseline configurations parameter by parameter across all domains.

    WHY: Proves that configuration layers maintain strict isolation and do not cause collateral parameter mutations.
    Validating that untouched ranges stay identical guarantees that stress scenarios have surgical precision.

    Returns:
        None upon asserting all unmentioned parameters strictly match baseline configs.

    Example:
        >>> test_stress_files_change_only_what_they_name()
    """
    # 1. Discover all scenario layer files in scenarios/stress/.
    stress_files = sorted(STRESS_DIR.glob("*.yaml"))

    # 2. Iterate through each stress layer and verify parameter isolation.
    for file_path in stress_files:
        layer = read_layer(file_path)
        base = read_world_configs()
        layered, _ = apply_layers(base, [layer])

        # 3. Compare each parameter setting across all four domain configuration partitions.
        for config_name in CONFIG_NAMES:
            base_params = getattr(base, config_name)["parameters"]
            layered_params = getattr(layered, config_name)["parameters"]
            layer_changes = layer.changes.get(config_name, {})
            for setting_name, base_block in base_params.items():
                if setting_name not in layer_changes:
                    assert layered_params[setting_name] == base_block, (
                        f"Layer {layer.name} modified unmentioned setting {config_name}.{setting_name}"
                    )


def test_big_festival_raises_the_uplift() -> None:
    """
    WHAT: Tests that applying big-festival layer shifts Diwali uplift_fraction into the 0.5 to 0.6 range.
    It compares festival calendar draws between the base world (0.05 to 0.25) and layered world for tiny seed 101.

    WHY: Confirms that festival parameter overrides actively alter stochastic calendar generation during simulation.

    Returns:
        None upon asserting festival uplift bounds reflect the applied stress layer.

    Example:
        >>> test_big_festival_raises_the_uplift()
    """
    # 1. Draw base world plan and check baseline Diwali festival uplift fraction.
    rng_base = np.random.default_rng(101)
    base_plan = draw_world_plan(rng_base, n_days=61, range_name="training")
    base_diwali = base_plan.festivals[0]
    assert 0.05 <= base_diwali.uplift_fraction <= 0.25, f"Unexpected base uplift: {base_diwali.uplift_fraction}"

    # 2. Load big-festival layer and apply overrides onto baseline configurations.
    layer = read_layer(STRESS_DIR / "big-festival.yaml")
    layered_cfgs, _ = apply_layers(read_world_configs(), [layer])

    # 3. Draw layered world plan and verify uplift falls strictly between 0.5 and 0.6.
    rng_layered = np.random.default_rng(101)
    bg_cfg = layered_cfgs.background_realism
    layered_plan = draw_world_plan(rng_layered, n_days=61, range_name="training", background_config=bg_cfg)
    layered_diwali = layered_plan.festivals[0]
    assert 0.5 <= layered_diwali.uplift_fraction <= 0.6, f"Uplift not in [0.5, 0.6]: {layered_diwali.uplift_fraction}"


def test_os_update_wave_is_fast_and_wide() -> None:
    """
    WHAT: Asserts that the os-update-wave layer increases the total count of os_update device events.
    It draws device event schedules under baseline and layered configurations for seed 101 and prints measured counts.

    WHY: Verifies that the stress scenario successfully accelerates hardware update adoption across the population.

    Returns:
        None upon asserting layered os_update events exceed baseline counts.

    Example:
        >>> test_os_update_wave_is_fast_and_wide()
    """
    # 1. Initialize simulation settings, town population, and base device schedule.
    from world.build_world import build_world
    from world.device_schedule import draw_device_event_schedule
    seed = 101
    town = build_world(seed, 60, 12, "training")
    plan_rng, _ = spawn_world_streams(seed)
    base_plan = draw_world_plan(plan_rng, 61, "training")
    dev_rng1, _ = spawn_life_streams(seed)
    base_cfgs = read_world_configs()
    base_events = draw_device_event_schedule(
        base_cfgs.device_realism, dev_rng1, town.people, base_plan.festivals, base_plan.n_days
    )

    # 2. Draw device events under os-update-wave layered configuration.
    layer = read_layer(STRESS_DIR / "os-update-wave.yaml")
    layered_cfgs, _ = apply_layers(base_cfgs, [layer])
    dev_rng2, _ = spawn_life_streams(seed)
    layered_events = draw_device_event_schedule(
        layered_cfgs.device_realism, dev_rng2, town.people, base_plan.festivals, base_plan.n_days
    )

    # 3. Tally and compare os_update event counts across both schedules.
    base_os = sum(1 for e in base_events if e.kind == "os_update")
    layered_os = sum(1 for e in layered_events if e.kind == "os_update")
    print(f"MEASURED: base os_update={base_os}, layered os_update={layered_os}")
    assert layered_os > base_os, f"Expected layered os_update > base os_update ({layered_os} <= {base_os})"


def test_bank_outage_long_lasts_longer() -> None:
    """
    WHAT: Asserts that applying the bank-outage-long scenario stretches every bank downtime incident to 480-720 minutes.
    It draws incident schedules with layered failure realism and checks durations across all generated bank downtimes.

    WHY: Proves that extended downtime configurations effectively govern individual bank outage durations.

    Returns:
        None upon verifying all bank downtime durations fall within the 8 to 12 hour window.

    Example:
        >>> test_bank_outage_long_lasts_longer()
    """
    # 1. Load bank-outage-long layer and apply parameter overrides.
    layer = read_layer(STRESS_DIR / "bank-outage-long.yaml")
    layered_cfgs, _ = apply_layers(read_world_configs(), [layer])

    # 2. Draw world plan using layered failure realism configuration.
    plan_rng, _ = spawn_world_streams(101)
    fail_cfg = layered_cfgs.failure_realism
    plan = draw_world_plan(plan_rng, 61, "training", failure_config=fail_cfg, n_banks=3)

    # 3. Filter bank downtime incidents and assert every duration is between 480 and 720 minutes.
    bank_incidents = [inc for inc in plan.incidents if inc.kind == "bank_downtime"]
    assert bank_incidents, "No bank downtime incidents generated"
    for inc in bank_incidents:
        assert 480 <= inc.duration_minutes <= 720, f"Downtime duration {inc.duration_minutes} not in [480, 720]"


def test_switch_outage_routes_nothing_during_the_incident() -> None:
    """
    WHAT: Tests that every payment attempted during an NPCI incident window under switch-outage fails with npci_central.
    It executes simulation run 101 with the switch-outage layer and inspects transaction outcomes inside incident spans.

    WHY: Proves that the central switch outage scenario produces complete transaction failure during disruption periods.

    Returns:
        None upon verifying all payments inside incident windows fail under the npci_central domain.

    Example:
        >>> test_switch_outage_routes_nothing_during_the_incident()
    """
    # 1. Apply switch-outage layer and run fast-mode simulation with tiny preset.
    layer = read_layer(STRESS_DIR / "switch-outage.yaml")
    layered_cfgs, _ = apply_layers(read_world_configs(), [layer])
    settings = FastRunSettings(101, 60, 12, 61, "training")
    output = run_fast_world(settings, configs=layered_cfgs)

    # 2. Identify all central NPCI incidents and filter payments falling within active windows.
    npci_incidents = [inc for inc in output.plan.incidents if inc.kind == "npci_incident"]
    assert npci_incidents, "Expected at least one NPCI central incident to exist"
    inside_payments = [
        ev for ev in output.result.events
        if any(inc.start_minute <= ev.minute < inc.start_minute + inc.duration_minutes for inc in npci_incidents)
    ]

    # 3. Assert that every transaction attempted during an incident failed with npci_central domain.
    assert inside_payments, "Expected attempted payments inside NPCI incident windows"
    for ev in inside_payments:
        assert ev.status == "failed", f"Payment succeeded during switch outage: {ev}"
        assert ev.failure_domain == "npci_central", f"Unexpected failure domain: {ev.failure_domain}"


def test_not_yet_effective_stress_tests_change_no_events() -> None:
    """
    WHAT: Asserts that registry-outage and complaint-backlog produce the exact golden 11C-1 events hash.
    Neither registry nor complaint backlog settings are read by simulation code, so outputs must remain byte-identical.

    WHY: Proves that deferred scenarios introduce zero functional drift until their respective levels are implemented.

    Returns:
        None upon asserting both layered runs yield EXPECTED_11C1_SHA256.

    Example:
        >>> test_not_yet_effective_stress_tests_change_no_events()
    """
    # 1. Set up standard tiny fast-mode simulation settings and temporary export directory.
    import tempfile
    settings = FastRunSettings(101, 60, 12, 61, "training")

    # 2. Test registry-outage layer against baseline hash.
    with tempfile.TemporaryDirectory() as td:
        l1 = read_layer(STRESS_DIR / "registry-outage.yaml")
        c1, _ = apply_layers(read_world_configs(), [l1])
        out1 = run_fast_world(settings, configs=c1)
        sha1 = write_events_file(out1, Path(td) / "reg_events.jsonl")
        assert sha1 == EXPECTED_11C1_SHA256, f"registry-outage altered events: {sha1}"

    # 3. Test complaint-backlog layer against baseline hash.
    with tempfile.TemporaryDirectory() as td:
        l2 = read_layer(STRESS_DIR / "complaint-backlog.yaml")
        c2, _ = apply_layers(read_world_configs(), [l2])
        out2 = run_fast_world(settings, configs=c2)
        sha2 = write_events_file(out2, Path(td) / "comp_events.jsonl")
        assert sha2 == EXPECTED_11C1_SHA256, f"complaint-backlog altered events: {sha2}"


def test_stress_label_is_printed() -> None:
    """
    WHAT: Verifies that describe_effective_world prints STRESS TEST banners and deferred activation notices.
    It inspects summary text generated from registry-outage to confirm both lines appear in output.

    WHY: Ensures simulation operators receive unambiguous notices regarding active and deferred scenario layers.

    Returns:
        None upon asserting required notice strings are present in summary text.

    Example:
        >>> test_stress_label_is_printed()
    """
    # 1. Load registry-outage layer and compute effective configuration changes.
    layer = read_layer(STRESS_DIR / "registry-outage.yaml")
    layered, changes = apply_layers(read_world_configs(), [layer])

    # 2. Generate summary description and verify stress test banner and activation level notice.
    summary = describe_effective_world([layer], changes, layered)
    assert "STRESS TEST registry-outage:" in summary, "Missing STRESS TEST banner"
    assert "takes effect from Level 13; nothing changes in this world yet." in summary, (
        "Missing deferred activation notice"
    )


def test_world_card_has_banks_regions_and_empty_base_rate() -> None:
    """
    WHAT: Asserts that build_world_card compiles three banks, regional allocations, and empty base-rate records.
    It checks that bank citizen counts and regional citizen counts each sum exactly to total population.

    WHY: Validates schema completeness and demographic integrity within generated world cards.

    Returns:
        None upon asserting all banks, regions, and base-rate fields satisfy structural requirements.

    Example:
        >>> test_world_card_has_banks_regions_and_empty_base_rate()
    """
    # 1. Execute tiny fast-mode simulation and construct world card dictionary.
    settings = FastRunSettings(101, 60, 12, 61, "training")
    output = run_fast_world(settings)
    card = build_world_card(settings, output)

    # 2. Verify banking institutions and citizen allocation sum.
    banks = card["banks"]
    assert len(banks) == 3, f"Expected 3 banks, got {len(banks)}"
    for b in banks:
        for key in ("bank_index", "kind", "customer_share", "weak_kyc_share", "hold_delay_factor", "people_count"):
            assert key in b, f"Missing bank key {key}"
    assert sum(b["people_count"] for b in banks) == settings.n_people

    # 3. Verify regional breakdown and empty placeholder base-rate values.
    regions = card["regions"]
    assert sum(regions["people_per_region"].values()) == settings.n_people
    assert card["base_rate"]["fraud_share"] is None
    assert card["base_rate"]["fraud_inflation_factor"] is None


def test_world_card_lists_applied_stress_tests() -> None:
    """
    WHAT: Tests that a layered run records applied stress tests in world_card.json and world_card.md.
    It runs fast mode with big-festival, asserting card metadata holds its SHA-256 and Markdown contains STRESS TEST.

    WHY: Guarantees cryptographic auditability and clear visual marking of stress tests across exported card artifacts.

    Returns:
        None upon verifying stress test provenance in JSON and Markdown card outputs.

    Example:
        >>> test_world_card_lists_applied_stress_tests()
    """
    # 1. Apply big-festival layer and execute simulation with temporary export directory.
    import tempfile
    layer = read_layer(STRESS_DIR / "big-festival.yaml")
    layered, _ = apply_layers(read_world_configs(), [layer])
    settings = FastRunSettings(101, 10, 4, 61, "training")

    # 2. Export simulation files and verify written world card artifacts.
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "run"
        _ = write_world_files(settings, out_dir, configs=layered, layers=[layer])
        card_data = json.loads((out_dir / "world_card.json").read_text(encoding="utf-8"))
        md_text = (out_dir / "world_card.md").read_text(encoding="utf-8")

        # 3. Assert card JSON holds layer name and hash, and Markdown highlights STRESS TEST.
        assert len(card_data["stress_tests"]) == 1
        assert card_data["stress_tests"][0]["name"] == "big-festival"
        assert card_data["stress_tests"][0]["sha256"] == layer.sha256
        assert "STRESS TEST" in md_text


def test_events_and_manifest_unchanged_by_the_card() -> None:
    """
    WHAT: Asserts that exporting world cards does not modify the contents or schema of events.jsonl or manifest.json.
    It runs an unlayered tiny simulation and asserts events SHA-256 matches 11C-1 and manifest has no new keys.

    WHY: Preserves strict byte-for-byte backwards compatibility across baseline fast-mode export files.

    Returns:
        None upon verifying events hash and manifest keys strictly match Level 11C-1 expectations.

    Example:
        >>> test_events_and_manifest_unchanged_by_the_card()
    """
    # 1. Export unlayered simulation run to a temporary directory.
    import tempfile
    settings = FastRunSettings(101, 60, 12, 61, "training")
    expected_manifest_keys = [
        "clock_start", "counts", "device_counts", "episode_count", "episode_report",
        "event_count", "events_file", "events_sha256", "format", "generator", "settings", "world_fingerprint"
    ]

    # 2. Read manifest metadata and verify event hash and schema keys.
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "run"
        manifest_path = write_world_files(settings, out_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["events_sha256"] == EXPECTED_11C1_SHA256
        assert sorted(manifest.keys()) == sorted(expected_manifest_keys)


def test_grounding_counts_every_setting() -> None:
    """
    WHAT: Asserts that grounding_rows extracts every setting across background, failure, and device configs.
    It verifies that total row count matches the computed sum of parameters and that label counts sum to it.

    WHY: Proves that the grounding catalog provides complete coverage with zero omitted parameters.

    Returns:
        None upon asserting row counts and label tallies match computed configuration parameters.

    Example:
        >>> test_grounding_counts_every_setting()
    """
    # 1. Read baseline configs and dynamically calculate parameter count across the three domains.
    configs = read_world_configs()
    expected_count = sum(
        len(getattr(configs, name)["parameters"])
        for name in ("background_realism", "failure_realism", "device_realism")
    )

    # 2. Generate grounding rows and tally counts by epistemic label.
    rows = grounding_rows(configs)
    label_tallies = count_by_label(rows)

    # 3. Assert row count equals expected total and label sums match.
    assert len(rows) == expected_count, f"Expected {expected_count} rows, got {len(rows)}"
    assert sum(label_tallies.values()) == expected_count, "Label counts do not sum to total settings"
