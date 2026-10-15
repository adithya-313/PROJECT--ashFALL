"""
WHAT: Unit test suite validating world sizing presets, parameter knobs, expert files, and machine limits.
It asserts configuration schemas, ratio scaling arithmetic, refusal diagnostic formatting, and capacity limits.
It proves that the legacy explicit CLI command and the new tiny preset produce bit-for-bit identical event files.

WHY: Guarantees that operators can scale simulations safely without risking host out-of-memory crashes.
Proving exact equivalence between legacy commands and presets protects backward compatibility for pipelines.
Automated checks prevent regressions in proportional shop scaling, parameter validation, and refusal suggestions.

TERM: preset: A named pre-configured world configuration specifying standard population and shop counts,
      e.g. the tiny preset.
TERM: knob: A parameter override that automatically scales dependent world dimensions by preset ratios,
      e.g. adjusting people with shops following proportionally.
TERM: expert file: A structured YAML file providing complete explicit simulation parameters for advanced scenarios,
      e.g. an expert.yaml file.
TERM: machine limit: A safety boundary on runtime, memory, or disk preventing execution of oversized worlds,
      e.g. a 30-minute limit.
TERM: refusal: A safety rejection preventing execution of oversized worlds with diagnostic feedback and suggestions,
      e.g. refusing a run exceeding RAM.
"""

import hashlib
from pathlib import Path

import pytest

from core.config_validator import validate_file
from world.estimates import (
    MachineProfile,
    estimate_world,
    read_sizing_config,
)
from world.fast_mode import FastRunSettings, main
from world.machine_limits import MachineState, find_problems, read_machine_state
from world.world_sizing import (
    decide_run,
    largest_fitting_people,
    resolve_settings,
)


def test_sizing_config_has_four_presets_and_limits() -> None:
    """
    WHAT: Validates that config/world_sizing.yaml conforms to schema and defines all required presets and limits.
    It runs validate_file to verify that label and reason fields exist on every policy block.
    It confirms that tiny, small, medium, and large presets exist along with machine limit parameters.

    WHY: Prevents runtime KeyError exceptions during simulation setup by ensuring sizing policy files are valid.
    Schema validation protects against accidental edits or deletions of operational policies in YAML files.
    Enforcing the presence of the four standard presets guarantees predictable sizing options for operators.

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None upon asserting configuration validity without schema errors.

    Example:
        # test_sizing_config_has_four_presets_and_limits()
        # assert len(validate_file(Path('config/world_sizing.yaml'))) == 0
    """
    # 1. Run core configuration validator to verify label and reason metadata.
    config_path = Path("config/world_sizing.yaml")
    validation_errors = validate_file(config_path)
    assert validation_errors == []

    # 2. Parse sizing configuration dictionary and verify preset collection.
    sizing_config = read_sizing_config(config_path)
    presets = sizing_config["presets"]
    for expected_preset in ("tiny", "small", "medium", "large"):
        assert expected_preset in presets

    # 3. Verify operational machine limits configuration block.
    limits = sizing_config["machine_limits"]
    for expected_key in ("max_minutes", "memory_share", "disk_share"):
        assert expected_key in limits


def test_preset_resolves_to_its_sizes() -> None:
    """
    WHAT: Asserts that resolving a named preset template yields its baseline population, shops, and duration.
    It resolves the small preset with seed 7, expecting 300 citizens, 40 shops, 61 days, and training split.
    All resolved dimensions are verified against the authoritative policy parameters.

    WHY: Confirms that standard preset templates correctly expand into complete simulation settings.
    Operators rely on preset templates to instantiate canonical test environments without typing numbers.
    Verifying that baseline values match configuration ensures reproducible simulation setups.

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None upon asserting exact equivalence with baseline preset dimensions.

    Example:
        # settings = resolve_settings(seed=7, preset='small')
        # assert (settings.n_people, settings.n_shops) == (300, 40)
    """
    # 1. Resolve standard small preset template with explicit random seed.
    sizing_config = read_sizing_config()
    settings = resolve_settings(seed=7, preset="small", sizing_config=sizing_config)

    # 2. Verify all resolved simulation dimensions match small preset baselines.
    assert settings.seed == 7
    assert settings.n_people == 300
    assert settings.n_shops == 40
    assert settings.n_days == 61
    assert settings.range_name == "training"


def test_people_knob_scales_shops_by_preset_ratio() -> None:
    """
    WHAT: Asserts that adjusting the citizen population knob scales commercial shops by preset ratio.
    Starting from the small preset baseline (300 people / 40 shops), 600 people scales to 80 shops.
    Similarly, scaling population to 450 people proportionally calculates 60 commercial shops.

    WHY: Preserves realistic merchant-to-citizen economic ratios when operators adjust world size.
    Proportional scaling prevents unrealistic economic imbalances such as 1,000 citizens with only 2 shops.
    Automated checks verify the mathematical rounding formula max(1, floor(people * shops / base + 0.5)).

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None upon asserting proportional shop counts across scaled populations.

    Example:
        # settings = resolve_settings(seed=7, preset='small', n_people=600)
        # assert settings.n_shops == 80
    """
    # 1. Scale small preset population up to 600 people and assert doubled shop count.
    sizing_config = read_sizing_config()
    settings_600 = resolve_settings(seed=7, preset="small", n_people=600, sizing_config=sizing_config)
    assert settings_600.n_people == 600
    assert settings_600.n_shops == 80

    # 2. Scale small preset population to 450 people and assert proportionally scaled shops.
    settings_450 = resolve_settings(seed=7, preset="small", n_people=450, sizing_config=sizing_config)
    assert settings_450.n_people == 450
    assert settings_450.n_shops == 60


def test_explicit_shops_override_scaling() -> None:
    """
    WHAT: Asserts that explicitly providing a shop count knob overrides proportional ratio scaling.
    When small preset is adjusted to 600 people with an explicit 55 shops override, shops remain 55.
    The explicit shop argument takes precedence over the calculated ratio value of 80 shops.

    WHY: Allows operators to design customized stress scenarios with non-standard economic densities.
    Permitting explicit overrides supports testing extreme merchant concentrations or sparse markets.
    Verifying precedence ensures operator intent is respected over automated scaling heuristics.

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None upon asserting that explicit shop overrides supersede ratio calculations.

    Example:
        # settings = resolve_settings(seed=7, preset='small', n_people=600, n_shops=55)
        # assert settings.n_shops == 55
    """
    # 1. Resolve small preset with both population override and explicit shop count.
    sizing_config = read_sizing_config()
    settings = resolve_settings(
        seed=7,
        preset="small",
        n_people=600,
        n_shops=55,
        sizing_config=sizing_config,
    )

    # 2. Assert explicit shop override was applied rather than calculated 80 shops.
    assert settings.n_people == 600
    assert settings.n_shops == 55


def test_expert_file_gives_exact_settings(tmp_path: Path) -> None:
    """
    WHAT: Asserts that an expert YAML file provides exact simulation parameters directly to settings.
    It writes a temporary YAML document defining five fields and resolves it through the sizing engine.
    The resolved FastRunSettings record is asserted to match every value specified in the document.

    WHY: Enables research operators to define reproducible, complex simulation scenarios from versioned files.
    Expert configuration files decouple parameter specification from interactive command shells.
    Verifying exact field mapping guarantees that file-defined experiments execute as authored.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated file creation.

    Returns:
        None upon asserting exact equivalence with expert YAML attributes.

    Example:
        # settings = resolve_settings(expert_path=tmp_path / 'expert.yaml')
        # assert (settings.seed, settings.n_people) == (42, 150)
    """
    # 1. Author valid expert YAML document specifying all five simulation parameters.
    expert_file = tmp_path / "valid_expert.yaml"
    expert_content = "seed: 42\nn_people: 150\nn_shops: 25\nn_days: 61\nrange_name: training\n"
    expert_file.write_text(expert_content, encoding="utf-8")

    # 2. Resolve simulation settings using expert configuration pathway.
    settings = resolve_settings(expert_path=expert_file)

    # 3. Assert all five resolved parameters match author specifications.
    assert settings.seed == 42
    assert settings.n_people == 150
    assert settings.n_shops == 25
    assert settings.n_days == 61
    assert settings.range_name == "training"


def test_expert_file_refuses_unknown_or_missing_fields(tmp_path: Path) -> None:
    """
    WHAT: Asserts that expert YAML files with misspelled or omitted fields raise descriptive ValueErrors.
    It verifies that a file containing 'n_peopel' is refused with a message naming the unknown field.
    It also verifies that omitting a required field such as 'seed' raises an error naming the missing key.

    WHY: Protects operators against silent parameter omission or unintended default fallbacks.
    Misspelling a configuration key such as n_peopel could otherwise lead to erroneous simulation results.
    Naming offending keys directly in error diagnostics accelerates operator debugging.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated file creation.

    Returns:
        None upon asserting appropriate ValueError exceptions naming erroneous keys.

    Example:
        # with pytest.raises(ValueError) as exc: resolve_settings(expert_path=bad_file)
        # assert 'n_peopel' in str(exc.value)
    """
    # 1. Verify unknown misspelled field 'n_peopel' is rejected naming the unknown key.
    bad_unknown = tmp_path / "unknown_field.yaml"
    bad_unknown.write_text("seed: 1\nn_peopel: 50\nn_shops: 10\nn_days: 61\nrange_name: training\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_unknown:
        resolve_settings(expert_path=bad_unknown)
    assert "n_peopel" in str(exc_unknown.value)

    # 2. Verify omitted required field 'seed' is rejected naming the missing key.
    bad_missing = tmp_path / "missing_field.yaml"
    bad_missing.write_text("n_people: 50\nn_shops: 10\nn_days: 61\nrange_name: training\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_missing:
        resolve_settings(expert_path=bad_missing)
    assert "seed" in str(exc_missing.value)


def test_mixed_or_incomplete_modes_are_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that conflicting configuration modes or incomplete explicit parameter sets are rejected.
    It checks that combining expert files with preset flags raises an explanatory ValueError.
    It also checks that invoking preset mode without --seed or omitting explicit arguments is refused.

    WHY: Enforces strict separation between parameterization pathways to eliminate configuration ambiguity.
    Requiring explicit random seeds prevents operators from accidentally running unseeded simulations.
    Refusing incomplete explicit invocations guarantees all world dimensions are intentionally chosen.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated file creation.

    Returns:
        None upon asserting ValueError exceptions for conflicting and incomplete invocations.

    Example:
        # with pytest.raises(ValueError): resolve_settings(expert_path='x.yaml', preset='tiny')
        # with pytest.raises(ValueError): resolve_settings(preset='tiny')
    """
    # 1. Author dummy expert file and verify mixing expert mode with preset flag is rejected.
    dummy_expert = tmp_path / "dummy.yaml"
    dummy_expert.write_text("seed: 1\nn_people: 10\nn_shops: 2\nn_days: 61\nrange_name: training\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_mixed:
        resolve_settings(expert_path=dummy_expert, preset="tiny")
    assert "--preset" in str(exc_mixed.value)

    # 2. Verify preset mode without an explicit seed parameter is rejected.
    with pytest.raises(ValueError) as exc_no_seed:
        resolve_settings(preset="tiny")
    assert "seed" in str(exc_no_seed.value)

    # 3. Verify explicit mode missing required parameter --shops is rejected.
    with pytest.raises(ValueError) as exc_missing_arg:
        resolve_settings(seed=1, n_people=60, n_days=61, range_name="training")
    assert "n_shops" in str(exc_missing_arg.value)


def test_too_short_world_is_refused_before_calibration() -> None:
    """
    WHAT: Asserts that worlds with durations shorter than festival season upper bounds are rejected early.
    It attempts to resolve settings with 50 days in training range, which has an upper bound of 50.
    The resulting ValueError must state that the smallest permissible duration for training is 51 days.

    WHY: Protects calendar planning algorithms from runtime index errors during festival season scheduling.
    Rejecting invalid durations before machine calibration prevents wasting CPU time on infeasible worlds.
    Directing operators to the smallest permissible duration guides immediate parameter correction.

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None upon asserting ValueError diagnostic identifying the minimum allowable duration.

    Example:
        # with pytest.raises(ValueError) as exc:
        #     resolve_settings(seed=1, n_people=60, n_shops=12, n_days=50, range_name='training')
        # assert '51' in str(exc.value)
    """
    # 1. Attempt parameter resolution with duration falling on or below festival peak bound.
    with pytest.raises(ValueError) as exc_info:
        resolve_settings(seed=1, n_people=60, n_shops=12, n_days=50, range_name="training")

    # 2. Assert diagnostic message identifies the smallest allowed horizon (51 days for training).
    error_message = str(exc_info.value)
    assert "51" in error_message
    assert "training" in error_message


def test_oversized_world_is_refused_with_estimate_and_suggestions() -> None:
    """
    WHAT: Asserts that an oversized world is refused with a structured diagnostic message and suggestions.
    Using made-up MachineProfile and MachineState values, it tests a world exceeding host capacity.
    The refusal message is verified to include Refused, Nothing was run, Estimate, preset, and --people.

    WHY: Enforces operator communication standards when proposed simulations exceed host capacity.
    Clear refusal diagnostics empower operators to adapt their workloads without guessing feasible scales.
    Using synthetic hardware parameters ensures this test runs deterministically on any machine.

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None upon asserting required diagnostic tags in the generated refusal message.

    Example:
        # decision = decide_run(huge_settings, profile, state, config, 30.0)
        # assert 'Refused' in decision.message and '--people' in decision.message
    """
    # 1. Define synthetic hardware profile and constrained machine state for deterministic testing.
    profile = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
    state = MachineState(free_memory_bytes=100_000_000, free_disk_bytes=500_000_000)

    # 2. Propose an oversized simulation exceeding host memory and disk thresholds.
    settings = FastRunSettings(seed=101, n_people=500_000, n_shops=50_000, n_days=61, range_name="training")
    decision = decide_run(settings, profile, state, read_sizing_config(), max_minutes=30.0)

    # 3. Assert refusal status and required structured guidance sections.
    assert not decision.fits
    assert "Refused: this world is too big for this machine. Nothing was run." in decision.message
    assert "Estimate:" in decision.message
    assert "Why:" in decision.message
    assert "What you can do:" in decision.message
    assert "--people" in decision.message


def test_time_problem_suggests_max_minutes_and_the_override_fits() -> None:
    """
    WHAT: Asserts that runs failing only runtime limits suggest --max-minutes and succeed when overridden.
    Using synthetic MachineProfile and MachineState parameters, it tests a workload breaking only time.
    The diagnostic is verified to suggest --max-minutes N, and decide_run with N is verified to fit.

    WHY: Verifies the adaptive suggestion mechanism for time-constrained simulation workloads.
    Providing exact runtime budget suggestions unblocks operators without requiring manual calculations.
    Confirming the suggested override succeeds ensures diagnostic guidance is mathematically sound.

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None upon asserting runtime suggestion presence and subsequent override approval.

    Example:
        # dec1 = decide_run(settings, profile, state, config, 5.0)
        # dec2 = decide_run(settings, profile, state, config, 12.0)
        # assert dec2.fits
    """
    # 1. Construct synthetic machine profile with moderate throughput and ample memory and disk.
    profile = MachineProfile(10.0, 1.0, 10.0, 10.0, 100)
    state = MachineState(free_memory_bytes=1_000_000_000, free_disk_bytes=1_000_000_000)
    sizing_config = read_sizing_config()

    # 2. Evaluate workload of 100 people for 61 days (~7,015 payments, ~702 s = ~11.7 min) with 5 min limit.
    settings = FastRunSettings(seed=101, n_people=100, n_shops=20, n_days=61, range_name="training")
    decision = decide_run(settings, profile, state, sizing_config, max_minutes=5.0)

    # 3. Assert time bottleneck is identified and suggests --max-minutes 12.
    assert not decision.fits
    assert any(p.startswith("time:") for p in decision.problems)
    assert "--max-minutes 12" in decision.message

    # 4. Re-evaluate with recommended 12-minute budget and assert approval.
    decision_override = decide_run(settings, profile, state, sizing_config, max_minutes=12.0)
    assert decision_override.fits


def test_suggested_people_count_is_the_largest_that_fits() -> None:
    """
    WHAT: Asserts that largest_fitting_people returns the exact maximum population fitting host capacity.
    Using synthetic profile and state values, it evaluates the calculated maximum population.
    It verifies that the suggested count produces zero limit problems, while count + 1 produces violations.

    WHY: Guarantees mathematical optimality and precision in population sizing recommendations.
    Recommending a population that still fails or under-provisions available resources harms usability.
    Proving that count fits while count + 1 fails confirms the stepped boundary search algorithm.

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None upon asserting boundary condition satisfaction for count and count + 1.

    Example:
        # count = largest_fitting_people(61, profile, state, config, 30.0)
        # assert fits(count) and not fits(count + 1)
    """
    # 1. Define synthetic machine parameters for deterministic capacity testing.
    profile = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
    state = MachineState(free_memory_bytes=100_000_000, free_disk_bytes=500_000_000)
    sizing_config = read_sizing_config()

    # 2. Calculate largest permissible citizen population for 61 days under a 30-minute budget.
    count = largest_fitting_people(61, profile, state, sizing_config, max_minutes=30.0)
    assert count > 0

    # 3. Assert that the calculated population fits within operational constraints.
    est_fitting = estimate_world(count, 61, profile, sizing_config)
    assert not find_problems(est_fitting, state, sizing_config["machine_limits"], max_minutes=30.0)

    # 4. Assert that incrementing population by a single person introduces a capacity violation.
    est_exceeded = estimate_world(count + 1, 61, profile, sizing_config)
    assert len(find_problems(est_exceeded, state, sizing_config["machine_limits"], max_minutes=30.0)) > 0


def test_machine_state_reads_real_memory_and_disk(tmp_path: Path) -> None:
    """
    WHAT: Asserts that read_machine_state measures physical host memory and disk capacity.
    It invokes read_machine_state on a non-existent child directory inside tmp_path.
    Both free_memory_bytes and free_disk_bytes are asserted to be strictly positive integers.

    WHY: Validates platform integration with psutil memory counters and shutil disk metrics.
    Checking non-existent child directories confirms parent folder resolution logic works cleanly.
    Ensuring positive capacities verifies that capacity evaluation routines receive real machine data.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated path resolution.

    Returns:
        None upon asserting positive integer memory and storage capacities.

    Example:
        # state = read_machine_state(tmp_path / 'new')
        # assert state.free_memory_bytes > 0 and state.free_disk_bytes > 0
    """
    # 1. Query real host machine physical memory and disk availability.
    target_path = tmp_path / "subfolder" / "destination"
    state = read_machine_state(target_path)

    # 2. Assert both measured hardware quantities are positive integers.
    assert isinstance(state.free_memory_bytes, int)
    assert isinstance(state.free_disk_bytes, int)
    assert state.free_memory_bytes > 0
    assert state.free_disk_bytes > 0


def test_estimate_only_runs_nothing(tmp_path: Path) -> None:
    """
    WHAT: Asserts that invoking fast mode with --estimate-only exits successfully without writing files.
    It executes main() targeting an empty directory with the tiny preset and the estimate-only flag.
    It verifies that exit status is 0 and that the target directory was not created on the filesystem.

    WHY: Allows operators and scripts to inspect workload feasibility safely without disk writes.
    Preventing directory creation guarantees that dry-run invocations leave no residual filesystem traces.
    Verifying exit code 0 confirms feasibility approval is communicated accurately to shell callers.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated directory verification.

    Returns:
        None upon asserting exit code 0 and non-existence of the target directory.

    Example:
        # code = main(['--preset', 'tiny', '--seed', '101', '--estimate-only', '--out', path])
        # assert code == 0 and not Path(path).exists()
    """
    # 1. Invoke fast mode CLI with --estimate-only targeting non-existent directory.
    out_dir = tmp_path / "dry_run_target"
    argv = [
        "--preset", "tiny",
        "--seed", "101",
        "--estimate-only",
        "--out", str(out_dir),
    ]
    exit_code = main(argv)

    # 2. Assert successful exit code and confirm no directory or files were created.
    assert exit_code == 0
    assert not out_dir.exists()


def test_failed_run_leaves_no_folder(tmp_path: Path) -> None:
    """
    WHAT: Asserts that an invalid fast-mode invocation exits with an error and creates no target folder.
    It invokes main() with an invalid population count of 0 targeting an uncreated directory.
    It asserts exit status 1 and confirms that the destination folder does not exist on disk.

    WHY: Enforces clean filesystem hygiene by preventing partial or empty folder creation on failures.
    Pre-creating folders on failed runs would obstruct subsequent invocations with FileExistsErrors.
    Verifying that validation failures leave no directory behind ensures transactional consistency.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated directory verification.

    Returns:
        None upon asserting error exit code 1 and target folder non-existence.

    Example:
        # code = main([
        #     '--seed', '101', '--people', '0', '--shops', '12', '--days', '61', '--range-name', 'training', '--out', p
        # ])
        # assert code == 1 and not Path(p).exists()
    """
    # 1. Execute CLI invocation with invalid non-positive citizen count.
    out_dir = tmp_path / "failed_execution_dir"
    argv = [
        "--seed", "101",
        "--people", "0",
        "--shops", "12",
        "--days", "61",
        "--range-name", "training",
        "--out", str(out_dir),
    ]
    exit_code = main(argv)

    # 2. Assert failure exit code and confirm target directory was never created.
    assert exit_code == 1
    assert not out_dir.exists()


def test_old_command_and_tiny_preset_give_identical_files(tmp_path: Path) -> None:
    """
    WHAT: Asserts that the legacy explicit command and the new tiny preset write byte-identical outputs.
    It runs both commands and verifies that both events.jsonl files match the 11C-1 golden SHA-256 digest.
    Binary file bytes are compared directly to verify 100% cryptographic parity across invocation modes.

    WHY: Proves that preset templates produce identical outputs to legacy explicit command-line flags.
    Preserving bit-for-bit parity guarantees that downstream regression benchmarks remain fully valid.
    Verifying against the golden SHA-256 digest proves that parameter defaults were configured faithfully.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated run comparisons.

    Returns:
        None upon asserting exact SHA-256 equivalence and byte equality.

    Example:
        # run_legacy_cli(dir1); run_preset_cli(dir2)
        # assert read_bytes(dir1 / 'events.jsonl') == read_bytes(dir2 / 'events.jsonl')
    """
    # 1. Execute legacy explicit fast-mode command into first temporary directory.
    dir_old = tmp_path / "run_explicit"
    exit_old = main([
        "--seed", "101",
        "--people", "60",
        "--shops", "12",
        "--days", "61",
        "--range-name", "training",
        "--out", str(dir_old),
    ])
    assert exit_old == 0

    # 2. Execute new tiny preset fast-mode command into second temporary directory.
    dir_tiny = tmp_path / "run_preset_tiny"
    exit_tiny = main([
        "--preset", "tiny",
        "--seed", "101",
        "--out", str(dir_tiny),
    ])
    assert exit_tiny == 0

    # 3. Assert both runs match authoritative 11C-1 golden SHA-256 digest and binary equality.
    expected_sha256 = "24fc0e25009d79462d508f040a109620dd4308486cc78aee589393fe921f92fd"
    events_old = dir_old / "events.jsonl"
    events_tiny = dir_tiny / "events.jsonl"
    assert hashlib.sha256(events_old.read_bytes()).hexdigest() == expected_sha256
    assert hashlib.sha256(events_tiny.read_bytes()).hexdigest() == expected_sha256
    assert events_old.read_bytes() == events_tiny.read_bytes()
