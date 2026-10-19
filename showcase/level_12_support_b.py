"""
WHAT: Support utilities and domain validators for Level 12 Showcase Part B checks.
Provides fixture loaders, environment resolution, runtime caller-guard test harnesses,
static import auditing helpers, and deterministic run comparator functions.

WHY: Isolates simulation execution orchestration and assertion mechanics from check runners,
ensuring all check runner functions remain modular, concise, and under the 30-line code ceiling.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: hidden truth: ground-truth records describing illicit roles, device controls, and payment stories.
TERM: gang configuration: specification of an illicit syndicate's tactical campaign and resource bounds.
TERM: truth store: queryable in-memory representation of cryptographically verified simulation ground truth.
TERM: device control: association between an external gang hardware device and a compromised account.
"""

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from core.truth_import_check import find_truth_violations
from evaluation.truth import TruthStore
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.fast_mode import FastRunSettings
from world.gang_config import GangConfig, gang_config_from_plain
from world.interactive_events import write_events_v2_file
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import device_event_stream, prepare_world, run_interactive_world
from world.truth_writer import write_truth_files

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
FROZEN_ANCHORS_PATH: Path = REPO_ROOT / "config" / "frozen_anchors.yaml"
SHOWCASE_12_CONFIG_PATH: Path = REPO_ROOT / "config" / "showcase_level_12.yaml"
GANG_FIXTURE_PATH: Path = REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_two_layers.yaml"


def load_showcase_b_fixture(path: Path = GANG_FIXTURE_PATH) -> tuple[GangConfig, int]:
    """
    WHAT: Loads the Level 12 Part B gang YAML fixture, returning an instantiated GangConfig
    and the configured over-budget money ceiling in paise.

    WHY: Decouples check execution from magic values, ensuring the gang typology and intentional
    budget overruns originate from committed configuration files.

    Args:
        path: Filesystem Path pointing to the gang fixture YAML file.

    Returns:
        Tuple containing the validated GangConfig and integer over_budget_money_paise.

    Example:
        gang, over_budget = load_showcase_b_fixture()
        assert gang.gang_id == "gang_two_layers" and over_budget == 100
    """
    # 1. Parse fixture data from disk and instantiate the gang configuration.
    data = load_yaml(path)
    gang = gang_config_from_plain(data)
    over_budget_money_paise = int(data["over_budget_money_paise"])

    return gang, over_budget_money_paise


def get_showcase_b_settings() -> tuple[FastRunSettings, str]:
    """
    WHAT: Resolves frozen-anchor simulation parameters (seed 101, 60 people, 12 shops, 61 days)
    and the pinned code_version recorded for Level 12 showcase executions.

    WHY: Enforces Ground Rule 27 by locking simulation scale and seed parameters to frozen anchors,
    guaranteeing bit-for-bit repeatability across independent environments.

    Args:
        None.

    Returns:
        Tuple containing FastRunSettings and the string code_version.

    Example:
        settings, code_version = get_showcase_b_settings()
        assert settings.seed == 101 and code_version == "showcase-12a"
    """
    # 1. Extract simulation dimensions from frozen anchors and code version from showcase config.
    anchors = load_yaml(FROZEN_ANCHORS_PATH)["frozen_anchors"]["fraud_off_events"]
    settings = FastRunSettings(
        seed=anchors["seed"],
        n_people=anchors["n_people"],
        n_shops=anchors["n_shops"],
        n_days=anchors["n_days"],
        range_name=anchors["range_name"],
    )
    code_version = str(load_yaml(SHOWCASE_12_CONFIG_PATH)["showcase_run"]["code_version"])

    return settings, code_version


def run_showcase_b_world(
    gang_config: GangConfig | None = None,
    truth_root: Path | None = None,
) -> tuple[InteractiveWorldOutput, Path | None]:
    """
    WHAT: Executes the frozen-anchor world under fraud-on mode with an optional GangConfig.
    If truth_root is provided, writes truth artifacts and returns the created run folder path.

    WHY: Standardizes world execution across all Part B showcase checks, eliminating boilerplate
    setup while preserving strict execution isolation.

    Args:
        gang_config: Optional GangConfig specification to plan and execute in the world.
        truth_root: Optional filesystem Path to write hidden truth artifacts into.

    Returns:
        Tuple containing the InteractiveWorldOutput and optional run directory Path.

    Example:
        # output, run_dir = run_showcase_b_world(gang, Path("/tmp/truth"))
        # assert output.result.run_result.event_count > 0
    """
    # 1. Resolve frozen-anchor parameters and invoke interactive runner.
    settings, code_version = get_showcase_b_settings()
    gangs: tuple[GangConfig, ...] = (gang_config,) if gang_config is not None else ()
    output = run_interactive_world(
        settings,
        RunnerSettings(fraud_enabled=True),
        gang_configs=gangs,
        code_version=code_version,
    )

    # 2. Serialize truth artifacts to the requested root directory if specified.
    run_dir = write_truth_files(output, truth_root) if truth_root is not None else None
    return output, run_dir


def verify_gang_money_rules(
    output: InteractiveWorldOutput,
    fixture_opening_balance: int,
) -> None:
    """
    WHAT: Validates Level 11 financial invariants (no negative balances, balance equality with
    entry sums, and money conservation including the fixture collector opening balance).

    WHY: Confirms that gang routing transfers strictly respect double-entry bookkeeping
    and never create, destroy, or leak funds.

    Args:
        output: InteractiveWorldOutput container holding post-run bank ledgers.
        fixture_opening_balance: Integer paise deposited into the fixture collector account.

    Returns:
        None. Raises AssertionError if any financial invariant rule fails.

    Example:
        # verify_gang_money_rules(output, 50_000_000)
    """
    # 1. Compute baseline world opening total plus fixture injection funds.
    settings, _ = get_showcase_b_settings()
    baseline_world = prepare_world(settings, None).world
    base_opening = sum(
        ledger.balance_of(acc) for ledger in baseline_world.ledgers for acc in ledger.accounts
    )
    expected_total = base_opening + fixture_opening_balance

    # 2. Assert all financial invariant rules hold across all ledgers.
    ledgers = output.world.ledgers
    neg_errors = check_no_negative_balances(ledgers)
    assert not neg_errors, f"Negative balance invariant broken: {neg_errors}"
    sum_errors = check_balances_equal_entry_sums(ledgers)
    assert not sum_errors, f"Balance entry sum invariant broken: {sum_errors}"
    cons_errors = check_money_conserved(ledgers, expected_total)
    assert not cons_errors, f"Money conservation invariant broken: {cons_errors}"


def verify_runtime_truth_caller_guard(truth_root: Path, run_id: str, temp_dir: Path) -> str:
    """
    WHAT: Dynamically imports a synthetic module named 'api.caller' located inside temp_dir,
    invokes evaluation.truth.load_truth through it, and asserts that PermissionError is raised.

    WHY: Proves runtime enforcement of Ground Rule 15: operational package callers cannot
    read hidden simulation ground truth.

    Args:
        truth_root: Filesystem Path pointing to the directory containing written truth.
        run_id: String run identifier designating the active simulation truth folder.
        temp_dir: Isolated filesystem Path where the mock operational package is constructed.

    Returns:
        String message captured from the PermissionError exception.

    Example:
        # msg = verify_runtime_truth_caller_guard(truth_root, run_id, temp_path)
        # assert "Operational package 'api' is forbidden" in msg
    """
    # 1. Construct temporary mock operational module under package name 'api'.
    api_dir = temp_dir / "mock_operational" / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    caller_py = api_dir / "caller.py"
    caller_code = (
        "from evaluation.truth import load_truth\n"
        "def try_read(root, rid):\n"
        "    return load_truth(root, rid)\n"
    )
    caller_py.write_text(caller_code, encoding="utf-8")

    # 2. Load module under spec 'api.caller' and assert PermissionError on invocation.
    spec = importlib.util.spec_from_file_location("api.caller", caller_py)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.try_read(truth_root, run_id)
    except PermissionError as error:
        return str(error)
    raise AssertionError("Caller guard failed: load_truth did not raise PermissionError for 'api.caller'.")


def verify_static_truth_import_guard(temp_dir: Path) -> list[str]:
    """
    WHAT: Creates a temporary project tree containing an unauthorized truth import in an 'api' module,
    invokes core.truth_import_check.find_truth_violations, and returns detected violations.

    WHY: Proves static AST detection sensitivity: forbidden truth imports outside allowed
    reader directories are caught unconditionally by CI static guards.

    Args:
        temp_dir: Isolated filesystem Path where the mock project tree is built.

    Returns:
        List of violation strings detected by the static analyzer.

    Example:
        # violations = verify_static_truth_import_guard(temp_path)
        # assert len(violations) == 1
    """
    # 1. Construct synthetic operational repository tree with forbidden import.
    mock_repo = temp_dir / "mock_static_repo"
    api_dir = mock_repo / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    (api_dir / "service.py").write_text("from evaluation.truth import load_truth\n", encoding="utf-8")

    # 2. Run static truth scanner against synthetic repository.
    return find_truth_violations(mock_repo)


def compare_reproducible_runs(
    dir1: Path,
    out1: InteractiveWorldOutput,
    dir2: Path,
    out2: InteractiveWorldOutput,
) -> tuple[int, int]:
    """
    WHAT: Compares artifacts between two independent runs of the same seed across events-v2 digest,
    manifest, roles, device control, payment stories, applied gang device events, and device event streams.

    WHY: Verifies deterministic parity and bit-for-bit repeatability across simulation execution,
    stream ordering, and cryptographic truth serialization.

    Args:
        dir1: Filesystem Path to the truth directory of the first run.
        out1: InteractiveWorldOutput from the first simulation execution.
        dir2: Filesystem Path to the truth directory of the second run.
        out2: InteractiveWorldOutput from the second simulation execution.

    Returns:
        Tuple containing (number of matching comparisons, total comparisons evaluated).

    Example:
        # matches, total = compare_reproducible_runs(dir1, out1, dir2, out2)
        # assert matches == total == 7
    """
    # 1. Compare events-v2 file SHA-256 digests.
    matches, total = 0, 7
    ev1 = write_events_v2_file(out1, dir1 / "events.jsonl")
    ev2 = write_events_v2_file(out2, dir2 / "events.jsonl")
    if ev1 == ev2:
        matches += 1

    # 2. Compare cryptographic digests of all 4 truth store artifacts.
    for fname in ("manifest.json", "roles.jsonl", "device_control.jsonl", "payment_stories.jsonl"):
        h1 = hashlib.sha256((dir1 / fname).read_bytes()).hexdigest()
        h2 = hashlib.sha256((dir2 / fname).read_bytes()).hexdigest()
        if h1 == h2:
            matches += 1

    # 3. Compare applied gang device events and merged device event streams.
    if out1.result.applied_gang_device_events == out2.result.applied_gang_device_events:
        matches += 1
    if device_event_stream(out1.result) == device_event_stream(out2.result):
        matches += 1

    return matches, total


def verify_temporal_mule_transition(store: TruthStore) -> tuple[str, dict[str, Any]]:
    """
    WHAT: Evaluates temporal transitions of a leased mule account: role is 'genuine' before recruitment,
    transitions to 'mule' at valid_from, and controllers_of returns the gang device with sim_swap and remote_access.

    WHY: Isolates temporal role resolution and device telemetry assertion from the check runner
    to ensure check runner functions remain well within the 30-line code limit.

    Args:
        store: Verified TruthStore instance loaded from simulation run artifacts.

    Returns:
        Tuple containing the formatted key_number summary string and evidence dictionary.

    Example:
        # key_number, evidence = verify_temporal_mule_transition(store)
        # assert "genuine at min" in key_number
    """
    # 1. Select an acquired mule account and assert role transition across valid_from boundary.
    mule_span = next(r for r in store._roles if r["role"] == "mule")
    mule_account = str(mule_span["account_id"])
    valid_from = int(mule_span["valid_from"])
    assert store.role_at(mule_account, valid_from - 1) == "genuine"
    assert store.role_at(mule_account, valid_from) == "mule"

    # 2. Inspect controlling gang device and assert takeover minute telemetry flags.
    device_span = next(d for d in store._device_control if d["account_id"] == mule_account)
    takeover_minute = int(device_span["valid_from"])
    controllers = store.controllers_of(mule_account, takeover_minute)
    expected_device = str(device_span["device_id"])
    assert controllers == [expected_device], f"Expected [{expected_device}], got {controllers}"
    assert device_span["sim_swap"] is True
    assert device_span["remote_access"] is True

    key_number = (
        f"{mule_account} genuine at min {valid_from - 1}, mule at min {valid_from}, "
        f"{expected_device} (sim_swap=True, remote_access=True) at min {takeover_minute}"
    )
    evidence = {
        "account_id": mule_account,
        "valid_from": valid_from,
        "takeover_minute": takeover_minute,
        "device_id": expected_device,
        "sim_swap": True,
        "remote_access": True,
    }
    return key_number, evidence

