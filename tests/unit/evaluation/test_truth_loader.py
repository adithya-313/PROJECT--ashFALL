"""
WHAT: Unit tests for evaluation truth loader (evaluation/truth.py): asserts temporal role and device
lookups follow half-open intervals, transaction narratives are queryable by payment_id, tampered truth files
are rejected via cryptographic digests and row counts, and operational packages are blocked by direct caller checks.

WHY: Ground truth must be cryptographically verified and strictly restricted to evaluation contexts.
Testing interval bounds, tamper detection, and caller isolation ensures benchmarks receive authoritative labels
while preventing simulated labels from leaking into bank-facing modules.

TERM: evaluation: post-simulation analysis, benchmarking, and detection scoring pipeline.
TERM: hidden truth: ground-truth records describing illicit roles, device controls, and payment stories.
TERM: truth store: evaluation interface providing verified forensic simulation reality to benchmarks.
TERM: half-open span: an interval [valid_from, valid_to) inclusive of start and exclusive of end.
TERM: tamper detection: verifying cryptographic SHA-256 digests and row counts against manifest metadata.
TERM: direct caller guard: runtime inspection of calling module frame to block operational packages.
"""

import importlib.util
from pathlib import Path

import pytest

from core.config_validator import load_yaml
from evaluation.truth import load_truth
from world.calendar_runner import PlannedPayment
from world.fast_mode import FastRunSettings
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.run_identity import REPO_ROOT
from world.truth_records import DeviceControlSpan, PaymentStory, RoleSpan, TruthFacts
from world.truth_writer import write_truth_files

# Tiny world configuration from frozen anchors.
FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
TEST_CODE_VERSION = "test-version-eval"


def test_temporal_span_transitions_and_device_controllers(tmp_path: Path) -> None:
    """
    WHAT: Asserts that TruthStore temporal queries (role_at, owner_at, depth_at, and controllers_of) respect
    half-open intervals: an account rented from minute 1,440 to 2,880 is 'genuine' at 1,439, 'mule' at 1,440,
    and returns to 'genuine' at 2,880. Also verifies controlling device IDs match the active interval.

    WHY: Temporal accuracy is essential for fraud benchmarking; an account cannot be labeled as compromised
    before takeover occurs or after illicit control terminates.

    Args:
        tmp_path: Pytest temporary directory fixture for truth file storage.

    Returns:
        None. Fails if boundary minute queries return incorrect role, ownership, or device states.

    Example:
        # Account with span [1440, 2880) is genuine at 1439, mule at 1440, and genuine at 2880.
    """
    # 1. Configure a temporary rental role span and device control span spanning [1440, 2880).
    account_id = "acc_person_000001"
    role_span = RoleSpan(account_id, "mule", "gang_alpha", 1, 1440, 2880)
    dev_span = DeviceControlSpan("dev_takeover_01", account_id, "gang_alpha", 1440, 2880, False, False, False, False)
    truth = TruthFacts(roles=(role_span,), device_control=(dev_span,))

    # 2. Run simulation, write truth files, and initialize TruthStore via load_truth.
    output = run_interactive_world(
        TINY, RunnerSettings(fraud_enabled=True),
        FraudInputs(truth=truth), code_version=TEST_CODE_VERSION,
    )
    run_dir = write_truth_files(output, tmp_path)
    store = load_truth(tmp_path, run_dir.name)

    # 3. Assert state before span starts (minute 1439): genuine with no gang, depth, or devices.
    assert store.role_at(account_id, 1439) == "genuine"
    assert store.owner_at(account_id, 1439) is None
    assert store.depth_at(account_id, 1439) is None
    assert store.controllers_of(account_id, 1439) == []

    # 4. Assert state during span (minute 1440): active mule under gang_alpha with takeover device.
    assert store.role_at(account_id, 1440) == "mule"
    assert store.owner_at(account_id, 1440) == "gang_alpha"
    assert store.depth_at(account_id, 1440) == 1
    assert store.controllers_of(account_id, 1440) == ["dev_takeover_01"]

    # 5. Assert state at boundary expiration (minute 2880): returns to genuine status.
    assert store.role_at(account_id, 2880) == "genuine"
    assert store.owner_at(account_id, 2880) is None
    assert store.depth_at(account_id, 2880) is None
    assert store.controllers_of(account_id, 2880) == []


def test_story_lookup_by_payment_id(tmp_path: Path) -> None:
    """
    WHAT: Asserts that TruthStore.story_of returns the exact forensic story dictionary corresponding
    to a given payment_id, and raises KeyError when querying an unknown payment identifier.

    WHY: Links transaction events to fraud attribution narratives during post-run evaluation.

    Args:
        tmp_path: Pytest temporary directory fixture for truth file storage.

    Returns:
        None. Fails if story attributes mismatch or unknown payment IDs do not raise KeyError.

    Example:
        # store.story_of(valid_payment_id)['gang_id'] == 'gang_alpha'
    """
    # 1. Run simulation with one fraud payment and narrative story.
    fraud_payment = PlannedPayment(10, "acc_person_000001", "acc_person_000002", 5000, 0, "everyday")
    story = PaymentStory("gang_delta", "cashing_out", 2, "terminal withdrawal hop")
    output = run_interactive_world(
        TINY, RunnerSettings(fraud_enabled=True),
        FraudInputs(payments=(fraud_payment,), truth=TruthFacts(payment_stories=(story,))),
        code_version=TEST_CODE_VERSION,
    )
    run_dir = write_truth_files(output, tmp_path)
    store = load_truth(tmp_path, run_dir.name)

    # 2. Retrieve story of the fraud event and verify all narrative fields.
    fraud_position = output.result.event_sources.index("fraud")
    fraud_payment_id = output.result.payment_ids[fraud_position]
    fraud_story = store.story_of(fraud_payment_id)
    assert fraud_story["gang_id"] == "gang_delta"
    assert fraud_story["block"] == "cashing_out"
    assert fraud_story["layer"] == 2
    assert fraud_story["description"] == "terminal withdrawal hop"

    # 3. Assert querying a non-existent payment identifier raises KeyError.
    with pytest.raises(KeyError, match="No payment story found for payment_id 'missing_id'"):
        store.story_of("missing_id")


def test_tampered_file_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Modifies a single byte in roles.jsonl and asserts that load_truth strictly refuses
    to load the directory with a ValueError describing the cryptographic hash mismatch.

    WHY: Ground rule 15 requires cryptographic tamper proofing: corrupted or maliciously modified
    ground truth must never be accepted into evaluation metrics.

    Args:
        tmp_path: Pytest temporary directory fixture for truth file storage.

    Returns:
        None. Fails if tampered files are loaded without raising ValueError.

    Example:
        # Modifying roles.jsonl causes load_truth to raise ValueError on digest verification.
    """
    # 1. Execute run, write valid truth files, and verify initial loading succeeds.
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    run_dir = write_truth_files(output, tmp_path)
    assert load_truth(tmp_path, run_dir.name) is not None

    # 2. Tamper with roles.jsonl by modifying one character in the file.
    roles_path = run_dir / "roles.jsonl"
    original_bytes = roles_path.read_bytes()
    tampered_bytes = original_bytes + b" "
    roles_path.write_bytes(tampered_bytes)

    # 3. Assert load_truth detects the hash mismatch and refuses to load.
    with pytest.raises(ValueError, match="Cryptographic hash mismatch for 'roles.jsonl'"):
        load_truth(tmp_path, run_dir.name)


def test_caller_guard_blocks_operational_package(tmp_path: Path) -> None:
    """
    WHAT: Creates a temporary module under a package named 'api' in tmp_path, imports load_truth
    from it, and asserts that calling load_truth raises PermissionError.

    WHY: Operational ashfall packages (api, serving, models, etc.) must never read ground truth.
    Testing this runtime guard via a temporary package ensures enforcement without modifying real repo files.

    Args:
        tmp_path: Pytest temporary directory fixture for isolated module creation.

    Returns:
        None. Fails if operational packages can invoke load_truth without PermissionError.

    Example:
        # A module inside package 'api' attempting to call load_truth receives PermissionError.
    """
    # 1. Generate valid truth files in tmp_path.
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    run_dir = write_truth_files(output, tmp_path)

    # 2. Create isolated dummy package directory structure 'api' inside tmp_path.
    pkg_root = tmp_path / "mock_operational"
    api_dir = pkg_root / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    caller_code = (
        "from evaluation.truth import load_truth\n"
        "def try_load(truth_root, run_id):\n"
        "    return load_truth(truth_root, run_id)\n"
    )
    (api_dir / "caller.py").write_text(caller_code, encoding="utf-8")

    # 3. Load caller module under package 'api.caller' using file spec.
    spec = importlib.util.spec_from_file_location("api.caller", api_dir / "caller.py")
    assert spec is not None and spec.loader is not None
    mock_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mock_module)

    # 4. Assert calling load_truth raises PermissionError naming operational package.
    with pytest.raises(PermissionError, match="Operational package 'api' is forbidden"):
        mock_module.try_load(tmp_path, run_dir.name)
