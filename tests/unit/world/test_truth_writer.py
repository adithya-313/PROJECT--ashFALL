"""
WHAT: Unit tests for truth_writer module and events-v2 file writer (world/truth_writer.py and
world/interactive_events.py): verifies that payment stories map one-to-one to events, split fraud payments
share stories while maintaining distinct payment IDs, non-empty folders and missing env vars are rejected,
repeated runs produce byte-identical truth artifacts, and events-v2 files contain 16 fields with no truth leaks.

WHY: Simulation ground truth must remain completely accurate and isolated from bank-facing feeds.
Verifying that truth exports and transaction logs are bit-for-bit reproducible and securely separated
ensures evaluation frameworks receive dependable labels without contaminating operational data stores.

TERM: hidden truth: ground-truth records describing illicit roles, device controls, and payment stories.
TERM: truth store: the file set (roles.jsonl, device_control.jsonl, payment_stories.jsonl, manifest.json).
TERM: manifest: metadata file linking file names, row counts, format version, run id, and cryptographic SHA-256.
TERM: payment story: narrative attributing a fraud payment to a gang, tactic block, layer, and description.
TERM: events-v2: Level 11 event line format plus payment_id and planned_payment_id (16 fields).
TERM: split payment: a payment exceeding UPI limit partitioned into multiple compliant transaction legs.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from core.config_validator import load_yaml
from world.calendar_runner import PlannedPayment
from world.fast_mode import FastRunSettings
from world.interactive_events import interactive_event_lines, write_events_v2_file
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.payments import find_ledger_for_account
from world.run_identity import REPO_ROOT
from world.truth_records import DeviceControlSpan, PaymentStory, RoleSpan, TruthFacts
from world.truth_writer import build_truth_rows, truth_root_from_env, write_truth_files

# Tiny world configuration from frozen anchors.
FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
TEST_CODE_VERSION = "test-version-truth"
FORBIDDEN_WORDS = ("source", "fraud", "truth", "role", "gang")


def test_payment_stories_rows_per_event_and_genuine_rows_have_no_gang() -> None:
    """
    WHAT: Asserts that build_truth_rows produces exactly one payment_stories row per simulated event,
    and that genuine events have None for all story attributes (gang_id, block, layer, description).

    WHY: Ground truth payment story files must correlate perfectly with transaction logs while keeping
    legitimate consumer transactions unannotated.

    Args:
        None.

    Returns:
        None. Fails if event counts differ or genuine payments receive spurious gang attribution.

    Example:
        # Seed-101 tiny world produces 2,915 events and 2,915 story rows.
    """
    # 1. Execute a genuine simulation run with fraud disabled.
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    event_count = len(output.result.run_result.events)
    assert event_count == 2915

    # 2. Build truth rows and verify payment story row alignment.
    rows = build_truth_rows(output)
    story_rows = rows["payment_stories"]
    assert len(story_rows) == 2915

    # 3. Assert genuine event attributes reflect uncompromised consumer traffic.
    first_story = story_rows[0]
    assert first_story["origin"] == "planned"
    assert first_story["resumed"] is False
    assert first_story["gang_id"] is None
    assert first_story["block"] is None
    assert first_story["layer"] is None
    assert first_story["description"] is None


def test_fraud_event_carries_story_and_split_payment_shares_story_with_different_payment_ids() -> None:
    """
    WHAT: Verifies that a fraud payment that splits across the UPI transaction limit creates multiple
    events that share the same story metadata while each receiving a distinct payment_id.

    WHY: Payment splitting is an infrastructure compliance rule. Both split legs originate from the
    same illicit intent and must share the payment story while retaining distinct event identifiers.

    Args:
        None.

    Returns:
        None. Fails if split fraud events do not share the story or fail to receive unique payment IDs.

    Example:
        # An oversized fraud transfer against the world's UPI limit creates 2 legs with the same story.
    """
    # 1. Obtain UPI limit from prepared world ledgers and configure an oversized fraud payment that splits.
    prepared = prepare_world(TINY, None)
    sender_ledger = find_ledger_for_account(prepared.world.ledgers, "acc_person_000001")
    assert sender_ledger is not None
    bank_upi_limit = sender_ledger.upi_limit_paise

    fraud_payment = PlannedPayment(
        minute=10,
        from_account="acc_person_000001",
        to_account="acc_person_000002",
        amount_paise=bank_upi_limit + 1000,
        creation_index=0,
        purpose="everyday",
    )
    story = PaymentStory("gang_mule_net", "layering_hop", 1, "mule split laundering hop")
    truth = TruthFacts(payment_stories=(story,))

    # 2. Execute simulation with fraud enabled and inspect generated truth rows.
    output = run_interactive_world(
        TINY, RunnerSettings(fraud_enabled=True),
        FraudInputs(payments=(fraud_payment,), truth=truth),
        code_version=TEST_CODE_VERSION,
    )
    rows = build_truth_rows(output)
    fraud_stories = [row for row in rows["payment_stories"] if row["origin"] == "fraud"]

    # 3. Assert exactly two split legs were recorded with identical stories and distinct payment IDs.
    assert len(fraud_stories) == 2
    assert fraud_stories[0]["payment_id"] != fraud_stories[1]["payment_id"]
    assert fraud_stories[0]["planned_payment_id"] == fraud_stories[1]["planned_payment_id"]
    for leg in fraud_stories:
        assert leg["gang_id"] == "gang_mule_net"
        assert leg["block"] == "layering_hop"
        assert leg["layer"] == 1
        assert leg["description"] == "mule split laundering hop"


def test_truth_refusals_on_missing_stories_and_unknown_account() -> None:
    """
    WHAT: Asserts that build_truth_rows refuses with ValueError when fraud events exist without payment stories,
    or when a role span or device control span references an account not registered in the world's ledgers.

    WHY: Ground truth must remain semantically consistent: fraud transactions cannot be written without narratives,
    and illicit operations cannot claim non-existent bank accounts.

    Args:
        None.

    Returns:
        None. Fails if invalid accounts or story omissions are accepted without raising ValueError.

    Example:
        # build_truth_rows raises ValueError if RoleSpan references an unregistered account.
    """
    # 1. Execute run with a fraud payment but empty payment stories in truth facts.
    fraud_payment = PlannedPayment(10, "acc_person_000001", "acc_person_000002", 5000, 0, "everyday")
    output_no_story = run_interactive_world(
        TINY, RunnerSettings(fraud_enabled=True),
        FraudInputs(payments=(fraud_payment,), truth=TruthFacts(payment_stories=())),
        code_version=TEST_CODE_VERSION,
    )
    with pytest.raises(ValueError, match="Fraud events exist in the run result but payment_stories is empty"):
        build_truth_rows(output_no_story)

    # 2. Execute genuine run and verify rejection of unknown account IDs in role spans.
    output_genuine = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    unknown_role = RoleSpan("acc_nonexistent_person", "mule", "gang_1", 1, 0, None)
    corrupted_roles_result = dataclasses.replace(
        output_genuine.result, truth=TruthFacts(roles=(unknown_role,))
    )
    output_corrupted_roles = dataclasses.replace(output_genuine, result=corrupted_roles_result)
    with pytest.raises(ValueError, match="is not registered in the world's ledgers"):
        build_truth_rows(output_corrupted_roles)

    # 3. Verify rejection of unknown account IDs in device control spans.
    unknown_device = DeviceControlSpan(
        "dev_1", "acc_nonexistent_person", "gang_1", 0, None, False, False, False, False
    )
    corrupted_dev_result = dataclasses.replace(
        output_genuine.result, truth=TruthFacts(device_control=(unknown_device,))
    )
    output_corrupted_dev = dataclasses.replace(output_genuine, result=corrupted_dev_result)
    with pytest.raises(ValueError, match="is not registered in the world's ledgers"):
        build_truth_rows(output_corrupted_dev)


def test_fraud_event_with_invalid_fraud_input_index_raises_value_error() -> None:
    """
    WHAT: Asserts that build_truth_rows raises ValueError naming the payment_id when a fraud event
    in the result has an event_fraud_input_index that is None or out of range.

    WHY: A fraud event cannot enter truth records without a valid index pointing to its payment story;
    missing or corrupt index mapping indicates internal runner inconsistency.

    Args:
        None.

    Returns:
        None. Fails if invalid fraud input indices are accepted or do not raise ValueError.

    Example:
        # build_truth_rows raises ValueError naming the payment_id if event_fraud_input_index is 99.
    """
    # 1. Execute run with a single fraud payment and story.
    payment = PlannedPayment(10, "acc_person_000001", "acc_person_000002", 5000, 0, "everyday")
    story = PaymentStory("gang_alpha", "route", 1, "test story")
    output = run_interactive_world(
        TINY, RunnerSettings(fraud_enabled=True),
        FraudInputs(payments=(payment,), truth=TruthFacts(payment_stories=(story,))),
        code_version=TEST_CODE_VERSION,
    )

    # 2. Corrupt event_fraud_input_index to 99 (out of range) for the fraud event.
    fraud_position = output.result.event_sources.index("fraud")
    corrupted_indices = list(output.result.event_fraud_input_index)
    corrupted_indices[fraud_position] = 99
    corrupted_result = dataclasses.replace(output.result, event_fraud_input_index=corrupted_indices)
    corrupted_output = dataclasses.replace(output, result=corrupted_result)

    # 3. Verify build_truth_rows raises ValueError naming the payment_id.
    payment_id = output.result.payment_ids[fraud_position]
    with pytest.raises(ValueError, match=f"Fraud event '{payment_id}' has invalid event_fraud_input_index: 99"):
        build_truth_rows(corrupted_output)


def test_truth_refusals_on_non_empty_folder_and_missing_env_var(tmp_path: Path) -> None:
    """
    WHAT: Asserts that write_truth_files raises FileExistsError when the destination folder exists and is not
    empty, and that truth_root_from_env raises ValueError when the environment variable is absent or empty.

    WHY: Transactional file emission prevents corrupting previously recorded simulation runs, and strict
    environment variable enforcement ensures truth root paths are never guessed.

    Args:
        tmp_path: Pytest temporary directory fixture for isolated disk operations.

    Returns:
        None. Fails if non-empty directories are overwritten or missing variables are accepted.

    Example:
        # write_truth_files raises FileExistsError if tmp_path / run_id contains files.
    """
    # 1. Create a non-empty directory at the expected run truth destination.
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    target_dir = tmp_path / output.result.run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "existing_artifact.txt").write_text("blocked", encoding="utf-8")

    # 2. Verify write_truth_files rejects the non-empty directory with FileExistsError.
    with pytest.raises(FileExistsError, match="already exists and is not empty; refusing to overwrite"):
        write_truth_files(output, tmp_path)

    # 3. Verify truth_root_from_env refuses empty or missing environment mappings.
    with pytest.raises(ValueError, match="is missing or empty"):
        truth_root_from_env({})
    with pytest.raises(ValueError, match="is missing or empty"):
        truth_root_from_env({"ASHFALL_TRUTH_DIR": "   "})
    assert truth_root_from_env({"ASHFALL_TRUTH_DIR": "runs/custom_truth"}) == Path("runs/custom_truth")


def test_identical_runs_produce_byte_identical_truth_files(tmp_path: Path) -> None:
    """
    WHAT: Executes two simulation runs from the same seed and inputs, writes truth files to separate
    directories, and asserts that all four generated files have identical SHA-256 digests.

    WHY: Simulation determinism is a foundational invariant: identical inputs must produce bit-for-bit
    identical truth artifacts across all runs and platforms.

    Args:
        tmp_path: Pytest temporary directory fixture for isolated file generation.

    Returns:
        None. Fails if cryptographic digests between identical runs differ.

    Example:
        # digests in run_dir_1 match digests in run_dir_2 across all four truth files.
    """
    # 1. Execute two independent runs with identical settings and fixed code version.
    out_1 = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version="frozen-truth-test")
    out_2 = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version="frozen-truth-test")

    # 2. Write truth artifacts to distinct folders in tmp_path.
    dir_1 = write_truth_files(out_1, tmp_path / "run_alpha")
    dir_2 = write_truth_files(out_2, tmp_path / "run_beta")

    # 3. Assert bit-for-bit identical SHA-256 digests across all five truth files.
    # Level 12g: gangs.jsonl records gang syndicate classifications (seen/held-out) in truth-v2.
    file_names = ["roles.jsonl", "device_control.jsonl", "payment_stories.jsonl", "gangs.jsonl", "manifest.json"]
    for name in file_names:
        hash_1 = hashlib.sha256((dir_1 / name).read_bytes()).hexdigest()
        hash_2 = hashlib.sha256((dir_2 / name).read_bytes()).hexdigest()
        assert hash_1 == hash_2


def test_write_events_v2_file_format_and_no_forbidden_words(tmp_path: Path) -> None:
    """
    WHAT: Writes an events-v2 file using write_events_v2_file and asserts that every line has exactly 16
    keys, matches interactive_event_lines output, and contains no revealing truth words.

    WHY: events-v2 is the public bank feed representation. It must remain free of fraud labels while
    including payment_id and planned_payment_id for transaction tracking.

    Args:
        tmp_path: Pytest temporary directory fixture for event log writing.

    Returns:
        None. Fails if line counts, key counts, or forbidden word checks fail.

    Example:
        # Every line in written events-v2 has 16 fields and zero forbidden words.
    """
    # 1. Execute simulation run and write events-v2 log to temporary path.
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    events_path = tmp_path / "events.jsonl"
    digest = write_events_v2_file(output, events_path)

    # 2. Assert digest matches file content and written lines equal interactive_event_lines.
    assert len(digest) == 64
    assert digest == hashlib.sha256(events_path.read_bytes()).hexdigest()
    written_lines = events_path.read_text(encoding="utf-8").splitlines(keepends=True)
    expected_lines = interactive_event_lines(output)
    assert written_lines == expected_lines

    # 3. Verify exactly 16 keys per line and ensure no forbidden words appear in keys.
    for line in written_lines:
        record = json.loads(line)
        assert len(record) == 16
        for key in record:
            assert not any(word in key for word in FORBIDDEN_WORDS)
