"""
WHAT: Unit test suite verifying the fast-mode world execution orchestrator, event logger, and run manifest.
It validates bit-for-bit determinism, parity with golden event/stamp benchmarks G2 and G3, and schema integrity.
It also ensures strict transactional safety by confirming non-empty destination directories are rejected.

WHY: Guarantees that fast-mode simulations accurately reproduce canonical benchmarks and provide reliable telemetry.
Downstream benchmark runners, evaluation pipelines, and fraud model trainers rely on deterministic transaction streams.
Automated checks prevent regressions in payment event schemas, hardware stamp fidelity, and manifest accounting.

TERM: fast mode: A streamlined simulation execution pipeline running an entire world directly in memory,
      e.g. executing 61 days of town transactions without disk overhead.
TERM: golden hash: A fixed cryptographic digest saved from an earlier known-good simulation run,
      e.g. verifying that event sequences match regression baseline G2.
TERM: JSON Lines: A text format storing structured records where each line is an independent JSON object,
      e.g. recording payment events line by line in events.jsonl.
TERM: run manifest: A metadata document summarizing simulation settings, entity counts, and file digests,
      e.g. manifest.json providing cryptographic provenance for a run.
TERM: determinism: The guarantee that identical seeds and configurations produce bit-for-bit identical outputs,
      e.g. generating identical event logs across repeated runs.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from core.config_validator import load_yaml
from world.fast_mode import (
    FastRunSettings,
    main,
    run_fast_world,
    write_world_files,
)
from world.run_hashes import hash_device_stamps, hash_payment_events


def test_full_run_reproduces_g2_and_g3() -> None:
    """
    WHAT: Asserts that running fast mode on the canonical baseline exactly reproduces golden G2 and G3 hashes.
    It executes a full 61-day world run with seed 101, 60 citizens, 12 shops, and training ranges.
    Event and device stamp hashes are computed and compared against values loaded from golden_values.yaml.

    WHY: Proves that the fast-mode orchestration sequence matches the canonical test execution pipeline.
    Any divergence in random stream generation or execution order would alter the resulting hashes.
    Printing measured hash prefixes provides human-readable verification during test suite execution.

    Args:
        None: This test takes no positional or keyword parameters.

    Returns:
        None: Asserts golden hash equivalence and returns nothing.

    Example:
        # Runs seed 101 through fast mode and verifies G2 and G3 golden hashes match
        # test_full_run_reproduces_g2_and_g3()
    """
    # 1. Initialize canonical baseline configuration matching golden benchmark settings.
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")

    # 2. Execute full world simulation directly in memory.
    output = run_fast_world(settings)

    # 3. Compute cryptographic hashes for simulated payment events and device telemetry stamps.
    event_hash = hash_payment_events(output.result.events)
    stamp_hash = hash_device_stamps(output.result.event_sender_devices)

    # 4. Load authoritative golden benchmark digests directly from the configuration repository.
    golden_doc = load_yaml(Path("scripts/golden_values.yaml"))
    g2_expected = golden_doc["g2"]["expected"]["event_hash"]
    g3_expected = golden_doc["g3"]["expected"]["stamp_hash"]

    # 5. Output measured event statistics and digest prefixes for auditing.
    print(f"MEASURED: events={len(output.result.events)} event_hash={event_hash[:8]} stamp_hash={stamp_hash[:8]}")

    # 6. Verify computed digests match golden expectations exactly.
    assert event_hash == g2_expected
    assert stamp_hash == g3_expected


def test_same_seed_gives_identical_files(tmp_path: Path) -> None:
    """
    WHAT: Asserts that executing fast mode twice with the same seed writes bit-for-bit identical files.
    It writes full simulation runs into two distinct temporary directories using identical parameters.
    Both the events.jsonl event logs and manifest.json metadata files are compared byte-for-byte.

    WHY: Guarantees absolute determinism across filesystem serialization routines.
    Eliminates non-deterministic serialization hazards such as unordered dictionary keys or OS line endings.
    Ensures that downstream machine learning pipelines receive completely reproducible datasets.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated file generation.

    Returns:
        None: Asserts exact binary equality between runs and returns nothing.

    Example:
        # Writes identical simulation runs to two folders and verifies byte equality
        # test_same_seed_gives_identical_files(tmp_path)
    """
    # 1. Initialize standard simulation settings for seed 101.
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")

    # 2. Write simulation output files into first temporary target directory.
    dir_first = tmp_path / "run_first"
    write_world_files(settings, dir_first)

    # 3. Write simulation output files into second temporary target directory.
    dir_second = tmp_path / "run_second"
    write_world_files(settings, dir_second)

    # 4. Compare raw binary bytes of both generated events files.
    events_first_bytes = (dir_first / "events.jsonl").read_bytes()
    events_second_bytes = (dir_second / "events.jsonl").read_bytes()
    assert events_first_bytes == events_second_bytes

    # 5. Compare raw binary bytes of both generated run manifests.
    manifest_first_bytes = (dir_first / "manifest.json").read_bytes()
    manifest_second_bytes = (dir_second / "manifest.json").read_bytes()
    assert manifest_first_bytes == manifest_second_bytes


def test_different_seed_gives_different_events(tmp_path: Path) -> None:
    """
    WHAT: Asserts that running fast mode with different seeds produces distinct event file digests.
    It executes simulation runs for seed 101 and seed 102, writing artifacts into separate directories.
    The SHA-256 digests recorded in the resulting manifest files are asserted to be unequal.

    WHY: Confirms that changing the master random seed alters the generated simulation trajectory.
    Prevents degenerate bugs where random streams fail to consume the provided seed parameter.
    Validates that each seed produces a distinct, statistically unique simulated economy.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated file generation.

    Returns:
        None: Asserts digest divergence across different seeds and returns nothing.

    Example:
        # Compares event hashes from seed 101 and seed 102 to ensure they differ
        # test_different_seed_gives_different_events(tmp_path)
    """
    # 1. Initialize settings for seed 101 and write output files.
    settings_101 = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    manifest_101_path = write_world_files(settings_101, tmp_path / "run_101")

    # 2. Initialize settings for seed 102 and write output files.
    settings_102 = FastRunSettings(seed=102, n_people=60, n_shops=12, n_days=61, range_name="training")
    manifest_102_path = write_world_files(settings_102, tmp_path / "run_102")

    # 3. Read recorded event SHA-256 digests from both run manifests.
    data_101 = json.loads(manifest_101_path.read_text(encoding="utf-8"))
    data_102 = json.loads(manifest_102_path.read_text(encoding="utf-8"))

    # 4. Assert that different random seeds produce distinct event logs.
    assert data_101["events_sha256"] != data_102["events_sha256"]


def test_manifest_matches_events_file(tmp_path: Path) -> None:
    """
    WHAT: Validates that the run manifest accurately describes the contents of the events file.
    It checks that manifest event_count matches the exact line count of events.jsonl.
    It also recomputes the SHA-256 digest of the events file to verify manifest checksum integrity.

    WHY: Guarantees that run manifests provide faithful cryptographic summaries of exported data.
    Downstream data ingestion pipelines rely on manifest counts and digests for validation.
    Prevents metadata desynchronization where summary counts diverge from serialized records.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated file generation.

    Returns:
        None: Asserts manifest consistency with events file and returns nothing.

    Example:
        # Verifies line count, recomputed SHA-256, and attempt counts match manifest
        # test_manifest_matches_events_file(tmp_path)
    """
    # 1. Execute simulation run and write artifacts to temporary directory.
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    out_dir = tmp_path / "run_manifest_test"
    manifest_path = write_world_files(settings, out_dir)

    # 2. Parse generated manifest data and read raw events file lines and bytes.
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    events_path = out_dir / "events.jsonl"
    events_lines = events_path.read_text(encoding="utf-8").splitlines()
    recomputed_sha256 = hashlib.sha256(events_path.read_bytes()).hexdigest()

    # 3. Verify event count matches the serialized line count.
    assert manifest_data["event_count"] == len(events_lines)

    # 4. Verify recorded events SHA-256 matches recomputed file digest.
    assert manifest_data["events_sha256"] == recomputed_sha256

    # 5. Verify total attempted transactions count equals recorded event count.
    assert manifest_data["counts"]["attempts"] == manifest_data["event_count"]


def test_event_records_have_all_fields(tmp_path: Path) -> None:
    """
    WHAT: Validates that every line in events.jsonl conforms strictly to the events-v1 schema.
    It verifies that each record parses as valid JSON and contains exactly the 14 mandatory fields.
    It also validates that simulation minute timestamps are monotonically non-decreasing.

    WHY: Protects downstream consumers from schema violations, missing keys, or temporal anomalies.
    Consistent JSON Lines records enable stream-processing pipelines to ingest events reliably.
    Monotonic minute ordering is essential for chronological replay and time-series modeling.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated file generation.

    Returns:
        None: Asserts schema compliance and chronological ordering and returns nothing.

    Example:
        # Checks 14 mandatory keys and monotonic time ordering across all event lines
        # test_event_records_have_all_fields(tmp_path)
    """
    # 1. Generate simulation output files in temporary directory.
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    out_dir = tmp_path / "run_fields_test"
    write_world_files(settings, out_dir)

    # 2. Define the mandatory 14-field schema required by events-v1 specification.
    expected_fields = {
        "timestamp", "minute", "from_account", "to_account", "amount_paise",
        "from_bank_index", "to_bank_index", "status", "reason", "failure_domain",
        "purpose", "sender_fingerprint", "sender_os_generation", "sender_is_new_device",
    }

    # 3. Iterate through every serialized event line and validate schema and ordering.
    previous_minute = -1
    for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert set(record.keys()) == expected_fields
        assert len(record) == 14
        assert record["minute"] >= previous_minute
        previous_minute = record["minute"]


def test_failed_events_keep_status_and_domain(tmp_path: Path) -> None:
    """
    WHAT: Asserts that failed payment transactions preserve error status and failure domain tags.
    It checks that the count of non-success event lines matches manifest failure statistics.
    It also confirms that every failed event record has a non-null failure domain attribution.

    WHY: Enforces compliance with issue R7 requirements regarding failure tracking and attribution.
    Failed transactions must be recorded without moving money while retaining diagnostic causes.
    Auditability requires downstream models to distinguish bank failures from customer balance limits.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated file generation.

    Returns:
        None: Asserts failure recording and domain attribution and returns nothing.

    Example:
        # Verifies non-success lines match manifest failure count and retain failure domains
        # test_failed_events_keep_status_and_domain(tmp_path)
    """
    # 1. Execute simulation run and write output files.
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    out_dir = tmp_path / "run_failures_test"
    manifest_path = write_world_files(settings, out_dir)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 2. Collect all failed transaction records from the events log.
    failed_events: list[dict[str, Any]] = []
    for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["status"] != "success":
            failed_events.append(record)
            # Verify each failed transaction has a non-null failure domain attribution.
            assert record["failure_domain"] is not None

    # 3. Assert total observed failures matches manifest count.
    assert len(failed_events) == manifest_data["counts"]["failures"]


def test_refuses_non_empty_output_folder(tmp_path: Path) -> None:
    """
    WHAT: Asserts that write_world_files raises FileExistsError if the target folder is not empty.
    It populates a target folder with an existing canary file before invoking file generation.
    It verifies that an error is raised and that the existing canary file remains undisturbed.

    WHY: Guarantees transactional safety by preventing accidental data loss or overwrite of past runs.
    Ensures that simulation outputs are written into clean, isolated filesystem directories.
    Protecting existing directories prevents corrupting previous benchmark or training artifacts.

    Args:
        tmp_path: Pytest temporary directory fixture path for isolated folder testing.

    Returns:
        None: Asserts exception raising and filesystem safety and returns nothing.

    Example:
        # Attempts to write to a non-empty folder and asserts FileExistsError is raised
        # test_refuses_non_empty_output_folder(tmp_path)
    """
    # 1. Create a non-empty target directory containing a canary file.
    out_dir = tmp_path / "occupied_dir"
    out_dir.mkdir(parents=True, exist_ok=True)
    canary_file = out_dir / "canary.txt"
    canary_file.write_text("existing data", encoding="utf-8")

    # 2. Attempt to write simulation files into the non-empty directory.
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    with pytest.raises(FileExistsError) as exc_info:
        write_world_files(settings, out_dir)

    # 3. Verify error message clearly identifies the offending directory.
    assert str(out_dir) in str(exc_info.value)

    # 4. Verify directory content remains untouched and contains only the original canary file.
    assert list(out_dir.iterdir()) == [canary_file]


def test_cli_writes_both_files(tmp_path: Path) -> None:
    """
    WHAT: Asserts that the command-line interface entry point writes both output files and handles collisions.
    It invokes main() with explicit CLI arguments targeting an empty temporary directory.
    It verifies exit code 0 and file creation, followed by exit code 1 when invoked on the same folder.

    WHY: Validates the command-line user interface for automated scripting and operator workflows.
    Ensures proper argument parsing, process exit status propagation, and error handling.
    Confirms that CLI execution correctly respects directory collision safeguards without crashing.

    Args:
        tmp_path: Pytest temporary directory fixture path for CLI execution testing.

    Returns:
        None: Asserts exit codes and file existence and returns nothing.

    Example:
        # Invokes main() twice to verify success exit code 0 and collision exit code 1
        # test_cli_writes_both_files(tmp_path)
    """
    # 1. Define command-line argument list pointing to clean target directory.
    out_dir = tmp_path / "cli_target"
    argv = [
        "--seed", "101",
        "--people", "60",
        "--shops", "12",
        "--days", "61",
        "--range-name", "training",
        "--out", str(out_dir),
    ]

    # 2. Execute CLI entry point on clean target directory and assert success exit code 0.
    exit_code_first = main(argv)
    assert exit_code_first == 0

    # 3. Verify both events.jsonl and manifest.json artifacts were written to disk.
    assert (out_dir / "events.jsonl").exists()
    assert (out_dir / "manifest.json").exists()

    # 4. Re-execute CLI entry point on now-occupied target directory and assert error exit code 1.
    exit_code_second = main(argv)
    assert exit_code_second == 1
