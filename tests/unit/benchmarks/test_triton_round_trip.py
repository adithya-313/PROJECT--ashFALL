"""
WHAT: Unit tests for pure helper functions in benchmarks/triton_round_trip.py.

WHY: Verifies statistical summarization, percentile calculation reuse, report payload construction,
SHA-256 verification of prepared rows, and environment-dependent URL defaults without requiring
an active Triton server.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks.triton_round_trip import (
    REQUIRED_VALIDATION_ROW_COUNT,
    build_report_payload,
    compute_file_sha256,
    default_triton_url,
    load_prepared_rows,
    summarize_level,
    verify_rows_sha256,
)


def test_compute_file_sha256(tmp_path: Path) -> None:
    """
    WHAT: Tests SHA-256 calculation for a temporary file.

    WHY: Accurate hashing of model artifacts and validation row files is required
    for report integrity.
    """
    test_file = tmp_path / "hello.txt"
    test_file.write_text("ashfall", encoding="utf-8")
    expected_digest = hashlib.sha256(b"ashfall").hexdigest()
    digest = compute_file_sha256(test_file)
    assert digest == expected_digest


def test_default_triton_url() -> None:
    """
    WHAT: Tests default Triton gRPC URL generation depending on 'where' parameter.

    WHY: Container runs must reach Triton at compose service host 'triton:8001',
    while native host runs target 'localhost:8001'.
    """
    assert default_triton_url("here") == "localhost:8001"
    assert default_triton_url("container") == "triton:8001"
    assert default_triton_url("unknown") == "localhost:8001"


def test_verify_rows_sha256_success_and_refusal(tmp_path: Path) -> None:
    """
    WHAT: Tests that verify_rows_sha256 succeeds on matching hash and raises ValueError on mismatch.

    WHY: Ground rule: container run must refuse to execute if prepared rows data has been modified.
    """
    row_file = tmp_path / "rows.npy"
    row_file.write_bytes(b"dummy_numpy_data_bytes")
    correct_sha = hashlib.sha256(b"dummy_numpy_data_bytes").hexdigest()

    # 1. Matching sha256 succeeds and returns the verified hash
    verified = verify_rows_sha256(row_file, correct_sha)
    assert verified == correct_sha

    # 2. Tampered / wrong sha256 raises ValueError
    wrong_sha = "0" * 64
    with pytest.raises(ValueError, match="Prepared rows SHA-256 mismatch"):
        verify_rows_sha256(row_file, wrong_sha)

    # 3. Missing file raises FileNotFoundError
    missing_file = tmp_path / "missing.npy"
    with pytest.raises(FileNotFoundError, match="Prepared rows file not found"):
        verify_rows_sha256(missing_file, correct_sha)


def test_load_prepared_rows(tmp_path: Path) -> None:
    """
    WHAT: Tests loading of verified prepared rows matrix and rejection of invalid shape.

    WHY: Ensures data fed into Triton matches the frozen 54-feature format and row count.
    """
    npy_path = tmp_path / "valid_rows.npy"
    sha_path = tmp_path / "valid_rows.sha256"

    # Create array with shape (1100, 54), float32
    valid_data = np.zeros((REQUIRED_VALIDATION_ROW_COUNT, 54), dtype=np.float32)
    np.save(npy_path, valid_data)
    real_sha = compute_file_sha256(npy_path)
    sha_path.write_text(real_sha, encoding="utf-8")

    matrix, metadata = load_prepared_rows(npy_path, sha_path)
    assert matrix.shape == (REQUIRED_VALIDATION_ROW_COUNT, 54)
    assert matrix.dtype == np.float32
    assert metadata["sha256"] == real_sha
    assert metadata["count"] == REQUIRED_VALIDATION_ROW_COUNT

    # Test invalid shape raises ValueError
    invalid_npy = tmp_path / "invalid_rows.npy"
    invalid_sha = tmp_path / "invalid_rows.sha256"
    np.save(invalid_npy, np.zeros((500, 54), dtype=np.float32))
    invalid_real_sha = compute_file_sha256(invalid_npy)
    invalid_sha.write_text(invalid_real_sha, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid prepared rows matrix shape"):
        load_prepared_rows(invalid_npy, invalid_sha)


def test_summarize_level_normal() -> None:
    """
    WHAT: Tests statistical summarization across runs for a single in-flight level.

    WHY: Guarantees correct computation of per-run p50/p99 and median-of-run values.
    """
    run_1 = [float(i) for i in range(1, 11)]
    run_2 = [float(i * 2) for i in range(1, 11)]
    run_3 = [float(i * 3) for i in range(1, 11)]

    summary = summarize_level(in_flight=4, run_latencies=[run_1, run_2, run_3])

    assert summary["in_flight"] == 4
    assert len(summary["per_run_p50"]) == 3
    assert len(summary["per_run_p99"]) == 3
    assert summary["p50"] == pytest.approx(11.0)
    assert summary["p99"] == pytest.approx(19.82)


def test_summarize_level_empty_raises() -> None:
    """
    WHAT: Tests that summarize_level raises ValueError when provided with no runs.

    WHY: Rejects empty measurement sequences to avoid silent fallback or division by zero.
    """
    with pytest.raises(ValueError, match="Cannot summarize level with zero runs"):
        summarize_level(in_flight=1, run_latencies=[])


def test_build_report_payload_normal() -> None:
    """
    WHAT: Tests assembling the benchmark report document.

    WHY: Verifies the presence of required schema fields, level-1 extraction, where, and git metadata.
    """
    method = {"warm_up_calls": 100, "measured_calls": 1000, "runs": 3, "judged_on": "median_of_run_p99"}
    rows_meta = {"count": 1100, "sources": [], "sha256": "abc123"}
    sweep_summary = [
        {"in_flight": 1, "p50": 1.2, "p99": 2.5, "per_run_p50": [1.1, 1.2, 1.3], "per_run_p99": [2.4, 2.5, 2.6]},
        {"in_flight": 2, "p50": 1.5, "p99": 3.0, "per_run_p50": [1.4, 1.5, 1.6], "per_run_p99": [2.9, 3.0, 3.1]},
    ]
    payload = build_report_payload(
        device_label="cpu",
        method=method,
        rows_meta=rows_meta,
        sweep_summary=sweep_summary,
        git_commit="deadbeef",
        triton_image_tag="nvcr.io/nvidia/tritonserver:24.01-py3",
        where="container",
    )

    assert payload["where"] == "container"
    assert payload["runtime"] == "container"
    assert payload["device_label"] == "cpu"
    assert payload["method"] == method
    assert payload["rows"] == rows_meta
    assert payload["per_run_p99"] == [2.4, 2.5, 2.6]
    assert payload["median_of_run_p99"] == 2.5
    assert len(payload["sweep"]) == 2
    assert payload["triton_image_tag"] == "nvcr.io/nvidia/tritonserver:24.01-py3"
    assert payload["git_commit"] == "deadbeef"


def test_build_report_payload_missing_level_1_raises() -> None:
    """
    WHAT: Tests that build_report_payload raises ValueError if level 1 is missing.

    WHY: Serving limit derivations strictly depend on in-flight 1 statistics.
    """
    method = {"warm_up_calls": 100}
    rows_meta = {"count": 1100}
    sweep_summary = [
        {"in_flight": 2, "p50": 1.5, "p99": 3.0, "per_run_p50": [1.4, 1.5, 1.6], "per_run_p99": [2.9, 3.0, 3.1]}
    ]
    with pytest.raises(ValueError, match="Sweep summary lacks in_flight level 1 results"):
        build_report_payload(
            device_label="cpu",
            method=method,
            rows_meta=rows_meta,
            sweep_summary=sweep_summary,
            git_commit="deadbeef",
        )
