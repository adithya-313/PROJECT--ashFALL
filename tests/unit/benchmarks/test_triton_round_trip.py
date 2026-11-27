"""
WHAT: Unit tests for pure helper functions in benchmarks/triton_round_trip.py.

WHY: Verifies statistical summarization, percentile calculation reuse, and report
payload construction without requiring an active Triton server.
"""

import hashlib
from pathlib import Path

import pytest

from benchmarks.triton_round_trip import (
    build_report_payload,
    compute_file_sha256,
    summarize_level,
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


def test_summarize_level_normal() -> None:
    """
    WHAT: Tests statistical summarization across runs for a single in-flight level.

    WHY: Guarantees correct computation of per-run p50/p99 and median-of-run values.
    """
    # 3 runs of 10 samples each
    run_1 = [float(i) for i in range(1, 11)]  # p50=5.5, p99=9.91
    run_2 = [float(i * 2) for i in range(1, 11)]  # p50=11.0, p99=19.82
    run_3 = [float(i * 3) for i in range(1, 11)]  # p50=16.5, p99=29.73

    summary = summarize_level(in_flight=4, run_latencies=[run_1, run_2, run_3])

    assert summary["in_flight"] == 4
    assert len(summary["per_run_p50"]) == 3
    assert len(summary["per_run_p99"]) == 3
    # Median of [5.5, 11.0, 16.5] is 11.0
    assert summary["p50"] == pytest.approx(11.0)
    # Median of [9.91, 19.82, 29.73] is 19.82
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

    WHY: Verifies the presence of required schema fields, level-1 extraction, and git metadata.
    """
    method = {"warm_up_calls": 100, "measured_calls": 1000, "runs": 3, "judged_on": "median_of_run_p99"}
    rows_meta = {"count": 1100, "sources": [], "matrix_sha256": "abc123"}
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
    )

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
