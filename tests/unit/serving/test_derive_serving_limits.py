"""
WHAT: Unit tests for serving limits derivation function in scripts/derive_level19_serving_limits.py.

WHY: Verifies correct calculation of device selection, safety margin, decision timeout,
and overload max in-flight level under normal conditions, and verifies that appropriate
STOP condition ValueErrors are raised when timeout or concurrency limits fail.
"""

from typing import Any

import pytest

from scripts.derive_level19_serving_limits import derive_serving_limits


def _create_sample_reports(
    decision_p99s: list[float],
    decision_median: float,
    cpu_median: float,
    gpu_median: float,
    gpu_sweep: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Helper creating minimal synthetic report structures for testing."""
    non_scoring = {
        "run_p99_ms": {"decision": decision_p99s},
        "medians_of_run_p99_ms": {"decision": decision_median},
    }
    cpu_report = {
        "device_label": "cpu",
        "median_of_run_p99": cpu_median,
        "sweep": [{"in_flight": 1, "p99": cpu_median}],
    }
    gpu_report = {
        "device_label": "gpu",
        "median_of_run_p99": gpu_median,
        "sweep": gpu_sweep,
    }
    return non_scoring, cpu_report, gpu_report


def test_derive_serving_limits_normal_case() -> None:
    """
    WHAT: Tests derivation when all stages fit within budget.

    WHY: Verifies normal-case device selection, margin, timeout calculation,
    and overload in-flight selection.
    """
    # Decision spread: max 12.0 - min 10.0 = margin 2.0. Median: 11.0.
    # Timeout = 100 - 11.0 - 2.0 = 87.0 ms.
    # GPU (1.5 ms) < CPU (2.5 ms) -> chosen GPU.
    # Largest GPU sweep level <= 87.0 is 16 (p99 15.0).
    non_scoring, cpu_rep, gpu_rep = _create_sample_reports(
        decision_p99s=[10.0, 11.0, 12.0],
        decision_median=11.0,
        cpu_median=2.5,
        gpu_median=1.5,
        gpu_sweep=[
            {"in_flight": 1, "p99": 1.5},
            {"in_flight": 2, "p99": 3.0},
            {"in_flight": 4, "p99": 6.0},
            {"in_flight": 8, "p99": 10.0},
            {"in_flight": 16, "p99": 15.0},
        ],
    )

    result = derive_serving_limits(
        non_scoring_data=non_scoring,
        cpu_data=cpu_rep,
        gpu_data=gpu_rep,
        budget_max_ms=100.0,
        non_scoring_source="reports/latency/non_scoring.json",
        cpu_source="reports/latency/cpu.json",
        gpu_source="reports/latency/gpu.json",
        non_scoring_sha="sha_ns",
        cpu_sha="sha_cpu",
        gpu_sha="sha_gpu",
    )

    assert result["triton_device"]["value"] == "gpu"
    assert result["margin_ms"]["value"] == 2.0
    assert result["decision_timeout_ms"]["value"] == 87.0
    assert result["overload_max_in_flight"]["value"] == 16


def test_derive_serving_limits_stop_condition_timeout() -> None:
    """
    WHAT: Tests stop condition when decision_timeout_ms < chosen Triton median p99 at in-flight 1.

    WHY: Project rules dictate hard stop if the non-scoring stages leave insufficient time
    even for a single in-flight model evaluation.
    """
    # Non-scoring takes 95 ms, spread = 4 ms -> timeout = 100 - 95 - 4 = 1.0 ms.
    # GPU in-flight 1 p99 is 2.0 ms (> 1.0 ms).
    non_scoring, cpu_rep, gpu_rep = _create_sample_reports(
        decision_p99s=[93.0, 95.0, 97.0],
        decision_median=95.0,
        cpu_median=5.0,
        gpu_median=2.0,
        gpu_sweep=[{"in_flight": 1, "p99": 2.0}],
    )

    with pytest.raises(ValueError, match="STOP CONDITION: derived decision_timeout_ms .* is less than chosen gpu"):
        derive_serving_limits(
            non_scoring_data=non_scoring,
            cpu_data=cpu_rep,
            gpu_data=gpu_rep,
            budget_max_ms=100.0,
            non_scoring_source="reports/latency/non_scoring.json",
            cpu_source="reports/latency/cpu.json",
            gpu_source="reports/latency/gpu.json",
            non_scoring_sha="sha_ns",
            cpu_sha="sha_cpu",
            gpu_sha="sha_gpu",
        )


def test_derive_serving_limits_stop_condition_no_level_fits() -> None:
    """
    WHAT: Tests stop condition when no swept in-flight level fits within decision_timeout_ms.

    WHY: Guarantees that the system fails loudly rather than selecting an unverified concurrency level.
    """
    # Timeout is 20.0 ms. Chosen device (CPU) has in-flight 1 at 25.0 ms, so even level 1 doesn't fit.
    # Note: GPU is 30.0 ms, CPU is 25.0 ms.
    # In-flight 1 median is 25.0 ms, but suppose in-flight 1 median was checked
    # before sweep or sweep entries start higher:
    non_scoring, cpu_rep, gpu_rep = _create_sample_reports(
        decision_p99s=[75.0, 80.0, 85.0],
        decision_median=80.0,
        cpu_median=5.0,  # in-flight 1 median is 5.0, so timeout (100 - 80 - 10 = 10.0) >= 5.0 passes timeout stop check
        gpu_median=10.0,
        gpu_sweep=[],
    )
    # But CPU sweep levels all exceed 10.0 ms:
    cpu_rep["sweep"] = [
        {"in_flight": 1, "p99": 12.0},
        {"in_flight": 2, "p99": 15.0},
    ]

    with pytest.raises(ValueError, match="STOP CONDITION: no swept in-flight level on cpu fits"):
        derive_serving_limits(
            non_scoring_data=non_scoring,
            cpu_data=cpu_rep,
            gpu_data=gpu_rep,
            budget_max_ms=100.0,
            non_scoring_source="reports/latency/non_scoring.json",
            cpu_source="reports/latency/cpu.json",
            gpu_source="reports/latency/gpu.json",
            non_scoring_sha="sha_ns",
            cpu_sha="sha_cpu",
            gpu_sha="sha_gpu",
        )
