# File: tests/unit/test_decision_latency.py
# Purpose: Unit tests verifying percentile and histogram arithmetic on fixed numerical sequences.
# Issue: Level 15c build round (D1: decision latency against evaluation budget).
"""
WHAT: Unit tests validating the mathematical precision, percentile interpolation, histogram binning,
and budget configuration loader logic in benchmarks/decision_latency.py.

WHY: Guarantees that latency metrics evaluated against the evaluation budget in config/evaluation.yaml
are computed accurately, without floating point drift or off-by-one errors on boundary values.
"""

from pathlib import Path

import pytest

from benchmarks.decision_latency import (
    build_histogram,
    compute_percentiles,
    load_latency_budget,
    load_latency_method,
    median_of_run_values,
)


def test_compute_percentiles_fixed_sequence() -> None:
    """Verifies percentile calculation on a fixed arithmetic progression 1.0 to 100.0."""
    values = [float(i) for i in range(1, 101)]
    stats = compute_percentiles(values)

    assert stats["count"] == 100.0
    assert stats["min"] == 1.0
    assert stats["max"] == 100.0
    assert abs(stats["p50"] - 50.5) < 1e-6
    assert abs(stats["p95"] - 95.05) < 1e-6
    assert abs(stats["p99"] - 99.01) < 1e-6


def test_compute_percentiles_single_element() -> None:
    """Verifies percentile behavior on a sequence containing a single measurement."""
    stats = compute_percentiles([42.5])
    assert stats["count"] == 1.0
    assert stats["min"] == 42.5
    assert stats["max"] == 42.5
    assert stats["p50"] == 42.5
    assert stats["p95"] == 42.5
    assert stats["p99"] == 42.5


def test_compute_percentiles_empty_raises() -> None:
    """Verifies that an empty sequence raises ValueError."""
    with pytest.raises(ValueError, match="Cannot compute percentiles on an empty sequence"):
        compute_percentiles([])


def test_build_histogram_fixed_bins() -> None:
    """Verifies exact histogram bin assignment across fixed bucket edges."""
    edges = (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 200.0)
    # One value per bucket
    test_values = [2.5, 7.5, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 250.0]

    counts, text = build_histogram(test_values, bucket_edges=edges)

    assert len(counts) == len(edges)  # 9 interval buckets + 1 overflow
    for label, count in counts.items():
        assert count == 1, f"Expected 1 item in bucket {label}, got {count}"

    assert "[  0.0,   5.0) ms" in text
    assert "[>=200.0   ) ms" in text
    assert "10.0%" in text


def test_build_histogram_boundaries() -> None:
    """Verifies that lower bucket bounds are inclusive and upper bounds are exclusive."""
    edges = (0.0, 10.0, 20.0)
    # 0.0 lands in [0, 10), 10.0 lands in [10, 20), 20.0 lands in overflow [>=20)
    counts, _ = build_histogram([0.0, 10.0, 20.0], bucket_edges=edges)
    assert counts["[  0.0,  10.0) ms"] == 1
    assert counts["[ 10.0,  20.0) ms"] == 1
    assert counts["[>= 20.0   ) ms"] == 1


def test_build_histogram_empty_raises() -> None:
    """Verifies building a histogram on empty values raises ValueError."""
    with pytest.raises(ValueError, match="Cannot build histogram on empty values"):
        build_histogram([])


def test_load_latency_budget_valid() -> None:
    """Verifies loading the canonical evaluation budget from config/evaluation.yaml."""
    budget = load_latency_budget()

    assert budget["endpoint"] == "POST /v1/decisions"
    assert budget["percentile"] == 99.0
    assert budget["max_ms"] == 100.0
    assert budget["feature_computation_max_ms"] == 50.0
    assert budget["measured"] == "server_side"
    assert budget["hardware"] == "author_laptop"
    assert budget["warm_up_required"] is True


def test_load_latency_budget_custom_file(tmp_path: Path) -> None:
    """Verifies loading custom evaluation budget and error handling on missing section."""
    custom_yaml = tmp_path / "custom_eval.yaml"
    custom_yaml.write_text(
        """
latency:
  endpoint: "POST /v1/decisions"
  percentile: 95
  max_ms: 80
  feature_computation_max_ms: 40
  measured: "server_side"
  hardware: "test_laptop"
  warm_up_required: false
""",
        encoding="utf-8",
    )
    b = load_latency_budget(custom_yaml)
    assert b["max_ms"] == 80.0
    assert b["percentile"] == 95.0
    assert b["feature_computation_max_ms"] == 40.0

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("something_else: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing 'latency' section"):
        load_latency_budget(bad_yaml)


def test_latency_method_comes_from_the_pre_registered_file() -> None:
    """The method (100 warm-up, 1,000 measured, 3 runs, median of run p99) is read from config/evaluation.yaml."""
    method = load_latency_method()
    assert method == {
        "warm_up_calls": 100,
        "measured_calls": 1000,
        "runs": 3,
        "judged_on": "median_of_run_p99",
        "slowest_calls_reported": 10,
    }


def test_latency_method_rejects_another_statistic(tmp_path: Path) -> None:
    config_file = tmp_path / "evaluation.yaml"
    config_file.write_text(
        "latency_method:\n  warm_up_calls: 1\n  measured_calls: 1\n  runs: 1\n  judged_on: mean\n"
        "  slowest_calls_reported: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported latency_method.judged_on"):
        load_latency_method(config_file)
    missing = tmp_path / "missing.yaml"
    missing.write_text("latency: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing 'latency_method'"):
        load_latency_method(missing)


def test_median_of_run_values_is_the_middle_p99() -> None:
    """One noisy run (the 120) does not decide the verdict."""
    assert median_of_run_values([48.0, 120.0, 51.0]) == 51.0
    assert median_of_run_values([40.0, 60.0]) == 50.0
    with pytest.raises(ValueError, match="at least one run"):
        median_of_run_values([])
