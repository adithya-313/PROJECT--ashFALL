# File: tests/unit/test_decision_latency.py
# Purpose: Unit tests verifying percentile and histogram arithmetic on fixed numerical sequences.
# Issue: Level 15c build round (D1: decision latency against evaluation budget).
"""
WHAT: Unit tests validating the mathematical precision, percentile interpolation, histogram binning,
and budget configuration loader logic in benchmarks/decision_latency.py.

WHY: Guarantees that latency metrics evaluated against the evaluation budget in config/evaluation.yaml
are computed accurately, without floating point drift or off-by-one errors on boundary values.
"""

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Any

import orjson
import pytest

import api.v1.bank_service as bank_service
from benchmarks import decision_latency
from benchmarks.decision_latency import (
    DecisionSample,
    MethodResult,
    PartsRecorder,
    build_histogram,
    collect_conditions,
    compute_percentiles,
    format_method_report,
    load_latency_budget,
    load_latency_method,
    median_of_run_values,
    method_result_to_dict,
    run_method,
    run_method_in_container,
    summarise_run,
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


# --- per-part timing, runs, medians, report and the container launcher ---


def _sample(total: float, registry: float, features: float, log_write: float) -> DecisionSample:
    return DecisionSample(
        total_ms=total, registry_wait_ms=registry, feature_compute_ms=features, feature_log_write_ms=log_write
    )


def test_decision_sample_splits_the_total_into_the_four_parts() -> None:
    """overhead = total - registry - features; other = overhead - the feature_log write."""
    sample = _sample(100.0, 30.0, 40.0, 6.0)
    assert sample.overhead_ms == 30.0
    assert sample.other_ms == 24.0


def test_parts_recorder_times_the_functions_record_decision_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each wrapped function adds its own time to its own part, and is called exactly as before."""
    calls: list[str] = []

    def prepare() -> str:
        calls.append("prepare")
        time.sleep(0.01)
        return "prepared"

    def compute() -> str:
        calls.append("compute")
        time.sleep(0.02)
        return "row"

    async def prefetch() -> str:
        calls.append("prefetch")
        await asyncio.sleep(0.03)
        return "answers"

    async def insert() -> str:
        calls.append("insert")
        await asyncio.sleep(0.005)
        return "inserted"

    monkeypatch.setattr(bank_service, "prepare_registry_questions", prepare)
    monkeypatch.setattr(bank_service, "compute_decision_row", compute)
    monkeypatch.setattr(bank_service, "prefetch_registry_answers", prefetch)
    monkeypatch.setattr(bank_service, "insert_feature_log_row", insert)

    recorder = PartsRecorder()

    async def drive() -> list[str]:
        with recorder.patched():
            return [
                bank_service.prepare_registry_questions(),
                await bank_service.prefetch_registry_answers(),
                bank_service.compute_decision_row(),
                await bank_service.insert_feature_log_row(),
            ]

    assert asyncio.run(drive()) == ["prepared", "answers", "row", "inserted"]
    assert calls == ["prepare", "prefetch", "compute", "insert"]
    assert recorder.feature_compute_s >= 0.028  # prepare + compute
    assert recorder.registry_wait_s >= 0.028
    assert recorder.feature_log_write_s >= 0.004
    sample = recorder.sample(0.1)
    assert sample.total_ms == pytest.approx(100.0)
    recorder.reset()
    assert recorder.feature_compute_s == recorder.registry_wait_s == recorder.feature_log_write_s == 0.0
    assert bank_service.compute_decision_row is compute, "the real function must be back after the block"


def test_summarise_run_reports_percentiles_per_part_and_the_slowest_calls() -> None:
    samples = [_sample(float(i), 1.0, float(i) / 2, 0.5) for i in range(1, 101)]
    run = summarise_run(1, 100, samples, registry_timeouts=3, slowest_count=10)
    assert run.timed_count == 100 and run.registry_timeouts == 3
    assert run.decision["p99"] == pytest.approx(99.01)
    assert run.feature_compute["p99"] == pytest.approx(49.505)
    assert [s.total_ms for s in run.slowest] == [float(i) for i in range(100, 90, -1)]


def _result(decision_p99s: list[float], feature_p99s: list[float], timeouts: list[int]) -> MethodResult:
    runs = []
    for index, (decision, feature, timed_out) in enumerate(zip(decision_p99s, feature_p99s, timeouts), start=1):
        samples = [_sample(decision, 5.0, feature, 1.0) for _ in range(10)]
        runs.append(summarise_run(index, 2, samples, timed_out, slowest_count=3))
    return MethodResult(
        mode="http",
        delay="default",
        runtime="container",
        runs=runs,
        questions_per_decision=3,
        budget=load_latency_budget(),
        method=load_latency_method(),
        conditions={"runtime": "container", "platform": "test-platform"},
    )


def test_verdict_uses_the_median_of_the_three_run_p99_values() -> None:
    """One noisy run (200 ms, 90 ms) does not fail the verdict; two do."""
    passing = _result([80.0, 200.0, 85.0], [30.0, 90.0, 35.0], [0, 0, 0])
    assert passing.median_p99("decision") == 85.0
    assert passing.median_p99("feature_compute") == 35.0
    assert passing.decision_passes and passing.feature_passes
    failing = _result([80.0, 200.0, 190.0], [30.0, 90.0, 80.0], [0, 0, 0])
    assert not failing.decision_passes and not failing.feature_passes


def test_timeout_rate_is_timeouts_over_questions_asked() -> None:
    result = _result([50.0, 50.0, 50.0], [20.0, 20.0, 20.0], [3, 0, 12])
    # 3 runs x 10 timed decisions x 3 questions = 90 questions; 15 timed out.
    assert result.timeout_rate == pytest.approx(15 / 90)


def test_method_report_shows_every_run_the_median_the_slowest_calls_and_the_formula() -> None:
    result = _result([80.0, 200.0, 85.0], [30.0, 90.0, 35.0], [0, 1, 0])
    text = format_method_report(result)
    for expected in ("run 1", "run 2", "run 3", "median", "spread", "registry wait", "slowest", "timeout formula"):
        assert expected in text, expected
    as_json = method_result_to_dict(result)
    assert as_json["medians_of_run_p99_ms"]["decision"] == 85.0
    assert as_json["run_p99_ms"]["decision"] == [80.0, 200.0, 85.0]
    assert len(as_json["runs"]) == 3 and len(as_json["runs"][0]["slowest"]) == 3


def test_conditions_record_platform_python_commit_clock_and_where_it_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASHFALL_BENCH_RUNTIME", "container")
    monkeypatch.setenv("BENCH_COMMIT", "abc1234")
    conditions = collect_conditions()
    assert conditions["runtime"] == "container"
    assert conditions["commit"] == "abc1234"
    assert "monotonic_clock" in conditions and "python" in conditions and "platform" in conditions
    assert "not detectable" in conditions["power"]
    monkeypatch.delenv("ASHFALL_BENCH_RUNTIME")
    assert collect_conditions()["runtime"] == "native"


def test_run_method_refuses_an_impossible_mode_or_delay() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        asyncio.run(run_method(fixture=None, mode="grpc", delay="off", registry_url=None))
    with pytest.raises(ValueError, match="delay must be one of"):
        asyncio.run(run_method(fixture=None, mode="http", delay="slowish", registry_url=None))
    with pytest.raises(ValueError, match="needs mode 'http'"):
        asyncio.run(run_method(fixture=None, mode="fake", delay="default", registry_url=None))


def test_container_launcher_starts_the_registry_with_the_delay_then_runs_the_bench_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """http: recreate nationalsim-bench with the delay in its environment, then run `bench` against it."""
    monkeypatch.setattr(decision_latency, "OUTPUT_REPORT_PATH", tmp_path / "level_15_decision_latency.json")
    issued: list[tuple[list[str], str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "abc1234", "")
        issued.append((command, kwargs["env"]["NATIONALSIM_SCREEN_DELAY"]))
        if "bench" in command and "run" in command:
            (tmp_path / "container_http_registry-slow.json").write_bytes(orjson.dumps({"ok": True}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(decision_latency.subprocess, "run", fake_run)
    result = run_method_in_container("http", "registry-slow", runs=1)
    assert result == {"ok": True}

    up, run = issued[-2], issued[-1]
    assert up[0][-5:] == ["up", "-d", "--wait", "--force-recreate", "nationalsim-bench"] and up[1] == "registry-slow"
    assert run[0][:5] == ["docker", "compose", "--profile", "benchmark", "run"]
    assert "--registry-url" in run[0] and "http://nationalsim-bench:8080" in run[0]
    assert run[0][run[0].index("--registry-delay") + 1] == "registry-slow"
    assert run[0][run[0].index("--runs") + 1] == "1"
    assert "/out/container_http_registry-slow.json" in run[0]


def test_container_launcher_does_not_start_a_registry_for_the_fake_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(decision_latency, "OUTPUT_REPORT_PATH", tmp_path / "level_15_decision_latency.json")
    issued: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "abc1234", "")
        issued.append(command)
        (tmp_path / "container_fake_off.json").write_bytes(orjson.dumps({"ok": True}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(decision_latency.subprocess, "run", fake_run)
    run_method_in_container("fake", "off")
    assert not any(command[-1] == "nationalsim-bench" for command in issued)
    with pytest.raises(ValueError, match="delay must be one of"):
        run_method_in_container("http", "sometimes")
