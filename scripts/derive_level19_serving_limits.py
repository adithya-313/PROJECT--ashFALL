"""
WHAT: Derives Level 19 production serving limits and timeout configuration from latency benchmark reports.
Reads the three measured latency reports (non-scoring stages, Triton CPU round-trip, and Triton GPU round-trip)
and config/evaluation.yaml, computes device selection, safety margin, decision timeout, and maximum in-flight
overload limits, and writes config/serving_limits.yaml.

WHY: Production serving needs mathematically grounded timeout and concurrency boundaries calibrated to the real
hardware measurements of this host, ensuring requests never exceed the end-to-end latency budget (100 ms).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def compute_file_sha256(path: Path) -> str:
    """
    PURPOSE:
        Computes the SHA-256 hexadecimal digest of a file on disk.

    INPUTS:
        path: Path to the target file.

    RETURNS:
        str: 64-character hexadecimal SHA-256 string.

    ERRORS:
        Raises FileNotFoundError if the file does not exist.
    """
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def load_max_latency_budget(evaluation_config_path: Path) -> float:
    """
    PURPOSE:
        Reads the overall latency budget max_ms from config/evaluation.yaml.

    INPUTS:
        evaluation_config_path: Path to config/evaluation.yaml.

    RETURNS:
        float: Maximum allowed latency budget in milliseconds (e.g. 100.0).

    ERRORS:
        Raises FileNotFoundError if evaluation config does not exist.
        Raises ValueError if latency or max_ms section is missing.
    """
    if not evaluation_config_path.is_file():
        raise FileNotFoundError(f"Evaluation configuration not found: {evaluation_config_path}")

    with open(evaluation_config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "latency" not in data:
        raise ValueError(f"Missing 'latency' block in {evaluation_config_path}")

    latency_block = data["latency"]
    if not isinstance(latency_block, dict) or "max_ms" not in latency_block:
        raise ValueError(f"Missing 'max_ms' under latency block in {evaluation_config_path}")

    return float(latency_block["max_ms"])


def derive_serving_limits(
    non_scoring_data: dict[str, Any],
    cpu_data: dict[str, Any],
    gpu_data: dict[str, Any],
    budget_max_ms: float,
    non_scoring_source: str,
    cpu_source: str,
    gpu_source: str,
    non_scoring_sha: str,
    cpu_sha: str,
    gpu_sha: str,
) -> dict[str, Any]:
    """
    PURPOSE:
        Calculates device selection, margin, decision timeout, and overload concurrency limits
        from benchmark report dictionaries and evaluates project STOP conditions.

    INPUTS:
        non_scoring_data: Parsed JSON content of non-scoring stages latency report.
        cpu_data: Parsed JSON content of Triton CPU benchmark report.
        gpu_data: Parsed JSON content of Triton GPU benchmark report.
        budget_max_ms: Overall end-to-end latency budget (100 ms).
        non_scoring_source: Relative path string to non-scoring report.
        cpu_source: Relative path string to Triton CPU report.
        gpu_source: Relative path string to Triton GPU report.
        non_scoring_sha: SHA-256 digest of non-scoring report.
        cpu_sha: SHA-256 digest of Triton CPU report.
        gpu_sha: SHA-256 digest of Triton GPU report.

    RETURNS:
        dict[str, Any]: Dictionary formatted for config/serving_limits.yaml with value, label,
        and derived_from provenance for each entry.

    ERRORS:
        Raises ValueError if any stop condition is triggered:
            - decision_timeout_ms < chosen-device Triton median_of_run_p99 at in-flight 1.
            - No swept in-flight concurrency level has p99 <= decision_timeout_ms.
    """
    # 1. Device selection: chosen device is the one with lower median_of_run_p99 at in-flight 1.
    cpu_median_inflight_1: float = float(cpu_data["median_of_run_p99"])
    gpu_median_inflight_1: float = float(gpu_data["median_of_run_p99"])

    if gpu_median_inflight_1 < cpu_median_inflight_1:
        chosen_device: str = "gpu"
        chosen_report: dict[str, Any] = gpu_data
        chosen_source: str = gpu_source
        chosen_sha: str = gpu_sha
        chosen_inflight_1_p99: float = gpu_median_inflight_1
    else:
        chosen_device = "cpu"
        chosen_report = cpu_data
        chosen_source = cpu_source
        chosen_sha = cpu_sha
        chosen_inflight_1_p99 = cpu_median_inflight_1

    # 2. Run-to-run spread margin: (max - min) of the three per-run whole-decision p99s.
    run_p99_decision: list[float] = [float(val) for val in non_scoring_data["run_p99_ms"]["decision"]]
    if len(run_p99_decision) != 3:
        raise ValueError(
            f"Expected exactly 3 per-run decision p99 values, found {len(run_p99_decision)}"
        )
    margin_ms: float = max(run_p99_decision) - min(run_p99_decision)

    # 3. Decision timeout: budget_max_ms - non_scoring_median_p99 - margin_ms.
    non_scoring_median_p99: float = float(non_scoring_data["medians_of_run_p99_ms"]["decision"])
    decision_timeout_ms: float = budget_max_ms - non_scoring_median_p99 - margin_ms

    # STOP CONDITION 1: decision_timeout_ms < chosen-device Triton median_of_run_p99 at in-flight 1.
    if decision_timeout_ms < chosen_inflight_1_p99:
        raise ValueError(
            f"STOP CONDITION: derived decision_timeout_ms ({decision_timeout_ms:.3f} ms) is less than "
            f"chosen {chosen_device} Triton median_of_run_p99 at in-flight 1 ({chosen_inflight_1_p99:.3f} ms)."
        )

    # 4. Overload max in-flight: largest swept in-flight level whose p99 <= decision_timeout_ms.
    fitting_levels: list[int] = []
    sweep_entries = chosen_report.get("sweep", [])
    for entry in sweep_entries:
        level: int = int(entry["in_flight"])
        p99: float = float(entry["p99"])
        if p99 <= decision_timeout_ms:
            fitting_levels.append(level)

    # STOP CONDITION 2: no in-flight level fits within decision_timeout_ms.
    if not fitting_levels:
        raise ValueError(
            f"STOP CONDITION: no swept in-flight level on {chosen_device} fits within "
            f"decision_timeout_ms ({decision_timeout_ms:.3f} ms)."
        )

    overload_max_in_flight: int = max(fitting_levels)

    # 5. Build configuration dictionary with full mathematical and provenance tracking.
    device_formula: str = (
        f"lower median_of_run_p99 at in-flight 1: {chosen_device} ({chosen_inflight_1_p99:.3f} ms) "
        f"vs other ({max(cpu_median_inflight_1, gpu_median_inflight_1):.3f} ms)"
    )
    margin_formula: str = (
        f"run-to-run spread (max - min) of 3 whole-decision p99s: "
        f"{max(run_p99_decision):.3f} - {min(run_p99_decision):.3f} = {margin_ms:.3f} ms"
    )
    timeout_formula: str = (
        f"budget max_ms ({budget_max_ms:.1f}) - non_scoring_median_p99 ({non_scoring_median_p99:.3f}) "
        f"- margin_ms ({margin_ms:.3f}) = {decision_timeout_ms:.3f} ms"
    )
    overload_formula: str = (
        f"largest swept in-flight level on {chosen_device} with p99 <= decision_timeout_ms "
        f"({decision_timeout_ms:.3f} ms)"
    )

    return {
        "triton_device": {
            "value": chosen_device,
            "label": "derived",
            "derived_from": (
                f"{device_formula}; CPU [{cpu_source} sha256:{cpu_sha}], "
                f"GPU [{gpu_source} sha256:{gpu_sha}]"
            ),
        },
        "margin_ms": {
            "value": round(margin_ms, 3),
            "label": "derived",
            "derived_from": f"{margin_formula}; source: {non_scoring_source} sha256:{non_scoring_sha}",
        },
        "decision_timeout_ms": {
            "value": round(decision_timeout_ms, 3),
            "label": "derived",
            "derived_from": (
                f"{timeout_formula}; non-scoring source: {non_scoring_source} sha256:{non_scoring_sha}, "
                f"budget: config/evaluation.yaml"
            ),
        },
        "overload_max_in_flight": {
            "value": overload_max_in_flight,
            "label": "derived",
            "derived_from": f"{overload_formula}; report source: {chosen_source} sha256:{chosen_sha}",
        },
    }


def main(argv: list[str] | None = None) -> int:
    """
    PURPOSE:
        CLI entry point for deriving serving limits from measured latency reports.
        Validates reports, checks stop conditions, and writes config/serving_limits.yaml.

    INPUTS:
        argv: Optional command-line argument list (defaults to sys.argv[1:]).

    RETURNS:
        int: 0 on successful configuration writing, 1 if stop condition or error encountered.
    """
    parser = argparse.ArgumentParser(description="Derive Level 19 serving limits.")
    parser.add_argument(
        "--non-scoring-report",
        type=Path,
        default=REPO_ROOT / "reports" / "latency" / "level_19_non_scoring_stages.json",
        help="Path to non-scoring stages latency report.",
    )
    parser.add_argument(
        "--cpu-report",
        type=Path,
        default=REPO_ROOT / "reports" / "latency" / "level_19_triton_round_trip_cpu.json",
        help="Path to Triton CPU round-trip report.",
    )
    parser.add_argument(
        "--gpu-report",
        type=Path,
        default=REPO_ROOT / "reports" / "latency" / "level_19_triton_round_trip_gpu.json",
        help="Path to Triton GPU round-trip report.",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=REPO_ROOT / "config" / "evaluation.yaml",
        help="Path to evaluation config.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=REPO_ROOT / "config" / "serving_limits.yaml",
        help="Path to output config/serving_limits.yaml.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # 1. Verify all input files exist before proceeding.
    for path_item in (args.non_scoring_report, args.cpu_report, args.gpu_report, args.evaluation_config):
        if not path_item.is_file():
            print(f"ERROR: Input report file does not exist: {path_item}", file=sys.stderr)
            return 1

    # 2. Compute cryptographic hashes for data provenance.
    non_scoring_sha: str = compute_file_sha256(args.non_scoring_report)
    cpu_sha: str = compute_file_sha256(args.cpu_report)
    gpu_sha: str = compute_file_sha256(args.gpu_report)

    # 3. Parse JSON reports and latency budget.
    with open(args.non_scoring_report, "r", encoding="utf-8") as f:
        non_scoring_data = json.load(f)
    with open(args.cpu_report, "r", encoding="utf-8") as f:
        cpu_data = json.load(f)
    with open(args.gpu_report, "r", encoding="utf-8") as f:
        gpu_data = json.load(f)

    try:
        budget_max_ms: float = load_max_latency_budget(args.evaluation_config)
    except Exception as err:
        print(f"ERROR loading budget from {args.evaluation_config}: {err}", file=sys.stderr)
        return 1

    non_scoring_source = args.non_scoring_report.relative_to(REPO_ROOT).as_posix()
    cpu_source = args.cpu_report.relative_to(REPO_ROOT).as_posix()
    gpu_source = args.gpu_report.relative_to(REPO_ROOT).as_posix()

    # 4. Perform derivation and check STOP conditions.
    try:
        serving_limits = derive_serving_limits(
            non_scoring_data=non_scoring_data,
            cpu_data=cpu_data,
            gpu_data=gpu_data,
            budget_max_ms=budget_max_ms,
            non_scoring_source=non_scoring_source,
            cpu_source=cpu_source,
            gpu_source=gpu_source,
            non_scoring_sha=non_scoring_sha,
            cpu_sha=cpu_sha,
            gpu_sha=gpu_sha,
        )
    except ValueError as stop_err:
        print(f"STOPPED: {stop_err}", file=sys.stderr)
        print("Per project rules, exiting non-zero and writing nothing to config/serving_limits.yaml.")
        return 1

    # 5. Write generated configuration if derivation was completely successful.
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(serving_limits, f, sort_keys=False)

    print(f"Successfully generated serving limits configuration: {args.output_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
