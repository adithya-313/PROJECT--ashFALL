# File: benchmarks/log_level6_to_mlflow.py
# Purpose: Level 6 - log the benchmark results that were ALREADY measured (stored in
#          reports/benchmarks/level6_batch.json) to MLflow as ONE run in the "level6_benchmarking" experiment, then
#          write the run ID back into that JSON file so the numbers can be traced to the run.
#
# Nothing here is measured or invented: every metric is read from level6_batch.json. If a section is missing the
# script stops with an error instead of logging a placeholder.
#
# Run from the repo root (no Docker needed, MLflow uses the local sqlite file):
#     PYTHONPATH=. python benchmarks/log_level6_to_mlflow.py
import json
import subprocess
import sys
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.mlflow_setup import setup_mlflow  # noqa: E402

REPORT_PATH = REPO_ROOT / "reports" / "benchmarks" / "level6_batch.json"
EXPERIMENT_NAME = "level6_benchmarking"
RUN_NAME = "level6_batch_optimization"
DECISION_CONCURRENCY = "50"  # the concurrency level the dynamic-batching values were chosen at
CHOSEN_BATCHING_LABEL = "delay1000us_pref8-32"


def require(report: dict, key: str) -> dict:
    """Return a section of the report, or stop with a clear message if it is missing (no placeholders)."""
    if key not in report:
        raise SystemExit(f"level6_batch.json has no '{key}' section; run the benchmark that produces it first")
    return report[key]


def gpu_kernel_metrics(report: dict) -> dict:
    """Per-size metrics for the Haversine GPU kernel: speed, occupancy, memory bandwidth, cost."""
    kernel = require(report, "gpu_kernel_profiling")["results"]
    cost = {row["rows"]: row for row in require(report, "gpu_cost_per_inference")["results"]}
    metrics = {}
    for row in kernel:
        prefix = f"gpu_kernel_{row['rows']}"
        metrics[f"{prefix}_inferences_per_sec"] = row["inferences_per_sec"]  # kernel time from Nsight Systems
        metrics[f"{prefix}_kernel_time_us"] = row["kernel_time_ms"] * 1000.0
        metrics[f"{prefix}_sm_occupancy_pct"] = row["sm_occupancy_pct"]
        metrics[f"{prefix}_gddr6_bandwidth_pct"] = row["hbm_bandwidth_util_pct"]  # the plan's "HBM bandwidth"
        metrics[f"{prefix}_cost_per_inference_usd"] = cost[row["rows"]]["cost_per_inference_usd"]
        metrics[f"{prefix}_sustained_inferences_per_sec"] = cost[row["rows"]]["sustained_inferences_per_sec"]
        metrics[f"{prefix}_avg_power_w"] = cost[row["rows"]]["avg_power_w"]
    return metrics


def vectorization_metrics(report: dict) -> dict:
    """Old per-row loop versus the vectorized NumPy path, per batch size."""
    metrics = {}
    for row in require(report, "vectorization")["results"]:
        prefix = f"vectorization_{row['batch_size']}"
        metrics[f"{prefix}_old_loop_ms"] = row["old_loop_median_ms"]
        metrics[f"{prefix}_vectorized_ms"] = row["vectorized_median_ms"]
        metrics[f"{prefix}_speedup_x"] = row["speedup_x"]
    return metrics


def dynamic_batching_metrics(report: dict) -> dict:
    """Triton throughput and p99 with dynamic batching OFF and with the chosen setting ON, at every level measured."""
    tuning = require(report, "dynamic_batching_tuning")
    chosen = tuning["candidates"][CHOSEN_BATCHING_LABEL]["levels"]
    metrics = {}
    for level, off in tuning["off_baseline"].items():
        metrics[f"triton_c{level}_off_throughput_rps"] = off["throughput_rps"]
        metrics[f"triton_c{level}_off_p99_ms"] = off["p99_ms"]
        metrics[f"triton_c{level}_on_throughput_rps"] = chosen[level]["throughput_rps"]
        metrics[f"triton_c{level}_on_p99_ms"] = chosen[level]["p99_ms"]
    return metrics


def git_commit_hash() -> str:
    """The commit the code was at when the results were logged."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> int:
    report = json.loads(REPORT_PATH.read_text())
    price = require(report, "gpu_cost_per_inference")["price_per_kwh_usd"]
    metrics = {**gpu_kernel_metrics(report), **vectorization_metrics(report), **dynamic_batching_metrics(report)}

    tracking_uri = setup_mlflow(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id

        # Write the run ID into the JSON first, so the copy of the JSON attached to the run already contains it.
        report["mlflow"] = {"experiment": EXPERIMENT_NAME, "run_id": run_id, "run_name": RUN_NAME,
                            "tracking_uri": "sqlite:///mlflow.db (local, gitignored, not backed up)",
                            "logged_by": "benchmarks/log_level6_to_mlflow.py",
                            "note": "every metric in the run is read from this file, none are typed in"}
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

        mlflow.log_params({"git_commit": git_commit_hash(), "price_per_kwh_usd": price,
                           "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU", "gpu_memory_type": "GDDR6 (not HBM)",
                           "nsys_version": "2026.1.3.425", "ncu_version": "2026.3.1.0",
                           "dynamic_batching_chosen": "preferred_batch_size [8, 32], max_queue_delay 1000 us",
                           "triton_max_batch_size": 128})
        mlflow.set_tags({
            "price_per_kwh_is": "an editable ASSUMPTION, not measured",
            "gpu_metrics_describe": "the Haversine geo-velocity kernel, not Triton inference",
            "triton_model_gpu_metrics": "N/A (fraud_xgboost runs on CPU: KIND_CPU, tree ensemble)",
            "warm_cache_caveat": "nsys and sustained-loop speeds reuse the same input, so the GPU L2 cache is warm "
                                 "(ESTIMATE); fresh data per batch is slower. ncu occupancy and DRAM% are cold-cache.",
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(REPORT_PATH))

    print(f"MLflow store: {tracking_uri}")
    print(f"Logged {len(metrics)} metrics to experiment '{EXPERIMENT_NAME}', run ID {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
