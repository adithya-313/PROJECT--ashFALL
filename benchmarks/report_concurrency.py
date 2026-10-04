# File: benchmarks/report_concurrency.py
# Purpose: Turns the saved concurrency benchmark JSON (and an optional container CPU monitor log) into
#          reports/benchmarks/level5_grpc_vs_http_concurrency.md and logs the numbers to MLflow, so no figure is typed
#          by hand. Also records the serving-transport decision.
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.mlflow_setup import setup_mlflow  # noqa: E402

REPORT_PATH = REPO_ROOT / "reports" / "benchmarks" / "level5_grpc_vs_http_concurrency.md"
EXPERIMENT_NAME = "level5_fraud_serving"
TRANSPORTS = ["grpc", "http"]
METRICS = ["p50_ms", "p95_ms", "p99_ms", "mean_ms", "throughput_rps"]
LOWER_IS_BETTER = {"p50_ms", "p95_ms", "p99_ms", "mean_ms"}


def stat_values(data: dict, level: int, transport: str, metric: str) -> list[float]:
    """The value of one metric in every repeated run."""
    return [run["levels"][str(level)][transport][metric] for run in data["runs"]]


def median_of(data: dict, level: int, transport: str, metric: str) -> float:
    """Median of one metric across the repeated runs."""
    return statistics.median(stat_values(data, level, transport, metric))


def grpc_better_in_runs(data: dict, level: int, metric: str) -> int:
    """In how many runs gRPC was better than HTTP on this metric (lower latency or higher throughput)."""
    grpc_values = stat_values(data, level, "grpc", metric)
    http_values = stat_values(data, level, "http", metric)
    if metric in LOWER_IS_BETTER:
        return sum(g < h for g, h in zip(grpc_values, http_values))
    return sum(g > h for g, h in zip(grpc_values, http_values))


def level_table(data: dict) -> list[str]:
    """Markdown table of the median across runs for every (concurrency, transport)."""
    lines = ["| Concurrency | Transport | p50 ms | p95 ms | p99 ms | mean ms | Throughput req/s | Errors |",
             "|---|---|---|---|---|---|---|---|"]
    for level in data["levels"]:
        for transport in TRANSPORTS:
            medians = [median_of(data, level, transport, metric) for metric in METRICS]
            errors = sum(run["levels"][str(level)][transport]["errors"] for run in data["runs"])
            lines.append(f"| {level} | {transport} | {medians[0]:.3f} | {medians[1]:.3f} | {medians[2]:.3f} | "
                         f"{medians[3]:.3f} | {medians[4]:,.0f} | {errors} |")
    return lines


def direction_table(data: dict) -> list[str]:
    """For each level: in how many runs gRPC beat HTTP on each metric (so the direction can be judged for noise)."""
    runs = len(data["runs"])
    lines = [f"Cells show how many of the {runs} runs gRPC was better (lower latency / higher throughput) than HTTP.",
             "", "| Concurrency | p50 | p95 | p99 | mean | Throughput |", "|---|---|---|---|---|---|"]
    for level in data["levels"]:
        cells = " | ".join(f"{grpc_better_in_runs(data, level, metric)} / {runs}" for metric in METRICS)
        lines.append(f"| {level} | {cells} |")
    return lines


def crossover_text(data: dict) -> list[str]:
    """Plain statements about where gRPC is ahead, generated from the data."""
    runs = len(data["runs"])
    levels = data["levels"]
    ahead = [lv for lv in levels if grpc_better_in_runs(data, lv, "throughput_rps") == runs]
    behind = [lv for lv in levels if grpc_better_in_runs(data, lv, "throughput_rps") == 0]
    tail_ahead = [lv for lv in levels if grpc_better_in_runs(data, lv, "p99_ms") == runs]
    p50_http = [lv for lv in levels if grpc_better_in_runs(data, lv, "p50_ms") == 0]
    lines = []
    if ahead:
        lines.append(f"- gRPC had higher throughput than HTTP in every run at concurrency {ahead}.")
    if behind:
        lines.append(f"- HTTP had higher throughput than gRPC in every run at concurrency {behind}.")
    mixed = [lv for lv in levels if lv not in ahead and lv not in behind]
    if mixed:
        lines.append(f"- Throughput direction was mixed between runs (within noise) at concurrency {mixed}.")
    if tail_ahead:
        lines.append(f"- gRPC had the lower p99 latency in every run at concurrency {tail_ahead}.")
    if p50_http:
        lines.append(f"- HTTP had the lower median (p50) latency in every run at concurrency {p50_http}.")
    return lines


def saturation_text(data: dict) -> list[str]:
    """Check whether throughput stopped growing between the two highest levels (a sign of resource limits)."""
    levels = data["levels"]
    low, high = levels[-2], levels[-1]
    lines = []
    for transport in TRANSPORTS:
        t_low, t_high = (median_of(data, lv, transport, "throughput_rps") for lv in (low, high))
        p_low, p_high = (median_of(data, lv, transport, "p50_ms") for lv in (low, high))
        lines.append(f"- {transport}: from concurrency {low} to {high}, median throughput changed "
                     f"{t_high / t_low:.2f}x "
                     f"({t_low:,.0f} to {t_high:,.0f} req/s) while median p50 latency changed {p_high / p_low:.2f}x "
                     f"({p_low:.1f} to {p_high:.1f} ms).")
    return lines


def parse_monitor(path: Path) -> dict:
    """Summarise the container CPU samples (docker stats) taken while the benchmark ran."""
    samples: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        match = re.match(r"\S+ (\S+) cpu=([\d.]+)%", line.strip())
        if match:
            samples.setdefault(match.group(1), []).append(float(match.group(2)))
    return {name: {"n": len(v), "median": statistics.median(v), "max": max(v)} for name, v in samples.items()}


def build_report(data: dict, monitor: dict | None) -> str:
    """Assemble the markdown report."""
    runs = len(data["runs"])
    levels = data["levels"]
    process_text = ", ".join(str(lv) + ": " + str(data["processes_for_level"][str(lv)]) for lv in levels)
    request_text = ", ".join(str(lv) + ": " + format(data["requests_for_level"][str(lv)], ",") for lv in levels)
    lines = [
        "# Level 5 - Triton gRPC vs HTTP under concurrency (micro-benchmark)",
        "",
        "Companion to `level5_grpc_vs_http.md`, which measured one request at a time. This measures the same model, "
        "with the same single-row inputs, over Triton's gRPC and HTTP endpoints while several requests are in flight "
        "at once. It is a TRANSPORT micro-benchmark on one shared laptop. It is not the production load test "
        "(Level 6/10, around 500 users) and its numbers must not be read as production capacity.",
        "",
        "## Method",
        "",
        "- Client: asyncio clients (`tritonclient.grpc.aio` and `tritonclient.http.aio`), run from one throwaway "
        "container on the compose network and talking to `triton:8001` and `triton:8000` directly, so both "
        "transports share one network path.",
        "- Concurrency = the number of requests in flight at once (a closed loop: each finished request is replaced by "
        "the next). To keep the Python client from becoming the bottleneck, the requests are spread over several "
        f"client processes (client processes per concurrency level: {process_text}).",
        f"- Timed requests per transport per level: {request_text}; "
        f"{data['warmup_total']} untimed warm-up requests first. Inputs: the same 1,000 real test-split rows "
        "(50 fraud, 950 non-fraud), one row per request.",
        f"- Each level was run {runs} times, alternating which transport ran first. Tables show the MEDIAN across the "
        "runs. A shared laptop is noisy, so look at the direction across runs, not at one figure.",
        f"- Both transports returned identical probabilities (largest difference {data['answer_gap']:.1e}). "
        "Failed requests are counted in the Errors column.",
        "- Latency is measured per request from sending to receiving the answer, so at high concurrency it includes "
        "time spent queueing. Throughput = completed requests / elapsed seconds.",
        "",
        "## Results (median across runs)",
        "",
        *level_table(data),
        "",
        "## Which transport was ahead, run by run",
        "",
        *direction_table(data),
        "",
        "## Crossover finding",
        "",
        *crossover_text(data),
        "",
        "## Saturation: where the machine, not the transport, sets the limit",
        "",
        *saturation_text(data),
    ]
    if monitor:
        triton = monitor.get("project-ashfall-triton-1")
        client = monitor.get("bench-concurrency")
        if triton and client:
            lines += [
                f"- Container CPU sampled during the whole run (100% = one core; Docker's VM has 12): Triton median "
                f"{triton['median']:.0f}%, peak {triton['max']:.0f}%; benchmark client container median "
                f"{client['median']:.0f}%, peak {client['max']:.0f}%.",
            ]
    lines += [
        "",
        "If throughput stops growing while latency keeps rising (Little's law: latency is roughly concurrency divided "
        "by throughput), the system has reached its service rate. Here Triton's container was using essentially all "
        "of the VM's CPU cores at the higher levels, and the client shares those same cores. So the highest levels "
        "measure the saturation of this machine and of this Triton configuration (one CPU model instance, no dynamic "
        "batching, not tuned), not a clean transport difference. 100 is therefore the ceiling for this benchmark; "
        "going higher would only measure the machine choking.",
        "",
        "## Serving transport decision",
        "",
        "**gRPC is the single serving path.** There is no HTTP/gRPC switch.",
        "",
        "- At one request at a time the two transports are within a few hundredths of a millisecond of each other, and "
        "which one is faster depends on the client library: with the synchronous clients (`level5_grpc_vs_http.md`) "
        "HTTP's median was about 0.02 ms lower; with the asyncio clients used here, gRPC's p50 was lower at "
        "concurrency 1. Either way the difference is negligible in practice.",
        "- Under concurrency gRPC is ahead where it matters for a hosted service: see the crossover finding and the "
        "run-by-run table above.",
        "- An adaptive HTTP/gRPC switch was considered and rejected: it adds complexity for no real payoff. The hard "
        "part would be measuring live load to decide when to switch, and HTTP's advantage exists only in a "
        "single-request scenario that a production service does not see.",
        "- The real production load test is Level 6/10. This document is a transport micro-benchmark only.",
        "",
    ]
    return "\n".join(lines)


def log_to_mlflow(data: dict, monitor: dict | None) -> str:
    """Log parameters, per-level medians and the report to MLflow; return the run ID."""
    first_level = str(data["levels"][0])
    with mlflow.start_run(run_name="grpc_vs_http_concurrency") as run:
        mlflow.log_params({
            "concurrency_levels": ",".join(str(lv) for lv in data["levels"]), "repeats": len(data["runs"]),
            "requests_per_transport_per_level": json.dumps(data["requests_for_level"]),
            "client_processes_per_level": json.dumps(data["processes_for_level"]),
            "warmup_total": data["warmup_total"], "grpc_endpoint": data["grpc_url"],
            "http_endpoint": data["http_url"], "client": "tritonclient asyncio (grpc.aio, http.aio)",
            "model": "fraud_xgboost", "first_level": first_level,
        })
        for level in data["levels"]:
            for transport in TRANSPORTS:
                for metric in METRICS:
                    value = median_of(data, level, transport, metric)
                    mlflow.log_metric(f"c{level}_{transport}_{metric}_median", value)
        if monitor:
            for name, stats in monitor.items():
                mlflow.log_metric(f"cpu_peak_pct_{name}", stats["max"])
                mlflow.log_metric(f"cpu_median_pct_{name}", stats["median"])
        mlflow.log_artifact(str(REPORT_PATH))
        return run.info.run_id


def main() -> int:
    """Build the report from the saved JSON and log it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--monitor-txt", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.json.read_text())
    monitor = parse_monitor(args.monitor_txt) if args.monitor_txt else None
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(data, monitor))
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT).as_posix()}")
    setup_mlflow(EXPERIMENT_NAME)
    print(f"MLflow experiment '{EXPERIMENT_NAME}', run ID {log_to_mlflow(data, monitor)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
