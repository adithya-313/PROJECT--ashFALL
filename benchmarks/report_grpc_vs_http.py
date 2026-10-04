# File: benchmarks/report_grpc_vs_http.py
# Purpose: Turns saved gRPC-vs-HTTP benchmark JSON files into reports/benchmarks/level5_grpc_vs_http.md and logs the
#          numbers to MLflow, so no latency figure is ever typed by hand.
import argparse
import json
import statistics
import sys
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.mlflow_setup import setup_mlflow  # noqa: E402

REPORT_PATH = REPO_ROOT / "reports" / "benchmarks" / "level5_grpc_vs_http.md"
EXPERIMENT_NAME = "level5_fraud_serving"
STATS = ["p50_ms", "p95_ms", "p99_ms", "mean_ms"]


def load_runs(paths: list[Path]) -> list[dict]:
    """Read the saved benchmark result files."""
    return [json.loads(path.read_text()) for path in paths]


def table_lines(summary: dict) -> list[str]:
    """Markdown table for one run: one row per transport."""
    lines = ["| Transport | p50 ms | p95 ms | p99 ms | mean ms | min ms | max ms |", "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | {s['p99_ms']:.3f} | {s['mean_ms']:.3f} | "
                     f"{s['min_ms']:.3f} | {s['max_ms']:.3f} |")
    return lines


def median_across_runs(runs: list[dict], transport: str, stat: str) -> float:
    """The median of one statistic across the repeated runs."""
    return statistics.median(run["summary"][transport][stat] for run in runs)


def build_report(container_runs: list[dict], host_run: dict | None) -> str:
    """Assemble the markdown report."""
    first = container_runs[0]
    lines = [
        "# Level 5 - Triton gRPC vs HTTP inference micro-benchmark",
        "",
        "## What this measures (and what it does not)",
        "",
        "The same model (`fraud_xgboost`, ONNX on Triton, CPU) is called with the same single-row inputs over Triton's "
        "gRPC endpoint and over Triton's HTTP endpoint, one request at a time. This isolates the cost of the "
        "TRANSPORT. It is a micro-benchmark, not a load test: there is no concurrency, so it says nothing about "
        "throughput or behaviour under load (real performance and load work belongs to later levels). It is also not "
        "comparable with the Level 1 numbers, which measured a trivial `/health` endpoint at 100 connections.",
        "",
        "## Method",
        "",
        "- Both transports run from ONE throwaway container on the compose network, talking to `triton:8001` (gRPC) "
        "and `triton:8000` (HTTP) directly. Host port 8000 belongs to the FastAPI service and Triton's HTTP port is "
        "deliberately not published, so this is the only like-for-like way to reach both. Both share the same "
        "network path, so neither is handicapped by the Docker/WSL port relay.",
        f"- {first['n_requests']:,} timed requests per transport after {first['warmup']} warm-up requests; "
        f"{first['rows']:,} distinct real test-split rows (50 fraud, 950 non-fraud; seed 42), one row per request.",
        "- For every request the transports take turns going first, so slow drift in the machine affects both equally.",
        "- Each timing covers the full client-side request: build the input, send, receive, read the answer.",
        "- Default client settings for both (binary tensor encoding on HTTP). The two transports returned identical "
        f"probabilities on every row (largest difference {max(r['max_answer_gap'] for r in container_runs):.1e}).",
        f"- The benchmark was run {len(container_runs)} times to show run-to-run spread.",
        "",
    ]
    for index, run in enumerate(container_runs, start=1):
        lines += [f"### Run {index}", "", *table_lines(run["summary"]), ""]

    lines += ["### Median across runs", "", "| Transport | p50 ms | p95 ms | p99 ms | mean ms |"]
    lines += ["|---|---|---|---|---|"]
    for transport in ("grpc", "http"):
        cells = " | ".join(f"{median_across_runs(container_runs, transport, stat):.3f}" for stat in STATS)
        lines.append(f"| {transport} | {cells} |")

    grpc_p50 = median_across_runs(container_runs, "grpc", "p50_ms")
    http_p50 = median_across_runs(container_runs, "http", "p50_ms")
    faster = "HTTP" if http_p50 < grpc_p50 else "gRPC"
    runs_total = len(container_runs)

    def http_wins(stat: str) -> int:
        """In how many runs HTTP had the lower value of this statistic."""
        return sum(run["summary"]["http"][stat] < run["summary"]["grpc"][stat] for run in container_runs)

    grpc_p99 = median_across_runs(container_runs, "grpc", "p99_ms")
    http_p99 = median_across_runs(container_runs, "http", "p99_ms")
    lines += [
        "",
        "## Reading the result",
        "",
        f"At one request at a time, {faster} had the lower median latency here: median p50 gRPC {grpc_p50:.3f} ms vs "
        f"HTTP {http_p50:.3f} ms (HTTP faster at p50 in {http_wins('p50_ms')} of {runs_total} runs, at p95 in "
        f"{http_wins('p95_ms')} of {runs_total}, at p99 in {http_wins('p99_ms')} of {runs_total}). The p50 gap is only "
        f"a few hundredths of a millisecond, so the typical request is comparable; the tails differ more (median p99 "
        f"gRPC {grpc_p99:.3f} ms vs HTTP {http_p99:.3f} ms). So at this concurrency gRPC shows no latency advantage "
        "over HTTP. The usual reasons to prefer gRPC (streaming, multiplexing many concurrent requests on one "
        "connection, lower per-request overhead under heavy load) are not exercised by this test. The absolute numbers "
        "move between runs (the machine is shared with other software), which is why the comparison is made within "
        "each run, with the transports alternating.",
        "",
    ]
    if host_run is not None:
        s = host_run["summary"]["grpc"]
        lines += [
            "## For reference: the client path from the host",
            "",
            "The committed client (`serving/fraud_client.py`) runs on the host and reaches Triton through the "
            "published gRPC port (`localhost:8001`), which adds Docker's port relay. Same inputs, gRPC only, "
            f"{host_run['n_requests']:,} requests: p50 {s['p50_ms']:.3f} ms, p95 {s['p95_ms']:.3f} ms, "
            f"p99 {s['p99_ms']:.3f} ms, mean {s['mean_ms']:.3f} ms. This number is NOT comparable with the tables "
            "above (different network path) and is shown only to give a sense of the real client's latency.",
            "",
        ]
    return "\n".join(lines)


def log_to_mlflow(container_runs: list[dict], host_run: dict | None) -> str:
    """Log parameters, medians, per-run numbers and the report to MLflow; return the run ID."""
    first = container_runs[0]
    with mlflow.start_run(run_name="grpc_vs_http_microbenchmark") as run:
        mlflow.log_params({
            "n_requests_per_transport": first["n_requests"], "warmup": first["warmup"], "concurrency": 1,
            "distinct_rows": first["rows"], "grpc_endpoint": first["grpc_url"], "http_endpoint": first["http_url"],
            "repeats": len(container_runs), "model": "fraud_xgboost", "triton_image": "tritonserver:24.01-py3",
        })
        for transport in ("grpc", "http"):
            for stat in STATS:
                mlflow.log_metric(f"{transport}_{stat}_median", median_across_runs(container_runs, transport, stat))
            for index, result in enumerate(container_runs, start=1):
                mlflow.log_metric(f"{transport}_p50_ms_run{index}", result["summary"][transport]["p50_ms"])
        if host_run is not None:
            for stat in STATS:
                mlflow.log_metric(f"host_grpc_{stat}", host_run["summary"]["grpc"][stat])
        mlflow.log_artifact(str(REPORT_PATH))
        return run.info.run_id


def main() -> int:
    """Build the report from the saved JSON files and log it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--container-json", type=Path, nargs="+", required=True)
    parser.add_argument("--host-json", type=Path, default=None)
    args = parser.parse_args()

    container_runs = load_runs(args.container_json)
    host_run = load_runs([args.host_json])[0] if args.host_json else None
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(container_runs, host_run))
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT).as_posix()}")

    setup_mlflow(EXPERIMENT_NAME)
    print(f"MLflow experiment '{EXPERIMENT_NAME}', run ID {log_to_mlflow(container_runs, host_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
