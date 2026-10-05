# File: benchmarks/benchmark_batch_sweep.py
# Purpose: Level 6 batch-size sweep. Measures how Triton's throughput and latency change as one request carries more
#          rows (batch sizes 1, 8, 32 and 128), using the SAME client code as production
#          (serving/fraud_client.py: tritonclient.grpc + set_data_from_numpy). Level 9 reuses this script.
#
# How the measurement is kept fair:
#   - ONE request is in flight at a time, so the batch size is the only thing that changes between rows of the table.
#   - Every batch size gets an untimed warm-up first (connection open, caches filled), then the timed requests.
#   - The input is deterministic: a fixed random seed, rows of 15 finite float32 numbers.
#   - The whole sweep is repeated (default 3 times) because a laptop is noisy. The headline numbers are the MEDIAN
#     across repeats; every individual repeat is saved too, so you can see how much they moved.
#
# Run from the repo root (Triton must be up: docker compose --profile serving up -d triton):
#     PYTHONPATH=. python benchmarks/benchmark_batch_sweep.py
import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpc_client

from serving import fraud_client

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "benchmarks" / "level6_batch.json"
SECTION_NAME = "triton_batch_sweep"
DEFAULT_BATCH_SIZES = [1, 8, 32, 128]
DEFAULT_WARMUP = 100  # untimed requests per batch size
DEFAULT_REQUESTS = 1000  # timed requests per batch size
DEFAULT_REPEATS = 3
DEFAULT_SEED = 42


def make_batches(batch_size: int, count: int, seed: int) -> list[np.ndarray]:
    """Make `count` different batches of random, finite float32 rows, the same every time for the same seed."""
    rng = np.random.default_rng(seed + batch_size)
    return [rng.random((batch_size, len(fraud_client.FEATURE_ORDER)), dtype=np.float32) for _ in range(count)]


def send_one_request(client, batch: np.ndarray) -> None:
    """Send one batch through the production client code and wait for the answer (validation, packing, gRPC, parse)."""
    fraud_client.raw_fraud_probability(client, batch)


def measure_batch_size(client, batch_size: int, warmup: int, requests: int, seed: int) -> dict:
    """Warm up, then time `requests` single in-flight requests of `batch_size` rows each."""
    batches = make_batches(batch_size, warmup + requests, seed)
    for batch in batches[:warmup]:
        send_one_request(client, batch)

    latencies_ms = []
    wall_start = time.perf_counter()
    for batch in batches[warmup:]:
        start = time.perf_counter()
        send_one_request(client, batch)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    wall_seconds = time.perf_counter() - wall_start

    rows_total = batch_size * requests
    return {
        "batch_size": batch_size,
        "requests_timed": requests,
        "rows_total": rows_total,
        "wall_time_s": round(wall_seconds, 4),
        "throughput_infers_per_s": round(rows_total / wall_seconds, 1),
        "latency_ms": {
            "p50": round(float(np.percentile(latencies_ms, 50)), 3),
            "p99": round(float(np.percentile(latencies_ms, 99)), 3),
            "mean": round(float(np.mean(latencies_ms)), 3),
        },
    }


def combine_repeats(repeats: list[list[dict]]) -> list[dict]:
    """Headline numbers per batch size: the median of each metric across the repeated sweeps."""
    combined = []
    for index in range(len(repeats[0])):
        runs = [repeat[index] for repeat in repeats]
        first = runs[0]
        combined.append({
            "batch_size": first["batch_size"],
            "requests_timed": first["requests_timed"],
            "rows_total": first["rows_total"],
            "wall_time_s": round(statistics.median(r["wall_time_s"] for r in runs), 4),
            "throughput_infers_per_s": round(statistics.median(r["throughput_infers_per_s"] for r in runs), 1),
            "latency_ms": {key: round(statistics.median(r["latency_ms"][key] for r in runs), 3)
                           for key in ("p50", "p99", "mean")},
            "p99_ms_each_repeat": [r["latency_ms"]["p99"] for r in runs],
            "throughput_each_repeat": [r["throughput_infers_per_s"] for r in runs],
        })
    return combined


def describe_environment(client, url: str) -> dict:
    """Record what the numbers were measured on, so nobody reads them as a different setup."""
    config = client.get_model_config(fraud_client.MODEL_NAME).config
    try:
        containers = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True,
                                    timeout=20).stdout.split()
    except (OSError, subprocess.SubprocessError):
        containers = ["could not run docker ps"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "triton_server_version": client.get_server_metadata().version,
        "model_backend": config.backend,
        "model_runs_on": "CPU (instance_group kind: KIND_CPU), so this sweep is CPU-bound, not GPU",
        "max_batch_size_in_config": config.max_batch_size,
        "dynamic_batching_in_config": config.HasField("dynamic_batching"),
        "client": f"host python {platform.python_version()}, tritonclient {tritonclient_version()}, url {url} "
                  "(Docker Desktop port forward, not a container on the compose network)",
        "other_containers_running": containers,
    }


def tritonclient_version() -> str:
    """tritonclient has no __version__ attribute, so read the installed package metadata instead."""
    from importlib.metadata import version
    return version("tritonclient")


def print_table(rows: list[dict]) -> None:
    """Print the 4-row summary table."""
    print(f"{'batch_size':>10} | {'requests':>8} | {'throughput_infers_per_s':>23} | {'p50_ms':>8} | {'p99_ms':>8}")
    for row in rows:
        print(f"{row['batch_size']:>10} | {row['requests_timed']:>8} | {row['throughput_infers_per_s']:>23,.1f} | "
              f"{row['latency_ms']['p50']:>8.3f} | {row['latency_ms']['p99']:>8.3f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton batch-size sweep (single client, one request in flight)")
    parser.add_argument("--url", default=fraud_client.TRITON_URL)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    client = grpc_client.InferenceServerClient(url=args.url)
    if not client.is_model_ready(fraud_client.MODEL_NAME):
        print(f"Model {fraud_client.MODEL_NAME} is not READY on {args.url}; start Triton first.")
        return 1
    print(f"Triton {client.get_server_metadata().version}: model {fraud_client.MODEL_NAME} is READY on {args.url}")

    repeats = []
    for repeat in range(args.repeats):
        sweep = []
        for batch_size in args.batch_sizes:
            sweep.append(measure_batch_size(client, batch_size, args.warmup, args.requests, args.seed))
        repeats.append(sweep)
        print(f"\nRepeat {repeat + 1} of {args.repeats}:")
        print_table(sweep)

    headline = combine_repeats(repeats)
    print("\nHeadline (median across repeats):")
    print_table(headline)

    section = {
        "what": "Triton fraud_xgboost over gRPC: throughput and latency per request at different batch sizes, "
                "one request in flight, production client path (serving/fraud_client.py)",
        "method": {"warmup_requests_per_size": args.warmup, "timed_requests_per_size": args.requests,
                   "repeats": args.repeats, "input_seed": args.seed,
                   "input": "random float32 rows in [0, 1), 15 features, all finite; a different batch each request",
                   "headline_numbers": "median across repeats, taken separately for each metric",
                   "throughput_definition": "total rows sent / total timed wall seconds"},
        "environment": describe_environment(client, args.url),
        "results": headline,
        "all_repeats": repeats,
    }
    report = json.loads(args.output_json.read_text()) if args.output_json.exists() else {}
    report[SECTION_NAME] = section
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote the '{SECTION_NAME}' section to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
