# File: benchmarks/benchmark_dynamic_batching.py
# Purpose: Level 6 load driver for tuning Triton dynamic batching. It fires many SEPARATE single-row gRPC requests
#          concurrently (the pattern dynamic batching exists for) and reports throughput and latency per concurrency
#          level. It reuses the load machinery of benchmarks/benchmark_concurrency.py (same client code, same
#          warm-up and multi-process closed loop), but measures gRPC only and writes a small JSON file per run.
#
# It is meant to run INSIDE a throwaway container on the compose network, like the Level 5 benchmarks, so the
# Docker/WSL port relay does not add noise:
#     python benchmark_dynamic_batching.py --rows-npy rows.npy --levels 10 50 --repeats 3 --output-json out.json
# benchmarks/tune_dynamic_batching.py drives it once per Triton configuration.
import argparse
import json
import time
from pathlib import Path

import numpy as np
from benchmark_concurrency import run_level


def median_of_runs(runs: list[dict]) -> dict:
    """Combine repeated measurements of the same level: the median of each number, plus every p99 for the spread."""
    keys = ("throughput_rps", "p50_ms", "p95_ms", "p99_ms", "mean_ms")
    combined = {key: float(np.median([run[key] for run in runs])) for key in keys}
    combined["errors"] = int(sum(run["errors"] for run in runs))
    combined["completed_per_repeat"] = runs[0]["completed"]
    combined["p99_ms_each_repeat"] = [round(run["p99_ms"], 3) for run in runs]
    combined["throughput_each_repeat"] = [round(run["throughput_rps"], 1) for run in runs]
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent single-row gRPC load for dynamic-batching tuning")
    parser.add_argument("--rows-npy", type=Path, required=True)
    parser.add_argument("--grpc-url", default="triton:8001")
    parser.add_argument("--levels", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = np.load(args.rows_npy).astype(np.float32)
    per_level = {}
    for level in args.levels:
        runs = []
        for _ in range(args.repeats):
            runs.append(run_level("grpc", args.grpc_url, rows, level))
            time.sleep(1.0)  # a short pause so one run's leftovers do not leak into the next
        per_level[str(level)] = median_of_runs(runs)
        stats = per_level[str(level)]
        print(f"c={level:<3} {stats['throughput_rps']:9.0f} req/s  p50 {stats['p50_ms']:7.3f}  "
              f"p99 {stats['p99_ms']:8.3f} ms  errors {stats['errors']}", flush=True)

    args.output_json.write_text(json.dumps({"levels": per_level}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
