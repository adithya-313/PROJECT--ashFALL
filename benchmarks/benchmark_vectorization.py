# File: benchmarks/benchmark_vectorization.py
# Purpose: Level 6 micro-benchmark. Times the old row-by-row CPU loop against the new vectorized NumPy path inside
#          compute_geo_velocity_batch, in one process (no Triton, no network), and writes the result into the
#          "vectorization" section of reports/benchmarks/level6_batch.json (other sections are kept).
#
# Run from the repo root:  PYTHONPATH=. python benchmarks/benchmark_vectorization.py
import json
import statistics
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import features.geo_velocity as geo_velocity
from features.geo_velocity import compute_geo_velocity, compute_geo_velocity_batch

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "benchmarks" / "level6_batch.json"

# Batch sizes to time. 10,000 is the headline size (the same batch size the Level 4.5 Haversine benchmark used).
# 100 is the GPU crossover size, and 1,000 sits in between.
BATCH_SIZES = [100, 1_000, 10_000]
HEADLINE_BATCH_SIZE = 10_000
TIMED_RUNS = 20
SEED = 42


def make_rows(n: int) -> tuple[list[str], list[str | None], list[float]]:
    """Make n fake transactions: random current/last locations as "lat,lon" text and random elapsed seconds."""
    rng = np.random.default_rng(SEED)
    lats, lons = rng.uniform(-90, 90, (2, n)), rng.uniform(-180, 180, (2, n))
    current = [f"{a!r},{b!r}" for a, b in zip(lats[0], lons[0])]
    last = [f"{a!r},{b!r}" for a, b in zip(lats[1], lons[1])]
    elapsed = rng.uniform(1.0, 86_400.0, n).tolist()
    return current, last, elapsed


def old_loop(current: list[str], last: list[str | None], elapsed: list[float]) -> list[float | None]:
    """The loop that used to run inside compute_geo_velocity_batch: one function call per row."""
    return [compute_geo_velocity(c, last_geo, e) if last_geo is not None and e > 0 else None
            for c, last_geo, e in zip(current, last, elapsed)]


def median_ms(func, runs: int) -> float:
    """Run func `runs` times (after one warm-up call) and return the median time in milliseconds."""
    func()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def identical(left: list, right: list) -> bool:
    """True when two result lists match exactly (same binary floats, same Nones)."""
    return [None if v is None else struct.pack("d", v) for v in left] == \
           [None if v is None else struct.pack("d", v) for v in right]


def main() -> None:
    # Force the CPU branch even for big batches, so we time the CPU path and never touch the GPU kernel.
    geo_velocity.GEO_KERNEL_CROSSOVER_BATCH_SIZE = 10**9

    results = []
    for n in BATCH_SIZES:
        current, last, elapsed = make_rows(n)
        old_ms = median_ms(lambda: old_loop(current, last, elapsed), TIMED_RUNS)
        new_ms = median_ms(lambda: compute_geo_velocity_batch(current, last, elapsed), TIMED_RUNS)
        same = identical(old_loop(current, last, elapsed), compute_geo_velocity_batch(current, last, elapsed))
        results.append({"batch_size": n, "old_loop_median_ms": round(old_ms, 3),
                        "vectorized_median_ms": round(new_ms, 3), "speedup_x": round(old_ms / new_ms, 2),
                        "outputs_bit_identical": same})
        print(f"N={n:>6}  old loop {old_ms:9.3f} ms  vectorized {new_ms:9.3f} ms  "
              f"x{old_ms / new_ms:.2f}  identical={same}")

    section = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "what": "compute_geo_velocity_batch CPU path: old per-row loop vs vectorized NumPy (parsing included in both)",
        "timed_runs_per_size": TIMED_RUNS,
        "headline_batch_size": HEADLINE_BATCH_SIZE,
        "note": "In-process micro-benchmark on the dev laptop; no Triton or network. Text parsing is still a Python "
                "loop in the new path, so the speedup comes from vectorizing the maths only.",
        "results": results,
    }

    # Merge into the existing file so other Level 6 sections are not overwritten.
    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["vectorization"] = section
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote the 'vectorization' section to {REPORT_PATH}")


if __name__ == "__main__":
    main()
