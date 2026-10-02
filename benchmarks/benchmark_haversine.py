# File: benchmarks/benchmark_haversine.py
# Purpose: Benchmarks the CPU NumPy Haversine path against the OpenAI Triton kernel language GPU kernel.
#          Finds the crossover batch size and writes reports/benchmarks/level4_5_haversine.json.
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import triton

# Allow running as "python benchmarks/benchmark_haversine.py" from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from features.geo_velocity import haversine_distance_km  # noqa: E402
from features.gpu_geo_velocity import BLOCK_SIZE, compute_geo_velocity_gpu, haversine_triton_kernel  # noqa: E402

BATCH_SIZE = 10_000
TIMED_RUNS = 100
CROSSOVER_RUNS = 20
CROSSOVER_BATCH_SIZES = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
REPORT_PATH = REPO_ROOT / "reports" / "benchmarks" / "level4_5_haversine.json"


def cpu_geo_velocity(lat1, lon1, lat2, lon2, elapsed) -> list[float]:
    """Run the CPU reference path: one NumPy Haversine call per element, in a Python loop."""
    return [
        haversine_distance_km(lat1[i], lon1[i], lat2[i], lon2[i]) / (elapsed[i] / 3600.0) for i in range(len(lat1))
    ]


def launch_kernel_only(tensors: list[torch.Tensor], output: torch.Tensor, n: int) -> None:
    """Launch the GPU kernel on tensors that are already on the GPU (no host-device transfer)."""
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    haversine_triton_kernel[grid](*tensors, output, n, BLOCK_SIZE=BLOCK_SIZE)


def time_ms(func, runs: int, sync_gpu: bool = False) -> list[float]:
    """Call func `runs` times and return each run's wall time in milliseconds."""
    times = []
    for _ in range(runs):
        # The GPU runs asynchronously: without synchronize() the timer would stop before the kernel finishes.
        if sync_gpu:
            torch.cuda.synchronize()
        start = time.perf_counter()
        func()
        if sync_gpu:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def summarize(times_ms: list[float], batch_size: int) -> dict[str, float]:
    """Return p50/p95/p99 latency in ms and median throughput in transactions per second."""
    p50, p95, p99 = (float(np.percentile(times_ms, q)) for q in (50, 95, 99))
    return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "throughput_txn_per_sec": batch_size / (p50 / 1000.0)}


def make_data(n: int, seed: int) -> tuple[np.ndarray, ...]:
    """Generate n random lat/lon pairs and elapsed times (60 s to 24 h)."""
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-90, 90, n),
        rng.uniform(-180, 180, n),
        rng.uniform(-90, 90, n),
        rng.uniform(-180, 180, n),
        rng.uniform(60.0, 86400.0, n),
    )


def main() -> None:
    """Run all benchmark sections and write the JSON report."""
    # Section 1: generate test data once. The GPU copies are made here so data transfer is outside the timed loop.
    np.random.seed(99)
    data = make_data(BATCH_SIZE, seed=99)
    gpu_tensors = [torch.as_tensor(a, dtype=torch.float32).to("cuda").contiguous() for a in data]
    gpu_output = torch.empty_like(gpu_tensors[0])

    # Section 2: warm-up. The first GPU launch compiles the kernel (JIT compilation) and initialises the CUDA
    # context, which takes far longer than a normal launch. Running both paths once first means the timed runs
    # measure steady-state performance, not compilation.
    cpu_geo_velocity(*data)
    launch_kernel_only(gpu_tensors, gpu_output, BATCH_SIZE)
    torch.cuda.synchronize()

    # Section 3: CPU timing, 100 runs over the full batch.
    cpu_times = time_ms(lambda: cpu_geo_velocity(*data), TIMED_RUNS)
    cpu_stats = summarize(cpu_times, BATCH_SIZE)

    # Section 4: GPU timing, 100 runs of the kernel only (data is already on the GPU).
    gpu_times = time_ms(lambda: launch_kernel_only(gpu_tensors, gpu_output, BATCH_SIZE), TIMED_RUNS, sync_gpu=True)
    gpu_stats = summarize(gpu_times, BATCH_SIZE)

    print(f"CPU (10,000 elements, {TIMED_RUNS} runs): {cpu_stats}")
    print(f"GPU kernel only (10,000 elements, {TIMED_RUNS} runs): {gpu_stats}")

    # Section 5: crossover point. GPU timing here INCLUDES host-to-device and device-to-host transfer, because that
    # overhead is what makes the GPU slower at small batch sizes.
    detail = []
    crossover = None
    for size in CROSSOVER_BATCH_SIZES:
        subset = tuple(a[:size] for a in data)
        # Warm up this size too: the kernel is re-specialised for some sizes and must not be timed compiling.
        compute_geo_velocity_gpu(*subset)
        cpu_median = float(np.median(time_ms(lambda s=subset: cpu_geo_velocity(*s), CROSSOVER_RUNS)))
        gpu_median = float(
            np.median(time_ms(lambda s=subset: compute_geo_velocity_gpu(*s), CROSSOVER_RUNS, sync_gpu=True))
        )
        faster = "gpu" if gpu_median < cpu_median else "cpu"
        if faster == "gpu" and crossover is None:
            crossover = size
        detail.append(
            {"batch_size": size, "cpu_median_ms": cpu_median, "gpu_median_ms": gpu_median, "faster": faster}
        )
        print(f"batch={size:>6}  cpu={cpu_median:9.4f} ms  gpu(with transfer)={gpu_median:9.4f} ms  faster={faster}")

    print(f"CROSSOVER BATCH SIZE: {crossover}")

    # Section 6: save the report.
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_size": BATCH_SIZE,
        "runs": TIMED_RUNS,
        "cpu": cpu_stats,
        "gpu": gpu_stats,
        "crossover_batch_size": crossover,
        "crossover_detail": detail,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
