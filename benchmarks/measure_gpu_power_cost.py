# File: benchmarks/measure_gpu_power_cost.py
# Purpose: Level 6 - measure the GPU power draw and the sustained speed of the Level 4.5 Haversine kernel IN THE SAME
#          RUN, then turn them into an electricity cost per inference with core/energy_cost.py. Writes the
#          "gpu_cost_per_inference" section of reports/benchmarks/level6_batch.json (other sections are kept).
#
# How it works, for each batch size (10,000 / 100,000 / 1,000,000 rows):
#   1. a short warm-up (the first launch compiles the kernel);
#   2. a SUSTAINED loop of back-to-back kernel launches for about DURATION seconds;
#   3. while that loop runs, nvidia-smi reports the GPU power draw every 0.3 s in a background process;
#   4. average power = mean of those readings; throughput = rows launched / loop time (same run, same moment).
#
# Note on speed: this sustained throughput is limited partly by how fast Python can launch the kernel (a launch costs
# far more time than a 10,000-row kernel runs), so it is lower than the kernel-only figure in the Nsight Systems
# trace. It is the speed the loop really achieved while the power was measured, which is what a cost needs.
# The loop reuses the same input every launch, so the GPU's L2 cache is warm (see the caveat in the JSON).
#
# Run inside WSL2 from the repo root, with Docker stopped:  python benchmarks/measure_gpu_power_cost.py
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import triton

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.energy_cost import DEFAULT_PRICE_PER_KWH_USD, cost_per_inference_usd, watts_to_kilowatts  # noqa: E402
from features.gpu_geo_velocity import BLOCK_SIZE, haversine_triton_kernel  # noqa: E402

REPORT_PATH = REPO_ROOT / "reports" / "benchmarks" / "level6_batch.json"
SIZES = [10_000, 100_000, 1_000_000]
DURATION_SECONDS = 15.0  # length of the sustained loop per size
POWER_SAMPLE_MS = 300
SYNC_EVERY = 200  # wait for the GPU every N launches so the launch queue cannot grow without limit
SEED = 7


def make_inputs(rows: int) -> list[torch.Tensor]:
    """Five random float32 arrays on the GPU (same ranges as benchmarks/profile_kernel_at_size.py)."""
    rng = np.random.default_rng(SEED)
    arrays = [rng.uniform(-90, 90, rows), rng.uniform(-180, 180, rows), rng.uniform(-90, 90, rows),
              rng.uniform(-180, 180, rows), rng.uniform(60.0, 86400.0, rows)]
    return [torch.as_tensor(a, dtype=torch.float32).to("cuda").contiguous() for a in arrays]


def start_power_sampler() -> subprocess.Popen:
    """Start nvidia-smi printing one power reading (watts) every POWER_SAMPLE_MS milliseconds."""
    return subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                             "-lms", str(POWER_SAMPLE_MS)], stdout=subprocess.PIPE, text=True)


def stop_power_sampler(sampler: subprocess.Popen) -> list[float]:
    """Stop nvidia-smi and return every reading it printed."""
    sampler.terminate()
    output, _ = sampler.communicate(timeout=10)
    return [float(line) for line in output.split() if line.replace(".", "", 1).isdigit()]


def measure_size(rows: int) -> dict:
    """Run the sustained loop for one size and return speed, power and cost."""
    tensors = make_inputs(rows)
    output = torch.empty_like(tensors[0])
    grid = (triton.cdiv(rows, BLOCK_SIZE),)

    def launch() -> None:
        haversine_triton_kernel[grid](*tensors, output, rows, BLOCK_SIZE=BLOCK_SIZE)

    for _ in range(20):  # warm-up: compile, load, let the clocks rise
        launch()
    torch.cuda.synchronize()

    sampler = start_power_sampler()
    time.sleep(1.0)  # let the first readings arrive before the loop starts
    launches = 0
    loop_start = time.perf_counter()
    while time.perf_counter() - loop_start < DURATION_SECONDS:
        for _ in range(SYNC_EVERY):
            launch()
        torch.cuda.synchronize()
        launches += SYNC_EVERY
    loop_seconds = time.perf_counter() - loop_start
    readings = stop_power_sampler(sampler)

    # Keep only readings taken while the loop was running: drop the first second (taken before it started).
    skip = int(1000 / POWER_SAMPLE_MS)
    loop_readings = readings[skip:skip + int(loop_seconds * 1000 / POWER_SAMPLE_MS)]
    avg_power_w = statistics.mean(loop_readings)
    inferences = launches * rows
    seconds_per_inference = loop_seconds / inferences
    return {
        "rows": rows,
        "loop_seconds": round(loop_seconds, 2),
        "launches": launches,
        "sustained_inferences_per_sec": round(inferences / loop_seconds),
        "avg_power_w": round(avg_power_w, 2),
        "power_readings_used": len(loop_readings),
        "power_w_min_max": [min(loop_readings), max(loop_readings)],
        "seconds_per_inference": seconds_per_inference,
        "cost_per_inference_usd": cost_per_inference_usd(watts_to_kilowatts(avg_power_w), seconds_per_inference),
    }


def idle_power_w() -> float:
    """Average GPU power for about 3 seconds with nothing running, for comparison (not used in the cost)."""
    sampler = start_power_sampler()
    time.sleep(3.5)
    return round(statistics.mean(stop_power_sampler(sampler)), 2)


def main() -> int:
    idle = idle_power_w()
    print(f"idle GPU power: {idle} W")
    results = []
    for rows in SIZES:
        result = measure_size(rows)
        results.append(result)
        print(f"{rows:>9} rows: {result['sustained_inferences_per_sec']:>14,} inferences/s  "
              f"{result['avg_power_w']:6.2f} W avg  ${result['cost_per_inference_usd']:.3e} per inference")

    section = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "what": "Electricity cost per inference of the Haversine GPU kernel (not Triton inference)",
        "formula": "cost_per_inference_usd = avg_power_kW * seconds_per_inference / 3600 * price_per_kWh "
                   "(core/energy_cost.py; the same formula Level 10.5 reuses)",
        "price_per_kwh_usd": DEFAULT_PRICE_PER_KWH_USD,
        "price_is": "an ASSUMPTION (editable default), not a measured value; set your real tariff to get your cost",
        "measured": "avg_power_w from nvidia-smi during a sustained loop, and throughput from the same loop",
        "method": {"loop_seconds_target": DURATION_SECONDS, "power_sample_ms": POWER_SAMPLE_MS,
                   "idle_gpu_power_w": idle, "same_input_every_launch": True},
        "caveats": [
            "Warm cache: the loop reuses the same input, so later launches probably read from the GPU's L2 cache "
            "instead of DRAM (ESTIMATE). With fresh data per batch the kernel is slower and the cost per inference "
            "higher.",
            "Sustained throughput is limited by Python kernel-launch time at small sizes, so it is lower than the "
            "kernel-only rows/sec in the Nsight Systems trace.",
            "Power is the whole GPU board power reported by nvidia-smi (including idle draw), not the kernel's share.",
        ],
        "results": results,
    }
    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["gpu_cost_per_inference"] = section
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote 'gpu_cost_per_inference' to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
