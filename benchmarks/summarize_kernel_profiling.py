# File: benchmarks/summarize_kernel_profiling.py
# Purpose: Level 6 - turn the profiler files in reports/benchmarks/level6_nsight/ into the "gpu_kernel_profiling" and
#          "triton_gpu_metrics" sections of reports/benchmarks/level6_batch.json (other sections are kept).
#
# Where each number comes from:
#   kernel_time_ms         Nsight Systems trace (.nsys-rep): the median GPU run time of the Haversine kernel over
#                          all its launches in that trace (`nsys stats --report cuda_gpu_kern_sum`).
#   inferences_per_sec     rows / kernel time. "Inference" here means one transaction row scored by the
#                          kernel; it counts the kernel only, not copying data to or from the GPU.
#   sm_occupancy_pct       Nsight Compute CSV, "Achieved Occupancy" (how full the GPU's compute units were with warps).
#   hbm_bandwidth_util_pct Nsight Compute CSV, "DRAM Throughput" (share of the memory bandwidth peak that was used).
#                          This GPU has GDDR6 memory, not HBM; the field name follows the Level 6 plan.
#
# Run inside WSL2 (it calls nsys), from the repo root:  python3 benchmarks/summarize_kernel_profiling.py
import csv
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO_ROOT / "reports" / "benchmarks" / "level6_nsight"
REPORT_PATH = REPO_ROOT / "reports" / "benchmarks" / "level6_batch.json"
NSYS = "/opt/nvidia/nsight-systems/2026.1.3/bin/nsys"
SIZES = [10_000, 100_000, 1_000_000]
KERNEL_NAME = "haversine_triton_kernel"


def kernel_stats_from_trace(trace: Path) -> dict:
    """Run `nsys stats` on a COPY of the trace (nsys writes a .sqlite file next to it) and read the kernel row."""
    with tempfile.TemporaryDirectory() as work:
        copy = Path(work) / trace.name
        shutil.copy(trace, copy)
        output = subprocess.run([NSYS, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(copy)],
                                check=True, capture_output=True, text=True).stdout
    lines = output.splitlines()
    header_at = next(i for i, line in enumerate(lines) if line.startswith("Time (%)"))
    for row in csv.DictReader(lines[header_at:]):
        if KERNEL_NAME in row["Name"]:
            return {"instances": int(row["Instances"]), "median_ns": float(row["Med (ns)"]),
                    "min_ns": float(row["Min (ns)"]), "max_ns": float(row["Max (ns)"])}
    raise RuntimeError(f"No {KERNEL_NAME} kernel events in {trace.name}: the trace is empty of CUDA kernels")


def counters_from_ncu_csv(path: Path) -> dict:
    """Pick the few Nsight Compute numbers we report out of the full details table."""
    # "Memory Throughput" appears twice in the table (once in percent, once in Gbyte/s), so the unit is part of the key.
    wanted = {("Achieved Occupancy", "%"): "sm_occupancy_pct",
              ("Theoretical Occupancy", "%"): "theoretical_occupancy_pct",
              ("DRAM Throughput", "%"): "hbm_bandwidth_util_pct",
              ("Compute (SM) Throughput", "%"): "sm_throughput_pct",
              ("Memory Throughput", "Gbyte/s"): "dram_gb_per_s",
              ("Duration", "us"): "ncu_duration_us_reference_only"}
    found = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = wanted.get((row["Metric Name"], row["Metric Unit"]))
            if key and key not in found:
                found[key] = float(row["Metric Value"].replace(",", ""))
    missing = set(wanted.values()) - set(found)
    if missing:
        raise RuntimeError(f"{path.name} is missing Nsight Compute metrics: {sorted(missing)}")
    return found


def main() -> int:
    per_size = []
    for rows in SIZES:
        trace_stats = kernel_stats_from_trace(TRACE_DIR / f"haversine_{rows}.nsys-rep")
        counters = counters_from_ncu_csv(TRACE_DIR / f"haversine_{rows}_ncu.csv")
        kernel_ms = trace_stats["median_ns"] / 1e6
        per_size.append({
            "rows": rows,
            "kernel_time_ms": round(kernel_ms, 5),
            "kernel_time_ms_min_max": [round(trace_stats["min_ns"] / 1e6, 5), round(trace_stats["max_ns"] / 1e6, 5)],
            "kernel_launches_in_trace": trace_stats["instances"],
            "inferences_per_sec": round(rows / (kernel_ms / 1000.0)),
            "sm_occupancy_pct": counters["sm_occupancy_pct"],
            "theoretical_occupancy_pct": counters["theoretical_occupancy_pct"],
            "hbm_bandwidth_util_pct": counters["hbm_bandwidth_util_pct"],
            "dram_gb_per_s": counters["dram_gb_per_s"],
            "sm_throughput_pct": counters["sm_throughput_pct"],
            "ncu_duration_us_reference_only": counters["ncu_duration_us_reference_only"],
            # The kernel reads 5 float32 arrays and writes 1, so it moves rows x 6 x 4 bytes. Divided by the nsys
            # kernel time this is the data rate the kernel actually achieved in the repeated-launch timeline.
            "bytes_moved": rows * 6 * 4,
            "implied_data_rate_gb_per_s_from_nsys": round(rows * 6 * 4 / trace_stats["median_ns"], 1),
            "dram_peak_gb_per_s_from_ncu": round(
                counters["dram_gb_per_s"] / counters["hbm_bandwidth_util_pct"] * 100, 1),
        })
        print(f"{rows:>9} rows  kernel {kernel_ms * 1000:8.2f} us  {per_size[-1]['inferences_per_sec']:>14,} rows/s  "
              f"occupancy {counters['sm_occupancy_pct']}%  DRAM {counters['hbm_bandwidth_util_pct']}%")

    section = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "what": "GPU profile of the Level 4.5 HAVERSINE geo-velocity kernel (features/gpu_geo_velocity.py). These "
                "numbers describe that kernel only, NOT Triton inference.",
        "hardware": "NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6), run in WSL2",
        "tools": {"kernel_time": "Nsight Systems 2026.1.3, --trace=cuda --sample=none --cpuctxsw=none",
                  "counters": "Nsight Compute 2026.3.1, one launch per size, sections SpeedOfLight, Occupancy, "
                              "MemoryWorkloadAnalysis"},
        "files": "reports/benchmarks/level6_nsight/ (.nsys-rep traces, .ncu-repz reports, _ncu.csv tables)",
        "how_numbers_are_derived": {
            "kernel_time_ms": "median GPU time of the kernel across the launches in the Nsight Systems trace",
            "inferences_per_sec": "rows / kernel_time; one row = one inference; kernel only, no host-device copies",
            "sm_occupancy_pct": "Nsight Compute achieved occupancy (separate profiling run from the nsys trace)",
            "hbm_bandwidth_util_pct": "Nsight Compute DRAM throughput as a percent of peak (the GPU uses GDDR6, not "
                                      "HBM)",
            "ncu_duration_us_reference_only": "kept for reference, not used for throughput: Nsight Compute runs the "
                                              "kernel under its own measurement settings (it normally locks clocks "
                                              "and flushes caches, ESTIMATE from its documented defaults), so its "
                                              "duration is not the free-running time"},
        "results": per_size,
        "verdict": "Memory-bound, and more so as the batch grows, when the data comes cold from DRAM (Nsight Compute, "
                   "caches flushed): DRAM throughput rises 23.5% -> 69.2% -> 93.4% of peak from 10k to 1M rows while "
                   "SM (compute) throughput only rises 6.2% -> 22.8% -> 31.3%. At 10k and 100k the GPU is also "
                   "under-filled (achieved occupancy 8.3% and 34.1%, against a theoretical 66.7%); at 1M occupancy "
                   "reaches 58.0% and memory bandwidth is the limit, not compute.",
        "caveat_cache": "The Nsight Systems timeline repeats the SAME inputs, so later launches probably read from the "
                        "GPU's large L2 cache instead of DRAM (ESTIMATE). Evidence: at 100k and 1M rows the data rate "
                        "implied by the nsys kernel time (see implied_data_rate_gb_per_s_from_nsys) is higher than the "
                        "DRAM peak Nsight Compute reports (dram_peak_gb_per_s_from_ncu), which DRAM alone cannot "
                        "deliver. So the nsys inferences_per_sec is a warm-cache figure; with fresh data for every "
                        "batch the kernel would be slower, closer to the Nsight Compute duration. Not measured "
                        "separately.",
        "nsys_note": "The older Nsight Systems 2022.4 from the Ubuntu package recorded no CUDA kernel events for this "
                     "CUDA 12.4 setup (its traces had no kernel tables), so it was replaced by 2026.1.3 from "
                     "NVIDIA's CUDA repo for WSL.",
    }
    triton_note = {
        "note": "The Triton fraud model (fraud_xgboost, ONNX Runtime) runs on the CPU: instance_group kind is "
                "KIND_CPU and the model is a tree ensemble. SM occupancy, memory-bandwidth utilization and "
                "inferences-per-second-per-GPU are therefore N/A for it. GPU profiling here applies only to the "
                "Haversine kernel above. No GPU numbers are recorded for the Triton model.",
        "triton_model_gpu_metrics": "N/A",
    }
    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["gpu_kernel_profiling"] = section
    report["triton_gpu_metrics"] = triton_note
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote 'gpu_kernel_profiling' and 'triton_gpu_metrics' to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
