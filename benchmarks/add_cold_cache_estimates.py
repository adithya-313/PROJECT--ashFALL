# File: benchmarks/add_cold_cache_estimates.py
# Purpose: Level 6 hardening - make the warm-cache caveat impossible to misread in reports/benchmarks/level6_batch.json.
#          No new measurement is taken. This script only (1) renames the warm-cache figures so their NAMES say they are
#          optimistic, and (2) adds a conservative "cold cache" estimate next to each one, derived from numbers that
#          are already in the file. It is safe to run twice: it does nothing the second time.
#
# The two figures bracket the truth:
#   warm (optimistic)   the Nsight Systems kernel time and the sustained loop REUSE the same input, so after the first
#                       launch the data probably sits in the GPU's L2 cache instead of coming from DRAM (ESTIMATE).
#   cold (conservative) the Nsight Compute kernel duration. Nsight Compute flushes caches before the kernel and by
#                       default locks the GPU clocks (documented defaults, not tested here), so the kernel looks slower
#                       than it would run for free. It is a lower bound on speed, an upper bound on cost.
# Neither figure includes copying data from the CPU to the GPU. Real serving would also pay for that.
#
# Run from the repo root:  PYTHONPATH=. python benchmarks/add_cold_cache_estimates.py
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.energy_cost import cost_per_inference_usd, watts_to_kilowatts  # noqa: E402

REPORT_PATH = REPO_ROOT / "reports" / "benchmarks" / "level6_batch.json"

HOW_TO_READ = ("Every speed and cost here comes in two flavours. *_warmcache_optimistic reuses the same input, so the "
               "GPU's L2 cache is probably warm (ESTIMATE). *_cold_estimate comes from Nsight Compute, which flushes "
               "caches and may lock clocks, so it is a conservative bound. The realistic value for fresh data every "
               "batch most likely lies between the two (ESTIMATE). Neither includes host-to-GPU data copies.")
THROUGHPUT_NOTE = ("Truth most likely sits between inferences_per_sec_cold_estimate (conservative, Nsight Compute "
                   "duration, caches flushed) and inferences_per_sec_warmcache_optimistic (Nsight Systems median, "
                   "same input reused so L2 is probably warm). ESTIMATE; neither was measured with fresh data per "
                   "batch.")
COST_BRACKET_NOTE = ("Truth most likely sits between cost_per_inference_usd_warmcache_optimistic (lowest cost, warm-"
                     "loop speed) and cost_per_inference_usd_cold_estimate (highest cost). The cold figure ASSUMES the "
                     "same average board power as the warm loop, which was not measured for a cold run (ESTIMATE).")
COST_NO_BRACKET_NOTE = (
    "NOT a bracket at this size. The warm sustained loop is limited by Python kernel-launch time (the GPU sits idle "
    "between launches), so its cost per inference is HIGHER than the kernel-only cold estimate. Read "
    "cost_per_inference_usd_warmcache_optimistic as the cost of this launch-limited loop, and "
    "cost_per_inference_usd_cold_estimate as a kernel-only figure that assumes the loop's average power "
    "(ESTIMATE, not measured for a cold run).")


def add_to_kernel_profiling(section: dict) -> None:
    """Rename the warm figures and add the cold throughput estimate to each size."""
    section["how_to_read"] = HOW_TO_READ
    for row in section["results"]:
        if "inferences_per_sec" in row:  # not yet migrated
            row["inferences_per_sec_warmcache_optimistic"] = row.pop("inferences_per_sec")
            row["kernel_time_ms_warmcache_optimistic"] = row.pop("kernel_time_ms")
        cold_seconds = row["ncu_duration_us_reference_only"] * 1e-6
        row["kernel_time_ms_cold_estimate"] = round(cold_seconds * 1000.0, 5)
        row["inferences_per_sec_cold_estimate"] = round(row["rows"] / cold_seconds)
        row["cold_estimate_derived_as"] = "rows / ncu_duration_us_reference_only (Nsight Compute, caches flushed)"
        row["note"] = THROUGHPUT_NOTE


def add_to_cost(section: dict, kernel_rows: dict) -> None:
    """Rename the warm cost figures and add a cold cost estimate (same measured power, longer time per inference)."""
    section["how_to_read"] = HOW_TO_READ
    price = section["price_per_kwh_usd"]
    for row in section["results"]:
        if "cost_per_inference_usd" in row:  # not yet migrated
            row["cost_per_inference_usd_warmcache_optimistic"] = row.pop("cost_per_inference_usd")
            row["sustained_inferences_per_sec_warmcache_optimistic"] = row.pop("sustained_inferences_per_sec")
            row["seconds_per_inference_warmcache_optimistic"] = row.pop("seconds_per_inference")
        cold_seconds_per_inference = kernel_rows[row["rows"]]["kernel_time_ms_cold_estimate"] / 1000.0 / row["rows"]
        row["seconds_per_inference_cold_estimate"] = cold_seconds_per_inference
        row["cost_per_inference_usd_cold_estimate"] = cost_per_inference_usd(
            watts_to_kilowatts(row["avg_power_w"]), cold_seconds_per_inference, price)
        row["cold_estimate_derived_as"] = ("cost_per_inference_usd(avg_power_w from the warm loop, "
                                           "ncu_duration / rows, price_per_kwh_usd)")
        warm = row["cost_per_inference_usd_warmcache_optimistic"]
        cold = row["cost_per_inference_usd_cold_estimate"]
        row["warm_and_cold_bracket_the_truth"] = warm <= cold
        row["note"] = COST_BRACKET_NOTE if warm <= cold else COST_NO_BRACKET_NOTE


def main() -> int:
    report = json.loads(REPORT_PATH.read_text())
    add_to_kernel_profiling(report["gpu_kernel_profiling"])
    kernel_rows = {row["rows"]: row for row in report["gpu_kernel_profiling"]["results"]}
    add_to_cost(report["gpu_cost_per_inference"], kernel_rows)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for row, cost in zip(report["gpu_kernel_profiling"]["results"], report["gpu_cost_per_inference"]["results"]):
        print(f"{row['rows']:>9} rows  inferences/s warm {row['inferences_per_sec_warmcache_optimistic']:>14,}  "
              f"cold {row['inferences_per_sec_cold_estimate']:>14,}  | cost warm "
              f"{cost['cost_per_inference_usd_warmcache_optimistic']:.2e}  cold "
              f"{cost['cost_per_inference_usd_cold_estimate']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
