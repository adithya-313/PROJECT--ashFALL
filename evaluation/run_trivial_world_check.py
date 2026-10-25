"""
WHAT: Evaluates trivial separability across all planned worlds under amendment A-026.
Runs 680 planned worlds (training 320, validation 120, test 120, stress 120) using
process pool workers, computes six candidate behavioral signals per account, scores
AP against ground truth mules, gates training/validation splits, and reports test/stress splits.
Executed via: py -3.12 -m evaluation.run_trivial_world_check.

WHY: Ground Rule 17 and amendment A-026 forbid trivial worlds where a single simple feature
separates fraud so well (AUPRC >= 0.90) that any model looks artificially competent.
Gates training/validation worlds to prevent easy worlds from entering training pipelines,
while auditing test/stress worlds as report-only indicators.

TERM: gate split: worlds (training, validation) where best_single_feature_auprc >= 0.90 causes a hard stop.
TERM: report-only split: worlds (test, stress) where results are audited and never used for tuning.
"""

import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config_validator import load_yaml  # noqa: E402
from evaluation.trivial_world import check_world  # noqa: E402
from world.sampled_world import run_sampled_world  # noqa: E402
from world.world_configs import read_world_configs  # noqa: E402
from world.world_sampler import WorldSpec, load_sampler_config, sample_block  # noqa: E402

PLANNED_BLOCKS: dict[str, int] = {
    "training": 8,
    "validation": 3,
    "test": 3,
    "stress": 3,
}

Task = tuple[WorldSpec, str]


def evaluate_world_task(task: Task) -> dict[str, Any]:
    """
    WHAT: Worker entry point executing a single sampled world simulation and evaluating signals.
    Runs simulation under the specified preset, calculates six behavioral signals across accounts,
    and returns a compact summary dictionary containing AP metrics and held-out gang evaluations.

    WHY: Executing inside child worker processes prevents simulation memory leaks and leverages
    multi-core parallelism across all 680 planned experimental worlds.

    Args:
        task: Tuple of (spec, preset) defining world parameters and simulation sizing.

    Returns:
        Compact dictionary containing world identity, fraud share, population counts, AP scores,
        pass status, and held-out tactic evaluations.
    """
    spec, preset = task
    sampler_cfg = load_sampler_config()
    eval_cfg = load_yaml(REPO_ROOT / "config" / "evaluation.yaml")

    output = run_sampled_world(spec, preset, None, sampler_cfg, "run-trivial-world-check")
    result = check_world(output, eval_cfg)

    return {
        "world_id": spec.world_id,
        "split": spec.split,
        "seed": spec.seed,
        "fraud_share": result["fraud_share"],
        "n_mules": result["n_mules"],
        "n_genuine": result["n_genuine"],
        "best_signal": result["best_signal"],
        "best_ap": result["best_ap"],
        "passed": result["passed"],
        "per_signal_ap": result["per_signal_ap"],
        "top_3": result["top_3"],
        "held_out": result["held_out"],
    }


def build_world_tasks(
    sampler_seed: int,
    preset: str,
    limit_per_split: int | None = None,
) -> list[Task]:
    """
    WHAT: Generates planned WorldSpec tasks across all four splits based on sampler configuration.
    Produces 320 training, 120 validation, 120 test, and 120 stress specs in block and seed order.

    WHY: Standardizes world specification generation against committed block sizing rules in A-023.

    Args:
        sampler_seed: Master integer seed feeding Latin hypercube stream sequences.
        preset: Sizing preset string from world_sizing.yaml (e.g., 'small').
        limit_per_split: Optional integer limiting specs per split for testing.

    Returns:
        List of Task tuples ready for pool execution.
    """
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_configs().world_ranges
    tasks: list[Task] = []

    for split, num_blocks in PLANNED_BLOCKS.items():
        split_specs: list[WorldSpec] = []
        for block_idx in range(num_blocks):
            block_specs = sample_block(split, block_idx, sampler_seed, sampler_cfg, world_ranges)
            split_specs.extend(block_specs)

        if limit_per_split is not None:
            split_specs = split_specs[:limit_per_split]

        for spec in split_specs:
            tasks.append((spec, preset))

    return tasks


def write_raw_results(results: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    """
    WHAT: Writes raw world evaluation results to CSV and JSON formats in the specified output directory.
    Guarantees that files are placed outside the repository to prevent accidental git tracking.

    WHY: Amendment A-026 demands complete archival of simulation outputs outside version control.

    Args:
        results: List of world check result dictionaries.
        out_dir: Target filesystem directory located outside the repository root.

    Returns:
        Tuple of (csv_path, json_path) pointing to written result artifacts.

    Raises:
        ValueError: If out_dir is located inside the repository workspace.
    """
    resolved_dir = out_dir.resolve()
    if REPO_ROOT == resolved_dir or REPO_ROOT in resolved_dir.parents:
        raise ValueError(f"Output directory {resolved_dir} is inside repository {REPO_ROOT}; forbidden.")

    resolved_dir.mkdir(parents=True, exist_ok=True)
    csv_path = resolved_dir / "trivial_world_results.csv"
    json_path = resolved_dir / "trivial_world_results.json"

    # 1. Write comprehensive JSON serialization.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 2. Write flattened tabular CSV export.
    fieldnames = [
        "world_id",
        "split",
        "seed",
        "fraud_share",
        "n_mules",
        "n_genuine",
        "best_signal",
        "best_ap",
        "passed",
        "devices",
        "dwell",
        "pass_through",
        "payer_count",
        "payer_regions",
        "routine_life",
        "held_out_family",
        "held_out_best_signal",
        "held_out_best_ap",
        "held_out_passed",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in results:
            signals = r["per_signal_ap"]
            ho = r.get("held_out")
            row = {
                "world_id": r["world_id"],
                "split": r["split"],
                "seed": r["seed"],
                "fraud_share": round(r["fraud_share"], 6),
                "n_mules": r["n_mules"],
                "n_genuine": r["n_genuine"],
                "best_signal": r["best_signal"],
                "best_ap": round(r["best_ap"], 6),
                "passed": r["passed"],
                "devices": round(signals.get("devices", 0.0), 6),
                "dwell": round(signals.get("dwell", 0.0), 6),
                "pass_through": round(signals.get("pass_through", 0.0), 6),
                "payer_count": round(signals.get("payer_count", 0.0), 6),
                "payer_regions": round(signals.get("payer_regions", 0.0), 6),
                "routine_life": round(signals.get("routine_life", 0.0), 6),
                "held_out_family": ho["family"] if ho else "",
                "held_out_best_signal": ho["best_signal"] if ho else "",
                "held_out_best_ap": round(ho["best_ap"], 6) if ho else "",
                "held_out_passed": ho["passed"] if ho else "",
            }
            writer.writerow(row)

    return csv_path, json_path


def format_frequency(counts: dict[str, int]) -> str:
    """
    WHAT: Formats a dictionary of string counts into a sorted, comma-separated display string.
    Lists signals by descending frequency count, then alphabetical name.

    WHY: Provides standardized, scannable summaries of dominant features across world splits.

    Args:
        counts: Mapping of signal names to occurrences.

    Returns:
        Formatted summary string (e.g., 'routine_life: 210, devices: 110').
    """
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return ", ".join(f"{k}: {v}" for k, v in sorted_items)


def build_markdown_summary(results: list[dict[str, Any]]) -> str:
    """
    WHAT: Assembles the structured markdown summary conforming exactly to amendment A-026 rules.
    Includes disclaimer header, gate performance tables, report-only tables, held-out family evaluations,
    fraud share distributions, and any report-only findings reaching threshold 0.90.

    WHY: Preserves clear, auditable records committed to git without polluting repository with raw rows.

    Args:
        results: Full list of simulation check result records.

    Returns:
        Markdown document string formatted for reports/trivial_world/summary.md.
    """
    lines: list[str] = [
        "Report-only results for test and stress worlds are never used to tune the world or the threshold (A-026).",
        "",
        "# Trivial-World Check Results (A-026)",
        "",
        "Evaluation of single-feature AUPRC separability across all 680 planned worlds under Level 12h.",
        "Candidate signals: `devices`, `dwell`, `pass_through`, `payer_count`, `payer_regions`, `routine_life`.",
        "Gate threshold: 0.90 (applied strictly to training and validation splits).",
        "",
        "## Gate Splits (Training and Validation)",
        "",
        "| Split | Worlds Checked | Failed (>= 0.90) | Best AP Min | Best AP Median | Best AP Max | Dominant Signals |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for split in ("training", "validation"):
        split_rows = [r for r in results if r["split"] == split]
        checked = len(split_rows)
        failed = sum(1 for r in split_rows if not r["passed"])
        best_aps = [r["best_ap"] for r in split_rows]
        ap_min = min(best_aps) if best_aps else 0.0
        ap_med = statistics.median(best_aps) if best_aps else 0.0
        ap_max = max(best_aps) if best_aps else 0.0

        signal_counts: dict[str, int] = {}
        for r in split_rows:
            sig = r["best_signal"]
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        freq_str = format_frequency(signal_counts)
        lines.append(
            f"| {split} | {checked} | {failed} | {ap_min:.4f} | {ap_med:.4f} | {ap_max:.4f} | {freq_str} |"
        )

    lines.extend([
        "",
        "## Report-Only Splits (Test and Stress)",
        "",
        "| Split | Worlds Checked | At/Above 0.90 | Best AP Min | Best AP Median | Best AP Max | Dominant Signals |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])

    for split in ("test", "stress"):
        split_rows = [r for r in results if r["split"] == split]
        checked = len(split_rows)
        failed = sum(1 for r in split_rows if r["best_ap"] >= 0.90)
        best_aps = [r["best_ap"] for r in split_rows]
        ap_min = min(best_aps) if best_aps else 0.0
        ap_med = statistics.median(best_aps) if best_aps else 0.0
        ap_max = max(best_aps) if best_aps else 0.0

        signal_counts = {}
        for r in split_rows:
            sig = r["best_signal"]
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        freq_str = format_frequency(signal_counts)
        lines.append(
            f"| {split} | {checked} | {failed} | {ap_min:.4f} | {ap_med:.4f} | {ap_max:.4f} | {freq_str} |"
        )

    lines.extend([
        "",
        "## Held-Out Tactic Families (Report-Only: Test and Stress)",
        "",
        "| Family | Worlds Evaluated | At/Above 0.90 | Best AP Min | Best AP Median | Best AP Max | Dominant Signals |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])

    report_rows = [r for r in results if r["split"] in ("test", "stress") and r.get("held_out")]
    for fam in ("slow_drip", "device_rotation"):
        fam_rows = [r for r in report_rows if r["held_out"]["family"] == fam]
        checked = len(fam_rows)
        failed = sum(1 for r in fam_rows if r["held_out"]["best_ap"] >= 0.90)
        best_aps = [r["held_out"]["best_ap"] for r in fam_rows]
        ap_min = min(best_aps) if best_aps else 0.0
        ap_med = statistics.median(best_aps) if best_aps else 0.0
        ap_max = max(best_aps) if best_aps else 0.0

        signal_counts = {}
        for r in fam_rows:
            sig = r["held_out"]["best_signal"]
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        freq_str = format_frequency(signal_counts)
        lines.append(
            f"| {fam} | {checked} | {failed} | {ap_min:.4f} | {ap_med:.4f} | {ap_max:.4f} | {freq_str} |"
        )

    lines.extend([
        "",
        "## Fraud Share Distribution",
        "",
        "| Split | Worlds | Fraud Share Min | Fraud Share Median | Fraud Share Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])

    for split in ("training", "validation", "test", "stress"):
        split_rows = [r for r in results if r["split"] == split]
        shares = [r["fraud_share"] for r in split_rows]
        s_min = min(shares) if shares else 0.0
        s_med = statistics.median(shares) if shares else 0.0
        s_max = max(shares) if shares else 0.0
        lines.append(f"| {split} | {len(split_rows)} | {s_min:.6f} | {s_med:.6f} | {s_max:.6f} |")

    # Audit for report-only findings reaching threshold 0.90.
    findings: list[str] = []
    for r in results:
        if r["split"] in ("test", "stress"):
            if r["best_ap"] >= 0.90:
                findings.append(
                    f"- World `{r['world_id']}` ({r['split']}): "
                    f"best signal `{r['best_signal']}` achieved AUPRC {r['best_ap']:.4f}."
                )
            ho = r.get("held_out")
            if ho and ho["best_ap"] >= 0.90:
                findings.append(
                    f"- Held-out family `{ho['family']}` in `{r['world_id']}` ({r['split']}): "
                    f"best signal `{ho['best_signal']}` achieved AUPRC {ho['best_ap']:.4f}."
                )

    if findings:
        lines.extend([
            "",
            "## Report-only findings (A-026 (c) applies)",
            "",
            "The following report-only signals or held-out families reached or exceeded 0.90 AUPRC:",
            "",
        ])
        lines.extend(findings)

    lines.append("")
    return "\n".join(lines)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """
    WHAT: Parses CLI arguments for trivial-world check execution.
    Configures worker count, preset sizing, sampler seed, and filesystem target destinations.

    WHY: Facilitates deterministic evaluation invocations across command line environments.

    Args:
        argv: List of raw string command line arguments.

    Returns:
        Populated argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(description="Run trivial world check across all planned worlds.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (default 4).")
    parser.add_argument("--preset", default="small", help="World sizing preset (default 'small').")
    parser.add_argument("--sampler-seed", type=int, default=1, help="Sampler master seed (default 1).")
    parser.add_argument(
        "--out-dir",
        default=str(Path(tempfile.gettempdir()) / "ashfall_trivial_world"),
        help="Directory outside repository to store raw CSV and JSON results.",
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "reports" / "trivial_world" / "summary.md"),
        help="Repository markdown summary file path.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional limit on worlds per split for smoke testing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    WHAT: Main orchestration flow running 680 worlds, saving artifacts, and enforcing gate checks.
    Halts execution before committing if any training or validation world fails the 0.90 gate.

    WHY: Enforces rigorous automated quality control defined in Level 12h Round 3.

    Args:
        argv: Optional command line arguments list.

    Returns:
        Integer exit code (0 on success, 1 on gate failure or error).
    """
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.out_dir).resolve()
    report_path = Path(args.report_path).resolve()

    tasks = build_world_tasks(args.sampler_seed, args.preset, args.limit_per_split)
    total_tasks = len(tasks)
    print(f"Starting trivial-world check across {total_tasks} planned worlds with {args.workers} workers...")

    started = time.perf_counter()
    results: list[dict[str, Any]]
    if args.workers <= 1:
        results = []
        for idx, t in enumerate(tasks, 1):
            results.append(evaluate_world_task(t))
            if idx % 50 == 0 or idx == total_tasks:
                now_elapsed = time.perf_counter() - started
                print(
                    f"[{idx}/{total_tasks}] worlds completed "
                    f"({now_elapsed:.1f}s elapsed, {now_elapsed / idx:.2f}s/world)..."
                )
    else:
        results = [None] * total_tasks  # type: ignore[list-item]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_idx = {pool.submit(evaluate_world_task, task): idx for idx, task in enumerate(tasks)}
            completed_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                results[idx] = res
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == total_tasks:
                    now_elapsed = time.perf_counter() - started
                    print(
                        f"[{completed_count}/{total_tasks}] worlds completed "
                        f"({now_elapsed:.1f}s elapsed, {now_elapsed / completed_count:.2f}s/world)..."
                    )
    elapsed = time.perf_counter() - started
    print(f"Completed {len(results)} worlds in {elapsed:.2f} seconds ({elapsed / max(len(results), 1):.2f}s/world).")

    # Save detailed CSV and JSON outside repository.
    csv_path, json_path = write_raw_results(results, out_dir)
    print(f"Archived raw results outside repository:\n  CSV:  {csv_path}\n  JSON: {json_path}")

    # Build and write markdown summary inside repository.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md = build_markdown_summary(results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Wrote committed summary to: {report_path}")

    # Enforce gate check on training and validation worlds.
    gate_failures = [
        r for r in results if r["split"] in ("training", "validation") and not r["passed"]
    ]
    if gate_failures:
        print(f"\nGATE CHECK FAILED: {len(gate_failures)} training/validation worlds exceeded 0.90 AUPRC!")
        for fail in gate_failures:
            print(
                f"  - {fail['world_id']} ({fail['split']}): best signal {fail['best_signal']} "
                f"AUPRC={fail['best_ap']:.4f} >= 0.90"
            )
        return 1

    print("\nGATE CHECK PASSED: All training and validation worlds have best single-feature AUPRC < 0.90.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
