# File: benchmarks/decision_latency.py
# Purpose: In-process timing benchmark of bank payment decision recording and feature computation.
# Issue: Level 15c build round (D1: decision latency against evaluation budget).
"""
WHAT: In-process latency benchmark measuring server-side POST /v1/decisions execution time
(record_decision) with feature logging enabled, alongside standalone M1 feature extraction time
(compute_decision_row) against a pre-seeded transactional and behavioral history in PostgreSQL and Redis.

WHY: Verifies compliance with the pre-registered latency budget specified in config/evaluation.yaml:
1. Decision latency: p99 <= max_ms (100 ms).
2. Feature extraction latency: p99 <= feature_computation_max_ms (50 ms).
Measurements are timed server-side with time.perf_counter(). HTTP serialization/network transport
and HMAC/asymmetric request signing are NOT included in this measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import orjson
import redis
import yaml

from api.v1.bank_service import record_decision
from api.v1.models import DecisionRequest
from core.event_clock import ManualClock
from features.decision_features import compute_decision_row
from features.registry_screening import RegistryScreener

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "evaluation.yaml"
OUTPUT_REPORT_PATH = REPO_ROOT / "reports" / "latency" / "level_15_decision_latency.json"

FIXED_BUCKET_EDGES = (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 200.0)


def load_latency_budget(config_file: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads the pre-registered latency budget from config/evaluation.yaml.

    WHY: Project rules strictly forbid typing or hardcoding evaluation numbers.
    All thresholds must be loaded dynamically from the canonical configuration file.
    """
    path = config_file or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "latency" not in data:
        raise ValueError(f"Missing 'latency' section in evaluation config: {path}")

    lat = data["latency"]
    return {
        "endpoint": str(lat["endpoint"]),
        "percentile": float(lat["percentile"]),
        "max_ms": float(lat["max_ms"]),
        "feature_computation_max_ms": float(lat["feature_computation_max_ms"]),
        "measured": str(lat["measured"]),
        "hardware": str(lat["hardware"]),
        "warm_up_required": bool(lat["warm_up_required"]),
    }


def compute_percentiles(values: Sequence[float]) -> dict[str, float]:
    """
    WHAT: Computes count, min, p50, p95, p99, and max for a sequence of floating-point timings.

    WHY: Evaluates latency tail percentiles matching evaluation budget requirements.
    """
    if not values:
        raise ValueError("Cannot compute percentiles on an empty sequence")

    arr = np.array(values, dtype=float)
    return {
        "count": float(len(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def build_histogram(
    values: Sequence[float],
    bucket_edges: Sequence[float] = FIXED_BUCKET_EDGES,
) -> tuple[dict[str, int], str]:
    """
    WHAT: Bins measured latencies into fixed bucket intervals and renders a text histogram.

    WHY: Provides clear visual inspection of timing distributions across warm and steady states.
    """
    if not values:
        raise ValueError("Cannot build histogram on empty values")

    counts: dict[str, int] = {}
    total = len(values)

    # Initialize buckets: [edges[i], edges[i+1]) and [edges[-1], +inf)
    labels: list[str] = []
    for i in range(len(bucket_edges) - 1):
        low = bucket_edges[i]
        high = bucket_edges[i + 1]
        label = f"[{low:5.1f}, {high:5.1f}) ms"
        labels.append(label)
        counts[label] = 0

    overflow_label = f"[>={bucket_edges[-1]:5.1f}   ) ms"
    labels.append(overflow_label)
    counts[overflow_label] = 0

    for v in values:
        placed = False
        for i in range(len(bucket_edges) - 1):
            if bucket_edges[i] <= v < bucket_edges[i + 1]:
                counts[labels[i]] += 1
                placed = True
                break
        if not placed:
            counts[overflow_label] += 1

    # Render ASCII text bar
    lines: list[str] = []
    max_count = max(counts.values()) if counts.values() else 1
    bar_width = 30
    for label in labels:
        c = counts[label]
        pct = (c / total) * 100.0
        bar_len = int(round((c / max_count) * bar_width)) if max_count > 0 else 0
        bar = "#" * bar_len
        lines.append(f"  {label}: {c:5d} ({pct:5.1f}%) {bar}")

    return counts, "\n".join(lines)


@dataclass
class LatencyBenchmarkResult:
    """Container for in-process decision latency benchmark measurements."""

    warmup_count: int
    timed_count: int
    record_decision_p50_ms: float
    record_decision_p95_ms: float
    record_decision_p99_ms: float
    record_decision_max_ms: float
    record_decision_histogram: dict[str, int]
    record_decision_histogram_text: str
    compute_row_p50_ms: float
    compute_row_p95_ms: float
    compute_row_p99_ms: float
    compute_row_max_ms: float
    compute_row_histogram: dict[str, int]
    compute_row_histogram_text: str
    budget: dict[str, Any]
    passed_budget: bool
    platform_name: str
    python_version: str
    scope_note: str


async def measure_decision_latency(
    connection: asyncpg.Connection,
    redis_client: redis.Redis,
    bank_id: str,
    run_id: str,
    payer_acc: str,
    payee_vpa: str,
    decision_time: datetime,
    screener: RegistryScreener,
    warmup_count: int = 5,
    timed_count: int = 30,
    config_file: Path | None = None,
) -> LatencyBenchmarkResult:
    """
    WHAT: Measures decision recording and feature extraction latency in-process.

    Args:
        connection: Active PostgreSQL connection.
        redis_client: Redis client with seeded state.
        bank_id: Bank identifier.
        run_id: 16-hex simulation run ID.
        payer_acc: Account ID with history.
        payee_vpa: Counterparty VPA.
        decision_time: Point-in-time timestamp.
        screener: Registry screener instance.
        warmup_count: Number of untimed warm-up calls (default: 5).
        timed_count: Number of timed calls (default: 30).
        config_file: Optional path to evaluation.yaml.

    Returns:
        LatencyBenchmarkResult with computed percentiles, histograms, and budget pass/fail verdict.
    """
    budget = load_latency_budget(config_file)
    clock = ManualClock(decision_time)

    # 1. Warm-up calls (unmeasured)
    for i in range(warmup_count):
        pid = f"{run_id}-warm-{i:04d}"
        req = DecisionRequest(
            payment_id=pid,
            sender_account_id=payer_acc,
            receiver_bank_id="bank_1",
            receiver_vpa=payee_vpa,
            payee_type="person",
            amount_paise=10000,
            event_time=decision_time,
        )
        # Warm compute_decision_row
        compute_decision_row(
            redis_client=redis_client,
            bank_id=bank_id,
            request=req,
            clock=clock,
            screener=screener,
        )
        # Warm record_decision
        await record_decision(
            connection=connection,
            bank_id=bank_id,
            request=req,
            redis_client=redis_client,
            decision_clock=clock,
            screener=screener,
        )

    # 2. Timed calls
    record_latencies_ms: list[float] = []
    compute_latencies_ms: list[float] = []

    for i in range(timed_count):
        pid = f"{run_id}-timed-{i:04d}"
        req = DecisionRequest(
            payment_id=pid,
            sender_account_id=payer_acc,
            receiver_bank_id="bank_1",
            receiver_vpa=payee_vpa,
            payee_type="person",
            amount_paise=15000 + i * 100,
            event_time=decision_time,
        )

        # Time compute_decision_row alone
        t_c0 = time.perf_counter()
        compute_decision_row(
            redis_client=redis_client,
            bank_id=bank_id,
            request=req,
            clock=clock,
            screener=screener,
        )
        t_c1 = time.perf_counter()
        compute_latencies_ms.append((t_c1 - t_c0) * 1000.0)

        # Time full record_decision (server-side insert + feature computation + log insert)
        t_r0 = time.perf_counter()
        await record_decision(
            connection=connection,
            bank_id=bank_id,
            request=req,
            redis_client=redis_client,
            decision_clock=clock,
            screener=screener,
        )
        t_r1 = time.perf_counter()
        record_latencies_ms.append((t_r1 - t_r0) * 1000.0)

    rec_stats = compute_percentiles(record_latencies_ms)
    comp_stats = compute_percentiles(compute_latencies_ms)

    rec_hist, rec_hist_text = build_histogram(record_latencies_ms)
    comp_hist, comp_hist_text = build_histogram(compute_latencies_ms)

    passed_rec = rec_stats["p99"] <= budget["max_ms"]
    passed_comp = comp_stats["p99"] <= budget["feature_computation_max_ms"]
    passed_budget = passed_rec and passed_comp

    plat_name = platform.platform()
    py_ver = sys.version.split()[0]

    scope_note = (
        "Timing measures in-process server-side record_decision and compute_decision_row with "
        "feature logging enabled against real PostgreSQL and Redis. HTTP serialization, network "
        "transport, and cryptographic request signing are NOT included."
    )

    return LatencyBenchmarkResult(
        warmup_count=warmup_count,
        timed_count=timed_count,
        record_decision_p50_ms=rec_stats["p50"],
        record_decision_p95_ms=rec_stats["p95"],
        record_decision_p99_ms=rec_stats["p99"],
        record_decision_max_ms=rec_stats["max"],
        record_decision_histogram=rec_hist,
        record_decision_histogram_text=rec_hist_text,
        compute_row_p50_ms=comp_stats["p50"],
        compute_row_p95_ms=comp_stats["p95"],
        compute_row_p99_ms=comp_stats["p99"],
        compute_row_max_ms=comp_stats["max"],
        compute_row_histogram=comp_hist,
        compute_row_histogram_text=comp_hist_text,
        budget=budget,
        passed_budget=passed_budget,
        platform_name=plat_name,
        python_version=py_ver,
        scope_note=scope_note,
    )


def format_benchmark_output(res: LatencyBenchmarkResult) -> str:
    """Format benchmark results into plain-text terminal output."""
    budget = res.budget
    verdict = "PASS" if res.passed_budget else "FAIL"
    lines = [
        "=" * 72,
        "LEVEL 15c DECISION LATENCY BENCHMARK REPORT [TAG: ESTIMATE]",
        "=" * 72,
        f"Platform:       {res.platform_name}",
        f"Python:         {res.python_version}",
        f"Hardware Note:  the budget is set for '{budget['hardware']}'; the verdict below is only the budget "
        "judgement if this platform is that machine.",
        f"Scope:          {res.scope_note}",
        f"Call Counts:    {res.warmup_count} warm-up calls, {res.timed_count} timed calls",
        "",
        "--- Budget Specifications (config/evaluation.yaml) ---",
        f"  Target Endpoint:           {budget['endpoint']}",
        f"  Evaluation Percentile:     p{budget['percentile']:.0f}",
        f"  Total Decision Budget:     <= {budget['max_ms']:.1f} ms",
        f"  Feature Computation Budget: <= {budget['feature_computation_max_ms']:.1f} ms",
        "",
        "--- Measurement 1: Full record_decision (server-side) ---",
        f"  Count:  {res.timed_count}",
        f"  p50:    {res.record_decision_p50_ms:6.2f} ms",
        f"  p95:    {res.record_decision_p95_ms:6.2f} ms",
        f"  p99:    {res.record_decision_p99_ms:6.2f} ms  (Budget: <= {budget['max_ms']:.1f} ms)",
        f"  Max:    {res.record_decision_max_ms:6.2f} ms",
        "  Distribution Histogram:",
        res.record_decision_histogram_text,
        "",
        "--- Measurement 2: compute_decision_row alone ---",
        f"  Count:  {res.timed_count}",
        f"  p50:    {res.compute_row_p50_ms:6.2f} ms",
        f"  p95:    {res.compute_row_p95_ms:6.2f} ms",
        f"  p99:    {res.compute_row_p99_ms:6.2f} ms  (Budget: <= {budget['feature_computation_max_ms']:.1f} ms)",
        f"  Max:    {res.compute_row_max_ms:6.2f} ms",
        "  Distribution Histogram:",
        res.compute_row_histogram_text,
        "",
        "=" * 72,
        f"OVERALL LATENCY VERDICT: [{verdict}]",
        "=" * 72,
    ]
    return "\n".join(lines)


async def run_benchmark(
    warmup: int = 5,
    timed: int = 30,
    output_path: Path | None = None,
) -> LatencyBenchmarkResult:
    """Orchestrates benchmark environment setup, measurement, and reporting."""
    # Imported here because the showcase support module imports this module's measurement code at top level.
    from showcase.isolation import cleanup_workspace, create_workspace
    from showcase.level_15_support_c import setup_part_c_history

    workspace = await create_workspace()
    try:
        fixture = await setup_part_c_history(workspace.connection)
        try:
            res = await measure_decision_latency(
                connection=fixture.connection,
                redis_client=fixture.live_redis,
                bank_id=fixture.bank_id,
                run_id=fixture.run_id,
                payer_acc=fixture.payer_acc,
                payee_vpa=fixture.payee_vpa,
                decision_time=fixture.decision_time,
                screener=fixture.screener,
                warmup_count=warmup,
                timed_count=timed,
            )
            write_report(res, output_path or OUTPUT_REPORT_PATH)
            return res
        finally:
            await fixture.cleanup()
    finally:
        await cleanup_workspace(workspace)


def write_report(res: LatencyBenchmarkResult, out_file: Path) -> None:
    """Write the benchmark result, histograms included, as JSON."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": res.platform_name,
        "python_version": res.python_version,
        "budget": res.budget,
        "scope_note": res.scope_note,
        "warmup_count": res.warmup_count,
        "timed_count": res.timed_count,
        "record_decision": {
            "p50_ms": res.record_decision_p50_ms,
            "p95_ms": res.record_decision_p95_ms,
            "p99_ms": res.record_decision_p99_ms,
            "max_ms": res.record_decision_max_ms,
            "histogram": res.record_decision_histogram,
        },
        "compute_decision_row": {
            "p50_ms": res.compute_row_p50_ms,
            "p95_ms": res.compute_row_p95_ms,
            "p99_ms": res.compute_row_p99_ms,
            "max_ms": res.compute_row_max_ms,
            "histogram": res.compute_row_histogram,
        },
        "overall_verdict": "PASS" if res.passed_budget else "FAIL",
    }
    out_file.write_bytes(orjson.dumps(report_data, option=orjson.OPT_INDENT_2))


def main() -> None:
    """CLI entrypoint for running the decision latency benchmark."""
    parser = argparse.ArgumentParser(description="In-process decision latency benchmark against evaluation budget.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of untimed warm-up calls (default: 5).",
    )
    parser.add_argument(
        "--timed",
        "-n",
        type=int,
        default=30,
        help="Number of timed measurement calls (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_REPORT_PATH,
        help=f"Path to write JSON benchmark report (default: {OUTPUT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced call count for rapid verification (warmup=2, timed=5).",
    )
    args = parser.parse_args()

    warmup_count = 2 if args.quick else args.warmup
    timed_count = 5 if args.quick else args.timed

    result = asyncio.run(run_benchmark(warmup=warmup_count, timed=timed_count, output_path=args.output))
    print(format_benchmark_output(result))


if __name__ == "__main__":
    main()
