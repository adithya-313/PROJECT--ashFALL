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
import socket
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import orjson
import redis
import uvicorn
import yaml
from prometheus_client import REGISTRY

from api.v1.bank_service import record_decision
from api.v1.models import DecisionRequest
from core.event_clock import IST, ManualClock
from features.decision_features import compute_decision_row
from features.ingest_watermark import get_bank_watermark
from features.m1_intelligence import get_bank_index
from features.registry_prefetch import list_registry_questions
from features.registry_screening import NationalsimRegistryScreener, RegistryScreener
from nationalsim.app import ScreenDelay, create_app, load_nationalsim_config, load_screen_delay_settings
from nationalsim.bank_client import NationalsimBankClient
from nationalsim.keyed_hash import compute_identifier_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "evaluation.yaml"
OUTPUT_REPORT_PATH = REPO_ROOT / "reports" / "latency" / "level_15_decision_latency.json"

FIXED_BUCKET_EDGES = (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 200.0)

# The participant key the benchmark's mock registry and banks share (a test value, never a real secret).
BENCHMARK_DPIP_KEY = "benchmark_dpip_key_32_bytes_long!!"
BENCHMARK_PAYER_DEVICE = "fp_c_dev"
BENCHMARK_PAYEE_BANK_ID = "bank_1"


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


def load_latency_method(config_file: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads the pre-registered measurement method (latency_method block) from config/evaluation.yaml:
    warm-up calls, measured calls per run, number of runs, and what the verdict is judged on.

    WHY: The method was fixed before any new measurement (amendment A-035); the numbers must come from the file.
    """
    path = config_file or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "latency_method" not in data:
        raise ValueError(f"Missing 'latency_method' section in evaluation config: {path}")
    method = data["latency_method"]
    if method["judged_on"] != "median_of_run_p99":
        raise ValueError(f"Unsupported latency_method.judged_on {method['judged_on']!r}; expected 'median_of_run_p99'")
    return {
        "warm_up_calls": int(method["warm_up_calls"]),
        "measured_calls": int(method["measured_calls"]),
        "runs": int(method["runs"]),
        "judged_on": str(method["judged_on"]),
        "slowest_calls_reported": int(method["slowest_calls_reported"]),
    }


def median_of_run_values(values: Sequence[float]) -> float:
    """The median of the per-run p99 values (the verdict statistic of the latency method)."""
    if not values:
        raise ValueError("median_of_run_values needs at least one run")
    return float(np.median(np.asarray(values, dtype=float)))


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
    registry_mode: str = "fake"
    registry_delay: str = "off"
    registry_timeouts: int = 0
    registry_questions: int = 0


def registry_timeouts_so_far() -> int:
    """Total of ashfall_registry_timeouts_total over both identifier types, read from the process registry."""
    total = 0.0
    for identifier_type in ("account", "device"):
        value = REGISTRY.get_sample_value("ashfall_registry_timeouts_total", {"identifier_type": identifier_type})
        total += 0.0 if value is None else value
    return int(total)


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
    timed_count: int = 300,
    config_file: Path | None = None,
    dpip_key: str | None = None,
    registry_mode: str = "fake",
    registry_delay: str = "off",
    registry_questions: int = 0,
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
        timed_count: Number of timed calls (default: 300).
        config_file: Optional path to evaluation.yaml.
        dpip_key: Optional DPIP key used to hash the identifiers screened by the registry.
        registry_mode: "fake" or "http", recorded in the result.
        registry_delay: "off" or "on", recorded in the result.
        registry_questions: Registry questions one decision asks (recorded in the result).

    Returns:
        LatencyBenchmarkResult with computed percentiles, histograms, and budget pass/fail verdict.
    """
    budget = load_latency_budget(config_file)
    clock = ManualClock(decision_time)

    # 1. Warm-up calls (unmeasured)
    for i in range(warmup_count):
        pid = f"{run_id}-warm-{registry_mode}-{registry_delay}-{i:04d}"
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
        await asyncio.to_thread(
            compute_decision_row,
            redis_client=redis_client,
            bank_id=bank_id,
            request=req,
            clock=clock,
            screener=screener,
            dpip_key=dpip_key,
        )
        # Warm record_decision
        await record_decision(
            connection=connection,
            bank_id=bank_id,
            request=req,
            redis_client=redis_client,
            decision_clock=clock,
            screener=screener,
            dpip_key=dpip_key,
        )

    # 2. Timed calls
    timeouts_before = registry_timeouts_so_far()
    record_latencies_ms: list[float] = []
    compute_latencies_ms: list[float] = []

    for i in range(timed_count):
        pid = f"{run_id}-timed-{registry_mode}-{registry_delay}-{i:04d}"
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
        await asyncio.to_thread(
            compute_decision_row,
            redis_client=redis_client,
            bank_id=bank_id,
            request=req,
            clock=clock,
            screener=screener,
            dpip_key=dpip_key,
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
            dpip_key=dpip_key,
        )
        t_r1 = time.perf_counter()
        record_latencies_ms.append((t_r1 - t_r0) * 1000.0)

    timeouts_in_timed_calls = registry_timeouts_so_far() - timeouts_before

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
    if registry_mode == "http":
        scope_note += (
            " Registry screening IS included: every decision asks the mock national registry over real HTTP "
            f"(localhost TCP, uvicorn in this process), added registry delay {registry_delay}."
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
        registry_mode=registry_mode,
        registry_delay=registry_delay,
        registry_timeouts=timeouts_in_timed_calls,
        registry_questions=registry_questions,
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
        f"Registry:       {res.registry_mode}, added delay {res.registry_delay}, "
        f"questions per decision {res.registry_questions}, registry timeouts in timed calls {res.registry_timeouts}",
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


class RunningMockRegistry:
    """The mock national registry served by uvicorn on a free localhost port, in a thread of this process."""

    def __init__(self, server: uvicorn.Server, thread: threading.Thread, base_url: str) -> None:
        self._server = server
        self._thread = thread
        self.base_url = base_url

    def stop(self) -> None:
        """Ask the server to exit and wait for its thread."""
        self._server.should_exit = True
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("the mock registry server did not stop within 10 seconds")


def start_mock_registry(
    run_id: str,
    dpip_key: str,
    reported: Sequence[tuple[str, str]],
    screen_delay: ScreenDelay | None,
) -> RunningMockRegistry:
    """
    WHAT: Start the real nationalsim app under uvicorn on 127.0.0.1 (free port) with the given
    (identifier_type, identifier_hash) entries already reported and visible.

    WHY: The rule is to benchmark against the real registry over real TCP, not httpx.ASGITransport
    and not an instant fake. The added /screen delay (for example the configured one) is attached only when given.
    """
    listing_delay, bank_settings = load_nationalsim_config()
    clock = ManualClock(datetime.now(IST))
    app = create_app(
        clock=clock,
        key_provider=lambda: dpip_key,
        settings={"listing_delay_minutes": int(listing_delay.total_seconds() // 60), "banks": bank_settings},
        screen_delay=screen_delay,
    )
    for number, (identifier_type, identifier_hash) in enumerate(reported):
        app.state.registry.ingest(
            bank_id="bank_0",
            report_id=f"bench-{number}",
            identifier_hash=identifier_hash,
            identifier_type=identifier_type,
            reason_code="fraud_confirmed",
            reported_at=clock.now_ist() - timedelta(days=1),
            run_id=run_id,
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    # Clients abandon calls that time out, so the server must not wait forever for those connections to close.
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the mock registry server did not start within 10 seconds")
        time.sleep(0.01)
    return RunningMockRegistry(server, thread, f"http://127.0.0.1:{port}")


def benchmark_registry_entries(run_id: str, payee_vpa: str) -> list[tuple[str, str]]:
    """The (identifier_type, identifier_hash) entries the mock registry reports: the payee and the payer's device."""
    return [
        (
            "account",
            compute_identifier_hash(
                key=BENCHMARK_DPIP_KEY,
                run_id=run_id,
                identifier_type="account",
                raw_value=payee_vpa,
                bank_index=get_bank_index(BENCHMARK_PAYEE_BANK_ID),
            ),
        ),
        (
            "device",
            compute_identifier_hash(
                key=BENCHMARK_DPIP_KEY,
                run_id=run_id,
                identifier_type="device",
                raw_value=BENCHMARK_PAYER_DEVICE,
                bank_index=None,
            ),
        ),
    ]


async def measure_against_mock_registry(
    connection: asyncpg.Connection,
    redis_client: redis.Redis,
    bank_id: str,
    run_id: str,
    payer_acc: str,
    payee_vpa: str,
    decision_time: datetime,
    delay_on: bool,
    warmup_count: int = 5,
    timed_count: int = 300,
) -> LatencyBenchmarkResult:
    """
    WHAT: measure_decision_latency with the real NationalsimBankClient pointed at the mock registry over HTTP.
    The registry holds one reported match for the payee account and one for the payer's device.
    """
    watermark = get_bank_watermark(redis_client, bank_id)
    if watermark is None:
        raise RuntimeError("the seeded history left no bank watermark, so there is nothing to benchmark")
    questions = list_registry_questions(
        redis_client,
        run_id,
        bank_id,
        payer_acc,
        BENCHMARK_PAYEE_BANK_ID,
        payee_vpa,
        decision_time,
        watermark,
        BENCHMARK_DPIP_KEY,
    )

    screen_delay = load_screen_delay_settings().build() if delay_on else None
    registry_server = start_mock_registry(
        run_id, BENCHMARK_DPIP_KEY, benchmark_registry_entries(run_id, payee_vpa), screen_delay
    )
    try:
        client = NationalsimBankClient(registry_server.base_url, bank_id=bank_id, run_id=run_id)
        return await measure_decision_latency(
            connection=connection,
            redis_client=redis_client,
            bank_id=bank_id,
            run_id=run_id,
            payer_acc=payer_acc,
            payee_vpa=payee_vpa,
            decision_time=decision_time,
            screener=NationalsimRegistryScreener(client),
            warmup_count=warmup_count,
            timed_count=timed_count,
            dpip_key=BENCHMARK_DPIP_KEY,
            registry_mode="http",
            registry_delay="on" if delay_on else "off",
            registry_questions=len(questions),
        )
    finally:
        registry_server.stop()


async def run_benchmark(
    warmup: int = 5,
    timed: int = 300,
    output_path: Path | None = None,
    registry: str = "fake",
    registry_delay: str = "off",
) -> LatencyBenchmarkResult:
    """Orchestrates benchmark environment setup, measurement, and reporting."""
    if registry not in ("fake", "http"):
        raise ValueError(f"registry must be 'fake' or 'http', got {registry!r}")
    if registry_delay not in ("off", "on"):
        raise ValueError(f"registry_delay must be 'off' or 'on', got {registry_delay!r}")
    if registry == "fake" and registry_delay == "on":
        raise ValueError("--registry-delay on needs --registry http (the fake registry answers instantly)")
    # Imported here because the showcase support module imports this module's measurement code at top level.
    from showcase.isolation import cleanup_workspace, create_workspace
    from showcase.level_15_support_c import setup_part_c_history

    workspace = await create_workspace()
    try:
        fixture = await setup_part_c_history(workspace.connection)
        try:
            if registry == "http":
                res = await measure_against_mock_registry(
                    connection=fixture.connection,
                    redis_client=fixture.live_redis,
                    bank_id=fixture.bank_id,
                    run_id=fixture.run_id,
                    payer_acc=fixture.payer_acc,
                    payee_vpa=fixture.payee_vpa,
                    decision_time=fixture.decision_time,
                    delay_on=registry_delay == "on",
                    warmup_count=warmup,
                    timed_count=timed,
                )
            else:
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
        "registry": {
            "mode": res.registry_mode,
            "delay": res.registry_delay,
            "questions_per_decision": res.registry_questions,
            "timeouts_in_timed_calls": res.registry_timeouts,
        },
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
        default=300,
        help="Number of timed measurement calls (default: 300).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_REPORT_PATH,
        help=f"Path to write JSON benchmark report (default: {OUTPUT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--registry",
        choices=("fake", "http"),
        default="fake",
        help="fake: the instant in-memory screener (default). http: the real mock registry over real HTTP.",
    )
    parser.add_argument(
        "--registry-delay",
        choices=("off", "on"),
        default="off",
        help="With --registry http: add the configured seeded delay to every /screen answer (default: off).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced call count for rapid verification (warmup=2, timed=5).",
    )
    args = parser.parse_args()

    warmup_count = 2 if args.quick else args.warmup
    timed_count = 5 if args.quick else args.timed

    result = asyncio.run(
        run_benchmark(
            warmup=warmup_count,
            timed=timed_count,
            output_path=args.output,
            registry=args.registry,
            registry_delay=args.registry_delay,
        )
    )
    print(format_benchmark_output(result))


if __name__ == "__main__":
    main()
