# File: benchmarks/decision_latency.py
# Purpose: Decision latency benchmark by the pre-registered method (warm-up, runs, median of run p99s).
# Issue: Level 15 decision latency (final measurement round).
"""
WHAT: Times server-side POST /v1/decisions work (record_decision) with feature logging on, against a seeded
history in PostgreSQL and Redis and, for mode http, the real mock national registry over HTTP through one shared
connection pool. Each decision is split into registry wait, feature compute (our own code), the feature_log write
and the rest. The pre-registered method (config/evaluation.yaml latency_method): 100 warm-up calls, 1,000 measured
calls per run, 3 runs, judged on the MEDIAN of the three p99 values.

WHY: The latency budget (config/evaluation.yaml) is: whole decision p99 <= max_ms (100 ms, registry wait included)
and feature computation p99 <= feature_computation_max_ms (50 ms, our own feature code only). The budget is judged
where the service runs: in the project's Linux API container (--where container). Running here (--where here) on
the Windows host is a harness check, not the verdict. HTTP serialization to the bank and request signing are NOT
timed. Timings use time.perf_counter() inside the process that runs the decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import asyncpg
import httpx
import numpy as np
import orjson
import redis
import uvicorn
import yaml
from prometheus_client import REGISTRY

import api.v1.bank_service as bank_service
from api.v1.bank_service import record_decision
from api.v1.models import DecisionRequest
from core.config_validator import load_yaml
from core.event_clock import IST, ManualClock
from features.ingest_watermark import get_bank_watermark
from features.m1_intelligence import get_bank_index
from features.registry_prefetch import (
    RegistryGateway,
    compute_registry_timeout_ms,
    list_registry_questions,
    load_registry_timeout_seconds,
)
from features.registry_screening import RegistryScreener
from nationalsim.app import ScreenDelay, create_app, load_nationalsim_config, load_screen_delay_settings
from nationalsim.bank_client import create_shared_http_client, load_bank_client_config, load_registry_pool_settings
from nationalsim.keyed_hash import compute_identifier_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "evaluation.yaml"
CONFIG_ACCOUNT_FEATURES = REPO_ROOT / "config" / "account_features.yaml"
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


class RunningMockRegistry:
    """The mock national registry served by uvicorn on a free localhost port, in a thread of this process."""

    def __init__(
        self,
        server: uvicorn.Server,
        thread: threading.Thread,
        base_url: str,
        clock: ManualClock,
        listing_delay: timedelta,
    ) -> None:
        self._server = server
        self._thread = thread
        self.base_url = base_url
        self._clock = clock
        self._listing_delay = listing_delay

    def make_entries_visible(self) -> None:
        """Move this registry's own clock past the listing delay, so entries received so far are listed (matchable)."""
        self._clock.set_time(self._clock.now_ist() + self._listing_delay + timedelta(minutes=1))

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
    # The registry lists an entry only after its listing delay, measured on its own clock.

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
    running = RunningMockRegistry(server, thread, f"http://127.0.0.1:{port}", clock, listing_delay)
    if reported:
        running.make_entries_visible()
    return running


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


RUNTIME_ENV = "ASHFALL_BENCH_RUNTIME"
COMMIT_ENV = "BENCH_COMMIT"
CONTAINER_REGISTRY_URL = "http://nationalsim-bench:8080"
CONTAINER_OUTPUT_DIR = "/out"
DELAY_CHOICES = ("off", "default", "registry-slow")


def registry_timeouts_so_far() -> int:
    """Total of ashfall_registry_timeouts_total over both identifier types, read from the process registry."""
    total = 0.0
    for identifier_type in ("account", "device"):
        value = REGISTRY.get_sample_value("ashfall_registry_timeouts_total", {"identifier_type": identifier_type})
        total += 0.0 if value is None else value
    return int(total)


def runtime_label() -> str:
    """'container' when run by the compose benchmark service, else 'native'."""
    return os.environ.get(RUNTIME_ENV, "native")


# ---------------------------------------------------------------------------------------------------------------------
# What one decision spends its time on
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class DecisionSample:
    """One timed record_decision call and where its time went (milliseconds, measured with perf_counter)."""

    total_ms: float
    registry_wait_ms: float
    feature_compute_ms: float
    feature_log_write_ms: float

    @property
    def overhead_ms(self) -> float:
        """Everything in the decision except the feature computation and the registry wait."""
        return self.total_ms - self.registry_wait_ms - self.feature_compute_ms

    @property
    def other_ms(self) -> float:
        """Overhead without the feature_log write: request handling, thread hand-offs, the response."""
        return self.overhead_ms - self.feature_log_write_ms


class PartsRecorder:
    """
    WHAT: Times the parts of record_decision by wrapping the functions it calls (only while the context is open):
    registry questions listed + features computed (feature_compute), the registry wait, and the feature_log write.

    WHY: Measures in the harness, so nothing in the production path stores or computes these durations.
    """

    def __init__(self) -> None:
        self.registry_wait_s = 0.0
        self.feature_compute_s = 0.0
        self.feature_log_write_s = 0.0

    def reset(self) -> None:
        self.registry_wait_s = 0.0
        self.feature_compute_s = 0.0
        self.feature_log_write_s = 0.0

    def _sync(self, real: Callable[..., Any], attribute: str) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return real(*args, **kwargs)
            finally:
                setattr(self, attribute, getattr(self, attribute) + time.perf_counter() - started)

        return wrapper

    def _async(self, real: Callable[..., Awaitable[Any]], attribute: str) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return await real(*args, **kwargs)
            finally:
                setattr(self, attribute, getattr(self, attribute) + time.perf_counter() - started)

        return wrapper

    @contextlib.contextmanager
    def patched(self) -> Iterator[None]:
        """Wrap the four functions in api.v1.bank_service for the duration of the block."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    bank_service,
                    "prepare_registry_questions",
                    self._sync(bank_service.prepare_registry_questions, "feature_compute_s"),
                )
            )
            stack.enter_context(
                patch.object(
                    bank_service,
                    "compute_decision_row",
                    self._sync(bank_service.compute_decision_row, "feature_compute_s"),
                )
            )
            stack.enter_context(
                patch.object(
                    bank_service,
                    "prefetch_registry_answers",
                    self._async(bank_service.prefetch_registry_answers, "registry_wait_s"),
                )
            )
            stack.enter_context(
                patch.object(
                    bank_service,
                    "insert_feature_log_row",
                    self._async(bank_service.insert_feature_log_row, "feature_log_write_s"),
                )
            )
            yield

    def sample(self, total_s: float) -> DecisionSample:
        return DecisionSample(
            total_ms=total_s * 1000.0,
            registry_wait_ms=self.registry_wait_s * 1000.0,
            feature_compute_ms=self.feature_compute_s * 1000.0,
            feature_log_write_ms=self.feature_log_write_s * 1000.0,
        )


@dataclass
class RunResult:
    """One run: the timed samples summarised per part, the slowest calls and the registry timeouts."""

    index: int
    warmup_count: int
    timed_count: int
    decision: dict[str, float]
    feature_compute: dict[str, float]
    registry_wait: dict[str, float]
    overhead: dict[str, float]
    registry_timeouts: int
    slowest: list[DecisionSample]
    decision_histogram: dict[str, int]
    decision_histogram_text: str


def summarise_run(
    index: int, warmup_count: int, samples: Sequence[DecisionSample], registry_timeouts: int, slowest_count: int
) -> RunResult:
    """Reduce a run's samples to percentiles per part plus its slowest `slowest_count` calls."""
    histogram, histogram_text = build_histogram([s.total_ms for s in samples])
    return RunResult(
        index=index,
        warmup_count=warmup_count,
        timed_count=len(samples),
        decision=compute_percentiles([s.total_ms for s in samples]),
        feature_compute=compute_percentiles([s.feature_compute_ms for s in samples]),
        registry_wait=compute_percentiles([s.registry_wait_ms for s in samples]),
        overhead=compute_percentiles([s.overhead_ms for s in samples]),
        registry_timeouts=registry_timeouts,
        slowest=sorted(samples, key=lambda s: s.total_ms, reverse=True)[:slowest_count],
        decision_histogram=histogram,
        decision_histogram_text=histogram_text,
    )


async def measure_run(
    *,
    connection: asyncpg.Connection,
    redis_client: redis.Redis,
    bank_id: str,
    run_id: str,
    payer_acc: str,
    payee_vpa: str,
    decision_time: datetime,
    screener: RegistryScreener | None,
    gateway: RegistryGateway | None,
    dpip_key: str | None,
    warmup_count: int,
    timed_count: int,
    slowest_count: int,
    tag: str,
    index: int,
) -> RunResult:
    """
    WHAT: One run of the method: `warmup_count` discarded record_decision calls, then `timed_count` timed ones, each
    split into registry wait / feature compute / feature_log write / the rest. Payment ids carry `tag`, so runs in
    the same history never collide.
    """
    clock = ManualClock(decision_time)
    recorder = PartsRecorder()

    def request(phase: str, number: int, amount_paise: int) -> DecisionRequest:
        return DecisionRequest(
            payment_id=f"{run_id}-{phase}-{tag}-{number:05d}",
            sender_account_id=payer_acc,
            receiver_bank_id=BENCHMARK_PAYEE_BANK_ID,
            receiver_vpa=payee_vpa,
            payee_type="person",
            amount_paise=amount_paise,
            event_time=decision_time,
        )

    async def decide(req: DecisionRequest) -> None:
        await record_decision(
            connection=connection,
            bank_id=bank_id,
            request=req,
            redis_client=redis_client,
            decision_clock=clock,
            screener=screener,
            dpip_key=dpip_key,
            registry=gateway,
        )

    samples: list[DecisionSample] = []
    with recorder.patched():
        for i in range(warmup_count):
            await decide(request("warm", i, 10000))
        timeouts_before = registry_timeouts_so_far()
        for i in range(timed_count):
            recorder.reset()
            req = request("timed", i, 15000 + i * 100)
            started = time.perf_counter()
            await decide(req)
            samples.append(recorder.sample(time.perf_counter() - started))
        timeouts = registry_timeouts_so_far() - timeouts_before
    return summarise_run(index, warmup_count, samples, timeouts, slowest_count)


# ---------------------------------------------------------------------------------------------------------------------
# The method: runs, medians, verdict, conditions
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class MethodResult:
    """All runs of one mode plus the medians the verdict uses."""

    mode: str
    delay: str
    runtime: str
    runs: list[RunResult]
    questions_per_decision: int
    budget: dict[str, Any]
    method: dict[str, Any]
    conditions: dict[str, str]
    per_group: list[dict[str, Any]] = field(default_factory=list)

    def run_p99s(self, part: str) -> list[float]:
        return [getattr(run, part)["p99"] for run in self.runs]

    def median_p99(self, part: str) -> float:
        return median_of_run_values(self.run_p99s(part))

    @property
    def decision_passes(self) -> bool:
        return self.median_p99("decision") <= self.budget["max_ms"]

    @property
    def feature_passes(self) -> bool:
        return self.median_p99("feature_compute") <= self.budget["feature_computation_max_ms"]

    @property
    def timeout_rate(self) -> float:
        asked = sum(run.timed_count for run in self.runs) * self.questions_per_decision
        return sum(run.registry_timeouts for run in self.runs) / asked if asked else 0.0


def _read_command(args: Sequence[str]) -> str:
    """Output of a short host command, or a plain statement that it could not be read (never an empty guess)."""
    if shutil.which(args[0]) is None:
        return f"not detectable: {args[0]} not found"
    completed = subprocess.run(list(args), capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode != 0:
        return f"not detectable: {args[0]} exited with code {completed.returncode}"
    return completed.stdout.strip()


def _windows_power_source() -> str:
    """'plugged in' or 'on battery' from the Windows power status, else a statement that it is not detectable."""

    class PowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
        return "not detectable: GetSystemPowerStatus failed"
    return {0: "on battery", 1: "plugged in"}.get(status.ACLineStatus, "not detectable: unknown AC line status")


def collect_conditions() -> dict[str, str]:
    """The conditions a result was measured under (platform, Python, commit, power, Docker, clock)."""
    runtime = runtime_label()
    clock = time.get_clock_info("monotonic")
    conditions = {
        "runtime": runtime,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "commit": os.environ.get(COMMIT_ENV) or _read_command(["git", "rev-parse", "HEAD"]),
        "monotonic_clock": f"{clock.implementation}, resolution {clock.resolution * 1000.0:.6f} ms",
    }
    if runtime == "container":
        conditions["power"] = "not detectable from inside the container (host laptop)"
        conditions["power_plan"] = "not detectable from inside the container (host laptop)"
        conditions["docker"] = "running (this is a compose container)"
    else:
        conditions["power"] = _windows_power_source() if sys.platform == "win32" else "not detectable on this OS"
        conditions["power_plan"] = (
            _read_command(["powercfg", "/getactivescheme"]) if sys.platform == "win32" else "not applicable"
        )
        conditions["docker"] = (
            "running"
            if _read_command(["docker", "info", "--format", "{{.ServerVersion}}"])[:1].isdigit()
            else "not running or not detectable"
        )
    return conditions


def seed_registry_over_http(base_url: str, run_id: str, entries: Sequence[tuple[str, str]]) -> None:
    """Report the (identifier_type, identifier_hash) entries to a running mock registry through POST /ingest."""
    reported_at = (datetime.now(IST) - timedelta(days=1)).isoformat()
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        for number, (identifier_type, identifier_hash) in enumerate(entries):
            response = client.post(
                "/ingest",
                json={
                    "bank_id": "bank_0",
                    "report_id": f"bench-{run_id}-{number}",
                    "identifier_hash": identifier_hash,
                    "identifier_type": identifier_type,
                    "reason_code": "fraud_confirmed",
                    "reported_at": reported_at,
                    "run_id": run_id,
                },
            )
            response.raise_for_status()


async def run_method(
    *,
    fixture: Any,
    mode: str,
    delay: str,
    registry_url: str | None,
    warmup_count: int | None = None,
    timed_count: int | None = None,
    runs: int | None = None,
    include_per_group: bool = True,
) -> MethodResult:
    """
    WHAT: The pre-registered latency method (config/evaluation.yaml latency_method) for one mode: warm-up calls
    discarded, `measured_calls` timed calls per run, `runs` runs; per-part percentiles for every run.

    mode "fake": the instant in-memory screener. mode "http": the real mock registry over HTTP, either started in
    this process (registry_url None) or an already running service at registry_url (seeded through /ingest, with
    its own configured delay). Decisions use one shared connection pool (RegistryGateway), as the API does.
    """
    if mode not in ("fake", "http"):
        raise ValueError(f"mode must be 'fake' or 'http', got {mode!r}")
    if delay not in DELAY_CHOICES:
        raise ValueError(f"delay must be one of {DELAY_CHOICES}, got {delay!r}")
    if mode == "fake" and delay != "off":
        raise ValueError("a registry delay needs mode 'http' (the fake registry answers instantly)")
    method = load_latency_method()
    budget = load_latency_budget()
    warmup = method["warm_up_calls"] if warmup_count is None else warmup_count
    timed = method["measured_calls"] if timed_count is None else timed_count
    run_count = method["runs"] if runs is None else runs

    gateway: RegistryGateway | None = None
    local_registry: RunningMockRegistry | None = None
    questions = 0
    try:
        if mode == "http":
            entries = benchmark_registry_entries(fixture.run_id, fixture.payee_vpa)
            if registry_url is None:
                screen_delay = None if delay == "off" else load_screen_delay_settings(profile=delay).build()
                local_registry = start_mock_registry(fixture.run_id, BENCHMARK_DPIP_KEY, entries, screen_delay)
                base_url = local_registry.base_url
            else:
                seed_registry_over_http(registry_url, fixture.run_id, entries)
                base_url = registry_url
            _, max_connections, max_keepalive = load_registry_pool_settings()
            client_timeout, retry_backoff = load_bank_client_config()
            gateway = RegistryGateway(
                http_client=create_shared_http_client(max_connections, max_keepalive, client_timeout),
                base_url=base_url,
                client_timeout_seconds=client_timeout,
                retry_backoff_seconds=retry_backoff,
                registry_timeout_seconds=load_registry_timeout_seconds(),
            )
            watermark = get_bank_watermark(fixture.live_redis, fixture.bank_id)
            if watermark is None:
                raise RuntimeError("the seeded history left no bank watermark, so there is nothing to benchmark")
            questions = len(
                list_registry_questions(
                    fixture.live_redis,
                    fixture.run_id,
                    fixture.bank_id,
                    fixture.payer_acc,
                    BENCHMARK_PAYEE_BANK_ID,
                    fixture.payee_vpa,
                    fixture.decision_time,
                    watermark,
                    BENCHMARK_DPIP_KEY,
                )
            )

        run_results: list[RunResult] = []
        for index in range(1, run_count + 1):
            run_results.append(
                await measure_run(
                    connection=fixture.connection,
                    redis_client=fixture.live_redis,
                    bank_id=fixture.bank_id,
                    run_id=fixture.run_id,
                    payer_acc=fixture.payer_acc,
                    payee_vpa=fixture.payee_vpa,
                    decision_time=fixture.decision_time,
                    screener=fixture.screener if mode == "fake" else None,
                    gateway=gateway,
                    dpip_key=BENCHMARK_DPIP_KEY if mode == "http" else None,
                    warmup_count=warmup,
                    timed_count=timed,
                    slowest_count=method["slowest_calls_reported"],
                    tag=f"{mode}-{delay}-r{index}",
                    index=index,
                )
            )
    finally:
        if gateway is not None:
            await gateway.aclose()
        if local_registry is not None:
            local_registry.stop()

    result = MethodResult(
        mode=mode,
        delay=delay,
        runtime=runtime_label(),
        runs=run_results,
        questions_per_decision=questions,
        budget=budget,
        method=method,
        conditions=collect_conditions(),
    )
    if include_per_group:
        result.per_group = await asyncio.to_thread(per_group_feature_times, fixture)
    return result


def per_group_feature_times(fixture: Any) -> list[dict[str, Any]]:
    """Per-group feature times and Redis round trips (the profile script's measurement), on this fixture."""
    from benchmarks.profile_decision_features import profile_decisions, summarise

    samples, _ = profile_decisions(fixture, decisions=50, warmup=5, screener=fixture.screener)
    return summarise(samples)


# ---------------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------------


def _spread(values: Sequence[float]) -> str:
    return f"{min(values):.1f} to {max(values):.1f} ms (spread {max(values) - min(values):.1f})"


def format_method_report(res: MethodResult) -> str:
    """Plain-text report: conditions, per-run p99s with medians, timeouts, slowest calls, per-group times, formula."""
    budget = res.budget
    lines = [
        "=" * 76,
        f"LEVEL 15 DECISION LATENCY: mode {res.mode}, registry delay {res.delay}, runtime {res.runtime}",
        "=" * 76,
    ]
    lines += [f"  {key:<16} {value}" for key, value in res.conditions.items()]
    method = res.method
    first = res.runs[0]
    lines += [
        f"  method           {method['warm_up_calls']} warm-up, {method['measured_calls']} measured calls x "
        f"{method['runs']} runs, judged on {method['judged_on']} "
        f"(this report: {first.warmup_count} warm-up, {first.timed_count} measured, {len(res.runs)} runs)",
        f"  registry         questions per decision {res.questions_per_decision}",
        "",
        f"{'p99 ms':<22}" + "".join(f"{'run ' + str(r.index):>10}" for r in res.runs) + f"{'median':>10}   spread",
    ]
    for label, part in (
        ("decision", "decision"),
        ("feature compute", "feature_compute"),
        ("registry wait", "registry_wait"),
        ("overhead (other)", "overhead"),
    ):
        values = res.run_p99s(part)
        lines.append(
            f"{label:<22}"
            + "".join(f"{v:>10.2f}" for v in values)
            + f"{res.median_p99(part):>10.2f}   {_spread(values)}"
        )
    lines += [
        "",
        f"  decision median p99 {res.median_p99('decision'):.2f} ms vs budget {budget['max_ms']:.0f} ms: "
        f"{'PASS' if res.decision_passes else 'FAIL'}",
        f"  feature compute median p99 {res.median_p99('feature_compute'):.2f} ms vs budget "
        f"{budget['feature_computation_max_ms']:.0f} ms: {'PASS' if res.feature_passes else 'FAIL'}",
        f"  registry timeouts per run: {[r.registry_timeouts for r in res.runs]} "
        f"(rate {res.timeout_rate * 100:.1f}% of {res.questions_per_decision} questions per decision)",
    ]
    for run in res.runs:
        lines += [
            "",
            f"Run {run.index}: the {len(run.slowest)} slowest calls (ms): total = registry wait + feature compute + "
            "feature_log write + other",
            f"  {'total':>8}{'registry':>10}{'features':>10}{'log write':>11}{'other':>9}",
        ]
        for s in run.slowest:
            lines.append(
                f"  {s.total_ms:>8.1f}{s.registry_wait_ms:>10.1f}{s.feature_compute_ms:>10.1f}"
                f"{s.feature_log_write_ms:>11.1f}{s.other_ms:>9.1f}"
            )
    if res.per_group:
        lines += ["", "Per-group feature time (50 decisions, each row includes its own snapshot read):"]
        lines.append(f"  {'item':<26}{'p50 ms':>9}{'p99 ms':>9}{'trips p50':>11}")
        for row in res.per_group:
            lines.append(
                f"  {row['item']:<26}{row['p50_ms']:>9.2f}{row['p99_ms']:>9.2f}{row['round_trips_p50']:>11.0f}"
            )
    lines += ["", formula_line(res)]
    return "\n".join(lines)


def formula_line(res: MethodResult) -> str:
    """The A2 registry-timeout formula with this result's measured overhead p99 and this runtime's clock resolution."""
    config_block = load_yaml(CONFIG_ACCOUNT_FEATURES)["decision_registry"]
    margin = float(config_block["timeout_margin_ms"]["value"])
    minimum = float(config_block["timeout_minimum_ms"]["value"])
    overhead_p99 = res.median_p99("overhead")
    resolution_ms = time.get_clock_info("monotonic").resolution * 1000.0
    inputs = (
        f"max_ms {res.budget['max_ms']:.0f} - feature limit {res.budget['feature_computation_max_ms']:.0f} "
        f"- overhead p99 {overhead_p99:.3f} - margin {margin:.0f} - clock resolution {resolution_ms:.6f}"
    )
    try:
        value = compute_registry_timeout_ms(
            res.budget["max_ms"],
            res.budget["feature_computation_max_ms"],
            overhead_p99,
            margin,
            resolution_ms,
            minimum,
        )
    except ValueError as err:
        return f"Registry timeout formula: {inputs} -> STOP: {err}"
    return f"Registry timeout formula: {inputs} = {value:.3f} ms"


def method_result_to_dict(res: MethodResult) -> dict[str, Any]:
    """JSON-ready result: conditions, medians (the verdict inputs), every run, slowest calls and per-group times."""
    return {
        "mode": res.mode,
        "delay": res.delay,
        "runtime": res.runtime,
        "conditions": res.conditions,
        "method": res.method,
        "budget": res.budget,
        "questions_per_decision": res.questions_per_decision,
        "medians_of_run_p99_ms": {
            "decision": res.median_p99("decision"),
            "feature_compute": res.median_p99("feature_compute"),
            "registry_wait": res.median_p99("registry_wait"),
            "overhead": res.median_p99("overhead"),
        },
        "run_p99_ms": {
            part: res.run_p99s(part) for part in ("decision", "feature_compute", "registry_wait", "overhead")
        },
        "decision_passes": res.decision_passes,
        "feature_passes": res.feature_passes,
        "timeout_rate": res.timeout_rate,
        "runs": [
            {
                "index": run.index,
                "warmup_count": run.warmup_count,
                "timed_count": run.timed_count,
                "decision_ms": run.decision,
                "feature_compute_ms": run.feature_compute,
                "registry_wait_ms": run.registry_wait,
                "overhead_ms": run.overhead,
                "registry_timeouts": run.registry_timeouts,
                "decision_histogram": run.decision_histogram,
                "slowest": [
                    {
                        "total_ms": s.total_ms,
                        "registry_wait_ms": s.registry_wait_ms,
                        "feature_compute_ms": s.feature_compute_ms,
                        "feature_log_write_ms": s.feature_log_write_ms,
                        "other_ms": s.other_ms,
                    }
                    for s in run.slowest
                ],
            }
            for run in res.runs
        ],
        "per_group": res.per_group,
        "formula": formula_line(res),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_method_report(res: MethodResult, out_file: Path) -> None:
    """Write the method result as JSON."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(orjson.dumps(method_result_to_dict(res), option=orjson.OPT_INDENT_2))


# ---------------------------------------------------------------------------------------------------------------------
# Running in the Linux container (where the service runs) and running here
# ---------------------------------------------------------------------------------------------------------------------


def run_method_in_container(
    registry: str,
    delay: str,
    *,
    warmup: int | None = None,
    timed: int | None = None,
    runs: int | None = None,
) -> dict[str, Any]:
    """
    WHAT: Run this benchmark inside the compose `bench` service (the project's API image, repo mounted read-only) on
    the compose network, against postgres, redis and redpanda by service name and, for http, the compose
    `nationalsim-bench` registry started with the chosen delay. Returns the JSON result the container wrote.

    WHY: The latency budget is judged where the service runs, on Linux, not on the Windows host.
    """
    if delay not in DELAY_CHOICES:
        raise ValueError(f"delay must be one of {DELAY_CHOICES}, got {delay!r}")
    env = os.environ.copy()
    env[COMMIT_ENV] = _read_command(["git", "rev-parse", "HEAD"])
    env["NATIONALSIM_SCREEN_DELAY"] = delay
    compose = ["docker", "compose", "--profile", "benchmark"]
    if registry == "http":
        subprocess.run(
            [*compose, "up", "-d", "--wait", "--force-recreate", "nationalsim-bench"],
            env=env,
            check=True,
            cwd=REPO_ROOT,
        )
    name = f"container_{registry}_{delay}.json"
    command = [
        *compose,
        "run",
        "--rm",
        "bench",
        "python",
        "-m",
        "benchmarks.decision_latency",
        "--where",
        "here",
        "--registry",
        registry,
        "--registry-delay",
        delay,
        "--output",
        f"{CONTAINER_OUTPUT_DIR}/{name}",
    ]
    if registry == "http":
        command += ["--registry-url", CONTAINER_REGISTRY_URL]
    for flag, value in (("--warmup", warmup), ("--timed", timed), ("--runs", runs)):
        if value is not None:
            command += [flag, str(value)]
    subprocess.run(command, env=env, check=True, cwd=REPO_ROOT)
    result = orjson.loads((OUTPUT_REPORT_PATH.parent / name).read_bytes())
    if not isinstance(result, dict):
        raise ValueError(f"the container result {name} is not a JSON object")
    return result


async def run_benchmark(
    *,
    mode: str,
    delay: str,
    registry_url: str | None,
    output_path: Path,
    warmup: int | None,
    timed: int | None,
    runs: int | None,
) -> MethodResult:
    """Build the seeded history in this process's environment, run the method, write the JSON report."""
    # Imported here because the showcase support module imports this module's measurement code at top level.
    from showcase.isolation import cleanup_workspace, create_workspace
    from showcase.level_15_support_c import setup_part_c_history

    workspace = await create_workspace()
    try:
        fixture = await setup_part_c_history(workspace.connection)
        try:
            result = await run_method(
                fixture=fixture,
                mode=mode,
                delay=delay,
                registry_url=registry_url,
                warmup_count=warmup,
                timed_count=timed,
                runs=runs,
            )
            write_method_report(result, output_path)
            return result
        finally:
            await fixture.cleanup()
    finally:
        await cleanup_workspace(workspace)


def main() -> None:
    """CLI entrypoint for the decision latency benchmark (the pre-registered method)."""
    parser = argparse.ArgumentParser(description="Decision latency benchmark by the pre-registered method.")
    parser.add_argument(
        "--where",
        choices=("here", "container"),
        default="here",
        help="here: this process; container: compose bench service.",
    )
    parser.add_argument(
        "--registry", choices=("fake", "http"), default="fake", help="fake: instant screener; http: real mock registry."
    )
    parser.add_argument(
        "--registry-delay",
        choices=DELAY_CHOICES,
        default="off",
        help="With --registry http: off, the default delay range, or the registry-slow stress range.",
    )
    parser.add_argument(
        "--registry-url", default=None, help="Use an already running mock registry at this URL (seeded via /ingest)."
    )
    parser.add_argument(
        "--warmup", type=int, default=None, help="Warm-up calls per run (default: from config/evaluation.yaml)."
    )
    parser.add_argument(
        "--timed", "-n", type=int, default=None, help="Measured calls per run (default: from config/evaluation.yaml)."
    )
    parser.add_argument("--runs", type=int, default=None, help="Number of runs (default: from config/evaluation.yaml).")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_REPORT_PATH, help=f"JSON report path (default: {OUTPUT_REPORT_PATH})."
    )
    parser.add_argument("--quick", action="store_true", help="Quick check: 2 warm-up calls, 5 measured calls, 1 run.")
    args = parser.parse_args()

    warmup, timed, runs = args.warmup, args.timed, args.runs
    if args.quick:
        warmup, timed, runs = 2, 5, 1

    if args.where == "container":
        result_dict = run_method_in_container(args.registry, args.registry_delay, warmup=warmup, timed=timed, runs=runs)
        medians = result_dict["medians_of_run_p99_ms"]
        print(
            f"Container result ({result_dict['conditions']['platform']}): decision median p99 "
            f"{medians['decision']:.2f} ms, feature compute median p99 {medians['feature_compute']:.2f} ms, "
            f"timeout rate {result_dict['timeout_rate'] * 100:.1f}%"
        )
        return

    result = asyncio.run(
        run_benchmark(
            mode=args.registry,
            delay=args.registry_delay,
            registry_url=args.registry_url,
            output_path=args.output,
            warmup=warmup,
            timed=timed,
            runs=runs,
        )
    )
    print(format_method_report(result))


if __name__ == "__main__":
    main()
