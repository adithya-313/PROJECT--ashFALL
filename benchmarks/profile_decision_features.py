# File: benchmarks/profile_decision_features.py
# Purpose: Per-group latency and Redis round-trip profile of the decision feature path.
# Issue: Level 15 decision latency profile (find where the time in compute_decision_row goes).
"""
WHAT: Times each M1 account feature group, the payment features and read_account_snapshot separately,
counts the Redis round trips each one makes, and explains the network group (BFS expansions, repeated
same_bank_counterparties calls, labelled accounts read). Also measures one plain Redis PING.

WHY: compute_decision_row misses its feature budget by a wide margin. Before anyone changes production code,
this shows which group costs the time and whether the cost is the NUMBER of Redis round trips (a network
cost) or the work done per call. Nothing under features/ or api/ is changed: counting happens in a proxy
around the Redis client and in monkeypatches that exist only while this script runs.
"""

import argparse
import asyncio
import platform
import sys
import time
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import orjson  # noqa: E402
import redis  # noqa: E402

import features.m1_network as m1_network  # noqa: E402
from benchmarks.decision_latency import compute_percentiles  # noqa: E402
from features.account_keys import AccountFeaturesSettings, get_default_settings  # noqa: E402
from features.account_reader import read_account_snapshot  # noqa: E402
from features.ingest_watermark import get_bank_watermark  # noqa: E402
from features.m1_features import compute_account_features  # noqa: E402
from features.m1_payment import ProposedPayment, compute_payment_features  # noqa: E402
from features.registry_screening import RegistryScreener  # noqa: E402
from showcase.isolation import cleanup_workspace, create_workspace  # noqa: E402
from showcase.level_15_support_c import PartCFixture, setup_part_c_history  # noqa: E402

OUTPUT_REPORT_PATH = REPO_ROOT / "reports" / "latency" / "level_15_decision_profile.json"

ACCOUNT_GROUPS = (
    "inflow",
    "outflow",
    "balance_pattern",
    "routine_life",
    "lifecycle",
    "control",
    "region",
    "network",
    "intelligence",
)
PING_SAMPLES = 200


class CountingPipeline:
    """Wraps a real pipeline. Buffering commands costs nothing; execute() is ONE round trip."""

    def __init__(self, pipeline: Any, owner: "CountingRedis") -> None:
        self._pipeline = pipeline
        self._owner = owner

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Count one round trip holding every buffered command, then send them."""
        self._owner.round_trips += 1
        self._owner.pipeline_executes += 1
        self._owner.pipeline_commands += len(self._pipeline.command_stack)
        return self._pipeline.execute(*args, **kwargs)

    def __enter__(self) -> "CountingPipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._pipeline.reset()

    def __getattr__(self, name: str) -> Any:
        # Command methods (get, zadd, ...) only buffer on a pipeline, so they pass straight through.
        return getattr(self._pipeline, name)


class CountingRedis:
    """
    Proxy around a real Redis client that counts round trips and forwards every call unchanged.

    Command methods are taken from the client's own class and bound to this proxy, so the client's internal
    calls (for example scan_iter -> scan -> execute_command) pass through execute_command here and are counted.
    A pipeline counts as one round trip however many commands it holds.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.round_trips = 0
        self.pipeline_executes = 0
        self.pipeline_commands = 0

    def reset(self) -> None:
        """Zero the counters before measuring the next item."""
        self.round_trips = 0
        self.pipeline_executes = 0
        self.pipeline_commands = 0

    def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        """Count one plain round trip and forward it."""
        self.round_trips += 1
        return self._client.execute_command(*args, **kwargs)

    def pipeline(self, *args: Any, **kwargs: Any) -> CountingPipeline:
        """Return a pipeline whose execute() is counted as one round trip."""
        return CountingPipeline(self._client.pipeline(*args, **kwargs), self)

    def __getattr__(self, name: str) -> Any:
        class_attribute = getattr(type(self._client), name, None)
        if isinstance(class_attribute, types.FunctionType):
            return types.MethodType(class_attribute, self)
        return getattr(self._client, name)


@dataclass
class ItemSamples:
    """Everything measured for one profiled item across all decisions."""

    name: str
    ms: list[float] = field(default_factory=list)
    round_trips: list[float] = field(default_factory=list)
    pipeline_executes: list[float] = field(default_factory=list)
    pipeline_commands: list[float] = field(default_factory=list)


@dataclass
class NetworkDetail:
    """What the network group did during one decision (counted in a separate, untimed pass)."""

    bfs_runs: int
    bfs_nodes_visited: int
    neighbour_calls: int
    repeat_neighbour_calls: int
    distinct_accounts_asked: int
    labelled_accounts_read: int


def measure_item(
    samples: ItemSamples, counter: CountingRedis, work: Callable[[], object], clock: Callable[[], float]
) -> None:
    """Run one item once: reset the counters, time the work, store the time and the round-trip counts."""
    counter.reset()
    started = clock()
    work()
    elapsed_ms = (clock() - started) * 1000.0
    samples.ms.append(elapsed_ms)
    samples.round_trips.append(float(counter.round_trips))
    samples.pipeline_executes.append(float(counter.pipeline_executes))
    samples.pipeline_commands.append(float(counter.pipeline_commands))


def measure_ping(client: redis.Redis, samples: int = PING_SAMPLES) -> dict[str, float]:
    """Time plain PING round trips on the unwrapped client (after a few untimed ones)."""
    for _ in range(10):
        client.ping()
    timings: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        client.ping()
        timings.append((time.perf_counter() - started) * 1000.0)
    return compute_percentiles(timings)


def measure_network_detail(
    counter: CountingRedis,
    fix: PartCFixture,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings,
) -> NetworkDetail:
    """
    Run the network group once with counting wrappers around its helpers (patched on m1_network only).

    Done in its own pass so the wrappers never slow the timed measurements.
    """
    real_neighbours = m1_network.same_bank_counterparties
    real_bfs = m1_network._compute_hops_to_target
    real_snapshot = m1_network.read_account_snapshot

    asked: list[str] = []
    bfs_seen: set[str] = set()
    state = {"bfs_runs": 0, "in_bfs": 0, "labelled_reads": 0}

    def counting_neighbours(client: Any, run_id: str, bank_id: str, account_id: str, *rest: Any) -> frozenset[str]:
        result = real_neighbours(client, run_id, bank_id, account_id, *rest)
        asked.append(account_id)
        if state["in_bfs"]:
            bfs_seen.add(account_id)
            bfs_seen.update(result)
        return result

    def counting_bfs(*args: Any, **kwargs: Any) -> int | None:
        state["bfs_runs"] += 1
        state["in_bfs"] += 1
        try:
            return real_bfs(*args, **kwargs)
        finally:
            state["in_bfs"] -= 1

    def counting_snapshot(*args: Any, **kwargs: Any) -> Any:
        state["labelled_reads"] += 1
        return real_snapshot(*args, **kwargs)

    with (
        patch.object(m1_network, "same_bank_counterparties", counting_neighbours),
        patch.object(m1_network, "_compute_hops_to_target", counting_bfs),
        patch.object(m1_network, "read_account_snapshot", counting_snapshot),
    ):
        compute_account_features(
            client=counter,
            run_id=fix.run_id,
            bank_id=fix.bank_id,
            account_id=fix.payer_acc,
            decision_time=fix.decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=settings,
            groups=("network",),
            screener=fix.screener,
        )

    distinct = len(set(asked))
    return NetworkDetail(
        bfs_runs=state["bfs_runs"],
        bfs_nodes_visited=len(bfs_seen),
        neighbour_calls=len(asked),
        repeat_neighbour_calls=len(asked) - distinct,
        distinct_accounts_asked=distinct,
        labelled_accounts_read=state["labelled_reads"],
    )


def profile_decisions(
    fix: PartCFixture,
    decisions: int,
    warmup: int,
    screener: RegistryScreener,
) -> tuple[list[ItemSamples], NetworkDetail]:
    """Profile every item for `decisions` decisions after `warmup` untimed rounds."""
    counter = CountingRedis(fix.live_redis)
    settings = get_default_settings()
    watermark = get_bank_watermark(counter, fix.bank_id)
    if watermark is None:
        raise RuntimeError("the seeded history left no bank watermark, so there is nothing to profile")
    arrival_cutoff: datetime = watermark

    items = {name: ItemSamples(name) for name in (*ACCOUNT_GROUPS, "payment_features", "read_account_snapshot")}

    def group_work(group: str) -> Callable[[], object]:
        """One group computed alone, at the same decision time and arrival cutoff as every other item."""
        return lambda: compute_account_features(
            client=counter,
            run_id=fix.run_id,
            bank_id=fix.bank_id,
            account_id=fix.payer_acc,
            decision_time=fix.decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=settings,
            groups=(group,),
            screener=screener,
        )

    def run_round(record: bool, index: int) -> None:
        # The proposed payment varies a little per decision, as real traffic would.
        proposed = ProposedPayment(
            run_id=fix.run_id,
            bank_id=fix.bank_id,
            payer_account_id=fix.payer_acc,
            payee_bank_id="bank_1",
            payee_vpa=fix.payee_vpa,
            payee_type="person",
            amount_paise=15000 + index * 100,
            event_time=fix.decision_time,
        )
        works: dict[str, Callable[[], object]] = {
            "payment_features": lambda: compute_payment_features(
                counter, proposed, arrival_cutoff, settings=settings, screener=screener
            ),
            "read_account_snapshot": lambda: read_account_snapshot(
                counter, fix.run_id, fix.bank_id, fix.payer_acc, fix.decision_time, arrival_cutoff, settings
            ),
        }
        for group in ACCOUNT_GROUPS:
            works[group] = group_work(group)
        for name, work in works.items():
            if record:
                measure_item(items[name], counter, work, time.perf_counter)
            else:
                work()

    for i in range(warmup):
        run_round(record=False, index=i)
    for i in range(decisions):
        run_round(record=True, index=i)

    detail = measure_network_detail(counter, fix, arrival_cutoff, settings)
    return list(items.values()), detail


def summarise(samples: Sequence[ItemSamples]) -> list[dict[str, Any]]:
    """Reduce each item to p50/p99 time and median round-trip counts, sorted by p99 descending."""
    rows: list[dict[str, Any]] = []
    for item in samples:
        timing = compute_percentiles(item.ms)
        rows.append(
            {
                "item": item.name,
                "p50_ms": timing["p50"],
                "p99_ms": timing["p99"],
                "round_trips_p50": compute_percentiles(item.round_trips)["p50"],
                "pipeline_executes_p50": compute_percentiles(item.pipeline_executes)["p50"],
                "pipeline_commands_p50": compute_percentiles(item.pipeline_commands)["p50"],
            }
        )
    return sorted(rows, key=lambda row: row["p99_ms"], reverse=True)


def format_report(
    rows: Sequence[dict[str, Any]],
    ping: dict[str, float],
    detail: NetworkDetail,
    decisions: int,
    warmup: int,
) -> str:
    """Plain-text table plus the PING line and the network detail."""
    lines = [
        f"Platform: {platform.platform()}   Python: {sys.version.split()[0]}",
        f"Decisions: {decisions} timed, {warmup} warm-up. Each group row includes its own read_account_snapshot.",
        "",
        f"{'item':<24}{'p50 ms':>10}{'p99 ms':>10}{'trips p50':>11}{'pipelines':>11}{'pipe cmds':>11}",
    ]
    for row in rows:
        lines.append(
            f"{row['item']:<24}{row['p50_ms']:>10.2f}{row['p99_ms']:>10.2f}{row['round_trips_p50']:>11.0f}"
            f"{row['pipeline_executes_p50']:>11.0f}{row['pipeline_commands_p50']:>11.0f}"
        )
    lines += [
        "",
        f"Redis PING round trip ({PING_SAMPLES} samples): p50 {ping['p50']:.3f} ms, p99 {ping['p99']:.3f} ms",
        "",
        "Network group, one decision:",
        f"  BFS runs: {detail.bfs_runs}, BFS nodes visited (start + neighbours seen): {detail.bfs_nodes_visited}",
        f"  same_bank_counterparties calls: {detail.neighbour_calls}, of which repeat calls "
        f"(account already asked): {detail.repeat_neighbour_calls}, distinct accounts asked: "
        f"{detail.distinct_accounts_asked}",
        f"  labelled accounts read by compute_shared_counterparties_with_flagged_7d: {detail.labelled_accounts_read}",
    ]
    return "\n".join(lines)


async def run_profile(decisions: int, warmup: int, output_path: Path) -> str:
    """Build the seeded history, profile it, write the JSON report and return the text report."""
    workspace = await create_workspace()
    try:
        fix = await setup_part_c_history(workspace.connection)
        try:
            # compute runs synchronously in this thread; only the history setup needed the event loop.
            samples, detail = profile_decisions(fix, decisions, warmup, fix.screener)
            ping = measure_ping(fix.live_redis)
        finally:
            await fix.cleanup()
    finally:
        await cleanup_workspace(workspace)

    rows = summarise(samples)
    report = format_report(rows, ping, detail, decisions, warmup)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "decisions": decisions,
        "warmup": warmup,
        "items": rows,
        "ping_ms": ping,
        "network_detail": detail.__dict__,
    }
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return report


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Per-group latency and Redis round-trip profile.")
    parser.add_argument("--decisions", "-n", type=int, default=50, help="Timed decisions (default: 50).")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed warm-up rounds (default: 5).")
    parser.add_argument("--output", type=Path, default=OUTPUT_REPORT_PATH, help="JSON report path.")
    args = parser.parse_args()
    print(asyncio.run(run_profile(args.decisions, args.warmup, args.output)))


if __name__ == "__main__":
    main()
