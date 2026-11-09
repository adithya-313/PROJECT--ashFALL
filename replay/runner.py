# File: replay/runner.py
# Purpose: Level 16b replay runner driving the full operational pipeline in world-time order.
# Issue: Level 16b replay runner execution (architect decisions C2, 16b Q1-Q3, P1, S1, S2, and 16b-3b ruling).
"""
WHAT: Executes historical simulation replay by driving the real operational pipeline in strictly-ordered
world-time steps across PostgreSQL, Redis, and Kafka.

WHY: Architect decisions C2, 16b Q1-Q3, P1, S1, S2, and Level 16b-3b Ruling:
Replay evaluation executes against real isolated infrastructure (dedicated Postgres schema, Redis database,
and single-partition Kafka topic) in causal world-time sequence. All pending outbox messages, Kafka streams,
and database pollers are fully drained and watermarked before evaluating each decision, while ensuring that
a payment's own sides are posted only after its decision.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import asyncpg
import orjson
import redis
from aiokafka import AIOKafkaConsumer, TopicPartition

from api.services.outbox_relay import count_waiting, relay_one_batch, start_producer
from api.v1.bank_service import InProcessBankClient
from core.config import config
from core.event_clock import ManualClock, WallClock
from features.account_keys import AccountFeaturesSettings, get_default_settings
from features.account_labels import apply_complaints_since, apply_feedback_since
from features.account_snapshots import select_daily_snapshot_accounts, write_account_snapshots
from features.account_stream import apply_account_events_since, apply_withdrawals_since, consume_available
from features.feature_log import canonical_feature_names
from features.ingest_watermark import get_payment_watermark, record_payment_watermark
from features.label_consumer import consume_complaint_labels_since
from replay.ordering import DecisionStep, FeedStep, build_replay_steps
from world.bank_connectors import BankTransport, MockBankConnectors, load_bank_modes
from world.bank_feed import BankFeed, resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.replay_recording import RecordedWorld
from world.replay_workspace import (
    ReplaySettings,
    ReplayWorkspace,
    allocate_workspace,
    drop_workspace,
    get_default_replay_settings,
    workspace_redis_client,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ReplayDrainTimeout",
    "ReplayDropContext",
    "ReplayHookContext",
    "ReplayPipelineError",
    "ReplayResult",
    "check_workspace_topic_partitions",
    "compute_feature_log_sha256",
    "day_boundaries_between",
    "drain",
    "idle_advance_target",
    "load_default_idle_partition_advance",
    "run_replay",
]


class ReplayDrainTimeout(Exception):
    """Raised when the drain loop exceeds the configured deadline without reaching convergence."""


class ReplayPipelineError(Exception):
    """Raised when an internal pipeline invariant is violated during replay."""


@dataclass(frozen=True)
class ReplayHookContext:
    """
    WHAT: Frozen context passed to test injection hooks prior to decision evaluation.
    WHY: Level 16b-3b Ruling: allows deterministic injection of transient pipeline faults
    (e.g. partition pause or watermark deletion) without ad-hoc inspection or mutation.
    """

    step_index: int
    decision_step: DecisionStep
    consumer: AIOKafkaConsumer
    workspace: ReplayWorkspace


@dataclass(frozen=True)
class ReplayDropContext:
    """
    WHAT: Context provided to before_drop_hook prior to isolated workspace teardown.
    WHY: Implements Decision D4. Exposes workspace credentials and logged payment IDs by bank
    for verification (e.g. point-in-time history recomputation).
    """

    workspace: ReplayWorkspace
    workspace_dsn: str
    logged_payment_ids_by_bank: dict[str, list[str]]


@dataclass(frozen=True)
class ReplayResult:
    """
    WHAT: Output metrics and verification metadata from a completed replay execution.
    WHY: Provides fail-closed observability across decision counts, feature row statuses,
    arrival cutoff lag, and drain convergence times.
    Level 16c dataset manifests must copy both registry_wired and registry_wired_reason fields.
    """

    decisions_sent: int
    feature_rows_by_status: dict[str, int]
    decisions_arrival_cutoff_behind: int
    max_arrival_cutoff_gap_seconds: float
    longest_drain_seconds: float
    decisions_before_first_payment: int
    unwatermarked_payment_ids: tuple[str, ...] = ()
    logged_features_by_payment_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_request_fields_by_payment_id: dict[str, list[str]] = field(default_factory=dict)
    registry_wired: bool = False
    registry_wired_reason: str = "no nationalsim screener or DPIP key in replay until 16d"
    feature_log_sha256: str = ""


def compute_feature_log_sha256(rows: Sequence[asyncpg.Record]) -> str:
    """
    WHAT: Computes canonical SHA-256 hash across workspace feature_log rows.
    WHY: Implements Decision D3. Rows are sorted by (bank_id, payment_id) and serialized
    as canonical JSON using all 16 deterministic schema columns. Excludes only created_at
    which is generated by clock_timestamp() (wall clock).
    """
    canonical_rows: list[dict[str, Any]] = []
    # Real columns from information_schema:
    # Included (16 deterministic columns):
    #   account_id, arrival_cutoff, bank_id, decided_at, decision_time, error_message,
    #   error_type, feature_names_sha256, feature_set_version, feature_values,
    #   missing_request_fields, payment_id, registry_answers, run_id, status,
    #   watermark_lag_ms
    # Excluded (1 real-clock column):
    #   created_at (DEFAULT clock_timestamp() - wall-clock real time)
    for r in rows:
        raw_vals = r["feature_values"]
        if isinstance(raw_vals, (str, bytes)):
            feature_vals = orjson.loads(raw_vals)
        else:
            feature_vals = raw_vals

        reg_raw = r["registry_answers"]
        if isinstance(reg_raw, (str, bytes)):
            reg_answers = orjson.loads(reg_raw)
        else:
            reg_answers = reg_raw

        row_dict: dict[str, Any] = {
            "account_id": str(r["account_id"]),
            "arrival_cutoff": r["arrival_cutoff"].isoformat() if r["arrival_cutoff"] is not None else None,
            "bank_id": str(r["bank_id"]),
            "decided_at": r["decided_at"].isoformat() if r["decided_at"] is not None else None,
            "decision_time": r["decision_time"].isoformat() if r["decision_time"] is not None else None,
            "error_message": str(r["error_message"]) if r["error_message"] is not None else None,
            "error_type": str(r["error_type"]) if r["error_type"] is not None else None,
            "feature_names_sha256": str(r["feature_names_sha256"]),
            "feature_set_version": str(r["feature_set_version"]),
            "feature_values": feature_vals,
            "missing_request_fields": list(r["missing_request_fields"] or []),
            "payment_id": str(r["payment_id"]),
            "registry_answers": reg_answers,
            "run_id": str(r["run_id"]),
            "status": str(r["status"]),
            "watermark_lag_ms": int(r["watermark_lag_ms"]) if r["watermark_lag_ms"] is not None else None,
        }
        canonical_rows.append(row_dict)

    payload = orjson.dumps(canonical_rows, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


def check_workspace_topic_partitions(consumer: AIOKafkaConsumer, topic: str) -> None:
    """
    WHAT: Enforces Section 9.4 R1: verifies that the workspace Kafka topic has exactly 1 partition.
    WHY: Prevents the unwatermarked partition hazard where a partition that never receives a message
    fails to hold the aggregate watermark back.
    Uses aiokafka API: consumer.partitions_for_topic(topic) (AIOKafkaConsumer.partitions_for_topic:620).
    """
    parts = consumer.partitions_for_topic(topic)
    if parts is None or len(parts) != 1:
        raise ValueError(
            f"Section 9.4 R1 violation: workspace topic {topic!r} must have exactly 1 partition, "
            f"got {len(parts) if parts is not None else 0} partitions: {parts}"
        )


IST = timezone(timedelta(hours=5, minutes=30))


def day_boundaries_between(
    prev_world_time: datetime,
    next_world_time: datetime,
) -> list[datetime]:
    """
    WHAT: Pure function returning all IST midnight boundaries strictly between prev_world_time and next_world_time.
    WHY: Implements architect decision D2: identifies completed simulation world days in IST
    to trigger daily account snapshot computation during replay.
    """
    t_prev = (
        prev_world_time
        if prev_world_time.tzinfo is not None
        else prev_world_time.replace(tzinfo=IST)
    ).astimezone(IST)
    t_next = (
        next_world_time
        if next_world_time.tzinfo is not None
        else next_world_time.replace(tzinfo=IST)
    ).astimezone(IST)

    if t_next < t_prev:
        raise ValueError(
            f"next_world_time ({next_world_time}) is earlier than prev_world_time ({prev_world_time})"
        )

    prev_date = t_prev.date()
    cur_midnight = datetime(
        prev_date.year, prev_date.month, prev_date.day, 0, 0, 0, tzinfo=IST
    ) + timedelta(days=1)

    boundaries: list[datetime] = []
    while cur_midnight < t_next:
        if cur_midnight > t_prev:
            boundaries.append(cur_midnight)
        cur_midnight += timedelta(days=1)

    return boundaries


async def drain(
    step_index: int,
    workspace_conn: asyncpg.Connection,
    producer: Any,
    consumer: AIOKafkaConsumer,
    workspace: ReplayWorkspace,
    redis_client: Any,
    acct_settings: AccountFeaturesSettings,
    banks: list[str],
    world_clock: ManualClock,
    run_id: str,
    poller_cursors: dict[str, dict[str, int]],
    deadline: float,
    batch_size: int = 100,
) -> float:
    """
    WHAT: Drains the outbox, consumer, and table pollers until pipeline convergence or deadline.
    WHY: Architect decision S2: guarantees that all prior events are fully committed and watermarked
    before scoring a decision. S2.3: consumer is polled on every iteration to prevent deadlocks.
    """
    t0 = time.monotonic()
    topic_parts = consumer.partitions_for_topic(workspace.kafka_topic) or {0}
    tps = [TopicPartition(workspace.kafka_topic, p) for p in sorted(topic_parts)]

    while True:
        now = time.monotonic()
        if now > deadline:
            end_dict = await consumer.end_offsets(tps)
            committed_dict = {tp: await consumer.committed(tp) for tp in tps}
            raise ReplayDrainTimeout(
                f"Drain deadline exceeded at step {step_index}: "
                f"partitions={[(tp.partition, end_dict.get(tp), committed_dict.get(tp)) for tp in tps]}, "
                f"poller_cursors={poller_cursors}"
            )

        # a. Outbox: relay batches until count_waiting == 0
        while await count_waiting(workspace_conn) > 0:
            if time.monotonic() > deadline:
                break
            await relay_one_batch(
                workspace_conn,
                producer,
                batch_size=batch_size,
                topic=workspace.kafka_topic,
            )
            # S2.3: poll consumer on every loop
            await consume_available(consumer, redis_client, acct_settings, timeout_ms=20)

        # b. Consumer: consume available messages until committed offset >= end offset for all partitions
        await consume_available(consumer, redis_client, acct_settings, timeout_ms=20)

        end_dict = await consumer.end_offsets(tps)
        committed_dict = {tp: await consumer.committed(tp) for tp in tps}
        consumer_caught_up = all(
            (committed_dict.get(tp) or 0) >= end_dict.get(tp, 0)
            for tp in tps
        )

        # c. Pollers: advance each of the 4 pollers for each bank until cursor >= table max id
        pollers_caught_up = True
        for bank_id in banks:
            # Events
            max_events = await workspace_conn.fetchval(
                "SELECT COALESCE(MAX(id), 0) FROM bank_account_events WHERE bank_id = $1",
                bank_id,
            )
            while poller_cursors[bank_id]["events"] < max_events:
                c = await apply_account_events_since(
                    workspace_conn,
                    redis_client,
                    settings=acct_settings,
                    bank_id=bank_id,
                    after_id=poller_cursors[bank_id]["events"],
                    clock=world_clock,
                    run_id=run_id,
                )
                if c <= poller_cursors[bank_id]["events"]:
                    break
                poller_cursors[bank_id]["events"] = c

            # Withdrawals
            max_wth = await workspace_conn.fetchval(
                "SELECT COALESCE(MAX(id), 0) FROM bank_withdrawals WHERE bank_id = $1",
                bank_id,
            )
            while poller_cursors[bank_id]["withdrawals"] < max_wth:
                c = await apply_withdrawals_since(
                    workspace_conn,
                    redis_client,
                    settings=acct_settings,
                    bank_id=bank_id,
                    after_id=poller_cursors[bank_id]["withdrawals"],
                    clock=world_clock,
                    run_id=run_id,
                )
                if c <= poller_cursors[bank_id]["withdrawals"]:
                    break
                poller_cursors[bank_id]["withdrawals"] = c

            # Feedback
            max_fb = await workspace_conn.fetchval(
                "SELECT COALESCE(MAX(id), 0) FROM bank_feedback WHERE bank_id = $1",
                bank_id,
            )
            while poller_cursors[bank_id]["feedback"] < max_fb:
                c = await apply_feedback_since(
                    workspace_conn,
                    redis_client,
                    settings=acct_settings,
                    bank_id=bank_id,
                    after_id=poller_cursors[bank_id]["feedback"],
                    clock=world_clock,
                    run_id=run_id,
                )
                if c <= poller_cursors[bank_id]["feedback"]:
                    break
                poller_cursors[bank_id]["feedback"] = c

            # Complaints
            max_cmp = await workspace_conn.fetchval(
                "SELECT COALESCE(MAX(id), 0) FROM bank_complaints WHERE bank_id = $1",
                bank_id,
            )
            while poller_cursors[bank_id]["complaints"] < max_cmp:
                c = await apply_complaints_since(
                    workspace_conn,
                    redis_client,
                    settings=acct_settings,
                    bank_id=bank_id,
                    after_id=poller_cursors[bank_id]["complaints"],
                    clock=world_clock,
                    run_id=run_id,
                )
                if c <= poller_cursors[bank_id]["complaints"]:
                    break
                poller_cursors[bank_id]["complaints"] = c

            # Labels (from complaints)
            while poller_cursors[bank_id]["labels"] < max_cmp:
                c_lbl = await consume_complaint_labels_since(
                    workspace_conn,
                    bank_id=bank_id,
                    run_id=run_id,
                    after_id=poller_cursors[bank_id]["labels"],
                    batch_size=batch_size,
                )
                if c_lbl <= poller_cursors[bank_id]["labels"]:
                    break
                poller_cursors[bank_id]["labels"] = c_lbl

            # Advance clock watermarks on empty or caught-up tables
            c_ev = await apply_account_events_since(
                workspace_conn, redis_client, settings=acct_settings, bank_id=bank_id,
                after_id=poller_cursors[bank_id]["events"], clock=world_clock, run_id=run_id,
            )
            poller_cursors[bank_id]["events"] = max(poller_cursors[bank_id]["events"], c_ev)

            c_wt = await apply_withdrawals_since(
                workspace_conn, redis_client, settings=acct_settings, bank_id=bank_id,
                after_id=poller_cursors[bank_id]["withdrawals"], clock=world_clock, run_id=run_id,
            )
            poller_cursors[bank_id]["withdrawals"] = max(poller_cursors[bank_id]["withdrawals"], c_wt)

            c_fb = await apply_feedback_since(
                workspace_conn, redis_client, settings=acct_settings, bank_id=bank_id,
                after_id=poller_cursors[bank_id]["feedback"], clock=world_clock, run_id=run_id,
            )
            poller_cursors[bank_id]["feedback"] = max(poller_cursors[bank_id]["feedback"], c_fb)

            c_cm = await apply_complaints_since(
                workspace_conn, redis_client, settings=acct_settings, bank_id=bank_id,
                after_id=poller_cursors[bank_id]["complaints"], clock=world_clock, run_id=run_id,
            )
            poller_cursors[bank_id]["complaints"] = max(poller_cursors[bank_id]["complaints"], c_cm)

            if (
                poller_cursors[bank_id]["events"] < max_events
                or poller_cursors[bank_id]["withdrawals"] < max_wth
                or poller_cursors[bank_id]["feedback"] < max_fb
                or poller_cursors[bank_id]["complaints"] < max_cmp
                or poller_cursors[bank_id]["labels"] < max_cmp
            ):
                pollers_caught_up = False

        waiting_count = await count_waiting(workspace_conn)
        if waiting_count == 0 and consumer_caught_up and pollers_caught_up:
            break

        await asyncio.sleep(0.01)

    return time.monotonic() - t0


def load_default_decisions_per_event_payment() -> bool:
    """
    WHAT: Reads the decisions_per_event_payment policy flag from config/replay.yaml.
    WHY: Decision D6: Replay and Level 16c datasets default to one decision per executed attempt.
    """
    from core.config_validator import load_yaml
    from world.replay_workspace import CONFIG_PATH

    data = load_yaml(CONFIG_PATH)
    block = data.get("replay_workspace", {})
    entry = block.get("decisions_per_event_payment")
    if entry is None or not isinstance(entry, dict) or "value" not in entry:
        raise ValueError(f"Missing 'decisions_per_event_payment' policy setting in {CONFIG_PATH}")
    return bool(entry["value"])


def validate_replay_speed(speed: Any) -> None:
    """
    WHAT: Enforces validity of replay_speed setting.
    WHY: Decision D2: Speed must be 'max' or a float/int strictly greater than 0.
    """
    if speed == "max":
        return
    if isinstance(speed, (int, float)) and not isinstance(speed, bool):
        if speed > 0:
            return
        raise ValueError(f"replay_speed must be > 0, got {speed!r}")
    raise ValueError(f"replay_speed must be 'max' or a number > 0, got {speed!r}")


def load_default_replay_speed() -> str | float:
    """
    WHAT: Reads the replay_speed policy setting from config/replay.yaml.
    WHY: Level 16 task 2: Replay pacing defaults to 'max' (no sleeping).
    """
    from core.config_validator import load_yaml
    from world.replay_workspace import CONFIG_PATH

    data = load_yaml(CONFIG_PATH)
    block = data.get("replay_workspace", {})
    entry = block.get("replay_speed")
    if entry is None or not isinstance(entry, dict) or "value" not in entry:
        raise ValueError(f"Missing 'replay_speed' policy setting in {CONFIG_PATH}")
    val = entry["value"]
    validate_replay_speed(val)
    return val


def load_default_idle_partition_advance() -> bool:
    """
    WHAT: Reads the idle_partition_advance policy setting from config/replay.yaml.
    WHY: Architect 16c-3 (b): Drained idle payment partitions advance their watermark to clock_now - commit_grace.
    """
    from core.config_validator import load_yaml
    from world.replay_workspace import CONFIG_PATH

    data = load_yaml(CONFIG_PATH)
    block = data.get("replay_workspace", {})
    entry = block.get("idle_partition_advance")
    if entry is None or not isinstance(entry, dict) or "value" not in entry:
        raise ValueError(f"Missing 'idle_partition_advance' policy setting in {CONFIG_PATH}")
    return bool(entry["value"])


def idle_advance_target(
    clock_now: datetime,
    commit_grace_seconds: int | float,
    partition_drained: bool,
    next_unsent_arrival: datetime | None,
) -> tuple[datetime | None, str]:
    """
    WHAT: Evaluates whether a drained, idle payment partition may safely advance its watermark.
    WHY: Architect 16c-3 (b): In replay, when a partition is drained and no unsent record has arrival <= target,
    the partition is proven idle and its watermark advances to clock_now - commit_grace (matching pollers).

    Args:
        clock_now: Current simulation world clock datetime (must be timezone-aware).
        commit_grace_seconds: Ingestion watermark lag / grace period in seconds.
        partition_drained: Boolean indicating whether all pending events have been relayed and consumed.
        next_unsent_arrival: World time of the next scheduled step not yet dispatched, or None if at replay end.

    Returns:
        tuple[datetime | None, str]: (target, "advance") if safe to advance; otherwise (None, reason).
    """
    if clock_now.tzinfo is None:
        raise ValueError("clock_now must be timezone-aware")

    target = clock_now - timedelta(seconds=float(commit_grace_seconds))

    if not partition_drained:
        return None, "partition not drained"

    if next_unsent_arrival is not None and next_unsent_arrival <= target:
        return (
            None,
            f"next unsent record arrival {next_unsent_arrival.isoformat()} <= target {target.isoformat()} (in flight)",
        )

    # Steps are ordered chronologically; next_unsent_arrival proves every record with arrival <= target was already sent
    assert next_unsent_arrival is None or next_unsent_arrival > target, (
        f"Unsent step at {next_unsent_arrival} violates advance target {target}"
    )

    return target, "advance"


def pace_seconds(
    previous_world_time: datetime | None,
    next_world_time: datetime,
    speed: str | float,
) -> float:
    """
    WHAT: Pure function computing real seconds to sleep between replay steps.
    WHY: Decision D2: 0 for 'max'; (next_world_time - previous_world_time) / speed otherwise.
    Raises ValueError on negative gap (backwards world time).
    """
    validate_replay_speed(speed)
    if speed == "max" or previous_world_time is None:
        return 0.0
    if not isinstance(speed, (int, float)):
        raise ValueError(f"replay_speed must be a positive number, got {speed!r}")
    gap = (next_world_time - previous_world_time).total_seconds()
    if gap < 0:
        raise ValueError(
            f"Backwards world time gap ({gap}s): previous {previous_world_time} > next {next_world_time}"
        )
    return float(gap / speed)


async def pace_step(
    previous_world_time: datetime | None,
    next_world_time: datetime,
    speed: str | float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> float:
    """
    WHAT: Pacing helper evaluating pace_seconds and awaiting sleep(pace) before each replay step.
    WHY: Encapsulates step pacing logic for standalone unit testing and run_replay invocation.
    """
    pace = pace_seconds(previous_world_time, next_world_time, speed)
    await sleep(pace)
    return pace


async def _snapshot_completed_day(
    drain_conn: asyncpg.Connection,
    boundary: datetime,
    workspace: ReplayWorkspace,
    ws_redis: Any,
    producer: Any,
    consumer: AIOKafkaConsumer,
    acct_cfg: AccountFeaturesSettings,
    banks: list[str],
    world_clock: ManualClock,
    run_id: str,
    poller_cursors: dict[str, dict[str, int]],
    drain_timeout_seconds: float,
    step_idx: int,
    idle_advance: bool = False,
    produced_partitions: set[int] | None = None,
    next_unsent_arrival: datetime | None = None,
) -> float:
    """
    WHAT: Drains the operational pipeline up to boundary and writes daily account snapshots for completed day.
    WHY: Implements architect decision D2: snapshots active accounts for [boundary - 1d, boundary) at each IST midnight.
    """
    world_clock.set_time(boundary)
    deadline = time.monotonic() + drain_timeout_seconds
    drain_duration = await drain(
        step_index=step_idx,
        workspace_conn=drain_conn,
        producer=producer,
        consumer=consumer,
        workspace=workspace,
        redis_client=ws_redis,
        acct_settings=acct_cfg,
        banks=banks,
        world_clock=world_clock,
        run_id=run_id,
        poller_cursors=poller_cursors,
        deadline=deadline,
    )
    if idle_advance and produced_partitions:
        target_ts, reason = idle_advance_target(
            clock_now=boundary,
            commit_grace_seconds=acct_cfg.commit_grace,
            partition_drained=True,
            next_unsent_arrival=next_unsent_arrival,
        )
        if target_ts is not None:
            for part in produced_partitions:
                record_payment_watermark(ws_redis, partition=part, timestamp=target_ts, run_id=run_id)
    day_start = boundary - timedelta(days=1)
    day_end = boundary
    for bank_id in banks:
        account_ids = await select_daily_snapshot_accounts(
            drain_conn,
            run_id=run_id,
            bank_id=bank_id,
            day_start=day_start,
            day_end=day_end,
        )
        if account_ids:
            await write_account_snapshots(
                drain_conn,
                ws_redis,
                run_id=run_id,
                bank_id=bank_id,
                snapshot_time=boundary,
                snapshot_kind="daily",
                account_ids=account_ids,
                settings=acct_cfg,
                run_scoped_watermarks=True,
            )
    return drain_duration


async def run_replay(
    recorded_world: RecordedWorld,
    feed: BankFeed,
    admin_pg: str,
    *,
    decisions_per_event_payment: bool | None = None,
    replay_speed: str | float | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    replay_settings: ReplaySettings | None = None,
    account_settings: AccountFeaturesSettings | None = None,
    live_redis: Any = None,
    base_redis_url: str | None = None,
    redis_client_factory: Callable[[ReplayWorkspace], Any] | None = None,
    before_decision_hook: Callable[[ReplayHookContext], Awaitable[None]] | None = None,
    before_drop_hook: Callable[[ReplayDropContext], Awaitable[None]] | None = None,
    workspace_name: str | None = None,
    idle_partition_advance: bool | None = None,
) -> ReplayResult:
    """
    WHAT: Drives the complete operational pipeline in world-time order to replay a recorded simulation run.
    WHY: Implements architect decisions C2, 16b Q1-Q3, P1, S1, S2, and 16b-3b Ruling (R1-R3).
    """
    # 1. Refuse live bank modes and interleave steps
    bank_modes = load_bank_modes()
    for mode in bank_modes.values():
        if mode == "live":
            raise ValueError("Live bank mode refused until Level 19")

    idle_advance = (
        load_default_idle_partition_advance()
        if idle_partition_advance is None
        else bool(idle_partition_advance)
    )

    per_event_mode = (
        load_default_decisions_per_event_payment()
        if decisions_per_event_payment is None
        else decisions_per_event_payment
    )
    steps = build_replay_steps(
        recorded_world,
        feed,
        decisions_per_event_payment=per_event_mode,
    )

    current_speed = (
        load_default_replay_speed()
        if replay_speed is None
        else replay_speed
    )
    validate_replay_speed(current_speed)

    r_cfg = replay_settings if replay_settings is not None else get_default_replay_settings()
    acct_cfg = account_settings if account_settings is not None else get_default_settings()

    run_id = recorded_world.output.result.run_id
    base_name = workspace_name if workspace_name is not None else f"rpl_{run_id[:12]}_{uuid.uuid4().hex[:6]}"

    if not isinstance(admin_pg, str):
        raise ValueError(f"admin_pg must be a DSN string, got {type(admin_pg)}")
    pool_dsn = admin_pg

    if live_redis is None:
        r_url = base_redis_url if base_redis_url is not None else config.redis_url
        live_redis_client = redis.Redis.from_url(r_url)
    else:
        live_redis_client = live_redis

    # 2. Allocate isolated workspace; ALWAYS drop in finally
    workspace = await allocate_workspace(
        admin_pg,
        live_redis_client,
        base_name,
        settings=r_cfg,
        base_redis_url=base_redis_url,
        redis_client_factory=redis_client_factory,
    )

    producer = None
    consumer = None
    workspace_pool = None
    ws_redis = None

    try:
        # 3. Provision isolated operational connections
        workspace_pool = await asyncpg.create_pool(
            dsn=pool_dsn,
            server_settings={"search_path": workspace.postgres_schema},
            min_size=1,
            max_size=5,
        )
        assert workspace_pool is not None

        # Pre-create daily partitions for feature_log and account_snapshots for all world dates in steps.
        # Derives the unique calendar days in both local world-time (IST) and UTC directly from the scheduled
        # replay steps, pre-creating feature_log and account_snapshots range partitions so that unprivileged tenant
        # roles do not trigger on-the-fly table creation DDL.
        partition_days = {s.world_time.date() for s in steps} | {
            s.world_time.astimezone(timezone.utc).date() for s in steps
        }
        if steps:
            last_dt = steps[-1].world_time
            next_dt = last_dt + timedelta(days=1)
            partition_days |= {
                next_dt.date(),
                next_dt.astimezone(timezone.utc).date(),
            }
        async with workspace_pool.acquire() as setup_conn:
            for d in sorted(partition_days):
                await setup_conn.execute("SELECT create_feature_log_partition($1::date)", d)
                await setup_conn.execute("SELECT create_account_snapshots_partition($1::date)", d)

        factory = (
            redis_client_factory
            if redis_client_factory is not None
            else (lambda ws: workspace_redis_client(ws, base_redis_url=base_redis_url))
        )
        ws_redis = factory(workspace)

        producer = await start_producer()

        consumer = AIOKafkaConsumer(
            workspace.kafka_topic,
            bootstrap_servers=config.kafka_bootstrap_servers,
            group_id=workspace.kafka_group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await consumer.start()

        # R1: At replay start: read workspace topic's real partition count from Kafka (aiokafka metadata).
        # Must be exactly 1; otherwise raise ValueError.
        check_workspace_topic_partitions(consumer, workspace.kafka_topic)

        # 4. Set up clocks and bank connector transports
        if not steps:
            raise ValueError("Cannot replay an empty step list: no replay steps were generated")
        initial_time = steps[0].world_time
        world_clock = ManualClock(initial_time)
        transport_clock = WallClock()

        feed_cfg = load_bank_feed_config()
        clock_start = recorded_world.output.plan.clock.start_date
        if isinstance(clock_start, datetime):
            sim_start_dt = clock_start.astimezone(IST)
        else:
            sim_start_dt = datetime(
                clock_start.year, clock_start.month, clock_start.day, 0, 0, 0, tzinfo=IST
            )

        account_bank_map: dict[str, str] = {}
        for ledger in recorded_world.output.world.ledgers:
            b_id = resolve_bank_id(ledger.bank_index, feed_cfg)
            for acc in ledger.accounts:
                account_bank_map[acc] = b_id

        for opened in recorded_world.output.result.opened_accounts:
            op_acc = opened[1]
            op_b_idx = opened[4]
            account_bank_map[op_acc] = resolve_bank_id(op_b_idx, feed_cfg)

        clients: dict[str, InProcessBankClient] = {}
        for bank_id in bank_modes:
            clients[bank_id] = InProcessBankClient(
                pool=workspace_pool,
                bank_id=bank_id,
                clock=transport_clock,
                redis_client=ws_redis,
                decision_clock=world_clock,
                settings=acct_cfg,
                ingestion_clock=world_clock,
                run_scoped_watermarks=True,
                decision_feature_logging=True,
            )

        connectors = MockBankConnectors(
            transports=cast(dict[str, BankTransport], clients),
            modes=bank_modes,
            account_bank_map=account_bank_map,
            clock_start=clock_start,
            feed_config=feed_cfg,
            decision_timeout_seconds=5.0,
            loop=asyncio.get_running_loop(),
        )

        banks = sorted(bank_modes.keys())
        poller_cursors: dict[str, dict[str, int]] = {
            b: {"events": 0, "withdrawals": 0, "feedback": 0, "complaints": 0, "labels": 0}
            for b in banks
        }

        decisions_sent = 0
        longest_drain_seconds = 0.0
        decisions_before_first_payment = 0
        produced_partitions: set[int] = set()

        # 5 & 6. Execute replay steps
        prev_world_time: datetime | None = None
        async with workspace_pool.acquire() as drain_conn:
            for step_idx, step in enumerate(steps):
                await pace_step(prev_world_time, step.world_time, current_speed, sleep=sleep)

                if prev_world_time is not None:
                    boundaries = [
                        b
                        for b in day_boundaries_between(prev_world_time, step.world_time)
                        if b > sim_start_dt
                    ]
                    for boundary in boundaries:
                        dur = await _snapshot_completed_day(
                            drain_conn=drain_conn,
                            boundary=boundary,
                            workspace=workspace,
                            ws_redis=ws_redis,
                            producer=producer,
                            consumer=consumer,
                            acct_cfg=acct_cfg,
                            banks=banks,
                            world_clock=world_clock,
                            run_id=run_id,
                            poller_cursors=poller_cursors,
                            drain_timeout_seconds=r_cfg.drain_timeout_seconds,
                            step_idx=step_idx,
                            idle_advance=idle_advance,
                            produced_partitions=produced_partitions,
                            next_unsent_arrival=step.world_time,
                        )
                        if dur > longest_drain_seconds:
                            longest_drain_seconds = dur

                prev_world_time = step.world_time

                if isinstance(step, FeedStep):
                    # 5. FeedStep: set world clock, then post feed record
                    world_clock.set_time(step.world_time)
                    await connectors.send_feed_record(step.kind, step.record)
                    if step.kind == "transaction":
                        produced_partitions.add(0)

                elif isinstance(step, DecisionStep):
                    # 6. DecisionStep: set world clock, drain, check R2/R3, post decision
                    world_clock.set_time(step.world_time)
                    deadline = time.monotonic() + r_cfg.drain_timeout_seconds
                    drain_duration = await drain(
                        step_index=step_idx,
                        workspace_conn=drain_conn,
                        producer=producer,
                        consumer=consumer,
                        workspace=workspace,
                        redis_client=ws_redis,
                        acct_settings=acct_cfg,
                        banks=banks,
                        world_clock=world_clock,
                        run_id=run_id,
                        poller_cursors=poller_cursors,
                        deadline=deadline,
                    )
                    if drain_duration > longest_drain_seconds:
                        longest_drain_seconds = drain_duration

                    if idle_advance and produced_partitions:
                        next_unsent = steps[step_idx + 1].world_time if step_idx + 1 < len(steps) else None
                        target_ts, reason = idle_advance_target(
                            clock_now=step.world_time,
                            commit_grace_seconds=acct_cfg.commit_grace,
                            partition_drained=True,
                            next_unsent_arrival=next_unsent,
                        )
                        if target_ts is not None:
                            for part in produced_partitions:
                                record_payment_watermark(ws_redis, partition=part, timestamp=target_ts, run_id=run_id)

                    # Optional test injection hook (e.g. to simulate deleting watermark key or pausing consumer)
                    if before_decision_hook is not None:
                        ctx = ReplayHookContext(
                            step_index=step_idx,
                            decision_step=step,
                            consumer=consumer,
                            workspace=workspace,
                        )
                        await before_decision_hook(ctx)

                    # R2: Every partition the runner has produced at least one message to must have a watermark
                    for part in produced_partitions:
                        wm = get_payment_watermark(ws_redis, partition=part, run_id=run_id)
                        if wm is None:
                            raise ReplayPipelineError(
                                f"Section 9.4 R2 violation at step {step_idx}: Produced partition {part} "
                                f"has no run-scoped payment watermark for run {run_id}"
                            )

                    # R3: Decisions that come before ANY payment has been produced
                    # legitimately get status 'no_watermark'
                    if not produced_partitions:
                        decisions_before_first_payment += 1

                    # Post decision
                    sender_bank = step.call.sender_bank_id
                    transport = connectors.transports[sender_bank]
                    await transport.post(
                        "/v1/decisions",
                        step.call.body_bytes,
                        step.call.idempotency_key,
                    )
                    decisions_sent += 1

            if prev_world_time is not None:
                prev_ist = (
                    prev_world_time
                    if prev_world_time.tzinfo is not None
                    else prev_world_time.replace(tzinfo=IST)
                ).astimezone(IST)
                final_boundary = datetime(
                    prev_ist.year, prev_ist.month, prev_ist.day, 0, 0, 0, tzinfo=IST
                ) + timedelta(days=1)
                if final_boundary > sim_start_dt:
                    dur = await _snapshot_completed_day(
                        drain_conn=drain_conn,
                        boundary=final_boundary,
                        workspace=workspace,
                        ws_redis=ws_redis,
                        producer=producer,
                        consumer=consumer,
                        acct_cfg=acct_cfg,
                        banks=banks,
                        world_clock=world_clock,
                        run_id=run_id,
                        poller_cursors=poller_cursors,
                        drain_timeout_seconds=r_cfg.drain_timeout_seconds,
                        step_idx=len(steps),
                        idle_advance=idle_advance,
                        produced_partitions=produced_partitions,
                        next_unsent_arrival=None,
                    )
                    if dur > longest_drain_seconds:
                        longest_drain_seconds = dur

            # Read feature rows from feature_log table before workspace teardown
            rows = await drain_conn.fetch(
                "SELECT bank_id, payment_id, decision_time, decided_at, watermark_lag_ms, "
                "status, feature_values, missing_request_fields, registry_answers, created_at, "
                "run_id, account_id, arrival_cutoff, feature_set_version, feature_names_sha256, "
                "error_type, error_message "
                "FROM feature_log ORDER BY bank_id, payment_id, decision_time"
            )

        sha256_hash = compute_feature_log_sha256(rows)
        feature_rows_by_status: dict[str, int] = defaultdict(int)
        decisions_arrival_cutoff_behind = 0
        max_gap = 0.0
        unwatermarked_ids: list[str] = []
        features_by_pid: dict[str, dict[str, Any]] = {}
        missing_by_pid: dict[str, list[str]] = {}
        logged_payment_ids_by_bank: dict[str, list[str]] = defaultdict(list)
        canonical_names = canonical_feature_names()

        for r in rows:
            st = str(r["status"])
            pid = str(r["payment_id"])
            bid = str(r["bank_id"])
            logged_payment_ids_by_bank[bid].append(pid)
            feature_rows_by_status[st] += 1
            if st == "no_watermark":
                unwatermarked_ids.append(pid)

            missing_by_pid[pid] = list(r["missing_request_fields"] or [])

            raw_vals = r["feature_values"]
            if isinstance(raw_vals, (str, bytes)):
                parsed_vals = orjson.loads(raw_vals)
            else:
                parsed_vals = raw_vals

            if isinstance(parsed_vals, list) and len(parsed_vals) == len(canonical_names):
                features_by_pid[pid] = dict(zip(canonical_names, parsed_vals))
            elif isinstance(parsed_vals, dict):
                features_by_pid[pid] = parsed_vals

            d_time = r["decision_time"]
            a_cutoff = r["arrival_cutoff"]
            if a_cutoff is not None and a_cutoff < d_time:
                decisions_arrival_cutoff_behind += 1
                gap = (d_time - a_cutoff).total_seconds()
                if gap > max_gap:
                    max_gap = gap

        if before_drop_hook is not None:
            sep = "&" if "?" in admin_pg else "?"
            workspace_dsn = f"{admin_pg}{sep}options=-csearch_path%3D{workspace.postgres_schema}"
            drop_ctx = ReplayDropContext(
                workspace=workspace,
                workspace_dsn=workspace_dsn,
                logged_payment_ids_by_bank=dict(logged_payment_ids_by_bank),
            )
            await before_drop_hook(drop_ctx)

        return ReplayResult(
            decisions_sent=decisions_sent,
            feature_rows_by_status=dict(feature_rows_by_status),
            decisions_arrival_cutoff_behind=decisions_arrival_cutoff_behind,
            max_arrival_cutoff_gap_seconds=max_gap,
            longest_drain_seconds=longest_drain_seconds,
            decisions_before_first_payment=decisions_before_first_payment,
            unwatermarked_payment_ids=tuple(unwatermarked_ids),
            logged_features_by_payment_id=features_by_pid,
            missing_request_fields_by_payment_id=missing_by_pid,
            registry_wired=False,
            registry_wired_reason="no nationalsim screener or DPIP key in replay until 16d",
            feature_log_sha256=sha256_hash,
        )

    finally:
        # Resource cleanup
        if consumer is not None:
            try:
                await consumer.stop()
            except Exception as exc:
                logger.warning("Error stopping consumer: %s", exc)

        if producer is not None:
            try:
                await producer.stop()
            except Exception as exc:
                logger.warning("Error stopping producer: %s", exc)

        if ws_redis is not None and hasattr(ws_redis, "close"):
            try:
                ws_redis.close()
            except Exception as exc:
                logger.warning("Error closing workspace redis client: %s", exc)

        if workspace_pool is not None:
            try:
                await workspace_pool.close()
            except Exception as exc:
                logger.warning("Error closing workspace asyncpg pool: %s", exc)

        # Drop workspace ALWAYS
        try:
            await drop_workspace(
                admin_pg,
                live_redis_client,
                workspace,
                settings=r_cfg,
                base_redis_url=base_redis_url,
                redis_client_factory=redis_client_factory,
            )
        except Exception as exc:
            logger.error("Error dropping workspace %s: %s", workspace.name, exc)
