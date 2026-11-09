# File: tests/integration/test_replay_labels_snapshots.py
# Purpose: Integration tests for replay labels ingestion and daily account snapshots.
# Issue: Level 16c round 2a architect decisions D1, D2.
"""
WHAT: Integration tests verifying that run_replay drives the real operational pipeline to ingest
complaint fraud labels and write daily account snapshots at IST midnight boundaries.

WHY: Architect decisions D1, D2:
D1: Replay drain runs complaint -> label consumer for each commercial bank into workspace labels table.
D2: Replay runner identifies completed world days in IST, draining and writing daily account snapshots
    for all active customer accounts.
"""

import asyncio
import dataclasses
import os
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
import orjson
import pytest
import redis

from core.config import config
from core.config_validator import load_yaml
from features.account_keys import get_default_settings
from features.account_records import account_event_from_row
from features.account_snapshots import (
    account_feature_names,
    select_daily_snapshot_accounts,
    write_account_snapshots,
)
from features.account_stream import apply_account_event
from features.ingest_watermark import record_payment_watermark, record_poller_watermark
from features.m1_features import SUPPORTED_GROUPS, compute_account_features
from replay.ordering import FeedStep, build_replay_steps
from replay.pipeline import feed_with_labels
from replay.runner import IST, ReplayDropContext, run_replay
from tests.integration.conftest import get_test_database_url
from tests.integration.test_replay_recording import RejectingTransport
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import BankFeedPaymentSide, resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.gang_config import BlockStep, GangConfig, gang_config_from_plain
from world.interactive_records import RunnerSettings
from world.interactive_runner import FastRunSettings, prepare_world
from world.replay_recording import record_world_for_replay
from world.replay_workspace import ReplayWorkspace, workspace_redis_client

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

WORLD_SETTINGS = FastRunSettings(101, 20, 6, 51, "training")


def host_side_redis_url() -> str:
    """Resolve Redis connection string for test runner on host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


def host_redis_factory(ws: ReplayWorkspace) -> redis.Redis:
    """Constructs a Redis client pointing to host-accessible URL for the workspace DB."""
    return workspace_redis_client(ws, base_redis_url=host_side_redis_url())


@pytest.fixture
def live_redis() -> Generator[redis.Redis, None, None]:
    """Provides real Redis client connected to the host live database."""
    client = redis.Redis.from_url(host_side_redis_url())
    yield client
    client.close()


@pytest.fixture
async def pg_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    """Provides direct connection to main test database."""
    conn = await asyncpg.connect(get_test_database_url())
    try:
        yield conn
    finally:
        await conn.close()


async def test_boundary_daily_snapshots(
    isolated_schema: tuple[asyncpg.Connection, str],
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Fast boundary test verifying that account snapshots partition daily windows strictly by arrival_time.
    WHY: Implements architect decision D2:
      - Given events with arrival_time = boundary - 1s and boundary + 1s.
      - select_daily_snapshot_accounts for [boundary - 1d, boundary) selects only the earlier account.
      - write_account_snapshots at boundary reflects only the earlier event in feature_values (sim_swapped=1).
    """
    conn, schema_name = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    boundary = datetime(2026, 9, 2, 0, 0, 0, tzinfo=IST)
    day_start = boundary - timedelta(days=1)
    day_end = boundary

    # Pre-create account snapshots partition for boundary dates
    await conn.execute("SELECT create_account_snapshots_partition($1::date)", boundary.date())
    await conn.execute(
        "SELECT create_account_snapshots_partition($1::date)",
        (boundary + timedelta(days=1)).date(),
    )

    acc_early = f"acc_early_{uuid.uuid4().hex[:8]}"
    acc_late = f"acc_late_{uuid.uuid4().hex[:8]}"

    # Event 1: earlier (arrival_time = boundary - 1s)
    t_early_event = boundary - timedelta(minutes=5)
    t_early_arrival = boundary - timedelta(seconds=1)
    evt_early_id = f"{run_id}-{uuid.uuid4().hex[:8]}"

    # Event 2: later (arrival_time = boundary + 1s)
    t_late_event = boundary + timedelta(minutes=5)
    t_late_arrival = boundary + timedelta(seconds=1)
    evt_late_id = f"{run_id}-{uuid.uuid4().hex[:8]}"

    row1 = await conn.fetchrow(
        """
        INSERT INTO bank_account_events (
            bank_id, event_id, account_id, kind, event_time, details, created_at, arrival_time
        ) VALUES (
            $1, $2, $3, 'sim_swap', $4, '{}'::jsonb, $4, $5
        ) RETURNING id, bank_id, event_id, account_id, kind, event_time, details, created_at, arrival_time
        """,
        bank_id,
        evt_early_id,
        acc_early,
        t_early_event,
        t_early_arrival,
    )
    assert row1 is not None

    row2 = await conn.fetchrow(
        """
        INSERT INTO bank_account_events (
            bank_id, event_id, account_id, kind, event_time, details, created_at, arrival_time
        ) VALUES (
            $1, $2, $3, 'sim_swap', $4, '{}'::jsonb, $4, $5
        ) RETURNING id, bank_id, event_id, account_id, kind, event_time, details, created_at, arrival_time
        """,
        bank_id,
        evt_late_id,
        acc_late,
        t_late_event,
        t_late_arrival,
    )
    assert row2 is not None

    # Apply both events to Redis
    cfg = get_default_settings()
    apply_account_event(live_redis, account_event_from_row(dict(row1)), cfg)
    apply_account_event(live_redis, account_event_from_row(dict(row2)), cfg)

    # Set run-scoped watermarks up to boundary
    b_unix = int(boundary.timestamp())
    record_payment_watermark(live_redis, 0, b_unix, run_id=run_id)
    record_poller_watermark(live_redis, bank_id, "events", b_unix, run_id=run_id)
    record_poller_watermark(live_redis, bank_id, "withdrawals", b_unix, run_id=run_id)
    record_poller_watermark(live_redis, bank_id, "feedback", b_unix, run_id=run_id)
    record_poller_watermark(live_redis, bank_id, "complaints", b_unix, run_id=run_id)

    # 1. select_daily_snapshot_accounts selects only acc_early
    active = await select_daily_snapshot_accounts(
        conn,
        run_id=run_id,
        bank_id=bank_id,
        day_start=day_start,
        day_end=day_end,
    )
    assert acc_early in active
    assert acc_late not in active

    # 2. write_account_snapshots at boundary
    inserted = await write_account_snapshots(
        conn,
        live_redis,
        run_id=run_id,
        bank_id=bank_id,
        snapshot_time=boundary,
        snapshot_kind="daily",
        account_ids=active,
        settings=cfg,
        run_scoped_watermarks=True,
    )
    assert inserted == 1

    # Verify sim_swapped in snapshot feature_values
    snap = await conn.fetchrow(
        "SELECT feature_values FROM account_snapshots WHERE run_id = $1 AND account_id = $2 AND snapshot_time = $3",
        run_id,
        acc_early,
        boundary,
    )
    assert snap is not None
    raw_feats = snap["feature_values"]
    fvals = orjson.loads(raw_feats) if isinstance(raw_feats, (str, bytes)) else raw_feats
    names = account_feature_names()
    sim_idx = names.index("sim_swapped")
    assert fvals[sim_idx] == 1

    # If acc_late were queried at boundary, arrival_cutoff filter ensures sim_swapped is 0
    late_feats = compute_account_features(
        client=live_redis,
        run_id=run_id,
        bank_id=bank_id,
        account_id=acc_late,
        decision_time=boundary,
        arrival_cutoff=boundary,
        settings=cfg,
        groups=tuple(SUPPORTED_GROUPS),
        screener=None,
    )
    assert late_feats["sim_swapped"] == 0


def _load_adapted_complaint_gang() -> GangConfig:
    """
    WHAT: Loads showcase/fixtures/level_12/gang_complaints.yaml adapted for the 20-person small world.
    WHY: Architect ruling: small world has 14 eligible victims; adapt gang to 1 receiving mule,
    12 victims, and family 'fast_pass_through' without modifying the fixture file on disk.
    """
    repo_root = Path(__file__).resolve().parents[2]
    gang_path = repo_root / "showcase" / "fixtures" / "level_12" / "gang_complaints.yaml"
    base_gang = gang_config_from_plain(load_yaml(gang_path))

    new_steps = []
    for s in base_gang.steps:
        if s.block == "scam":
            p = dict(s.params)
            p["receiving_mules"] = 1
            p["victims_per_receiving_mule"] = 12
            new_steps.append(BlockStep(block="scam", params=tuple(p.items())))
        else:
            new_steps.append(s)

    return dataclasses.replace(base_gang, family="fast_pass_through", steps=tuple(new_steps))


async def test_end_to_end_replay_labels_snapshots(
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Runs end-to-end replay verifying complaint label consumption and daily account snapshots.
    WHY: Implements architect decisions D1, D2:
      - Labels table receives complaint labels during replay drain (>0 with adapted gang).
      - Account snapshots table receives >= 1 daily snapshot per bank per active world day.
      - Daily snapshot reflects transaction from the last step of the completed day.
    """
    cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()
    prepared = prepare_world(WORLD_SETTINGS, None)
    account_bank_map: dict[str, str] = {
        acc: resolve_bank_id(ledger.bank_index, cfg)
        for ledger in prepared.world.ledgers
        for acc in ledger.accounts
    }

    loop = asyncio.get_running_loop()
    connectors = MockBankConnectors(
        transports={b: RejectingTransport() for b in bank_modes},
        modes=bank_modes,
        account_bank_map=account_bank_map,
        clock_start=prepared.plan.clock.start_date,
        feed_config=cfg,
        decision_timeout_seconds=5.0,
        loop=loop,
    )
    runner = RunnerSettings(fraud_enabled=True)
    adapted_gang = _load_adapted_complaint_gang()
    recorded_world = record_world_for_replay(
        WORLD_SETTINGS, runner, connectors, gang_configs=(adapted_gang,)
    )
    feed = feed_with_labels(recorded_world)

    steps = build_replay_steps(recorded_world, feed, decisions_per_event_payment=True)
    cs = recorded_world.output.plan.clock.start_date
    sim_start_date = cs.astimezone(IST).date() if isinstance(cs, datetime) else cs
    day1_boundary = (
        datetime(sim_start_date.year, sim_start_date.month, sim_start_date.day, 0, 0, 0, tzinfo=IST)
        + timedelta(days=1)
    )
    day1_payments = [
        s for s in steps
        if sim_start_date <= s.world_time.astimezone(IST).date() < day1_boundary.date()
        and isinstance(s, FeedStep)
        and isinstance(s.record, BankFeedPaymentSide)
    ]
    assert len(day1_payments) > 0, "Expected payments on first simulation day"
    last_day1_step = day1_payments[-1]
    assert isinstance(last_day1_step, FeedStep)
    assert isinstance(last_day1_step.record, BankFeedPaymentSide)
    target_bank = last_day1_step.record.bank_id
    target_account = last_day1_step.record.account_id
    assert last_day1_step.record.side == "credit"

    labels_count = -1
    snapshots_by_bank_day: dict[tuple[str, datetime], int] = defaultdict(int)
    total_snapshots = 0
    last_step_feature_values_off: list[float | int | None] | None = None

    async def inspect_hook_off(drop_ctx: ReplayDropContext) -> None:
        nonlocal labels_count, snapshots_by_bank_day, total_snapshots, last_step_feature_values_off
        ws_conn = await asyncpg.connect(drop_ctx.workspace_dsn)
        try:
            labels_count = await ws_conn.fetchval("SELECT count(*) FROM labels WHERE source = 'complaint'")
            by_source = await ws_conn.fetch(
                "SELECT bank_id, source, count(*) AS n FROM labels GROUP BY 1, 2 ORDER BY 1, 2"
            )
            print("LABELS PER SOURCE PER BANK:", [(r["bank_id"], r["source"], r["n"]) for r in by_source])
            snap_rows = await ws_conn.fetch(
                "SELECT bank_id, snapshot_time, account_id FROM account_snapshots WHERE snapshot_kind = 'daily'"
            )
            total_snapshots = len(snap_rows)
            for r in snap_rows:
                b_id = str(r["bank_id"])
                s_time = r["snapshot_time"].astimezone(timezone.utc)
                snapshots_by_bank_day[(b_id, s_time)] += 1

            snap_val_query = (
                "SELECT feature_values FROM account_snapshots "
                "WHERE bank_id = $1 AND account_id = $2 AND snapshot_time = $3 AND snapshot_kind = 'daily'"
            )
            snap_val_row = await ws_conn.fetchrow(
                snap_val_query,
                target_bank,
                target_account,
                day1_boundary.astimezone(timezone.utc),
            )
            if snap_val_row is not None:
                raw_v = snap_val_row["feature_values"]
                last_step_feature_values_off = orjson.loads(raw_v) if isinstance(raw_v, str) else raw_v
        finally:
            await ws_conn.close()

    # Run 1: idle_partition_advance=False (legacy behaviour)
    result_off = await run_replay(
        recorded_world,
        feed,
        admin_pg=get_test_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        idle_partition_advance=False,
        before_drop_hook=inspect_hook_off,
    )

    # Run 2: idle_partition_advance=True (new behaviour)
    last_step_feature_values_on: list[float | int | None] | None = None

    async def inspect_hook_on(drop_ctx: ReplayDropContext) -> None:
        nonlocal last_step_feature_values_on
        ws_conn = await asyncpg.connect(drop_ctx.workspace_dsn)
        try:
            snap_val_query = (
                "SELECT feature_values FROM account_snapshots "
                "WHERE bank_id = $1 AND account_id = $2 AND snapshot_time = $3 AND snapshot_kind = 'daily'"
            )
            snap_val_row = await ws_conn.fetchrow(
                snap_val_query,
                target_bank,
                target_account,
                day1_boundary.astimezone(timezone.utc),
            )
            if snap_val_row is not None:
                raw_v = snap_val_row["feature_values"]
                last_step_feature_values_on = orjson.loads(raw_v) if isinstance(raw_v, str) else raw_v
        finally:
            await ws_conn.close()

    result_on = await run_replay(
        recorded_world,
        feed,
        admin_pg=get_test_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        idle_partition_advance=True,
        before_drop_hook=inspect_hook_on,
    )

    # 1. Decisions sent: exactly matches event count from simulation run
    events = recorded_world.output.result.run_result.events
    assert result_off.decisions_sent == len(events)
    assert result_on.decisions_sent == len(events)

    # 2. Complaint labels (Decision D1):
    all_complaints = [c for cmps in feed.complaints_by_bank.values() for c in cmps]
    same_bank_cmps = [c for c in all_complaints if c.reported_bank_id == c.bank_id]
    cross_bank_cmps = [c for c in all_complaints if c.reported_bank_id != c.bank_id]

    assert len(same_bank_cmps) >= 2, f"Expected >= 2 same-bank complaints, found {len(same_bank_cmps)}"
    received_times = {c.received_at for c in same_bank_cmps}
    assert len(received_times) == len(same_bank_cmps), "Expected distinct received_at times for same-bank complaints"

    assert labels_count == len(same_bank_cmps)
    assert labels_count > 0
    if cross_bank_cmps:
        assert labels_count < len(all_complaints)
    else:
        assert labels_count == len(all_complaints)

    # 3. Daily account snapshots (Decision D2):
    assert total_snapshots > 0

    active_bank_days: set[tuple[str, datetime]] = set()
    for s in steps:
        s_date = s.world_time.astimezone(IST).date()
        if s_date >= sim_start_date and isinstance(s, FeedStep) and hasattr(s.record, "bank_id"):
            boundary = datetime(s_date.year, s_date.month, s_date.day, 0, 0, 0, tzinfo=IST) + timedelta(days=1)
            active_bank_days.add((s.record.bank_id, boundary.astimezone(timezone.utc)))

    for bank_id, boundary_utc in active_bank_days:
        assert snapshots_by_bank_day[(bank_id, boundary_utc)] >= 1

    # 4. Snapshot ordering check: daily snapshot at Day 1 boundary reflects the last transaction on Day 1
    assert last_step_feature_values_off is not None, (
        f"Snapshot for {target_account} at {day1_boundary} must exist (off)"
    )
    assert last_step_feature_values_on is not None, (
        f"Snapshot for {target_account} at {day1_boundary} must exist (on)"
    )
    feature_names = account_feature_names()

    def _counterparty(r: BankFeedPaymentSide) -> str:
        return f"{r.counterparty_bank_id}:{r.counterparty_vpa}" if r.counterparty_vpa else r.counterparty_bank_id

    expected_with_last = len({
        _counterparty(s.record)
        for s in day1_payments
        if s.record.account_id == target_account and s.record.side == "credit" and s.record.status == "success"
    })
    expected_without_last = len({
        _counterparty(s.record)
        for s in day1_payments[:-1]
        if s.record.account_id == target_account and s.record.side == "credit" and s.record.status == "success"
    })
    assert expected_with_last != expected_without_last, (
        "Test setup error: expected last payment to introduce a new distinct counterparty"
    )
    payer_idx = feature_names.index("distinct_payers_24h")
    val_off = last_step_feature_values_off[payer_idx]
    val_on = last_step_feature_values_on[payer_idx]
    assert val_off == expected_without_last, (
        f"Expected off run to undercount: got {val_off}, expected {expected_without_last}"
    )
    assert val_on == expected_with_last, (
        f"Expected on run to include last step: got {val_on}, expected {expected_with_last}"
    )

    # Residual gap: count feed payments whose arrival falls in the last commit_grace seconds before any day boundary
    residual_gap_payments = 0
    commit_grace = 5
    for s in steps:
        if isinstance(s, FeedStep) and isinstance(s.record, BankFeedPaymentSide):
            w_time = s.world_time.astimezone(IST)
            b = datetime(w_time.year, w_time.month, w_time.day, 0, 0, 0, tzinfo=IST) + timedelta(days=1)
            gap_start = b - timedelta(seconds=commit_grace)
            if gap_start <= w_time < b:
                residual_gap_payments += 1

    print("\n[REPLAY METRICS]")
    print(f"OFF: feature_log_sha256={result_off.feature_log_sha256}")
    print(
        f"OFF: decisions_arrival_cutoff_behind={result_off.decisions_arrival_cutoff_behind}, "
        f"max_gap={result_off.max_arrival_cutoff_gap_seconds:.3f}s"
    )
    print(f"OFF: distinct_payers_24h={val_off} (expected_without_last={expected_without_last})")
    print(f"ON:  feature_log_sha256={result_on.feature_log_sha256}")
    print(
        f"ON:  decisions_arrival_cutoff_behind={result_on.decisions_arrival_cutoff_behind}, "
        f"max_gap={result_on.max_arrival_cutoff_gap_seconds:.3f}s"
    )
    print(f"ON:  distinct_payers_24h={val_on} (expected_with_last={expected_with_last})")
    print(f"Residual gap payments count (within {commit_grace}s of midnight): {residual_gap_payments}")
