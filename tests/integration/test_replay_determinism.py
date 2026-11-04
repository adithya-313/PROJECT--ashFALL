# File: tests/integration/test_replay_determinism.py
# Purpose: Level 16b round 4 integration test: replay determinism and replayed == recomputed features.
"""
WHAT: Integration test demonstrating replay determinism and point-in-time feature recomputation.
WHY: Level 16b Round 4 (Decisions D3, D4):
- Replay A runs with before_drop_hook executing recompute_logged_row against isolated workspace tables.
- Recomputed features strictly match logged features for every 'ok' row; 'no_watermark' row consistently
  raises ValueError("Feature log row status is 'no_watermark'; nothing to recompute").
- Replay B on the same recorded world yields bitwise-identical feature_log_sha256 across all 16 deterministic columns.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator

import asyncpg
import fakeredis
import pytest
import redis

from core.config import config
from features.feature_replay import recompute_logged_row
from replay.runner import (
    ReplayDropContext,
    run_replay,
)
from tests.integration.conftest import get_test_database_url
from tests.integration.test_replay_recording import RejectingTransport
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import project_run, resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
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
async def pg_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    """Provides a fresh direct asyncpg connection to the test database."""
    conn = await asyncpg.connect(get_test_database_url())
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def live_redis() -> Generator[redis.Redis, None, None]:
    """Provides real Redis client connected to the host live database."""
    client = redis.Redis.from_url(host_side_redis_url())
    yield client
    client.close()


async def test_replay_determinism_and_recomputed_features(
    pg_conn: asyncpg.Connection,
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Tests end-to-end replay determinism and recomputed == logged feature equality.
    WHY: Implements Commit 3 requirements for Level 16b Round 4:
      - Replay A executes with before_drop_hook running recompute_logged_row with fakeredis scratch store.
      - Replay B executes on the identical recorded world and produces identical feature_log_sha256.
    """
    cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()
    prepared = prepare_world(WORLD_SETTINGS, None)
    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

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
    recorded_world = record_world_for_replay(WORLD_SETTINGS, runner, connectors)
    feed = project_run(recorded_world.output)
    events = recorded_world.output.result.run_result.events

    recompute_matches: list[str] = []
    recompute_mismatches: list[str] = []
    no_watermark_results: list[tuple[str, str]] = []

    async def verify_recompute_hook(ctx: ReplayDropContext) -> None:
        hook_conn = await asyncpg.connect(ctx.workspace_dsn)
        try:
            for bank_id, payment_ids in ctx.logged_payment_ids_by_bank.items():
                for payment_id in payment_ids:
                    scratch_redis = fakeredis.FakeRedis()
                    try:
                        rec_res = await recompute_logged_row(
                            pg_connection=hook_conn,
                            scratch_redis=scratch_redis,
                            bank_id=bank_id,
                            payment_id=payment_id,
                        )
                        if len(rec_res.mismatches) == 0 and rec_res.logged == rec_res.recomputed:
                            recompute_matches.append(payment_id)
                        else:
                            recompute_mismatches.append(f"{bank_id}/{payment_id}: {rec_res.mismatches}")
                    except ValueError as exc:
                        no_watermark_results.append((payment_id, str(exc)))
        finally:
            await hook_conn.close()

    # Replay A: with before_drop_hook recomputing all logged payment decisions
    result_a = await run_replay(
        recorded_world,
        feed,
        admin_pg=get_test_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        before_drop_hook=verify_recompute_hook,
    )

    # Verify Replay A metrics
    assert result_a.decisions_sent == len(events)
    assert result_a.feature_rows_by_status == {"ok": len(events) - 1, "no_watermark": 1}
    assert len(recompute_matches) == len(events) - 1
    assert len(recompute_mismatches) == 0
    assert len(no_watermark_results) == 1
    assert no_watermark_results[0][0] == recorded_world.output.result.payment_ids[0]
    assert "Feature log row status is 'no_watermark'; nothing to recompute" in no_watermark_results[0][1]
    assert len(result_a.feature_log_sha256) == 64

    # Replay B: fresh workspace on identical recorded world
    result_b = await run_replay(
        recorded_world,
        feed,
        admin_pg=get_test_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
    )

    # Verify bitwise-identical determinism and matching counts
    assert result_b.feature_log_sha256 == result_a.feature_log_sha256
    assert result_b.decisions_sent == result_a.decisions_sent == len(events)
    assert result_b.feature_rows_by_status == result_a.feature_rows_by_status
    assert result_b.unwatermarked_payment_ids == result_a.unwatermarked_payment_ids
