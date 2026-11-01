# File: tests/integration/test_replay_recording.py
# Purpose: Integration tests for replay recording and run-scoped decision feature logging.
import asyncio
from datetime import datetime, timezone
from typing import Any

import asyncpg
import fakeredis
import orjson
import pytest

from api.v1.bank_scope import bank_scoped_transaction
from api.v1.bank_service import InProcessBankClient
from core.config import config
from core.event_clock import IST, ManualClock
from features.feature_log import read_feature_log_row
from features.ingest_watermark import record_payment_watermark, record_poller_watermark
from tests.integration.conftest import get_test_database_url
from world.bank_connectors import BankTransport, MockBankConnectors, load_bank_modes
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.interactive_records import RunnerSettings, allow_all
from world.interactive_runner import FastRunSettings, prepare_world, run_interactive_world
from world.replay_recording import record_world_for_replay

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

START_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
RUN_ID = "66db9aa439b092ee"


@pytest.fixture
async def env(isolated_schema):
    """Sets up an isolated schema with feature log partitions and an asyncpg pool."""
    connection, schema_name = isolated_schema
    await connection.execute("SELECT create_feature_log_partition($1::date)", START_TIME.date())

    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(),
        min_size=1,
        max_size=3,
        server_settings={"search_path": schema_name},
    )
    assert pool is not None
    try:
        yield {
            "connection": connection,
            "pool": pool,
            "schema_name": schema_name,
        }
    finally:
        await pool.close()


async def test_in_process_client_run_scoped_watermarks_and_feature_logging_override(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Proves that InProcessBankClient with run_scoped_watermarks=True and decision_feature_logging=True
    logs a feature row whose arrival_cutoff is the run-scoped MIN while config.decision_feature_logging_enabled
    remains False.
    """
    # Global config stays False
    monkeypatch.setattr(config, "decision_feature_logging_enabled", False)
    assert config.decision_feature_logging_enabled is False

    connection = env["connection"]
    pool = env["pool"]
    clock = ManualClock(START_TIME)
    redis_client = fakeredis.FakeRedis()

    # Seed run-scoped watermarks
    t_run_min = datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc)
    ts_run_min = int(t_run_min.timestamp())
    record_payment_watermark(redis_client, 0, ts_run_min, run_id=RUN_ID)
    record_poller_watermark(redis_client, "bank_0", "events", ts_run_min + 50, run_id=RUN_ID)
    record_poller_watermark(redis_client, "bank_0", "withdrawals", ts_run_min + 150, run_id=RUN_ID)
    record_poller_watermark(redis_client, "bank_0", "feedback", ts_run_min + 200, run_id=RUN_ID)
    record_poller_watermark(redis_client, "bank_0", "complaints", ts_run_min + 300, run_id=RUN_ID)

    in_process = InProcessBankClient(
        pool=pool,
        bank_id="bank_0",
        clock=clock,
        redis_client=redis_client,
        decision_clock=clock,
        run_scoped_watermarks=True,
        decision_feature_logging=True,
    )

    payment_id = f"{RUN_ID}-dec001"
    payload = {
        "payment_id": payment_id,
        "sender_account_id": "acc_001",
        "receiver_bank_id": "bank_1",
        "receiver_vpa": "acc_recv_1",
        "payee_type": "person",
        "amount_paise": 75000,
        "event_time": START_TIME.isoformat(),
    }
    body_bytes = orjson.dumps(payload)

    outcome = await in_process.post("/v1/decisions", body_bytes, idempotency_key="idem_run_scoped_01")
    assert outcome.status == 200
    assert outcome.body["decision"] == "unscored"

    # Invariant: Global config remained False throughout
    assert config.decision_feature_logging_enabled is False

    # Verify that feature log row was written with arrival_cutoff equal to run MIN
    async with bank_scoped_transaction(connection, "bank_0") as scoped:
        log_row = await read_feature_log_row(scoped, "bank_0", payment_id)
        assert log_row is not None
        assert log_row.payment_id == payment_id
        assert log_row.arrival_cutoff == t_run_min


class RejectingTransport:
    """Mock transport that ensures no network or database calls occur during recording."""

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> Any:
        raise AssertionError("Transport should not be called")


async def test_record_world_for_replay_records_calls_and_matches_baseline() -> None:
    """
    Test 1: Run with FastRunSettings(101, 20, 6, 51, "training") and RunnerSettings(fraud_enabled=True).
    Connectors uses a fake transport whose post() raises AssertionError('Transport should not be called').
    Assert:
      recorded calls match hook invocations in order (payment.planned_payment_id == call.idempotency_key stripped
      of 'dec:')
      every call.body_bytes is valid JSON with the expected fields
      world output equals plain run_interactive_world(fast_settings, runner_settings, hook=allow_all) — same
      run_id, same event counts, same payment IDs.
    """
    fast_settings = FastRunSettings(101, 20, 6, 51, "training")
    runner_settings = RunnerSettings(fraud_enabled=True)

    prepared = prepare_world(fast_settings, None)
    clock_start = prepared.plan.clock.start_date
    cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()

    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

    transports: dict[str, BankTransport] = {b: RejectingTransport() for b in bank_modes}
    loop = asyncio.get_running_loop()
    connectors = MockBankConnectors(
        transports=transports,
        modes=bank_modes,
        account_bank_map=account_bank_map,
        clock_start=clock_start,
        feed_config=cfg,
        decision_timeout_seconds=cfg.decision_timeout_seconds,
        loop=loop,
    )

    recorded_world = record_world_for_replay(fast_settings, runner_settings, connectors)
    plain_output = run_interactive_world(fast_settings, runner_settings, hook=allow_all)

    # 1. Output parity with plain allow_all execution
    assert recorded_world.output.result.run_id == plain_output.result.run_id
    assert len(recorded_world.output.result.run_result.events) == len(plain_output.result.run_result.events)
    assert recorded_world.output.result.payment_ids == plain_output.result.payment_ids

    # 2. Recorded calls non-empty and matching hook invocations in order
    assert len(recorded_world.calls) > 0
    expected_fields = {
        "payment_id",
        "sender_account_id",
        "receiver_bank_id",
        "amount_paise",
        "event_time",
        "channel",
        "payee_type",
    }
    for payment, call in recorded_world.calls:
        assert call.idempotency_key == f"dec:{payment.planned_payment_id}"
        assert call.idempotency_key.removeprefix("dec:") == payment.planned_payment_id
        body = orjson.loads(call.body_bytes)
        assert isinstance(body, dict)
        assert expected_fields.issubset(body.keys())
        assert body["payment_id"] == payment.planned_payment_id
        assert body["sender_account_id"] == payment.from_account
        assert body["amount_paise"] == payment.amount_paise
        assert body["receiver_bank_id"] == call.receiver_bank_id


async def test_record_world_for_replay_refuses_live_bank_mode() -> None:
    """
    Test 2: When a connector bank has mode 'live', record_world_for_replay raises ValueError before
    run_interactive_world runs (assert with pytest.raises(ValueError, match="Live bank mode refused until Level 19")).
    """
    fast_settings = FastRunSettings(101, 20, 6, 51, "training")
    runner_settings = RunnerSettings(fraud_enabled=True)

    prepared = prepare_world(fast_settings, None)
    clock_start = prepared.plan.clock.start_date
    cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()
    # Force one bank to live
    first_bank = next(iter(bank_modes.keys()))
    live_modes = dict(bank_modes)
    live_modes[first_bank] = "live"

    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

    transports: dict[str, BankTransport] = {b: RejectingTransport() for b in live_modes}
    loop = asyncio.get_running_loop()
    connectors = MockBankConnectors(
        transports=transports,
        modes=live_modes,
        account_bank_map=account_bank_map,
        clock_start=clock_start,
        feed_config=cfg,
        decision_timeout_seconds=cfg.decision_timeout_seconds,
        loop=loop,
    )

    with pytest.raises(ValueError, match="Live bank mode refused until Level 19"):
        record_world_for_replay(fast_settings, runner_settings, connectors)


