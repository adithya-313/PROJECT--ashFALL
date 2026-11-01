# File: tests/integration/test_replay_recording.py
# Purpose: Integration tests for replay recording and run-scoped decision feature logging.
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

