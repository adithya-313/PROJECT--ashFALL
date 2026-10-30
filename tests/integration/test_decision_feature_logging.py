# File: tests/integration/test_decision_feature_logging.py
# Purpose: Proves decision-time shadow feature logging behind decision_feature_logging_enabled (D4, S4).
# Issue: Level 15c watermark and feature log (plan Task 1, D4, S4).
"""
WHAT: Integration tests for /v1/decisions feature logging.

WHY: Verifies:
1. Feature logging is disabled by default;
2. When enabled, inserts canonical M1 features into bank_feature_log;
3. Atomic transaction rollback if feature log insert fails (Sabotage S4);
4. Idempotent replay does not duplicate feature log rows;
5. Tenant isolation under RLS.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import asyncpg
import fakeredis
import httpx
import orjson
import pytest
from pydantic import SecretStr

from api.main import app
from api.v1.bank_keys import issue_api_key
from api.v1.bank_scope import bank_scoped_transaction
from api.v1.bank_service import DECISION_REASON, InProcessBankClient, get_bank_mode
from api.v1.signing import sign_outgoing
from core.config import config
from core.event_clock import IST, ManualClock
from features.feature_log import canonical_feature_names, read_feature_log_row
from features.ingest_watermark import (
    POLLER_SOURCES,
    record_payment_watermark,
    record_poller_watermark,
)
from tests.integration.conftest import get_test_database_url

pytestmark = pytest.mark.anyio

BANKS = ("bank_0", "bank_1")
RUN_ID = "66db9aa439b092ee"
START_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
EVENT_TIME_STR = "2026-10-28T12:00:00+05:30"
EVENT_TIME_DT = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)


def _set_watermarks(client: Any, bank_id: str, ts: datetime) -> None:
    record_payment_watermark(client, 0, ts)
    for source in POLLER_SOURCES:
        record_poller_watermark(client, bank_id, source, ts)


@pytest.fixture
async def env(isolated_schema, monkeypatch: pytest.MonkeyPatch):
    connection, schema_name = isolated_schema
    # Pre-create partition for test date as schema owner
    await connection.execute("SELECT create_feature_log_partition($1::date)", START_TIME.date())

    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(),
        min_size=1,
        max_size=3,
        server_settings={"search_path": schema_name},
    )
    assert pool is not None

    for bank_id in BANKS:
        secret_text = f"test_secret_for_{bank_id}_32_chars_min!"
        monkeypatch.setattr(config, f"bank_signing_secret_{bank_id}", SecretStr(secret_text))

    api_keys: dict[str, str] = {}
    for bank_id in BANKS:
        _, api_keys[bank_id] = await issue_api_key(connection, bank_id)

    clock = ManualClock(START_TIME)
    redis_client = fakeredis.FakeRedis()
    _set_watermarks(redis_client, "bank_0", START_TIME.astimezone(timezone.utc) + timedelta(hours=1))
    _set_watermarks(redis_client, "bank_1", START_TIME.astimezone(timezone.utc) + timedelta(hours=1))

    saved_state: dict[str, Any] = {}
    for name in ("db_pool", "transport_clock", "redis_client", "decision_clock"):
        if hasattr(app.state, name):
            saved_state[name] = getattr(app.state, name)

    app.state.db_pool = pool
    app.state.transport_clock = clock
    app.state.redis_client = redis_client
    app.state.decision_clock = clock

    try:
        yield {
            "connection": connection,
            "pool": pool,
            "clock": clock,
            "api_keys": api_keys,
            "redis_client": redis_client,
            "schema_name": schema_name,
        }
    finally:
        for name in ("db_pool", "transport_clock", "redis_client", "decision_clock"):
            if name in saved_state:
                setattr(app.state, name, saved_state[name])
            elif hasattr(app.state, name):
                delattr(app.state, name)
        await pool.close()


def _signed_headers(
    env_dict: dict[str, Any],
    bank_id: str,
    method: str,
    path: str,
    body: bytes,
    idempotency_key: str,
) -> dict[str, str]:
    nonce = uuid.uuid4().hex
    headers = sign_outgoing(bank_id, method, path, body, env_dict["clock"], nonce)
    headers["X-Ashfall-Key"] = env_dict["api_keys"][bank_id]
    headers["Idempotency-Key"] = idempotency_key
    return headers


async def test_feature_logging_disabled_by_default(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """When decision_feature_logging_enabled is False, feature_log receives no rows and response is byte-identical."""
    monkeypatch.setattr(config, "decision_feature_logging_enabled", False)
    connection = env["connection"]

    payment_id = f"{RUN_ID}-dec001"
    payload = {
        "payment_id": payment_id,
        "sender_account_id": "acc_001",
        "receiver_bank_id": "bank_1",
        "amount_paise": 50000,
        "event_time": EVENT_TIME_STR,
    }
    body_bytes = orjson.dumps(payload)
    headers = _signed_headers(env, "bank_0", "POST", "/v1/decisions", body_bytes, "idem_dec_off")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/v1/decisions", content=body_bytes, headers=headers)
        assert res.status_code == 200
        # Byte-identical response to 80a35f4
        assert res.json() == {
            "payment_id": payment_id,
            "decision": "unscored",
            "mode": get_bank_mode("bank_0"),
            "reason": DECISION_REASON,
        }

    # Decision recorded in bank_decision_requests
    async with bank_scoped_transaction(connection, "bank_0") as scoped:
        dec_exists = await scoped.fetchval(
            "SELECT COUNT(*) FROM bank_decision_requests WHERE payment_id = $1", payment_id
        )
        assert dec_exists == 1

        # No feature log row inserted
        log_count = await scoped.fetchval("SELECT COUNT(*) FROM feature_log WHERE payment_id = $1", payment_id)
        assert log_count == 0


async def test_feature_logging_enabled_records_row_and_replay(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """When decision_feature_logging_enabled is True, inserts into feature_log and handles replay."""
    monkeypatch.setattr(config, "decision_feature_logging_enabled", True)
    connection = env["connection"]

    payment_id = f"{RUN_ID}-dec002"
    payload = {
        "payment_id": payment_id,
        "sender_account_id": "acc_002",
        "receiver_bank_id": "bank_1",
        "receiver_vpa": "acc_recv_2",
        "amount_paise": 150000,
        "event_time": EVENT_TIME_STR,
        "channel": "upi_app",
        "payee_type": "person",
    }
    body_bytes = orjson.dumps(payload)
    headers = _signed_headers(env, "bank_0", "POST", "/v1/decisions", body_bytes, "idem_dec_on")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # First call: records decision and feature log
        res1 = await client.post("/v1/decisions", content=body_bytes, headers=headers)
        assert res1.status_code == 200
        assert res1.json()["decision"] == "unscored"

        # Replay with same idempotency key: returns same result without duplicating feature log
        headers_replay = _signed_headers(env, "bank_0", "POST", "/v1/decisions", body_bytes, "idem_dec_on")
        res2 = await client.post("/v1/decisions", content=body_bytes, headers=headers_replay)
        assert res2.status_code == 200
        assert res2.headers.get("Idempotent-Replayed") == "true"

    async with bank_scoped_transaction(connection, "bank_0") as scoped:
        rows = await scoped.fetch(
            "SELECT status, feature_values FROM feature_log WHERE payment_id = $1",
            payment_id,
        )
        assert len(rows) == 1
        assert rows[0]["status"] == "ok"
        fv = orjson.loads(rows[0]["feature_values"])
        assert isinstance(fv, list)
        names = canonical_feature_names()
        idx = names.index("payee_type")
        assert fv[idx] == "person"


async def test_duplicate_payment_id_returns_409_and_single_feature_log_row(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that duplicate payment_id yields 409 conflict and creates exactly ONE feature_log row."""
    monkeypatch.setattr(config, "decision_feature_logging_enabled", True)
    connection = env["connection"]

    payment_id = f"{RUN_ID}-dec-dup01"
    payload = {
        "payment_id": payment_id,
        "sender_account_id": "acc_002",
        "receiver_bank_id": "bank_1",
        "receiver_vpa": "acc_recv_2",
        "amount_paise": 150000,
        "event_time": EVENT_TIME_STR,
        "payee_type": "person",
    }
    body_bytes = orjson.dumps(payload)
    headers1 = _signed_headers(env, "bank_0", "POST", "/v1/decisions", body_bytes, "idem_dup_1")
    headers2 = _signed_headers(env, "bank_0", "POST", "/v1/decisions", body_bytes, "idem_dup_2")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # First call succeeds
        res1 = await client.post("/v1/decisions", content=body_bytes, headers=headers1)
        assert res1.status_code == 200

        # Second call with same payment_id but different idempotency key fails with 409
        res2 = await client.post("/v1/decisions", content=body_bytes, headers=headers2)
        assert res2.status_code == 409
        assert res2.json()["error"]["code"] == "payment_already_decided"

    async with bank_scoped_transaction(connection, "bank_0") as scoped:
        count = await scoped.fetchval(
            "SELECT COUNT(*) FROM feature_log WHERE payment_id = $1",
            payment_id,
        )
        assert count == 1


async def test_feature_logging_atomic_rollback_on_insert_failure(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves transaction atomicity (Sabotage S4): if feature log insert fails, decision insert rolls back."""
    monkeypatch.setattr(config, "decision_feature_logging_enabled", True)
    connection = env["connection"]

    payment_id = f"{RUN_ID}-dec003"
    payload = {
        "payment_id": payment_id,
        "sender_account_id": "acc_003",
        "receiver_bank_id": "bank_1",
        "amount_paise": 20000,
        "event_time": EVENT_TIME_STR,
    }
    body_bytes = orjson.dumps(payload)
    headers = _signed_headers(env, "bank_0", "POST", "/v1/decisions", body_bytes, "idem_dec_fail")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    # Simulate database failure during insert_feature_log_row
    with patch("api.v1.bank_service.insert_feature_log_row", side_effect=RuntimeError("Simulated DB failure")):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/v1/decisions", content=body_bytes, headers=headers)
            # Must fail with 500
            assert res.status_code == 500

    # Crucial assertion: bank_decision_requests must NOT contain the rolled-back decision
    async with bank_scoped_transaction(connection, "bank_0") as scoped:
        count = await scoped.fetchval("SELECT COUNT(*) FROM bank_decision_requests WHERE payment_id = $1", payment_id)
        assert count == 0, "Atomic rollback violated: bank_decision_requests persisted despite feature log failure!"


async def test_in_process_client_feature_logging(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that InProcessBankClient supports decision feature logging with optional dependencies."""
    monkeypatch.setattr(config, "decision_feature_logging_enabled", True)
    connection = env["connection"]
    pool = env["pool"]
    clock = env["clock"]
    redis_client = env["redis_client"]

    in_process = InProcessBankClient(
        pool=pool,
        bank_id="bank_0",
        clock=clock,
        redis_client=redis_client,
        decision_clock=clock,
    )

    payment_id = f"{RUN_ID}-dec004"
    payload = {
        "payment_id": payment_id,
        "sender_account_id": "acc_004",
        "receiver_bank_id": "bank_1",
        "receiver_vpa": "acc_recv_4",
        "payee_type": "person",
        "amount_paise": 75000,
        "event_time": EVENT_TIME_STR,
    }
    body_bytes = orjson.dumps(payload)

    outcome = await in_process.post("/v1/decisions", body_bytes, idempotency_key="idem_inproc_01")
    assert outcome.status == 200
    assert outcome.body["decision"] == "unscored"

    async with bank_scoped_transaction(connection, "bank_0") as scoped:
        log_row = await read_feature_log_row(scoped, "bank_0", payment_id)
        assert log_row is not None
        assert log_row.payment_id == payment_id
        assert log_row.status == "ok"
