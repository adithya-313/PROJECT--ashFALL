# File: tests/integration/test_bank_clients.py
# Purpose: Integration tests proving SignedHttpBankClient against real FastAPI endpoints over ASGITransport.
#
# Verifies that SignedHttpBankClient correctly signs requests, transmits idempotency keys,
# interprets 2xx responses and Idempotent-Replayed headers, and converts 4xx errors into BankCallError.

from collections.abc import AsyncGenerator
from typing import Any

import asyncpg
import httpx
import orjson
import pytest
from pydantic import SecretStr

from api.main import app
from api.v1.bank_clients import BankCallError, SignedHttpBankClient
from api.v1.bank_keys import issue_api_key
from core.config import config
from core.event_clock import ManualClock
from tests.integration.conftest import get_test_database_url
from tests.integration.test_bank_endpoints import (
    BANKS,
    START_TIME,
    EndpointEnv,
    decision_payload,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
async def env(isolated_schema: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[EndpointEnv, None]:
    """Gives the real app a pool on the isolated schema and a ManualClock, issues keys, and restores app.state."""
    connection, schema_name = isolated_schema
    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(), min_size=1, max_size=3, server_settings={"search_path": schema_name}
    )
    assert pool is not None
    for bank_id in BANKS:
        secret_text = f"test_secret_for_{bank_id}_32_chars_min!"
        monkeypatch.setattr(config, f"bank_signing_secret_{bank_id}", SecretStr(secret_text))
    api_keys: dict[str, str] = {}
    for bank_id in BANKS:
        _, api_keys[bank_id] = await issue_api_key(connection, bank_id)

    clock = ManualClock(START_TIME)
    saved_state: dict[str, Any] = {}
    for name in ("db_pool", "transport_clock"):
        if hasattr(app.state, name):
            saved_state[name] = getattr(app.state, name)
    app.state.db_pool = pool
    app.state.transport_clock = clock
    try:
        yield EndpointEnv(connection=connection, clock=clock, api_keys=api_keys)
    finally:
        for name in ("db_pool", "transport_clock"):
            if name in saved_state:
                setattr(app.state, name, saved_state[name])
            else:
                delattr(app.state, name)
        await pool.close()


async def test_signed_http_bank_client_decision_post_and_replay(env: EndpointEnv) -> None:
    """Proves that SignedHttpBankClient handles initial execution (replayed=False) and replay (replayed=True)."""
    payload = decision_payload("pmt_client_001")
    body_bytes = orjson.dumps(payload)
    idempotency_key = "dec:pmt_client_001"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        client = SignedHttpBankClient(
            http_client=http_client,
            bank_id="bank_0",
            api_key=env.api_keys["bank_0"],
            clock=env.clock,
        )

        # 1. Initial submission -> 200 OK, replayed=False
        outcome1 = await client.post("/v1/decisions", body_bytes, idempotency_key)
        assert outcome1.status == 200
        assert outcome1.replayed is False
        assert outcome1.body["decision"] == "unscored"
        assert outcome1.body["payment_id"] == "pmt_client_001"

        # 2. Identical replay submission -> 200 OK, replayed=True
        outcome2 = await client.post("/v1/decisions", body_bytes, idempotency_key)
        assert outcome2.status == 200
        assert outcome2.replayed is True
        assert outcome2.body["decision"] == "unscored"
        assert outcome2.body["payment_id"] == "pmt_client_001"


async def test_signed_http_bank_client_conflict_raises_bank_call_error(env: EndpointEnv) -> None:
    """Proves that a 409 conflict (reusing same key with different body) raises BankCallError."""
    payload1 = decision_payload("pmt_client_002")
    body1 = orjson.dumps(payload1)
    idempotency_key = "dec:pmt_client_002"

    payload2 = decision_payload("pmt_client_002")
    payload2["amount_paise"] = 999999
    body2 = orjson.dumps(payload2)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        client = SignedHttpBankClient(
            http_client=http_client,
            bank_id="bank_0",
            api_key=env.api_keys["bank_0"],
            clock=env.clock,
        )

        outcome1 = await client.post("/v1/decisions", body1, idempotency_key)
        assert outcome1.status == 200
        assert outcome1.replayed is False

        with pytest.raises(BankCallError) as exc_info:
            await client.post("/v1/decisions", body2, idempotency_key)

        assert exc_info.value.status == 409
        assert exc_info.value.code == "idempotency_key_reused"
        assert "idempotency_key_reused" in str(exc_info.value)
