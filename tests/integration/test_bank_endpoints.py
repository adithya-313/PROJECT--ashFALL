# File: tests/integration/test_bank_endpoints.py
# Purpose: Proves the /v1 bank endpoints through the real FastAPI app: contracts, authentication before parsing,
#          idempotency, bank isolation and input validation.
#
# The app is called through httpx.ASGITransport WITHOUT running its lifespan. Each test gives the app a pool on its
# own throwaway schema and a ManualClock for transport time, signs requests with the real signing code, and puts
# the app's state back afterwards.
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import asyncpg
import httpx
import orjson
import pytest
from pydantic import SecretStr

from api.main import app
from api.v1.bank_keys import issue_api_key
from api.v1.signing import sign_outgoing
from core.config import config
from core.event_clock import IST, ManualClock
from tests.integration.conftest import get_test_database_url

pytestmark = pytest.mark.anyio

BANKS = ("bank_0", "bank_1", "bank_2")
START_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
OLD_EVENT_TIME = "2020-01-01T10:00:00+05:30"
RECENT_EVENT_TIME = "2026-10-28T11:30:00+05:30"


@dataclass
class EndpointEnv:
    """Everything a test needs: the owner connection, the transport clock and one API key per bank."""

    connection: asyncpg.Connection
    clock: ManualClock
    api_keys: dict[str, str]


@pytest.fixture
async def env(isolated_schema, monkeypatch: pytest.MonkeyPatch):
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


def signed_headers(
    env: EndpointEnv,
    bank_id: str,
    method: str,
    path: str,
    body: bytes,
    idempotency_key: str | None = None,
    signing_clock: ManualClock | None = None,
) -> dict[str, str]:
    """Signs a request with the real signing code, then swaps in the bank's issued API key for the key header."""
    nonce = uuid.uuid4().hex
    headers = sign_outgoing(bank_id, method, path, body, signing_clock or env.clock, nonce)
    headers["X-Ashfall-Key"] = env.api_keys[bank_id]
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


async def post_json(
    env: EndpointEnv,
    bank_id: str,
    path: str,
    payload: dict[str, Any],
    idempotency_key: str,
    signing_clock: ManualClock | None = None,
) -> httpx.Response:
    """Sends one signed POST with a JSON body to the real app."""
    body = orjson.dumps(payload)
    headers = signed_headers(env, bank_id, "POST", path, body, idempotency_key, signing_clock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, content=body, headers=headers)


async def get_signed(env: EndpointEnv, bank_id: str, path: str) -> httpx.Response:
    """Sends one signed GET to the real app."""
    headers = signed_headers(env, bank_id, "GET", path, b"")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path, headers=headers)


def transaction_payload(payment_id: str = "pmt_ep_001", **overrides: Any) -> dict[str, Any]:
    """A valid debit side from bank_0 to bank_1; tests change single fields through overrides."""
    payload: dict[str, Any] = {
        "payment_id": payment_id,
        "side": "debit",
        "account_id": "acc_bank0_a",
        "counterparty_vpa": "shop@bank1",
        "counterparty_bank_id": "bank_1",
        "sender_bank_id": "bank_0",
        "receiver_bank_id": "bank_1",
        "amount_paise": 150000,
        "event_time": RECENT_EVENT_TIME,
        "status": "success",
    }
    payload.update(overrides)
    return payload


def decision_payload(payment_id: str = "pmt_dec_001") -> dict[str, Any]:
    """A valid decision request."""
    return {
        "payment_id": payment_id,
        "sender_account_id": "acc_bank0_a",
        "receiver_bank_id": "bank_1",
        "receiver_vpa": "shop@bank1",
        "amount_paise": 250000,
        "event_time": RECENT_EVENT_TIME,
        "channel": "upi_app",
        "payee_type": "merchant",
    }


def account_event_payload(event_id: str = "evt_ep_001") -> dict[str, Any]:
    """A valid account event."""
    return {
        "event_id": event_id,
        "account_id": "acc_bank0_a",
        "kind": "sim_swap",
        "event_time": RECENT_EVENT_TIME,
        "details": {"carrier": "example"},
    }


async def test_decisions_contract(env: EndpointEnv) -> None:
    """POST /v1/decisions answers 200 with an unscored decision in shadow mode and stores one request row."""
    response = await post_json(env, "bank_0", "/v1/decisions", decision_payload(), "idem_decision_0001")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"payment_id", "decision", "mode", "reason"}
    assert body["payment_id"] == "pmt_dec_001"
    assert body["decision"] == "unscored"
    assert body["mode"] == "shadow"
    assert body["reason"] == "scoring is not enabled until Level 19"
    rows = await env.connection.fetch("SELECT bank_id, payment_id, decision, mode FROM bank_decision_requests")
    assert [tuple(row.values()) for row in rows] == [("bank_0", "pmt_dec_001", "unscored", "shadow")]


async def test_same_payment_under_a_different_key_is_already_decided(env: EndpointEnv) -> None:
    """The same payment id with a new idempotency key is a 409 payment_already_decided, with nothing extra stored."""
    first = await post_json(env, "bank_0", "/v1/decisions", decision_payload(), "idem_decision_0001")
    second = await post_json(env, "bank_0", "/v1/decisions", decision_payload(), "idem_decision_0002")

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "payment_already_decided"
    assert await env.connection.fetchval("SELECT count(*) FROM bank_decision_requests") == 1
    assert await env.connection.fetchval("SELECT count(*) FROM bank_idempotency_keys") == 1


async def test_transactions_contract(env: EndpointEnv) -> None:
    """POST /v1/transactions answers 201 with ids and was_replay false, and stores the row and its outbox event."""
    response = await post_json(env, "bank_0", "/v1/transactions", transaction_payload(), "idem_transaction_001")

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"transaction_id", "event_id", "was_replay"}
    assert body["was_replay"] is False
    row = await env.connection.fetchrow("SELECT bank_id, payment_id, side, amount_paise FROM transactions")
    stored = (row["bank_id"], row["payment_id"], row["side"], row["amount_paise"])
    assert stored == ("bank_0", "pmt_ep_001", "debit", 150000)
    assert await env.connection.fetchval("SELECT count(*) FROM outbox") == 1


async def test_account_events_contract(env: EndpointEnv) -> None:
    """POST /v1/account-events answers 201 once, and 200 duplicate true when the same event id comes under a new key."""
    first = await post_json(env, "bank_0", "/v1/account-events", account_event_payload(), "idem_event_0000001")
    second = await post_json(env, "bank_0", "/v1/account-events", account_event_payload(), "idem_event_0000002")

    assert first.status_code == 201
    assert first.json() == {"event_id": "evt_ep_001", "duplicate": False}
    assert second.status_code == 200
    assert second.json() == {"event_id": "evt_ep_001", "duplicate": True}
    assert await env.connection.fetchval("SELECT count(*) FROM bank_account_events") == 1


async def test_account_risk_contract(env: EndpointEnv) -> None:
    """GET /v1/accounts/{id}/risk answers 200 unscored for an account the bank has a transaction for."""
    await post_json(env, "bank_0", "/v1/transactions", transaction_payload(), "idem_transaction_001")

    response = await get_signed(env, "bank_0", "/v1/accounts/acc_bank0_a/risk")

    assert response.status_code == 200
    assert response.json() == {"account_id": "acc_bank0_a", "risk": "unscored", "mode": "shadow"}


async def test_account_risk_for_an_invalid_id_is_422(env: EndpointEnv) -> None:
    """An account id with characters outside the identifier alphabet is refused with 422 after authentication."""
    # The server checks the signature over the decoded path (request.url.path), so the test signs the path with a
    # real space; httpx then sends it as %20 on the wire.
    response = await get_signed(env, "bank_0", "/v1/accounts/bad id/risk")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_account_id"


async def test_same_key_same_body_creates_one_record(env: EndpointEnv) -> None:
    """Sending the same key and body twice stores one transaction and one outbox event, and replays the answer."""
    payload = transaction_payload()
    first = await post_json(env, "bank_0", "/v1/transactions", payload, "idem_transaction_001")
    second = await post_json(env, "bank_0", "/v1/transactions", payload, "idem_transaction_001")

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json() == first.json()
    assert "idempotent-replayed" not in first.headers
    assert second.headers["Idempotent-Replayed"] == "true"
    assert await env.connection.fetchval("SELECT count(*) FROM transactions") == 1
    assert await env.connection.fetchval("SELECT count(*) FROM outbox") == 1


async def test_same_key_different_body_returns_409(env: EndpointEnv) -> None:
    """Reusing an idempotency key with a different body is a 409 idempotency_key_reused and stores nothing new."""
    first = await post_json(env, "bank_0", "/v1/transactions", transaction_payload(), "idem_transaction_001")
    second = await post_json(
        env, "bank_0", "/v1/transactions", transaction_payload(amount_paise=999), "idem_transaction_001"
    )

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "idempotency_key_reused"
    assert await env.connection.fetchval("SELECT count(*) FROM transactions") == 1


async def test_fast_replay_old_event_time_fresh_signature_accepted(env: EndpointEnv) -> None:
    """A payment from 2020 is accepted when its signature is fresh: freshness is transport time, not event time."""
    response = await post_json(
        env, "bank_0", "/v1/transactions", transaction_payload(event_time=OLD_EVENT_TIME), "idem_transaction_001"
    )

    assert 200 <= response.status_code < 300


async def test_stale_signature_refused(env: EndpointEnv) -> None:
    """A signature whose timestamp is 301 seconds old is refused with 401 stale_signature."""
    stale_clock = ManualClock(START_TIME - timedelta(seconds=301))

    response = await post_json(
        env, "bank_0", "/v1/transactions", transaction_payload(), "idem_transaction_001", signing_clock=stale_clock
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "stale_signature"


async def test_unsigned_malformed_body_gets_401_not_422(env: EndpointEnv) -> None:
    """An unsigned request with a broken body is answered 401: authentication runs before the body is parsed."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/transactions", content=b"{this is not json", headers={})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_auth_headers"


async def test_bank_1_cannot_see_bank_2_account_risk(env: EndpointEnv) -> None:
    """Bank 1 gets the same 404 for bank 2's account as for an account that does not exist."""
    payload = transaction_payload(
        "pmt_bank2_001", account_id="acc_bank2_x", sender_bank_id="bank_2", receiver_bank_id="bank_1",
        counterparty_bank_id="bank_1",
    )
    stored = await post_json(env, "bank_2", "/v1/transactions", payload, "idem_bank2_00001")
    assert stored.status_code == 201

    other_banks_account = await get_signed(env, "bank_1", "/v1/accounts/acc_bank2_x/risk")
    no_such_account = await get_signed(env, "bank_1", "/v1/accounts/acc_nobody_at_all/risk")
    owner_view = await get_signed(env, "bank_2", "/v1/accounts/acc_bank2_x/risk")

    assert other_banks_account.status_code == 404
    assert no_such_account.status_code == 404
    assert other_banks_account.json() == no_such_account.json()
    assert other_banks_account.json()["error"]["code"] == "account_not_found"
    assert owner_view.status_code == 200


async def test_body_with_bank_id_field_is_rejected(env: EndpointEnv) -> None:
    """A body that names a bank_id is refused with 422 listing the field, because the bank comes from the key."""
    payload = transaction_payload()
    payload["bank_id"] = "bank_1"

    response = await post_json(env, "bank_0", "/v1/transactions", payload, "idem_transaction_001")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_body"
    assert "bank_id" in error["message"]
    assert "bank_1" not in error["message"]
    assert await env.connection.fetchval("SELECT count(*) FROM transactions") == 0


async def test_transaction_side_bank_must_match_authenticated_bank(env: EndpointEnv) -> None:
    """With bank 1's key, a debit side whose sender bank is bank 2 is refused with 422 and stores nothing."""
    payload = transaction_payload(sender_bank_id="bank_2")

    response = await post_json(env, "bank_1", "/v1/transactions", payload, "idem_transaction_001")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_body"
    assert await env.connection.fetchval("SELECT count(*) FROM transactions") == 0
