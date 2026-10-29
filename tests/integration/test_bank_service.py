# File: tests/integration/test_bank_service.py
# Purpose: Proves that invoking bank operations via InProcessBankClient produces identical
#          database rows, request hashes, response structures, and idempotency guarantees
#          as requests submitted over HTTP transport via FastAPI endpoints.

from collections.abc import AsyncGenerator
from typing import Any

import asyncpg
import orjson
import pytest
from pydantic import SecretStr

from api.main import app
from api.v1.bank_keys import issue_api_key
from api.v1.bank_service import InProcessBankClient
from api.v1.errors import ApiError
from core.config import config
from core.event_clock import ManualClock
from scripts.run_migrations import apply_pending_migrations
from tests.integration.conftest import (
    connect_or_skip,
    create_schema,
    drop_schema,
    get_test_database_url,
    make_schema_name,
)
from tests.integration.test_bank_endpoints import (
    START_TIME,
    EndpointEnv,
    account_event_payload,
    complaint_payload,
    decision_payload,
    post_json,
    transaction_payload,
)

pytestmark = pytest.mark.anyio

BANKS = ("bank_0", "bank_1", "bank_2")


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

# Explicitly named excluded columns that are inherently non-deterministic between executions:
# synthetic sequential primary keys, generated UUIDs, and database-assigned current timestamps.
EXCLUDED_DECISION_COLUMNS: frozenset[str] = frozenset({"created_at"})
EXCLUDED_TRANSACTION_COLUMNS: frozenset[str] = frozenset({"id", "transaction_id", "created_at"})
EXCLUDED_OUTBOX_COLUMNS: frozenset[str] = frozenset({"id", "transaction_id", "created_at"})
EXCLUDED_ACCOUNT_EVENT_COLUMNS: frozenset[str] = frozenset({"id", "created_at"})
EXCLUDED_COMPLAINT_COLUMNS: frozenset[str] = frozenset({"id", "created_at"})
EXCLUDED_IDEMPOTENCY_COLUMNS: frozenset[str] = frozenset({"created_at", "expires_at"})


def sanitize_row(row_dict: dict[str, Any], excluded_columns: frozenset[str]) -> dict[str, Any]:
    """Filters out excluded volatile columns (e.g. timestamps, autoincrement IDs) for row comparison."""
    return {col: val for col, val in row_dict.items() if col not in excluded_columns}


@pytest.fixture
async def secondary_schema() -> AsyncGenerator[tuple[asyncpg.Connection, asyncpg.Pool], None]:
    """
    Provisions a completely separate isolated database schema with all migrations applied and an active pool.
    This provides an isolated database environment for InProcessBankClient runs to compare against HTTP runs.
    """
    database_url = get_test_database_url()
    schema_name = make_schema_name()
    await create_schema(database_url, schema_name)
    conn = await connect_or_skip(database_url, server_settings={"search_path": schema_name})
    pool: asyncpg.Pool | None = None
    try:
        await apply_pending_migrations(conn)
        pool = await asyncpg.create_pool(
            dsn=database_url, min_size=1, max_size=3, server_settings={"search_path": schema_name}
        )
        assert pool is not None
        yield conn, pool
    finally:
        if pool is not None:
            await pool.close()
        await conn.close()
        await drop_schema(database_url, schema_name)


async def test_decisions_parity_http_vs_in_process(
    env: EndpointEnv, secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool]
) -> None:
    """POST /v1/decisions yields identical stored row, response body, and request_hash across transports."""
    conn_inproc, pool_inproc = secondary_schema
    payload = decision_payload("pmt_parity_dec_001")
    body_bytes = orjson.dumps(payload)
    idem_key = "idem_parity_dec_001"

    # 1. Execute via HTTP
    http_resp = await post_json(env, "bank_0", "/v1/decisions", payload, idem_key)
    assert http_resp.status_code == 200

    # 2. Execute via InProcessBankClient
    client = InProcessBankClient(pool_inproc, "bank_0", env.clock)
    outcome = await client.post("/v1/decisions", body_bytes, idem_key)
    assert outcome.status == 200
    assert outcome.replayed is False
    assert outcome.body == http_resp.json()

    # 3. Compare stored domain records
    row_http = await env.connection.fetchrow(
        "SELECT * FROM bank_decision_requests WHERE payment_id = 'pmt_parity_dec_001'"
    )
    row_inproc = await conn_inproc.fetchrow(
        "SELECT * FROM bank_decision_requests WHERE payment_id = 'pmt_parity_dec_001'"
    )
    assert row_http is not None and row_inproc is not None
    assert sanitize_row(dict(row_http), EXCLUDED_DECISION_COLUMNS) == sanitize_row(
        dict(row_inproc), EXCLUDED_DECISION_COLUMNS
    )

    # 4. Compare idempotency records and request hashes
    idem_http = await env.connection.fetchrow(
        "SELECT * FROM bank_idempotency_keys WHERE idempotency_key = $1", idem_key
    )
    idem_inproc = await conn_inproc.fetchrow(
        "SELECT * FROM bank_idempotency_keys WHERE idempotency_key = $1", idem_key
    )
    assert idem_http is not None and idem_inproc is not None
    assert idem_http["request_hash"] == idem_inproc["request_hash"]
    assert sanitize_row(dict(idem_http), EXCLUDED_IDEMPOTENCY_COLUMNS) == sanitize_row(
        dict(idem_inproc), EXCLUDED_IDEMPOTENCY_COLUMNS
    )


async def test_transactions_parity_http_vs_in_process(
    env: EndpointEnv, secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool]
) -> None:
    """POST /v1/transactions yields identical stored row, outbox payload, and request_hash across transports."""
    conn_inproc, pool_inproc = secondary_schema
    payload = transaction_payload(
        payment_id="pmt_parity_tx_001",
        sender_device_fingerprint="fp_android_v3_parity",
        sender_os_generation=3,
    )
    body_bytes = orjson.dumps(payload)
    idem_key = "idem_parity_tx_001"

    # 1. Execute via HTTP
    http_resp = await post_json(env, "bank_0", "/v1/transactions", payload, idem_key)
    assert http_resp.status_code == 201

    # 2. Execute via InProcessBankClient
    client = InProcessBankClient(pool_inproc, "bank_0", env.clock)
    outcome = await client.post("/v1/transactions", body_bytes, idem_key)
    assert outcome.status == 201
    assert outcome.replayed is False
    assert outcome.body["event_id"] == http_resp.json()["event_id"]
    assert outcome.body["was_replay"] == http_resp.json()["was_replay"]

    # 3. Compare stored transaction records
    row_http = await env.connection.fetchrow(
        "SELECT * FROM transactions WHERE payment_id = 'pmt_parity_tx_001'"
    )
    row_inproc = await conn_inproc.fetchrow(
        "SELECT * FROM transactions WHERE payment_id = 'pmt_parity_tx_001'"
    )
    assert row_http is not None and row_inproc is not None
    assert sanitize_row(dict(row_http), EXCLUDED_TRANSACTION_COLUMNS) == sanitize_row(
        dict(row_inproc), EXCLUDED_TRANSACTION_COLUMNS
    )

    # 4. Compare outbox payloads
    outbox_http = await env.connection.fetchrow(
        """
        SELECT o.* FROM outbox o
        JOIN transactions t ON o.transaction_id = t.transaction_id
        WHERE t.payment_id = 'pmt_parity_tx_001'
        """
    )
    outbox_inproc = await conn_inproc.fetchrow(
        """
        SELECT o.* FROM outbox o
        JOIN transactions t ON o.transaction_id = t.transaction_id
        WHERE t.payment_id = 'pmt_parity_tx_001'
        """
    )
    assert outbox_http is not None and outbox_inproc is not None
    # Compare outer payload after parsing JSON to avoid formatting discrepancies
    payload_http = orjson.loads(outbox_http["payload"])
    payload_inproc = orjson.loads(outbox_inproc["payload"])
    # Disregard generated internal UUIDs and created_at timestamps in outbox payload comparison
    payload_http.pop("transaction_id", None)
    payload_inproc.pop("transaction_id", None)
    payload_http.pop("recorded_at", None)
    payload_inproc.pop("recorded_at", None)
    assert payload_http == payload_inproc

    # 5. Compare idempotency records and request hashes
    idem_http = await env.connection.fetchrow(
        "SELECT * FROM bank_idempotency_keys WHERE idempotency_key = $1", idem_key
    )
    idem_inproc = await conn_inproc.fetchrow(
        "SELECT * FROM bank_idempotency_keys WHERE idempotency_key = $1", idem_key
    )
    assert idem_http is not None and idem_inproc is not None
    assert idem_http["request_hash"] == idem_inproc["request_hash"]


async def test_account_events_parity_http_vs_in_process(
    env: EndpointEnv, secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool]
) -> None:
    """POST /v1/account-events yields identical stored row, response body, and request_hash across transports."""
    conn_inproc, pool_inproc = secondary_schema
    payload = account_event_payload("evt_parity_001")
    body_bytes = orjson.dumps(payload)
    idem_key = "idem_parity_evt_001"

    # 1. Execute via HTTP
    http_resp = await post_json(env, "bank_0", "/v1/account-events", payload, idem_key)
    assert http_resp.status_code == 201

    # 2. Execute via InProcessBankClient
    client = InProcessBankClient(pool_inproc, "bank_0", env.clock)
    outcome = await client.post("/v1/account-events", body_bytes, idem_key)
    assert outcome.status == 201
    assert outcome.replayed is False
    assert outcome.body == http_resp.json()

    # 3. Compare stored domain records
    row_http = await env.connection.fetchrow(
        "SELECT * FROM bank_account_events WHERE event_id = 'evt_parity_001'"
    )
    row_inproc = await conn_inproc.fetchrow(
        "SELECT * FROM bank_account_events WHERE event_id = 'evt_parity_001'"
    )
    assert row_http is not None and row_inproc is not None
    assert sanitize_row(dict(row_http), EXCLUDED_ACCOUNT_EVENT_COLUMNS) == sanitize_row(
        dict(row_inproc), EXCLUDED_ACCOUNT_EVENT_COLUMNS
    )

    # 4. Compare idempotency records and request hashes
    idem_http = await env.connection.fetchrow(
        "SELECT * FROM bank_idempotency_keys WHERE idempotency_key = $1", idem_key
    )
    idem_inproc = await conn_inproc.fetchrow(
        "SELECT * FROM bank_idempotency_keys WHERE idempotency_key = $1", idem_key
    )
    assert idem_http is not None and idem_inproc is not None
    assert idem_http["request_hash"] == idem_inproc["request_hash"]


async def test_complaints_parity_http_vs_in_process(
    env: EndpointEnv, secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool]
) -> None:
    """POST /v1/complaints yields identical stored row, response body, and request_hash across transports."""
    conn_inproc, pool_inproc = secondary_schema
    payload = complaint_payload("cmp_parity_001")
    body_bytes = orjson.dumps(payload)
    idem_key = "idem_parity_cmp_001"

    # 1. Execute via HTTP
    http_resp = await post_json(env, "bank_0", "/v1/complaints", payload, idem_key)
    assert http_resp.status_code == 201

    # 2. Execute via InProcessBankClient
    client = InProcessBankClient(pool_inproc, "bank_0", env.clock)
    outcome = await client.post("/v1/complaints", body_bytes, idem_key)
    assert outcome.status == 201
    assert outcome.replayed is False
    assert outcome.body == http_resp.json()

    # 3. Compare stored domain records
    row_http = await env.connection.fetchrow(
        "SELECT * FROM bank_complaints WHERE complaint_id = 'cmp_parity_001'"
    )
    row_inproc = await conn_inproc.fetchrow(
        "SELECT * FROM bank_complaints WHERE complaint_id = 'cmp_parity_001'"
    )
    assert row_http is not None and row_inproc is not None
    assert sanitize_row(dict(row_http), EXCLUDED_COMPLAINT_COLUMNS) == sanitize_row(
        dict(row_inproc), EXCLUDED_COMPLAINT_COLUMNS
    )

    # 4. Compare idempotency records and request hashes
    idem_http = await env.connection.fetchrow(
        "SELECT * FROM bank_idempotency_keys WHERE idempotency_key = $1", idem_key
    )
    idem_inproc = await conn_inproc.fetchrow(
        "SELECT * FROM bank_idempotency_keys WHERE idempotency_key = $1", idem_key
    )
    assert idem_http is not None and idem_inproc is not None
    assert idem_http["request_hash"] == idem_inproc["request_hash"]


async def test_in_process_replay_returns_replayed_true(
    secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool],
) -> None:
    """An in-process replay with the same key returns replayed=True and cached response."""
    _, pool = secondary_schema
    client = InProcessBankClient(pool, "bank_0", ManualClock(START_TIME))
    payload = decision_payload("pmt_inproc_replay_001")
    body_bytes = orjson.dumps(payload)
    idem_key = "idem_inproc_replay_001"

    first_outcome = await client.post("/v1/decisions", body_bytes, idem_key)
    assert first_outcome.status == 200
    assert first_outcome.replayed is False

    second_outcome = await client.post("/v1/decisions", body_bytes, idem_key)
    assert second_outcome.status == 200
    assert second_outcome.replayed is True
    assert second_outcome.body == first_outcome.body


async def test_in_process_same_key_different_body_raises_409(
    secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool],
) -> None:
    """Reusing an idempotency key with a different body via InProcessBankClient raises 409 ApiError."""
    _, pool = secondary_schema
    client = InProcessBankClient(pool, "bank_0", ManualClock(START_TIME))
    idem_key = "idem_inproc_conflict_001"

    first_payload = decision_payload("pmt_inproc_conflict_001")
    first_payload["amount_paise"] = 100000
    await client.post("/v1/decisions", orjson.dumps(first_payload), idem_key)

    second_payload = decision_payload("pmt_inproc_conflict_001")
    second_payload["amount_paise"] = 200000
    with pytest.raises(ApiError) as exc_info:
        await client.post("/v1/decisions", orjson.dumps(second_payload), idem_key)

    assert exc_info.value.status == 409
    assert exc_info.value.code == "idempotency_key_reused"


async def test_in_process_unknown_path_raises_value_error(
    secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool],
) -> None:
    """Invoking an unmapped route on InProcessBankClient raises a ValueError naming the unknown path."""
    _, pool = secondary_schema
    client = InProcessBankClient(pool, "bank_0", ManualClock(START_TIME))

    with pytest.raises(ValueError, match="Unknown in-process path: '/v1/unknown'"):
        await client.post("/v1/unknown", b"{}", "idem_unmapped_001")

