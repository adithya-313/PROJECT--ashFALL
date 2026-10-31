# File: tests/integration/test_bank_feed_additions_end_to_end.py
# Purpose: End-to-end integration tests for Level 15b bank feed additions: migration 0009, withdrawals, and balance.

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import asyncpg
import httpx
import pytest
from pydantic import SecretStr

from api.main import app
from api.v1.bank_clients import SignedHttpBankClient
from api.v1.bank_keys import issue_api_key
from api.v1.bank_service import InProcessBankClient
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
    BANKS,
    START_TIME,
    EndpointEnv,
)

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

EXCLUDED_WITHDRAWAL_COLUMNS: frozenset[str] = frozenset({"id", "recorded_at"})
EXCLUDED_TRANSACTION_COLUMNS: frozenset[str] = frozenset({"id", "transaction_id", "created_at"})


def sanitize_row(row_dict: dict[str, Any], excluded_columns: frozenset[str]) -> dict[str, Any]:
    """Filters out excluded volatile columns for row comparison."""
    return {col: val for col, val in row_dict.items() if col not in excluded_columns}


@pytest.fixture
async def env(isolated_schema: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[EndpointEnv, None]:
    """Gives the real app a pool on the isolated schema and a ManualClock, issues keys, and restores app.state."""
    connection, schema_name = isolated_schema
    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(), min_size=1, max_size=5, server_settings={"search_path": schema_name}
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


@pytest.fixture
async def secondary_schema() -> AsyncGenerator[tuple[asyncpg.Connection, asyncpg.Pool], None]:
    """Provisions a separate isolated schema and pool for InProcess vs HTTP parity testing."""
    database_url = get_test_database_url()
    schema_name = make_schema_name()
    await create_schema(database_url, schema_name)
    conn = await connect_or_skip(database_url, server_settings={"search_path": schema_name})
    pool: asyncpg.Pool | None = None
    try:
        await apply_pending_migrations(conn)
        pool = await asyncpg.create_pool(
            dsn=database_url, min_size=1, max_size=5, server_settings={"search_path": schema_name}
        )
        assert pool is not None
        yield conn, pool
    finally:
        if pool is not None:
            await pool.close()
        await conn.close()
        await drop_schema(database_url, schema_name)


async def test_migration_0009_structure(env: EndpointEnv) -> None:
    """Proves migration 0009 added balance_after_paise to transactions and created bank_withdrawals table."""
    # 1. Check transactions column
    col_row = await env.connection.fetchrow(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'transactions' AND column_name = 'balance_after_paise'
        """
    )
    assert col_row is not None
    assert col_row["data_type"] == "bigint"
    assert col_row["is_nullable"] == "YES"

    # 2. Check bank_withdrawals table exists
    tbl_row = await env.connection.fetchrow(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'bank_withdrawals'
        """
    )
    assert tbl_row is not None

    # 3. Check bank_withdrawals columns
    cols = await env.connection.fetch(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'bank_withdrawals'
        ORDER BY ordinal_position
        """
    )
    col_map = {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in cols}
    assert "withdrawal_id" in col_map
    assert "bank_id" in col_map
    assert "account_id" in col_map
    assert "channel" in col_map
    assert "amount_paise" in col_map
    assert "balance_after_paise" in col_map
    assert "status" in col_map
    assert "failure_domain" in col_map
    assert "event_time" in col_map
    assert "recorded_at" in col_map


async def test_withdrawal_in_process_vs_signed_http_parity(
    env: EndpointEnv,
    secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool],
) -> None:
    """
    Proves InProcessBankClient and SignedHttpBankClient record withdrawals with
    100% row parity and 201/200 idempotency.
    """
    conn_proc, pool_proc = secondary_schema
    bank_id = "bank_0"
    event_time = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)

    # InProcess client on secondary schema
    proc_client = InProcessBankClient(pool_proc, bank_id, env.clock)
    withdrawal_payload = {
        "bank_id": bank_id,
        "withdrawal_id": "wth_test_001",
        "account_id": "acc_person_000001",
        "channel": "atm",
        "amount_paise": 100000,
        "balance_after_paise": 400000,
        "event_time": event_time.isoformat(),
        "status": "success",
        "failure_domain": None,
    }

    # 1. Post new withdrawal via in-process client (returns 201 recorded)
    out_proc_1 = await proc_client.record_withdrawal(
        withdrawal_payload,
        idempotency_key="wth:wth_test_001",
    )
    assert out_proc_1.status == 201
    assert out_proc_1.body["status"] == "recorded"
    assert out_proc_1.replayed is False

    # 2. Replay same withdrawal via in-process client with same idempotency key (replays 201 recorded)
    out_proc_2 = await proc_client.record_withdrawal(
        withdrawal_payload,
        idempotency_key="wth:wth_test_001",
    )
    assert out_proc_2.status == 201
    assert out_proc_2.body == out_proc_1.body
    assert out_proc_2.replayed is True

    # 3. Post duplicate withdrawal with fresh idempotency key (returns 200 duplicate)
    out_proc_3 = await proc_client.record_withdrawal(
        withdrawal_payload,
        idempotency_key="wth:wth_test_001:dup",
    )
    assert out_proc_3.status == 200
    assert out_proc_3.body["status"] == "duplicate"
    assert out_proc_3.replayed is False

    # 4. Post identical withdrawal via SignedHttpBankClient on env schema
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        signed_client = SignedHttpBankClient(http_client, bank_id, env.api_keys[bank_id], env.clock)

        out_http_1 = await signed_client.record_withdrawal(
            withdrawal_payload,
            idempotency_key="wth:wth_test_001",
        )
        assert out_http_1.status == 201
        assert out_http_1.body == out_proc_1.body
        assert out_http_1.replayed is False

        # Replay via HTTP client with same idempotency key
        out_http_2 = await signed_client.record_withdrawal(
            withdrawal_payload,
            idempotency_key="wth:wth_test_001",
        )
        assert out_http_2.status == 201
        assert out_http_2.body == out_proc_2.body
        assert out_http_2.replayed is True

        # Duplicate via HTTP client with fresh idempotency key
        out_http_3 = await signed_client.record_withdrawal(
            withdrawal_payload,
            idempotency_key="wth:wth_test_001:dup",
        )
        assert out_http_3.status == 200
        assert out_http_3.body == out_proc_3.body
        assert out_http_3.replayed is False

    # 4. Compare stored database rows between schemas
    rows_proc = await conn_proc.fetch("SELECT * FROM bank_withdrawals WHERE withdrawal_id = 'wth_test_001'")
    rows_http = await env.connection.fetch("SELECT * FROM bank_withdrawals WHERE withdrawal_id = 'wth_test_001'")
    assert len(rows_proc) == 1
    assert len(rows_http) == 1

    clean_proc = sanitize_row(dict(rows_proc[0]), EXCLUDED_WITHDRAWAL_COLUMNS)
    clean_http = sanitize_row(dict(rows_http[0]), EXCLUDED_WITHDRAWAL_COLUMNS)
    assert clean_proc == clean_http


async def test_transaction_balance_after_persistence(
    env: EndpointEnv,
    secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool],
) -> None:
    """Proves balance_after_paise is stored and matches between InProcess and SignedHttp clients."""
    import orjson

    conn_proc, pool_proc = secondary_schema
    bank_id = "bank_0"
    event_time = datetime(2026, 10, 28, 11, 0, 0, tzinfo=timezone.utc)

    tx_payload = {
        "payment_id": "pmt_bal_001",
        "side": "debit",
        "account_id": "acc_person_000001",
        "counterparty_vpa": None,
        "counterparty_bank_id": "bank_1",
        "sender_bank_id": "bank_0",
        "receiver_bank_id": "bank_1",
        "amount_paise": 50000,
        "event_time": event_time.isoformat(),
        "status": "success",
        "failure_domain": None,
        "payee_type": "merchant",
        "channel": "upi_app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": 750000,
    }
    body_bytes = orjson.dumps(tx_payload)
    idempotency_key = "tx:pmt_bal_001:debit"

    proc_client = InProcessBankClient(pool_proc, bank_id, env.clock)
    out_proc = await proc_client.post("/v1/transactions", body_bytes, idempotency_key)
    assert out_proc.status == 201

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        signed_client = SignedHttpBankClient(http_client, bank_id, env.api_keys[bank_id], env.clock)
        out_http = await signed_client.post("/v1/transactions", body_bytes, idempotency_key)
        assert out_http.status == 201

    rows_proc = await conn_proc.fetch("SELECT * FROM transactions WHERE payment_id = 'pmt_bal_001'")
    rows_http = await env.connection.fetch("SELECT * FROM transactions WHERE payment_id = 'pmt_bal_001'")
    assert len(rows_proc) == 1
    assert len(rows_http) == 1
    assert rows_proc[0]["balance_after_paise"] == 750000
    assert rows_http[0]["balance_after_paise"] == 750000

    clean_proc = sanitize_row(dict(rows_proc[0]), EXCLUDED_TRANSACTION_COLUMNS)
    clean_http = sanitize_row(dict(rows_http[0]), EXCLUDED_TRANSACTION_COLUMNS)
    assert clean_proc == clean_http

