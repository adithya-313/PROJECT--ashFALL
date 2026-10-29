# File: tests/integration/test_bank_connectors_end_to_end.py
# Purpose: Integration tests proving MockBankConnectors end-to-end and InProcess vs SignedHTTP parity.
#
# Proves that running an interactive world through mock bank connectors populates decision requests,
# transaction sides, and account events with 100% fidelity, guarantees byte-identical world output
# in shadow mode, matches HTTP wire transport row-for-row, and preserves R7 failure domain visibility.

from collections.abc import AsyncGenerator
from typing import Any, cast

import asyncpg
import httpx
import orjson
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
from world.bank_connectors import (
    BankTransport,
    run_world_through_banks,
)
from world.bank_feed import project_run
from world.interactive_events import interactive_event_lines
from world.interactive_records import allow_all
from world.interactive_runner import (
    FastRunSettings,
    RunnerSettings,
    run_interactive_world,
)

pytestmark = pytest.mark.anyio

# FastRunSettings choice: FastRunSettings(101, 20, 6, 51, "training")
# WHY: A world with n_days <= 50 is rejected by simulator validation due to festival peak at day 50.
# This 51-day world completes quickly while producing 795 payment events with failures across all 5 failure domains.
WORLD_SETTINGS = FastRunSettings(101, 20, 6, 51, "training")
WORLD_RUNNER_SETTINGS = RunnerSettings(fraud_enabled=True)

# Excluded volatile non-deterministic columns for cross-schema parity comparisons.
EXCLUDED_DECISION_COLUMNS: frozenset[str] = frozenset({"created_at"})
EXCLUDED_TRANSACTION_COLUMNS: frozenset[str] = frozenset({"id", "transaction_id", "created_at"})
EXCLUDED_ACCOUNT_EVENT_COLUMNS: frozenset[str] = frozenset({"id", "created_at"})
EXCLUDED_IDEMPOTENCY_COLUMNS: frozenset[str] = frozenset({"created_at", "expires_at"})


def sanitize_row(row_dict: dict[str, Any], excluded_columns: frozenset[str]) -> dict[str, Any]:
    """Filters out excluded volatile columns (e.g. timestamps, autoincrement IDs) for row comparison."""
    return {col: val for col, val in row_dict.items() if col not in excluded_columns}


def sanitize_idempotency_row(row_dict: dict[str, Any]) -> dict[str, Any]:
    """Filters volatile idempotency columns and strips generated transaction UUIDs from response bodies."""
    cleaned = sanitize_row(row_dict, EXCLUDED_IDEMPOTENCY_COLUMNS)
    if "response_body" in cleaned and isinstance(cleaned["response_body"], str):
        body = orjson.loads(cleaned["response_body"])
        if isinstance(body, dict):
            body.pop("transaction_id", None)
            cleaned["response_body"] = body
    return cleaned


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


async def test_a_in_process_connectors_end_to_end(env: EndpointEnv) -> None:
    """Test A: Runs world simulation through InProcessBankClient asserting all rows match computed counts."""
    pool = app.state.db_pool
    transports: dict[str, BankTransport] = cast(
        dict[str, BankTransport],
        {b: InProcessBankClient(pool, b, env.clock) for b in BANKS},
    )

    report = await run_world_through_banks(
        WORLD_SETTINGS,
        WORLD_RUNNER_SETTINGS,
        transports=transports,
    )

    feed = project_run(report.output)
    events_count = len(report.output.result.run_result.events)
    total_projected_sides = sum(len(sides) for sides in feed.payments_by_bank.values())
    total_projected_acct_events = sum(len(evts) for evts in feed.account_events_by_bank.values())

    decisions_stored = await env.connection.fetchval("SELECT count(*) FROM bank_decision_requests")
    assert decisions_stored == report.decisions_sent + report.decisions_replayed == events_count

    tx_rows = await env.connection.fetchval("SELECT count(*) FROM transactions")
    assert tx_rows == total_projected_sides

    acct_rows = await env.connection.fetchval("SELECT count(*) FROM bank_account_events")
    assert acct_rows == total_projected_acct_events

    assert len(report.decision_failures) == 0

    for b in BANKS:
        stored_b_tx = await env.connection.fetchval(
            "SELECT count(*) FROM transactions WHERE bank_id = $1", b
        )
        assert stored_b_tx == len(feed.payments_by_bank[b])

        stored_b_acct = await env.connection.fetchval(
            "SELECT count(*) FROM bank_account_events WHERE bank_id = $1", b
        )
        assert stored_b_acct == len(feed.account_events_by_bank[b])

        stored_b_dec = await env.connection.fetchval(
            "SELECT count(*) FROM bank_decision_requests WHERE bank_id = $1", b
        )
        assert stored_b_dec > 0


async def test_b_shadow_mode_leaves_world_unchanged(env: EndpointEnv) -> None:
    """Test B: Asserts that shadow mode execution produces event lines identical to a plain allow_all run."""
    plain_output = run_interactive_world(WORLD_SETTINGS, WORLD_RUNNER_SETTINGS, hook=allow_all)
    plain_lines = interactive_event_lines(plain_output)

    pool = app.state.db_pool
    transports: dict[str, BankTransport] = cast(
        dict[str, BankTransport],
        {b: InProcessBankClient(pool, b, env.clock) for b in BANKS},
    )

    report = await run_world_through_banks(
        WORLD_SETTINGS,
        WORLD_RUNNER_SETTINGS,
        transports=transports,
    )
    connector_lines = interactive_event_lines(report.output)

    assert connector_lines == plain_lines


async def test_c_in_process_vs_http_parity(
    env: EndpointEnv,
    secondary_schema: tuple[asyncpg.Connection, asyncpg.Pool],
) -> None:
    """Test C: Compares stored database rows between InProcessBankClient and SignedHttpBankClient runs."""
    conn_proc, pool_proc = secondary_schema

    # 1. Run in-process on secondary schema
    proc_transports: dict[str, BankTransport] = cast(
        dict[str, BankTransport],
        {b: InProcessBankClient(pool_proc, b, env.clock) for b in BANKS},
    )
    await run_world_through_banks(
        WORLD_SETTINGS,
        WORLD_RUNNER_SETTINGS,
        transports=proc_transports,
    )

    # 2. Run over HTTP with SignedHttpBankClient on env schema
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        http_transports: dict[str, BankTransport] = cast(
            dict[str, BankTransport],
            {b: SignedHttpBankClient(http_client, b, env.api_keys[b], env.clock) for b in BANKS},
        )
        await run_world_through_banks(
            WORLD_SETTINGS,
            WORLD_RUNNER_SETTINGS,
            transports=http_transports,
        )

    # 3. Assert transactions parity
    tx_proc_raw = await conn_proc.fetch("SELECT * FROM transactions ORDER BY payment_id, side")
    tx_http_raw = await env.connection.fetch("SELECT * FROM transactions ORDER BY payment_id, side")
    tx_proc = [sanitize_row(dict(r), EXCLUDED_TRANSACTION_COLUMNS) for r in tx_proc_raw]
    tx_http = [sanitize_row(dict(r), EXCLUDED_TRANSACTION_COLUMNS) for r in tx_http_raw]
    assert tx_proc == tx_http

    # 4. Assert bank_decision_requests parity
    dec_proc_raw = await conn_proc.fetch("SELECT * FROM bank_decision_requests ORDER BY payment_id")
    dec_http_raw = await env.connection.fetch("SELECT * FROM bank_decision_requests ORDER BY payment_id")
    dec_proc = [sanitize_row(dict(r), EXCLUDED_DECISION_COLUMNS) for r in dec_proc_raw]
    dec_http = [sanitize_row(dict(r), EXCLUDED_DECISION_COLUMNS) for r in dec_http_raw]
    assert dec_proc == dec_http

    # 5. Assert bank_account_events parity
    acct_proc_raw = await conn_proc.fetch("SELECT * FROM bank_account_events ORDER BY event_id")
    acct_http_raw = await env.connection.fetch("SELECT * FROM bank_account_events ORDER BY event_id")
    acct_proc = [sanitize_row(dict(r), EXCLUDED_ACCOUNT_EVENT_COLUMNS) for r in acct_proc_raw]
    acct_http = [sanitize_row(dict(r), EXCLUDED_ACCOUNT_EVENT_COLUMNS) for r in acct_http_raw]
    assert acct_proc == acct_http

    # 6. Assert bank_idempotency_keys parity
    idem_proc_raw = await conn_proc.fetch("SELECT * FROM bank_idempotency_keys ORDER BY idempotency_key")
    idem_http_raw = await env.connection.fetch("SELECT * FROM bank_idempotency_keys ORDER BY idempotency_key")
    idem_proc = [sanitize_idempotency_row(dict(r)) for r in idem_proc_raw]
    idem_http = [sanitize_idempotency_row(dict(r)) for r in idem_http_raw]
    assert idem_proc == idem_http


async def test_d_failed_payment_r7_visibility(env: EndpointEnv) -> None:
    """Test D: Proves that failed payments preserve R7 visibility rules across sender and receiver banks."""
    pool = app.state.db_pool
    transports: dict[str, BankTransport] = cast(
        dict[str, BankTransport],
        {b: InProcessBankClient(pool, b, env.clock) for b in BANKS},
    )

    report = await run_world_through_banks(
        WORLD_SETTINGS,
        WORLD_RUNNER_SETTINGS,
        transports=transports,
    )

    failed_events = [e for e in report.output.result.run_result.events if e.failure_domain is not None]
    unique_domains = {e.failure_domain for e in failed_events}
    assert len(unique_domains) == 5

    for domain in unique_domains:
        debit_rows = await env.connection.fetch(
            "SELECT * FROM transactions WHERE status = 'failed' AND side = 'debit' AND failure_domain = $1",
            domain,
        )
        assert len(debit_rows) > 0

    credit_rows = await env.connection.fetch(
        "SELECT * FROM transactions WHERE status = 'failed' AND side = 'credit'"
    )
    assert len(credit_rows) > 0
    for r in credit_rows:
        assert r["failure_domain"] == "receiver_bank"
