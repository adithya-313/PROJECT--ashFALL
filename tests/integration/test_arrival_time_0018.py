# File: tests/integration/test_arrival_time_0018.py
# Purpose: Level 16a Round 3a - integration tests for migration 0018
# (arrival_time on transactions, events, withdrawals).
import uuid
from datetime import datetime
from typing import Any

import asyncpg
import httpx
import orjson
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from api.v1.bank_keys import issue_api_key
from api.v1.bank_router import router as bank_router
from api.v1.bank_service import (
    InProcessBankClient,
    record_account_event,
    record_transaction,
    record_withdrawal,
)
from api.v1.models import (
    AccountEventRequest,
    TransactionSideRequest,
    WithdrawalRequest,
)
from api.v1.signing import sign_outgoing
from core.config import config
from core.event_clock import IST, ManualClock
from tests.integration.conftest import get_test_database_url

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

START_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
INGESTION_TIME = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)


class StubNoneClock:
    """A faulty clock returning None to verify GUARD 1 rejection."""

    def now_ist(self) -> Any:
        return None


class StubNaiveClock:
    """A faulty clock returning naive datetime to verify GUARD 1 rejection."""

    def now_ist(self) -> Any:
        return datetime(2026, 9, 5, 14, 0, 0)


async def test_arrival_time_columns_exist_and_nullable(isolated_schema: Any) -> None:
    """Verifies that arrival_time column exists, is nullable, and is TIMESTAMPTZ on all three tables."""
    conn, schema_name = isolated_schema

    for table in ("transactions", "bank_account_events", "bank_withdrawals"):
        col = await conn.fetchrow(
            """
            SELECT column_name, data_type, is_nullable, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2 AND column_name = 'arrival_time'
            """,
            schema_name,
            table,
        )
        assert col is not None, f"arrival_time column missing from {table}"
        assert col["is_nullable"] == "YES"
        assert col["data_type"] == "timestamp with time zone"


async def test_transactions_arrival_time_is_last_column(isolated_schema: Any) -> None:
    """Verifies that arrival_time column was appended as the last column of transactions table."""
    conn, schema_name = isolated_schema

    cols = await conn.fetch(
        """
        SELECT column_name, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = 'transactions'
        ORDER BY ordinal_position
        """,
        schema_name,
    )
    col_names = [r["column_name"] for r in cols]
    assert len(col_names) == 32
    assert col_names[-1] == "arrival_time"


async def test_writers_clock_stamping_and_guard1(isolated_schema: Any) -> None:
    """
    Verifies that for transactions, account events, and withdrawals:
    1. Writer with ManualClock stores exactly 2026-09-05 14:00 IST in arrival_time.
    2. Writer without clock (None) stores NULL in arrival_time.
    3. GUARD 1: writer with stub clock returning None raises ValueError and inserts no row.
    """
    conn, schema_name = isolated_schema
    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(),
        min_size=1,
        max_size=3,
        server_settings={"search_path": schema_name},
    )
    assert pool is not None
    try:
        run_id = uuid.uuid4().hex[:16]
        bank_id = "bank_0"

        # -------------------------------------------------------------
        # 1. Transactions / Payments
        # -------------------------------------------------------------
        tx_clock = TransactionSideRequest(
            payment_id=f"{run_id}-pmt-clock",
            side="debit",
            account_id="acc_debit_01",
            counterparty_bank_id="bank_1",
            sender_bank_id="bank_0",
            receiver_bank_id="bank_1",
            amount_paise=150000,
            event_time=START_TIME,
            status="success",
        )
        await record_transaction(conn, bank_id, tx_clock, ingestion_clock=ManualClock(INGESTION_TIME))
        row_tx_1 = await conn.fetchrow(
            "SELECT arrival_time FROM transactions WHERE payment_id = $1", tx_clock.payment_id
        )
        assert row_tx_1 is not None
        assert row_tx_1["arrival_time"] == INGESTION_TIME

        tx_noclock = TransactionSideRequest(
            payment_id=f"{run_id}-pmt-noclock",
            side="debit",
            account_id="acc_debit_02",
            counterparty_bank_id="bank_1",
            sender_bank_id="bank_0",
            receiver_bank_id="bank_1",
            amount_paise=250000,
            event_time=START_TIME,
            status="success",
        )
        await record_transaction(conn, bank_id, tx_noclock, ingestion_clock=None)
        row_tx_2 = await conn.fetchrow(
            "SELECT arrival_time FROM transactions WHERE payment_id = $1", tx_noclock.payment_id
        )
        assert row_tx_2 is not None
        assert row_tx_2["arrival_time"] is None

        # GUARD 1: stub clock returning None raises ValueError and inserts NO row
        tx_stub = TransactionSideRequest(
            payment_id=f"{run_id}-pmt-stub",
            side="debit",
            account_id="acc_debit_03",
            counterparty_bank_id="bank_1",
            sender_bank_id="bank_0",
            receiver_bank_id="bank_1",
            amount_paise=350000,
            event_time=START_TIME,
            status="success",
        )
        with pytest.raises(ValueError, match="ingestion_clock must provide timezone-aware datetime"):
            await record_transaction(conn, bank_id, tx_stub, ingestion_clock=StubNoneClock())  # type: ignore[arg-type]

        # Verify no row was inserted
        row_tx_stub = await conn.fetchrow(
            "SELECT 1 FROM transactions WHERE payment_id = $1", tx_stub.payment_id
        )
        assert row_tx_stub is None

        # -------------------------------------------------------------
        # 2. Account Events
        # -------------------------------------------------------------
        ev_clock = AccountEventRequest(
            event_id=f"{run_id}-ev-clock",
            account_id="acc_user_01",
            kind="device_binding_change",
            event_time=START_TIME,
            details={"device_fingerprint": "dev_001"},
        )
        await record_account_event(conn, bank_id, ev_clock, ingestion_clock=ManualClock(INGESTION_TIME))
        row_ev_1 = await conn.fetchrow(
            "SELECT arrival_time FROM bank_account_events WHERE event_id = $1", ev_clock.event_id
        )
        assert row_ev_1 is not None
        assert row_ev_1["arrival_time"] == INGESTION_TIME

        ev_noclock = AccountEventRequest(
            event_id=f"{run_id}-ev-noclock",
            account_id="acc_user_02",
            kind="device_binding_change",
            event_time=START_TIME,
            details={"device_fingerprint": "dev_002"},
        )
        await record_account_event(conn, bank_id, ev_noclock, ingestion_clock=None)
        row_ev_2 = await conn.fetchrow(
            "SELECT arrival_time FROM bank_account_events WHERE event_id = $1", ev_noclock.event_id
        )
        assert row_ev_2 is not None
        assert row_ev_2["arrival_time"] is None

        # GUARD 1: stub clock returning None raises ValueError and inserts NO row
        ev_stub = AccountEventRequest(
            event_id=f"{run_id}-ev-stub",
            account_id="acc_user_03",
            kind="device_binding_change",
            event_time=START_TIME,
            details={"device_fingerprint": "dev_003"},
        )
        with pytest.raises(ValueError, match="ingestion_clock must provide timezone-aware datetime"):
            await record_account_event(conn, bank_id, ev_stub, ingestion_clock=StubNoneClock())  # type: ignore[arg-type]

        row_ev_stub = await conn.fetchrow(
            "SELECT 1 FROM bank_account_events WHERE event_id = $1", ev_stub.event_id
        )
        assert row_ev_stub is None

        # -------------------------------------------------------------
        # 3. Withdrawals
        # -------------------------------------------------------------
        wth_clock = WithdrawalRequest(
            withdrawal_id=f"{run_id}-wth-clock",
            bank_id=bank_id,
            account_id="acc_user_01",
            channel="atm",
            amount_paise=100000,
            status="success",
            event_time=START_TIME,
            balance_after_paise=900000,
        )
        await record_withdrawal(conn, bank_id, wth_clock, ingestion_clock=ManualClock(INGESTION_TIME))
        row_wth_1 = await conn.fetchrow(
            "SELECT arrival_time FROM bank_withdrawals WHERE withdrawal_id = $1", wth_clock.withdrawal_id
        )
        assert row_wth_1 is not None
        assert row_wth_1["arrival_time"] == INGESTION_TIME

        wth_noclock = WithdrawalRequest(
            withdrawal_id=f"{run_id}-wth-noclock",
            bank_id=bank_id,
            account_id="acc_user_02",
            channel="atm",
            amount_paise=200000,
            status="success",
            event_time=START_TIME,
            balance_after_paise=800000,
        )
        await record_withdrawal(conn, bank_id, wth_noclock, ingestion_clock=None)
        row_wth_2 = await conn.fetchrow(
            "SELECT arrival_time FROM bank_withdrawals WHERE withdrawal_id = $1", wth_noclock.withdrawal_id
        )
        assert row_wth_2 is not None
        assert row_wth_2["arrival_time"] is None

        # GUARD 1: stub clock returning None raises ValueError and inserts NO row
        wth_stub = WithdrawalRequest(
            withdrawal_id=f"{run_id}-wth-stub",
            bank_id=bank_id,
            account_id="acc_user_03",
            channel="atm",
            amount_paise=300000,
            status="success",
            event_time=START_TIME,
            balance_after_paise=700000,
        )
        with pytest.raises(ValueError, match="ingestion_clock must provide timezone-aware datetime"):
            await record_withdrawal(conn, bank_id, wth_stub, ingestion_clock=StubNoneClock())  # type: ignore[arg-type]

        row_wth_stub = await conn.fetchrow(
            "SELECT 1 FROM bank_withdrawals WHERE withdrawal_id = $1", wth_stub.withdrawal_id
        )
        assert row_wth_stub is None

    finally:
        await pool.close()


async def test_in_process_client_and_router_stamping(
    isolated_schema: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that InProcessBankClient and bank_router pass ingestion_clock through to all three writers."""
    conn, schema_name = isolated_schema
    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(),
        min_size=1,
        max_size=3,
        server_settings={"search_path": schema_name},
    )
    assert pool is not None

    bank_id = "bank_0"
    secret_text = "test_secret_for_bank_0_32_chars_min!"
    monkeypatch.setattr(config, f"bank_signing_secret_{bank_id}", SecretStr(secret_text))
    _, api_key = await issue_api_key(conn, bank_id)

    test_app = FastAPI()
    test_app.include_router(bank_router)
    test_app.state.db_pool = pool
    test_app.state.transport_clock = ManualClock(START_TIME)

    try:
        run_id = uuid.uuid4().hex[:16]

        # 1. InProcessBankClient with clock
        client = InProcessBankClient(
            pool=pool,
            bank_id=bank_id,
            clock=ManualClock(START_TIME),
            ingestion_clock=ManualClock(INGESTION_TIME),
        )

        res_tx = await client.post(
            "/v1/transactions",
            orjson.dumps({
                "payment_id": f"{run_id}-ipc-tx",
                "side": "debit",
                "account_id": "acc_ipc_01",
                "counterparty_bank_id": "bank_1",
                "sender_bank_id": "bank_0",
                "receiver_bank_id": "bank_1",
                "amount_paise": 50000,
                "event_time": START_TIME.isoformat(),
                "status": "success",
            }),
            idempotency_key="idemp_ipc_tx",
        )
        assert res_tx.status == 201
        row_tx = await conn.fetchrow(
            "SELECT arrival_time FROM transactions WHERE payment_id = $1", f"{run_id}-ipc-tx"
        )
        assert row_tx is not None
        assert row_tx["arrival_time"] == INGESTION_TIME

        res_ev = await client.post(
            "/v1/account-events",
            orjson.dumps({
                "event_id": f"{run_id}-ipc-ev",
                "account_id": "acc_ipc_01",
                "kind": "account_opened",
                "event_time": START_TIME.isoformat(),
                "details": {"account_type": "savings"},
            }),
            idempotency_key="idemp_ipc_ev",
        )
        assert res_ev.status == 201
        row_ev = await conn.fetchrow(
            "SELECT arrival_time FROM bank_account_events WHERE event_id = $1", f"{run_id}-ipc-ev"
        )
        assert row_ev is not None
        assert row_ev["arrival_time"] == INGESTION_TIME

        res_wth = await client.record_withdrawal(
            {
                "withdrawal_id": f"{run_id}-ipc-wth",
                "bank_id": bank_id,
                "account_id": "acc_ipc_01",
                "channel": "atm",
                "amount_paise": 40000,
                "status": "success",
                "event_time": START_TIME.isoformat(),
                "balance_after_paise": 600000,
            },
            idempotency_key="idemp_ipc_wth",
        )
        assert res_wth.status == 201
        row_wth = await conn.fetchrow(
            "SELECT arrival_time FROM bank_withdrawals WHERE withdrawal_id = $1", f"{run_id}-ipc-wth"
        )
        assert row_wth is not None
        assert row_wth["arrival_time"] == INGESTION_TIME

        # 2. HTTP router with app.state.ingestion_clock set
        test_app.state.ingestion_clock = ManualClock(INGESTION_TIME)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app),
            base_url="http://test",
        ) as http_client:
            body_tx = orjson.dumps({
                "payment_id": f"{run_id}-http-tx",
                "side": "debit",
                "account_id": "acc_http_01",
                "counterparty_bank_id": "bank_1",
                "sender_bank_id": "bank_0",
                "receiver_bank_id": "bank_1",
                "amount_paise": 70000,
                "event_time": START_TIME.isoformat(),
                "status": "success",
            })
            hdrs_tx = sign_outgoing(
                bank_id=bank_id,
                method="POST",
                path="/v1/transactions",
                body_bytes=body_tx,
                clock=ManualClock(START_TIME),
                nonce=uuid.uuid4().hex,
            )
            hdrs_tx["X-Ashfall-Key"] = api_key
            hdrs_tx["Idempotency-Key"] = "idemp_http_tx"
            resp1 = await http_client.post("/v1/transactions", content=body_tx, headers=hdrs_tx)
            assert resp1.status_code == 201

            row_http_tx = await conn.fetchrow(
                "SELECT arrival_time FROM transactions WHERE payment_id = $1", f"{run_id}-http-tx"
            )
            assert row_http_tx is not None
            assert row_http_tx["arrival_time"] == INGESTION_TIME

    finally:
        await pool.close()
