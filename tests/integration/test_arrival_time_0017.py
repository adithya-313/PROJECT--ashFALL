# File: tests/integration/test_arrival_time_0017.py
# Purpose: Level 16a Round 2 - integration tests for migration 0017 (arrival_time on complaints and feedback).
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
from api.v1.bank_service import InProcessBankClient
from api.v1.signing import sign_outgoing
from core.config import config
from core.event_clock import IST, ManualClock
from tests.integration.conftest import get_test_database_url

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

START_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
INGESTION_TIME = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)


async def test_arrival_time_columns_exist_and_nullable(isolated_schema: Any) -> None:
    """Verifies that arrival_time column exists and is nullable on both bank_complaints and bank_feedback."""
    conn, schema_name = isolated_schema

    for table in ("bank_complaints", "bank_feedback"):
        col = await conn.fetchrow(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2 AND column_name = 'arrival_time'
            """,
            schema_name,
            table,
        )
        assert col is not None, f"arrival_time column missing from {table}"
        assert col["is_nullable"] == "YES"
        assert col["data_type"] == "timestamp with time zone"


async def test_complaint_via_in_process_client_clock_stamping(isolated_schema: Any) -> None:
    """
    Verifies that InProcessBankClient:
    - with an injected Clock stores arrival_time from that clock;
    - without an injected Clock (None) stores NULL.
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

        # 1. With injected Clock
        ingestion_clock = ManualClock(INGESTION_TIME)
        client_with_clock = InProcessBankClient(
            pool=pool,
            bank_id="bank_0",
            clock=ManualClock(START_TIME),
            ingestion_clock=ingestion_clock,
        )

        cid_with_clock = f"{run_id}-cmp-clock"
        payload_1 = {
            "complaint_id": cid_with_clock,
            "complainant_account": "acc_vic_01",
            "reported_account": "acc_rep_01",
            "reported_bank_id": "bank_0",
            "payment_ids": ["pay_01"],
            "claimed_type": "phishing",
            "received_at": "2026-09-05T13:30:00+05:30",
        }
        res1 = await client_with_clock.post(
            "/v1/complaints",
            orjson.dumps(payload_1),
            idempotency_key="idemp_cmp_1",
        )
        assert res1.status == 201

        row1 = await conn.fetchrow(
            "SELECT arrival_time FROM bank_complaints WHERE complaint_id = $1",
            cid_with_clock,
        )
        assert row1 is not None
        assert row1["arrival_time"] == INGESTION_TIME

        # 2. Without injected Clock (None)
        client_no_clock = InProcessBankClient(
            pool=pool,
            bank_id="bank_0",
            clock=ManualClock(START_TIME),
        )

        cid_no_clock = f"{run_id}-cmp-noclock"
        payload_2 = {
            "complaint_id": cid_no_clock,
            "complainant_account": "acc_vic_02",
            "reported_account": "acc_rep_02",
            "reported_bank_id": "bank_0",
            "payment_ids": ["pay_02"],
            "claimed_type": "scam",
            "received_at": "2026-09-05T13:30:00+05:30",
        }
        res2 = await client_no_clock.post(
            "/v1/complaints",
            orjson.dumps(payload_2),
            idempotency_key="idemp_cmp_2",
        )
        assert res2.status == 201

        row2 = await conn.fetchrow(
            "SELECT arrival_time FROM bank_complaints WHERE complaint_id = $1",
            cid_no_clock,
        )
        assert row2 is not None
        assert row2["arrival_time"] is None

    finally:
        await pool.close()


async def test_feedback_via_router_clock_stamping(
    isolated_schema: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifies that POST /v1/feedback over the FastAPI app:
    - with app.state.ingestion_clock set stores arrival_time from that clock;
    - without app.state.ingestion_clock set stores NULL.
    """
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

        # 1. With app.state.ingestion_clock set
        test_app.state.ingestion_clock = ManualClock(INGESTION_TIME)

        fb_id_1 = f"{run_id}-fb-clock"
        payload_1 = {
            "feedback_id": fb_id_1,
            "payment_id": f"{run_id}-pmt-001",
            "outcome": "confirmed_fraud",
            "note": "fraud confirmed by customer",
            "event_time": "2026-09-05T13:30:00+05:30",
        }
        body_1 = orjson.dumps(payload_1)
        headers_1 = sign_outgoing(
            bank_id=bank_id,
            method="POST",
            path="/v1/feedback",
            body_bytes=body_1,
            clock=ManualClock(START_TIME),
            nonce=uuid.uuid4().hex,
        )
        headers_1["X-Ashfall-Key"] = api_key
        headers_1["Idempotency-Key"] = "idemp_fb_1"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app),
            base_url="http://test",
        ) as client:
            resp1 = await client.post("/v1/feedback", content=body_1, headers=headers_1)
            assert resp1.status_code == 201

        row1 = await conn.fetchrow(
            "SELECT arrival_time FROM bank_feedback WHERE feedback_id = $1",
            fb_id_1,
        )
        assert row1 is not None
        assert row1["arrival_time"] == INGESTION_TIME

        # 2. Without app.state.ingestion_clock
        if hasattr(test_app.state, "ingestion_clock"):
            delattr(test_app.state, "ingestion_clock")

        fb_id_2 = f"{run_id}-fb-noclock"
        payload_2 = {
            "feedback_id": fb_id_2,
            "payment_id": f"{run_id}-pmt-002",
            "outcome": "not_fraud",
            "note": "false positive",
            "event_time": "2026-09-05T13:45:00+05:30",
        }
        body_2 = orjson.dumps(payload_2)
        headers_2 = sign_outgoing(
            bank_id=bank_id,
            method="POST",
            path="/v1/feedback",
            body_bytes=body_2,
            clock=ManualClock(START_TIME),
            nonce=uuid.uuid4().hex,
        )
        headers_2["X-Ashfall-Key"] = api_key
        headers_2["Idempotency-Key"] = "idemp_fb_2"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app),
            base_url="http://test",
        ) as client:
            resp2 = await client.post("/v1/feedback", content=body_2, headers=headers_2)
            assert resp2.status_code == 201

        row2 = await conn.fetchrow(
            "SELECT arrival_time FROM bank_feedback WHERE feedback_id = $1",
            fb_id_2,
        )
        assert row2 is not None
        assert row2["arrival_time"] is None

    finally:
        await pool.close()
