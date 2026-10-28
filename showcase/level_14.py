# File: showcase/level_14.py
# Purpose: Level 14 showcase: Commercial bank UPI APIs, database isolation, and RLS (Parts 14a, 14b).
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import orjson
from pydantic import SecretStr

import showcase.level_07 as level_07
from api.main import app
from api.services.outbox_relay import choose_topic
from api.v1.bank_keys import issue_api_key, resolve_bank_for_key
from api.v1.bank_scope import bank_scoped_transaction
from api.v1.signing import load_bank_api_policy, sign_outgoing
from api.v1.upi_outbox import (
    EVENT_TYPE_UPI_PAYMENT_RECORDED,
    UPI_SCHEMA_VERSION,
    UpiPaymentSide,
    record_upi_side,
)
from core.config import config
from core.event_clock import IST, ManualClock
from showcase.framework import CheckDefinition, CheckOutcome, NotRun
from showcase.isolation import Workspace, get_showcase_database_url
from showcase.preconditions import check_database
from tests.contract.test_bank_api_fuzz import (
    generate_fuzz_cases,
    run_signed_fuzz_pass,
    run_unsigned_fuzz_pass,
)

LEVEL_NUMBER = 14
LEVEL_TITLE = "Commercial bank UPI APIs and isolation (built: parts 14a, 14b)"
NEEDS_DATABASE = True

WHAT_YOU_SAW = (
    "Level 14 introduces commercial bank UPI integration APIs and PostgreSQL Row Level Security (RLS) tenant "
    "isolation. Part 14a establishes schema migration 0006 with UPI columns and hashed API keys, enforces "
    "bank-scoped transaction isolation using a non-superuser role (ashfall_bank_api), proves that cross-bank reads "
    "and writes are refused, verifies connection pool safety without state bleed across bank transactions, and "
    "records cross-bank payments into transactional outbox with UUIDv5 identities and routing to upi.payments. "
    "Part 14b implements authenticated, signed, and idempotent bank REST APIs, raw percent-encoded path signing, "
    "replay protection with transport clock isolation, and comprehensive schemathesis fuzzing rejecting malformed "
    "requests with 4xx while preventing unhandled 500 server crashes."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = [
    REPO_ROOT / "config" / "bank_api.yaml",
    REPO_ROOT / "api" / "v1" / "bank_scope.py",
    REPO_ROOT / "api" / "v1" / "bank_keys.py",
    REPO_ROOT / "api" / "v1" / "upi_outbox.py",
    REPO_ROOT / "api" / "v1" / "request_auth.py",
    REPO_ROOT / "api" / "v1" / "bank_router.py",
    REPO_ROOT / "api" / "v1" / "signing.py",
    REPO_ROOT / "migrations" / "0006_add_upi_columns_and_bank_isolation.sql",
    REPO_ROOT / "tests" / "contract" / "test_bank_api_fuzz.py",
]


async def find_missing_preconditions() -> list[str]:
    """Check that PostgreSQL is reachable and all mandatory Level 14 implementation files exist."""
    problems: list[str] = []
    db_problem = await check_database(get_showcase_database_url(), "docker compose up -d postgres")
    if db_problem is not None:
        problems.append(db_problem)

    for f_path in REQUIRED_FILES:
        if not f_path.is_file():
            problems.append(f"{f_path.name} is missing - it is part of Level 14 files")

    return problems


async def check_1_bank_isolation_negative(workspace: Workspace) -> CheckOutcome:
    """
    Check 1 (negative): bank_1's key reads bank_2's account -> 0 rows, AND the cross-bank insert
    is refused with SQLSTATE 42501 (insufficient_privilege).
    Evidence: the row counts under both banks.
    """
    conn = workspace.connection
    now = datetime.now(timezone.utc)

    # 1. Issue API keys for bank_1 and bank_2 on the workspace connection
    _, key_bank_1 = await issue_api_key(conn, "bank_1")
    _, key_bank_2 = await issue_api_key(conn, "bank_2")

    # 2. Insert one world row for bank_2 account X as superuser
    await conn.execute(
        """
        INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                 amount_paise, event_time, status)
        VALUES ('world', 'acc_bank2_x', 500.00, 'pmt_showcase_1', 'bank_2', 'debit', 50000, $1, 'success')
        """,
        now,
    )

    # 3. Resolve key for bank_1 and verify isolation: query yields 0 rows
    bank_1_id = await resolve_bank_for_key(conn, key_bank_1)
    assert bank_1_id == "bank_1"

    async with bank_scoped_transaction(conn, "bank_1") as scoped_conn:
        rows_under_bank_1 = await scoped_conn.fetch(
            "SELECT id, account_id, amount FROM transactions WHERE account_id = 'acc_bank2_x'"
        )
        count_bank_1 = len(rows_under_bank_1)
        assert count_bank_1 == 0

        # Attempt to insert a row for bank_2 while scoped under bank_1
        observed_sqlstate = ""
        try:
            await scoped_conn.execute(
                """
                INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                         amount_paise, event_time, status)
                VALUES ('world', 'acc_cross_insert', 100.00, 'pmt_cross_1', 'bank_2', 'debit', 10000, $1, 'success')
                """,
                now,
            )
        except asyncpg.exceptions.InsufficientPrivilegeError as exc:
            observed_sqlstate = exc.sqlstate

        assert observed_sqlstate == "42501", f"Expected SQLSTATE 42501, observed {observed_sqlstate!r}"

    # 4. Under bank_2 scope, the row for account X is visible (proves non-empty table)
    bank_2_id = await resolve_bank_for_key(conn, key_bank_2)
    assert bank_2_id == "bank_2"

    async with bank_scoped_transaction(conn, "bank_2") as scoped_conn_2:
        rows_under_bank_2 = await scoped_conn_2.fetch(
            "SELECT id, account_id, amount FROM transactions WHERE account_id = 'acc_bank2_x'"
        )
        count_bank_2 = len(rows_under_bank_2)
        assert count_bank_2 == 1

    return CheckOutcome(
        key_number=f"bank_1 rows={count_bank_1}, bank_2 rows={count_bank_2}, sqlstate={observed_sqlstate}",
        evidence={
            "rows_under_bank_1": count_bank_1,
            "rows_under_bank_2": count_bank_2,
            "cross_bank_insert_sqlstate": observed_sqlstate,
        },
    )


async def check_2_pooled_interleaving_edge(workspace: Workspace) -> CheckOutcome:
    """
    Check 2 (edge): two banks interleaved on a 1-connection pool, >= 3 rounds, same backend pid,
    each sees only its own rows, and no leftover role or bank after release.
    """
    now = datetime.now(timezone.utc)
    pool = await asyncpg.create_pool(
        dsn=workspace.dsn,
        min_size=1,
        max_size=1,
        server_settings={"search_path": workspace.schema_name},
    )
    assert pool is not None

    rounds_completed = 0
    fixed_backend_pid = None

    try:
        # Seed rows for bank_0 and bank_1 as superuser
        async with pool.acquire() as conn:
            orig_user = await conn.fetchval("SELECT current_user")
            fixed_backend_pid = await conn.fetchval("SELECT pg_backend_pid()")

            await conn.execute(
                """
                INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                         amount_paise, event_time, status)
                VALUES ('world', 'acc_sc_pool_0', 100.00, 'pmt_sc_pool_0', 'bank_0', 'debit', 10000, $1, 'success'),
                       ('world', 'acc_sc_pool_1', 200.00, 'pmt_sc_pool_1', 'bank_1', 'debit', 20000, $1, 'success')
                """,
                now,
            )

        for round_idx in range(3):
            # Interleave bank_0
            async with pool.acquire() as conn:
                pid = await conn.fetchval("SELECT pg_backend_pid()")
                assert pid == fixed_backend_pid, f"Backend PID changed on round {round_idx}!"

                async with bank_scoped_transaction(conn, "bank_0"):
                    rows_0 = await conn.fetch("SELECT account_id FROM transactions")
                    assert len(rows_0) == 1
                    assert rows_0[0]["account_id"] == "acc_sc_pool_0"

            # Verify clean reset after release
            async with pool.acquire() as conn:
                user_after_0 = await conn.fetchval("SELECT current_user")
                setting_after_0 = await conn.fetchval("SELECT current_setting('app.bank_id', true)")
                assert user_after_0 == orig_user
                assert setting_after_0 in (None, "")

                # Interleave bank_1
                async with bank_scoped_transaction(conn, "bank_1"):
                    rows_1 = await conn.fetch("SELECT account_id FROM transactions")
                    assert len(rows_1) == 1
                    assert rows_1[0]["account_id"] == "acc_sc_pool_1"

            # Verify clean reset after release
            async with pool.acquire() as conn:
                user_after_1 = await conn.fetchval("SELECT current_user")
                setting_after_1 = await conn.fetchval("SELECT current_setting('app.bank_id', true)")
                assert user_after_1 == orig_user
                assert setting_after_1 in (None, "")

            rounds_completed += 1

    finally:
        await pool.close()

    return CheckOutcome(
        key_number=f"rounds={rounds_completed}, pid={fixed_backend_pid}, clean_resets=verified",
        evidence={
            "rounds_completed": rounds_completed,
            "fixed_backend_pid": fixed_backend_pid,
            "session_user_preserved": True,
            "setting_reset_verified": True,
        },
    )


async def check_3_level_7_consistency(workspace: Workspace) -> CheckOutcome:
    """
    Check 3 (consistency): run [sys.executable, "-m", "showcase", "--level", "7", "--quick"]
    as a subprocess (timeout 900 s). PASS only on exit code 0; evidence is the exit code plus
    the last 20 output lines. If the Level 7 preconditions are missing, report NotRun the same
    way other levels do; never PASS.
    """
    # Check Level 7 preconditions before running subprocess
    l7_missing = await level_07.find_missing_preconditions()
    if l7_missing:
        raise NotRun(f"Level 7 preconditions not met: {'; '.join(l7_missing)}")

    cmd = [sys.executable, "-m", "showcase", "--level", "7", "--quick"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"Level 7 quick showcase timed out after 900 seconds: {exc}") from exc

    lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
    last_20_lines = lines[-20:] if len(lines) >= 20 else lines

    if proc.returncode != 0:
        raise AssertionError(
            f"Level 7 quick showcase exited with code {proc.returncode}.\n"
            f"Last output lines:\n" + "\n".join(last_20_lines)
        )

    return CheckOutcome(
        key_number=f"exit_code={proc.returncode}",
        evidence={
            "exit_code": proc.returncode,
            "last_20_lines": last_20_lines,
        },
    )


async def check_4_cross_bank_payment_normal(workspace: Workspace) -> CheckOutcome:
    """
    Check 4 (normal): A cross-bank payment written the real way: bank_0 debit side under bank_0 scope,
    bank_2 credit side under bank_2 scope. Shows 2 transaction rows, 2 outbox rows, 2 different
    version-5 event IDs, both schema_version '2', and choose_topic routes both to 'upi.payments'.
    """
    conn = workspace.connection
    now = datetime.now(timezone.utc)
    pmt_id = "pmt_sc14_cross_001"

    debit_side = UpiPaymentSide(
        payment_id=pmt_id,
        bank_id="bank_0",
        side="debit",
        account_id="acc_sc14_debit_0",
        counterparty_bank_id="bank_2",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_2",
        amount_paise=250000,
        event_time=now,
        status="success",
        counterparty_vpa="user2@bank2",
    )

    credit_side = UpiPaymentSide(
        payment_id=pmt_id,
        bank_id="bank_2",
        side="credit",
        account_id="acc_sc14_credit_2",
        counterparty_bank_id="bank_0",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_2",
        amount_paise=250000,
        event_time=now,
        status="success",
        counterparty_vpa="user0@bank0",
    )

    async with bank_scoped_transaction(conn, "bank_0") as scoped_0:
        res_debit = await record_upi_side(scoped_0, debit_side)
        assert res_debit.was_replay is False

    async with bank_scoped_transaction(conn, "bank_2") as scoped_2:
        res_credit = await record_upi_side(scoped_2, credit_side)
        assert res_credit.was_replay is False

    tx_count = await conn.fetchval("SELECT count(*) FROM transactions WHERE payment_id = $1", pmt_id)
    outbox_rows = await conn.fetch(
        "SELECT event_id, event_type, payload FROM outbox WHERE event_id IN ($1, $2) ORDER BY id",
        res_debit.event_id,
        res_credit.event_id,
    )
    outbox_count = len(outbox_rows)
    assert tx_count == 2
    assert outbox_count == 2

    assert res_debit.event_id != res_credit.event_id
    uuid_debit = uuid.UUID(res_debit.event_id)
    uuid_credit = uuid.UUID(res_credit.event_id)
    assert uuid_debit.version == 5
    assert uuid_credit.version == 5

    topics_routed: list[str] = []
    for r in outbox_rows:
        assert r["event_type"] == EVENT_TYPE_UPI_PAYMENT_RECORDED
        payload = orjson.loads(r["payload"])
        assert payload["schema_version"] == UPI_SCHEMA_VERSION
        topic = choose_topic(r)
        topics_routed.append(topic)
        assert topic == "upi.payments"

    return CheckOutcome(
        key_number=f"tx_rows={tx_count}, outbox_rows={outbox_count}, routed={topics_routed}",
        evidence={
            "tx_rows": tx_count,
            "outbox_rows": outbox_count,
            "debit_event_id": res_debit.event_id,
            "credit_event_id": res_credit.event_id,
            "debit_uuid_version": uuid_debit.version,
            "credit_uuid_version": uuid_credit.version,
            "schema_versions": [UPI_SCHEMA_VERSION, UPI_SCHEMA_VERSION],
            "topics_routed": topics_routed,
        },
    )


async def check_5_replay_idempotency_edge(workspace: Workspace) -> CheckOutcome:
    """
    Check 5 (edge): Sending the same payment side twice stores it once. Shows 1 transaction row,
    1 outbox row, was_replay=True on the second call, and the same transaction_id.
    """
    conn = workspace.connection
    now = datetime.now(timezone.utc)
    pmt_id = "pmt_sc14_replay_002"

    side = UpiPaymentSide(
        payment_id=pmt_id,
        bank_id="bank_1",
        side="debit",
        account_id="acc_sc14_replay_1",
        counterparty_bank_id="bank_0",
        sender_bank_id="bank_1",
        receiver_bank_id="bank_0",
        amount_paise=75000,
        event_time=now,
        status="success",
    )

    async with bank_scoped_transaction(conn, "bank_1") as scoped_conn:
        first_res = await record_upi_side(scoped_conn, side)
        assert first_res.was_replay is False

        second_res = await record_upi_side(scoped_conn, side)
        assert second_res.was_replay is True
        assert second_res.transaction_id == first_res.transaction_id
        assert second_res.event_id == first_res.event_id

    tx_count = await conn.fetchval("SELECT count(*) FROM transactions WHERE payment_id = $1", pmt_id)
    outbox_count = await conn.fetchval("SELECT count(*) FROM outbox WHERE event_id = $1", first_res.event_id)
    assert tx_count == 1
    assert outbox_count == 1

    return CheckOutcome(
        key_number=(
            f"tx_id={first_res.transaction_id}, was_replay={second_res.was_replay}, "
            f"tx_rows={tx_count}, outbox_rows={outbox_count}"
        ),
        evidence={
            "first_transaction_id": first_res.transaction_id,
            "second_transaction_id": second_res.transaction_id,
            "second_was_replay": second_res.was_replay,
            "total_transactions_rows": tx_count,
            "total_outbox_rows": outbox_count,
        },
    )


@asynccontextmanager
async def _in_process_bank_app(workspace: Workspace, bank_id: str = "bank_1"):
    db_url = get_showcase_database_url()
    pool = await asyncpg.create_pool(
        dsn=db_url,
        min_size=1,
        max_size=3,
        server_settings={"search_path": workspace.schema_name},
    )
    assert pool is not None

    secret_key_attr = f"bank_signing_secret_{bank_id}"
    old_secret = getattr(config, secret_key_attr, None)
    secret_text = f"showcase_secret_{bank_id}_32_characters_min!"
    setattr(config, secret_key_attr, SecretStr(secret_text))

    # Issue API key for bank_id in workspace schema
    _, api_key = await issue_api_key(workspace.connection, bank_id)

    clock = ManualClock(datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST))
    saved_state: dict[str, Any] = {}
    for name in ("db_pool", "transport_clock"):
        if hasattr(app.state, name):
            saved_state[name] = getattr(app.state, name)
    app.state.db_pool = pool
    app.state.transport_clock = clock

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, api_key, clock, pool
    finally:
        for name in ("db_pool", "transport_clock"):
            if name in saved_state:
                setattr(app.state, name, saved_state[name])
            else:
                delattr(app.state, name)
        if old_secret is not None:
            setattr(config, secret_key_attr, old_secret)
        await pool.close()


def _make_signed_headers(
    bank_id: str,
    api_key: str,
    method: str,
    path: str,
    body: bytes,
    clock: ManualClock,
    idempotency_key: str | None = None,
) -> dict[str, str]:
    nonce = f"sc14_{uuid.uuid4().hex}"
    headers = sign_outgoing(
        bank_id=bank_id,
        method=method,
        path=path,
        body_bytes=body,
        clock=clock,
        nonce=nonce,
    )
    headers["X-Ashfall-Key"] = api_key
    if body:
        headers["Content-Type"] = "application/json"
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


async def check_6_post_transaction_and_get_risk_normal(workspace: Workspace) -> CheckOutcome:
    """
    Check 6 (normal): signed POST /v1/transactions creates a row and an outbox event;
    then a signed GET /v1/accounts/{account_id}/risk returns 200 "unscored".
    """
    async with _in_process_bank_app(workspace, "bank_1") as (client, api_key, clock, _pool):
        payment_id = f"pmt_sc14_norm_{uuid.uuid4().hex[:12]}"
        account_id = f"acc_sc14_norm_{uuid.uuid4().hex[:12]}"
        body_dict = {
            "payment_id": payment_id,
            "side": "debit",
            "account_id": account_id,
            "counterparty_bank_id": "bank_0",
            "sender_bank_id": "bank_1",
            "receiver_bank_id": "bank_0",
            "amount_paise": 50000,
            "event_time": "2026-10-28T12:00:00+05:30",
            "status": "success",
        }
        body_bytes = orjson.dumps(body_dict)
        headers = _make_signed_headers(
            "bank_1",
            api_key,
            "POST",
            "/v1/transactions",
            body_bytes,
            clock,
            idempotency_key=f"idem_sc14_{uuid.uuid4().hex}",
        )
        post_resp = await client.post("/v1/transactions", content=body_bytes, headers=headers)
        assert post_resp.status_code == 201
        post_data = post_resp.json()
        assert "transaction_id" in post_data
        assert "event_id" in post_data

        tx_count = await workspace.connection.fetchval(
            f"SELECT count(*) FROM {workspace.schema_name}.transactions WHERE payment_id = $1", payment_id
        )
        outbox_count = await workspace.connection.fetchval(
            f"SELECT count(*) FROM {workspace.schema_name}.outbox WHERE event_id = $1", post_data["event_id"]
        )
        assert tx_count == 1
        assert outbox_count == 1

        risk_path = f"/v1/accounts/{account_id}/risk"
        risk_headers = _make_signed_headers("bank_1", api_key, "GET", risk_path, b"", clock)
        risk_resp = await client.get(risk_path, headers=risk_headers)
        assert risk_resp.status_code == 200
        risk_data = risk_resp.json()
        assert risk_data["account_id"] == account_id
        assert risk_data["risk"] == "unscored"

        return CheckOutcome(
            key_number=(
                f"tx_id={post_data['transaction_id']}, risk={risk_data['risk']}, "
                f"tx_rows={tx_count}, outbox_rows={outbox_count}"
            ),
            evidence={
                "transaction_id": post_data["transaction_id"],
                "event_id": post_data["event_id"],
                "account_id": account_id,
                "risk_response": risk_data,
                "tx_rows": tx_count,
                "outbox_rows": outbox_count,
            },
        )


async def check_7_idempotency_same_key_same_body_edge(workspace: Workspace) -> CheckOutcome:
    """
    Check 7 (edge): same request twice with one idempotency key -> one DB row (count = 1).
    """
    async with _in_process_bank_app(workspace, "bank_1") as (client, api_key, clock, _pool):
        idem_key = f"idem_sc14_rep_{uuid.uuid4().hex[:12]}"
        payment_id = f"pmt_sc14_rep_{uuid.uuid4().hex[:12]}"
        body_dict = {
            "payment_id": payment_id,
            "side": "debit",
            "account_id": f"acc_sc14_rep_{uuid.uuid4().hex[:8]}",
            "counterparty_bank_id": "bank_0",
            "sender_bank_id": "bank_1",
            "receiver_bank_id": "bank_0",
            "amount_paise": 25000,
            "event_time": "2026-10-28T12:00:00+05:30",
            "status": "success",
        }
        body_bytes = orjson.dumps(body_dict)
        headers1 = _make_signed_headers(
            "bank_1", api_key, "POST", "/v1/transactions", body_bytes, clock, idempotency_key=idem_key
        )
        resp1 = await client.post("/v1/transactions", content=body_bytes, headers=headers1)
        assert resp1.status_code == 201
        data1 = resp1.json()

        headers2 = _make_signed_headers(
            "bank_1", api_key, "POST", "/v1/transactions", body_bytes, clock, idempotency_key=idem_key
        )
        resp2 = await client.post("/v1/transactions", content=body_bytes, headers=headers2)
        assert resp2.status_code == 201
        assert resp2.headers.get("Idempotent-Replayed") == "true"
        data2 = resp2.json()
        assert data2["transaction_id"] == data1["transaction_id"]

        tx_count = await workspace.connection.fetchval(
            f"SELECT count(*) FROM {workspace.schema_name}.transactions WHERE payment_id = $1", payment_id
        )
        idem_count = await workspace.connection.fetchval(
            f"SELECT count(*) FROM {workspace.schema_name}.bank_idempotency_keys WHERE idempotency_key = $1", idem_key
        )
        assert tx_count == 1
        assert idem_count == 1

        return CheckOutcome(
            key_number=f"tx_rows={tx_count}, idem_rows={idem_count}, replayed=true",
            evidence={
                "first_status": resp1.status_code,
                "second_status": resp2.status_code,
                "replayed_header": resp2.headers.get("Idempotent-Replayed"),
                "tx_rows": tx_count,
                "idem_rows": idem_count,
            },
        )


async def check_8_idempotency_different_body_conflict_edge(workspace: Workspace) -> CheckOutcome:
    """
    Check 8 (edge): same key with different body -> 409 idempotency_key_reused.
    """
    async with _in_process_bank_app(workspace, "bank_1") as (client, api_key, clock, _pool):
        idem_key = f"idem_sc14_conf_{uuid.uuid4().hex[:12]}"
        payment_id = f"pmt_sc14_conf_{uuid.uuid4().hex[:12]}"
        body1 = orjson.dumps({
            "payment_id": payment_id,
            "side": "debit",
            "account_id": f"acc_sc14_conf_{uuid.uuid4().hex[:8]}",
            "counterparty_bank_id": "bank_0",
            "sender_bank_id": "bank_1",
            "receiver_bank_id": "bank_0",
            "amount_paise": 10000,
            "event_time": "2026-10-28T12:00:00+05:30",
            "status": "success",
        })
        headers1 = _make_signed_headers(
            "bank_1", api_key, "POST", "/v1/transactions", body1, clock, idempotency_key=idem_key
        )
        resp1 = await client.post("/v1/transactions", content=body1, headers=headers1)
        assert resp1.status_code == 201

        # Reusing same idempotency key with different body
        body2 = orjson.dumps({
            "payment_id": payment_id,
            "side": "debit",
            "account_id": f"acc_sc14_conf_{uuid.uuid4().hex[:8]}",
            "counterparty_bank_id": "bank_0",
            "sender_bank_id": "bank_1",
            "receiver_bank_id": "bank_0",
            "amount_paise": 99999,
            "event_time": "2026-10-28T12:00:00+05:30",
            "status": "success",
        })
        headers2 = _make_signed_headers(
            "bank_1", api_key, "POST", "/v1/transactions", body2, clock, idempotency_key=idem_key
        )
        resp2 = await client.post("/v1/transactions", content=body2, headers=headers2)
        assert resp2.status_code == 409
        data2 = resp2.json()
        assert data2["error"]["code"] == "idempotency_key_reused"

        return CheckOutcome(
            key_number=(
                f"first_status={resp1.status_code}, second_status={resp2.status_code}, "
                f"error_code={data2['error']['code']}"
            ),
            evidence={
                "first_status": resp1.status_code,
                "second_status": resp2.status_code,
                "error": data2["error"],
            },
        )


async def check_9_fresh_signature_historical_event_edge(workspace: Workspace) -> CheckOutcome:
    """
    Check 9 (edge): replay with 2020 event_time and fresh signature is accepted; stale signature refused.
    """
    async with _in_process_bank_app(workspace, "bank_1") as (client, api_key, clock, _pool):
        old_event_time = "2020-01-01T10:00:00+05:30"
        payment_id_old = f"pmt_sc14_old_{uuid.uuid4().hex[:12]}"
        body_old = orjson.dumps({
            "payment_id": payment_id_old,
            "side": "debit",
            "account_id": f"acc_sc14_old_{uuid.uuid4().hex[:8]}",
            "counterparty_bank_id": "bank_0",
            "sender_bank_id": "bank_1",
            "receiver_bank_id": "bank_0",
            "amount_paise": 15000,
            "event_time": old_event_time,
            "status": "success",
        })
        headers_fresh = _make_signed_headers(
            "bank_1",
            api_key,
            "POST",
            "/v1/transactions",
            body_old,
            clock,
            idempotency_key=f"idem_sc14_{uuid.uuid4().hex}",
        )
        resp_fresh = await client.post("/v1/transactions", content=body_old, headers=headers_fresh)
        assert resp_fresh.status_code == 201

        stale_clock = ManualClock(clock.now_ist() - timedelta(seconds=301))
        headers_stale = _make_signed_headers(
            "bank_1",
            api_key,
            "POST",
            "/v1/transactions",
            body_old,
            stale_clock,
            idempotency_key=f"idem_sc14_{uuid.uuid4().hex}",
        )
        resp_stale = await client.post("/v1/transactions", content=body_old, headers=headers_stale)
        assert resp_stale.status_code == 401
        data_stale = resp_stale.json()
        assert data_stale["error"]["code"] == "stale_signature"

        return CheckOutcome(
            key_number=(
                f"fresh_status={resp_fresh.status_code}, stale_status={resp_stale.status_code}, "
                f"stale_error={data_stale['error']['code']}"
            ),
            evidence={
                "fresh_status": resp_fresh.status_code,
                "stale_status": resp_stale.status_code,
                "stale_error": data_stale["error"],
            },
        )


async def check_10_fuzz_endpoints_negative(workspace: Workspace, quick: bool = False) -> CheckOutcome:
    """
    Check 10 (negative): malformed and unsigned requests get clear 4xx, never 5xx
    (call generate_fuzz_cases, run_unsigned_fuzz_pass, run_signed_fuzz_pass with
    max_examples=10 in quick mode, configured value otherwise; evidence: request count and status histogram).
    """
    async with _in_process_bank_app(workspace, "bank_1") as (client, api_key, clock, _pool):
        max_examples = 10 if quick else load_bank_api_policy().fuzz_max_examples_per_endpoint
        cases = generate_fuzz_cases(max_examples=max_examples)
        unsigned_hist = await run_unsigned_fuzz_pass(client, cases)
        signed_hist = await run_signed_fuzz_pass(client, clock, "bank_1", api_key, cases)
        total_requests = len(cases) * 2

        return CheckOutcome(
            key_number=(
                f"fuzz_cases={len(cases)}, total_requests={total_requests}, "
                f"unsigned={unsigned_hist}, signed={signed_hist}"
            ),
            evidence={
                "quick_mode": quick,
                "max_examples_per_endpoint": max_examples,
                "fuzz_cases_count": len(cases),
                "total_requests_executed": total_requests,
                "unsigned_status_histogram": {str(k): v for k, v in unsigned_hist.items()},
                "signed_status_histogram": {str(k): v for k, v in signed_hist.items()},
            },
        )


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """
    Return the Level 14 verification checks.
    Tagged with part='a' and part='b' for selective Level 14 showcase execution.
    """
    async def run_check_10(workspace: Workspace) -> CheckOutcome:
        return await check_10_fuzz_endpoints_negative(workspace, quick=quick)

    return [
        CheckDefinition(
            name="1. bank_1 key reads bank_2 account yields 0 rows and cross-bank insert refused",
            kind="negative",
            proves="bank_1's key reads bank_2's account -> 0 rows, and cross-bank insert is refused (42501).",
            run=check_1_bank_isolation_negative,
            likely_cause="PostgreSQL RLS policy missing or bank_scoped_transaction did not set local role.",
            where_to_look="migrations/0006_add_upi_columns_and_bank_isolation.sql, api/v1/bank_scope.py",
            part="a",
        ),
        CheckDefinition(
            name="2. two banks interleaved on a 1-connection pool without state bleed",
            kind="edge",
            proves=(
                "Two banks alternating on a single connection pool maintain complete isolation "
                "with no leftover role or bank setting."
            ),
            run=check_2_pooled_interleaving_edge,
            likely_cause="SET LOCAL ROLE or set_config bled across transaction boundaries on pooled connection.",
            where_to_look="api/v1/bank_scope.py",
            part="a",
        ),
        CheckDefinition(
            name="3. Level 7 outbox consistency",
            kind="consistency",
            proves="Level 7 transactional outbox showcase passes cleanly without regression after Level 14 changes.",
            run=check_3_level_7_consistency,
            likely_cause="Migration 0006 or outbox relay routing broke Level 7 transactional outbox.",
            where_to_look="migrations/0006_add_upi_columns_and_bank_isolation.sql, api/services/outbox_relay.py",
            part="a",
        ),
        CheckDefinition(
            name="4. cross-bank payment records debit and credit sides with v5 UUIDs and UPI topic routing",
            kind="normal",
            proves=(
                "A cross-bank payment records 2 tx rows, 2 outbox rows with schema 2 and UUIDv5, "
                "routed to upi.payments."
            ),
            run=check_4_cross_bank_payment_normal,
            likely_cause="UPI outbox writer or relay routing logic failed.",
            where_to_look="api/v1/upi_outbox.py, api/services/outbox_relay.py",
            part="a",
        ),
        CheckDefinition(
            name="5. duplicate payment side idempotent replay without duplicate outbox event",
            kind="edge",
            proves=(
                "Submitting the same payment side twice yields was_replay=True, matching tx ID, "
                "and no duplicate outbox event."
            ),
            run=check_5_replay_idempotency_edge,
            likely_cause="ON CONFLICT handling or replay detection in record_upi_side failed.",
            where_to_look="api/v1/upi_outbox.py",
            part="a",
        ),
        CheckDefinition(
            name="6. signed POST /v1/transactions creates record and outbox event, and GET risk returns unscored",
            kind="normal",
            proves=(
                "Signed POST /v1/transactions creates a database row and outbox event; "
                "subsequent signed GET /v1/accounts/{account_id}/risk returns 200 unscored."
            ),
            run=check_6_post_transaction_and_get_risk_normal,
            likely_cause="POST /v1/transactions handler or GET /v1/accounts/{account_id}/risk query failed.",
            where_to_look="api/v1/bank_router.py, api/v1/upi_outbox.py",
            part="b",
        ),
        CheckDefinition(
            name="7. replaying same request with same idempotency key creates exactly one database row",
            kind="edge",
            proves=(
                "Submitting the same signed request twice with the same idempotency key yields a replay "
                "and exactly one database row."
            ),
            run=check_7_idempotency_same_key_same_body_edge,
            likely_cause="Idempotency replay detection or response cache failed in bank_router.",
            where_to_look="api/v1/bank_router.py, api/v1/bank_idempotency.py",
            part="b",
        ),
        CheckDefinition(
            name="8. reusing same idempotency key with different body is refused with 409 conflict",
            kind="edge",
            proves=(
                "Reusing an existing idempotency key with a differing payload is rejected "
                "with 409 idempotency_key_reused."
            ),
            run=check_8_idempotency_different_body_conflict_edge,
            likely_cause="Payload hash comparison or 409 conflict raising failed in bank_router.",
            where_to_look="api/v1/bank_router.py, api/v1/bank_idempotency.py",
            part="b",
        ),
        CheckDefinition(
            name="9. replay with 2020 event_time and fresh signature accepted, stale signature refused",
            kind="edge",
            proves=(
                "A historical transaction with 2020 event_time is accepted with a fresh signature, "
                "but rejected with 401 when signature is stale."
            ),
            run=check_9_fresh_signature_historical_event_edge,
            likely_cause="Transport clock verification mixed event_time with transport timestamp.",
            where_to_look="api/v1/request_auth.py, api/v1/signing.py",
            part="b",
        ),
        CheckDefinition(
            name="10. malformed and unsigned fuzz requests yield 4xx, never 5xx server errors",
            kind="negative",
            proves=(
                "Unsigned requests are rejected with 401/413 and signed fuzz requests "
                "never produce 5xx server crashes."
            ),
            run=run_check_10,
            likely_cause="Unhandled exception in bank_router or missing authentication guard.",
            where_to_look="api/v1/bank_router.py, api/v1/request_auth.py, tests/contract/test_bank_api_fuzz.py",
            part="b",
        ),
    ]
