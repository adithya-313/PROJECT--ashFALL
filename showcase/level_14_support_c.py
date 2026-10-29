# File: showcase/level_14_support_c.py
# Purpose: Level 14 part c support helpers: bank connectors, shadow mode parity, R7 visibility, leak checks.
#
# Provides the execution harness for Part 14c showcase checks:
#   - Check 11: HTTP bank connectors end-to-end and outbox relay to throwaway Kafka topic
#   - Check 12: In-process shadow mode parity preserving byte-for-byte world output
#   - Check 13: R7 failure domain visibility rules (npci_central debit-only vs receiver_bank both sides)
#   - Check 14: Strict boundary refusal of non-bank-visible fields ("purpose") in projection and API schema

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, cast

import asyncpg
import httpx
import orjson
from pydantic import SecretStr

from api.main import app
from api.services.outbox_relay import relay_one_batch
from api.v1.bank_clients import SignedHttpBankClient
from api.v1.bank_keys import issue_api_key
from api.v1.bank_service import InProcessBankClient
from core.config import config
from core.event_clock import ManualClock
from showcase.framework import CheckOutcome
from showcase.isolation import Workspace, cleanup_workspace, create_workspace
from showcase.kafka_tools import (
    create_topic,
    delete_topic,
    make_producer,
    make_topic_name,
    read_events,
)
from world.bank_connectors import BankTransport, run_world_through_banks
from world.bank_feed import (
    BANK_VISIBLE_PAYMENT_FIELDS,
    ProjectionLeakError,
    check_bank_visible,
    project_run,
)
from world.interactive_events import interactive_event_lines
from world.interactive_records import allow_all
from world.interactive_runner import (
    FastRunSettings,
    RunnerSettings,
    run_interactive_world,
)

BANKS: tuple[str, ...] = ("bank_0", "bank_1", "bank_2")
START_TIME: datetime = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


@asynccontextmanager
async def workspace_bank_env(
    workspace: Workspace,
) -> AsyncGenerator[tuple[dict[str, str], ManualClock, asyncpg.Pool], None]:
    """
    WHAT: Sets up an isolated FastAPI execution environment for all three commercial banks on workspace schema.
    Configures signing secrets in runtime config, generates authenticated API keys, establishes a connection
    pool bound to workspace.schema_name, initializes a ManualClock, and updates app.state.

    WHY: Allows showcase checks to execute InProcessBankClient and SignedHttpBankClient operations against
    the active showcase workspace with complete tenant isolation and reproducible clocks.
    """
    saved_secrets: dict[str, Any] = {}
    for b in BANKS:
        attr = f"bank_signing_secret_{b}"
        saved_secrets[attr] = getattr(config, attr, None)
        setattr(config, attr, SecretStr(f"showcase_secret_for_{b}_32_characters_min!"))

    api_keys: dict[str, str] = {}
    for b in BANKS:
        _, api_keys[b] = await issue_api_key(workspace.connection, b)

    pool = await asyncpg.create_pool(
        dsn=workspace.dsn,
        min_size=1,
        max_size=5,
        server_settings={"search_path": workspace.schema_name},
    )
    assert pool is not None

    clock = ManualClock(START_TIME)
    saved_state: dict[str, Any] = {}
    for name in ("db_pool", "transport_clock"):
        if hasattr(app.state, name):
            saved_state[name] = getattr(app.state, name)
    app.state.db_pool = pool
    app.state.transport_clock = clock

    try:
        yield api_keys, clock, pool
    finally:
        for name in ("db_pool", "transport_clock"):
            if name in saved_state:
                setattr(app.state, name, saved_state[name])
            else:
                delattr(app.state, name)
        for attr, val in saved_secrets.items():
            if val is not None:
                setattr(config, attr, val)
        await pool.close()


def verify_failure_visibility(
    payment_id: str,
    sides: list[asyncpg.Record],
    expected_domain: str,
    expect_credit: bool,
) -> None:
    """
    WHAT: Validates that database transaction records for a failed payment comply with R7 visibility rules.
    WHY: Enforces that non-receiver bank failures (e.g. npci_central) are visible only to the sending bank,
    whereas receiver_bank failures produce records for both sending and receiving banks.
    """
    side_names = sorted(r["side"] for r in sides)
    if expect_credit:
        assert side_names == ["credit", "debit"], (
            f"Payment {payment_id}: expected debit and credit rows for {expected_domain}, got {side_names}"
        )
    else:
        assert side_names == ["debit"], (
            f"Payment {payment_id}: expected debit row only for {expected_domain}, got {side_names}"
        )

    for r in sides:
        assert r["status"] == "failed", f"Payment {payment_id}: expected failed status, got {r['status']}"
        assert r["failure_domain"] == expected_domain, (
            f"Payment {payment_id}: expected failure domain {expected_domain}, got {r['failure_domain']}"
        )


async def check_11_http_connectors_and_outbox_relay_normal(
    workspace: Workspace, quick: bool = False
) -> CheckOutcome:
    """
    Check 11 (normal, part c): Runs world through APIs over HTTP (SignedHttpBankClient, httpx.ASGITransport,
    all three banks, shadow mode). Full mode: FastRunSettings(101, 20, 6, 51, 'training'); Quick mode:
    FastRunSettings(101, 10, 5, 51, 'training'). Drains workspace outbox with relay_one_batch to a throwaway
    Kafka topic, verifies message counts, and cleans up the topic.
    """
    fast_settings = (
        FastRunSettings(101, 10, 5, 51, "training")
        if quick
        else FastRunSettings(101, 20, 6, 51, "training")
    )
    runner_settings = RunnerSettings(fraud_enabled=True)

    isolated_ws = await create_workspace()
    try:
        async with workspace_bank_env(isolated_ws) as (api_keys, clock, _pool):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
                http_transports: dict[str, BankTransport] = cast(
                    dict[str, BankTransport],
                    {b: SignedHttpBankClient(http_client, b, api_keys[b], clock) for b in BANKS},
                )
                report = await run_world_through_banks(
                    fast_settings,
                    runner_settings,
                    transports=http_transports,
                )

        feed = project_run(report.output)
        expected_sides = sum(len(sides) for sides in feed.payments_by_bank.values())
        events_count = len(report.output.result.run_result.events)

        decisions_stored = await isolated_ws.connection.fetchval("SELECT count(*) FROM bank_decision_requests")
        assert decisions_stored == report.decisions_sent + report.decisions_replayed == events_count
        assert len(report.decision_failures) == 0

        tx_rows = await isolated_ws.connection.fetchval("SELECT count(*) FROM transactions")
        assert tx_rows == expected_sides

        # Relay isolated workspace outbox to throwaway Kafka topic
        producer = await make_producer()
        topic = make_topic_name(prefix="l14c")
        await create_topic(topic)
        try:
            while True:
                batch_res = await relay_one_batch(
                    isolated_ws.connection, producer, batch_size=100, topic=topic
                )
                if batch_res.waiting == 0:
                    break
                if batch_res.error is not None and not batch_res.poison_pending:
                    raise RuntimeError(f"Outbox relay error: {batch_res.error}")

            msgs = await read_events(topic, expected_count=tx_rows, idle_timeout_seconds=2.0)
            kafka_count = len(msgs)
        finally:
            await producer.stop()
            await delete_topic(topic)

        schema_2_outbox_rows = await isolated_ws.connection.fetchval(
            "SELECT count(*) FROM outbox WHERE payload->>'schema_version' = '2'"
        )
        assert schema_2_outbox_rows == tx_rows == kafka_count

        key_number = f"decisions={decisions_stored}, transaction sides={tx_rows}, kafka events={kafka_count}"
        evidence = {
            "decisions_stored": decisions_stored,
            "decisions_sent": report.decisions_sent,
            "decisions_replayed": report.decisions_replayed,
            "transactions_stored": tx_rows,
            "projected_sides": expected_sides,
            "schema_2_outbox_rows": schema_2_outbox_rows,
            "kafka_messages": kafka_count,
            "topic": topic,
        }
        return CheckOutcome(key_number=key_number, evidence=evidence)
    finally:
        await cleanup_workspace(isolated_ws)


async def check_12_shadow_mode_parity_edge(workspace: Workspace) -> CheckOutcome:
    """
    Check 12 (edge, part c): Proves that shadow mode execution leaves the simulation output completely unchanged.
    Runs FastRunSettings(101, 10, 5, 51) in-process, verifying that every decision request records mode 'shadow'
    and that interactive_event_lines matches a plain allow_all run line-for-line.
    """
    fast_settings = FastRunSettings(101, 10, 5, 51, "training")
    runner_settings = RunnerSettings(fraud_enabled=True)

    plain_output = run_interactive_world(fast_settings, runner_settings, hook=allow_all)
    plain_lines = interactive_event_lines(plain_output)

    async with workspace_bank_env(workspace) as (_api_keys, clock, pool):
        in_proc_transports: dict[str, BankTransport] = cast(
            dict[str, BankTransport],
            {b: InProcessBankClient(pool, b, clock) for b in BANKS},
        )
        report = await run_world_through_banks(
            fast_settings,
            runner_settings,
            transports=in_proc_transports,
        )

    connector_lines = interactive_event_lines(report.output)
    assert connector_lines == plain_lines

    non_shadow_decisions = await workspace.connection.fetchval(
        "SELECT count(*) FROM bank_decision_requests WHERE mode != 'shadow'"
    )
    assert non_shadow_decisions == 0

    decisions_recorded = await workspace.connection.fetchval("SELECT count(*) FROM bank_decision_requests")
    assert decisions_recorded > 0

    key_number = f"decisions recorded={decisions_recorded}, event lines identical=yes"
    evidence = {
        "decisions_recorded": decisions_recorded,
        "non_shadow_decisions": non_shadow_decisions,
        "interactive_event_lines_matched": len(plain_lines),
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def check_13_r7_visibility_edge(workspace: Workspace) -> CheckOutcome:
    """
    Check 13 (edge, part c, R7): From stored rows of an in-process run of FastRunSettings(101, 10, 5, 51),
    demonstrates one npci_central failure (debit row only) and one receiver_bank failure (debit and credit rows).
    Identifies the earliest occurrence of each failure domain by event_time without hardcoded IDs.
    """
    # Ensure in-process rows for 101/10/5/51 exist in the workspace database
    stored_tx = await workspace.connection.fetchval(
        "SELECT count(*) FROM transactions WHERE failure_domain = 'npci_central'"
    )
    if not stored_tx:
        fast_settings = FastRunSettings(101, 10, 5, 51, "training")
        runner_settings = RunnerSettings(fraud_enabled=True)
        async with workspace_bank_env(workspace) as (_api_keys, clock, pool):
            in_proc_transports: dict[str, BankTransport] = cast(
                dict[str, BankTransport],
                {b: InProcessBankClient(pool, b, clock) for b in BANKS},
            )
            await run_world_through_banks(
                fast_settings,
                runner_settings,
                transports=in_proc_transports,
            )

    # 1. First npci_central failure by event_time
    npci_row = await workspace.connection.fetchrow(
        "SELECT payment_id, event_time FROM transactions "
        "WHERE failure_domain = 'npci_central' ORDER BY event_time, id LIMIT 1"
    )
    assert npci_row is not None, "No npci_central failure found in transactions"
    npci_payment_id = npci_row["payment_id"]

    npci_sides = await workspace.connection.fetch(
        "SELECT side, status, failure_domain FROM transactions WHERE payment_id = $1 ORDER BY side",
        npci_payment_id,
    )
    verify_failure_visibility(npci_payment_id, npci_sides, "npci_central", expect_credit=False)

    # 2. First receiver_bank failure by event_time
    recv_row = await workspace.connection.fetchrow(
        "SELECT payment_id, event_time FROM transactions "
        "WHERE failure_domain = 'receiver_bank' ORDER BY event_time, id LIMIT 1"
    )
    assert recv_row is not None, "No receiver_bank failure found in transactions"
    recv_payment_id = recv_row["payment_id"]

    recv_sides = await workspace.connection.fetch(
        "SELECT side, status, failure_domain FROM transactions WHERE payment_id = $1 ORDER BY side",
        recv_payment_id,
    )
    verify_failure_visibility(recv_payment_id, recv_sides, "receiver_bank", expect_credit=True)

    npci_sides_list = [r["side"] for r in npci_sides]
    recv_sides_list = [r["side"] for r in recv_sides]
    key_number = "npci_central: debit only; receiver_bank: both sides"
    evidence = {
        "npci_central_payment_id": npci_payment_id,
        "npci_central_event_time": str(npci_row["event_time"]),
        "npci_central_sides": npci_sides_list,
        "receiver_bank_payment_id": recv_payment_id,
        "receiver_bank_event_time": str(recv_row["event_time"]),
        "receiver_bank_sides": recv_sides_list,
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def check_14_boundary_purpose_leak_negative(workspace: Workspace) -> CheckOutcome:
    """
    Check 14 (negative, part c): Verifies that non-bank-visible metadata ('purpose') is rejected at boundaries:
    (a) check_bank_visible on a projected record carrying 'purpose' raises ProjectionLeakError;
    (b) signed POST /v1/transactions with 'purpose' in body returns 422, storing 0 database rows.
    """
    # (a) Verify projection leak prevention
    leaked_record = {
        "payment_id": "pmt_leak_test_001",
        "side": "debit",
        "account_id": "acc_person_000000",
        "counterparty_vpa": "",
        "counterparty_bank_id": "bank_2",
        "sender_bank_id": "bank_1",
        "receiver_bank_id": "bank_2",
        "amount_paise": 10000,
        "event_time": datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc),
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "upi_app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "bank_id": "bank_1",
        "purpose": "unauthorized_rent_tag",
    }
    projection_refused = False
    try:
        check_bank_visible(leaked_record, BANK_VISIBLE_PAYMENT_FIELDS)
    except ProjectionLeakError as exc:
        if "purpose" in str(exc):
            projection_refused = True
    assert projection_refused, "check_bank_visible failed to raise ProjectionLeakError on 'purpose'"

    # (b) Verify API schema boundary rejects 'purpose' with 422
    from showcase.level_14 import _make_signed_headers

    leak_payment_id = "pmt_purpose_leak_api_001"
    tx_body = {
        "payment_id": leak_payment_id,
        "side": "debit",
        "account_id": "acc_person_000000",
        "counterparty_vpa": "",
        "counterparty_bank_id": "bank_2",
        "sender_bank_id": "bank_1",
        "receiver_bank_id": "bank_2",
        "amount_paise": 10000,
        "event_time": datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "upi_app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "purpose": "unauthorized_rent_tag",
    }
    body_bytes = orjson.dumps(tx_body)

    async with workspace_bank_env(workspace) as (api_keys, clock, _pool):
        headers = _make_signed_headers(
            bank_id="bank_1",
            api_key=api_keys["bank_1"],
            method="POST",
            path="/v1/transactions",
            body=body_bytes,
            clock=clock,
            idempotency_key=f"idem_{leak_payment_id}",
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/transactions", content=body_bytes, headers=headers)
            assert resp.status_code == 422, f"Expected 422 for extra field 'purpose', got {resp.status_code}"

        stored_rows = await workspace.connection.fetchval(
            "SELECT count(*) FROM transactions WHERE payment_id = $1", leak_payment_id
        )
        assert stored_rows == 0, f"Expected 0 stored rows for rejected payment, got {stored_rows}"

    key_number = f"projection: refused; API: 422; rows stored: {stored_rows}"
    evidence = {
        "projection_leak_prevented": True,
        "api_status_code": resp.status_code,
        "database_rows_stored": stored_rows,
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)
