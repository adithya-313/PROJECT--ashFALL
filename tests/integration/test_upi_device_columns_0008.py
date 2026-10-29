# File: tests/integration/test_upi_device_columns_0008.py
# Purpose: Proves Level 14c migration 0008: optional sender device columns on transactions,
#          debit-only validation at endpoint and database constraint levels, outbox payload integration,
#          and backward compatibility when device fields are omitted.

from datetime import datetime
from typing import Any

import asyncpg
import orjson
import pytest
from pydantic import SecretStr

from api.main import app
from api.v1.bank_keys import issue_api_key
from core.config import config
from core.event_clock import IST, ManualClock
from tests.integration.conftest import get_test_database_url
from tests.integration.test_bank_endpoints import EndpointEnv, post_json, transaction_payload

pytestmark = pytest.mark.anyio

BANKS = ("bank_0", "bank_1", "bank_2")
START_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)


@pytest.fixture
async def env(isolated_schema: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
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

pytestmark = pytest.mark.anyio


async def test_debit_side_with_device_fields_stores_and_populates_outbox(env: EndpointEnv) -> None:
    """A debit side carrying device fields stores them in transactions and populates outbox payload data."""
    payload = transaction_payload(
        payment_id="pmt_dev_001",
        sender_device_fingerprint="fp_android_v3_abcdef12",
        sender_os_generation=4,
    )
    response = await post_json(env, "bank_0", "/v1/transactions", payload, "idem_dev_001")

    assert response.status_code == 201
    body = response.json()
    assert body["was_replay"] is False

    # Check transactions table row
    row = await env.connection.fetchrow(
        """
        SELECT payment_id, side, sender_device_fingerprint, sender_os_generation
        FROM transactions
        WHERE payment_id = 'pmt_dev_001'
        """
    )
    assert row is not None
    assert row["payment_id"] == "pmt_dev_001"
    assert row["side"] == "debit"
    assert row["sender_device_fingerprint"] == "fp_android_v3_abcdef12"
    assert row["sender_os_generation"] == 4

    # Check outbox event payload data
    outbox_row = await env.connection.fetchrow(
        """
        SELECT o.payload
        FROM outbox o
        JOIN transactions t ON o.transaction_id = t.transaction_id
        WHERE t.payment_id = 'pmt_dev_001'
        """
    )
    assert outbox_row is not None
    payload_dict = orjson.loads(outbox_row["payload"])
    assert payload_dict["data"]["sender_device_fingerprint"] == "fp_android_v3_abcdef12"
    assert payload_dict["data"]["sender_os_generation"] == 4


async def test_credit_side_with_device_fields_refused_via_endpoint_422(env: EndpointEnv) -> None:
    """A credit side request containing device fields is rejected with 422 through the endpoint."""
    credit_payload = {
        "payment_id": "pmt_dev_credit_fail",
        "side": "credit",
        "account_id": "acc_bank1_b",
        "counterparty_vpa": "user@bank0",
        "counterparty_bank_id": "bank_0",
        "sender_bank_id": "bank_0",
        "receiver_bank_id": "bank_1",
        "amount_paise": 150000,
        "event_time": "2026-10-28T11:30:00+05:30",
        "status": "success",
        "sender_device_fingerprint": "fp_android_v3_abcdef12",
        "sender_os_generation": 4,
    }
    response = await post_json(env, "bank_1", "/v1/transactions", credit_payload, "idem_dev_credit_fail")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_body"

    # Confirm no rows written
    tx_count = await env.connection.fetchval(
        "SELECT count(*) FROM transactions WHERE payment_id = 'pmt_dev_credit_fail'"
    )
    assert tx_count == 0


async def test_credit_side_with_device_fields_refused_by_db_check(isolated_schema: Any) -> None:
    """Direct insertion of credit side with device fields violates chk_device_columns_debit_only."""
    connection, _ = isolated_schema
    with pytest.raises(asyncpg.CheckViolationError, match="chk_device_columns_debit_only"):
        await connection.execute(
            """
            INSERT INTO transactions (
                source, account_id, amount, payment_id, bank_id, side,
                counterparty_bank_id, amount_paise, event_time, status,
                sender_device_fingerprint, sender_os_generation
            ) VALUES (
                'world', 'acc_credit_0', 15.00, 'pmt_db_chk_fail', 'bank_1', 'credit',
                'bank_0', 1500, now(), 'success',
                'fp_android_v3_forbidden', 2
            )
            """
        )


async def test_transaction_without_device_fields_works_backward_compatible(env: EndpointEnv) -> None:
    """A request omitting device fields succeeds and stores NULL in both columns."""
    payload = transaction_payload(payment_id="pmt_dev_compat_001")
    assert "sender_device_fingerprint" not in payload
    assert "sender_os_generation" not in payload

    response = await post_json(env, "bank_0", "/v1/transactions", payload, "idem_compat_001")
    assert response.status_code == 201

    row = await env.connection.fetchrow(
        """
        SELECT sender_device_fingerprint, sender_os_generation
        FROM transactions
        WHERE payment_id = 'pmt_dev_compat_001'
        """
    )
    assert row is not None
    assert row["sender_device_fingerprint"] is None
    assert row["sender_os_generation"] is None

    outbox_row = await env.connection.fetchrow(
        """
        SELECT o.payload
        FROM outbox o
        JOIN transactions t ON o.transaction_id = t.transaction_id
        WHERE t.payment_id = 'pmt_dev_compat_001'
        """
    )
    assert outbox_row is not None
    payload_dict = orjson.loads(outbox_row["payload"])
    assert payload_dict["data"]["sender_device_fingerprint"] is None
    assert payload_dict["data"]["sender_os_generation"] is None


async def test_ashfall_bank_api_can_insert_new_columns_under_rls(isolated_schema: Any) -> None:
    """Proves that ashfall_bank_api role can insert new device columns under Row Level Security."""
    connection, _ = isolated_schema
    async with connection.transaction():
        await connection.execute("SET LOCAL ROLE ashfall_bank_api")
        await connection.execute("SET LOCAL app.bank_id = 'bank_0'")
        row = await connection.fetchrow(
            """
            INSERT INTO transactions (
                source, account_id, amount, payment_id, bank_id, side,
                counterparty_bank_id, amount_paise, event_time, status,
                sender_device_fingerprint, sender_os_generation
            ) VALUES (
                'world', 'acc_bank0_rls', 20.00, 'pmt_rls_dev_001', 'bank_0', 'debit',
                'bank_1', 2000, now(), 'success',
                'fp_android_v3_rls_allowed', 3
            )
            RETURNING transaction_id, sender_device_fingerprint, sender_os_generation
            """
        )
        assert row is not None
        print(
            f"RLS_INSERT_ROW: transaction_id={row['transaction_id']} "
            f"fingerprint={row['sender_device_fingerprint']} "
            f"os_gen={row['sender_os_generation']}"
        )
        assert row["sender_device_fingerprint"] == "fp_android_v3_rls_allowed"
        assert row["sender_os_generation"] == 3

