# File: tests/integration/test_bank_isolation.py
# Purpose: Proves PostgreSQL Row Level Security (RLS) tenant isolation and bank API key lifecycle.
from datetime import datetime, timezone

import asyncpg
import pytest

from api.v1.bank_keys import issue_api_key, resolve_bank_for_key, revoke_api_key
from api.v1.bank_scope import BANK_ROLE, bank_scoped_transaction
from tests.integration.conftest import get_test_database_url

pytestmark = pytest.mark.anyio


async def test_bank_1_key_cannot_read_bank_2_account(isolated_schema):
    """Under bank_1 scope, bank_2 rows are invisible; under bank_2 scope, the row is visible."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    # 1. Issue API keys for bank_1 and bank_2 as superuser/owner
    _, key_bank_1 = await issue_api_key(connection, "bank_1")
    _, key_bank_2 = await issue_api_key(connection, "bank_2")

    # 2. Insert one world row for bank_2 account X as superuser
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                 amount_paise, event_time, status)
        VALUES ('world', 'acc_bank2_x', 500.00, 'pmt_iso_1', 'bank_2', 'debit', 50000, $1, 'success')
        """,
        now,
    )

    # 3. Resolve presented key to bank_1
    resolved_bank = await resolve_bank_for_key(connection, key_bank_1)
    assert resolved_bank == "bank_1"

    # 4. Under bank_1 scoped transaction, querying transactions with NO bank filter yields 0 rows for account X
    async with bank_scoped_transaction(connection, resolved_bank) as scoped_conn:
        rows_bank_1 = await scoped_conn.fetch(
            "SELECT id, account_id, amount FROM transactions WHERE account_id = 'acc_bank2_x'"
        )
        assert len(rows_bank_1) == 0

    # 5. Under bank_2 scoped transaction, querying transactions with NO bank filter yields exactly 1 row
    resolved_bank_2 = await resolve_bank_for_key(connection, key_bank_2)
    assert resolved_bank_2 == "bank_2"

    async with bank_scoped_transaction(connection, resolved_bank_2) as scoped_conn:
        rows_bank_2 = await scoped_conn.fetch(
            "SELECT id, account_id, amount FROM transactions WHERE account_id = 'acc_bank2_x'"
        )
        assert len(rows_bank_2) == 1
        assert rows_bank_2[0]["account_id"] == "acc_bank2_x"


async def test_bank_cannot_insert_row_for_another_bank(isolated_schema):
    """A bank in bank_1 scope attempting to insert a row with bank_id='bank_2' fails with sqlstate 42501."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    async with bank_scoped_transaction(connection, "bank_1") as scoped_conn:
        with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc_info:
            await scoped_conn.execute(
                """
                INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                         amount_paise, event_time, status)
                VALUES ('world', 'acc_rogue_insert', 100.00, 'pmt_rogue_1', 'bank_2', 'debit', 10000, $1, 'success')
                """,
                now,
            )

        assert exc_info.value.sqlstate == "42501"


async def test_unscoped_bank_role_sees_nothing(isolated_schema):
    """Assuming ashfall_bank_api without set_config('app.bank_id') matches no rows and fails closed."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    # Insert rows for bank_0 and bank_1 as superuser
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                 amount_paise, event_time, status)
        VALUES ('world', 'acc_unscoped_0', 10.00, 'pmt_unscoped_0', 'bank_0', 'debit', 1000, $1, 'success'),
               ('world', 'acc_unscoped_1', 20.00, 'pmt_unscoped_1', 'bank_1', 'debit', 2000, $1, 'success')
        """,
        now,
    )

    async with connection.transaction():
        await connection.execute(f"SET LOCAL ROLE {BANK_ROLE}")
        # Note: app.bank_id is NOT set
        rows = await connection.fetch("SELECT id FROM transactions")
        assert len(rows) == 0


async def test_two_banks_interleaved_on_one_pooled_connection(isolated_schema):
    """Alternating bank scopes on a single-connection pool maintains complete isolation without state bleed."""
    _, schema_name = isolated_schema
    dsn = get_test_database_url()
    now = datetime.now(timezone.utc)

    # Use a pool with strictly 1 connection
    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=1,
        server_settings={"search_path": schema_name},
    )
    assert pool is not None

    try:
        # 1. Insert distinct rows for bank_0 and bank_1 as superuser
        async with pool.acquire() as conn:
            orig_user = await conn.fetchval("SELECT current_user")
            fixed_backend_pid = await conn.fetchval("SELECT pg_backend_pid()")

            await conn.execute(
                """
                INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                         amount_paise, event_time, status)
                VALUES ('world', 'acc_pool_0', 100.00, 'pmt_pool_0', 'bank_0', 'debit', 10000, $1, 'success'),
                       ('world', 'acc_pool_1', 200.00, 'pmt_pool_1', 'bank_1', 'debit', 20000, $1, 'success')
                """,
                now,
            )

        # 2. Interleave acquire -> bank_0 scope -> release -> bank_1 scope -> release for 3 rounds
        for round_idx in range(3):
            # Round: bank_0
            async with pool.acquire() as conn:
                pid = await conn.fetchval("SELECT pg_backend_pid()")
                assert pid == fixed_backend_pid, f"Pool connection pid changed on round {round_idx}!"

                async with bank_scoped_transaction(conn, "bank_0"):
                    rows = await conn.fetch("SELECT account_id FROM transactions")
                    assert len(rows) == 1
                    assert rows[0]["account_id"] == "acc_pool_0"

                # Immediately after exiting context manager on the same connection:
                user = await conn.fetchval("SELECT current_user")
                assert user == orig_user
                setting = await conn.fetchval("SELECT current_setting('app.bank_id', true)")
                assert setting is None or setting == ""

            # Round: bank_1
            async with pool.acquire() as conn:
                pid = await conn.fetchval("SELECT pg_backend_pid()")
                assert pid == fixed_backend_pid, f"Pool connection pid changed on round {round_idx}!"

                async with bank_scoped_transaction(conn, "bank_1"):
                    rows = await conn.fetch("SELECT account_id FROM transactions")
                    assert len(rows) == 1
                    assert rows[0]["account_id"] == "acc_pool_1"

                # Verify clean reset after release
                user = await conn.fetchval("SELECT current_user")
                assert user == orig_user
                setting = await conn.fetchval("SELECT current_setting('app.bank_id', true)")
                assert setting is None or setting == ""

    finally:
        await pool.close()


async def test_rotation_old_and_new_key_work_until_revoked(isolated_schema):
    """Both old and new keys authenticate the bank during rotation overlap until old key is revoked."""
    connection, _ = isolated_schema

    # Issue initial key
    key_id_1, key_plaintext_1 = await issue_api_key(connection, "bank_0")
    assert await resolve_bank_for_key(connection, key_plaintext_1) == "bank_0"

    # Issue replacement key (rotation overlap)
    key_id_2, key_plaintext_2 = await issue_api_key(connection, "bank_0")
    assert key_id_1 != key_id_2
    assert await resolve_bank_for_key(connection, key_plaintext_1) == "bank_0"
    assert await resolve_bank_for_key(connection, key_plaintext_2) == "bank_0"

    # Revoke old key
    revoked = await revoke_api_key(connection, key_id_1)
    assert revoked is True

    # Old key fails, new key succeeds
    assert await resolve_bank_for_key(connection, key_plaintext_1) is None
    assert await resolve_bank_for_key(connection, key_plaintext_2) == "bank_0"


async def test_revoked_key_resolves_to_none(isolated_schema):
    """A revoked key resolves to None, and repeated revocation returns False."""
    connection, _ = isolated_schema

    key_id, plaintext = await issue_api_key(connection, "bank_1")
    assert await resolve_bank_for_key(connection, plaintext) == "bank_1"

    assert await revoke_api_key(connection, key_id) is True
    assert await resolve_bank_for_key(connection, plaintext) is None

    # Repeating revocation returns False because key is already revoked
    assert await revoke_api_key(connection, key_id) is False


async def test_malformed_and_wrong_secret_keys_resolve_to_none(isolated_schema):
    """Malformed key formats, non-existent key IDs, and corrupted secrets resolve to None."""
    connection, _ = isolated_schema

    key_id, plaintext = await issue_api_key(connection, "bank_2")

    # Malformed keys
    assert await resolve_bank_for_key(connection, "") is None
    assert await resolve_bank_for_key(connection, "not_a_key") is None
    assert await resolve_bank_for_key(connection, "ak_missing_dot") is None
    assert await resolve_bank_for_key(connection, "bk_wrong_prefix.secret") is None
    assert await resolve_bank_for_key(connection, "ak_.") is None

    # Non-existent key ID
    assert await resolve_bank_for_key(connection, "ak_0000000000000000.secret") is None

    # Tampered secret
    prefix, secret = plaintext.split(".", 1)
    corrupted_secret = secret[:-1] + ("A" if secret[-1] != "A" else "B")
    assert await resolve_bank_for_key(connection, f"{prefix}.{corrupted_secret}") is None
