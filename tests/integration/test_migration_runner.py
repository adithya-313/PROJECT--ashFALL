# File: tests/integration/test_migration_runner.py
# Purpose: Proves scripts/run_migrations.py applies files in order, is safe to repeat, and never records a failed file.
#
# Each test starts from an EMPTY throwaway schema (the empty_schema fixture in conftest.py, which does not run any
# migrations), so the runner itself is what builds the tables.
import shutil

import asyncpg
import pytest

from scripts.run_migrations import MIGRATIONS_DIR, apply_pending_migrations, find_migration_files

pytestmark = pytest.mark.anyio

# The transactions table columns, in order. The Level 10.5 agent reads this table, so changing it must break this
# test on purpose and force a deliberate decision.
EXPECTED_TRANSACTION_COLUMNS = [
    "id",
    "transaction_id",
    "source",
    "account_id",
    "merchant_id",
    "amount",
    "device_fingerprint",
    "geo_coordinates",
    "txn_type",
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
    "fraud_score",
    "model_version",
    "details",
    "created_at",
    "payment_id",
    "bank_id",
    "side",
    "counterparty_vpa",
    "counterparty_bank_id",
    "amount_paise",
    "event_time",
    "status",
    "failure_domain",
    "payee_type",
    "channel",
    "sender_device_fingerprint",
    "sender_os_generation",
    "balance_after_paise",
]


async def applied_filenames(connection: asyncpg.Connection) -> list[str]:
    """The migration files the tracking table says were applied, sorted by name."""
    rows = await connection.fetch("SELECT filename FROM schema_migrations ORDER BY filename")
    return [row["filename"] for row in rows]


async def table_names_in_schema(connection: asyncpg.Connection, schema_name: str) -> set[str]:
    """The names of the tables that exist in this test's schema only."""
    rows = await connection.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = $1", schema_name
    )
    return {row["table_name"] for row in rows}


async def test_fresh_schema_applies_all_migrations_in_order(empty_schema):
    """On an empty schema the runner applies every real migration file and creates every expected table."""
    connection, schema_name = empty_schema
    real_filenames = [path.name for path in find_migration_files()]

    applied_count = await apply_pending_migrations(connection)

    # All files applied (we expect 12 today), and the tracking table lists exactly those names.
    assert applied_count == len(real_filenames) == 12
    assert await applied_filenames(connection) == real_filenames
    assert real_filenames == sorted(real_filenames)

    # The tables were created inside THIS schema, not in public.
    tables = await table_names_in_schema(connection, schema_name)
    expected_tables = {
        "transactions",
        "outbox",
        "processed_events",
        "demo_event_counts",
        "outbox_dead_letter",
        "national_outbox",
        "bank_api_keys",
        "schema_migrations",
        "bank_request_nonces",
        "bank_idempotency_keys",
        "bank_decision_requests",
        "bank_account_events",
        "bank_complaints",
        "bank_feedback",
        "feature_log",
    }
    assert expected_tables <= tables


async def test_second_run_applies_nothing(empty_schema):
    """Running the runner twice is safe: the second run applies nothing and records nothing new."""
    connection, _ = empty_schema

    first_count = await apply_pending_migrations(connection)
    second_count = await apply_pending_migrations(connection)

    assert first_count == len(find_migration_files()) == 12
    assert second_count == 0
    assert len(await applied_filenames(connection)) == 12


async def test_failed_migration_is_not_recorded(empty_schema, tmp_path):
    """A migration file with bad SQL fails, is not recorded as applied, and does not undo the earlier good files."""
    connection, _ = empty_schema

    # Copy the real migrations into a temp folder and add one broken file that sorts last.
    copied_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, copied_dir)
    (copied_dir / "0099_broken.sql").write_text("THIS IS NOT VALID SQL;", encoding="utf-8")

    # The runner must raise when it reaches the broken file.
    with pytest.raises(asyncpg.PostgresError):
        await apply_pending_migrations(connection, copied_dir)

    # The three good files stay recorded, and the broken one is absent. Each file is its own transaction, so the
    # failure rolled back only itself.
    recorded = await applied_filenames(connection)
    assert "0099_broken.sql" not in recorded
    assert recorded == [path.name for path in find_migration_files()]


async def test_transactions_table_column_count_is_locked(isolated_schema):
    """The transactions table has exactly the 31 locked columns, in order."""
    connection, schema_name = isolated_schema

    rows = await connection.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = 'transactions'
        ORDER BY ordinal_position
        """,
        schema_name,
    )
    column_names = [row["column_name"] for row in rows]

    assert len(column_names) == 31
    assert column_names == EXPECTED_TRANSACTION_COLUMNS


async def column_names_of(connection: asyncpg.Connection, schema_name: str, table_name: str) -> list[str]:
    """The column names of one table in the test schema, in table order."""
    rows = await connection.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2
        ORDER BY ordinal_position
        """,
        schema_name,
        table_name,
    )
    return [row["column_name"] for row in rows]


async def test_outbox_and_dead_letter_columns_are_locked(isolated_schema):
    """The outbox has the four dead-letter tracking columns, and outbox_dead_letter has exactly its listed columns."""
    connection, schema_name = isolated_schema

    # Any future change to these tables must break this test on purpose, so it is a deliberate decision.
    outbox_columns = await column_names_of(connection, schema_name, "outbox")
    for new_column in ("attempt_count", "last_error", "first_failed_at", "dead_lettered_at"):
        assert new_column in outbox_columns

    dead_letter_columns = await column_names_of(connection, schema_name, "outbox_dead_letter")
    assert dead_letter_columns == [
        "id",
        "outbox_id",
        "event_id",
        "transaction_id",
        "payload",
        "error_type",
        "error_message",
        "attempt_count",
        "first_failed_at",
        "dead_lettered_at",
    ]
