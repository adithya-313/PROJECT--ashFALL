# File: tests/integration/test_national_outbox_table.py
# Purpose: Proves migrations/0005_create_national_outbox.sql creates the national_outbox table with locked columns,
#          topic check constraint, event_id uniqueness, and unprocessed rows partial index.
import uuid

import asyncpg
import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

EXPECTED_NATIONAL_OUTBOX_COLUMNS = [
    "id",
    "event_id",
    "run_id",
    "topic",
    "bank_key",
    "schema_version",
    "payload",
    "created_at",
    "processed_at",
]


async def test_national_outbox_columns_are_locked(isolated_schema) -> None:
    """The national_outbox table has exactly the 9 locked columns, in expected order."""
    connection, schema_name = isolated_schema

    rows = await connection.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = 'national_outbox'
        ORDER BY ordinal_position
        """,
        schema_name,
    )
    columns = [row["column_name"] for row in rows]
    assert columns == EXPECTED_NATIONAL_OUTBOX_COLUMNS


async def test_national_outbox_topic_constraint_accepts_valid_and_rejects_invalid(isolated_schema) -> None:
    """Topic must be either 'registry.events' or 'hold.requests'."""
    connection, _ = isolated_schema
    event_id = uuid.uuid4()

    # Valid topic
    await connection.execute(
        """
        INSERT INTO national_outbox (event_id, run_id, topic, bank_key, schema_version, payload)
        VALUES ($1, 'run-1', 'registry.events', 'run-1:0', 'national-1', '{"test": 1}'::jsonb)
        """,
        event_id,
    )

    # Invalid topic should raise CheckViolationError
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO national_outbox (event_id, run_id, topic, bank_key, schema_version, payload)
            VALUES ($1, 'run-1', 'invalid.topic', 'run-1:0', 'national-1', '{"test": 2}'::jsonb)
            """,
            uuid.uuid4(),
        )


async def test_national_outbox_event_id_unique(isolated_schema) -> None:
    """Duplicate event_id is rejected by unique constraint."""
    connection, _ = isolated_schema
    event_id = uuid.uuid4()

    await connection.execute(
        """
        INSERT INTO national_outbox (event_id, run_id, topic, bank_key, schema_version, payload)
        VALUES ($1, 'run-1', 'hold.requests', 'run-1:1', 'national-1', '{"hold": 100}'::jsonb)
        """,
        event_id,
    )

    with pytest.raises(asyncpg.UniqueViolationError):
        await connection.execute(
            """
            INSERT INTO national_outbox (event_id, run_id, topic, bank_key, schema_version, payload)
            VALUES ($1, 'run-1', 'hold.requests', 'run-1:1', 'national-1', '{"hold": 200}'::jsonb)
            """,
            event_id,
        )


async def test_national_outbox_unprocessed_index_exists(isolated_schema) -> None:
    """The partial index on unprocessed rows exists."""
    connection, schema_name = isolated_schema

    index_name = await connection.fetchval(
        """
        SELECT indexname FROM pg_indexes
        WHERE schemaname = $1 AND tablename = 'national_outbox' AND indexname = 'idx_national_outbox_unprocessed'
        """,
        schema_name,
    )
    assert index_name == "idx_national_outbox_unprocessed"
