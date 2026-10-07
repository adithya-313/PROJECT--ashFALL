# File: tests/integration/test_idempotent_consumer.py
# Purpose: Proves api/services/idempotent_consumer.py handle_once really gives "twice = once" and rolls back on error.
#
# Where the database comes from: the REAL local Postgres from docker compose (migrations 0001-0003 must be applied).
# The test connects with the environment variable TEST_DATABASE_URL if set, otherwise with the app's DATABASE_URL
# with the Docker host name "postgres" swapped for "localhost" (the compose file publishes port 5432). If Postgres or
# the tables are not reachable, the tests are SKIPPED, not failed.
#
# Clean-up: every test uses its own unique consumer_name ("test-<random>") and deletes ONLY the rows with that name
# afterwards, so it never touches data belonging to other consumers.
import asyncio
import os
import uuid

import asyncpg
import pytest

from api.services.idempotent_consumer import handle_once
from core.config import config


def database_url_for_tests() -> str:
    """The database address the tests connect to (see the header comment)."""
    return os.environ.get("TEST_DATABASE_URL") or config.database_url.replace("@postgres:", "@localhost:")


async def open_connection() -> asyncpg.Connection:
    """Connect, or skip the test if the database or the tables are not there."""
    try:
        conn = await asyncpg.connect(dsn=database_url_for_tests(), timeout=5)
    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as error:
        pytest.skip(f"local Postgres not reachable: {error}")
    try:
        await conn.fetchval("SELECT count(*) FROM processed_events")
        await conn.fetchval("SELECT count(*) FROM demo_event_counts")
    except asyncpg.UndefinedTableError:
        await conn.close()
        pytest.skip("migration 0003 has not been applied (run scripts/run_migrations.py)")
    return conn


async def counter_value(conn: asyncpg.Connection, consumer_name: str) -> int:
    """Read the counter for this test's consumer (0 if it has no row yet)."""
    return await conn.fetchval(
        "SELECT coalesce(max(events_counted), 0) FROM demo_event_counts WHERE consumer_name = $1", consumer_name)


def make_counting_handler(consumer_name: str):
    """A handler that adds 1 to this test's counter, using the connection handle_once passes in."""
    async def handler(conn: asyncpg.Connection) -> None:
        await conn.execute(
            """
            INSERT INTO demo_event_counts (consumer_name, events_counted) VALUES ($1, 1)
            ON CONFLICT (consumer_name) DO UPDATE SET events_counted = demo_event_counts.events_counted + 1
            """,
            consumer_name,
        )
    return handler


async def delete_test_rows(conn: asyncpg.Connection, consumer_name: str) -> None:
    """Remove only the rows this test created."""
    await conn.execute("DELETE FROM processed_events WHERE consumer_name = $1", consumer_name)
    await conn.execute("DELETE FROM demo_event_counts WHERE consumer_name = $1", consumer_name)


def test_same_event_twice_runs_the_handler_once():
    async def scenario():
        conn = await open_connection()
        consumer_name, event_id = f"test-{uuid.uuid4()}", uuid.uuid4()
        handler = make_counting_handler(consumer_name)
        try:
            first = await handle_once(conn, consumer_name, event_id, handler)
            second = await handle_once(conn, consumer_name, event_id, handler)
            return first, second, await counter_value(conn, consumer_name)
        finally:
            await delete_test_rows(conn, consumer_name)
            await conn.close()

    first, second, count = asyncio.run(scenario())
    assert first is True  # the first delivery ran the handler
    assert second is False  # the duplicate was skipped
    assert count == 1  # and the counter went up by exactly one


def test_two_consumers_each_handle_the_same_event_once():
    async def scenario():
        conn = await open_connection()
        name_a, name_b, event_id = f"test-{uuid.uuid4()}", f"test-{uuid.uuid4()}", uuid.uuid4()
        try:
            a = await handle_once(conn, name_a, event_id, make_counting_handler(name_a))
            b = await handle_once(conn, name_b, event_id, make_counting_handler(name_b))
            return a, b
        finally:
            for name in (name_a, name_b):
                await delete_test_rows(conn, name)
            await conn.close()

    assert asyncio.run(scenario()) == (True, True)  # "once" means once PER consumer


def test_a_failing_handler_rolls_back_and_the_event_can_be_retried():
    async def scenario():
        conn = await open_connection()
        consumer_name, event_id = f"test-{uuid.uuid4()}", uuid.uuid4()
        counting_handler = make_counting_handler(consumer_name)

        async def failing_handler(conn_in_transaction: asyncpg.Connection) -> None:
            await counting_handler(conn_in_transaction)  # does real work first...
            raise RuntimeError("handler blew up")  # ...then fails, so everything must be undone

        try:
            with pytest.raises(RuntimeError):
                await handle_once(conn, consumer_name, event_id, failing_handler)
            marked_after_failure = await conn.fetchval(
                "SELECT count(*) FROM processed_events WHERE consumer_name = $1", consumer_name)
            count_after_failure = await counter_value(conn, consumer_name)
            retry = await handle_once(conn, consumer_name, event_id, counting_handler)
            return marked_after_failure, count_after_failure, retry, await counter_value(conn, consumer_name)
        finally:
            await delete_test_rows(conn, consumer_name)
            await conn.close()

    marked, count_after_failure, retry, final_count = asyncio.run(scenario())
    assert marked == 0  # the failed attempt left no "handled" note
    assert count_after_failure == 0  # and the handler's own write was rolled back too
    assert retry is True  # so the retry ran the handler
    assert final_count == 1
