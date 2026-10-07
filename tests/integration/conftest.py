# File: tests/integration/conftest.py
# Purpose: Shared fixtures for the database integration tests. The main one, isolated_schema, gives each test its own
#          throwaway Postgres schema with freshly migrated tables, so tests never touch real data and never see each
#          other's rows.
#
# Where the database comes from: TEST_DATABASE_URL if set, otherwise the app's DATABASE_URL with the Docker host
# name "postgres" swapped for "localhost" (same rule as test_idempotent_consumer.py). If Postgres is not reachable,
# the tests are SKIPPED, not failed.
import asyncio
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import asyncpg
import pytest

# Make the repo root importable so "scripts.run_migrations" can be found.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import config  # noqa: E402
from scripts.run_migrations import apply_pending_migrations  # noqa: E402


def get_test_database_url() -> str:
    """The database address the tests connect to (see the header comment)."""
    return os.environ.get("TEST_DATABASE_URL") or config.database_url.replace("@postgres:", "@localhost:")


def describe_host(database_url: str) -> str:
    """Host and port only, so skip messages never show a password."""
    parsed = urlparse(database_url)
    return f"{parsed.hostname}:{parsed.port}"


def make_schema_name() -> str:
    """
    Build a unique schema name such as test_3f9a1c0b7d2e.

    This name comes ONLY from our own code (a fixed prefix plus random hex), never from user input or the
    environment. That is why it is safe to put it straight into SQL text below: SQL cannot take a schema name as a
    $1 parameter, and nothing outside our control can reach this string.
    """
    return "test_" + uuid.uuid4().hex[:12]


async def connect_or_skip(database_url: str, **connect_options) -> asyncpg.Connection:
    """Open a connection, or skip the whole test if Postgres is not reachable."""
    try:
        return await asyncpg.connect(dsn=database_url, timeout=5, **connect_options)
    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as error:
        pytest.skip(f"Postgres not reachable at {describe_host(database_url)}: {error}")


async def create_schema(database_url: str, schema_name: str) -> None:
    """Create the empty throwaway schema using a short-lived connection."""
    admin_connection = await connect_or_skip(database_url)
    try:
        await admin_connection.execute(f"CREATE SCHEMA {schema_name}")
    finally:
        await admin_connection.close()


async def drop_schema(database_url: str, schema_name: str) -> None:
    """Delete the throwaway schema and everything in it, using a fresh connection (the test's may be broken)."""
    admin_connection = await connect_or_skip(database_url)
    try:
        await admin_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    finally:
        await admin_connection.close()


@pytest.fixture
async def empty_schema():
    """
    Yield (connection, schema_name) for a brand-new schema with NO tables yet.

    The connection's search_path points at the new schema, so unqualified table names (like "transactions") land
    inside it and never touch the real "public" tables.
    """
    database_url = get_test_database_url()
    schema_name = make_schema_name()
    await create_schema(database_url, schema_name)
    connection = None
    try:
        connection = await connect_or_skip(database_url, server_settings={"search_path": schema_name})
        yield connection, schema_name
    finally:
        # Always clean up, even if the test failed or the connection could not be opened.
        if connection is not None:
            await connection.close()
        await drop_schema(database_url, schema_name)


@pytest.fixture
async def isolated_schema(empty_schema):
    """Yield (connection, schema_name) for a throwaway schema that already has all migrations applied."""
    connection, schema_name = empty_schema
    await apply_pending_migrations(connection)
    yield connection, schema_name
