# File: showcase/isolation.py
# Purpose: Gives a showcase its own private playground: a throwaway Postgres schema (with freshly migrated tables),
#          a temp folder, and a fixed seed. Nothing a showcase does can touch real data, and everything is removed
#          afterwards, even when a check fails.
import asyncio
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import asyncpg

# Make the repo root importable, so "scripts.run_migrations" and "core.config" are found (same trick the tests use).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import config  # noqa: E402
from scripts.run_migrations import apply_pending_migrations  # noqa: E402
from showcase.framework import FIXED_SEED  # noqa: E402

# Every showcase schema starts with this. Cleanup refuses to drop anything that does not.
SCHEMA_PREFIX = "showcase_"
CONNECT_TIMEOUT_SECONDS = 5


@dataclass
class Workspace:
    """Everything a showcase check may use."""

    dsn: str
    schema_name: str
    connection: asyncpg.Connection
    temp_dir: Path
    seed: int


def get_showcase_database_url() -> str:
    """
    Same rule as tests/integration/conftest.py: TEST_DATABASE_URL if set, else DATABASE_URL pointing at localhost.

    This is a deliberate small duplicate. Production-side code (showcase/) must not import from tests/, and the rule
    is only three lines, so copying it is safer than creating that dependency.
    """
    return os.environ.get("TEST_DATABASE_URL") or config.database_url.replace("@postgres:", "@localhost:")


def make_schema_name() -> str:
    """
    Build a unique schema name such as showcase_3f9a1c0b7d2e.

    The name comes ONLY from our own code (a fixed prefix plus random hex), never from user input or the
    environment. That is why it is safe to put it straight into SQL text: SQL cannot take a schema name as a $1
    parameter, and nothing outside our control can reach this string.
    """
    return SCHEMA_PREFIX + uuid.uuid4().hex[:12]


def check_schema_name_is_ours(schema_name: str) -> None:
    """
    Raise ValueError unless the name starts with 'showcase_'.

    This protects real data: even if a bug passed in "public" or a real schema name, cleanup would refuse to drop it.
    """
    if not schema_name.startswith(SCHEMA_PREFIX):
        raise ValueError(f"refusing to touch schema '{schema_name}': it does not start with '{SCHEMA_PREFIX}'")


async def create_schema(dsn: str, schema_name: str) -> None:
    """Create the empty schema using a short-lived plain connection."""
    check_schema_name_is_ours(schema_name)
    admin_connection = await asyncpg.connect(dsn=dsn, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        await admin_connection.execute(f"CREATE SCHEMA {schema_name}")
    finally:
        await admin_connection.close()


async def create_workspace() -> Workspace:
    """Make the temp folder and schema, connect inside the schema, and run the migrations there."""
    dsn = get_showcase_database_url()
    schema_name = make_schema_name()
    temp_dir = Path(tempfile.mkdtemp(prefix="ashfall_showcase_"))
    try:
        await create_schema(dsn, schema_name)
        # search_path makes every unqualified table name land inside our schema, never in "public".
        connection = await asyncpg.connect(
            dsn=dsn, timeout=CONNECT_TIMEOUT_SECONDS, server_settings={"search_path": schema_name})
    except BaseException:
        # Setup failed halfway: undo whatever was already made, then report the original problem.
        await drop_schema_quietly(dsn, schema_name)
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    workspace = Workspace(dsn=dsn, schema_name=schema_name, connection=connection, temp_dir=temp_dir,
                          seed=FIXED_SEED)
    try:
        await apply_pending_migrations(connection)
    except BaseException:
        await cleanup_workspace(workspace)
        raise
    return workspace


async def drop_schema(dsn: str, schema_name: str) -> None:
    """Drop the schema and everything in it, on a fresh plain connection."""
    check_schema_name_is_ours(schema_name)
    fresh_connection = await asyncpg.connect(dsn=dsn, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        await fresh_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    finally:
        await fresh_connection.close()


async def drop_schema_quietly(dsn: str, schema_name: str) -> None:
    """Try to drop the schema and ignore errors (used while already handling another error)."""
    try:
        await drop_schema(dsn, schema_name)
    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError):
        pass


async def cleanup_workspace(workspace: Workspace) -> None:
    """
    Remove everything the workspace made. Each step is attempted even if an earlier one fails.

    The name check runs FIRST so a bad name never reaches the database.
    """
    check_schema_name_is_ours(workspace.schema_name)
    problems: list[str] = []

    # Step 1: close the workspace connection (it may already be broken, which is fine).
    try:
        await workspace.connection.close()
    except Exception as error:  # noqa: BLE001 - keep cleaning up whatever happens
        problems.append(f"close connection: {error}")

    # Step 2: drop the schema using a fresh connection.
    try:
        await drop_schema(workspace.dsn, workspace.schema_name)
    except Exception as error:  # noqa: BLE001
        problems.append(f"drop schema: {error}")

    # Step 3: delete the temp folder.
    shutil.rmtree(workspace.temp_dir, ignore_errors=True)

    if problems:
        raise RuntimeError("workspace cleanup had problems: " + "; ".join(problems))
