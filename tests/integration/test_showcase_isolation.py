# File: tests/integration/test_showcase_isolation.py
# Purpose: Proves showcase/isolation.py gives a showcase its own schema and temp folder, keeps real data untouched,
#          removes everything afterwards, and refuses to drop a schema that is not its own.
#
# Real Postgres (same address rule as the other integration tests). If it is not reachable the tests are SKIPPED.
import asyncio
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from showcase.isolation import (
    SCHEMA_PREFIX,
    Workspace,
    check_schema_name_is_ours,
    cleanup_workspace,
    create_workspace,
    get_showcase_database_url,
)
from tests.db_skip import run_or_skip

EXPECTED_TABLES = {"transactions", "outbox", "processed_events", "demo_event_counts"}


async def fetch_value(query: str, *args: Any) -> Any:
    """Run one query on a fresh plain connection and return the first value."""
    connection = await asyncpg.connect(dsn=get_showcase_database_url(), timeout=5)
    try:
        return await connection.fetchval(query, *args)
    finally:
        await connection.close()


async def public_transaction_count() -> int:
    """Count rows in the REAL public.transactions table."""
    return await fetch_value("SELECT count(*) FROM public.transactions")


async def schema_exists(schema_name: str) -> bool:
    """Is there a schema with this name?"""
    count = await fetch_value("SELECT count(*) FROM information_schema.schemata WHERE schema_name = $1", schema_name)
    return count == 1


async def table_names(workspace: Workspace) -> set[str]:
    """Names of the tables inside the workspace's schema."""
    rows = await workspace.connection.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = $1", workspace.schema_name)
    return {row["table_name"] for row in rows}


def test_workspace_has_all_tables_inside_its_own_schema():
    """create_workspace builds the expected tables inside the showcase_ schema."""
    async def scenario() -> None:
        workspace = await create_workspace()
        try:
            assert workspace.schema_name.startswith(SCHEMA_PREFIX)
            assert EXPECTED_TABLES <= await table_names(workspace)
        finally:
            await cleanup_workspace(workspace)
    run_or_skip(scenario())


def test_rows_written_in_workspace_are_not_visible_in_public():
    """A row written inside the workspace never shows up in the real public.transactions table."""
    async def scenario() -> None:
        before = await public_transaction_count()
        workspace = await create_workspace()
        try:
            await workspace.connection.execute(
                "INSERT INTO transactions (source, account_id, amount) VALUES ('ecommerce', 'showcase-test', 1.00)")
            inside = await workspace.connection.fetchval("SELECT count(*) FROM transactions")
            assert inside == 1
        finally:
            await cleanup_workspace(workspace)
        assert await public_transaction_count() == before
    run_or_skip(scenario())


def test_cleanup_removes_schema_and_temp_folder():
    """cleanup_workspace drops the schema and deletes the temp folder."""
    async def scenario() -> None:
        workspace = await create_workspace()
        assert workspace.temp_dir.exists()
        assert await schema_exists(workspace.schema_name)
        await cleanup_workspace(workspace)
        assert not workspace.temp_dir.exists()
        assert not await schema_exists(workspace.schema_name)
    run_or_skip(scenario())


def test_cleanup_refuses_a_schema_name_that_is_not_ours():
    """A name that does not start with 'showcase_' (such as 'public') is refused before anything is dropped."""
    with pytest.raises(ValueError, match="refusing"):
        check_schema_name_is_ours("public")
    with pytest.raises(ValueError, match="refusing"):
        check_schema_name_is_ours("test_abc123")


def test_cleanup_workspace_itself_refuses_a_foreign_schema():
    """cleanup_workspace raises ValueError for a workspace whose schema is 'public', before touching anything."""
    fake_workspace = Workspace(dsn="unused", schema_name="public", connection=None,  # type: ignore[arg-type]
                               temp_dir=Path("unused"), seed=0)
    with pytest.raises(ValueError, match="refusing"):
        asyncio.run(cleanup_workspace(fake_workspace))
