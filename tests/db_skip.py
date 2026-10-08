# File: tests/db_skip.py
# Purpose: One shared helper for database tests that run a coroutine and SKIP only when the database is truly
#          unreachable. Any other error (for example a missing table) must make the test FAIL with its real message,
#          because hiding it as a "skip" would let a broken setup pass quietly.
import asyncio
import os
from collections.abc import Awaitable
from typing import TypeVar

import asyncpg
import pytest

from core.config import config

T = TypeVar("T")

# Only genuine "could not connect" problems:
#   OSError                         - refused connection, unknown host, network down (includes ConnectionError)
#   asyncio.TimeoutError            - the server did not answer in time
#   asyncpg.CannotConnectNowError   - the server is starting up or shutting down
#   asyncpg.PostgresConnectionError - the connection broke or was rejected (a family of connection-level errors)
# Deliberately NOT here: asyncpg.PostgresError as a whole, because it also covers real bugs such as
# UndefinedTableError (a missing table), which must fail the test.
CONNECTION_ERRORS = (
    OSError,
    asyncio.TimeoutError,
    asyncpg.CannotConnectNowError,
    asyncpg.PostgresConnectionError,
)


def run_or_skip(coroutine: Awaitable[T]) -> T:
    """Run a coroutine; skip the test if the database cannot be reached, but let every other error propagate."""
    try:
        return asyncio.run(coroutine)  # type: ignore[arg-type]
    except CONNECTION_ERRORS as error:
        pytest.skip(f"Postgres could not be reached at the showcase database address ({type(error).__name__})")


def host_side_database_url() -> str:
    """
    The database address to use from the host machine (a laptop or a CI runner).

    TEST_DATABASE_URL wins if set. Otherwise we take DATABASE_URL and swap the Docker-only host name "postgres" for
    "localhost", because docker compose publishes Postgres on localhost:5432.
    """
    return os.environ.get("TEST_DATABASE_URL") or config.database_url.replace("@postgres:", "@localhost:")
