# File: tests/db_skip.py
# Purpose: One shared helper for database tests that run a coroutine and SKIP only when the database is truly
#          unreachable. Any other error (for example a missing table) must make the test FAIL with its real message,
#          because hiding it as a "skip" would let a broken setup pass quietly.
import asyncio
from collections.abc import Awaitable
from typing import TypeVar

import asyncpg
import pytest

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
