# File: tests/unit/test_db_skip.py
# Purpose: Proves run_or_skip skips only for connection-type errors and re-raises every other error. No database is
#          needed: the coroutines below simply raise the errors.
import asyncpg
import pytest

from tests.db_skip import run_or_skip


async def raise_error(error: Exception) -> None:
    """A coroutine that fails with the given error."""
    raise error


async def return_value() -> int:
    """A coroutine that works."""
    return 42


def test_connection_errors_cause_a_skip():
    """Refused connections, timeouts and server-starting-up errors are turned into a skip."""
    for error in (ConnectionRefusedError("refused"), TimeoutError("slow"), asyncpg.CannotConnectNowError("starting")):
        with pytest.raises(pytest.skip.Exception):
            run_or_skip(raise_error(error))


def test_missing_table_error_is_not_hidden_as_a_skip():
    """An UndefinedTableError is a real failure, so it is re-raised with its own message."""
    with pytest.raises(asyncpg.UndefinedTableError, match="no such table"):
        run_or_skip(raise_error(asyncpg.UndefinedTableError("no such table")))


def test_normal_result_is_returned():
    """When nothing goes wrong, the coroutine's result is returned."""
    assert run_or_skip(return_value()) == 42
