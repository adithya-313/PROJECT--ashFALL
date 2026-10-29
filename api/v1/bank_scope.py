"""
File: api/v1/bank_scope.py
Purpose: Scopes database operations to a specific bank using PostgreSQL Row Level Security (RLS).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg

from core.config_validator import load_yaml

BANK_ROLE: str = "ashfall_bank_api"
DEFAULT_CONFIG_PATH: str = "config/bank_api.yaml"


class UnknownBankError(ValueError):
    """Raised when an operation is attempted with an unrecognized bank identifier."""


def load_known_banks(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """
    WHAT: Reads the list of authorized bank identifiers from bank API configuration.

    WHY: Validates incoming bank identifiers against the ground truth configuration before any
    database calls, rejecting unauthorized banks fail-closed.

    Args:
        config_path: Path to bank API YAML configuration file.

    Returns:
        List of known bank identifiers (e.g. ['bank_0', 'bank_1', 'bank_2']).

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    data = load_yaml(config_path)
    bank_list = data.get("bank_api", {}).get("banks", [])
    return list(bank_list)


@asynccontextmanager
async def bank_scoped_transaction(
    conn: asyncpg.Connection,
    bank_id: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> AsyncIterator[asyncpg.Connection]:
    """
    WHAT: Opens an isolated database transaction scoped to the specified commercial bank.

    WHY: Enforces multi-tenant data isolation at the PostgreSQL database engine layer. By assuming the
    least-privileged role 'ashfall_bank_api' and injecting 'app.bank_id', row-level security (RLS)
    policies transparently filter all queries and validate all inserts against the authenticated bank.

    HOW POOLED CONNECTIONS REMAIN SAFE:
    Both `SET LOCAL ROLE` and `set_config('app.bank_id', $1, true)` operate with transaction-local scope
    (the third argument `true` specifies `is_local=true`). In PostgreSQL, transaction-local settings and
    roles are strictly bound to the lifetime of the enclosing transaction block.
    When this context manager commits or rolls back, PostgreSQL automatically resets the session user
    back to the connection's authenticated superuser/owner and reverts 'app.bank_id' to its default unset
    state (NULL or ''). Consequently, when a connection is returned to an asyncpg connection pool, no
    lingering bank identity or reduced privileges leak to subsequent callers sharing the pooled connection.

    Args:
        conn: Open asyncpg database connection.
        bank_id: Commercial bank identifier (e.g. 'bank_0').
        config_path: Optional override path for bank_api.yaml.

    Yields:
        The database connection within the active bank-scoped transaction.

    Raises:
        UnknownBankError: If bank_id is not in the list of known banks.
    """
    known_banks = load_known_banks(config_path)
    if bank_id not in known_banks:
        raise UnknownBankError(
            f"Unknown bank_id '{bank_id}'. Expected one of {known_banks}"
        )

    async with conn.transaction():
        # Drop superuser privileges to the restricted bank role for this transaction only
        await conn.execute(f"SET LOCAL ROLE {BANK_ROLE}")
        # Inject the active bank identity into session settings for RLS policy evaluation
        await conn.execute("SELECT set_config('app.bank_id', $1, true)", bank_id)
        yield conn
