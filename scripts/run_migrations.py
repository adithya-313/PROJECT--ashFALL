# File: scripts/run_migrations.py
# Purpose: A tiny, safe-to-repeat migration runner. It applies every .sql file in migrations/ to the database, in
#          filename order, and remembers which files it has already applied so running it again does nothing.
#
# Run from the repo root:   PYTHONPATH=. python scripts/run_migrations.py
#
# Which database: it uses the SAME setting the API uses (config.database_url from core/config.py, which reads
# DATABASE_URL from the environment or from .env). Inside Docker that URL points at the host name "postgres". From
# your own machine that name does not exist, so override it for one run, for example:
#     DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/ashfall PYTHONPATH=. python scripts/run_migrations.py
import asyncio
import sys
from pathlib import Path

import asyncpg

# Make the repo root importable, so "from core.config import config" works when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import config  # noqa: E402  (the same config object api/main.py uses)

MIGRATIONS_DIR = REPO_ROOT / "migrations"


def find_migration_files(migrations_dir: Path = MIGRATIONS_DIR) -> list[Path]:
    """List every .sql file in the migrations folder, sorted by name. The 0001_, 0002_ prefixes make this the right order."""
    return sorted(migrations_dir.glob("*.sql"))


async def ensure_tracking_table(connection: asyncpg.Connection) -> None:
    """Create the notebook where we write down which migration files have already been applied."""
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          filename   TEXT PRIMARY KEY,
          applied_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )


async def get_applied_filenames(connection: asyncpg.Connection) -> set[str]:
    """Read the notebook: the names of the migration files that were already applied."""
    rows = await connection.fetch("SELECT filename FROM schema_migrations")
    return {row["filename"] for row in rows}


async def apply_one_file(connection: asyncpg.Connection, path: Path) -> None:
    """
    Run one migration file and record it, all inside ONE database transaction.

    Why a transaction: if any statement in the file fails, the database undoes everything the file did, and the
    file is NOT recorded as applied. The record is written inside the same transaction, so "the tables exist" and
    "the file is marked applied" always happen together or not at all.
    """
    sql_text = path.read_text(encoding="utf-8")
    async with connection.transaction():
        await connection.execute(sql_text)
        await connection.execute("INSERT INTO schema_migrations (filename) VALUES ($1)", path.name)


async def apply_pending_migrations(connection: asyncpg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> int:
    """
    Apply every migration file that has not been applied yet, using a connection the caller already opened.
    Returns how many files were applied.

    Taking the connection as a parameter (instead of opening one here) lets tests point the runner at any database
    or schema, for example a throwaway one.
    """
    await ensure_tracking_table(connection)
    already_applied = await get_applied_filenames(connection)

    applied_count = 0
    for path in find_migration_files(migrations_dir):
        if path.name in already_applied:
            print(f"skip   {path.name} (already applied)")
            continue
        await apply_one_file(connection, path)
        print(f"apply  {path.name}")
        applied_count += 1
    return applied_count


async def run_migrations() -> int:
    """Connect to the configured database and apply every pending migration. Returns how many were applied."""
    connection = await asyncpg.connect(dsn=config.database_url)
    try:
        return await apply_pending_migrations(connection)
    finally:
        await connection.close()


def main() -> int:
    try:
        applied_count = asyncio.run(run_migrations())
    except Exception as error:  # noqa: BLE001 - show a clear message and a non-zero exit code to the caller
        print(f"Migration failed: {type(error).__name__}: {error}")
        return 1
    print(f"Done. {applied_count} migration file(s) applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
