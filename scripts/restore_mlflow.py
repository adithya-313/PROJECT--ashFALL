# File: scripts/restore_mlflow.py
# Purpose: Restore a backup made by scripts/backup_mlflow.py into a new, empty folder, after checking that no file in
#          the backup was damaged, and fix the folder paths stored inside the database.
#
# Run:
#     python scripts/restore_mlflow.py --backup-folder <one mlflow-backup-... folder> --target-dir <new empty folder>
#
# Why a path fix is needed: MLflow stores ABSOLUTE paths such as file:C:/Users/me/repo/mlruns/1/<run>/artifacts in its
# database. If the data now lives somewhere else, those paths point to the wrong place. After copying, this script
# replaces the old prefix (recorded in manifest.json) with the new mlruns location, in the RESTORED COPY ONLY.
import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

import orjson

# Make the repo root importable, so "from scripts.backup_mlflow import ..." works when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.backup_mlflow import sha256_of  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description="Restore an MLflow backup and fix the paths inside the database.")
    parser.add_argument("--backup-folder", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument(
        "--new-runs-path", type=Path, default=None, help="where mlruns goes (default: <target-dir>/mlruns)"
    )
    return parser.parse_args(argv)


def load_manifest(backup_folder: Path) -> dict:
    """Read manifest.json, or raise ValueError with a plain message if it is missing."""
    manifest_path = backup_folder / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"manifest.json is missing in {backup_folder}, so the backup cannot be checked.")
    return orjson.loads(manifest_path.read_bytes())


def find_bad_files(backup_folder: Path, manifest: dict) -> list[str]:
    """Compare every file with the manifest and list the ones that are missing or have a different SHA-256."""
    bad_files = []
    for entry in manifest["files"]:
        path = backup_folder / entry["path"]
        if not path.is_file() or sha256_of(path) != entry["sha256"]:
            bad_files.append(entry["path"])
    return bad_files


def check_target(target_dir: Path) -> None:
    """Refuse to restore into a folder that already has something in it."""
    if target_dir.exists() and any(target_dir.iterdir()):
        raise ValueError(f"The target folder {target_dir} is not empty. Use a new, empty folder.")


def make_new_prefix(new_runs_path: Path) -> str:
    """The new prefix, in the same style as the old one (file:C:/...)."""
    return "file:" + new_runs_path.resolve().as_posix()


def text_columns(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    """Find every (table, column) that holds text, by asking SQLite instead of hard-coding table names."""
    found = []
    tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    for (table,) in tables:
        for column in connection.execute(f'PRAGMA table_info("{table}")').fetchall():
            declared_type = (column[2] or "").upper()
            if "CHAR" in declared_type or "TEXT" in declared_type or "CLOB" in declared_type:
                found.append((table, column[1]))
    return found


def count_prefix(connection: sqlite3.Connection, prefix: str) -> dict[tuple[str, str], int]:
    """For every text column: how many rows contain the prefix (only columns with at least one are returned)."""
    counts = {}
    for table, column in text_columns(connection):
        sql = f'SELECT count(*) FROM "{table}" WHERE instr("{column}", ?) > 0'
        number = connection.execute(sql, (prefix,)).fetchone()[0]
        if number:
            counts[(table, column)] = number
    return counts


def replace_prefix(db_path: Path, old_prefix: str, new_prefix: str) -> dict[tuple[str, str], int]:
    """Replace the old prefix everywhere in one transaction, then require that none is left."""
    connection = sqlite3.connect(db_path)
    try:
        before = count_prefix(connection, old_prefix)
        with connection:  # one transaction: if anything fails, nothing is changed
            for table, column in before:
                sql = f'UPDATE "{table}" SET "{column}" = replace("{column}", ?, ?) WHERE instr("{column}", ?) > 0'
                connection.execute(sql, (old_prefix, new_prefix, old_prefix))
            if count_prefix(connection, old_prefix):
                raise RuntimeError("Some old paths are still in the database after the fix. Nothing was changed.")
        return before
    finally:
        connection.close()


def restore(backup_folder: Path, target_dir: Path, new_runs_path: Path | None) -> dict[tuple[str, str], int]:
    """Check, copy and fix. Returns the per-column counts of rewritten rows."""
    manifest = load_manifest(backup_folder)
    bad_files = find_bad_files(backup_folder, manifest)
    if bad_files:
        raise ValueError("These files do not match the manifest: " + ", ".join(bad_files))
    check_target(target_dir)
    runs_path = new_runs_path or target_dir / "mlruns"
    old_prefix = manifest["old_runs_prefix"]
    if not old_prefix:
        raise ValueError("The manifest has no old runs prefix, so the paths cannot be fixed.")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_folder / "mlflow.db", target_dir / "mlflow.db")
    shutil.copytree(backup_folder / "mlruns", runs_path)
    return replace_prefix(target_dir / "mlflow.db", old_prefix, make_new_prefix(runs_path))


def main(argv: list[str] | None = None) -> int:
    """Run the restore and print a short summary."""
    args = parse_args(argv)
    try:
        counts = restore(args.backup_folder, args.target_dir, args.new_runs_path)
    except (ValueError, RuntimeError) as error:
        print(f"Restore stopped: {error}")
        return 1
    print("Manifest verified: every file matches its SHA-256.")
    for (table, column), number in sorted(counts.items()):
        print(f"  rewrote {number} row(s) in {table}.{column}")
    print("Old prefix count after the fix: 0")
    print(f"Restored into {args.target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
