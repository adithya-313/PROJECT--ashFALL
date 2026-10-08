# File: scripts/backup_mlflow.py
# Purpose: Make a safe, checked copy of the MLflow data (the SQLite database mlflow.db and the folder mlruns/).
#
# Run from the repo root:
#     python scripts/backup_mlflow.py --backup-dir <existing folder to put the backup in>
#
# What it does: it makes a NEW folder inside --backup-dir named mlflow-backup-<UTC time>, copies the database with
# SQLite's own backup function (a consistent snapshot, even if something is writing), copies mlruns/, and writes
# manifest.json listing every file with its size and SHA-256 so a restore can check nothing was damaged.
# It never changes the originals and never overwrites an earlier backup.
import argparse
import hashlib
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
RESTORE_NOTE = (
    "Restore with: python scripts/restore_mlflow.py --backup-folder <this folder> --target-dir <new empty folder>"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description="Back up mlflow.db and mlruns/ with a checksum manifest.")
    parser.add_argument("--backup-dir", required=True, type=Path, help="existing folder that will hold the backup")
    parser.add_argument("--source-db", type=Path, default=REPO_ROOT / "mlflow.db")
    parser.add_argument("--source-runs", type=Path, default=REPO_ROOT / "mlruns")
    return parser.parse_args(argv)


def check_inputs(source_db: Path, source_runs: Path, backup_dir: Path) -> str | None:
    """Return a plain-English problem, or None when everything needed exists."""
    if not source_db.is_file():
        return f"The database {source_db} does not exist."
    if not source_runs.is_dir():
        return f"The runs folder {source_runs} does not exist."
    if not backup_dir.is_dir():
        return f"The backup folder {backup_dir} does not exist. Create it first."
    return None


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in chunks so big files do not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_database(source_db: Path, destination: Path) -> None:
    """Copy the database with SQLite's backup function. The source is opened read-only, the destination is new."""
    source = sqlite3.connect(f"{source_db.resolve().as_uri()}?mode=ro", uri=True)
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def find_old_prefix(db_path: Path) -> str:
    """
    Find the folder prefix the database uses for artifacts, such as file:C:/Users/me/repo/mlruns.

    It is read from the first experiment's artifact_location (cut at "/mlruns"). Empty text means none was found.
    """
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT artifact_location FROM experiments WHERE artifact_location IS NOT NULL"
        ).fetchone()
    finally:
        connection.close()
    if row is None or "/mlruns" not in row[0]:
        return ""
    return row[0].split("/mlruns")[0] + "/mlruns"


def list_files(backup_folder: Path) -> list[dict[str, object]]:
    """Every file inside the backup (except the manifest), with relative path, size and SHA-256."""
    entries = []
    for path in sorted(backup_folder.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            relative = path.relative_to(backup_folder).as_posix()
            entries.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_of(path)})
    return entries


def write_manifest(backup_folder: Path, old_prefix: str) -> list[dict[str, object]]:
    """Write manifest.json and return the file list."""
    files = list_files(backup_folder)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "old_runs_prefix": old_prefix,
        "restore_note": RESTORE_NOTE,
        "files": files,
    }
    (backup_folder / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    return files


def make_backup(source_db: Path, source_runs: Path, backup_dir: Path) -> Path:
    """Create one new backup folder and return its path. Raises FileExistsError if the folder name is taken."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_folder = backup_dir / f"mlflow-backup-{stamp}"
    backup_folder.mkdir()  # fails if it already exists, so an earlier backup is never overwritten
    copy_database(source_db, backup_folder / "mlflow.db")
    shutil.copytree(source_runs, backup_folder / "mlruns")
    write_manifest(backup_folder, find_old_prefix(source_db))
    return backup_folder


def main(argv: list[str] | None = None) -> int:
    """Run the backup and print a short summary."""
    args = parse_args(argv)
    problem = check_inputs(args.source_db, args.source_runs, args.backup_dir)
    if problem:
        print(problem)
        return 1
    try:
        backup_folder = make_backup(args.source_db, args.source_runs, args.backup_dir)
    except FileExistsError:
        print("A backup with this timestamp already exists. Wait a second and try again.")
        return 1
    files = list_files(backup_folder)
    total_bytes = sum(int(entry["size"]) for entry in files)  # type: ignore[call-overload]
    print(f"Backup written to {backup_folder}")
    print(f"{len(files)} files, {total_bytes} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
