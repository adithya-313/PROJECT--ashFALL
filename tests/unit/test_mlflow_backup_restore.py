# File: tests/unit/test_mlflow_backup_restore.py
# Purpose: Proves the MLflow backup and restore scripts keep every file intact, fix the paths stored in the database,
#          catch tampering, refuse to restore into a non-empty folder, and never overwrite an earlier backup.
#          Everything is built from tiny fake files in a temporary folder; no real MLflow data or package is needed.
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import backup_mlflow, restore_mlflow

OLD_PREFIX = "file:C:/fake/repo/mlruns"


def make_fake_mlflow(root: Path) -> tuple[Path, Path]:
    """Create a tiny database and a tiny mlruns folder that look like the real thing."""
    db_path = root / "mlflow.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE experiments (id INTEGER, artifact_location TEXT)")
    connection.execute("CREATE TABLE runs (id INTEGER, artifact_uri VARCHAR(200), note TEXT)")
    connection.execute("INSERT INTO experiments VALUES (1, ?)", (OLD_PREFIX + "/1",))
    connection.execute("INSERT INTO runs VALUES (1, ?, 'hello')", (OLD_PREFIX + "/1/abc/artifacts",))
    connection.commit()
    connection.close()
    runs_dir = root / "mlruns"
    (runs_dir / "1" / "abc").mkdir(parents=True)
    (runs_dir / "1" / "abc" / "metric.txt").write_text("0.99", encoding="utf-8")
    (runs_dir / "1" / "meta.yaml").write_text("name: demo", encoding="utf-8")
    return db_path, runs_dir


def make_backup_in(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Make the fake data and one backup of it. Returns (db, runs folder, backup folder)."""
    source = tmp_path / "source"
    source.mkdir()
    db_path, runs_dir = make_fake_mlflow(source)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    return db_path, runs_dir, backup_mlflow.make_backup(db_path, runs_dir, backup_dir)


def test_backup_then_restore_keeps_every_file_identical(tmp_path: Path):
    """Every mlruns file in the restored copy has the same SHA-256 as the original."""
    _, runs_dir, backup_folder = make_backup_in(tmp_path)
    target = tmp_path / "restored"

    restore_mlflow.restore(backup_folder, target, None)

    for original in runs_dir.rglob("*"):
        if original.is_file():
            restored = target / "mlruns" / original.relative_to(runs_dir)
            assert backup_mlflow.sha256_of(restored) == backup_mlflow.sha256_of(original)


def test_restore_replaces_the_old_prefix_in_the_database(tmp_path: Path):
    """After restoring, the old prefix is gone from the database and the new location is present."""
    _, _, backup_folder = make_backup_in(tmp_path)
    target = tmp_path / "restored"

    restore_mlflow.restore(backup_folder, target, None)

    connection = sqlite3.connect(target / "mlflow.db")
    try:
        assert restore_mlflow.count_prefix(connection, OLD_PREFIX) == {}
        new_prefix = restore_mlflow.make_new_prefix(target / "mlruns")
        uri = connection.execute("SELECT artifact_uri FROM runs").fetchone()[0]
    finally:
        connection.close()
    assert uri.startswith(new_prefix)


def test_a_tampered_backup_file_stops_the_restore(tmp_path: Path):
    """If one backed-up file is changed, the restore refuses and names the file."""
    _, _, backup_folder = make_backup_in(tmp_path)
    (backup_folder / "mlruns" / "1" / "meta.yaml").write_text("name: tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="meta.yaml"):
        restore_mlflow.restore(backup_folder, tmp_path / "restored", None)
    assert not (tmp_path / "restored").exists()


def test_restoring_into_a_non_empty_folder_is_refused(tmp_path: Path):
    """A target folder that already has a file in it is never used."""
    _, _, backup_folder = make_backup_in(tmp_path)
    target = tmp_path / "restored"
    target.mkdir()
    (target / "keep.txt").write_text("do not touch", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        restore_mlflow.restore(backup_folder, target, None)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "do not touch"


def test_a_second_backup_never_overwrites_the_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two backups in the same second collide on the folder name; the second stops and the first stays intact."""
    class FixedClock:
        """A clock that always says the same time, to force the same folder name."""

        @staticmethod
        def now(tz: object = None) -> datetime:
            return datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    source = tmp_path / "source"
    source.mkdir()
    db_path, runs_dir = make_fake_mlflow(source)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(backup_mlflow, "datetime", FixedClock)

    first_backup = backup_mlflow.make_backup(db_path, runs_dir, backup_dir)
    manifest_before = (first_backup / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        backup_mlflow.make_backup(db_path, runs_dir, backup_dir)

    assert (first_backup / "manifest.json").read_bytes() == manifest_before
