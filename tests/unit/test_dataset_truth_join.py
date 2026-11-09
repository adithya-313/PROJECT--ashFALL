"""
WHAT: Unit tests for evaluation/dataset_truth_join.py using a tiny hand-made dataset and truth folder.
Covers the half-open span rule, the clock_start minute rule, refusal cases, and that the join never
changes original columns or the dataset folder.

WHY: The truth join decides what counts as a mule for every scored row; an off-by-one minute or a
wrong clock start would silently corrupt evaluation.

TERM: half-open span: active from valid_from (included) up to valid_to (excluded).
TERM: world minute: whole minutes since IST midnight of clock_start.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.config_validator import load_yaml
from core.event_clock import IST
from evaluation.dataset_truth_join import join_truth
from world.run_identity import REPO_ROOT

RUN_ID = "run_x"
WORLD_ID = "world_x"
START = datetime(2026, 9, 1, 0, 0, tzinfo=IST)
SPAN_FROM = 600
SPAN_TO = 660
MULE = "acc_mule"


def _at(minute: int) -> datetime:
    """WHAT: UTC time of a world minute for clock_start 2026-09-01. WHY: builds row times. Example: _at(0)."""
    return (START + timedelta(minutes=minute)).astimezone(UTC)


def _write_truth(truth_root: Path) -> None:
    """WHAT: Writes a valid truth folder with one mule span. WHY: load_truth checks digests. Example: x."""
    run_dir = truth_root / RUN_ID
    run_dir.mkdir(parents=True)
    roles = [
        {
            "account_id": MULE, "active_from": SPAN_FROM, "burned_from": None, "depth": 1,
            "gang_id": "gang_1", "origin": "recruited", "role": "mule",
            "valid_from": SPAN_FROM, "valid_to": SPAN_TO,
        }
    ]
    files: dict[str, dict[str, object]] = {}
    for name, rows in (
        ("roles.jsonl", roles), ("device_control.jsonl", []),
        ("payment_stories.jsonl", []), ("gangs.jsonl", []),
    ):
        text = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
        (run_dir / name).write_text(text, encoding="utf-8", newline="\n")
        files[name] = {"row_count": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    fmt = load_yaml(REPO_ROOT / "config" / "truth_store.yaml")["truth_store"]["format"]
    manifest = {"code_version": "t", "files": files, "format": fmt, "run_id": RUN_ID}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_dataset(
    root: Path, clock_start: str = "2026-09-01", minutes: tuple[int, ...] = (599, 600, 659, 660)
) -> Path:
    """WHAT: Writes a tiny dataset folder. WHY: join_truth reads a built dataset. Example: _write_dataset(p)."""
    ds = root / "dataset"
    ds.mkdir()
    n = len(minutes)
    payments = pa.table(
        {
            "world_id": [WORLD_ID] * n,
            "payment_id": [f"p{i}" for i in range(n)],
            "receiver_account_id": [MULE] * n,
            "decision_time": pa.array([_at(m) for m in minutes], type=pa.timestamp("us", tz="UTC")),
            "amount_feature": [1.5] * n,
        }
    )
    snapshots = pa.table(
        {
            "world_id": [WORLD_ID] * n,
            "account_id": [MULE] * n,
            "snapshot_time": pa.array([_at(m) for m in minutes], type=pa.timestamp("us", tz="UTC")),
        }
    )
    pq.write_table(payments, ds / "payments.parquet")
    pq.write_table(snapshots, ds / "snapshots.parquet")
    manifest = {"worlds": [{"world_id": WORLD_ID, "run_id": RUN_ID, "clock_start": clock_start}]}
    (ds / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds


def _hashes(folder: Path) -> dict[str, str]:
    """WHAT: SHA-256 of every file in a folder. WHY: proves the folder is untouched. Example: _hashes(p)."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path]:
    """WHAT: Dataset dir and truth root. WHY: shared by tests. Example: ds, truth = setup."""
    truth_root = tmp_path / "truth"
    _write_truth(truth_root)
    return _write_dataset(tmp_path), truth_root


def test_roles_follow_half_open_span(setup: tuple[Path, Path]) -> None:
    """WHAT: mule inside span; genuine one minute before and exactly at valid_to. WHY: half-open rule."""
    ds, truth = setup
    payments, snapshots = join_truth(ds, truth)
    assert payments.column("truth_role_receiver").to_pylist() == ["genuine", "mule", "mule", "genuine"]
    assert snapshots.column("truth_role").to_pylist() == ["genuine", "mule", "mule", "genuine"]
    assert payments.column("truth_owner").to_pylist() == [None, "gang_1", "gang_1", None]
    assert payments.column("truth_depth").to_pylist() == [None, 1, 1, None]


def test_minute_uses_clock_start(tmp_path: Path) -> None:
    """WHAT: shifting clock_start one day earlier changes the result. WHY: minute is relative to clock_start."""
    truth_root = tmp_path / "truth"
    _write_truth(truth_root)
    ds = _write_dataset(tmp_path, minutes=(600,))
    assert join_truth(ds, truth_root)[0].column("truth_role_receiver").to_pylist() == ["mule"]
    manifest = json.loads((ds / "manifest.json").read_text())
    manifest["worlds"][0]["clock_start"] = "2026-08-31"
    (ds / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert join_truth(ds, truth_root)[0].column("truth_role_receiver").to_pylist() == ["genuine"]


def test_original_columns_and_folder_unchanged(setup: tuple[Path, Path]) -> None:
    """WHAT: original columns equal, folder files and hashes unchanged. WHY: join is read-only."""
    ds, truth = setup
    before_files = _hashes(ds)
    orig_p = pq.read_table(ds / "payments.parquet")
    orig_s = pq.read_table(ds / "snapshots.parquet")
    payments, snapshots = join_truth(ds, truth)
    assert payments.select(orig_p.column_names).equals(orig_p)
    assert snapshots.select(orig_s.column_names).equals(orig_s)
    assert _hashes(ds) == before_files


def test_refuses_bad_manifest_or_early_row(tmp_path: Path) -> None:
    """WHAT: ValueError for missing clock_start/run_id or a row before clock_start. WHY: refuse wrong joins."""
    truth_root = tmp_path / "truth"
    _write_truth(truth_root)
    ds = _write_dataset(tmp_path, minutes=(0,))
    base = json.loads((ds / "manifest.json").read_text())
    for key in ("clock_start", "run_id"):
        bad = json.loads(json.dumps(base))
        del bad["worlds"][0][key]
        (ds / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            join_truth(ds, truth_root)
    late = json.loads(json.dumps(base))
    late["worlds"][0]["clock_start"] = "2026-09-02"
    (ds / "manifest.json").write_text(json.dumps(late), encoding="utf-8")
    with pytest.raises(ValueError, match="before clock_start"):
        join_truth(ds, truth_root)
