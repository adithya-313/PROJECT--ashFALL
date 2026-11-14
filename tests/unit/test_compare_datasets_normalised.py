# File: tests/unit/test_compare_datasets_normalised.py
# Purpose: Unit tests for scripts/compare_datasets_normalised.py (Level 18a1 output-identity check).
"""
WHAT: Verifies the run-id-normalised dataset comparison: identical data with different run_ids,
one changed cell, a run_id inside a list cell, and mismatched column sets.

WHY: The 18a1 fast-replay verdict rests on this comparison, so it must compare exactly and
normalise every run-id-bearing string cell.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.compare_datasets_normalised import compare_datasets


def write_dataset(
    folder: Path,
    run_id: str,
    *,
    amount_paise: int = 100,
    label_payment_ids: list[str] | None = None,
    extra_column: bool = False,
) -> None:
    """Writes one minimal dataset folder with payments, snapshots and labels tables."""
    folder.mkdir()
    manifest = {"worlds": [{"world_id": "w0", "run_id": run_id}]}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    payments: dict[str, list[object]] = {
        "run_id": [run_id],
        "world_id": ["w0"],
        "decision_time": ["2026-09-01T10:00:00+00:00"],
        "payment_id": ["p1"],
        "amount_paise": [amount_paise],
    }
    if extra_column:
        payments["extra"] = ["x"]
    pq.write_table(pa.table(payments), folder / "payments.parquet")

    snapshots = {
        "run_id": [run_id],
        "world_id": ["w0"],
        "snapshot_time": ["2026-09-01T10:30:00+00:00"],
        "bank_id": ["bank_0"],
        "account_id": ["a1"],
        "snapshot_kind": ["daily"],
        "balance_paise": [5_000],
    }
    pq.write_table(pa.table(snapshots), folder / "snapshots.parquet")

    labels = {
        "run_id": [run_id],
        "world_id": ["w0"],
        "arrival_time": ["2026-09-01T11:00:00+00:00"],
        "label_id": ["l1"],
        "account_id": ["a1"],
        "payment_ids": [label_payment_ids if label_payment_ids is not None else [f"{run_id}:p1"]],
        "source": ["complaint"],
    }
    pq.write_table(pa.table(labels), folder / "labels.parquet")


def test_same_data_with_different_run_ids_is_identical(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that two folders whose only difference is the run_id compare as identical."""
    write_dataset(tmp_path / "new", "runNew")
    write_dataset(tmp_path / "fixture", "runFixture")

    assert compare_datasets(tmp_path / "new", tmp_path / "fixture") == 0
    assert "VERDICT: IDENTICAL" in capsys.readouterr().out


def test_one_changed_cell_differs_with_one_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that one changed payment cell yields differing_rows=1 and the DIFFERS verdict."""
    write_dataset(tmp_path / "new", "runNew", amount_paise=101)
    write_dataset(tmp_path / "fixture", "runFixture", amount_paise=100)

    assert compare_datasets(tmp_path / "new", tmp_path / "fixture") == 1
    output = capsys.readouterr().out
    assert "payments: rows_new=1 rows_fixture=1 differing_rows=1" in output
    assert "VERDICT: DIFFERS" in output


def test_run_id_inside_a_list_cell_is_normalised(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that a run_id embedded in labels.payment_ids is replaced before the compare."""
    write_dataset(tmp_path / "new", "runNew", label_payment_ids=["runNew:p1"])
    write_dataset(tmp_path / "fixture", "runFixture", label_payment_ids=["runFixture:p1"])

    assert compare_datasets(tmp_path / "new", tmp_path / "fixture") == 0
    assert "VERDICT: IDENTICAL" in capsys.readouterr().out


def test_mismatched_columns_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that differing column sets return exit code 2 with a clear error."""
    write_dataset(tmp_path / "new", "runNew")
    write_dataset(tmp_path / "fixture", "runFixture", extra_column=True)

    assert compare_datasets(tmp_path / "new", tmp_path / "fixture") == 2
    assert "column sets differ" in capsys.readouterr().err
