"""
File: scripts/compare_datasets_normalised.py
Purpose: Compares two dataset folders table by table after replacing each world's own run_id
         with a fixed placeholder in every string cell.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PLACEHOLDER = "RUNID"

# Sort keys copied from replay/dataset_builder.py content_hash.
TABLES: dict[str, tuple[str, ...]] = {
    "payments": ("world_id", "decision_time", "payment_id"),
    "snapshots": ("world_id", "snapshot_time", "bank_id", "account_id", "snapshot_kind"),
    "labels": ("world_id", "arrival_time", "label_id", "account_id"),
}


class DatasetCompareError(Exception):
    """Raised when a folder cannot be compared unambiguously."""


def world_run_ids(folder: Path) -> dict[str, str]:
    """
    WHAT: Maps each world_id in the folder's manifest to its run_id.
    WHY: Normalisation must use the run_id that belongs to each row's world.
    """
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for world in manifest["worlds"]:
        world_id = world["world_id"]
        run_id = world["run_id"]
        if world_id in mapping and mapping[world_id] != run_id:
            raise DatasetCompareError(
                f"{folder}: world {world_id} has more than one run_id: {sorted({mapping[world_id], run_id})}"
            )
        mapping[world_id] = run_id
    if not mapping:
        raise DatasetCompareError(f"{folder}: manifest lists no worlds")
    return mapping


def normalise(value: Any, run_id: str) -> Any:
    """Replaces run_id inside every string, recursing into lists and dicts."""
    if isinstance(value, str):
        return value.replace(run_id, PLACEHOLDER)
    if isinstance(value, list):
        return [normalise(item, run_id) for item in value]
    if isinstance(value, dict):
        return {key: normalise(item, run_id) for key, item in value.items()}
    return value


def sort_value(value: Any) -> tuple[bool, str]:
    """Returns a total-order key for one sort column value, with None last."""
    if hasattr(value, "isoformat"):
        return (value is None, value.isoformat())
    return (value is None, str(value))


def load(folder: Path, table: str, run_ids: dict[str, str]) -> tuple[set[str], list[dict[str, Any]]]:
    """Loads one parquet table, normalises each row with its world's run_id, and sorts by the content keys."""
    arrow_table = pq.read_table(folder / f"{table}.parquet")
    rows: list[dict[str, Any]] = []
    for row in arrow_table.to_pylist():
        world_id = row["world_id"]
        if world_id not in run_ids:
            raise DatasetCompareError(f"{folder / (table + '.parquet')}: world {world_id} is not in the manifest")
        rows.append(normalise(row, run_ids[world_id]))
    keys = TABLES[table]
    rows.sort(key=lambda row: tuple(sort_value(row[key]) for key in keys))
    return set(arrow_table.column_names), rows


def differing_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    """Counts row positions whose cells differ, plus any rows left over from a length mismatch."""
    different = sum(1 for a, b in zip(left, right) if any(a[column] != b[column] for column in a))
    return different + abs(len(left) - len(right))


def compare_datasets(new_dir: Path, fixture_dir: Path) -> int:
    """
    WHAT: Compares the three dataset tables exactly after run-id normalisation.
    WHY: Returns the exit code the CLI uses: 0 identical, 1 differing, 2 incomparable input.
    """
    try:
        run_ids_new = world_run_ids(new_dir)
        run_ids_fixture = world_run_ids(fixture_dir)
        identical = True
        for table in TABLES:
            columns_new, rows_new = load(new_dir, table, run_ids_new)
            columns_fixture, rows_fixture = load(fixture_dir, table, run_ids_fixture)
            if columns_new != columns_fixture:
                raise DatasetCompareError(
                    f"{table}: column sets differ; only in new={sorted(columns_new - columns_fixture)} "
                    f"only in fixture={sorted(columns_fixture - columns_new)}"
                )
            different = differing_rows(rows_new, rows_fixture)
            if different:
                identical = False
            print(f"{table}: rows_new={len(rows_new)} rows_fixture={len(rows_fixture)} differing_rows={different}")
    except DatasetCompareError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("VERDICT: IDENTICAL" if identical else "VERDICT: DIFFERS")
    return 0 if identical else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Parses two dataset folders and returns the comparison exit code."""
    parser = argparse.ArgumentParser(description="Compare two dataset folders after run-id normalisation")
    parser.add_argument("new_dir", type=Path)
    parser.add_argument("fixture_dir", type=Path)
    args = parser.parse_args(argv)
    return compare_datasets(args.new_dir, args.fixture_dir)


if __name__ == "__main__":
    sys.exit(main())
