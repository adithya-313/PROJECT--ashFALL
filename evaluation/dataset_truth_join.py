"""
WHAT: Evaluation-only join that attaches hidden-truth columns to the tables of a built dataset.
Reads payments.parquet, snapshots.parquet and manifest.json from a dataset folder and returns new
in-memory tables with extra truth columns. Never writes into the dataset folder.

WHY: Offline scoring needs to know, for every payment receiver and every snapshot account, what the
account really was (mule or genuine) at the moment of the row. That answer key is hidden truth and
must stay out of operational code, so the join lives in the evaluation package only.

TERM: world minute: whole minutes since IST midnight of the world's clock_start date (floor).
TERM: half-open span: a role is active from valid_from (included) up to valid_to (excluded).
TERM: truth join: looking up role, owner and depth at the row's own world minute.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from core.event_clock import ist_to_world_minute
from evaluation.truth import TruthStore, load_truth

__all__ = ["join_truth"]


def _world_entries(manifest: dict[str, Any]) -> dict[str, tuple[str, date]]:
    """
    WHAT: Maps each world_id in the manifest to its (run_id, clock_start date).
    WHY: Truth is stored per run and the minute rule needs each world's own clock start.

    Args:
        manifest: Parsed dataset manifest.json.

    Returns:
        Mapping world_id -> (run_id, clock_start).

    Raises:
        ValueError: If worlds, run_id or clock_start are missing or malformed.

    Example:
        # _world_entries({"worlds": [{"world_id": "w", "run_id": "r", "clock_start": "2026-09-01"}]})
    """
    worlds = manifest.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise ValueError("Dataset manifest has no 'worlds' entries; cannot join truth.")
    entries: dict[str, tuple[str, date]] = {}
    for world in worlds:
        world_id = world.get("world_id")
        run_id = world.get("run_id")
        clock_start = world.get("clock_start")
        if not world_id or not run_id or not clock_start:
            raise ValueError(f"Manifest world entry is missing world_id/run_id/clock_start: {world!r}")
        try:
            start = date.fromisoformat(str(clock_start))
        except ValueError as exc:
            raise ValueError(f"Manifest clock_start {clock_start!r} is not an ISO date.") from exc
        entries[str(world_id)] = (str(run_id), start)
    return entries


def _minute_of(moment: Any, start: date, label: str) -> int:
    """
    WHAT: Converts one row time to its world minute and refuses times before clock_start.
    WHY: A negative minute means the dataset and clock_start disagree; joining would be silent nonsense.

    Args:
        moment: Timezone-aware datetime from the row.
        start: The world's clock_start date.
        label: Column name, for the error message.

    Returns:
        Non-negative world minute.

    Raises:
        ValueError: If the time is missing or earlier than clock_start.

    Example:
        # _minute_of(datetime(2026, 9, 1, tzinfo=IST), date(2026, 9, 1), "decision_time") == 0
    """
    if moment is None:
        raise ValueError(f"Row has no {label}; cannot join truth.")
    minute = ist_to_world_minute(moment, start)
    if minute < 0:
        raise ValueError(f"Row {label} {moment.isoformat()} is before clock_start {start.isoformat()}.")
    return minute


def _append_columns(table: pa.Table, columns: list[tuple[str, list[Any], pa.DataType]]) -> pa.Table:
    """
    WHAT: Appends new columns to a table, leaving existing columns untouched.
    WHY: The join must never change the original dataset columns.

    Args:
        table: Source table.
        columns: (name, values, arrow type) triples.

    Returns:
        New table with the extra columns at the end.

    Raises:
        ValueError: If a new column name already exists.

    Example:
        # _append_columns(t, [("truth_role", ["genuine"], pa.string())])
    """
    for name, values, dtype in columns:
        if name in table.column_names:
            raise ValueError(f"Column {name!r} already exists in the dataset table.")
        table = table.append_column(name, pa.array(values, type=dtype))
    return table


def _join_table(
    table: pa.Table,
    account_col: str,
    time_col: str,
    role_name: str,
    entries: dict[str, tuple[str, date]],
    stores: dict[str, TruthStore],
) -> pa.Table:
    """
    WHAT: Looks up role, owner and depth for every row of one table at that row's own world minute.
    WHY: Shared by payments (receiver) and snapshots (account) so both follow one minute rule.

    Args:
        table: payments or snapshots table.
        account_col: Column holding the account id to look up.
        time_col: Column holding the UTC row time.
        role_name: Name of the added role column.
        entries: world_id -> (run_id, clock_start).
        stores: run_id -> loaded TruthStore.

    Returns:
        Table with role, truth_owner and truth_depth columns appended.

    Raises:
        ValueError: If a row's world_id is not in the manifest or its time is before clock_start.

    Example:
        # _join_table(payments, "receiver_account_id", "decision_time", "truth_role_receiver", e, s)
    """
    roles: list[str] = []
    owners: list[str | None] = []
    depths: list[int | None] = []
    for world_id, account, moment in zip(
        table.column("world_id").to_pylist(),
        table.column(account_col).to_pylist(),
        table.column(time_col).to_pylist(),
        strict=True,
    ):
        if world_id not in entries:
            raise ValueError(f"Row world_id {world_id!r} is not listed in the dataset manifest.")
        run_id, start = entries[world_id]
        minute = _minute_of(moment, start, time_col)
        store = stores[run_id]
        roles.append(store.role_at(account, minute))
        owners.append(store.owner_at(account, minute))
        depths.append(store.depth_at(account, minute))
    return _append_columns(
        table,
        [
            (role_name, roles, pa.string()),
            ("truth_owner", owners, pa.string()),
            ("truth_depth", depths, pa.int64()),
        ],
    )


def join_truth(dataset_dir: Path, truth_root: Path) -> tuple[pa.Table, pa.Table]:
    """
    WHAT: Returns (payments, snapshots) tables with hidden-truth columns added, looked up at each row's
    own world minute. Payments get truth_role_receiver; snapshots get truth_role; both get truth_owner
    and truth_depth. Nothing is written into dataset_dir and no existing column changes.

    WHY: Lets evaluation score a model against what each account really was when the row happened.

    Args:
        dataset_dir: Folder written by write_dataset (payments.parquet, snapshots.parquet, manifest.json).
        truth_root: Root folder holding one truth folder per run_id.

    Returns:
        (payments_with_truth, snapshots_with_truth).

    Raises:
        ValueError: If the manifest lacks run_id/clock_start or a row time is before clock_start.
        PermissionError: If called from an operational package (raised by load_truth).

    Example:
        # payments, snapshots = join_truth(Path("data/datasets/training-..."), Path("runs/truth"))
    """
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = _world_entries(manifest)
    stores = {run_id: load_truth(truth_root, run_id) for run_id, _ in entries.values()}
    payments = pq.read_table(dataset_dir / "payments.parquet")
    snapshots = pq.read_table(dataset_dir / "snapshots.parquet")
    return (
        _join_table(payments, "receiver_account_id", "decision_time", "truth_role_receiver", entries, stores),
        _join_table(snapshots, "account_id", "snapshot_time", "truth_role", entries, stores),
    )
