# File: evaluation/compact_rows.py
# Purpose: Builds the Level 18 account-model training matrix and validation account-day rows one
#          simulated world at a time, so the full-scale run never holds every row dictionary at once.
"""
WHAT:
    Loads dataset folders one world at a time, reuses the existing training-row builder and the
    existing one-row-per-account-day selector, and returns numpy arrays: a float32 feature matrix
    with integer targets and key arrays for the model, and optionally every account-day as flat
    arrays, precomputed ScoredDay records, a rules-tuning matrix and the feature matrix the model
    scores at the training-row keys.

WHY:
    Materialising every snapshot as a Python dictionary costs about 3,000 bytes per row, so the
    full-scale pilot (about 1.86 M training and 1.4 M validation rows) needs roughly 19 GB and does
    not fit a 16 GB laptop. A world is the natural unit: build_training_rows already groups rows by
    world and processes each world independently, so loading one world at a time produces exactly
    the same rows with memory bounded by the largest single world.

TERM: world: one simulated world, identified by world_id and stored under its own run_id.
TERM: key arrays: one entry per model row, the (world, bank, account, IST day) key build_training_rows
      returns, stored as string arrays plus the day's integer ordinal.

Every dataset folder, manifest entry and truth folder is required. Missing or malformed inputs raise
a plain ValueError naming the offending path or world; nothing is guessed.
"""

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy
import pyarrow as pa
import pyarrow.parquet as pq

from core.event_clock import IST
from decisions.anomaly_detector import TrainedAnomalyDetector
from decisions.mule_account_model import account_model_feature_names, build_training_rows
from decisions.rules_config import RulesConfig
from evaluation.dataset_truth_join import _join_table, _world_entries
from evaluation.level17_scoring import ScoredDay, one_row_per_account_day, score_account_days
from evaluation.rules_tuning import RulesDayMatrix, build_rules_day_matrix
from evaluation.truth import TruthStore, load_truth

__all__ = [
    "CompactDayRows",
    "CompactRows",
    "CompactSplit",
    "load_compact_split",
]


@dataclass(frozen=True)
class CompactRows:
    """
    WHAT: The training rows one split's worlds produce, stored as arrays instead of dictionaries.

    Attributes:
        feature_names: Model feature order, exactly account_model_feature_names().
        matrix: (row_count, feature_count) float32 feature values, NaN for a missing feature.
        targets: (row_count,) int64 binary labels, 1 for a positive account-day.
        world_ids: (row_count,) object array of world_id strings.
        bank_ids: (row_count,) object array of bank_id strings.
        account_ids: (row_count,) object array of account_id strings.
        day_ordinals: (row_count,) int64 date.toordinal() of each row's IST day.
    """

    feature_names: tuple[str, ...]
    matrix: numpy.ndarray
    targets: numpy.ndarray
    world_ids: numpy.ndarray
    bank_ids: numpy.ndarray
    account_ids: numpy.ndarray
    day_ordinals: numpy.ndarray


@dataclass(frozen=True)
class CompactDayRows:
    """
    WHAT: Every account-day of a split (not cut at the first label), with truth role and depth,
    plus the results later steps need from those rows.

    Why separate from CompactRows:
        Validation scoring and tuning need every account-day with its hidden role, while the model
        training matrix keeps only the days a bank could see. Both are derived from the same
        per-world snapshot load, so a caller asks for both in one pass, and the day dictionaries
        themselves are released as soon as these arrays exist.

    Attributes:
        feature_names: Model feature order, exactly account_model_feature_names().
        features: (row_count, feature_count) float64 feature values, NaN for a missing feature.
        world_ids: (row_count,) object array of world_id strings.
        bank_ids: (row_count,) object array of bank_id strings.
        account_ids: (row_count,) object array of account_id strings.
        day_ordinals: (row_count,) int64 date.toordinal() of each row's IST day.
        roles: (row_count,) object array of truth_role strings.
        depths: (row_count,) object array of truth_depth integers or None.
        scored: One ScoredDay per account-day, in row order, built by score_account_days with the
            supplied rules config and detector.
        rules_matrix: The same account-days as the flat arrays the vectorised rules tuner reads.
        aligned_features: (model_row_count, feature_count) float32 feature values for the model
            training-row keys, in build_training_rows key order.
    """

    feature_names: tuple[str, ...]
    features: numpy.ndarray
    world_ids: numpy.ndarray
    bank_ids: numpy.ndarray
    account_ids: numpy.ndarray
    day_ordinals: numpy.ndarray
    roles: numpy.ndarray
    depths: numpy.ndarray
    scored: tuple[ScoredDay, ...]
    rules_matrix: RulesDayMatrix
    aligned_features: numpy.ndarray


@dataclass(frozen=True)
class CompactSplit:
    """
    WHAT: One split's compact model rows, plus the account-day arrays when they were requested.

    Attributes:
        rows: Training matrix, targets and key arrays, concatenated in sorted world order.
        days: Per-account-day arrays and their precomputed results, or None when the caller did
            not ask. No day dictionaries are retained, so memory follows the arrays, not the rows.
    """

    rows: CompactRows
    days: CompactDayRows | None


def _world_sources(dataset_dirs: Sequence[Path]) -> list[tuple[str, Path]]:
    """
    WHAT: Returns (world_id, dataset folder) pairs for every world in the given folders.

    WHY: A dataset folder can hold several worlds, so the loader needs the world-to-folder map
    before it can read a single world. Sorting by world_id reproduces the order
    build_training_rows uses when it groups a split's rows by world.

    Args:
        dataset_dirs: Dataset folders written by write_dataset.

    Returns:
        (world_id, folder) pairs sorted by world_id.

    Raises:
        ValueError: If a folder has no usable manifest, a world repeats across folders, or a
            dataset is marked superseded.
    """
    if not dataset_dirs:
        raise ValueError("load_compact_split requires at least one dataset folder")
    pairs: list[tuple[str, Path]] = []
    for dataset_dir in dataset_dirs:
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Dataset manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("superseded_by"):
            raise ValueError(f"{dataset_dir.name} is superseded by {manifest['superseded_by']}; refusing to load it")
        for world_id in _world_entries(manifest):
            pairs.append((world_id, dataset_dir))
    pairs.sort(key=lambda pair: pair[0])
    seen: set[str] = set()
    for world_id, _ in pairs:
        if world_id in seen:
            raise ValueError(f"World {world_id!r} appears in more than one dataset folder")
        seen.add(world_id)
    return pairs


def _read_world_table(
    dataset_dir: Path,
    table_name: str,
    world_id: str,
) -> pa.Table:
    """
    WHAT: Reads only one world's rows of a dataset parquet table.

    WHY: The per-world filter is the whole memory saving: the other worlds' rows are never
    materialised, so peak memory follows one world rather than the split.

    Args:
        dataset_dir: Dataset folder holding the table.
        table_name: "snapshots" or "labels".
        world_id: World to keep.

    Returns:
        The world's rows.
    """
    return pq.read_table(dataset_dir / f"{table_name}.parquet", filters=[("world_id", "=", world_id)])


def _with_truth(
    snapshots: pa.Table,
    dataset_dir: Path,
    truth_root: Path,
    stores: dict[str, TruthStore],
) -> pa.Table:
    """
    WHAT: Appends truth_role, truth_owner and truth_depth to one world's snapshot rows.

    WHY: Validation metrics need the hidden role of every account-day. Reusing the snapshot join
    that join_truth applies keeps the columns and the minute rule identical to the full-table join;
    only the table is already filtered to one world.

    Args:
        snapshots: Snapshot table already filtered to one world.
        dataset_dir: Dataset folder, used to read the manifest's world entries.
        truth_root: Root folder holding one truth folder per run_id.
        stores: run_id -> loaded TruthStore cache filled by this function.

    Returns:
        The snapshot table with the three truth columns appended.

    Raises:
        ValueError: If the manifest is missing world entries or a row's time precedes clock_start.
    """
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = _world_entries(manifest)
    for run_id in {run_id for run_id, _ in entries.values()}:
        if run_id not in stores:
            stores[run_id] = load_truth(truth_root, run_id)
    return _join_table(snapshots, "account_id", "snapshot_time", "truth_role", entries, stores)


def _float_feature_columns(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> numpy.ndarray:
    """
    WHAT: Builds a float64 feature matrix from row dictionaries, with NaN for a None value.

    WHY: The rules tuner compares feature values with config thresholds using exact Python
    arithmetic, so it needs every value at full float precision; only the model matrix is float32.

    Args:
        rows: Account-day rows.
        feature_names: Feature order.

    Returns:
        (len(rows), len(feature_names)) float64 matrix.

    Raises:
        ValueError: If any present value is not numeric.
    """
    if not rows:
        return numpy.empty((0, len(feature_names)), dtype=numpy.float64)
    columns: list[numpy.ndarray] = []
    for name in feature_names:
        values = [row.get(name) for row in rows]
        for value in values:
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature {name!r} must be numeric or None; got {value!r}")
        columns.append(numpy.asarray(values, dtype=numpy.float64))
    return numpy.column_stack(columns)


def _string_column(rows: Sequence[dict[str, Any]], key: str, caller: str) -> numpy.ndarray:
    """
    WHAT: Returns one required non-empty string field of every row as an object array.

    Raises:
        ValueError: If a row's field is missing, empty, or not a string.
    """
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{caller} requires a non-empty string '{key}' on every row")
        values.append(value)
    return numpy.asarray(values, dtype=object)


def _day_ordinals(rows: Sequence[dict[str, Any]], caller: str) -> numpy.ndarray:
    """
    WHAT: Returns each row's IST day as an integer ordinal.

    Raises:
        ValueError: If a row's snapshot_time is missing or timezone-naive.
    """
    ordinals: list[int] = []
    for row in rows:
        moment = row.get("snapshot_time")
        if not isinstance(moment, datetime):
            raise ValueError(f"{caller} requires a timezone-aware 'snapshot_time' on every row")
        if moment.tzinfo is None:
            raise ValueError(f"{caller} requires a timezone-aware 'snapshot_time' on every row")
        ordinals.append(moment.astimezone(IST).date().toordinal())
    return numpy.asarray(ordinals, dtype=numpy.int64)


def _day_key(row: dict[str, Any]) -> tuple[str, str, str, date]:
    """
    WHAT: Returns the (world, bank, account, IST day) key of one account-day row.

    WHY: build_training_rows reports its rows as key tuples only, so matching them back to the
    deduplicated account-day arrays needs the same tuple shape.

    Raises:
        ValueError: If a required identifier or snapshot time is missing or malformed.
    """
    world_id = row.get("world_id")
    bank_id = row.get("bank_id")
    account_id = row.get("account_id")
    moment = row.get("snapshot_time")
    if not isinstance(world_id, str) or not world_id:
        raise ValueError("load_compact_split requires a non-empty string 'world_id' on every row")
    if not isinstance(bank_id, str) or not bank_id:
        raise ValueError("load_compact_split requires a non-empty string 'bank_id' on every row")
    if not isinstance(account_id, str) or not account_id:
        raise ValueError("load_compact_split requires a non-empty string 'account_id' on every row")
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("load_compact_split requires a timezone-aware 'snapshot_time' on every row")
    return (world_id, bank_id, account_id, moment.astimezone(IST).date())


def _aligned_day_features(
    world_features: numpy.ndarray,
    world_days: Sequence[dict[str, Any]],
    world_keys: Sequence[tuple[str, str, str, date]],
) -> numpy.ndarray:
    """
    WHAT: Returns one world's account-day features in the order of that world's training-row keys.

    WHY: The pilot scores the model on the account-day rows behind each training-row key, in key
    order. Those rows are the same deduplicated rows one_row_per_account_day produced, so their
    feature columns can be selected instead of rebuilding dictionaries.

    Args:
        world_features: (day_count, feature_count) float64 features of one world's account-days.
        world_days: The same world's account-day rows, in array order, used for their keys.
        world_keys: build_training_rows keys of that world, in key order.

    Returns:
        (len(world_keys), feature_count) float32 features in key order; the model consumes float32
        anyway, so storing the converted values changes no prediction.

    Raises:
        ValueError: If a key has no matching account-day row.
    """
    if not world_keys:
        return numpy.empty((0, world_features.shape[1]), dtype=numpy.float32)
    index_by_key = {_day_key(row): index for index, row in enumerate(world_days)}
    indices: list[int] = []
    for key in world_keys:
        index = index_by_key.get(key)
        if index is None:
            raise ValueError(f"Training row key {key!r} has no matching account-day snapshot")
        indices.append(index)
    return numpy.asarray(world_features[indices], dtype=numpy.float32)


def _concat_rule_matrices(pieces: Sequence[RulesDayMatrix]) -> RulesDayMatrix:
    """
    WHAT: Concatenates per-world rules-tuning matrices into one matrix, world by world.

    WHY: tune_rules_matrix needs one split-wide matrix, while the loader only ever holds a single
    world's rows. Concatenating the per-world arrays in sorted world order reproduces the matrix a
    single build_rules_day_matrix call over the whole split would build.

    Args:
        pieces: Non-empty per-world rules matrices, in sorted world order.

    Returns:
        One RulesDayMatrix whose rows follow the concatenation order.

    Raises:
        ValueError: If no piece was supplied.
    """
    if not pieces:
        raise ValueError("_concat_rule_matrices requires at least one per-world matrix")
    feature_names = tuple(pieces[0].features)
    features: dict[str, numpy.ndarray] = {}
    for name in feature_names:
        features[name] = numpy.concatenate([piece.features[name] for piece in pieces])
    return RulesDayMatrix(
        features=features,
        account_ids=numpy.concatenate([piece.account_ids for piece in pieces]),
        day_ordinals=numpy.concatenate([piece.day_ordinals for piece in pieces]),
        roles=numpy.concatenate([piece.roles for piece in pieces]),
    )


def load_compact_split(
    dataset_dirs: Sequence[Path],
    truth_root: Path | None = None,
    *,
    include_days: bool = False,
    world_ids: Sequence[str] | None = None,
    rules_config: RulesConfig | None = None,
    detector: TrainedAnomalyDetector | None = None,
) -> CompactSplit:
    """
    WHAT: Builds one split's compact arrays by loading and releasing one world at a time.

    WHY: Memory, see the module docstring. build_training_rows is called on a single world's rows,
    so the label cut, maturity window and feature order are the production ones, and concatenating
    the worlds in sorted world order reproduces the single-call row order exactly.

    Args:
        dataset_dirs: Dataset folders of one split, in any order.
        truth_root: Root folder holding one truth folder per run_id; required when include_days
            is true, because validation metrics need truth_role and truth_depth.
        include_days: When true, also return every account-day as arrays, its ScoredDay records,
            the rules-tuning matrix and the aligned model feature matrix. The day dictionaries are
            released world by world and are never part of the result.
        world_ids: Optional subset of world ids to load, for a bounded probe; None loads every
            world in the folders.
        rules_config: Rules table scored into the ScoredDay records and the rules matrix; required
            when include_days is true.
        detector: Trained anomaly detector scored into every ScoredDay. When None the anomaly
            score is None, exactly as score_account_days defines it; the probe uses that form
            because it measures memory and never scores.

    Returns:
        A CompactSplit whose rows equal build_training_rows over the whole split, first row first.

    Raises:
        ValueError: If a folder or manifest is missing, a world repeats, the truth root or rules
            config is absent when account-days are requested, or a row lacks a required field.

    Complexity:
        Time O(rows * feature_count) plus one parquet read per world. Without include_days, peak
        memory is O(largest world), because each world's dictionaries are released before the next
        world is read; the returned arrays hold O(rows * feature_count) float32 and O(rows) keys.
        With include_days, the result additionally holds O(account_days) float64 feature values,
        one ScoredDay per account-day, O(account_days * rule features) tuning arrays and
        O(model_rows * feature_count) float32 aligned features; the snapshot dictionaries are
        released after each world.
    """
    pairs = _world_sources(dataset_dirs)
    if world_ids is not None:
        wanted = set(world_ids)
        available = {world_id for world_id, _ in pairs}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"Requested worlds are not in the dataset folders: {missing}")
        pairs = [pair for pair in pairs if pair[0] in wanted]
    if include_days and truth_root is None:
        raise ValueError("include_days requires truth_root so account-days carry truth_role")
    if include_days and rules_config is None:
        raise ValueError("include_days requires rules_config so account-days carry rules scores")

    feature_names = account_model_feature_names()
    stores: dict[str, TruthStore] = {}
    matrices: list[numpy.ndarray] = []
    targets: list[int] = []
    keys: list[tuple[str, str, str, date]] = []
    day_features: list[numpy.ndarray] = []
    day_worlds: list[numpy.ndarray] = []
    day_banks: list[numpy.ndarray] = []
    day_accounts: list[numpy.ndarray] = []
    day_ordinals: list[numpy.ndarray] = []
    day_roles: list[numpy.ndarray] = []
    day_depths: list[numpy.ndarray] = []
    scored_days: list[ScoredDay] = []
    rule_matrices: list[RulesDayMatrix] = []
    aligned_features: list[numpy.ndarray] = []

    for world_id, dataset_dir in pairs:
        snapshots = _read_world_table(dataset_dir, "snapshots", world_id)
        labels = _read_world_table(dataset_dir, "labels", world_id)
        if truth_root is not None:
            snapshots = _with_truth(snapshots, dataset_dir, truth_root, stores)
        snapshot_rows = snapshots.to_pylist()
        label_rows = labels.to_pylist()

        world_matrix, world_targets, world_keys = build_training_rows(snapshot_rows, label_rows)
        matrices.append(
            numpy.asarray(world_matrix, dtype=numpy.float32)
            if world_matrix
            else numpy.empty((0, len(feature_names)), dtype=numpy.float32)
        )
        targets.extend(world_targets)
        keys.extend(world_keys)

        if include_days:
            world_days = one_row_per_account_day(snapshot_rows)
            world_features = _float_feature_columns(world_days, feature_names)
            day_features.append(world_features)
            day_worlds.append(_string_column(world_days, "world_id", "load_compact_split"))
            day_banks.append(_string_column(world_days, "bank_id", "load_compact_split"))
            day_accounts.append(_string_column(world_days, "account_id", "load_compact_split"))
            day_ordinals.append(_day_ordinals(world_days, "load_compact_split"))
            day_roles.append(_string_column(world_days, "truth_role", "load_compact_split"))
            day_depths.append(numpy.asarray([row.get("truth_depth") for row in world_days], dtype=object))
            scored_days.extend(score_account_days(world_days, rules_config, detector))
            rule_matrices.append(build_rules_day_matrix(world_days, rules_config))
            aligned_features.append(_aligned_day_features(world_features, world_days, world_keys))

        del snapshot_rows, label_rows, snapshots, labels

    if matrices:
        matrix_array = numpy.concatenate(matrices, axis=0)
        feature_count = len(feature_names)
        matrix_array = matrix_array.reshape((len(targets), feature_count))
    else:
        matrix_array = numpy.empty((0, len(feature_names)), dtype=numpy.float32)
    rows = CompactRows(
        feature_names=tuple(feature_names),
        matrix=matrix_array,
        targets=numpy.asarray(targets, dtype=numpy.int64),
        world_ids=numpy.asarray([key[0] for key in keys], dtype=object),
        bank_ids=numpy.asarray([key[1] for key in keys], dtype=object),
        account_ids=numpy.asarray([key[2] for key in keys], dtype=object),
        day_ordinals=numpy.asarray([key[3].toordinal() for key in keys], dtype=numpy.int64),
    )

    if not include_days:
        return CompactSplit(rows=rows, days=None)
    days = CompactDayRows(
        feature_names=tuple(feature_names),
        features=numpy.concatenate(day_features, axis=0),
        world_ids=numpy.concatenate(day_worlds, axis=0),
        bank_ids=numpy.concatenate(day_banks, axis=0),
        account_ids=numpy.concatenate(day_accounts, axis=0),
        day_ordinals=numpy.concatenate(day_ordinals, axis=0),
        roles=numpy.concatenate(day_roles, axis=0),
        depths=numpy.concatenate(day_depths, axis=0),
        scored=tuple(scored_days),
        rules_matrix=_concat_rule_matrices(rule_matrices),
        aligned_features=numpy.concatenate(aligned_features, axis=0),
    )
    return CompactSplit(rows=rows, days=days)
