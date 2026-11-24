# File: evaluation/calibration.py
# Purpose: Fit isotonic calibration on validation scores for the frozen Level 18 models and check it
# on test.
"""
WHAT:
    Fits one isotonic calibrator per frozen model on validation probabilities against validation
    truth, applies it unchanged to test, proves the calibration never reorders rows, and reports the
    test Brier score, expected calibration error and reliability points before and after
    calibration. The account model is fitted on account-day probabilities against the hidden role;
    the payment model is fitted on validation payment probabilities from `load_split_rows` against
    the receiver-reported target. The payment test rows are rebuilt with the production task 4
    scoring path and must reproduce task 4's recorded totals and truth AP exactly.

WHY:
    A-060 and A-060a fix the truth definitions, the fit split, the monotone rule, the bins and the
    output shape before any calibration value exists, so the reported numbers measure the frozen
    models instead of a choice made after seeing test. No model, artifact or threshold changes, and
    the test split is read only for the pre-registered checks.

TERM: isotonic calibration: a non-decreasing step function that maps a raw probability to its
      observed rate on the fit split.
TERM: equal-count bin: one of ten contiguous groups of the test rows after sorting by probability,
      each holding an equal share of the rows; the last groups differ by at most one row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost
from sklearn.isotonic import IsotonicRegression

from core.event_clock import ist_to_world_minute
from decisions.mule_account_model import predict_account_days
from decisions.pu_learning import PUBaggedAccountModel
from evaluation.dataset_truth_join import _world_entries
from evaluation.level17_scoring import GENUINE_ROLES, MULE_ROLES, one_row_per_account_day
from evaluation.payment_model import TARGET, build_matrix, load_feature_order, load_split_rows, predict_scores
from evaluation.payment_oof_risk import (
    _CommitSampler,
    _git_head,
    _read_world_snapshots,
    load_frozen_recipe,
    validate_dataset_split,
)
from evaluation.payment_test_score import (
    _average_precision_or_none,
    aggregate,
    score_split,
    validate_scoring_split,
    verify_booster_rounds,
    verify_frozen_artifacts,
)
from evaluation.reason_codes import verify_account_checkpoint
from evaluation.truth import load_truth

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "full" / "payment_model_v1"
DEFAULT_OOF_ROOT = REPO_ROOT / "runs" / "full" / "payment_oof"
DEFAULT_TRUTH_ROOT = REPO_ROOT / "runs" / "full" / "truth"
DEFAULT_SCORING_RESULT = DEFAULT_MODEL_DIR / "test_score" / "result.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "full" / "level18b" / "calibration"

AMENDMENTS = ("A-060", "A-060a")
BINS = 10
APPLICATION_RULE = "linear interpolation between the thresholds, clipped to [y_min, y_max]"
#: A-060a fixes the task 4 reproduction totals before any calibration value exists.
EXPECTED_TEST_ROWS = 1_846_587
EXPECTED_TEST_POSITIVES = 43_239


@dataclass(frozen=True)
class Calibrator:
    """One fitted isotonic calibrator: its breakpoints and the clipping bounds."""

    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    y_min: float = 0.0
    y_max: float = 1.0

    def payload(self) -> dict[str, Any]:
        """Returns the committed JSON payload for this calibrator."""
        return {
            "x_thresholds": list(self.x_thresholds),
            "y_thresholds": list(self.y_thresholds),
            "y_min": self.y_min,
            "y_max": self.y_max,
            "application": APPLICATION_RULE,
        }


@dataclass(frozen=True)
class WorldRecord:
    """One test world's rebuilt row count, positive count and truth average precision."""

    world_id: str
    rows: int
    positives: int
    truth_ap: float | None


def _sha256_file(path: Path) -> str:
    """Returns the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _probability_truth(
    probabilities: Sequence[float] | np.ndarray,
    truth: Sequence[int] | np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    WHAT: Returns the probabilities and 0/1 truth as validated float64 and int8 arrays.

    WHY: Every calibration function shares one boundary check, so a length mismatch, a non-finite
    score, an out-of-range probability or a truth value other than 0 and 1 raises instead of being
    silently absorbed into an average.

    Raises:
        ValueError: If the arrays differ in length, are empty or not one-dimensional, hold a
            non-finite or out-of-range probability, or hold a truth value other than 0 and 1.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int8)
    if values.ndim != 1 or labels.ndim != 1:
        raise ValueError(f"{label} needs one-dimensional probabilities and truth")
    if values.size == 0:
        raise ValueError(f"{label} holds no rows")
    if values.size != labels.size:
        raise ValueError(f"{label} holds {values.size} probabilities for {labels.size} truth values")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{label} holds a non-finite probability")
    if bool(np.any(values < 0.0)) or bool(np.any(values > 1.0)):
        raise ValueError(f"{label} holds a probability outside [0, 1]")
    if bool(np.any((labels != 0) & (labels != 1))):
        raise ValueError(f"{label} holds a truth value other than 0 and 1")
    return values, labels


def truth_label(role: str) -> int:
    """
    WHAT: Maps a hidden account role to the 1/0 calibration truth.

    WHY: A-060 fixes mule, collector and cash-out as positive and genuine and victim as negative,
    so an unexpected role must stop the phase instead of being counted as one side by accident.

    Raises:
        ValueError: If the role is neither a mule role nor a genuine role.
    """
    if role in MULE_ROLES:
        return 1
    if role in GENUINE_ROLES:
        return 0
    raise ValueError(f"hidden truth role {role!r} is neither a mule role nor a genuine role")


def isotonic_fit(probabilities: Sequence[float] | np.ndarray, truth: Sequence[int] | np.ndarray) -> Calibrator:
    """
    WHAT: Fits the A-060 isotonic calibrator on validation probabilities against validation truth.

    WHY: A-060 ruling 4 fixes the sklearn class and its bounds, and ruling 5 fixes that the fitted
    thresholds must be non-decreasing; a single-class validation split cannot say which way to
    calibrate and stops the phase.

    Raises:
        ValueError: If the input fails the probability/truth checks or holds only one class.
    """
    values, labels = _probability_truth(probabilities, truth, "isotonic fit")
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        raise ValueError("the isotonic fit needs both classes in its validation truth")
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(values, labels)
    x_thresholds = tuple(float(value) for value in model.X_thresholds_)
    y_thresholds = tuple(float(value) for value in model.y_thresholds_)
    if not x_thresholds:
        raise ValueError("the isotonic fit produced no thresholds")
    if any(left > right for left, right in zip(y_thresholds, y_thresholds[1:])):
        raise ValueError("the fitted isotonic thresholds are not non-decreasing")
    return Calibrator(x_thresholds=x_thresholds, y_thresholds=y_thresholds)


def isotonic_apply(calibrator: Calibrator, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    WHAT: Applies a calibrator with linear interpolation between its thresholds and a final clip.

    WHY: A-060 ruling 7 fixes this application rule, so the number written in the calibrator file and
    the number this run reports come from the same arithmetic.

    Raises:
        ValueError: If the values are empty, not one-dimensional or hold a non-finite value.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("isotonic_apply needs a non-empty one-dimensional score list")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("isotonic_apply received a non-finite score")
    calibrated = np.interp(
        values,
        np.asarray(calibrator.x_thresholds, dtype=np.float64),
        np.asarray(calibrator.y_thresholds, dtype=np.float64),
    )
    return np.clip(calibrated, calibrator.y_min, calibrator.y_max)


def assert_ranking_preserved(
    probabilities: Sequence[float] | np.ndarray,
    calibrated: Sequence[float] | np.ndarray,
    label: str,
) -> dict[str, Any]:
    """
    WHAT: Proves a calibration leaves the rows in their raw-probability order.

    WHY: A-060 ruling 5 says isotonic calibration may not reorder rows; ordering by the raw
    probability must leave the calibrated probability non-decreasing, and a violation stops the
    phase rather than being reported as a curiosity.

    Raises:
        ValueError: If the arrays differ in length or shape, hold a non-finite value, or the
            calibrated value decreases along the stable ascending raw order.
    """
    raw_values = np.asarray(probabilities, dtype=np.float64)
    calibrated_values = np.asarray(calibrated, dtype=np.float64)
    if raw_values.ndim != 1 or calibrated_values.ndim != 1:
        raise ValueError(f"{label} needs one-dimensional arrays")
    if raw_values.size == 0:
        raise ValueError(f"{label} holds no rows")
    if raw_values.size != calibrated_values.size:
        raise ValueError(f"{label} holds {raw_values.size} raw and {calibrated_values.size} calibrated values")
    if not bool(np.all(np.isfinite(raw_values))) or not bool(np.all(np.isfinite(calibrated_values))):
        raise ValueError(f"{label} holds a non-finite value")
    order = np.argsort(raw_values, kind="stable")
    ordered = calibrated_values[order]
    if not bool(np.all(np.diff(ordered) >= 0.0)):
        raise ValueError(f"{label} calibration reorders rows by raw probability")
    return {"rows": int(raw_values.size), "monotone": True}


def calibration_metrics(
    probabilities: Sequence[float] | np.ndarray,
    truth: Sequence[int] | np.ndarray,
    bins: int = BINS,
) -> dict[str, Any]:
    """
    WHAT: Returns the Brier score, the expected calibration error and the reliability points.

    WHY: A-060 ruling 6 fixes the Brier formula and the ten equal-count bins, so the raw and the
    calibrated probability are summarised by one function and the bins always cover every row.

    Raises:
        ValueError: If the input fails the probability/truth checks, the bin count is not positive,
            or the rows are fewer than the bins.
    """
    values, labels = _probability_truth(probabilities, truth, "calibration metrics")
    if bins <= 0:
        raise ValueError(f"calibration bins must be positive; got {bins!r}")
    if values.size < bins:
        raise ValueError(f"calibration bins need at least {bins} rows; got {values.size}")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_labels = labels[order]
    points: list[dict[str, Any]] = []
    error = 0.0
    for index, positions in enumerate(np.array_split(np.arange(values.size), bins)):
        count = int(positions.size)
        bin_values = ordered_values[positions]
        bin_labels = ordered_labels[positions]
        mean_probability = float(bin_values.mean())
        observed_rate = float(bin_labels.mean())
        error += (count / values.size) * abs(observed_rate - mean_probability)
        points.append(
            {
                "bin": index,
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "min_probability": float(bin_values.min()),
                "max_probability": float(bin_values.max()),
            }
        )
    return {
        "rows": int(values.size),
        "positives": int(labels.sum()),
        "brier": float(np.mean((values - labels) ** 2)),
        "ece": float(error),
        "bins": points,
    }


def write_calibrator(path: Path, calibrator: Calibrator) -> None:
    """Writes one calibrator as JSON with LF newlines so its sha256 is stable."""
    text = json.dumps(calibrator.payload(), indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def load_calibrator(path: Path) -> Calibrator:
    """
    WHAT: Loads a calibrator file after proving it holds only the A-060 ruling 7 fields.

    WHY: A reviewer must be able to apply exactly the committed thresholds; an extra statistic or a
    changed application rule would mean the file is not the one this module writes.

    Raises:
        ValueError: If the JSON is not a mapping with exactly the fixed keys, the bounds or the
            application rule differ, or the thresholds are malformed or decreasing.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON mapping")
    expected_keys = {"x_thresholds", "y_thresholds", "y_min", "y_max", "application"}
    if set(payload) != expected_keys:
        raise ValueError(f"{path} holds keys {sorted(payload)}; expected {sorted(expected_keys)}")
    if payload["y_min"] != 0.0 or payload["y_max"] != 1.0:
        raise ValueError(f"{path} does not hold the fixed [0, 1] bounds")
    if payload["application"] != APPLICATION_RULE:
        raise ValueError(f"{path} does not hold the fixed application rule")
    x_thresholds = payload["x_thresholds"]
    y_thresholds = payload["y_thresholds"]
    if not isinstance(x_thresholds, list) or not isinstance(y_thresholds, list):
        raise ValueError(f"{path} holds no threshold lists")
    if not x_thresholds or len(x_thresholds) != len(y_thresholds):
        raise ValueError(f"{path} holds {len(x_thresholds)} x and {len(y_thresholds)} y thresholds")
    x_values = tuple(float(value) for value in x_thresholds)
    y_values = tuple(float(value) for value in y_thresholds)
    if any(left > right for left, right in zip(x_values, x_values[1:])):
        raise ValueError(f"{path} holds decreasing x thresholds")
    if any(left > right for left, right in zip(y_values, y_values[1:])):
        raise ValueError(f"{path} holds decreasing y thresholds")
    return Calibrator(x_thresholds=x_values, y_thresholds=y_values)


def verify_task4_reproduction(
    records: Sequence[WorldRecord],
    reference: Mapping[str, Any],
    expected_rows: int,
    expected_positives: int,
) -> dict[str, Any]:
    """
    WHAT: Proves rebuilt test worlds equal task 4's recorded worlds, totals and truth AP median.

    WHY: A-060a makes this the same-input proof for the payment test check; a rebuilt split that
    differs in rows, positives or per-world truth AP means the check is measuring something else,
    so it stops before a calibrator is fitted.

    Raises:
        ValueError: If the reference lacks the test block or world list, the world count, a world
            id, a row count, a positive count or a truth AP differs, a total differs from the
            pre-registered value, a world has no defined AP, or the AP median differs.
    """
    splits = reference.get("splits")
    test_block = splits.get("test") if isinstance(splits, Mapping) else None
    if not isinstance(test_block, Mapping):
        raise ValueError("the task 4 result holds no test split block")
    reference_worlds = test_block.get("worlds")
    if not isinstance(reference_worlds, list):
        raise ValueError("the task 4 result holds no test world list")
    if len(records) != len(reference_worlds):
        raise ValueError(f"rebuilt {len(records)} test worlds but task 4 recorded {len(reference_worlds)}")
    rows = 0
    positives = 0
    for record, expected in zip(records, reference_worlds, strict=True):
        if not isinstance(expected, Mapping):
            raise ValueError("the task 4 result holds a test world that is not a mapping")
        for name, actual in (
            ("world_id", record.world_id),
            ("rows", record.rows),
            ("positives", record.positives),
            ("truth_ap", record.truth_ap),
        ):
            if actual != expected.get(name):
                raise ValueError(
                    f"test world {record.world_id!r} {name} {actual!r} differs from task 4's {expected.get(name)!r}"
                )
        rows += record.rows
        positives += record.positives
    if rows != expected_rows:
        raise ValueError(f"rebuilt test rows {rows} differ from the pre-registered {expected_rows}")
    if positives != expected_positives:
        raise ValueError(f"rebuilt test positives {positives} differ from the pre-registered {expected_positives}")
    truth_aps: list[float] = []
    for record in records:
        if record.truth_ap is None:
            raise ValueError(f"test world {record.world_id!r} has no truth AP")
        truth_aps.append(record.truth_ap)
    computed = aggregate(truth_aps)
    aggregate_block = test_block.get("aggregate")
    recorded_median = (
        aggregate_block.get("truth_ap", {}).get("median") if isinstance(aggregate_block, Mapping) else None
    )
    if computed["median"] != recorded_median:
        raise ValueError(f"rebuilt truth AP median {computed['median']!r} differs from task 4's {recorded_median!r}")
    return {
        "worlds": len(records),
        "rows": rows,
        "positives": positives,
        "truth_ap_median": float(recorded_median),
    }


def _split_datasets(dirs: Sequence[Path], split: str) -> list[Path]:
    """
    WHAT: Returns the split's dataset folders sorted, proving each belongs to that split and their
    names differ.

    Raises:
        ValueError: If no folder is given, two folders share a name, or a folder records another
            split.
    """
    if not dirs:
        raise ValueError(f"calibration needs at least one {split} dataset")
    datasets = sorted(Path(directory) for directory in dirs)
    names = [directory.name for directory in datasets]
    if len(names) != len(set(names)):
        raise ValueError(f"{split} dataset folder names repeat: {names}")
    validator = validate_dataset_split if split == "validation" else validate_scoring_split
    for directory in datasets:
        validator(directory, split)
    return datasets


def _content_hashes(datasets: Sequence[Path], split: str) -> dict[str, Any]:
    """Returns each dataset's recorded content hash by folder name."""
    validator = validate_dataset_split if split == "validation" else validate_scoring_split
    return {dataset.name: validator(dataset, split).get("content_hash") for dataset in datasets}


def account_split_rows(
    datasets: Sequence[Path],
    split: str,
    model: PUBaggedAccountModel,
    truth_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    WHAT: Scores every account-day of one split and labels it with the hidden role at its kept
    snapshot's world minute.

    WHY: A-060 ruling 2 fixes one row per account-day per account, the production scorer and the
    truth lookup; each world is read on its own so only one world's snapshots are held at a time.

    Raises:
        ValueError: If a folder records another split, a world entry or snapshot lacks a key field,
            a snapshot time is before the world clock start, or the role is not a truth role.
    """
    validator = validate_dataset_split if split == "validation" else validate_scoring_split
    probabilities: list[float] = []
    truth: list[int] = []
    world_count = 0
    for dataset_dir in datasets:
        manifest = validator(dataset_dir, split)
        entries = _world_entries(manifest)
        stores = {run_id: load_truth(truth_root, run_id) for run_id, _ in entries.values()}
        world_entries = manifest.get("worlds")
        if not isinstance(world_entries, list) or not world_entries:
            raise ValueError(f"{dataset_dir.name} manifest holds no world list")
        for entry in world_entries:
            world_id = entry.get("world_id") if isinstance(entry, Mapping) else None
            if not isinstance(world_id, str) or not world_id:
                raise ValueError(f"{dataset_dir.name} manifest holds a world entry without world_id")
            if world_id not in entries:
                raise ValueError(f"{dataset_dir.name} manifest does not map world {world_id!r} to a run")
            run_id, clock_start = entries[world_id]
            day_rows = one_row_per_account_day(_read_world_snapshots(dataset_dir, world_id))
            if not day_rows:
                raise ValueError(f"world {world_id!r} holds no account-day rows in {dataset_dir.name}")
            scores = predict_account_days(model, day_rows)
            for row, probability in zip(day_rows, scores, strict=True):
                account_id = row.get("account_id")
                if not isinstance(account_id, str) or not account_id:
                    raise ValueError(f"world {world_id!r} holds an account-day without an account_id")
                moment = row.get("snapshot_time")
                if not isinstance(moment, datetime):
                    raise ValueError(f"world {world_id!r} holds an account-day without a snapshot_time")
                minute = ist_to_world_minute(moment, clock_start)
                if minute < 0:
                    raise ValueError(f"world {world_id!r} holds an account-day before clock_start")
                role = stores[run_id].role_at(account_id, minute)
                probabilities.append(float(probability))
                truth.append(truth_label(role))
            world_count += 1
    if not truth:
        raise ValueError(f"the {split} datasets hold no account-day rows")
    labels = np.asarray(truth, dtype=np.int8)
    return (
        np.asarray(probabilities, dtype=np.float64),
        labels,
        {
            "rows": int(labels.size),
            "positives": int(labels.sum()),
            "negatives": int(labels.size - int(labels.sum())),
            "worlds": world_count,
        },
    )


def payment_validation_rows(
    datasets: Sequence[Path],
    features: Sequence[str],
    encodings: Mapping[str, Sequence[str]],
    model: xgboost.XGBClassifier,
    oof_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    WHAT: Scores every validation payment with the frozen model and labels it with the
    receiver-reported target.

    WHY: A-060 ruling 3 fixes validation payments as the production `load_split_rows` rows and their
    `label_receiver_reported_by_cutoff` column, so the fit sees the same rows and label the payment
    model was trained on.

    Raises:
        ValueError: If the validation rows are empty or their targets are not 0/1.
    """
    table = load_split_rows(list(datasets), "validation", features, oof_root)
    if table.num_rows == 0:
        raise ValueError("the validation datasets hold no payment rows")
    matrix = build_matrix(table, features, encodings)
    probabilities = np.asarray(predict_scores(model, matrix), dtype=np.float64)
    truth = np.asarray(table[TARGET].to_pylist(), dtype=np.int8)
    if truth.size != probabilities.size:
        raise ValueError(f"{truth.size} targets do not match {probabilities.size} scores")
    return (
        probabilities,
        truth,
        {
            "rows": int(truth.size),
            "positives": int(truth.sum()),
            "negatives": int(truth.size - int(truth.sum())),
        },
    )


def payment_test_rows(
    datasets: Sequence[Path],
    features: Sequence[str],
    encodings: Mapping[str, Sequence[str]],
    model: xgboost.XGBClassifier,
    truth_root: Path,
    account_model: PUBaggedAccountModel,
    account_config: Path,
    reference_path: Path,
    expected_rows: int,
    expected_positives: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    WHAT: Scores every test payment with the production task 4 path and proves it reproduces task 4.

    WHY: A-060a fixes the test check to the exact task 4 functions, and the reproduction check makes
    a changed split, model or payee risk stop the phase before a calibrator is fitted.

    Raises:
        ValueError: If the production path, the reproduction check or the rebuilt totals fail.
    """
    worlds, reason_counts, dataset_records = score_split(
        list(datasets),
        "test",
        features,
        encodings,
        model,
        truth_root,
        account_model,
        account_config,
    )
    records = [
        WorldRecord(
            world_id=world.world_id,
            rows=len(world.scores),
            positives=int(sum(world.labels)),
            truth_ap=_average_precision_or_none(list(world.scores), list(world.labels)),
        )
        for world in worlds
    ]
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if not isinstance(reference, dict):
        raise ValueError(f"{reference_path} does not hold a JSON mapping")
    reproduction = verify_task4_reproduction(records, reference, expected_rows, expected_positives)
    reproduction["reference"] = str(reference_path)
    reproduction["reference_sha256"] = _sha256_file(reference_path)
    probabilities = np.asarray([score for world in worlds for score in world.scores], dtype=np.float64)
    truth = np.asarray([label for world in worlds for label in world.labels], dtype=np.int8)
    return (
        probabilities,
        truth,
        {
            "rows": int(truth.size),
            "positives": int(truth.sum()),
            "negatives": int(truth.size - int(truth.sum())),
            "worlds": len(worlds),
            "datasets": dataset_records,
            "nan_reason_counts": reason_counts,
            "reproduction": reproduction,
        },
    )


def _calibrate_model(
    name: str,
    validation_probabilities: Sequence[float] | np.ndarray,
    validation_truth: Sequence[int] | np.ndarray,
    test_probabilities: Sequence[float] | np.ndarray,
    test_truth: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """
    WHAT: Fits one model's calibrator on validation and returns its fit diagnostics, ranking checks
    and test metrics before and after calibration.

    WHY: Keeping the fit and both checks in one function makes the calibrator that produced the
    numbers the calibrator the run writes.

    Raises:
        ValueError: If the fit or any ranking check fails.
    """
    calibrator = isotonic_fit(validation_probabilities, validation_truth)
    validation_calibrated = isotonic_apply(calibrator, validation_probabilities)
    test_calibrated = isotonic_apply(calibrator, test_probabilities)
    ranking = {
        "validation": assert_ranking_preserved(validation_probabilities, validation_calibrated, f"{name} validation"),
        "test": assert_ranking_preserved(test_probabilities, test_calibrated, f"{name} test"),
    }
    values, labels = _probability_truth(validation_probabilities, validation_truth, f"{name} validation")
    return {
        "calibrator": calibrator,
        "fit": {
            "split": "validation",
            "rows": int(values.size),
            "positives": int(labels.sum()),
            "negatives": int(values.size - int(labels.sum())),
            "distinct_scores": int(np.unique(values).size),
            "calibrated_min": float(validation_calibrated.min()),
            "calibrated_max": float(validation_calibrated.max()),
        },
        "ranking_checks": ranking,
        "test_before_calibration": calibration_metrics(test_probabilities, test_truth),
        "test_after_calibration": calibration_metrics(test_calibrated, test_truth),
    }


def _model_output(result: Mapping[str, Any], calibrator_path: Path) -> dict[str, Any]:
    """Returns one model's calibration record for calibration_data.json."""
    calibrator = result["calibrator"]
    return {
        "calibrator": {
            "path": str(calibrator_path),
            "sha256": _sha256_file(calibrator_path),
            "x_thresholds": list(calibrator.x_thresholds),
            "y_thresholds": list(calibrator.y_thresholds),
            "y_min": calibrator.y_min,
            "y_max": calibrator.y_max,
            "application": APPLICATION_RULE,
        },
        "fit_diagnostics": result["fit"],
        "ranking_checks": result["ranking_checks"],
        "test": {
            "before_calibration": result["test_before_calibration"],
            "after_calibration": result["test_after_calibration"],
        },
    }


def _console_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the short console summary of one calibration run."""
    models = data["models"]
    return {
        "account": {
            "brier_before": models["account"]["test"]["before_calibration"]["brier"],
            "brier_after": models["account"]["test"]["after_calibration"]["brier"],
            "ece_before": models["account"]["test"]["before_calibration"]["ece"],
            "ece_after": models["account"]["test"]["after_calibration"]["ece"],
            "ranking": models["account"]["ranking_checks"],
        },
        "payment": {
            "brier_before": models["payment"]["test"]["before_calibration"]["brier"],
            "brier_after": models["payment"]["test"]["after_calibration"]["brier"],
            "ece_before": models["payment"]["test"]["before_calibration"]["ece"],
            "ece_after": models["payment"]["test"]["after_calibration"]["ece"],
            "ranking": models["payment"]["ranking_checks"],
        },
        "task4_reproduction": data["task4_reproduction"],
        "peak_committed_gb": data["peak_committed_gb"],
        "wall_seconds": data["wall_seconds"],
        "out_root": data["out_root"],
    }


def run(
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    out_root: Path = DEFAULT_OUT_ROOT,
) -> dict[str, Any]:
    """
    WHAT: Fits both calibrators on validation, checks them on test and writes the three output files.

    WHY: One entry point keeps the frozen-artifact checks, the task 4 reproduction, the fit and the
    reported checks in one run, so every number in calibration_data.json is traceable to the
    artifacts that produced it.

    Raises:
        ValueError: If a folder records another split, a frozen artifact differs, the payment test
            split does not reproduce task 4, or any calibration check fails.
    """
    validation_datasets = _split_datasets(validation_dirs, "validation")
    test_datasets = _split_datasets(test_dirs, "test")
    started = time.perf_counter()

    recipe = load_frozen_recipe()
    checkpoint_hashes = verify_account_checkpoint(recipe.checkpoint_dir.parent)
    account_model = PUBaggedAccountModel.load(recipe.checkpoint_dir)
    if account_model.n_bags != recipe.n_bags:
        raise ValueError(f"loaded account ensemble has {account_model.n_bags} bags, recipe says {recipe.n_bags}")
    if list(account_model.member_seeds) != list(range(recipe.seed, recipe.seed + recipe.n_bags)):
        raise ValueError("loaded account ensemble member seeds are not the frozen seed block")

    model_sha256, feature_file_sha256 = verify_frozen_artifacts(DEFAULT_MODEL_DIR)
    feature_order = load_feature_order(DEFAULT_MODEL_DIR / "feature_order.yaml")
    features = feature_order["features"]
    encodings = feature_order["categorical_encodings"]
    if not isinstance(features, list) or not isinstance(encodings, dict):
        raise ValueError(f"{DEFAULT_MODEL_DIR} holds malformed feature-order data")
    metadata = json.loads((DEFAULT_MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or feature_order["feature_sha256"] != metadata.get("feature_sha256")
        or len(features) != metadata.get("feature_count")
    ):
        raise ValueError("frozen feature order does not match model_metadata.json")
    payment_model = xgboost.XGBClassifier()
    payment_model.load_model(str(DEFAULT_MODEL_DIR / "frozen_model.json"))
    booster = payment_model.get_booster()
    if booster.num_features() != len(features):
        raise ValueError(f"frozen model takes {booster.num_features()} features but the order holds {len(features)}")
    rounds = verify_booster_rounds(booster, metadata)

    sampler = _CommitSampler()
    with sampler:
        loading_started = time.perf_counter()
        account_validation_scores, account_validation_truth, account_validation_diagnostics = account_split_rows(
            validation_datasets, "validation", account_model, DEFAULT_TRUTH_ROOT
        )
        payment_validation_scores, payment_validation_truth, payment_validation_diagnostics = payment_validation_rows(
            validation_datasets, features, encodings, payment_model, DEFAULT_OOF_ROOT
        )
        account_test_scores, account_test_truth, account_test_diagnostics = account_split_rows(
            test_datasets, "test", account_model, DEFAULT_TRUTH_ROOT
        )
        payment_test_scores, payment_test_truth, payment_test_diagnostics = payment_test_rows(
            test_datasets,
            features,
            encodings,
            payment_model,
            DEFAULT_TRUTH_ROOT,
            account_model,
            recipe.config_path,
            DEFAULT_SCORING_RESULT,
            EXPECTED_TEST_ROWS,
            EXPECTED_TEST_POSITIVES,
        )
        load_seconds = time.perf_counter() - loading_started
        fit_started = time.perf_counter()
        account = _calibrate_model(
            "account",
            account_validation_scores,
            account_validation_truth,
            account_test_scores,
            account_test_truth,
        )
        payment = _calibrate_model(
            "payment",
            payment_validation_scores,
            payment_validation_truth,
            payment_test_scores,
            payment_test_truth,
        )
        fit_seconds = time.perf_counter() - fit_started
    peak = sampler.peak_gb
    wall_seconds = time.perf_counter() - started

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    account_path = out_root / "account_isotonic.json"
    payment_path = out_root / "payment_isotonic.json"
    write_calibrator(account_path, account["calibrator"])
    write_calibrator(payment_path, payment["calibrator"])

    data: dict[str, Any] = {
        "amendment": "A-060",
        "amendments": list(AMENDMENTS),
        "simulated": True,
        "git_hash": _git_head(),
        "datasets": {
            "validation": {
                "folders": [dataset.name for dataset in validation_datasets],
                "content_hash": _content_hashes(validation_datasets, "validation"),
            },
            "test": {
                "folders": [dataset.name for dataset in test_datasets],
                "content_hash": _content_hashes(test_datasets, "test"),
            },
        },
        "models": {
            "account": {
                **_model_output(account, account_path),
                "truth": {
                    "validation": "hidden role at the kept snapshot's world minute: mule role = 1, genuine role = 0",
                    "test": "hidden role at the kept snapshot's world minute: mule role = 1, genuine role = 0",
                },
                "splits": {
                    "validation": account_validation_diagnostics,
                    "test": account_test_diagnostics,
                },
                "recipe": {
                    "config": str(recipe.config_path),
                    "config_sha256": recipe.config_sha256,
                    "checkpoint": str(recipe.checkpoint_dir),
                    "seed": recipe.seed,
                    "n_bags": recipe.n_bags,
                    "n_estimators": recipe.n_estimators,
                    "feature_sha256": recipe.feature_sha256,
                    "feature_count": len(recipe.feature_names),
                },
                "checkpoint_sha256": checkpoint_hashes,
            },
            "payment": {
                **_model_output(payment, payment_path),
                "truth": {
                    "validation": "the target column derived from each validation dataset's labels.parquet",
                    "test": "the receiver has a mule role on the payment's decision day, the task 4 label",
                },
                "splits": {
                    "validation": payment_validation_diagnostics,
                    "test": payment_test_diagnostics,
                },
                "model": {
                    "path": str(DEFAULT_MODEL_DIR / "frozen_model.json"),
                    "sha256": model_sha256,
                    "feature_order_path": str(DEFAULT_MODEL_DIR / "feature_order.yaml"),
                    "feature_order_file_sha256": feature_file_sha256,
                    "feature_sha256": feature_order["feature_sha256"],
                    "feature_count": len(features),
                    "rounds": rounds,
                    "seed": metadata.get("seed"),
                    "device": metadata.get("device"),
                },
            },
        },
        "task4_reproduction": payment_test_diagnostics["reproduction"],
        "load_seconds": load_seconds,
        "fit_seconds": fit_seconds,
        "wall_seconds": wall_seconds,
        "peak_committed_gb": peak,
        "out_root": str(out_root),
    }
    (out_root / "calibration_data.json").write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(_console_summary(data), indent=2), flush=True)
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """Fits and checks the isotonic calibrators, or loads a written calibrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Fit the calibrators on validation and check them on test")
    run_parser.add_argument(
        "--validation", action="append", required=True, help="Validation dataset folder (repeatable)"
    )
    run_parser.add_argument("--test", action="append", required=True, help="Test dataset folder (repeatable)")
    run_parser.add_argument(
        "--out-root", default=str(DEFAULT_OUT_ROOT), help="Output folder for the three result files"
    )
    args = parser.parse_args(argv)
    run([Path(path) for path in args.validation], [Path(path) for path in args.test], Path(args.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
