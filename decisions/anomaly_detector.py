# File: decisions/anomaly_detector.py
# Purpose: Unsupervised anomaly detection via Isolation Forest trained on early unlabelled transaction snapshots.
# Issue: Level 17b Round 1 anomaly detector and cold-start policy.
"""
WHAT:
    Implements feature matrix construction with missingness indicators, Isolation Forest model
    training on early unlabelled traffic, anomaly scoring, and cryptographic artifact serialization.

WHY:
    In cold-start environments before verified labels exist, unsupervised anomaly detection
    identifies accounts with outlier behavioral velocity. By scoring with -score_samples(X),
    higher scores indicate higher anomalousness without depending on ground-truth labels.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from decisions.anomaly_config import (
    AnomalyDetectorConfig,
    input_feature_names,
    load_anomaly_detector_config,
)

__all__ = [
    "TrainedAnomalyDetector",
    "build_feature_matrix",
    "load_detector",
    "save_detector",
    "score_rows",
    "train_anomaly_detector",
]


@dataclass(frozen=True)
class TrainedAnomalyDetector:
    """
    PURPOSE:
        Encapsulates a fitted Isolation Forest detector, feature metadata, training provenance, and digests.

    ATTRIBUTES:
        model: Fitted sklearn.ensemble.IsolationForest instance.
        feature_names: Tuple of canonical input feature names.
        column_names: Tuple of matrix column names including missing indicators.
        missing_features: Tuple of feature names that required a missingness indicator during training.
        medians: Mapping of feature names to their training imputation medians.
        training_row_count: Number of early unlabelled snapshot rows used to fit the model.
        training_rows_sha256: Cryptographic SHA-256 digest of training rows.
        model_sha256: SHA-256 digest of serialized model weights.
        sklearn_version: Version string of scikit-learn used during training.
    """

    model: IsolationForest
    feature_names: tuple[str, ...]
    column_names: tuple[str, ...]
    missing_features: tuple[str, ...]
    medians: dict[str, float]
    training_row_count: int
    training_rows_sha256: str
    model_sha256: str
    sklearn_version: str


def build_feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    medians: Mapping[str, float],
    missing_features: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    PURPOSE:
        Converts snapshot dictionaries into a dense numeric 2D numpy array with missingness indicators.

    INPUTS:
        rows: Sequence of dictionary records (each containing account features).
        feature_names: Ordered sequence of canonical feature keys to extract.
        medians: Imputation median values for each feature when a value is None.
        missing_features: Optional sequence of feature names that must produce a <name>__missing column.
                          If None, derived dynamically from features having None in rows.

    KEY LOGIC:
        1. Identifies features requiring missingness indicators.
        2. Iterates over rows and features, imputing None with median and coercing booleans to 0/1.
        3. Appends a 0/1 missing indicator column for every feature in missing_features.
        4. Rejects any non-numeric and non-boolean value with ValueError (no silent fallbacks).

    RETURNS:
        tuple[np.ndarray, list[str]]: (2D float64 numpy array, list of column names).

    ERRORS:
        Raises ValueError: If any feature value is a string, list, or other non-numeric type.
    """
    if missing_features is None:
        active_missing: list[str] = [
            f for f in feature_names if any(row.get(f) is None for row in rows)
        ]
    else:
        active_missing = list(missing_features)

    column_names = list(feature_names) + [f"{f}__missing" for f in active_missing]
    n_rows = len(rows)
    n_cols = len(column_names)
    matrix = np.empty((n_rows, n_cols), dtype=np.float64)

    for i, row in enumerate(rows):
        # 1. Base feature values
        for j, feat in enumerate(feature_names):
            val = row.get(feat)
            if val is None:
                matrix[i, j] = medians[feat]
            elif isinstance(val, bool):
                matrix[i, j] = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                matrix[i, j] = float(val)
            else:
                raise ValueError(
                    f"Invalid non-numeric value {val!r} for feature '{feat}' in row {i}; expected number or bool."
                )

        # 2. Missing indicator columns (0 or 1)
        offset = len(feature_names)
        for k, feat in enumerate(active_missing):
            val = row.get(feat)
            matrix[i, offset + k] = 1.0 if val is None else 0.0

    return matrix, column_names


def _compute_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Computes a deterministic SHA-256 hex digest of serialized dataset rows."""
    serialized_items: list[str] = []
    for r in rows:
        filtered = {k: v for k, v in r.items() if not isinstance(v, (datetime, bytes))}
        serialized_items.append(json.dumps(filtered, sort_keys=True, default=str))
    serialized_items.sort()
    combined = "\n".join(serialized_items).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def _ensure_utc(dt: datetime) -> datetime:
    """Normalizes datetime to UTC timezone for accurate epoch comparison."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def train_anomaly_detector(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    world_start_by_world: Mapping[str, datetime],
    config: AnomalyDetectorConfig | None = None,
) -> TrainedAnomalyDetector:
    """
    PURPOSE:
        Trains an unsupervised IsolationForest model using only early unlabelled account snapshot rows.

    INPUTS:
        rows: Sequence of snapshot dictionaries.
        split: Dataset split identifier (e.g. 'training').
        world_start_by_world: Mapping of world_id to world start timestamp.
        config: Optional pre-loaded AnomalyDetectorConfig. If None, default config is loaded.

    KEY LOGIC:
        1. Enforces split constraint: split must be in config.allowed_training_splits.
        2. Filters rows strictly to snapshot_time < world_start + early_window_days.
        3. Enforces sample size constraint: refuses if fewer than 2 rows remain.
        4. Computes feature medians exclusively from the filtered early rows.
        5. Builds numeric feature matrix with missing indicators and fits IsolationForest.
        6. Note: Takes NO labels parameter and inspects zero label data.

    RETURNS:
        TrainedAnomalyDetector: Container with fitted model, imputation medians, and provenance.

    ERRORS:
        Raises ValueError: If split is not allowed, or if fewer than 2 rows remain after filtering.
    """
    if config is None:
        config = load_anomaly_detector_config()

    if split not in config.allowed_training_splits:
        raise ValueError(
            f"Split '{split}' is not permitted for anomaly detector training; "
            f"allowed splits: {config.allowed_training_splits}"
        )

    early_window = timedelta(days=config.early_window_days)
    kept_rows: list[Mapping[str, Any]] = []

    for row in rows:
        world_id = row.get("world_id")
        if not world_id or world_id not in world_start_by_world:
            raise ValueError(f"world_id '{world_id}' not found in world_start_by_world mapping.")

        world_start = _ensure_utc(world_start_by_world[world_id])

        raw_snapshot_time = row.get("snapshot_time")
        if raw_snapshot_time is None:
            raise ValueError("Row missing 'snapshot_time' required for early window filtering.")

        if isinstance(raw_snapshot_time, str):
            snapshot_dt = datetime.fromisoformat(raw_snapshot_time)
        elif isinstance(raw_snapshot_time, datetime):
            snapshot_dt = raw_snapshot_time
        else:
            raise ValueError(f"Unsupported snapshot_time type: {type(raw_snapshot_time)}")

        snapshot_dt = _ensure_utc(snapshot_dt)

        # Retain only early unlabelled records
        if snapshot_dt < world_start + early_window:
            kept_rows.append(row)

    if len(kept_rows) < 2:
        raise ValueError(
            f"Fewer than 2 snapshot rows remain after early window filtering; got {len(kept_rows)} rows."
        )

    feature_names = input_feature_names(config)

    # Compute imputation medians strictly from kept rows
    medians: dict[str, float] = {}
    for feat in feature_names:
        present_values: list[float] = []
        for r in kept_rows:
            v = r.get(feat)
            if v is not None:
                if isinstance(v, bool):
                    present_values.append(1.0 if v else 0.0)
                elif isinstance(v, (int, float)):
                    present_values.append(float(v))
                else:
                    raise ValueError(f"Non-numeric value {v!r} for feature '{feat}'.")

        if present_values:
            medians[feat] = float(np.median(present_values))
        else:
            # Fallback for features that are entirely None across early rows
            medians[feat] = 0.0

    missing_features = tuple(
        f for f in feature_names if any(r.get(f) is None for r in kept_rows)
    )

    X_train, column_names = build_feature_matrix(
        kept_rows,
        feature_names,
        medians,
        missing_features=missing_features,
    )

    model = IsolationForest(
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,
        contamination=config.contamination,
        random_state=config.random_state,
    )
    model.fit(X_train)

    training_digest = _compute_rows_digest(kept_rows)

    return TrainedAnomalyDetector(
        model=model,
        feature_names=tuple(feature_names),
        column_names=tuple(column_names),
        missing_features=missing_features,
        medians=medians,
        training_row_count=len(kept_rows),
        training_rows_sha256=training_digest,
        model_sha256="",  # Computed upon file serialization
        sklearn_version=sklearn.__version__,
    )


def score_rows(
    detector: TrainedAnomalyDetector,
    rows: Sequence[Mapping[str, Any]],
) -> list[float]:
    """
    PURPOSE:
        Scores a sequence of snapshot rows producing continuous anomaly scores.

    INPUTS:
        detector: Fitted TrainedAnomalyDetector instance.
        rows: Sequence of snapshot dictionaries.

    KEY LOGIC:
        Calculates anomaly score as:
            anomaly_score = -model.score_samples(X)
        In scikit-learn, score_samples returns opposite values (lower = more abnormal, higher = normal).
        Negating score_samples inverts this ordering so that HIGHER scores represent MORE UNUSUAL / ANOMALOUS profiles.

    RETURNS:
        list[float]: Continuous anomaly score for each row in input sequence order.
    """
    if not rows:
        return []

    X, _ = build_feature_matrix(
        rows,
        detector.feature_names,
        detector.medians,
        missing_features=detector.missing_features,
    )

    raw_scores = detector.model.score_samples(X)
    # Higher anomaly score represents higher anomaly / risk
    anomaly_scores = -raw_scores
    return [float(score) for score in anomaly_scores]


def save_detector(detector: TrainedAnomalyDetector, directory: Path | str) -> None:
    """
    PURPOSE:
        Persists a trained anomaly detector and cryptographic manifest to disk.

    INPUTS:
        detector: TrainedAnomalyDetector instance to serialize.
        directory: Destination folder path.

    KEY LOGIC:
        1. Serializes model weights using joblib to 'isolation_forest.joblib'.
        2. Computes SHA-256 digest of the serialized model file.
        3. Writes 'manifest.json' recording scikit-learn version, feature columns,
           medians, training row count, and cryptographic digests.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / "isolation_forest.joblib"
    joblib.dump(detector.model, model_path)

    model_bytes = model_path.read_bytes()
    model_digest = hashlib.sha256(model_bytes).hexdigest()

    manifest = {
        "sklearn_version": detector.sklearn_version,
        "feature_column_list": list(detector.column_names),
        "feature_names": list(detector.feature_names),
        "missing_features": list(detector.missing_features),
        "medians": detector.medians,
        "training_row_count": detector.training_row_count,
        "training_rows_sha256": detector.training_rows_sha256,
        "model_sha256": model_digest,
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_detector(directory: Path | str) -> TrainedAnomalyDetector:
    """
    PURPOSE:
        Loads and verifies a persisted anomaly detector from disk.

    INPUTS:
        directory: Folder containing 'manifest.json' and 'isolation_forest.joblib'.

    KEY LOGIC:
        1. Reads manifest.json and asserts file existence.
        2. Validates scikit-learn version matches current runtime version.
        3. Recomputes SHA-256 digest of 'isolation_forest.joblib' and verifies against manifest.
        4. Deserializes model weights with joblib.

    RETURNS:
        TrainedAnomalyDetector: Verified detector ready for inference.

    ERRORS:
        Raises FileNotFoundError: If manifest.json or model file is missing.
        Raises ValueError: If sklearn version or model file cryptographic hash does not match manifest.
    """
    target_dir = Path(directory)
    manifest_path = target_dir / "manifest.json"
    model_path = target_dir / "isolation_forest.joblib"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file missing at: {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing at: {model_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Check scikit-learn version
    recorded_version = manifest_data.get("sklearn_version")
    current_version = sklearn.__version__
    if recorded_version != current_version:
        raise ValueError(
            f"scikit-learn version mismatch in {target_dir}: "
            f"manifest recorded {recorded_version}, but runtime is {current_version}"
        )

    # 2. Verify SHA-256 digest of the model file
    actual_model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected_model_sha256 = manifest_data.get("model_sha256")
    if actual_model_sha256 != expected_model_sha256:
        raise ValueError(
            f"Cryptographic hash mismatch for model file in {target_dir}: "
            f"expected {expected_model_sha256}, got {actual_model_sha256}"
        )

    model = joblib.load(model_path)

    return TrainedAnomalyDetector(
        model=model,
        feature_names=tuple(manifest_data["feature_names"]),
        column_names=tuple(manifest_data["feature_column_list"]),
        missing_features=tuple(manifest_data["missing_features"]),
        medians={k: float(v) for k, v in manifest_data["medians"].items()},
        training_row_count=int(manifest_data["training_row_count"]),
        training_rows_sha256=str(manifest_data["training_rows_sha256"]),
        model_sha256=actual_model_sha256,
        sklearn_version=current_version,
    )

