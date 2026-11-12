"""
Unit tests for decisions.anomaly_detector: matrix construction, isolation forest training,
scoring, determinism, early-window filtering, missingness indicators, and cryptographic serialization.
"""

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from decisions.anomaly_config import input_feature_names, load_anomaly_detector_config
from decisions.anomaly_detector import (
    build_feature_matrix,
    load_detector,
    save_detector,
    score_rows,
    train_anomaly_detector,
)

T0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
WORLD_ID = "world_001"
WORLD_STARTS = {WORLD_ID: T0}


def _make_synth_row(
    row_idx: int,
    features: list[str],
    snapshot_time: datetime,
    is_extreme: bool = False,
    none_features: set[str] | None = None,
) -> dict[str, Any]:
    """Helper creating a synthetic snapshot row shaped like Level 16 dataset tables."""
    row: dict[str, Any] = {
        "world_id": WORLD_ID,
        "account_id": f"acc_{row_idx:06d}",
        "snapshot_time": snapshot_time,
    }
    rng = np.random.default_rng(seed=1000 + row_idx)
    for feat in features:
        if none_features and feat in none_features:
            row[feat] = None
        elif is_extreme:
            row[feat] = 100000000.0  # Massive outlier value across all features
        else:
            # Baseline normal behavior: modest randomized amounts
            row[feat] = float(rng.uniform(10.0, 500.0))
    return row


def test_anomaly_detector_train_and_extreme_planted_row_has_highest_score() -> None:
    """
    Test (a):
    Trains on 300 ordinary rows + 1 planted extreme row.
    Asserts the planted outlier row has the strictly highest anomaly score.
    """
    config = load_anomaly_detector_config()
    features = list(input_feature_names(config))

    rows: list[dict[str, Any]] = []
    # 300 ordinary rows inside the early window
    for i in range(300):
        t = T0 + timedelta(days=1, hours=i % 24)
        rows.append(_make_synth_row(i, features, t, is_extreme=False))

    # 1 planted extreme row (index 300)
    extreme_t = T0 + timedelta(days=2)
    rows.append(_make_synth_row(300, features, extreme_t, is_extreme=True))

    detector = train_anomaly_detector(rows, split="training", world_start_by_world=WORLD_STARTS, config=config)
    scores = score_rows(detector, rows)

    matrix_cols = len(detector.column_names)
    top_idx = int(np.argmax(scores))

    print(f"\n[EVIDENCE] matrix column count in test (a): {matrix_cols}")
    print(f"[EVIDENCE] top score row index in test (a): {top_idx}")

    assert top_idx == 300
    assert scores[300] == max(scores)
    # The extreme row score should strictly exceed the 99th percentile of normal scores
    normal_p99 = np.percentile(scores[:300], 99)
    assert scores[300] > normal_p99


def test_anomaly_detector_training_determinism() -> None:
    """
    Test (b):
    Verifies that training twice on the exact same rows produces bit-for-bit identical scores.
    """
    config = load_anomaly_detector_config()
    features = list(input_feature_names(config))

    rows: list[dict[str, Any]] = []
    for i in range(50):
        t = T0 + timedelta(days=1, minutes=i * 10)
        rows.append(_make_synth_row(i, features, t))

    det1 = train_anomaly_detector(rows, split="training", world_start_by_world=WORLD_STARTS, config=config)
    det2 = train_anomaly_detector(rows, split="training", world_start_by_world=WORLD_STARTS, config=config)

    scores1 = score_rows(det1, rows)
    scores2 = score_rows(det2, rows)

    assert scores1 == scores2


@pytest.mark.parametrize("forbidden_split", ["test", "stress"])
def test_anomaly_detector_refuses_test_and_stress_splits(forbidden_split: str) -> None:
    """
    Test (c):
    Verifies that train_anomaly_detector raises ValueError when split is 'test' or 'stress'.
    """
    features = list(input_feature_names())
    rows = [_make_synth_row(i, features, T0 + timedelta(days=1)) for i in range(5)]

    with pytest.raises(ValueError, match="is not permitted for anomaly detector training"):
        train_anomaly_detector(rows, split=forbidden_split, world_start_by_world=WORLD_STARTS)


def test_anomaly_detector_filters_rows_after_early_window() -> None:
    """
    Test (d):
    Verifies that rows occurring after world_start + early_window_days are excluded from training.
    """
    config = load_anomaly_detector_config()
    features = list(input_feature_names(config))

    early_rows = [_make_synth_row(i, features, T0 + timedelta(days=2)) for i in range(12)]
    late_rows = [_make_synth_row(100 + i, features, T0 + timedelta(days=20)) for i in range(8)]
    all_rows = early_rows + late_rows

    detector = train_anomaly_detector(all_rows, split="training", world_start_by_world=WORLD_STARTS, config=config)

    print(f"\n[EVIDENCE] training row count in test (d): {detector.training_row_count}")
    assert detector.training_row_count == 12


def test_missing_values_filled_with_median_and_indicator_is_one() -> None:
    """
    Test (e):
    Verifies that None feature values are imputed with median and the corresponding
    <feature>__missing indicator is set to 1.0 for that row.
    """
    features = ["current_balance_paise", "pass_through_ratio_24h"]
    medians = {"current_balance_paise": 5000.0, "pass_through_ratio_24h": 0.5}

    row0 = {"current_balance_paise": 1000.0, "pass_through_ratio_24h": None}
    row1 = {"current_balance_paise": 9000.0, "pass_through_ratio_24h": 0.8}

    matrix, column_names = build_feature_matrix([row0, row1], features, medians)

    assert "pass_through_ratio_24h__missing" in column_names
    missing_idx = column_names.index("pass_through_ratio_24h__missing")
    val_idx = column_names.index("pass_through_ratio_24h")

    # Row 0 had None: value must equal median (0.5), indicator must equal 1.0
    assert matrix[0, val_idx] == 0.5
    assert matrix[0, missing_idx] == 1.0

    # Row 1 had 0.8: value must equal 0.8, indicator must equal 0.0
    assert matrix[1, val_idx] == 0.8
    assert matrix[1, missing_idx] == 0.0


def test_save_and_load_detector_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    """
    Test (f):
    Verifies detector save and load roundtrip yields identical scores,
    and modifying 1 byte of the serialized model file causes load_detector to raise ValueError.
    """
    config = load_anomaly_detector_config()
    features = list(input_feature_names(config))
    rows = [_make_synth_row(i, features, T0 + timedelta(days=1)) for i in range(25)]

    detector = train_anomaly_detector(rows, split="training", world_start_by_world=WORLD_STARTS, config=config)
    original_scores = score_rows(detector, rows)

    save_dir = tmp_path / "detector_export"
    save_detector(detector, save_dir)

    loaded_detector = load_detector(save_dir)
    loaded_scores = score_rows(loaded_detector, rows)
    assert original_scores == loaded_scores

    # Tamper with 1 byte of the serialized model file
    model_file = save_dir / "isolation_forest.joblib"
    raw_bytes = bytearray(model_file.read_bytes())
    raw_bytes[10] ^= 0xFF
    model_file.write_bytes(raw_bytes)

    with pytest.raises(ValueError, match="Cryptographic hash mismatch for model file"):
        load_detector(save_dir)


def test_train_anomaly_detector_has_no_label_parameters() -> None:
    """
    Test (g):
    Verifies that train_anomaly_detector's signature contains zero parameters with 'label'.
    """
    sig = inspect.signature(train_anomaly_detector)
    for param_name in sig.parameters:
        assert "label" not in param_name.lower(), f"Forbidden parameter '{param_name}' contains 'label'"


def test_load_detector_raises_on_sklearn_version_mismatch(tmp_path: Path) -> None:
    """Verifies that load_detector raises ValueError mentioning version on mismatch."""
    config = load_anomaly_detector_config()
    features = list(input_feature_names(config))
    rows = [_make_synth_row(i, features, T0 + timedelta(days=1)) for i in range(10)]

    detector = train_anomaly_detector(rows, split="training", world_start_by_world=WORLD_STARTS, config=config)
    save_dir = tmp_path / "detector_version_test"
    save_detector(detector, save_dir)

    manifest_file = save_dir / "manifest.json"
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["sklearn_version"] = "0.0.0"
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="0.0.0"):
        load_detector(save_dir)


def test_all_missing_features_recorded_when_feature_is_none_in_every_row(tmp_path: Path) -> None:
    """Verifies that features with None in every kept row are recorded in all_missing_features."""
    config = load_anomaly_detector_config()
    features = list(input_feature_names(config))
    target_feature = features[0]

    rows = [
        _make_synth_row(i, features, T0 + timedelta(days=1), none_features={target_feature})
        for i in range(10)
    ]

    detector = train_anomaly_detector(rows, split="training", world_start_by_world=WORLD_STARTS, config=config)
    assert target_feature in detector.all_missing_features
    assert detector.medians[target_feature] == 0.0

    print(f"\n[EVIDENCE] all_missing_features count in commit 1 test: {len(detector.all_missing_features)}")

    # Verify roundtrip through save and load
    save_dir = tmp_path / "all_missing_export"
    save_detector(detector, save_dir)
    loaded = load_detector(save_dir)
    assert loaded.all_missing_features == detector.all_missing_features

