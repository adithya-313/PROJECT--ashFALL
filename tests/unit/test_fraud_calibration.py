# File: tests/unit/test_fraud_calibration.py
# Purpose: Checks the 5.5 calibration helpers: bin check, [0,1] and monotonic checks, threshold selection rules,
#          and that a saved XGBoost model reloads with identical scores.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.lazy_script import lazy_script

pytestmark = pytest.mark.ml  # needs the requirements-ml.txt libraries: CI's fast job leaves it out, the ml job runs it

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "fraud_calibration.py"
fraud_calibration = lazy_script(SCRIPT_PATH, ("mlflow", "optuna", "xgboost", "shap", "joblib", "yaml"))


def make_calibrated_data(rows: int = 50_000):
    """Scores whose true fraud rate equals the score, so the data is perfectly calibrated by construction."""
    rng = np.random.default_rng(0)
    probability = rng.random(rows)
    y = pd.Series((rng.random(rows) < probability).astype(int))
    return y, probability


def test_well_calibrated_scores_pass_the_ten_point_check():
    y, probability = make_calibrated_data()
    rows = fraud_calibration.reliability_table(y, probability)
    assert len(rows) == 10
    assert fraud_calibration.check_bins(rows)


def test_badly_calibrated_scores_fail_the_ten_point_check():
    y, probability = make_calibrated_data()
    rows = fraud_calibration.reliability_table(y, np.clip(probability * 0.3, 0, 1) + 0.6)  # shifted far from truth
    assert not fraud_calibration.check_bins(rows)


def test_empty_bins_are_reported_but_not_judged():
    y = pd.Series([0, 1, 0, 1])
    rows = fraud_calibration.reliability_table(y, np.array([0.05, 0.95, 0.04, 0.96]))
    empty = [r for r in rows if r["count"] == 0]
    assert empty and all(r["deviation"] is None for r in empty)
    assert sum(r["count"] for r in rows) == 4


def test_isotonic_output_is_in_range_and_monotonic():
    y, raw = make_calibrated_data()
    calibrator = fraud_calibration.fit_calibrator(raw, y)
    checks = fraud_calibration.check_calibrated_probabilities(calibrator, raw)
    assert checks["in_range"] and checks["monotonic"]


def test_threshold_rules_pick_the_right_points():
    rows = [
        {"threshold": 0.1, "precision": 0.5, "recall": 1.00, "f1": 0.667},
        {"threshold": 0.3, "precision": 0.8, "recall": 0.996, "f1": 0.887},
        {"threshold": 0.5, "precision": 0.9, "recall": 0.90, "f1": 0.900},
        {"threshold": 0.7, "precision": 0.95, "recall": 0.60, "f1": 0.735},
    ]
    assert fraud_calibration.choose_f1_optimal(rows)["threshold"] == 0.5
    # Highest threshold that keeps recall >= 0.995 is 0.3 (0.5 drops to 0.90).
    assert fraud_calibration.choose_high_recall(rows)["threshold"] == 0.3
    assert fraud_calibration.choose_high_recall(rows[2:]) is None


def test_metrics_at_counts_are_exact():
    y = pd.Series([1, 1, 0, 0, 0, 1])
    probability = np.array([0.9, 0.4, 0.6, 0.1, 0.2, 0.8])
    row = fraud_calibration.metrics_at(y, probability, 0.5)
    assert (row["tp"], row["fp"], row["fn"]) == (2, 1, 1)
    assert row["precision"] == pytest.approx(2 / 3) and row["recall"] == pytest.approx(2 / 3)


def test_saved_model_reloads_with_identical_scores(tmp_path, monkeypatch):
    import xgboost

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((500, 4)), columns=list("abcd"))
    y = (X["a"] > 0.6).astype(int)
    model = xgboost.XGBClassifier(n_estimators=20, tree_method="hist", device="cpu", random_state=42).fit(X, y)

    monkeypatch.setattr(fraud_calibration, "MODEL_PATH", tmp_path / "model.ubj")
    model.save_model(str(fraud_calibration.MODEL_PATH))
    reloaded = fraud_calibration.load_saved_model()
    assert np.abs(model.predict_proba(X)[:, 1] - reloaded.predict_proba(X)[:, 1]).max() <= 1e-6


def test_frozen_output_paths():
    assert fraud_calibration.CALIBRATOR_PATH.as_posix().endswith("models/calibration/isotonic_calibrator.pkl")
    assert fraud_calibration.MODEL_PATH.as_posix().endswith("models/xgboost/model.ubj")
    assert fraud_calibration.RELIABILITY_PLOT_PATH.as_posix().endswith("reports/calibration/reliability_diagram.png")
