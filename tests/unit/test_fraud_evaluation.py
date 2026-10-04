# File: tests/unit/test_fraud_evaluation.py
# Purpose: Checks the 5.4 evaluation helpers: type conversion of tuned params, winner rule, segment scoring with
#          'undefined' where no fraud exists, and that a SHAP explainer pickled with joblib loads back.
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

for library in ("mlflow", "optuna", "xgboost", "catboost", "shap", "joblib", "yaml"):
    pytest.importorskip(library)

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "fraud_evaluation.py"
spec = importlib.util.spec_from_file_location("fraud_evaluation", SCRIPT_PATH)
fraud_evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fraud_evaluation)


def test_convert_param_gives_real_types():
    convert = fraud_evaluation.convert_param
    assert convert("5") == 5 and isinstance(convert("5"), int)
    assert convert("0.1160184349121161") == pytest.approx(0.1160184349121161)
    assert convert("0.5") == 0.5 and isinstance(convert("0.5"), float)  # random forest max_features fraction
    assert convert("sqrt") == "sqrt" and convert("log2") == "log2"
    assert convert("None") is None


def test_winner_is_the_highest_pr_auc():
    results = {
        "xgboost": {"pr_auc": 0.90, "roc_auc": 0.99},
        "catboost": {"pr_auc": 0.95, "roc_auc": 0.98},
        "random_forest": {"pr_auc": 0.93, "roc_auc": 0.999},
    }
    assert fraud_evaluation.select_winner(results) == "catboost"


def test_exact_pr_auc_tie_is_broken_by_roc_auc():
    results = {
        "xgboost": {"pr_auc": 0.95, "roc_auc": 0.97},
        "catboost": {"pr_auc": 0.95, "roc_auc": 0.99},
    }
    assert fraud_evaluation.select_winner(results) == "catboost"


def test_segment_without_fraud_is_undefined_not_invented():
    y = pd.Series([0, 0, 0, 1, 1, 0])
    probability = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.2])
    no_fraud_rows = pd.Series([True, True, True, False, False, True])
    row = fraud_evaluation.segment_scores(y, probability, no_fraud_rows, "Type: PAYMENT")
    assert row["pr_auc"] is None and row["roc_auc"] is None
    assert "undefined" in row["note"]
    assert row["rows"] == 4 and row["fraud"] == 0


def test_segment_with_both_classes_has_real_numbers():
    y = pd.Series([0, 0, 0, 1, 1, 0])
    probability = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.2])
    row = fraud_evaluation.segment_scores(y, probability, pd.Series(True, index=y.index), "Overall")
    assert row["pr_auc"] == pytest.approx(1.0) and row["roc_auc"] == pytest.approx(1.0)


def test_frozen_shap_path_is_exact():
    assert fraud_evaluation.SHAP_PICKLE_PATH.as_posix().endswith("models/xgboost/shap_model.pkl")


def test_pickled_tree_explainer_loads_back(tmp_path):
    import joblib
    import shap
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((300, 4)), columns=list("abcd"))
    y = (X["a"] > 0.7).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    explainer = shap.TreeExplainer(model)

    path = tmp_path / "shap_model.pkl"
    joblib.dump(explainer, path)
    reloaded = joblib.load(path)
    assert isinstance(reloaded, shap.TreeExplainer)
    values = fraud_evaluation.positive_class_shap_values(reloaded.shap_values(X.iloc[:20]))
    assert values.shape == (20, 4)
