# File: tests/unit/test_fraud_onnx_export.py
# Purpose: Checks the 5.6 ONNX export helpers on a tiny model: conversion works, the fraud probability column is
#          read correctly, scores match native within 1e-5, ONNX Runtime loads on CPU, and a too-large gap fails.
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

for library in ("mlflow", "optuna", "xgboost", "shap", "onnxruntime", "onnxmltools", "yaml"):
    pytest.importorskip(library)

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "fraud_onnx_export.py"
spec = importlib.util.spec_from_file_location("fraud_onnx_export", SCRIPT_PATH)
fraud_onnx_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fraud_onnx_export)

FEATURES = [f"feature_{i}" for i in range(15)]


@pytest.fixture
def tiny_model(tmp_path):
    """A tiny trained 15-feature model saved to disk with real feature names, plus the data it was trained on."""
    import xgboost

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.random((2000, 15)) * 1000, columns=FEATURES)
    y = ((X["feature_0"] + X["feature_3"]) > 1200).astype(int)
    model = xgboost.XGBClassifier(n_estimators=30, max_depth=4, tree_method="hist", device="cpu", random_state=42)
    model.fit(X, y)
    path = tmp_path / "model.ubj"
    model.save_model(str(path))
    return path, X


def export_to(path: Path, onnx_path: Path) -> None:
    """Convert the saved model at `path` and write the ONNX file to `onnx_path`."""
    model = fraud_onnx_export.load_model_for_conversion(path)
    fraud_onnx_export.save_onnx(fraud_onnx_export.convert_to_onnx(model), onnx_path)


def test_converted_model_matches_native_within_tolerance(tiny_model, tmp_path):
    import xgboost

    path, X = tiny_model
    onnx_path = tmp_path / "model.onnx"
    export_to(path, onnx_path)
    session = fraud_onnx_export.open_session(onnx_path)

    native_model = xgboost.XGBClassifier()
    native_model.load_model(str(path))
    fraud_onnx_export.check_class_order(native_model)
    native = native_model.predict_proba(X)[:, 1]
    summary = fraud_onnx_export.compare_scores(native, fraud_onnx_export.onnx_fraud_probability(session, X))
    fraud_onnx_export.assert_within_tolerance(summary)
    assert summary["max_abs_diff"] <= 1e-5 and summary["rows_over_tolerance"] == 0


def test_onnx_runtime_uses_cpu_provider(tiny_model, tmp_path):
    path, _ = tiny_model
    onnx_path = tmp_path / "model.onnx"
    export_to(path, onnx_path)
    assert fraud_onnx_export.open_session(onnx_path).get_providers() == ["CPUExecutionProvider"]


def test_conversion_does_not_change_the_saved_model_feature_names(tiny_model):
    import xgboost

    path, _ = tiny_model
    fraud_onnx_export.load_model_for_conversion(path)
    reloaded = xgboost.XGBClassifier()
    reloaded.load_model(str(path))
    assert reloaded.get_booster().feature_names == FEATURES


def test_a_gap_larger_than_the_tolerance_fails_the_assertion():
    native = np.array([0.1, 0.5, 0.9])
    off_by_a_little = native + np.array([0.0, 2e-5, 0.0])
    summary = fraud_onnx_export.compare_scores(native, off_by_a_little)
    assert summary["rows_over_tolerance"] == 1
    with pytest.raises(AssertionError):
        fraud_onnx_export.assert_within_tolerance(summary)


def test_frozen_onnx_path():
    assert fraud_onnx_export.ONNX_PATH.as_posix().endswith("models/xgboost/model.onnx")
