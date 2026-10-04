# File: scripts/fraud_onnx_export.py
# Purpose: Sub-phase 5.6 - export the saved winning XGBoost (models/xgboost/model.ubj) to ONNX and hard-assert that
#          the ONNX scores match the native XGBoost scores within 1e-5 on the full test split.
#
# What models/xgboost/model.onnx is: the winning XGBoost classifier, taking a float32 matrix of the 15 contract
# features (in contract order) and returning the fraud probability; it is the file Triton serves in sub-phase 5.7.
import sys
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime
import pandas as pd
import xgboost
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Make the repo root and this scripts/ folder importable, so the file works when run directly or loaded by tests.
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from fraud_baseline import read_dvc_data_hash  # noqa: E402
from fraud_calibration import MODEL_PATH, load_features  # noqa: E402

from core.mlflow_setup import setup_mlflow  # noqa: E402

TEST_PATH = REPO_ROOT / "data" / "processed" / "test.parquet"
ONNX_PATH = REPO_ROOT / "models" / "xgboost" / "model.onnx"

EXPERIMENT_NAME = "level5_fraud_onnx"
RANDOM_SEED = 42
N_FEATURES = 15
INPUT_NAME = "input"
PROBABILITY_OUTPUT = "probabilities"  # ONNX output holding one column per class: [non-fraud, fraud]
TOLERANCE = 1e-5  # the acceptance limit on the largest allowed difference between ONNX and native scores


def load_model_for_conversion(model_path: Path = MODEL_PATH) -> xgboost.XGBClassifier:
    """Load a fresh copy of the saved model, with its feature names cleared, ready for the ONNX converter."""
    model = xgboost.XGBClassifier()
    model.load_model(str(model_path))

    # The converter insists on feature names of the form f0, f1, ... but the saved model carries the real column
    # names (amount, oldbalanceOrg, ...). Clearing the names on this in-memory copy fixes that. Nothing changes
    # in the saved file, and the column ORDER is untouched: the ONNX model takes one plain matrix, so features are
    # matched by position, which is the same contract order used everywhere else.
    model.get_booster().feature_names = None
    return model


def convert_to_onnx(model: xgboost.XGBClassifier):
    """Convert the XGBoost classifier to an ONNX model taking a float32 matrix with 15 columns."""
    # target_opset is left at its default, which means the converter's own maximum supported version.
    return convert_xgboost(model, initial_types=[(INPUT_NAME, FloatTensorType([None, N_FEATURES]))])


def save_onnx(onnx_model, path: Path = ONNX_PATH) -> None:
    """Write the ONNX model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_model.SerializeToString())


def open_session(path: Path = ONNX_PATH) -> onnxruntime.InferenceSession:
    """Open the ONNX file in ONNX Runtime on the CPU (the only provider installed here)."""
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_fraud_probability(session: onnxruntime.InferenceSession, features: pd.DataFrame) -> np.ndarray:
    """Run the ONNX model and return the fraud probability for each row."""
    # ONNX Runtime wants float32 input. XGBoost itself also reads features as float32, so nothing extra is lost.
    matrix = features.to_numpy(dtype=np.float32)
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, session.run(None, {INPUT_NAME: matrix})))

    # The model emits a class label plus a (rows x 2) probability array; column 1 is the fraud class because the
    # classes are ordered [0, 1] (checked by check_class_order).
    probabilities = outputs[PROBABILITY_OUTPUT]
    assert probabilities.shape == (len(features), 2), f"Unexpected probability shape {probabilities.shape}"
    return probabilities[:, 1].astype(np.float64)


def check_class_order(model: xgboost.XGBClassifier) -> None:
    """Confirm the class order is [0, 1], so ONNX probability column 1 really is the fraud class."""
    assert list(model.classes_) == [0, 1], f"Unexpected class order {list(model.classes_)}"


def compare_scores(native: np.ndarray, onnx_scores: np.ndarray, tolerance: float = TOLERANCE) -> dict:
    """Summarise how far the ONNX scores are from the native scores."""
    difference = np.abs(native - onnx_scores)
    return {
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "rows_over_tolerance": int((difference > tolerance).sum()),
        "rows": int(len(difference)),
    }


def assert_within_tolerance(summary: dict, tolerance: float = TOLERANCE) -> None:
    """Hard check: the largest difference must not exceed the tolerance (the tolerance is never loosened here)."""
    assert summary["max_abs_diff"] <= tolerance, (
        f"ONNX differs from native by up to {summary['max_abs_diff']:.3e}; "
        f"{summary['rows_over_tolerance']:,} of {summary['rows']:,} rows exceed {tolerance}"
    )


def opset_description(onnx_model) -> str:
    """Describe the operator-set versions the exported model declares."""
    return ", ".join(f"{(item.domain or 'ai.onnx')}:{item.version}" for item in onnx_model.opset_import)


def log_to_mlflow(summary: dict, opset_text: str, data_hash: str) -> str:
    """Log the export check to MLflow and return the run ID."""
    with mlflow.start_run(run_name="onnx_export_xgboost") as run:
        mlflow.log_params({"seed": RANDOM_SEED, "dvc_data_hash": data_hash, "opset": opset_text,
                           "input_dtype": "float32", "n_features": N_FEATURES, "tolerance": TOLERANCE,
                           "converter": "onnxmltools.convert_xgboost"})
        mlflow.log_metric("max_abs_diff", summary["max_abs_diff"])
        mlflow.log_metric("mean_abs_diff", summary["mean_abs_diff"])
        mlflow.log_metric("test_rows", summary["rows"])
        mlflow.log_metric("rows_over_tolerance", summary["rows_over_tolerance"])
        mlflow.log_artifact(str(ONNX_PATH))
        return run.info.run_id


def main() -> int:
    """Convert the saved model, save the ONNX file, and verify it against the native model on the full test split."""
    setup_mlflow(EXPERIMENT_NAME)
    data_hash = read_dvc_data_hash()

    native_model = xgboost.XGBClassifier()
    native_model.load_model(str(MODEL_PATH))
    check_class_order(native_model)

    onnx_model = convert_to_onnx(load_model_for_conversion())
    save_onnx(onnx_model)
    opset_text = opset_description(onnx_model)
    onnx_location = ONNX_PATH.relative_to(REPO_ROOT).as_posix()
    print(f"Saved {onnx_location} ({ONNX_PATH.stat().st_size:,} bytes), opsets: {opset_text}")

    session = open_session()
    providers = session.get_providers()
    print(f"ONNX Runtime providers: {providers}")

    X_test, _ = load_features(TEST_PATH)
    print(f"Test rows: {len(X_test):,}")
    native = native_model.predict_proba(X_test)[:, 1]
    onnx_scores = onnx_fraud_probability(session, X_test)

    summary = compare_scores(native, onnx_scores)
    print(f"Max abs difference:  {summary['max_abs_diff']:.3e}")
    print(f"Mean abs difference: {summary['mean_abs_diff']:.3e}")
    print(f"Rows over {TOLERANCE}: {summary['rows_over_tolerance']:,} of {summary['rows']:,}")

    run_id = log_to_mlflow(summary, opset_text, data_hash)
    print(f"MLflow experiment '{EXPERIMENT_NAME}', run ID {run_id}")

    # The hard acceptance checks come last, after everything is saved and logged, so a failure still leaves the
    # numbers on disk and in MLflow to look at.
    assert providers == ["CPUExecutionProvider"], f"Unexpected providers {providers}"
    assert_within_tolerance(summary)
    print(f"\nAcceptance: ONNX matches native within {TOLERANCE} on all {summary['rows']:,} test rows - PASS")
    print("Acceptance: ONNX loads cleanly in onnxruntime (CPU) - PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
