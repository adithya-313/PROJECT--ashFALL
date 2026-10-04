# File: serving/fraud_client.py
# Purpose: Thin gRPC client for the Triton-served fraud model; the single sanctioned way to get a calibrated fraud
#          probability (Triton scores a row, then the isotonic calibrator is applied here).
#
# Scope: this takes an ALREADY-PREPARED row of 15 features. Turning a raw payment request into those 15 features
# is feature enrichment, which is deferred to a later level and is deliberately not done here.
from pathlib import Path

import joblib
import numpy as np
import tritonclient.grpc as grpc_client

REPO_ROOT = Path(__file__).resolve().parent.parent

TRITON_URL = "localhost:8001"  # Triton's gRPC port on this machine (see docker-compose.yml)
MODEL_NAME = "fraud_xgboost"
INPUT_NAME = "input"
PROBABILITY_OUTPUT = "probabilities"  # two columns per row: [P(not fraud), P(fraud)]
MAX_BATCH = 128  # the model's max_batch_size in config.pbtxt (raised from 8 in Level 6 for the batch-size sweep)

# The canonical calibrator. This pickle is the source of truth (later levels load it too); this file only READS it.
CALIBRATOR_PATH = REPO_ROOT / "models" / "calibration" / "isotonic_calibrator.pkl"

# The 15 features, in the fixed order the model was trained on. The row must be given in exactly this order.
FEATURE_ORDER = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "account_avg_amount", "many_small_inbound_count", "large_outbound_amount",
    "new_device_flag", "geo_velocity_spike",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]


def load_calibration_arrays(path: Path = CALIBRATOR_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Read the calibrator pickle once and return its step positions (raw scores) and step heights (calibrated)."""
    # The pickle stays the canonical calibrator. An isotonic calibrator is just a list of (raw score, calibrated
    # probability) points joined by straight lines, so we read those two arrays out of it. Applying them with
    # numpy.interp then needs no scikit-learn and is very cheap per request.
    calibrator = joblib.load(path)
    raw_points = np.asarray(calibrator.X_thresholds_, dtype=np.float64)
    calibrated_points = np.asarray(calibrator.y_thresholds_, dtype=np.float64)
    return raw_points, calibrated_points


def apply_calibration(raw_scores: np.ndarray, arrays: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Turn raw model scores into calibrated fraud probabilities."""
    raw_points, calibrated_points = arrays
    # np.interp holds the first/last value for scores outside the fitted range, like the calibrator's "clip" setting.
    return np.interp(np.asarray(raw_scores, dtype=np.float64), raw_points, calibrated_points)


def make_client(url: str = TRITON_URL) -> grpc_client.InferenceServerClient:
    """Open a gRPC connection to Triton."""
    return grpc_client.InferenceServerClient(url=url)


def check_rows(rows: np.ndarray) -> np.ndarray:
    """Make sure the input is a (n_rows, 15) float32 table of real numbers with at most MAX_BATCH rows."""
    table = np.atleast_2d(np.asarray(rows, dtype=np.float32))
    if table.ndim != 2 or table.shape[1] != len(FEATURE_ORDER):
        raise ValueError(f"Expected rows of {len(FEATURE_ORDER)} features, got shape {table.shape}")
    if not 1 <= table.shape[0] <= MAX_BATCH:
        raise ValueError(f"Expected between 1 and {MAX_BATCH} rows per request, got {table.shape[0]}")
    if not np.isfinite(table).all():
        raise ValueError("Features must be finite numbers (no NaN or infinity)")
    return table


def raw_fraud_probability(client: grpc_client.InferenceServerClient, rows: np.ndarray) -> np.ndarray:
    """Send rows to Triton over gRPC and return the RAW (uncalibrated) fraud probability for each row."""
    table = check_rows(rows)
    request = grpc_client.InferInput(INPUT_NAME, list(table.shape), "FP32")
    request.set_data_from_numpy(table)
    response = client.infer(MODEL_NAME, [request], outputs=[grpc_client.InferRequestedOutput(PROBABILITY_OUTPUT)])
    # Column 1 is the fraud class (the model's classes are ordered [0, 1]).
    return response.as_numpy(PROBABILITY_OUTPUT)[:, 1]


def predict_batch(rows: np.ndarray, client=None, arrays=None) -> np.ndarray:
    """Calibrated fraud probability for each of up to 8 rows."""
    # This (with predict below) is the ONLY sanctioned way to get a prediction. The raw model output is never
    # returned as a final answer, so the model and its calibrator always travel together.
    client = client if client is not None else make_client()
    arrays = arrays if arrays is not None else load_calibration_arrays()
    return apply_calibration(raw_fraud_probability(client, rows), arrays)


def predict(features_15, client=None, arrays=None) -> float:
    """Calibrated fraud probability for ONE row of 15 features (in FEATURE_ORDER)."""
    return float(predict_batch(np.asarray(features_15, dtype=np.float32).reshape(1, -1), client, arrays)[0])
