# File: tests/unit/test_fraud_client.py
# Purpose: Checks the serving client's own logic without needing Triton: input validation, the calibration step
#          (numpy.interp read from the canonical pickle versus the pickle's own predict), and fixed constants.
from pathlib import Path

import numpy as np
import pytest

from tests.lazy_script import lazy_script

pytestmark = pytest.mark.ml  # needs the requirements-ml.txt libraries: CI's fast job leaves it out, the ml job runs it

CLIENT_PATH = Path(__file__).resolve().parent.parent.parent / "serving" / "fraud_client.py"
fraud_client = lazy_script(CLIENT_PATH, ("tritonclient.grpc", "joblib"))


def test_row_must_have_15_features():
    with pytest.raises(ValueError):
        fraud_client.check_rows(np.zeros(14))


def test_row_must_be_finite():
    with pytest.raises(ValueError):
        fraud_client.check_rows(np.full(15, np.nan))


def test_at_most_128_rows_per_request():
    assert fraud_client.check_rows(np.zeros((128, 15))).shape == (128, 15)
    with pytest.raises(ValueError):
        fraud_client.check_rows(np.zeros((129, 15)))


def test_single_row_becomes_a_one_row_float32_table():
    table = fraud_client.check_rows(np.arange(15))
    assert table.shape == (1, 15) and table.dtype == np.float32


def test_calibration_maps_with_a_toy_step_function():
    arrays = (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.1, 1.0]))
    result = fraud_client.apply_calibration(np.array([-1.0, 0.25, 0.5, 0.75, 2.0]), arrays)
    assert result == pytest.approx([0.0, 0.05, 0.1, 0.55, 1.0])  # outside the range it holds the end values


@pytest.mark.needs_dvc  # the calibrator pickle comes from `dvc pull`
def test_numpy_calibration_matches_the_canonical_pickle():
    if not fraud_client.CALIBRATOR_PATH.exists():
        pytest.skip("calibrator pickle not present (gitignored)")
    joblib = pytest.importorskip("joblib")
    # Triton returns float32 scores, so test on float32-valued scores (what the client really receives).
    calibrator = joblib.load(fraud_client.CALIBRATOR_PATH)
    arrays = fraud_client.load_calibration_arrays()
    scores = np.random.default_rng(0).random(50_000).astype(np.float32).astype(np.float64)
    ours = fraud_client.apply_calibration(scores, arrays)
    assert np.abs(ours - calibrator.predict(scores)).max() <= 1e-6
    assert ours.min() >= 0.0 and ours.max() <= 1.0
    assert np.all(np.diff(fraud_client.apply_calibration(np.sort(scores), arrays)) >= 0)  # never decreases


def test_constants_match_the_served_contract():
    assert fraud_client.MODEL_NAME == "fraud_xgboost"
    assert fraud_client.TRITON_URL.endswith(":8001")
    assert len(fraud_client.FEATURE_ORDER) == 15 and fraud_client.MAX_BATCH == 128
