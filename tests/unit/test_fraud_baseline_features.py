# File: tests/unit/test_fraud_baseline_features.py
# Purpose: Checks that the fraud baseline feature contract is exact: named columns only, bools as 0/1, fixed one-hot.
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# The script imports several heavy libraries; skip cleanly if any are missing in this environment.
for library in ("mlflow", "xgboost", "lightgbm", "catboost", "yaml"):
    pytest.importorskip(library)

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "fraud_baseline.py"
spec = importlib.util.spec_from_file_location("fraud_baseline", SCRIPT_PATH)
fraud_baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fraud_baseline)

EXPECTED_FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "account_avg_amount", "many_small_inbound_count", "large_outbound_amount",
    "new_device_flag", "geo_velocity_spike",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]
EXCLUDED = [
    "step", "merchant_risk_tier", "isFlaggedFraud", "is_adversarial", "adversarial_pattern_type",
    "nameOrig", "nameDest", "geo_coordinates", "device_fingerprint", "__index_level_0__",
]


def make_raw_frame() -> pd.DataFrame:
    """A tiny frame holding every column the real split has, plus a stray index column."""
    return pd.DataFrame(
        {
            "step": [1, 2], "type": ["TRANSFER", "PAYMENT"], "amount": [10.0, 20.0],
            "nameOrig": ["C1", "C2"], "oldbalanceOrg": [100.0, 200.0], "newbalanceOrig": [90.0, 180.0],
            "nameDest": ["C3", "M4"], "oldbalanceDest": [0.0, 0.0], "newbalanceDest": [10.0, 0.0],
            "isFraud": [1, 0], "isFlaggedFraud": [0, 0], "geo_coordinates": ["1,1", "2,2"],
            "device_fingerprint": ["dev_id=a", "dev_id=b"], "merchant_risk_tier": ["low", "high"],
            "new_device_flag": [True, False], "geo_velocity_spike": [False, True],
            "account_avg_amount": [5.0, 6.0], "many_small_inbound_count": [3, 4],
            "large_outbound_amount": [1.0, 2.0], "is_adversarial": [True, False],
            "adversarial_pattern_type": ["mule_account", None], "__index_level_0__": [7, 8],
        }
    )


def test_features_are_exactly_the_contract():
    features, _ = fraud_baseline.prepare_features(make_raw_frame())
    assert sorted(features.columns) == sorted(EXPECTED_FEATURES)


def test_excluded_columns_never_reach_the_model():
    features, _ = fraud_baseline.prepare_features(make_raw_frame())
    assert not set(EXCLUDED) & set(features.columns)


def test_bool_features_are_cast_to_zero_one_ints():
    features, _ = fraud_baseline.prepare_features(make_raw_frame())
    assert features["new_device_flag"].tolist() == [1, 0]
    assert features["geo_velocity_spike"].tolist() == [0, 1]
    assert str(features["new_device_flag"].dtype).startswith("int")


def test_type_one_hot_has_all_five_columns_even_if_a_type_is_missing():
    # The frame above has only TRANSFER and PAYMENT, yet all five type columns must exist, in a fixed order.
    features, _ = fraud_baseline.prepare_features(make_raw_frame())
    type_columns = [c for c in features.columns if c.startswith("type_")]
    assert type_columns == ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
    assert features["type_CASH_IN"].sum() == 0
    assert features["type_TRANSFER"].tolist() == [1, 0]


def test_target_is_isfraud():
    _, target = fraud_baseline.prepare_features(make_raw_frame())
    assert target.tolist() == [1, 0]


def test_scale_pos_weight_is_negatives_over_positives():
    labels = pd.Series([0] * 199 + [1])
    assert fraud_baseline.compute_scale_pos_weight(labels) == pytest.approx(199.0)
