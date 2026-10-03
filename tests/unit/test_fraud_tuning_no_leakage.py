# File: tests/unit/test_fraud_tuning_no_leakage.py
# Purpose: Proves the Optuna tuning script only touches the train split, samples it stratified, and feeds the
#          models only the 15 contract features (no labels, identifiers or other leakage columns).
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# The script imports several heavy libraries; skip cleanly if any are missing in this environment.
for library in ("mlflow", "optuna", "xgboost", "catboost", "yaml"):
    pytest.importorskip(library)

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "fraud_tuning.py"
spec = importlib.util.spec_from_file_location("fraud_tuning", SCRIPT_PATH)
fraud_tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fraud_tuning)

FORBIDDEN_FEATURES = [
    "isFraud", "isFlaggedFraud", "is_adversarial", "adversarial_pattern_type", "nameOrig", "nameDest",
    "geo_coordinates", "device_fingerprint", "merchant_risk_tier", "step", "__index_level_0__",
]


def make_raw_split(rows: int = 10_000, fraud_rows: int = 50) -> pd.DataFrame:
    """A synthetic stand-in for the train split: every real column, a 0.5% fraud rate, a unique amount per row."""
    return pd.DataFrame(
        {
            "step": 1, "type": ["TRANSFER" if i % 2 else "PAYMENT" for i in range(rows)],
            "amount": [float(i) for i in range(rows)],
            "nameOrig": [f"C{i}" for i in range(rows)], "oldbalanceOrg": 100.0, "newbalanceOrig": 90.0,
            "nameDest": "C0", "oldbalanceDest": 0.0, "newbalanceDest": 10.0,
            "isFraud": [1 if i < fraud_rows else 0 for i in range(rows)], "isFlaggedFraud": 0,
            "geo_coordinates": "1,1", "device_fingerprint": "dev_id=a", "merchant_risk_tier": "low",
            "new_device_flag": True, "geo_velocity_spike": False, "account_avg_amount": 5.0,
            "many_small_inbound_count": 3, "large_outbound_amount": 1.0, "is_adversarial": False,
            "adversarial_pattern_type": None,
        }
    )


def test_source_never_references_the_other_two_splits():
    # Look at the code only (comment lines are skipped): the other splits' file names must not appear at all.
    code_lines = [line for line in SCRIPT_PATH.read_text().splitlines() if not line.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert "calibration.parquet" not in code
    assert "test.parquet" not in code
    assert not hasattr(fraud_tuning, "TEST_PATH")
    assert not hasattr(fraud_tuning, "CALIBRATION_PATH")


def test_loader_reads_only_the_train_file_and_no_leakage_columns(monkeypatch):
    requested = {}

    def fake_read_parquet(path, columns=None):
        requested["path"] = Path(path)
        requested["columns"] = columns
        return make_raw_split()[columns]

    monkeypatch.setattr(fraud_tuning.pd, "read_parquet", fake_read_parquet)
    X, y = fraud_tuning.load_train_features()

    assert requested["path"] == fraud_tuning.TRAIN_PATH
    assert requested["path"].name == "train.parquet"
    assert not set(FORBIDDEN_FEATURES) & set(requested["columns"] or []) - {"isFraud"}
    assert not set(FORBIDDEN_FEATURES) & set(X.columns)
    assert len(X.columns) == 15
    assert y.name == "isFraud"


def test_sample_is_a_stratified_subset_of_the_train_rows():
    X, y = fraud_tuning.prepare_features(make_raw_split())
    X_sample, y_sample = fraud_tuning.make_tuning_sample(X, y, 0.25, 42)

    assert len(X_sample) == 2_500
    # amount is unique per synthetic row, so every sampled amount must exist in the original train rows.
    assert set(X_sample["amount"]) <= set(X["amount"])
    # The fraud ratio is preserved (0.5%), to within one fraud row.
    assert abs(int(y_sample.sum()) - 12.5) <= 1
    assert list(X_sample.index) == list(range(len(X_sample)))


def test_folds_are_identical_every_time_and_cover_each_row_once():
    _, y = fraud_tuning.prepare_features(make_raw_split())
    first = fraud_tuning.make_folds(y, 5, 42)
    second = fraud_tuning.make_folds(y, 5, 42)

    for (train_a, valid_a), (train_b, valid_b) in zip(first, second):
        assert list(valid_a) == list(valid_b)
        assert not set(train_a) & set(valid_a)
    all_valid = sorted(index for _, valid in first for index in valid)
    assert all_valid == list(range(len(y)))


def test_feature_matrix_excludes_every_forbidden_column():
    X, _ = fraud_tuning.prepare_features(make_raw_split())
    assert not set(FORBIDDEN_FEATURES) & set(X.columns)


def test_every_model_has_a_search_space_folds_and_baseline():
    for name in ("xgboost", "catboost", "random_forest"):
        assert name in fraud_tuning.SEARCH_SPACES
        assert name in fraud_tuning.N_FOLDS
        assert name in fraud_tuning.BASELINE_TRIAL_PARAMS
    assert fraud_tuning.N_FOLDS == {"xgboost": 5, "catboost": 5, "random_forest": 3}


def test_baseline_trial_is_accepted_exactly_by_every_search_space():
    # The queued baseline must come back unchanged from the search space (values inside every range), otherwise the
    # first trial would not really be the baseline.
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    baseline_params = {
        name: dict(values) for name, values in fraud_tuning.BASELINE_TRIAL_PARAMS.items()
    }
    baseline_params["catboost"]["learning_rate"] = 0.09
    for name, params in baseline_params.items():
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        study.enqueue_trial(params)
        trial = study.ask()
        assert fraud_tuning.SEARCH_SPACES[name](trial) == params
