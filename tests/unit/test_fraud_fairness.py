# File: tests/unit/test_fraud_fairness.py
# Purpose: Checks the 5.9 fairness code and model-card helpers: false-positive-rate arithmetic, labelled threshold
#          scales, the merchant-tier caveat, interval clamping, and the markdown table readers used by the card.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.lazy_script import lazy_script

pytestmark = pytest.mark.ml  # needs the requirements-ml.txt libraries: CI's fast job leaves it out, the ml job runs it

LIBRARIES = ("mlflow", "optuna", "xgboost", "shap", "joblib", "yaml")
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"

evaluation = lazy_script(SCRIPTS_DIR / "fraud_evaluation.py", LIBRARIES)
card = lazy_script(SCRIPTS_DIR / "fraud_model_card.py", LIBRARIES)


def tiny_data():
    """Six rows: three in segment A (one fraud) and three in segment B (no fraud); known scores."""
    y = pd.Series([1, 0, 0, 0, 0, 0])
    raw = np.array([0.9, 0.7, 0.2, 0.6, 0.1, 0.05])
    calibrated = np.array([0.8, 0.1, 0.01, 0.05, 0.001, 0.0])
    segments = pd.DataFrame({"segment": ["A", "A", "A", "B", "B", "B"]})
    return y, {"raw": raw, "calibrated": calibrated}, segments


def test_false_positive_rate_is_false_positives_over_non_fraud():
    y, scores, segments = tiny_data()
    row = evaluation.rate_row(y, scores, segments["segment"] == "A", "group", "A")
    cell = row["thresholds"]["raw_0.5"]  # rows flagged at raw >= 0.5 in A: scores 0.9 (fraud) and 0.7 (non-fraud)
    assert (row["rows"], row["fraud"], row["non_fraud"]) == (3, 1, 2)
    assert cell["fp"] == 1 and cell["fpr"] == pytest.approx(0.5) and cell["recall"] == pytest.approx(1.0)


def test_segment_with_no_fraud_has_an_fpr_but_no_recall():
    y, scores, segments = tiny_data()
    row = evaluation.rate_row(y, scores, segments["segment"] == "B", "group", "B")
    assert row["thresholds"]["raw_0.5"]["fp"] == 1 and row["thresholds"]["raw_0.5"]["recall"] is None
    assert row["thresholds"]["raw_0.5"]["fpr"] == pytest.approx(1 / 3)


def test_raw_and_calibrated_thresholds_use_their_own_scales():
    y, scores, segments = tiny_data()
    row = evaluation.rate_row(y, scores, segments["segment"] == "A", "group", "A")
    # Same segment, three thresholds: raw>=0.5 flags 2 rows, calibrated>=0.42 flags only the fraud row.
    assert row["thresholds"]["raw_0.5"]["fp"] == 1
    assert row["thresholds"]["cal_0.42"]["fp"] == 0 and row["thresholds"]["cal_0.42"]["recall"] == pytest.approx(1.0)
    scales = {t["key"]: t["scale"] for t in evaluation.FAIRNESS_THRESHOLDS}
    assert scales == {"raw_0.5": "raw", "cal_0.42": "calibrated", "cal_0.03": "calibrated"}


def test_wilson_interval_never_leaves_zero_to_one():
    low, high = evaluation.wilson_interval(0, 280_726)
    assert low == 0.0 and 0.0 < high < 0.001
    assert evaluation.wilson_interval(0, 0)[0] != evaluation.wilson_interval(0, 0)[0]  # NaN when there is no data
    low, high = evaluation.wilson_interval(0, 421)
    assert high == pytest.approx(0.0090, abs=0.0005)  # the ~0.90% upper bound quoted for the 421-row tier


def test_merchant_caveat_names_the_confound_and_the_small_tier():
    rows = [
        {"group": "Merchant risk tier (confounded)", "segment": "high", "rows": 421, "fraud": 0, "non_fraud": 421,
         "thresholds": {"raw_0.5": {"fp": 0, "fpr": 0.0, "recall": None}}, "fpr_interval": (0.0, 0.009)},
    ]
    confound = {
        "high": {"types": ["PAYMENT"], "fraud": 0, "rows": 421},
        "low": {"types": ["PAYMENT"], "fraud": 0, "rows": 5},
        "medium": {"types": ["CASH_IN", "TRANSFER"], "fraud": 12, "rows": 100},
    }
    text = " ".join(evaluation.merchant_caveat(rows, confound))
    assert "confounded with transaction type" in text and "NOT evidence of fair treatment" in text
    assert "421 rows" in text and "supports no conclusion" in text


def test_markdown_table_helpers_read_a_table():
    text = "## Results\n\n| Model | PR-AUC |\n|---|---|\n| xgboost | 0.99 |\n| catboost | 0.98 |\n\nafter\n"
    table = card.table_after_heading(text, "## Results")
    rows = card.parse_table(table)
    assert rows == [{"Model": "xgboost", "PR-AUC": "0.99"}, {"Model": "catboost", "PR-AUC": "0.98"}]
    assert card.row_where(rows, "Model", "cat")["PR-AUC"] == "0.98"


def test_runner_up_gap_is_the_largest_gap():
    models = [{"PR-AUC": "0.9947"}, {"PR-AUC": "0.9922"}, {"PR-AUC": "0.9912"}]
    assert card.runner_up_gap(models) == pytest.approx(0.0035)


def test_calibration_sentence_only_claims_worst_bin_when_it_is_the_thinnest():
    bins = [
        {"Bin": "0.0-0.1", "Rows": "1,000,000", "Deviation": "0.0000"},
        {"Bin": "0.1-0.2", "Rows": "500", "Deviation": "0.2000"},
        {"Bin": "0.2-0.3", "Rows": "5", "Deviation": "0.0100"},
        {"Bin": "0.9-1.0", "Rows": "6,000", "Deviation": "0.0005"},
    ]
    sentence = card.calibration_sparsity_sentence(bins)
    assert "0.2-0.3 (5 rows)" in sentence and "worst-calibrated bin on test is 0.1-0.2" in sentence
    assert "also the worst" not in sentence
