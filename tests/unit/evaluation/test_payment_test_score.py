import json
import math

import numpy as np
import pyarrow as pa
import pytest
import xgboost

from evaluation.payment_test_score import (
    HOLD_SHARE,
    WorldScores,
    aggregate,
    attach_payee_risk,
    evaluate_splits,
    hold_threshold,
    validate_scoring_split,
    verify_booster_rounds,
    verify_frozen_artifacts,
    world_metrics,
)


def _manifest(folder, split):
    folder.mkdir()
    (folder / "manifest.json").write_text(
        json.dumps({"split": split, "worlds": [{"world_id": "w1"}]}),
        encoding="utf-8",
    )
    return folder


def test_hold_threshold_is_the_linear_quantile_of_the_validation_scores():
    threshold = hold_threshold([float(value) for value in range(10)])
    assert HOLD_SHARE == 0.005
    assert math.isclose(threshold, 8.955, rel_tol=0.0, abs_tol=1e-12)
    with pytest.raises(ValueError):
        hold_threshold([])
    with pytest.raises(ValueError):
        hold_threshold([0.5], hold_share=0.0)


def test_the_frozen_artifact_hashes_must_match_a_058(tmp_path):
    (tmp_path / "frozen_model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feature_order.yaml").write_text("features: []\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        verify_frozen_artifacts(tmp_path)
    message = str(excinfo.value)
    assert "frozen_model.json" in message
    assert "feature_order.yaml" in message


def test_the_booster_round_count_must_match_the_model_metadata():
    train = xgboost.DMatrix(np.array([[0.0, 1.0], [1.0, 0.0]]), label=[0, 1])
    booster = xgboost.train({"max_depth": 1}, train, num_boost_round=1)
    assert booster.num_boosted_rounds() == 1
    assert verify_booster_rounds(booster, {"best_params": {"n_estimators": 1}}) == 1
    with pytest.raises(ValueError):
        verify_booster_rounds(booster, {"best_params": {"n_estimators": 2}})


def test_test_and_stress_scores_never_move_the_validation_threshold():
    validation_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    test_world = WorldScores("test", "test-a", "w1", [0.9, 0.95, 0.99], [0, 1, 1], [1.0, math.nan, 2.0])
    stress_world = WorldScores("stress", "stress-a", "w2", [0.99, 0.999], [1, 0], [math.nan, math.nan])
    result = evaluate_splits(validation_scores, {"test": [test_world], "stress": [stress_world]})
    expected = hold_threshold(validation_scores)
    assert result["threshold"] == expected

    extreme = [
        WorldScores("test", "test-a", "w1", [1.0e9, 1.0e9, 1.0e9], [0, 1, 1], [1.0, math.nan, 2.0]),
        WorldScores("stress", "stress-a", "w2", [1.0e9, 1.0e9], [1, 0], [math.nan, math.nan]),
    ]
    extreme_result = evaluate_splits(validation_scores, {"test": [extreme[0]], "stress": [extreme[1]]})
    assert extreme_result["threshold"] == expected


def test_world_metrics_are_the_hand_computed_values_at_the_threshold():
    world = WorldScores(
        split="test",
        dataset="test-a",
        world_id="w1",
        scores=[0.9, 0.2, 0.8, 0.1, 0.05],
        labels=[1, 1, 0, 0, 0],
        payee_risk=[3.0, math.nan, 1.0, 2.0, math.nan],
    )
    metrics = world_metrics(world, 0.5)
    assert metrics["rows"] == 5
    assert metrics["positives"] == 2
    assert metrics["held"] == 2
    assert metrics["true_positives"] == 1
    assert math.isclose(metrics["truth_ap"], 5.0 / 6.0, rel_tol=0.0, abs_tol=1e-12)
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["hold_rate"] == 0.4
    assert metrics["same_bank"] == {"rows": 3, "positives": 1, "truth_ap": 1.0}
    assert metrics["other_bank"] == {"rows": 2, "positives": 1, "truth_ap": 1.0}
    assert metrics["same_bank_truth_ap"] == 1.0
    assert metrics["other_bank_truth_ap"] == 1.0
    assert metrics["nan_payee_risk_share"] == 0.4


def test_average_precision_is_null_when_a_world_or_subset_lacks_a_class():
    no_positives = WorldScores("test", "test-a", "w1", [0.5, 0.6], [0, 0], [1.0, 2.0])
    metrics = world_metrics(no_positives, 0.55)
    assert metrics["truth_ap"] is None
    assert metrics["recall"] is None
    assert metrics["precision"] == 0.0
    assert metrics["same_bank"]["truth_ap"] is None

    no_holdings = world_metrics(no_positives, 0.9)
    assert no_holdings["precision"] is None
    assert no_holdings["hold_rate"] == 0.0

    all_positive = WorldScores("stress", "stress-a", "w2", [0.1, 0.2], [1, 1], [math.nan, math.nan])
    all_positive_metrics = world_metrics(all_positive, 0.05)
    assert all_positive_metrics["truth_ap"] is None
    assert all_positive_metrics["recall"] == 1.0
    assert all_positive_metrics["other_bank"]["truth_ap"] is None


def test_aggregate_reports_median_p10_p90_over_the_non_null_worlds():
    aggregated = aggregate([0.1, 0.2, 0.3, None, 0.4])
    assert aggregated["worlds_used"] == 4
    assert math.isclose(aggregated["median"], 0.25, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(aggregated["p10"], 0.13, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(aggregated["p90"], 0.37, rel_tol=0.0, abs_tol=1e-12)
    assert aggregate([None, None]) == {"median": None, "p10": None, "p90": None, "worlds_used": 0}


def test_scoring_split_guard_accepts_only_test_and_stress(tmp_path):
    test_folder = _manifest(tmp_path / "test-a", "test")
    validation_folder = _manifest(tmp_path / "validation-a", "validation")
    training_folder = _manifest(tmp_path / "training-a", "training")

    assert validate_scoring_split(test_folder, "test")["split"] == "test"
    with pytest.raises(ValueError):
        validate_scoring_split(test_folder, "stress")
    with pytest.raises(ValueError):
        validate_scoring_split(validation_folder, "test")
    with pytest.raises(ValueError):
        validate_scoring_split(training_folder, "stress")
    with pytest.raises(ValueError):
        validate_scoring_split(test_folder, "validation")


def test_attach_payee_risk_proves_the_keys_line_up():
    table = pa.table({"world_id": ["w", "w"], "payment_id": ["p1", "p2"]})
    risk_rows = [
        {"world_id": "w", "payment_id": "p1", "payee_risk": 0.4, "nan_reason": ""},
        {"world_id": "w", "payment_id": "p2", "payee_risk": math.nan, "nan_reason": "no_prior_account_day"},
    ]
    attached = attach_payee_risk(table, risk_rows)
    assert attached["payee_risk"].to_pylist()[0] == 0.4
    assert math.isnan(attached["payee_risk"].to_pylist()[1])
    assert attached["nan_reason"].to_pylist() == ["", "no_prior_account_day"]

    swapped = [risk_rows[1], risk_rows[0]]
    with pytest.raises(ValueError):
        attach_payee_risk(table, swapped)
    with pytest.raises(ValueError):
        attach_payee_risk(table, risk_rows[:1])


def test_evaluate_splits_reports_validation_hold_rate_and_world_aggregates():
    validation_scores = [0.2] * 10
    test_worlds = [
        WorldScores("test", "test-a", "w1", [0.3, 0.1, 0.2, 0.05], [1, 0, 1, 0], [1.0, 2.0, math.nan, math.nan]),
        WorldScores("test", "test-a", "w2", [0.05, 0.04], [1, 0], [3.0, 4.0]),
    ]
    stress_worlds = [WorldScores("stress", "stress-a", "w3", [0.25, 0.15], [1, 0], [5.0, math.nan])]
    result = evaluate_splits(validation_scores, {"test": test_worlds, "stress": stress_worlds})

    assert result["threshold"] == 0.2
    assert result["validation"] == {"rows": 10, "held": 10, "hold_rate": 1.0}
    aggregate_test = result["splits"]["test"]["aggregate"]
    assert aggregate_test["truth_ap"]["worlds_used"] == 2
    assert aggregate_test["truth_ap"]["median"] == 1.0
    assert aggregate_test["same_bank_truth_ap"]["worlds_used"] == 2
    assert aggregate_test["other_bank_truth_ap"]["worlds_used"] == 1
    assert aggregate_test["recall"]["median"] == 0.5
    assert aggregate_test["precision"]["worlds_used"] == 1
    assert aggregate_test["hold_rate"]["median"] == 0.25
    stress_aggregate = result["splits"]["stress"]["aggregate"]
    assert stress_aggregate["truth_ap"]["worlds_used"] == 1
    with pytest.raises(ValueError):
        evaluate_splits(validation_scores, {"test": [], "stress": stress_worlds})
