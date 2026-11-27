# File: tests/unit/evaluation/test_reason_codes.py
# Purpose: Checks the A-059 reason-code rules: the generated map, the positive-SHAP ranking, the
# group code mapping, the bagged account SHAP mean and the seeded sample selection.
import numpy as np
import pytest
import shap
import xgboost
import yaml

from decisions.pu_learning import PUBaggedAccountModel
from evaluation.reason_codes import (
    DEFAULT_CODES_PATH,
    DEFAULT_SCHEMA_PATH,
    account_shap_values,
    build_reason_code_map,
    payment_shap_values,
    reason_codes_for_features,
    row_reasons,
    select_sample,
    top_reason_features,
)


def test_reason_code_map_matches_the_committed_yaml_and_schema_groups():
    reason_map = build_reason_code_map()
    committed = yaml.safe_load(DEFAULT_CODES_PATH.read_text(encoding="utf-8"))
    assert committed == reason_map.payload()

    schema = yaml.safe_load(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_names = [record["name"] for record in schema["features"]]
    assert set(schema_names) <= set(reason_map.features)
    assert len(reason_map.features) == len(schema_names) + 1
    assert reason_map.features["distinct_payers_1h"] == "INFLOW"
    assert reason_map.features["payee_type"] == "PAYMENT_LEVEL"
    assert reason_map.features["payee_risk"] == "PAYEE_RISK"

    codes = [entry.code for entry in reason_map.codes]
    assert len(codes) == len(set(codes))
    assert codes == [entry.group.upper() for entry in reason_map.codes]


def test_driving_feature_is_the_first_reason():
    rng = np.random.default_rng(11)
    driver = rng.normal(size=600)
    noise = rng.normal(size=600)
    matrix = np.column_stack([driver, noise])
    labels = (driver > 0.5).astype(int)
    model = xgboost.XGBClassifier(
        n_estimators=40,
        max_depth=3,
        tree_method="hist",
        device="cpu",
        random_state=7,
        objective="binary:logistic",
    )
    model.fit(matrix, labels)

    high = np.array([[2.0, 0.0]])
    low = np.array([[-2.0, 0.0]])
    assert model.predict_proba(high)[0, 1] > 0.9
    assert model.predict_proba(low)[0, 1] < 0.1

    values = payment_shap_values(model, high, feature_count=2)
    assert values.shape == (1, 2)

    sample = matrix[[0]]
    sample_values = payment_shap_values(model, sample, feature_count=2)
    contribs = model.get_booster().predict(xgboost.DMatrix(sample), pred_contribs=True)
    margin = float(model.get_booster().predict(xgboost.DMatrix(sample), output_margin=True)[0])
    assert abs(float(contribs[0, -1]) + float(sample_values[0].sum()) - margin) < 1e-5

    reasons = top_reason_features(values[0].tolist(), ("driver", "noise"))
    assert reasons[0].feature == "driver"
    assert reasons[0].shap > 0.0
    coded, codes = row_reasons(values[0].tolist(), ("driver", "noise"), {"driver": "INFLOW", "noise": "CONTROL"})
    assert coded[0].feature == "driver"
    assert codes[0] == "INFLOW"


def test_top_reason_features_keeps_positive_values_in_rank_order_and_stops_at_three():
    names = ("f0", "f1", "f2", "f3", "f4", "f5")
    reasons = top_reason_features([0.0, 2.0, -1.0, 3.0, 1.0, 0.5], names)
    assert [reason.feature for reason in reasons] == ["f3", "f1", "f4"]
    assert [reason.shap for reason in reasons] == [3.0, 2.0, 1.0]
    assert top_reason_features([0.0, -1.0], ("a", "b")) == []
    with pytest.raises(ValueError):
        top_reason_features([1.0], ("a", "b"))
    with pytest.raises(ValueError):
        top_reason_features([float("nan")], ("a",))
    with pytest.raises(ValueError):
        top_reason_features([1.0], ("a",), top_n=0)


def test_reason_codes_deduplicate_groups_in_rank_order_and_reject_unknown_features():
    feature_codes = {"a": "INFLOW", "b": "INFLOW", "c": "CONTROL"}
    assert reason_codes_for_features(["a", "b", "c"], feature_codes) == ["INFLOW", "CONTROL"]
    assert reason_codes_for_features(["c", "a"], feature_codes) == ["CONTROL", "INFLOW"]
    assert reason_codes_for_features([], feature_codes) == []
    with pytest.raises(ValueError):
        reason_codes_for_features(["missing"], feature_codes)


def test_account_shap_is_the_mean_over_the_bagged_members():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(40, 3))
    labels = (matrix[:, 0] > 0).astype(int)
    members = []
    for seed in (1, 2):
        member = xgboost.XGBClassifier(
            n_estimators=12,
            max_depth=2,
            tree_method="hist",
            device="cpu",
            random_state=seed,
            objective="binary:logistic",
        )
        member.fit(matrix + rng.normal(scale=0.1, size=matrix.shape), labels)
        members.append(member)

    bagged = PUBaggedAccountModel(
        members=tuple(members),
        member_seeds=(1, 2),
        n_bags=2,
        n_estimators=12,
        unlabelled_per_positive=1,
    )
    values = account_shap_values(bagged, matrix.tolist(), 3)
    expected = np.mean(
        [np.asarray(shap.TreeExplainer(member).shap_values(matrix), dtype=float) for member in members],
        axis=0,
    )
    assert values.shape == (40, 3)
    assert np.allclose(values, expected, rtol=1e-12, atol=1e-12)


def test_sample_selection_is_seeded_deterministic_and_distinct():
    rows = [{"key": f"row-{index}"} for index in range(500)]
    first = select_sample(rows, ("key",), seed=1717, sample_rows=50)
    second = select_sample(rows, ("key",), seed=1717, sample_rows=50)
    other = select_sample(rows, ("key",), seed=1718, sample_rows=50)
    assert [row["key"] for row in first] == [row["key"] for row in second]
    assert [row["key"] for row in first] != [row["key"] for row in other]
    assert len(first) == 50
    assert len({row["key"] for row in first}) == 50

    assert len(select_sample(rows[:10], ("key",), seed=1717, sample_rows=50)) == 10
    with pytest.raises(ValueError):
        select_sample(rows, ("key",), seed=1717, sample_rows=0)
    with pytest.raises(ValueError):
        select_sample([{"key": "a"}, {"key": "a"}], ("key",), seed=1717, sample_rows=5)
    with pytest.raises(ValueError):
        select_sample([{}], ("key",), seed=1717, sample_rows=1)
