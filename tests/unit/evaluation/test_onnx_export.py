# File: tests/unit/evaluation/test_onnx_export.py
# Purpose: Pin the A-061 ONNX rules: the leaf-scaled 50-bag merge, the merge preconditions, the
# classifier export contract, the parity summary and the dataset split guard.

import json

import numpy as np
import pytest
import xgboost

from evaluation.onnx_export import (
    EXPECTED_OUTPUTS,
    MERGE_TOLERANCE,
    PARITY_TOLERANCE,
    _nan_rows,
    _split_datasets,
    export_model,
    merge_member_models,
    onnx_probabilities,
    open_session,
    parity_summary,
    save_onnx,
    write_margin_booster,
)
from evaluation.payment_model import predict_scores

N_FEATURES = 4
ROUNDS = 3


def _toy_booster(seed: int) -> xgboost.XGBClassifier:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(240, N_FEATURES))
    features[::7, 0] = np.nan
    labels = (features[:, 1] > 0).astype(np.int8)
    model = xgboost.XGBClassifier(
        n_estimators=ROUNDS,
        max_depth=2,
        learning_rate=0.3,
        objective="binary:logistic",
        base_score=0.5,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(features, labels)
    return model


def _raw(model: xgboost.XGBClassifier) -> dict:
    return json.loads(model.get_booster().save_raw("json"))


def _leaves(tree: dict) -> list[float]:
    return [value for value, left in zip(tree["split_conditions"], tree["left_children"], strict=True) if left == -1]


def test_merge_member_models_scales_every_leaf_and_matches_the_mean_margin(tmp_path):
    models = [_toy_booster(11), _toy_booster(23)]
    raws = [_raw(model) for model in models]

    merged = merge_member_models(raws, len(models), N_FEATURES)
    booster = write_margin_booster(merged, tmp_path / "merged.json", 2 * ROUNDS, N_FEATURES)

    merged_model = merged["learner"]["gradient_booster"]["model"]
    assert merged_model["gbtree_model_param"]["num_trees"] == str(2 * ROUNDS)
    assert merged_model["iteration_indptr"] == list(range(2 * ROUNDS + 1))
    assert merged_model["tree_info"] == [0] * (2 * ROUNDS)
    member_trees = raws[0]["learner"]["gradient_booster"]["model"]["trees"]
    leaf_count = 0
    for merged_tree, member_tree in zip(merged_model["trees"][:ROUNDS], member_trees, strict=True):
        member_leaves = _leaves(member_tree)
        merged_leaves = _leaves(merged_tree)
        assert member_leaves
        assert len(merged_leaves) == len(member_leaves)
        np.testing.assert_allclose(merged_leaves, np.asarray(member_leaves) / 2.0, rtol=0.0, atol=0.0)
        leaf_count += len(merged_leaves)
    assert leaf_count > 0

    rng = np.random.default_rng(31)
    rows = rng.normal(size=(64, N_FEATURES))
    rows[5, 2] = np.nan
    dmatrix = xgboost.DMatrix(rows)
    native = np.mean([model.get_booster().predict(dmatrix, output_margin=True) for model in models], axis=0)
    merged_margins = booster.predict(dmatrix, output_margin=True)
    difference = np.abs(np.asarray(native) - np.asarray(merged_margins))
    assert difference.shape == (64,)
    assert difference.max() <= MERGE_TOLERANCE
    assert _nan_rows(rows) == 1


def test_merge_member_models_rejects_a_disagreeing_base_score():
    raws = [_raw(_toy_booster(11)), _raw(_toy_booster(23))]
    base_scores = {raw["learner"]["learner_model_param"]["base_score"] for raw in raws}
    assert len(base_scores) == 1
    changed = json.loads(json.dumps(raws[1]))
    changed["learner"]["learner_model_param"]["base_score"] = "[9E-1]"

    with pytest.raises(ValueError, match="base_score"):
        merge_member_models([raws[0], changed], 2, N_FEATURES)
    assert merge_member_models(raws, 2, N_FEATURES)["learner"]["gradient_booster"]["model"]["trees"]


def test_merge_member_models_rejects_a_non_logistic_objective():
    raws = [_raw(_toy_booster(11)), _raw(_toy_booster(23))]
    assert raws[0]["learner"]["objective"]["name"] == "binary:logistic"
    changed = json.loads(json.dumps(raws[0]))
    changed["learner"]["objective"]["name"] = "binary:hinge"

    with pytest.raises(ValueError, match="binary:logistic"):
        merge_member_models([changed, raws[1]], 2, N_FEATURES)


def test_merge_member_models_rejects_a_wrong_member_count():
    raws = [_raw(_toy_booster(11)), _raw(_toy_booster(23))]

    with pytest.raises(ValueError, match="do not match the 3 frozen bags"):
        merge_member_models(raws, 3, N_FEATURES)


def test_export_model_keeps_the_classifier_contract_and_matches_the_native_probability(tmp_path):
    model = _toy_booster(41)
    converted = export_model(model.get_booster(), N_FEATURES)
    assert [node.op_type for node in converted.graph.node] == ["TreeEnsembleClassifier"]
    assert [output.name for output in converted.graph.output] == EXPECTED_OUTPUTS
    assert converted.graph.input[0].name == "input"

    save_onnx(converted, tmp_path / "toy.onnx")
    session = open_session(tmp_path / "toy.onnx")
    rng = np.random.default_rng(53)
    rows = rng.normal(size=(128, N_FEATURES))
    rows[3, 1] = np.nan
    rows[90, 2] = np.nan
    native = np.asarray(predict_scores(model, rows), dtype=np.float64)
    converted_scores = onnx_probabilities(session, rows)

    summary = parity_summary(native, converted_scores, _nan_rows(rows))
    assert summary["rows"] == 128
    assert summary["nan_rows"] == 2
    assert summary["max_abs_diff"] <= PARITY_TOLERANCE
    assert np.isfinite(native).all()
    np.testing.assert_allclose(converted_scores, native, rtol=0.0, atol=PARITY_TOLERANCE)


def test_parity_summary_enforces_the_tolerance_and_reports_the_nan_rows():
    summary = parity_summary([0.1, 0.2], [0.1, 0.2 + 1e-6], 3)
    assert summary["rows"] == 2
    assert summary["nan_rows"] == 3
    assert summary["max_abs_diff"] == pytest.approx(1e-6)
    assert summary["mean_abs_diff"] == pytest.approx(5e-7)

    with pytest.raises(ValueError, match="over the 1e-05 limit"):
        parity_summary([0.1, 0.2], [0.1, 0.2 + 2e-5], 0)
    with pytest.raises(ValueError, match="ONNX scores"):
        parity_summary([0.1, 0.2], [0.1], 0)
    with pytest.raises(ValueError, match="at least one row"):
        parity_summary([], [], 0)


def test_write_margin_booster_rejects_the_wrong_round_count(tmp_path):
    raws = [_raw(_toy_booster(11)), _raw(_toy_booster(23))]
    merged = merge_member_models(raws, 2, N_FEATURES)

    with pytest.raises(ValueError, match="merged booster holds 6 rounds, expected 7"):
        write_margin_booster(merged, tmp_path / "merged.json", 2 * ROUNDS + 1, N_FEATURES)
    booster = write_margin_booster(merged, tmp_path / "merged.json", 2 * ROUNDS, N_FEATURES)
    assert booster.num_boosted_rounds() == 2 * ROUNDS


def test_split_datasets_guards_the_split_and_the_folder_names(tmp_path):
    with pytest.raises(ValueError, match="no validation dataset folder"):
        _split_datasets([], "validation")

    first = tmp_path / "a" / "split_a"
    second = tmp_path / "b" / "split_a"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    with pytest.raises(ValueError, match="not unique"):
        _split_datasets([first, second], "validation")
    with pytest.raises(ValueError, match="Dataset manifest is missing"):
        _split_datasets([first], "validation")

    (first / "manifest.json").write_text(json.dumps({"split": "validation", "worlds": []}), encoding="utf-8")
    assert [path.name for path in _split_datasets([first], "validation")] == ["split_a"]
    with pytest.raises(ValueError, match="final scoring accepts only test and stress"):
        _split_datasets([first], "test")
