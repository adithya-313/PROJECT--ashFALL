"""Tests for the A-066 payment-model v2 candidates: constraint, segments, selection and scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from evaluation.payment_candidates import (
    CandidateModel,
    fit_candidate_from_params,
    fit_candidate_model,
    load_candidate_model,
    monotone_vector,
    pooled_truth_ap,
    replacement_decision,
    segment_masks,
    select_candidate,
    validate_winner_scoring_folder,
    verify_candidate_artifacts,
    world_p10,
    world_truth_ap,
    write_candidate_artifacts,
)
from evaluation.payment_model import feature_list_sha256

SMALL_PARAMS = {"n_estimators": 4, "max_depth": 2, "learning_rate": 0.5, "min_child_weight": 1.0}
ROUTING_PARAMS = {**SMALL_PARAMS, "n_estimators": 8, "min_child_weight": 0.0}


def _manifest(folder: Path, split: str) -> Path:
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"split": split}), encoding="utf-8")
    return folder


def _feature_order(folder: Path, features: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "feature_order.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "features": features,
                "feature_sha256": feature_list_sha256(features),
                "categorical_encodings": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_monotone_vector_constrains_only_payee_risk() -> None:
    features = ["amount", "payee_risk", "hour_of_day"]
    assert monotone_vector(features) == [0, 1, 0]
    with pytest.raises(ValueError, match="payee_risk"):
        monotone_vector(["amount", "hour_of_day"])


def test_segment_masks_split_on_finite_payee_risk_and_refuse_an_empty_segment() -> None:
    masks = segment_masks([0.1, math.nan, 0.2, float("nan")])
    assert masks["present"].tolist() == [True, False, True, False]
    assert masks["nan"].tolist() == [False, True, False, True]
    with pytest.raises(ValueError, match="present"):
        segment_masks([math.nan, math.nan])
    with pytest.raises(ValueError, match="nan"):
        segment_masks([0.1, 0.2])
    with pytest.raises(ValueError, match="one-dimensional"):
        segment_masks(np.array([[0.1, math.nan]]))


def test_the_fitted_c1_constraint_makes_the_payee_risk_response_non_decreasing() -> None:
    matrix = np.array([[0.0], [1.0], [0.0], [1.0]] * 4, dtype=np.float32)
    labels = np.array([1, 0, 1, 0] * 4)
    grid = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)

    free = fit_candidate_model(matrix, labels, SMALL_PARAMS, "cpu", 7, None)
    constrained = fit_candidate_model(matrix, labels, SMALL_PARAMS, "cpu", 7, monotone_vector(["payee_risk"]))

    free_scores = free.predict_proba(grid)[:, 1]
    constrained_scores = constrained.predict_proba(grid)[:, 1]
    assert free_scores[0] > free_scores[-1], "the data must oppose an increasing payee_risk response"
    assert np.all(np.diff(constrained_scores) >= -1e-12)


def test_candidate_models_route_each_row_to_its_own_booster() -> None:
    matrix = np.array(
        [
            [0.1, 1.0],
            [0.2, 2.0],
            [0.8, 3.0],
            [0.9, 4.0],
            [math.nan, 1.0],
            [math.nan, 2.0],
            [math.nan, 3.0],
            [math.nan, 4.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    masks = segment_masks(matrix[:, 0])
    present_rows = [0, 1, 2, 3]
    absent_rows = [4, 5, 6, 7]

    segmented = fit_candidate_from_params(
        "C2",
        {"present": ROUTING_PARAMS, "nan": ROUTING_PARAMS},
        None,
        matrix,
        labels,
        masks,
        "cpu",
        11,
    )
    scores = segmented.predict(matrix)
    assert len(set(scores.tolist())) > 1, "the two segment models must disagree on some rows"
    np.testing.assert_allclose(
        scores[present_rows], segmented.models["present"].predict_proba(matrix[present_rows])[:, 1]
    )
    np.testing.assert_allclose(scores[absent_rows], segmented.models["nan"].predict_proba(matrix[absent_rows])[:, 1])

    single = fit_candidate_from_params("C1", {"all": ROUTING_PARAMS}, None, matrix, labels, masks, "cpu", 11)
    np.testing.assert_allclose(single.predict(matrix), single.models["all"].predict_proba(matrix)[:, 1])

    malformed = CandidateModel(
        candidate="C1",
        payee_risk_index=0,
        models={"all": single.models["all"], "nan": segmented.models["nan"]},
    )
    with pytest.raises(ValueError, match="single-model"):
        malformed.predict(matrix)


def test_pooled_truth_ap_ranks_both_segments_as_one_sequence() -> None:
    roles = ["mule", "genuine", "cash_out", "genuine"]
    scores = [0.9, 0.8, 0.1, 0.2]
    assert pooled_truth_ap(scores, roles) == pytest.approx(0.75)


def test_world_truth_ap_groups_rows_and_leaves_single_class_worlds_null() -> None:
    world_ids = ["w1", "w1", "w2", "w2", "w3", "w3", "w4", "w4"]
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.4, 0.3]
    labels = [1, 0, 0, 1, 1, 1, 0, 0]
    rows = world_truth_ap(world_ids, scores, labels)

    assert [row["world_id"] for row in rows] == ["w1", "w2", "w3", "w4"]
    assert [(row["rows"], row["positives"]) for row in rows] == [(2, 1), (2, 1), (2, 2), (2, 0)]
    assert rows[0]["truth_ap"] == pytest.approx(1.0)
    assert rows[1]["truth_ap"] == pytest.approx(0.5)
    assert rows[2]["truth_ap"] is None
    assert rows[3]["truth_ap"] is None
    with pytest.raises(ValueError, match="differ in length"):
        world_truth_ap(world_ids, scores, labels[:-1])


def test_world_p10_is_the_linear_cross_world_aggregate() -> None:
    rows: list[dict[str, Any]] = [
        {"world_id": "w1", "truth_ap": 0.1},
        {"world_id": "w2", "truth_ap": 0.2},
        {"world_id": "w3", "truth_ap": 0.3},
        {"world_id": "w4", "truth_ap": None},
        {"world_id": "w5", "truth_ap": 0.4},
    ]
    aggregated = world_p10(rows)
    assert aggregated["worlds_used"] == 4
    assert aggregated["median"] == pytest.approx(0.25)
    assert aggregated["p10"] == pytest.approx(0.13)
    assert aggregated["p90"] == pytest.approx(0.37)


def test_selection_uses_inclusive_p10_eligibility_then_the_fixed_tie_priority() -> None:
    inclusive = select_candidate(
        [
            {"candidate": "C1", "validation_ap": 0.60, "validation_p10": 0.50},
            {"candidate": "C2", "validation_ap": 0.70, "validation_p10": 0.49},
        ],
        0.50,
    )
    assert inclusive["eligible"] == ["C1"]
    assert inclusive["winner"] == "C1"

    tied = select_candidate(
        [
            {"candidate": "C1", "validation_ap": 0.600, "validation_p10": 0.55},
            {"candidate": "C2", "validation_ap": 0.590, "validation_p10": 0.55},
            {"candidate": "C3", "validation_ap": 0.601, "validation_p10": 0.55},
        ],
        0.50,
    )
    assert tied["tied"] == ["C1", "C3"]
    assert tied["winner"] == "C1"

    none_eligible = select_candidate(
        [{"candidate": "C1", "validation_ap": 0.70, "validation_p10": 0.40}],
        0.50,
    )
    assert none_eligible["winner"] is None
    assert "no candidate" in none_eligible["reason"]

    with pytest.raises(ValueError, match="unknown candidate"):
        select_candidate([{"candidate": "C4", "validation_ap": 0.7, "validation_p10": 0.5}], 0.5)
    with pytest.raises(ValueError, match="validation_ap"):
        select_candidate([{"candidate": "C1", "validation_ap": None, "validation_p10": 0.5}], 0.5)


def test_the_replacement_gate_is_inclusive_at_the_literal_ap_floor_and_v1_p10() -> None:
    at_floor = replacement_decision(0.4659, 0.30, 0.30)
    assert at_floor["validation_ap_floor"] == 0.4659
    assert at_floor["replaces_v1"] is True

    below_floor = replacement_decision(0.465899999, 0.30, 0.30)
    assert below_floor["replaces_v1"] is False

    below_p10 = replacement_decision(0.5, 0.2999, 0.30)
    assert below_p10["replaces_v1"] is False

    above_both = replacement_decision(0.5, 0.31, 0.30)
    assert above_both["replaces_v1"] is True


def test_written_candidate_artifacts_are_verified_by_hash_and_loaded(tmp_path: Path) -> None:
    features = ["payee_risk", "amount"]
    order_path = _feature_order(tmp_path / "config", features)
    matrix = np.array([[0.1, 1.0], [0.9, 2.0], [math.nan, 3.0], [0.8, 4.0]], dtype=np.float32)
    labels = np.array([1, 0, 0, 1])
    masks = segment_masks(matrix[:, 0])
    model = fit_candidate_from_params("C1", {"all": SMALL_PARAMS}, None, matrix, labels, masks, "cpu", 5)

    written = write_candidate_artifacts(
        tmp_path / "C1",
        "C1",
        model,
        features,
        order_path,
        {
            "feature_sha256": feature_list_sha256(features),
            "feature_count": len(features),
            "seed": 5,
            "device": "cpu",
            "best_params": {"all": {"n_estimators": SMALL_PARAMS["n_estimators"]}},
        },
    )
    assert written["structure"] == "all"
    assert written["monotone_constraints"] == [1, 0]
    verify_candidate_artifacts(tmp_path / "C1")

    loaded = load_candidate_model(tmp_path / "C1")
    np.testing.assert_allclose(loaded.predict(matrix), model.predict(matrix))

    booster_path = tmp_path / "C1" / "frozen_model.json"
    booster_path.write_bytes(booster_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen_model.json"):
        verify_candidate_artifacts(tmp_path / "C1")


def test_the_winner_scoring_guard_accepts_only_the_matching_scoring_split(tmp_path: Path) -> None:
    test_folder = _manifest(tmp_path / "test-a", "test")
    stress_folder = _manifest(tmp_path / "stress-a", "stress")
    validation_folder = _manifest(tmp_path / "validation-a", "validation")

    assert validate_winner_scoring_folder(test_folder, "test")["split"] == "test"
    assert validate_winner_scoring_folder(stress_folder, "stress")["split"] == "stress"
    with pytest.raises(ValueError):
        validate_winner_scoring_folder(test_folder, "stress")
    with pytest.raises(ValueError):
        validate_winner_scoring_folder(stress_folder, "test")
    with pytest.raises(ValueError):
        validate_winner_scoring_folder(validation_folder, "test")
