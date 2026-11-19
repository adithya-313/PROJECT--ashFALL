"""
WHAT: Unit tests for PU bagging and weighted Elkan-Noto in decisions/pu_learning.py.

WHY: PU learning changes which rows each member sees and how unlabelled rows are weighted, so every
test checks the exact contract: bag composition, margin averaging, SHAP additivity, persistence,
and the Elkan-Noto c estimate and weight formula including its clip.
"""

from pathlib import Path

import numpy
import pytest

import decisions.pu_learning as pu_learning
from decisions.mule_account_model import train_account_model
from decisions.pu_learning import (
    PUBaggedAccountModel,
    _elkan_noto_weights,
    _positive_probability_mean,
    train_elkan_noto_model,
    train_pu_bagged_model,
)


def _matrix_and_targets() -> tuple[list[list[float]], list[int]]:
    """Builds a small matrix whose three positive rows come first, then twelve unlabelled rows."""
    rows = [
        [0.0, 0.1],
        [0.1, 0.9],
        [0.9, 0.2],
        [0.2, 0.3],
        [0.4, 0.7],
        [0.6, 0.1],
        [0.3, 0.6],
        [0.7, 0.4],
        [0.5, 0.5],
        [0.8, 0.9],
        [0.9, 0.6],
        [0.1, 0.4],
        [0.2, 0.8],
        [0.7, 0.2],
        [0.4, 0.9],
    ]
    targets = [1, 1, 1] + [0] * 12
    return rows, targets


def _probe_rows() -> list[list[float]]:
    """Returns a few feature rows used to compare predictions."""
    return [[0.15, 0.25], [0.85, 0.75], [0.45, 0.55]]


def test_one_full_coverage_bag_equals_the_full_data_model() -> None:
    """A single bag that draws every unlabelled row reproduces the full-data model exactly."""
    rows, targets = _matrix_and_targets()

    ensemble = train_pu_bagged_model(rows, targets, seed=3, n_estimators=5, n_bags=1, unlabelled_per_positive=100)
    full_model, _ = train_account_model(rows, targets, seed=3, n_estimators=5)

    numpy.testing.assert_array_equal(
        numpy.asarray(ensemble.predict_margins(_probe_rows())),
        full_model.predict(_probe_rows(), output_margin=True),
    )


def test_each_bag_has_all_positives_and_the_formula_sample_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every bag passes all positive rows and exactly min(unlabelled, ratio * positives) negatives."""
    rows, targets = _matrix_and_targets()
    captured: list[tuple[list[list[float]], list[int], int]] = []

    def fake_train(
        bag_rows: list[list[float]],
        bag_targets: list[int],
        seed: int,
        n_estimators: int = 100,
        *,
        sample_weight: list[float] | None = None,
        scale_pos_weight_override: float | None = None,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
    ) -> tuple[object, float]:
        """Records each bag's rows, targets and seed and returns a stand-in model."""
        captured.append((bag_rows, bag_targets, seed))
        return object(), 1.0

    monkeypatch.setattr(pu_learning, "train_account_model", fake_train)

    train_pu_bagged_model(rows, targets, seed=5, n_estimators=10, n_bags=4, unlabelled_per_positive=2)

    assert len(captured) == 4
    expected_sample_size = min(12, 2 * 3)
    for bag_index, (bag_rows, bag_targets, bag_seed) in enumerate(captured):
        assert bag_seed == 5 + bag_index
        assert bag_rows[:3] == rows[:3]
        assert bag_targets[:3] == [1, 1, 1]
        assert len(bag_rows) == 3 + expected_sample_size
        assert len([target for target in bag_targets if target == 0]) == expected_sample_size
        assert len(set(map(tuple, bag_rows[3:]))) == expected_sample_size
        for sampled_row in bag_rows[3:]:
            assert sampled_row in rows[3:]


def test_same_seed_reproduces_and_a_different_seed_changes_the_ensemble() -> None:
    """Sampling and fitting are seeded, so one seed is repeatable and another seed is not."""
    rows, targets = _matrix_and_targets()
    probe = _probe_rows()

    first = train_pu_bagged_model(rows, targets, seed=11, n_estimators=5, n_bags=3, unlabelled_per_positive=2)
    second = train_pu_bagged_model(rows, targets, seed=11, n_estimators=5, n_bags=3, unlabelled_per_positive=2)
    other = train_pu_bagged_model(rows, targets, seed=12, n_estimators=5, n_bags=3, unlabelled_per_positive=2)

    numpy.testing.assert_array_equal(
        numpy.asarray(first.predict_margins(probe)),
        numpy.asarray(second.predict_margins(probe)),
    )
    assert not numpy.array_equal(
        numpy.asarray(first.predict_margins(probe)),
        numpy.asarray(other.predict_margins(probe)),
    )


def test_predict_margins_is_the_hand_computed_member_mean() -> None:
    """The ensemble margin is exactly the arithmetic mean of the members' margins."""
    rows, targets = _matrix_and_targets()
    probe = _probe_rows()
    ensemble = train_pu_bagged_model(rows, targets, seed=7, n_estimators=4, n_bags=3, unlabelled_per_positive=2)

    member_margins = numpy.vstack([member.predict(probe, output_margin=True) for member in ensemble.members]).astype(
        float
    )
    expected = numpy.mean(member_margins, axis=0)

    numpy.testing.assert_array_equal(numpy.asarray(ensemble.predict_margins(probe)), expected)
    probabilities = numpy.asarray(ensemble.predict_probabilities(probe))
    numpy.testing.assert_allclose(probabilities, 1.0 / (1.0 + numpy.exp(-expected)), rtol=0.0)


def test_mean_shap_plus_mean_expected_value_reconstructs_the_margins() -> None:
    """Averaged SHAP rows plus averaged expected values sum to the averaged margins."""
    import shap

    rows, targets = _matrix_and_targets()
    probe = _probe_rows()
    ensemble = train_pu_bagged_model(rows, targets, seed=13, n_estimators=4, n_bags=3, unlabelled_per_positive=2)

    values = ensemble.shap_values(probe)
    expected_values: list[float] = []
    for member in ensemble.members:
        explainer = shap.TreeExplainer(member)
        # Tree-path-dependent explainers fill expected_value only while shap_values runs.
        explainer.shap_values(probe)
        expected_values.append(float(numpy.asarray(explainer.expected_value).reshape(-1)[0]))
    reconstructed = values.sum(axis=1) + numpy.mean(expected_values)

    assert values.shape == (len(probe), len(probe[0]))
    numpy.testing.assert_allclose(reconstructed, numpy.asarray(ensemble.predict_margins(probe)), atol=1e-4)


def test_save_and_load_round_trip_gives_identical_predictions(tmp_path: Path) -> None:
    """The sidecar rebuilds every member and the same sampling settings without a score change."""
    rows, targets = _matrix_and_targets()
    probe = _probe_rows()
    ensemble = train_pu_bagged_model(rows, targets, seed=17, n_estimators=4, n_bags=3, unlabelled_per_positive=2)

    ensemble.save(tmp_path, {"training_dataset_ids": ["fixture-training"]})
    loaded = PUBaggedAccountModel.load(tmp_path)

    assert loaded.member_seeds == ensemble.member_seeds
    assert loaded.n_bags == ensemble.n_bags
    assert loaded.n_estimators == ensemble.n_estimators
    assert loaded.unlabelled_per_positive == ensemble.unlabelled_per_positive
    numpy.testing.assert_array_equal(
        numpy.asarray(loaded.predict_margins(probe)),
        numpy.asarray(ensemble.predict_margins(probe)),
    )


def test_positive_probability_mean_is_the_mean_over_positive_rows() -> None:
    """c averages only the rows whose target is 1."""
    targets = [1, 0, 1, 0, 1]
    probabilities = [0.2, 0.9, 0.4, 0.9, 0.6]

    c = _positive_probability_mean(probabilities, targets)

    assert c == pytest.approx((0.2 + 0.4 + 0.6) / 3)


def test_positive_probability_mean_refuses_a_validation_set_without_positives() -> None:
    """A validation set with no labelled row cannot estimate c and is refused."""
    with pytest.raises(ValueError, match="at least one positive validation row"):
        _positive_probability_mean([0.2, 0.8], [0, 0])


def test_elkan_noto_weight_formula_on_hand_values_including_the_clip() -> None:
    """The weight is ((1 - c) / c) * p / (1 - p), capped at 1, with the clip count recorded."""
    weights, clip_count = _elkan_noto_weights([0.1, 0.5, 0.9], c=0.5, probability_clip=0.8)

    assert weights[0] == pytest.approx(0.1 / 0.9)
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)
    assert clip_count == 1


def test_elkan_noto_weight_formula_direction_and_probability_clip() -> None:
    """With c = 0.2 the factor is 4, not 1, and the probability clip is applied before the formula."""
    weights, clip_count = _elkan_noto_weights([0.1, 0.25], c=0.2, probability_clip=0.999999)
    clipped_weights, clipped_count = _elkan_noto_weights([0.1], c=0.2, probability_clip=0.05)

    assert weights[0] == pytest.approx(0.4444444444444444)
    assert weights[1] == pytest.approx(1.0)
    assert clip_count == 1
    assert clipped_weights[0] == pytest.approx(0.21052631578947367)
    assert clipped_count == 0


def test_elkan_noto_end_to_end_records_the_estimated_c_and_weight_summary() -> None:
    """A real fit estimates c from g on validation and reports weights within [0, 1]."""
    train_matrix, train_targets = _matrix_and_targets()
    validation_matrix = [[0.05, 0.15], [0.95, 0.85], [0.3, 0.4], [0.6, 0.6], [0.2, 0.7]]
    validation_targets = [1, 1, 0, 0, 0]
    seed = 19
    n_estimators = 5

    model, c, summary, clip_count = train_elkan_noto_model(
        train_matrix,
        train_targets,
        validation_matrix,
        validation_targets,
        seed,
        n_estimators,
        probability_clip=0.999999,
    )

    g, _ = train_account_model(
        train_matrix,
        train_targets,
        seed,
        n_estimators,
        scale_pos_weight_override=1.0,
    )
    g_validation = g.predict_proba(validation_matrix)[:, 1]
    expected_c = float(numpy.mean([g_validation[0], g_validation[1]]))

    assert c == pytest.approx(expected_c)
    assert 0.0 <= summary["min"] <= summary["median"] <= summary["max"] <= 1.0
    assert clip_count >= 0
    assert len(model.predict_proba(validation_matrix)) == len(validation_matrix)


def test_elkan_noto_refuses_a_validation_set_without_positives() -> None:
    """Elkan-Noto cannot estimate c without a labelled validation row and stops naming the reason."""
    train_matrix, train_targets = _matrix_and_targets()
    validation_matrix = [[0.05, 0.15], [0.95, 0.85]]

    with pytest.raises(ValueError, match="at least one positive validation row"):
        train_elkan_noto_model(
            train_matrix,
            train_targets,
            validation_matrix,
            [0, 0],
            19,
            5,
            probability_clip=0.999999,
        )
