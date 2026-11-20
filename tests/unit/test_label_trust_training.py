"""
WHAT: Tests that label trust weights reach every account model trainer and that leaving them off
changes nothing.

WHY: Weights are only safe to ship if the unweighted calls stay bit-for-bit what they were, and if
every trainer applies each row's own weight. The bit-identical tests compare against the old call
rebuilt by hand, so they cannot pass just because both sides run the same new code.
"""

import numpy
import pytest
import xgboost

import decisions.pu_learning as pu_learning
from decisions.mule_account_model import train_account_model
from decisions.pu_learning import (
    _elkan_noto_weights,
    _positive_probability_mean,
    train_elkan_noto_model,
    train_pu_bagged_model,
)

ROW_COUNT = 120
FEATURE_COUNT = 4
N_ESTIMATORS = 5
SEED = 11


def _matrix_and_targets() -> tuple[numpy.ndarray, list[int]]:
    """Returns a seeded random float32 matrix and targets with every fifth row positive."""
    generator = numpy.random.default_rng(2024)
    matrix = generator.random((ROW_COUNT, FEATURE_COUNT)).astype(numpy.float32)
    targets = [1 if index % 5 == 0 else 0 for index in range(ROW_COUNT)]
    return matrix, targets


def _probe_matrix() -> numpy.ndarray:
    """Returns a fixed seeded matrix used to compare predictions."""
    return numpy.random.default_rng(99).random((40, FEATURE_COUNT)).astype(numpy.float32)


def _old_plain_model(matrix: numpy.ndarray, targets: list[int]) -> xgboost.XGBClassifier:
    """Rebuilds the pre-weighting trainer call by hand: count ratio, no sample weights."""
    positives = sum(1 for target in targets if target == 1)
    negatives = sum(1 for target in targets if target == 0)
    model = xgboost.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=SEED,
        scale_pos_weight=negatives / positives,
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.3,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
    )
    model.fit(matrix, targets)
    return model


def test_plain_trainer_without_weights_is_bit_identical_to_the_old_call() -> None:
    matrix, targets = _matrix_and_targets()

    new_model, _ = train_account_model(matrix, targets, SEED, N_ESTIMATORS, sample_weight=None)

    numpy.testing.assert_array_equal(
        new_model.predict_proba(_probe_matrix()),
        _old_plain_model(matrix, targets).predict_proba(_probe_matrix()),
    )


def test_plain_trainer_with_all_ones_weights_changes_nothing() -> None:
    matrix, targets = _matrix_and_targets()
    ones = numpy.ones(ROW_COUNT, dtype=numpy.float32)

    weighted_model, weighted_ratio = train_account_model(matrix, targets, SEED, N_ESTIMATORS, sample_weight=ones)
    plain_model, plain_ratio = train_account_model(matrix, targets, SEED, N_ESTIMATORS)

    assert weighted_ratio == plain_ratio
    numpy.testing.assert_array_equal(
        weighted_model.predict_proba(_probe_matrix()),
        plain_model.predict_proba(_probe_matrix()),
    )


def test_bagging_without_weights_is_bit_identical_to_the_old_member_loop() -> None:
    matrix, targets = _matrix_and_targets()
    n_bags = 3
    unlabelled_per_positive = 2

    ensemble = train_pu_bagged_model(
        matrix, targets, SEED, N_ESTIMATORS, n_bags, unlabelled_per_positive, sample_weight=None
    )

    # The old loop rebuilt by hand: uniform seeded draw, positives first, one plain fit per member.
    positive_indices = [index for index, target in enumerate(targets) if target == 1]
    unlabelled_array = numpy.asarray([index for index, target in enumerate(targets) if target == 0])
    sample_size = min(len(unlabelled_array), unlabelled_per_positive * len(positive_indices))
    old_members = []
    for bag_index in range(n_bags):
        generator = numpy.random.default_rng(SEED + bag_index)
        sampled = generator.choice(unlabelled_array, size=sample_size, replace=False)
        ordered = positive_indices + sorted(int(index) for index in sampled)
        member, _ = train_account_model(
            [matrix[index] for index in ordered], [targets[index] for index in ordered], SEED + bag_index, N_ESTIMATORS
        )
        old_members.append(member)

    for new_member, old_member in zip(ensemble.members, old_members, strict=True):
        numpy.testing.assert_array_equal(
            new_member.predict(_probe_matrix(), output_margin=True),
            old_member.predict(_probe_matrix(), output_margin=True),
        )


def test_elkan_noto_without_weights_is_bit_identical_to_the_old_path() -> None:
    matrix, targets = _matrix_and_targets()
    validation_matrix, validation_targets = _matrix_and_targets()
    probability_clip = 0.95

    new_model, new_c, _, _ = train_elkan_noto_model(
        matrix, targets, validation_matrix, validation_targets, SEED, N_ESTIMATORS, probability_clip, sample_weight=None
    )

    # The old path rebuilt by hand: unbalanced g, c on validation positives, doubled unlabelled rows.
    g, _ = train_account_model(matrix, targets, SEED, N_ESTIMATORS, scale_pos_weight_override=1.0)
    c = _positive_probability_mean([float(v) for v in g.predict_proba(validation_matrix)[:, 1]], validation_targets)
    g_train = [float(v) for v in g.predict_proba(matrix)[:, 1]]
    unlabelled_probabilities = [p for p, t in zip(g_train, targets, strict=True) if t == 0]
    unlabelled_weights, _ = _elkan_noto_weights(unlabelled_probabilities, c, probability_clip)
    expanded_rows = [row for row, t in zip(matrix, targets, strict=True) if t == 1]
    expanded_targets = [1] * len(expanded_rows)
    sample_weights = [1.0] * len(expanded_rows)
    unlabelled_rows = [row for row, t in zip(matrix, targets, strict=True) if t == 0]
    for row, weight in zip(unlabelled_rows, unlabelled_weights, strict=True):
        expanded_rows.extend([row, row])
        expanded_targets.extend([1, 0])
        sample_weights.extend([weight, 1.0 - weight])
    positive_total = sum(w for w, t in zip(sample_weights, expanded_targets, strict=True) if t == 1)
    negative_total = sum(w for w, t in zip(sample_weights, expanded_targets, strict=True) if t == 0)
    old_model, _ = train_account_model(
        expanded_rows,
        expanded_targets,
        SEED,
        N_ESTIMATORS,
        sample_weight=sample_weights,
        scale_pos_weight_override=negative_total / positive_total,
    )

    assert new_c == c
    numpy.testing.assert_array_equal(
        new_model.predict_proba(_probe_matrix()),
        old_model.predict_proba(_probe_matrix()),
    )


def test_each_bagging_member_receives_the_weights_of_its_own_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    row_count = 30
    # The first feature of every row is its own index, so a member's rows identify its indices.
    matrix = numpy.zeros((row_count, 2), dtype=numpy.float32)
    matrix[:, 0] = numpy.arange(row_count)
    targets = [1 if index < 3 else 0 for index in range(row_count)]
    weights = numpy.asarray([0.5 + index / 100 for index in range(row_count)], dtype=numpy.float32)
    recorded: list[tuple[list[numpy.ndarray], numpy.ndarray]] = []

    def fake_train(bag_rows, bag_targets, seed, n_estimators=100, *, sample_weight=None, **_: object):  # type: ignore[no-untyped-def]
        recorded.append((bag_rows, sample_weight))
        return xgboost.XGBClassifier(), 1.0

    monkeypatch.setattr(pu_learning, "train_account_model", fake_train)

    train_pu_bagged_model(matrix, targets, SEED, N_ESTIMATORS, 3, 2, sample_weight=weights)

    assert len(recorded) == 3
    for bag_rows, member_weights in recorded:
        expected = numpy.asarray([weights[int(row[0])] for row in bag_rows], dtype=numpy.float32)
        numpy.testing.assert_array_equal(member_weights, expected)


def test_weighted_scale_pos_weight_is_the_weight_ratio_by_hand() -> None:
    matrix = numpy.random.default_rng(5).random((5, 2)).astype(numpy.float32)
    targets = [1, 0, 0, 1, 0]
    weights = [1.0, 0.5, 0.25, 1.0, 0.25]

    _, ratio = train_account_model(matrix, targets, SEED, 2, sample_weight=weights)

    # Target-0 weights sum to 1.0 and target-1 weights sum to 2.0.
    assert ratio == 0.5


def test_weighted_scale_pos_weight_refuses_a_class_with_zero_weight() -> None:
    matrix = numpy.random.default_rng(5).random((4, 2)).astype(numpy.float32)
    targets = [1, 0, 0, 1]

    with pytest.raises(ValueError, match="target-0 rows have a zero total"):
        train_account_model(matrix, targets, SEED, 2, sample_weight=[1.0, 0.0, 0.0, 1.0])


def test_override_wins_over_the_weighted_ratio() -> None:
    matrix = numpy.random.default_rng(5).random((4, 2)).astype(numpy.float32)
    targets = [1, 0, 0, 1]

    _, ratio = train_account_model(
        matrix, targets, SEED, 2, sample_weight=[1.0, 0.5, 0.5, 1.0], scale_pos_weight_override=3.0
    )

    assert ratio == 3.0


def test_elkan_noto_multiplies_both_copies_of_each_unlabelled_row_by_its_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, targets = _matrix_and_targets()
    validation_matrix, validation_targets = _matrix_and_targets()
    trust = numpy.asarray([1.0 if target == 1 else 0.25 + (index % 4) / 8 for index, target in enumerate(targets)])
    real_train = pu_learning.train_account_model
    recorded_weights: list[object] = []

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        recorded_weights.append(kwargs.get("sample_weight"))
        return real_train(*args, **kwargs)

    monkeypatch.setattr(pu_learning, "train_account_model", spy)

    train_elkan_noto_model(
        matrix, targets, validation_matrix, validation_targets, SEED, N_ESTIMATORS, 0.95, sample_weight=trust
    )

    assert recorded_weights[0] is trust
    final_weights = recorded_weights[1]
    assert isinstance(final_weights, list)
    positive_count = sum(targets)
    assert final_weights[:positive_count] == [1.0] * positive_count
    unlabelled_trust = [float(value) for value, target in zip(trust, targets, strict=True) if target == 0]
    for offset, expected_trust in enumerate(unlabelled_trust):
        positive_copy = final_weights[positive_count + 2 * offset]
        negative_copy = final_weights[positive_count + 2 * offset + 1]
        # w * trust + (1 - w) * trust = trust, whatever w is.
        assert positive_copy + negative_copy == pytest.approx(expected_trust)
