"""
WHAT: Equivalence tests for evaluation/compact_rows.py: the per-world compact arrays must equal the
single-call training-row builder and one-row-per-account-day selector, row for row.

WHY: The compact loader is only usable if it is an exact data-path change. A single reordered or
rounded row would move every validation number, so the tests compare the float32 matrix (NaN
positions included), the targets, the row keys, and the validation account-day features, roles and
depths against the existing production functions on the committed fixtures.
"""

from datetime import date
from pathlib import Path

import numpy
import pyarrow.parquet as pq

from core.event_clock import IST
from decisions.mule_account_model import account_model_feature_names, build_training_rows
from decisions.rules_config import load_rules_config
from evaluation.compact_rows import load_compact_split
from evaluation.dataset_truth_join import join_truth
from evaluation.level17_scoring import one_row_per_account_day, score_account_days
from evaluation.rules_tuning import build_rules_day_matrix
from showcase.level_17_checks import TEST_DATASET, TRAINING_DATASET, VALIDATION_DATASET

TRUTH_ROOT = Path(__file__).resolve().parents[3] / "showcase" / "fixtures" / "level_17" / "truth"


def _old_training(
    dataset_dir: Path,
    truth_root: Path | None = None,
) -> tuple[list[list[float]], list[int], list[tuple[str, str, str, date]], list[dict[str, object]]]:
    """Runs the existing one-call path and returns its matrix, targets, keys and snapshot rows."""
    if truth_root is None:
        snapshots_rows = pq.read_table(dataset_dir / "snapshots.parquet").to_pylist()
    else:
        _, snapshots = join_truth(dataset_dir, truth_root)
        snapshots_rows = snapshots.to_pylist()
    label_rows = pq.read_table(dataset_dir / "labels.parquet").to_pylist()
    matrix, targets, row_keys = build_training_rows(snapshots_rows, label_rows)
    return matrix, targets, row_keys, snapshots_rows


def _assert_keys_equal(
    world_ids: numpy.ndarray,
    bank_ids: numpy.ndarray,
    account_ids: numpy.ndarray,
    day_ordinals: numpy.ndarray,
    row_keys: list[tuple[str, str, str, date]],
) -> None:
    """Asserts the compact key arrays equal the builder's key tuples, in the same order."""
    assert list(world_ids) == [key[0] for key in row_keys]
    assert list(bank_ids) == [key[1] for key in row_keys]
    assert list(account_ids) == [key[2] for key in row_keys]
    assert list(day_ordinals) == [key[3].toordinal() for key in row_keys]


def test_compact_training_rows_match_the_single_call_matrix() -> None:
    """Three fixtures: float32 matrix, targets and keys equal the existing builder exactly."""
    for dataset_dir in (TRAINING_DATASET, VALIDATION_DATASET, TEST_DATASET):
        old_matrix, old_targets, old_keys, _ = _old_training(dataset_dir)

        split = load_compact_split([dataset_dir])

        assert split.rows.feature_names == account_model_feature_names()
        numpy.testing.assert_array_equal(
            split.rows.matrix,
            numpy.asarray(old_matrix, dtype=numpy.float32),
        )
        assert split.rows.matrix.dtype == numpy.float32
        assert list(split.rows.targets) == old_targets
        _assert_keys_equal(
            split.rows.world_ids,
            split.rows.bank_ids,
            split.rows.account_ids,
            split.rows.day_ordinals,
            old_keys,
        )
        assert split.days is None


def test_compact_validation_days_equal_one_row_per_account_day_with_truth() -> None:
    """The joined validation account-days match the existing dedupe on features, roles and depths,
    and the precomputed ScoredDays, rules matrix and aligned features match the existing producers."""
    old_matrix, old_targets, old_keys, snapshot_rows = _old_training(VALIDATION_DATASET, TRUTH_ROOT)
    day_rows = one_row_per_account_day(snapshot_rows)
    feature_names = account_model_feature_names()
    rules_config = load_rules_config()

    split = load_compact_split(
        [VALIDATION_DATASET],
        TRUTH_ROOT,
        include_days=True,
        rules_config=rules_config,
    )

    numpy.testing.assert_array_equal(
        split.rows.matrix,
        numpy.asarray(old_matrix, dtype=numpy.float32),
    )
    assert list(split.rows.targets) == old_targets
    _assert_keys_equal(
        split.rows.world_ids,
        split.rows.bank_ids,
        split.rows.account_ids,
        split.rows.day_ordinals,
        old_keys,
    )

    assert split.days is not None
    assert not hasattr(split, "day_rows")
    expected_features = numpy.asarray(
        [[row.get(name) for name in feature_names] for row in day_rows],
        dtype=numpy.float64,
    )
    numpy.testing.assert_array_equal(split.days.features, expected_features)
    _assert_keys_equal(
        split.days.world_ids,
        split.days.bank_ids,
        split.days.account_ids,
        split.days.day_ordinals,
        [
            (row["world_id"], row["bank_id"], row["account_id"], row["snapshot_time"].astimezone(IST).date())
            for row in day_rows
        ],
    )
    assert list(split.days.roles) == [row["truth_role"] for row in day_rows]
    assert list(split.days.depths) == [row["truth_depth"] for row in day_rows]

    assert list(split.days.scored) == score_account_days(day_rows, rules_config, None)

    expected_rules = build_rules_day_matrix(day_rows, rules_config)
    assert tuple(split.days.rules_matrix.features) == tuple(expected_rules.features)
    for name, column in expected_rules.features.items():
        numpy.testing.assert_array_equal(split.days.rules_matrix.features[name], column)
    assert list(split.days.rules_matrix.account_ids) == list(expected_rules.account_ids)
    assert list(split.days.rules_matrix.day_ordinals) == list(expected_rules.day_ordinals)
    assert list(split.days.rules_matrix.roles) == list(expected_rules.roles)

    rows_by_key: dict[tuple[str, str, str, date], dict[str, object]] = {}
    for row in day_rows:
        row_key = (
            str(row["world_id"]),
            str(row["bank_id"]),
            str(row["account_id"]),
            row["snapshot_time"].astimezone(IST).date(),
        )
        rows_by_key[row_key] = row
    expected_aligned = numpy.asarray(
        [[rows_by_key[key].get(name) for name in feature_names] for key in old_keys],
        dtype=numpy.float32,
    )
    numpy.testing.assert_array_equal(split.days.aligned_features, expected_aligned)


def test_world_subset_loads_only_the_requested_world() -> None:
    """Restricting the load to one world yields exactly that world's rows and keys."""
    full = load_compact_split([TRAINING_DATASET])
    world_ids = sorted(set(full.rows.world_ids))
    assert len(world_ids) == 2

    subset = load_compact_split([TRAINING_DATASET], world_ids=[world_ids[1]])

    assert set(subset.rows.world_ids) == {world_ids[1]}
    keep = numpy.asarray(full.rows.world_ids) == world_ids[1]
    numpy.testing.assert_array_equal(subset.rows.matrix, full.rows.matrix[keep])
    list_targets = list(subset.rows.targets)
    assert list_targets == list(full.rows.targets[keep])
