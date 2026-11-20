"""
WHAT: Tests that loading only the detector's columns changes nothing the detector computes.

WHY: _detector_snapshot_rows exists to cut memory at full scale. It is only safe if the world starts
and the fitted detector are exactly what the all-columns rows would give, and if a missing column
fails loudly instead of silently training on less.
"""

import numpy
import pytest

from decisions.anomaly_config import input_feature_names, load_anomaly_detector_config
from decisions.anomaly_detector import score_rows, train_anomaly_detector
from evaluation.level17_evaluation import world_start_by_world
from evaluation.run_level_17 import load_split_datasets
from evaluation.run_level_18_pilot import _detector_snapshot_rows
from showcase.level_17_checks import TRAINING_DATASET


def test_narrow_rows_give_identical_world_starts_and_detector_scores() -> None:
    config = load_anomaly_detector_config()
    datasets = load_split_datasets("training", [TRAINING_DATASET])
    full_rows = [row for dataset in datasets for row in dataset.snapshots.to_pylist()]
    narrow_rows = _detector_snapshot_rows(datasets, config)

    full_starts = world_start_by_world(full_rows)
    narrow_starts = world_start_by_world(narrow_rows)
    full_detector = train_anomaly_detector(full_rows, "training", full_starts, config)
    narrow_detector = train_anomaly_detector(narrow_rows, "training", narrow_starts, config)

    assert len(narrow_rows) == len(full_rows)
    assert narrow_starts == full_starts
    full_scores = score_rows(full_detector, full_rows)
    narrow_scores = score_rows(narrow_detector, full_rows)
    assert numpy.array_equal(numpy.asarray(full_scores), numpy.asarray(narrow_scores))
    # The row digest hashes every column of the kept rows, so it is expected to differ; the same
    # early rows must still be kept.
    assert narrow_detector.training_row_count == full_detector.training_row_count


def test_narrow_rows_hold_only_the_needed_columns_in_row_order() -> None:
    config = load_anomaly_detector_config()
    datasets = load_split_datasets("training", [TRAINING_DATASET])
    full_rows = [row for dataset in datasets for row in dataset.snapshots.to_pylist()]

    narrow_rows = _detector_snapshot_rows(datasets, config)

    assert set(narrow_rows[0]) == {"world_id", "snapshot_time", *input_feature_names(config)}
    assert [row["world_id"] for row in narrow_rows] == [row["world_id"] for row in full_rows]
    assert [row["snapshot_time"] for row in narrow_rows] == [row["snapshot_time"] for row in full_rows]


def test_a_missing_required_column_raises_naming_it() -> None:
    config = load_anomaly_detector_config()
    datasets = load_split_datasets("training", [TRAINING_DATASET])
    datasets[0].snapshots = datasets[0].snapshots.drop_columns(["world_id"])

    with pytest.raises(ValueError, match="world_id"):
        _detector_snapshot_rows(datasets, config)
