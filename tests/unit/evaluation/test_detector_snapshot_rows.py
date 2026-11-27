"""
WHAT: Tests that cutting the training rows to the detector's early window before building dicts
changes nothing the detector computes.

WHY: _early_window_detector_rows exists to cut memory at 320 worlds. It is only safe if its world
starts, kept rows and fitted detector are exactly what the full-rows path gives, if its cut sits on
the same strict boundary as train_anomaly_detector, and if bad input fails loudly.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import numpy
import pyarrow
import pytest

from decisions.anomaly_config import input_feature_names, load_anomaly_detector_config
from decisions.anomaly_detector import score_rows, train_anomaly_detector
from evaluation.level17_evaluation import world_start_by_world
from evaluation.run_level_17 import load_split_datasets
from evaluation.run_level_18_pilot import _early_window_detector_rows
from showcase.level_17_checks import TRAINING_DATASET

WORLD_START = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)


def _snapshot_dataset(
    world_ids: list[Any],
    times: list[Any],
    time_type: pyarrow.DataType | None = None,
    drop_column: str | None = None,
) -> Any:
    """Returns a stand-in dataset whose snapshots table holds the given ids and times and null features."""
    config = load_anomaly_detector_config()
    resolved_time_type = time_type if time_type is not None else pyarrow.timestamp("us", tz="UTC")
    columns: dict[str, pyarrow.Array] = {
        "world_id": pyarrow.array(world_ids, pyarrow.string()),
        "snapshot_time": pyarrow.array(times, resolved_time_type),
    }
    for feature_name in input_feature_names(config):
        columns[feature_name] = pyarrow.array([None] * len(world_ids), pyarrow.float64())
    if drop_column is not None:
        del columns[drop_column]
    return SimpleNamespace(snapshots=pyarrow.table(columns))


def test_early_window_rows_match_the_full_rows_path() -> None:
    config = load_anomaly_detector_config()
    datasets = load_split_datasets("training", [TRAINING_DATASET])
    needed_columns = ["world_id", "snapshot_time", *input_feature_names(config)]
    full_rows = [row for dataset in datasets for row in dataset.snapshots.select(needed_columns).to_pylist()]

    new_rows, new_starts = _early_window_detector_rows(datasets, config)

    old_starts = world_start_by_world(full_rows)
    old_detector = train_anomaly_detector(full_rows, "training", old_starts, config)
    new_detector = train_anomaly_detector(new_rows, "training", new_starts, config)
    assert new_starts == old_starts
    assert new_detector.training_row_count == old_detector.training_row_count
    assert len(new_rows) == old_detector.training_row_count
    assert new_detector.training_rows_sha256 == old_detector.training_rows_sha256
    old_scores = score_rows(old_detector, full_rows)
    new_scores = score_rows(new_detector, full_rows)
    assert numpy.array_equal(numpy.asarray(old_scores), numpy.asarray(new_scores))


def test_kept_rows_hold_only_the_needed_columns_in_row_order() -> None:
    config = load_anomaly_detector_config()
    datasets = load_split_datasets("training", [TRAINING_DATASET])

    kept_rows, _ = _early_window_detector_rows(datasets, config)

    assert set(kept_rows[0]) == {"world_id", "snapshot_time", *input_feature_names(config)}
    needed_columns = ["world_id", "snapshot_time", *input_feature_names(config)]
    full_rows = datasets[0].snapshots.select(needed_columns).to_pylist()
    remaining_full_rows = iter(full_rows)
    for kept_row in kept_rows:
        # Consuming the iterator while searching proves the kept rows appear in the same order.
        found = False
        for full_row in remaining_full_rows:
            if full_row == kept_row:
                found = True
                break
        assert found, f"kept row out of order or not in the dataset: {kept_row['world_id']}"


def test_a_row_exactly_at_the_window_edge_is_dropped_and_one_microsecond_earlier_is_kept() -> None:
    config = load_anomaly_detector_config()
    window = timedelta(days=config.early_window_days)
    one_microsecond = timedelta(microseconds=1)
    dataset = _snapshot_dataset(
        ["w1", "w1", "w1", "w1"],
        [
            WORLD_START,
            WORLD_START + window - one_microsecond,
            WORLD_START + window,
            WORLD_START + window + one_microsecond,
        ],
    )

    kept_rows, world_starts = _early_window_detector_rows([dataset], config)

    assert [row["snapshot_time"] for row in kept_rows] == [WORLD_START, WORLD_START + window - one_microsecond]
    assert world_starts == {"w1": WORLD_START}


def test_each_world_is_cut_from_its_own_start() -> None:
    config = load_anomaly_detector_config()
    window = timedelta(days=config.early_window_days)
    later_start = WORLD_START + timedelta(days=30)
    dataset = _snapshot_dataset(
        ["early", "late", "early", "late"],
        [WORLD_START, later_start, WORLD_START + window, later_start + window - timedelta(microseconds=1)],
    )

    kept_rows, world_starts = _early_window_detector_rows([dataset], config)

    assert [(row["world_id"], row["snapshot_time"]) for row in kept_rows] == [
        ("early", WORLD_START),
        ("late", later_start),
        ("late", later_start + window - timedelta(microseconds=1)),
    ]
    assert world_starts == {"early": WORLD_START, "late": later_start}


def test_a_timezone_naive_snapshot_time_raises() -> None:
    config = load_anomaly_detector_config()
    dataset = _snapshot_dataset(["w1"], [datetime(2026, 9, 1)], time_type=pyarrow.timestamp("us"))

    with pytest.raises(ValueError, match="timezone-aware"):
        _early_window_detector_rows([dataset], config)


def test_a_null_snapshot_time_raises() -> None:
    config = load_anomaly_detector_config()
    dataset = _snapshot_dataset(["w1", "w1"], [WORLD_START, None])

    with pytest.raises(ValueError, match="snapshot_time"):
        _early_window_detector_rows([dataset], config)


@pytest.mark.parametrize("bad_world_ids", [["w1", None], ["w1", ""]])
def test_a_null_or_empty_world_id_raises(bad_world_ids: list[Any]) -> None:
    config = load_anomaly_detector_config()
    dataset = _snapshot_dataset(bad_world_ids, [WORLD_START, WORLD_START])

    with pytest.raises(ValueError, match="world_id"):
        _early_window_detector_rows([dataset], config)


@pytest.mark.parametrize("missing_column", ["world_id", "snapshot_time"])
def test_a_missing_required_column_raises_naming_it(missing_column: str) -> None:
    config = load_anomaly_detector_config()
    dataset = _snapshot_dataset(["w1"], [WORLD_START], drop_column=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        _early_window_detector_rows([dataset], config)


def test_a_feature_column_missing_raises_naming_it() -> None:
    config = load_anomaly_detector_config()
    feature_name = input_feature_names(config)[0]
    dataset = _snapshot_dataset(["w1"], [WORLD_START], drop_column=feature_name)

    with pytest.raises(ValueError, match=feature_name):
        _early_window_detector_rows([dataset], config)


def test_a_world_id_in_two_datasets_raises() -> None:
    config = load_anomaly_detector_config()
    first = _snapshot_dataset(["w1"], [WORLD_START])
    second = _snapshot_dataset(["w1"], [WORLD_START + timedelta(days=1)])

    with pytest.raises(ValueError, match="more than one training dataset"):
        _early_window_detector_rows([first, second], config)
