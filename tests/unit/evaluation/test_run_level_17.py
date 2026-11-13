"""
WHAT: Unit tests for evaluation/run_level_17.py.

WHY: The runner must refuse a dataset whose recorded split differs from the flag it was passed
under, refuse a missing world card, key a result by its datasets alone, and mark every result as
simulated without recording any generation timestamp.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.event_clock import IST
from evaluation.level17_evaluation import evaluate_world
from evaluation.level17_scoring import ScoredDay
from evaluation.run_level_17 import (
    load_split_datasets,
    read_world_base_rate,
    run_key,
    summarise_worlds,
    write_result,
)


def _write_minimal_dataset(root: Path, name: str, split: str) -> Path:
    """Writes a loadable dataset folder whose manifest records the given split."""
    dataset_dir = root / name
    dataset_dir.mkdir()
    table = pa.table({"marker": [1]})
    pq.write_table(table, dataset_dir / "payments.parquet")
    pq.write_table(table, dataset_dir / "snapshots.parquet")
    pq.write_table(table, dataset_dir / "labels.parquet")
    (dataset_dir / "manifest.json").write_bytes(orjson.dumps({"split": split, "content_hash": "abc123"}))
    return dataset_dir


def _all_keys(value: Any) -> list[str]:
    """Collects every mapping key at every nesting level."""
    if isinstance(value, dict):
        keys = [str(key) for key in value]
        for nested in value.values():
            keys.extend(_all_keys(nested))
        return keys
    if isinstance(value, list):
        keys = []
        for item in value:
            keys.extend(_all_keys(item))
        return keys
    return []


def test_load_split_datasets_refuses_a_recorded_split_mismatch(tmp_path: Path) -> None:
    """A validation dataset passed as test data is refused with its recorded split named."""
    dataset_dir = _write_minimal_dataset(tmp_path, "validation-20261112T0000-aaaaaaaaaaaa", "validation")

    with pytest.raises(ValueError, match="records split 'validation'"):
        load_split_datasets("test", [dataset_dir])

    loaded = load_split_datasets("validation", [dataset_dir])
    assert [dataset.manifest["split"] for dataset in loaded] == ["validation"]


def test_read_world_base_rate_names_the_missing_file(tmp_path: Path) -> None:
    """A world whose base_rate.json is absent is refused, naming the missing file."""
    with pytest.raises(ValueError, match="base_rate.json"):
        read_world_base_rate(tmp_path / "world_cards", "run_missing")


def test_run_key_is_stable_for_the_same_dataset_ids_in_any_order() -> None:
    """The key depends only on the sorted dataset names, not on the flag order."""
    first = run_key(["test-1", "stress-1", "test-2"])
    second = run_key(["stress-1", "test-2", "test-1"])

    assert first == second
    assert len(first) == 12
    assert first != run_key(["test-1", "stress-1", "test-3"])


def _evaluated_two_depth_world() -> dict[str, Any]:
    """Runs evaluate_world on one day with mules at depths 1 and 2, returning its real metrics."""
    day = date(2025, 1, 1)
    scored = [
        ScoredDay(
            bank_id="bank_a",
            account_id="mule_1",
            day=day,
            rules_score=1,
            rules_fired=("RB3_MANY_PAYERS_ONE_DAY",),
            anomaly_score=None,
            role="mule",
            depth=1,
        ),
        ScoredDay(
            bank_id="bank_a",
            account_id="mule_2",
            day=day,
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role="mule",
            depth=2,
        ),
        ScoredDay(
            bank_id="bank_a",
            account_id="genuine_1",
            day=day,
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role="genuine",
            depth=None,
        ),
    ]
    return evaluate_world(
        scored=scored,
        scorer="rules",
        anomaly_threshold=None,
        capacity_pct=100.0,
        inflation_factor=1.0,
        persona_of=lambda account_id: "person",
        weights={"person": 1},
        default_weight=1,
        snapshot_time_by_account_day={
            (scored_day.account_id, day): datetime(2025, 1, 1, 12, 0, tzinfo=IST) for scored_day in scored
        },
        first_credit={
            "mule_1": datetime(2025, 1, 1, 10, 0, tzinfo=IST),
            "mule_2": datetime(2025, 1, 1, 10, 0, tzinfo=IST),
        },
        reported=set(),
    )


def test_write_result_writes_depth_keys_as_text(tmp_path: Path) -> None:
    """evaluate_world keys depth recall by integer depth; result.json must carry "1" and "2"."""
    result = _evaluated_two_depth_world()
    per_world = {"world_a": {"run_id": "run_a", "split": "test", "scorers": {"rules": result}}}

    path = write_result(tmp_path, "0123456789ab", {}, per_world, {})

    payload = orjson.loads(path.read_bytes())
    depth_recall = payload["per_world"]["world_a"]["scorers"]["rules"]["hidden_mule_recall_by_depth"]
    assert set(depth_recall) == {"1", "2"}
    assert depth_recall == {
        str(depth): value for depth, value in result["hidden_mule_recall_by_depth"].items()
    }


def test_write_result_refuses_a_non_text_key_and_names_its_path(tmp_path: Path) -> None:
    """A key that is not text anywhere else in the payload fails loudly with its path."""
    per_world = {
        "world_a": {
            "run_id": "run_a",
            "split": "test",
            "scorers": {"rules": {"analyst_load": {1: 0.5}}},
        }
    }

    with pytest.raises(ValueError, match=r"result\.per_world\.world_a\.scorers\.rules\.analyst_load"):
        write_result(tmp_path, "0123456789ab", {}, per_world, {})


def _world(split: str, recall: float) -> dict[str, Any]:
    """Builds a world block with every headline metric, all three scorers, and the given recall."""
    scorers = {
        scorer: {
            "recall_at_capacity": recall,
            "yield_rate": {"measured": recall, "reweighted": recall},
            "genuine_harm": {"genuine_restriction_rate": 0.0},
            "time_to_detection": {"median": 1.0},
        }
        for scorer in ("rules", "anomaly", "rules_and_anomaly")
    }
    return {"run_id": f"run_{split}", "split": split, "scorers": scorers}


def test_summarise_worlds_keeps_stress_out_of_the_test_median() -> None:
    """
    Two test worlds give a median recall of 0.5; adding a perfect stress world leaves that median
    at 0.5 (mixing them would give 0.75) and reports the stress window on its own.
    """
    test_worlds = {"test_a": _world("test", 0.25), "test_b": _world("test", 0.75)}
    stress_world = _world("stress", 1.0)

    test_only = summarise_worlds(test_worlds, min_worlds=2)
    with_stress = summarise_worlds({**test_worlds, "stress_a": stress_world}, min_worlds=2)

    assert with_stress["test"] == test_only["test"]
    assert with_stress["test"]["rules"]["recall_at_capacity"]["median"] == 0.5
    assert with_stress["test"]["rules"]["recall_at_capacity"]["world_count"] == 2
    assert with_stress["stress"]["rules"]["recall_at_capacity"]["median"] == 1.0
    assert with_stress["stress"]["rules"]["recall_at_capacity"]["world_count"] == 1
    assert test_only["stress"]["rules"]["recall_at_capacity"] == {
        "world_count": 0,
        "status": "not run: no stress worlds supplied",
        "unit": "fraction",
    }


def test_summarise_worlds_records_the_unit_of_every_metric() -> None:
    """Each headline metric says whether its number is a fraction or a time in hours."""
    summaries = summarise_worlds({"test_a": _world("test", 0.5)}, min_worlds=2)

    units = {metric: entry["unit"] for metric, entry in summaries["test"]["rules"].items()}

    assert units == {
        "recall_at_capacity": "fraction",
        "yield_measured": "fraction",
        "yield_reweighted": "fraction",
        "genuine_restriction_rate": "fraction",
        "median_time_to_detection_hours": "hours",
    }


def test_summarise_worlds_says_not_run_for_a_window_without_worlds() -> None:
    """A stress window never supplied is reported as not run, not as a metric that found nothing."""
    summaries = summarise_worlds({"test_a": _world("test", 0.5)}, min_worlds=2)

    assert summaries["stress"]["anomaly"]["recall_at_capacity"] == {
        "world_count": 0,
        "status": "not run: no stress worlds supplied",
        "unit": "fraction",
    }


def test_summarise_worlds_keeps_no_numeric_values_when_a_world_has_none() -> None:
    """A test world with a null detection time reports no numeric values, while its recall stands."""
    world = _world("test", 0.5)
    world["scorers"]["rules"]["time_to_detection"] = {"median": None}

    summaries = summarise_worlds({"test_a": world}, min_worlds=2)

    assert summaries["test"]["rules"]["median_time_to_detection_hours"] == {
        "world_count": 0,
        "status": "no numeric values",
        "unit": "hours",
    }
    assert summaries["test"]["rules"]["recall_at_capacity"]["median"] == 0.5


def test_write_result_marks_simulated_and_records_no_timestamp(tmp_path: Path) -> None:
    """result.json is written under the key folder, marked simulated, with no generation time."""
    manifest = {"code_version": "abc", "datasets": [{"dataset_id": "test-1"}]}
    per_world = {"test-1": {"run_id": "run_1", "split": "test", "scorers": {"rules": {"recall_at_capacity": 0.5}}}}
    across_worlds = {"rules": {"recall_at_capacity": {"world_count": 1, "median": 0.5, "spread": "not enough worlds"}}}

    path = write_result(tmp_path, "0123456789ab", manifest, per_world, across_worlds)

    assert path == tmp_path / "0123456789ab" / "result.json"
    payload = orjson.loads(path.read_bytes())
    assert payload["simulated"] is True
    assert payload["manifest"] == manifest
    assert payload["per_world"] == per_world
    assert payload["across_worlds"] == across_worlds
    for field in _all_keys(payload):
        lowered = field.lower()
        assert "timestamp" not in lowered
        assert not lowered.endswith("_at")
