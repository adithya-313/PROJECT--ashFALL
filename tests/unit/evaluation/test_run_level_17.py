"""
WHAT: Unit tests for evaluation/run_level_17.py.

WHY: The runner must refuse a dataset whose recorded split differs from the flag it was passed
under, refuse a missing world card, key a result by its datasets alone, and mark every result as
simulated without recording any generation timestamp.
"""

from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
    assert test_only["stress"]["rules"]["recall_at_capacity"] == {"world_count": 0, "status": "no numeric values"}


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
