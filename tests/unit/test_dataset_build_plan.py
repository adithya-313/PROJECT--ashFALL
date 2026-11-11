"""
File: tests/unit/test_dataset_build_plan.py
Purpose: Fast unit tests for Level 16 dataset build planning, split sampling,
         prior version superseding, and cutoff calculation without running slow replays.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pyarrow as pa
import pytest

from replay.dataset_builder import Dataset
from replay.runner import IST
from scripts.build_level_16_datasets import (
    derive_dataset_cutoff,
    load_datasets_config,
    plan_datasets,
    summarize_labels_by_source_and_via,
    supersede_prior_datasets,
)


def test_load_datasets_config_reads_expected_splits() -> None:
    """Verifies that load_datasets_config reads config/datasets.yaml with all 4 splits and expected world counts."""
    cfg = load_datasets_config()
    assert isinstance(cfg, dict)
    assert set(cfg.keys()) == {"training", "validation", "test", "stress"}
    assert cfg["training"] == 2
    assert cfg["validation"] == 1
    assert cfg["test"] == 1
    assert cfg["stress"] == 1


def test_plan_datasets_generates_expected_world_specs() -> None:
    """Verifies that plan_datasets plans the exact world counts and seeds for each split."""
    plan = plan_datasets(sampler_seed=1)
    assert len(plan["training"]) == 2
    assert len(plan["validation"]) == 1
    assert len(plan["test"]) == 1
    assert len(plan["stress"]) == 1

    # Verify seeds conform to configured seed blocks
    assert plan["training"][0].seed == 1000000
    assert plan["training"][1].seed == 1000001
    assert plan["validation"][0].seed == 2000000
    assert plan["test"][0].seed == 3000000
    assert plan["stress"][0].seed == 4000000

    # Verify world_ids
    assert plan["training"][0].world_id == "training-0000"
    assert plan["training"][1].world_id == "training-0001"
    assert plan["validation"][0].world_id == "validation-0000"
    assert plan["test"][0].world_id == "test-0000"
    assert plan["stress"][0].world_id == "stress-0000"


def test_supersede_prior_datasets_updates_matching_split_only(tmp_path: Path) -> None:
    """Verifies that supersede_prior_datasets marks only older datasets of the same split."""
    # Create an old training dataset folder
    old_train = tmp_path / "training-20260901T0000-old111111111"
    old_train.mkdir()
    (old_train / "manifest.json").write_bytes(
        orjson.dumps({"split": "training", "content_hash": "oldhash1"})
    )

    # Create an old validation dataset folder (different split, must NOT be marked)
    old_val = tmp_path / "validation-20260901T0000-old222222222"
    old_val.mkdir()
    (old_val / "manifest.json").write_bytes(
        orjson.dumps({"split": "validation", "content_hash": "oldhash2"})
    )

    # Create the new training dataset folder
    new_train = tmp_path / "training-20261101T0000-new333333333"
    new_train.mkdir()
    (new_train / "manifest.json").write_bytes(
        orjson.dumps({"split": "training", "content_hash": "newhash3"})
    )

    superseded = supersede_prior_datasets("training", new_train, datasets_root=tmp_path)
    assert superseded == [old_train]

    # Verify old training manifest has superseded_by pointing to new_train.name
    updated_train = orjson.loads((old_train / "manifest.json").read_bytes())
    assert updated_train.get("superseded_by") == new_train.name

    # Verify validation manifest is completely untouched
    untouched_val = orjson.loads((old_val / "manifest.json").read_bytes())
    assert "superseded_by" not in untouched_val

    # Second run should find 0 newly superseded datasets
    superseded_again = supersede_prior_datasets("training", new_train, datasets_root=tmp_path)
    assert superseded_again == []


def test_supersede_prior_datasets_handles_empty_root(tmp_path: Path) -> None:
    """Verifies that supersede_prior_datasets returns empty list on non-existent or empty directory."""
    non_existent = tmp_path / "does_not_exist"
    assert supersede_prior_datasets("training", tmp_path / "new", datasets_root=non_existent) == []

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    new_dir = empty_dir / "new_dataset"
    new_dir.mkdir()
    (new_dir / "manifest.json").write_bytes(orjson.dumps({"split": "training"}))
    assert supersede_prior_datasets("training", new_dir, datasets_root=empty_dir) == []


def test_derive_dataset_cutoff_uses_max_world_end() -> None:
    """Verifies that derive_dataset_cutoff derives the maximum simulation end timestamp."""
    # World 1: starts 2026-09-01, 61 days -> ends 2026-11-01 00:00:00 IST
    w1 = MagicMock()
    w1.output.plan.clock.start_date = datetime(2026, 9, 1, 9, 0, 0, tzinfo=IST)
    w1.output.plan.n_days = 61

    # World 2: starts 2026-09-01, 120 days -> ends 2026-12-30 00:00:00 IST
    w2 = MagicMock()
    w2.output.plan.clock.start_date = datetime(2026, 9, 1, 9, 0, 0, tzinfo=IST)
    w2.output.plan.n_days = 120

    cutoff = derive_dataset_cutoff([w1, w2])
    expected = datetime(2026, 9, 1, 0, 0, 0, tzinfo=IST) + timedelta(days=120)
    assert cutoff == expected

    # Empty list must raise ValueError
    with pytest.raises(ValueError, match="empty recorded worlds"):
        derive_dataset_cutoff([])


def test_summarize_labels_by_source_and_via() -> None:
    """Verifies label aggregation breakdown across complaint direct, hold request, registry, police."""
    labels_table = pa.Table.from_arrays(
        [
            pa.array(["complaint", "complaint", "registry", "police_bulk"]),
            pa.array([None, "hold_request", None, None]),
        ],
        names=["source", "via"],
    )
    mock_dataset = Dataset(
        payments=pa.table({}),
        snapshots=pa.table({}),
        labels=labels_table,
        manifest={},
    )
    summary = summarize_labels_by_source_and_via(mock_dataset)
    assert summary == {
        "complaint": 1,
        "complaint (via hold_request)": 1,
        "registry": 1,
        "police_bulk": 1,
    }
