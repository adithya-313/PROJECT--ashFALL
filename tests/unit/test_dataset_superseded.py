"""
Tests for superseded datasets: mark_superseded, load_dataset and the truth join refuse an old dataset unless asked.

WHY: A rebuilt dataset replaces an older one; the old files stay for history but nobody should use them by accident.

TERM: superseded: a dataset whose manifest.json names its replacement in superseded_by.
"""

from datetime import datetime, timezone
from pathlib import Path

import orjson
import pyarrow as pa
import pytest

from evaluation.dataset_truth_join import join_truth
from replay.dataset_builder import Dataset, SupersededDatasetError, load_dataset, mark_superseded, write_dataset


def make(root: Path, tag: str) -> Path:
    """Writes a tiny dataset whose hash differs by tag, and returns its folder."""
    manifest = {
        "split": "training", "cutoff": datetime(2026, 9, 10, tzinfo=timezone.utc).isoformat(),
        "content_hash": tag * 64, "superseded_by": None, "worlds": [],
    }
    table = pa.table({"x": [1, 2]})
    return write_dataset(Dataset(payments=table, snapshots=table, labels=table, manifest=manifest), root)


def test_fresh_dataset_loads_without_asking(tmp_path: Path) -> None:
    """Checks a dataset nobody replaced loads normally."""
    assert load_dataset(make(tmp_path, "a")).manifest["superseded_by"] is None


def test_marking_sets_superseded_by_and_keeps_every_old_file(tmp_path: Path) -> None:
    """Checks the old manifest names the new folder and all old files are still there."""
    old, new = make(tmp_path, "a"), make(tmp_path, "b")
    before = sorted(p.name for p in old.iterdir())
    mark_superseded(old, new)
    assert orjson.loads((old / "manifest.json").read_bytes())["superseded_by"] == new.name
    assert sorted(p.name for p in old.iterdir()) == before


def test_loading_a_superseded_dataset_is_refused_unless_asked(tmp_path: Path) -> None:
    """Checks load_dataset raises for a superseded dataset and loads it with allow_superseded=True."""
    old, new = make(tmp_path, "a"), make(tmp_path, "b")
    mark_superseded(old, new)
    with pytest.raises(SupersededDatasetError, match=new.name):
        load_dataset(old)
    assert load_dataset(old, allow_superseded=True).labels.num_rows == 2


def test_truth_join_refuses_a_superseded_dataset_unless_asked(tmp_path: Path) -> None:
    """Checks join_truth refuses a superseded dataset before touching any truth."""
    old, new = make(tmp_path, "a"), make(tmp_path, "b")
    mark_superseded(old, new)
    with pytest.raises(ValueError, match="superseded"):
        join_truth(old, tmp_path)


def test_a_dataset_cannot_supersede_itself_or_a_missing_folder(tmp_path: Path) -> None:
    """Checks the two bad calls to mark_superseded are refused."""
    old = make(tmp_path, "a")
    with pytest.raises(ValueError):
        mark_superseded(old, old)
    with pytest.raises(ValueError):
        mark_superseded(old, tmp_path / "nope")
