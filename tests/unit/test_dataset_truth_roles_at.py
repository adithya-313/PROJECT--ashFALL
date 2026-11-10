"""
Tests for evaluation.dataset_truth_join.roles_at: role and owner of single accounts at single moments.

WHY: Showcase checks ask "what was this account really at that moment" for accounts with no payment row; the answer
must follow the same half-open span and clock_start minute rules as the join.

TERM: half-open span: active from valid_from (included) up to valid_to (excluded).
"""

import json
from pathlib import Path

import pytest

from evaluation.dataset_truth_join import roles_at
from tests.unit.test_dataset_truth_join import MULE, SPAN_FROM, SPAN_TO, WORLD_ID, _at, _write_dataset, _write_truth


def test_role_and_owner_follow_the_half_open_span(tmp_path: Path) -> None:
    """Checks genuine before the span, mule inside it with its gang as owner, genuine again at valid_to."""
    _write_truth(tmp_path / "truth")
    ds = _write_dataset(tmp_path)
    queries = [(WORLD_ID, MULE, _at(m)) for m in (SPAN_FROM - 1, SPAN_FROM, SPAN_TO - 1, SPAN_TO)]
    got = roles_at(ds, tmp_path / "truth", queries)
    assert got == [("genuine", None), ("mule", "gang_1"), ("mule", "gang_1"), ("genuine", None)]


def test_unknown_world_is_refused(tmp_path: Path) -> None:
    """Checks a world id missing from the manifest is refused."""
    _write_truth(tmp_path / "truth")
    ds = _write_dataset(tmp_path)
    with pytest.raises(ValueError, match="not listed"):
        roles_at(ds, tmp_path / "truth", [("other_world", MULE, _at(SPAN_FROM))])


def test_superseded_dataset_is_refused_unless_asked(tmp_path: Path) -> None:
    """Checks roles_at refuses a superseded dataset and answers when allow_superseded is True."""
    _write_truth(tmp_path / "truth")
    ds = _write_dataset(tmp_path)
    manifest = json.loads((ds / "manifest.json").read_text())
    manifest["superseded_by"] = "newer"
    (ds / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    query = [(WORLD_ID, MULE, _at(SPAN_FROM))]
    with pytest.raises(ValueError, match="superseded"):
        roles_at(ds, tmp_path / "truth", query)
    assert roles_at(ds, tmp_path / "truth", query, allow_superseded=True) == [("mule", "gang_1")]
