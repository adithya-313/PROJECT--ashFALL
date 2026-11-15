# File: tests/unit/test_write_fixture_manifest.py
# Purpose: Unit tests for scripts/write_fixture_manifest.py (Level 17 fixture manifest writer).
"""
WHAT: Runs the manifest writer on a tiny fake fixture tree and checks the result against the same
Level 17 precondition function the showcase uses, including a run whose byte count crosses from
three digits to four.

WHY: The manifest counts its own bytes, so a writer that measures before writing claims a size the
showcase then rejects; these tests prove the written file matches the tree on disk.
"""

import json
from pathlib import Path
from typing import Any

from scripts.write_fixture_manifest import write_fixture_manifest
from showcase.level_17_checks import find_missing_preconditions

_MANIFEST_KEYS = {
    "code_version",
    "file_count",
    "preset",
    "purpose",
    "sampler_seed",
    "splits",
    "total_bytes",
}

_DATASETS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "test-20261115T0000-cccccccccccc": ("test", [("runTest0", "test-0000")]),
    "training-20261115T0000-aaaaaaaaaaaa": (
        "training",
        [("runTrain0", "training-0000"), ("runTrain1", "training-0001")],
    ),
    "validation-20261115T0000-bbbbbbbbbbbb": ("validation", [("runValid0", "validation-0000")]),
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes one small JSON file, creating its parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_fixture_tree(fixtures_root: Path) -> None:
    """
    WHAT: Creates datasets/, truth/ and world_cards/ in the committed fixture layout.

    WHY: The writer reads only the dataset manifests, but the precondition counts every file under
    the fixture root, so the tree must look like the real one.
    """
    for dataset_id, (split, worlds) in _DATASETS.items():
        dataset_dir = fixtures_root / "datasets" / dataset_id
        write_json(
            dataset_dir / "manifest.json",
            {
                "split": split,
                "worlds": [{"run_id": run_id, "world_id": world_id} for run_id, world_id in worlds],
            },
        )
        for run_id, _world_id in worlds:
            write_json(fixtures_root / "truth" / run_id / "manifest.json", {"run_id": run_id})
            write_json(fixtures_root / "world_cards" / run_id / "base_rate.json", {"run_id": run_id})


def files_under(root: Path) -> list[Path]:
    """Returns every file under root, the same set the Level 17 precondition counts."""
    return [path for path in root.rglob("*") if path.is_file()]


def test_writer_manifest_matches_fixture_tree(tmp_path: Path) -> None:
    """Verifies the written manifest names the on-disk datasets and passes the Level 17 precondition."""
    fixtures_root = tmp_path / "level_17"
    build_fixture_tree(fixtures_root)

    manifest_path = write_fixture_manifest(fixtures_root, "abc1234", "tiny", "Unit test fixtures", 2)

    assert find_missing_preconditions(fixtures_root) == []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == _MANIFEST_KEYS
    files = files_under(fixtures_root)
    assert manifest["file_count"] == len(files)
    assert manifest["total_bytes"] == sum(path.stat().st_size for path in files)
    assert manifest["code_version"] == "abc1234"
    assert manifest["preset"] == "tiny"
    assert manifest["purpose"] == "Unit test fixtures"
    assert manifest["sampler_seed"] == 2
    assert manifest["splits"]["training"]["dataset_id"] == "training-20261115T0000-aaaaaaaaaaaa"
    assert manifest["splits"]["training"]["worlds"] == [
        {"run_id": "runTrain0", "world_id": "training-0000"},
        {"run_id": "runTrain1", "world_id": "training-0001"},
    ]


def test_writer_matches_across_a_digit_boundary(tmp_path: Path) -> None:
    """
    WHAT: Pads the tree so its total size crosses from three digits to four while the manifest is
    written, then checks the manifest still counts itself exactly.

    WHY: The manifest's own size grows when total_bytes gains a digit; a writer that measures once
    would leave the stated total one byte short.
    """
    fixtures_root = tmp_path / "level_17"
    build_fixture_tree(fixtures_root)

    bytes_before_manifest = sum(path.stat().st_size for path in files_under(fixtures_root))
    assert bytes_before_manifest < 1000
    padding_size = 1000 - bytes_before_manifest + 50
    (fixtures_root / "datasets" / "padding.bin").write_bytes(b"x" * padding_size)

    manifest_path = write_fixture_manifest(fixtures_root, "abc1234", "tiny", "Unit test fixtures", 2)

    assert find_missing_preconditions(fixtures_root) == []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = files_under(fixtures_root)
    assert manifest["total_bytes"] >= 1000
    assert len(str(manifest["total_bytes"])) == 4
    assert manifest["file_count"] == len(files)
    assert manifest["total_bytes"] == sum(path.stat().st_size for path in files)
