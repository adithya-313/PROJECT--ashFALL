"""
File: scripts/write_fixture_manifest.py
Purpose: Writes the Level 17 FIXTURE_MANIFEST.json from the dataset manifests on disk.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from showcase.level_17_checks import find_missing_preconditions

STABILITY_LIMIT = 10


def load_splits(datasets_root: Path) -> dict[str, dict[str, Any]]:
    """
    WHAT: Reads every dataset manifest under datasets_root and groups them by split name.

    WHY: The fixture manifest must name the datasets that actually exist, with the run ids stored
    beside them, rather than values typed in from memory.

    Args:
        datasets_root: Folder holding one directory per dataset.

    Returns:
        Split name -> {"dataset_id": dataset folder name, "worlds": [{"run_id", "world_id"}]}.

    Raises:
        ValueError: If the folder or a manifest is missing, a field has the wrong type, or two
            datasets claim the same split.
    """
    if not datasets_root.is_dir():
        raise ValueError(f"Dataset folder not found: {datasets_root}")

    splits: dict[str, dict[str, Any]] = {}
    for dataset_dir in sorted(dataset_dir for dataset_dir in datasets_root.iterdir() if dataset_dir.is_dir()):
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Dataset manifest not found: {manifest_path}")
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if "split" not in dataset_manifest:
            raise ValueError(f"Dataset manifest has no split: {manifest_path}")
        split = dataset_manifest["split"]
        if not isinstance(split, str):
            raise ValueError(f"Dataset manifest split is not a string: {manifest_path}")
        if split in splits:
            raise ValueError(f"Two datasets claim split {split}: {splits[split]['dataset_id']} and {dataset_dir.name}")

        if "worlds" not in dataset_manifest:
            raise ValueError(f"Dataset manifest has no worlds: {manifest_path}")
        raw_worlds = dataset_manifest["worlds"]
        if not isinstance(raw_worlds, list) or not raw_worlds:
            raise ValueError(f"Dataset manifest worlds is not a non-empty list: {manifest_path}")

        worlds: list[dict[str, str]] = []
        for world in raw_worlds:
            run_id = world["run_id"]
            world_id = world["world_id"]
            if not isinstance(run_id, str) or not isinstance(world_id, str):
                raise ValueError(f"Dataset manifest world ids are not strings: {manifest_path}")
            worlds.append({"run_id": run_id, "world_id": world_id})

        splits[split] = {"dataset_id": dataset_dir.name, "worlds": worlds}

    if not splits:
        raise ValueError(f"No dataset manifests found under: {datasets_root}")
    return splits


def measure_fixtures(fixtures_root: Path) -> tuple[int, int]:
    """
    WHAT: Counts every file under fixtures_root and sums its size, the manifest file included.

    WHY: The showcase's precondition compares the manifest with the whole tree, so the writer must
    measure the same files the check measures.

    Args:
        fixtures_root: Level 17 fixture folder.

    Returns:
        The file count and the total byte size.
    """
    files = [path for path in fixtures_root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def build_manifest(
    splits: dict[str, dict[str, Any]],
    code_version: str,
    preset: str,
    purpose: str,
    sampler_seed: int,
    file_count: int,
    total_bytes: int,
) -> dict[str, Any]:
    """Returns the fixture manifest payload with the given counts."""
    return {
        "code_version": code_version,
        "file_count": file_count,
        "preset": preset,
        "purpose": purpose,
        "sampler_seed": sampler_seed,
        "splits": splits,
        "total_bytes": total_bytes,
    }


def write_manifest_file(manifest_path: Path, payload: dict[str, Any]) -> None:
    """Writes the payload as sorted two-space JSON with a trailing newline, the committed format."""
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_fixture_manifest(
    fixtures_root: Path,
    code_version: str,
    preset: str,
    purpose: str,
    sampler_seed: int,
) -> Path:
    """
    WHAT: Writes fixtures_root/FIXTURE_MANIFEST.json until its own file count and byte size match
    the tree on disk, then re-runs the Level 17 precondition check on the result.

    WHY: The manifest counts itself, so its byte size depends on the numbers it states; the file is
    written and measured again until those numbers stop changing.

    Args:
        fixtures_root: Level 17 fixture folder holding datasets/, truth/ and world_cards/.
        code_version: Commit hash the fixtures were recorded at.
        preset: World preset the fixtures were recorded with.
        purpose: One-sentence statement of what the fixtures are for.
        sampler_seed: Sampler seed the worlds were recorded with.

    Returns:
        The path of the written manifest.

    Raises:
        RuntimeError: If the counts do not stabilise within STABILITY_LIMIT attempts, or if the
            Level 17 precondition check still reports a problem after writing.
        ValueError: If the dataset manifests under datasets_root are missing or inconsistent.
    """
    if not fixtures_root.is_dir():
        raise ValueError(f"Fixtures folder not found: {fixtures_root}")

    splits = load_splits(fixtures_root / "datasets")
    manifest_path = fixtures_root / "FIXTURE_MANIFEST.json"
    file_count = 0
    total_bytes = 0

    for _attempt in range(STABILITY_LIMIT):
        payload = build_manifest(splits, code_version, preset, purpose, sampler_seed, file_count, total_bytes)
        write_manifest_file(manifest_path, payload)
        measured_count, measured_bytes = measure_fixtures(fixtures_root)
        if (measured_count, measured_bytes) == (file_count, total_bytes):
            problems = find_missing_preconditions(fixtures_root)
            if problems:
                raise RuntimeError(
                    "The written FIXTURE_MANIFEST.json does not match the fixtures tree: " + "; ".join(problems)
                )
            return manifest_path
        file_count = measured_count
        total_bytes = measured_bytes

    raise RuntimeError(f"FIXTURE_MANIFEST.json did not stabilise after {STABILITY_LIMIT} attempts: {manifest_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parses the fixture settings, writes the manifest and returns the process exit code."""
    parser = argparse.ArgumentParser(description="Write the Level 17 FIXTURE_MANIFEST.json from dataset manifests")
    parser.add_argument("--fixtures-root", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--sampler-seed", type=int, required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args(argv)

    try:
        manifest_path = write_fixture_manifest(
            args.fixtures_root,
            code_version=args.code_version,
            preset=args.preset,
            purpose=args.purpose,
            sampler_seed=args.sampler_seed,
        )
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
