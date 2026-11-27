"""
File: scripts/dataset_shards.py
Purpose: Splits one Level 16 split's planned worlds across worker processes, notices a STOP file,
         and loads the per-world checkpoints of a split in plan order for assembly.

A worker handles the worlds whose plan position satisfies position % N == K, so every world belongs
to exactly one worker. Assembly always reads checkpoints in plan order, never in completion order,
because the parquet row order follows the extract order.
"""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from replay.dataset_builder import WorldExtract
from scripts.dataset_checkpoints import (
    checkpoint_path,
    key_values_fingerprint,
    load_checkpoint,
    verify_checkpoint_identity,
)
from world.world_sampler import WorldSpec

# Existing convention: 0 means complete, 1 means a failure, 2 means a usage refusal or a compare
# error (scripts/compare_datasets_normalised.py, scripts/check.py, showcase/__main__.py). 3 is
# unused, so it means "stopped or incomplete: run the same command again to continue".
EXIT_STOPPED = 3

STOP_FILE_NAME = "STOP"


class MissingCheckpointsError(ValueError):
    """WHAT: Raised when an assemble-only build lacks a checkpoint.
    WHY: The dataset must not be written from partial work."""

    def __init__(self, split: str, world_ids: Sequence[str]) -> None:
        self.split = split
        self.world_ids = tuple(world_ids)
        super().__init__(
            f"cannot assemble split {split!r}: no checkpoint for {', '.join(self.world_ids)}"
        )


class WorldShard:
    """WHAT: One worker's share of a split's worlds (index K of count N)."""

    def __init__(self, index: int, count: int) -> None:
        if count < 1:
            raise ValueError(f"world shard count must be at least 1, got {count}")
        if index < 0 or index >= count:
            raise ValueError(f"world shard index {index} is outside 0..{count - 1}")
        self.index = index
        self.count = count

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WorldShard) and (self.index, self.count) == (other.index, other.count)

    def __hash__(self) -> int:
        return hash((self.index, self.count))

    def __repr__(self) -> str:
        return f"WorldShard({self.index}/{self.count})"


def parse_world_shard(text: str) -> WorldShard:
    """
    WHAT: Parses a "K/N" shard selection.
    Raises:
        ValueError: On missing or repeated "/", a non-integer part, or a value outside 0 <= K < N.
    """
    left, separator, right = text.strip().partition("/")
    if not separator or "/" in right:
        raise ValueError(f"world shard {text!r} is not K/N")
    try:
        index = int(left)
        count = int(right)
    except ValueError:
        raise ValueError(f"world shard {text!r} is not two integers") from None
    return WorldShard(index, count)


def shard_specs(specs: Sequence[WorldSpec], shard: WorldShard) -> list[WorldSpec]:
    """WHAT: Selects the worlds whose plan position satisfies position % count == index."""
    return [spec for position, spec in enumerate(specs) if position % shard.count == shard.index]


def stop_file_path(checkpoint_dir: Path, split: str) -> Path:
    """WHAT: Returns the STOP file of one split. WHY: One file per split, beside its checkpoints."""
    return checkpoint_dir / split / STOP_FILE_NAME


def stop_requested(checkpoint_dir: Path, split: str) -> bool:
    """WHAT: Reports whether a STOP file asks this split's build to start no further world."""
    return stop_file_path(checkpoint_dir, split).is_file()


def load_plan_checkpoints(
    checkpoint_dir: Path,
    split: str,
    specs: Sequence[WorldSpec],
    *,
    preset_name: str,
    sampler_seed: int,
    code_version: str,
) -> tuple[list[WorldExtract], list[datetime]]:
    """
    WHAT: Loads every planned world's checkpoint in plan order and checks each against its spec.
    WHY: build_dataset keeps the extract order in its parquet rows, so assembly must not depend on
    which worker finished first.
    Raises:
        MissingCheckpointsError: If any planned world has no checkpoint; nothing is assembled.
    """
    missing = [
        spec.world_id
        for spec in specs
        if not checkpoint_path(checkpoint_dir, split, spec.world_id).is_file()
    ]
    if missing:
        raise MissingCheckpointsError(split, missing)

    extracts: list[WorldExtract] = []
    world_ends = []
    for spec in specs:
        checkpoint = load_checkpoint(checkpoint_path(checkpoint_dir, split, spec.world_id))
        verify_checkpoint_identity(
            checkpoint,
            split=split,
            world_id=spec.world_id,
            seed=spec.seed,
            preset_name=preset_name,
            sampler_seed=sampler_seed,
            code_version=code_version,
            key_values_sha256=key_values_fingerprint(spec.key_values),
        )
        extracts.append(checkpoint.extract)
        world_ends.append(checkpoint.world_end)
    return extracts, world_ends
