"""
File: scripts/dataset_checkpoints.py
Purpose: Saves and loads one resumable build checkpoint per recorded Level 16 world, and verifies
         that a loaded checkpoint belongs to the world and settings of the current build.

The checkpoints are pickles. Pickle is allowed here only because these files are written and read
by scripts/build_level_16_datasets.py on the owner's machine. Never load a checkpoint that came
from anywhere else: unpickling can execute code.
"""

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from replay.dataset_builder import WorldExtract

CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_SUFFIX = ".pkl"
CHECKPOINT_TEMP_SUFFIX = ".pkl.tmp"
IDENTITY_FIELDS = ("split", "world_id", "seed", "preset_name", "sampler_seed", "code_version", "key_values_sha256")


def key_values_fingerprint(key_values: Iterable[tuple[str, float | int]]) -> str:
    """
    WHAT: Returns the SHA-256 of a world's key values written as canonical JSON (sorted keys, no spaces).
    WHY: Key values are the sampled settings that make a world; the same world id and seed with different
    values describe a different world, so a checkpoint must fingerprint them to be refused on a mismatch.
    """
    canonical = json.dumps([[name, value] for name, value in key_values], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetCheckpoint:
    """
    WHAT: One finished world of a build: its identity, the timings, the world end and its extract.
    WHY: Lets a rerun skip recording, replaying and extracting a world that already finished.
    """

    format_version: int
    split: str
    world_id: str
    seed: int
    preset_name: str
    sampler_seed: int
    code_version: str
    key_values_sha256: str
    run_id: str
    world_end: datetime
    record_seconds: float
    replay_seconds: float
    extract: WorldExtract


def checkpoint_path(checkpoint_dir: Path, split: str, world_id: str) -> Path:
    """WHAT: Returns the checkpoint file for one world. WHY: Keeps each world a separate atomic file."""
    return checkpoint_dir / split / f"{world_id}{CHECKPOINT_SUFFIX}"


def checkpoint_temp_path(path: Path) -> Path:
    """WHAT: Returns the temporary sibling a checkpoint is written to first."""
    return path.with_name(path.name + ".tmp")


def save_checkpoint(path: Path, checkpoint: DatasetCheckpoint) -> None:
    """
    WHAT: Writes a checkpoint to a temporary file, forces it to disk, then renames it into place.
    WHY: A crash mid-write leaves only the .tmp file, which no loader ever reads, never a half file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_temp_path(path)
    with temp_path.open("wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def load_checkpoint(path: Path) -> DatasetCheckpoint:
    """
    WHAT: Reads one checkpoint file and checks its type, format version and internal run id.
    WHY: A checkpoint from another format or with an extract from a different recording must be
    refused instead of silently feeding the dataset. The run id cannot be recomputed without
    recording the world again, so it is checked inside the file: once the extract is known, its
    run id must equal the recorded identity run id.
    """
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {path} does not exist")
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, DatasetCheckpoint):
        raise ValueError(f"checkpoint {path} holds {type(checkpoint).__name__}, not DatasetCheckpoint")
    if checkpoint.format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint {path} format version is {checkpoint.format_version}, "
            f"this build reads version {CHECKPOINT_FORMAT_VERSION}"
        )
    if checkpoint.extract.run_id != checkpoint.run_id:
        raise ValueError(
            f"checkpoint {path} identity run_id is {checkpoint.run_id!r} "
            f"but its extract run_id is {checkpoint.extract.run_id!r}"
        )
    return checkpoint


def verify_checkpoint_identity(
    checkpoint: DatasetCheckpoint,
    *,
    split: str,
    world_id: str,
    seed: int,
    preset_name: str,
    sampler_seed: int,
    code_version: str,
    key_values_sha256: str,
) -> None:
    """
    WHAT: Compares a checkpoint's identity fields with the current build's spec and settings.
    WHY: A changed seed, preset or sampler seed describes a different world, so its checkpoint
    must be refused instead of reused; a changed key-value fingerprint means the same world id and
    seed now sample different settings, which is equally a different world.
    """
    expected: dict[str, object] = {
        "split": split,
        "world_id": world_id,
        "seed": seed,
        "preset_name": preset_name,
        "sampler_seed": sampler_seed,
        "code_version": code_version,
        "key_values_sha256": key_values_sha256,
    }
    mismatches = [
        f"{field}: checkpoint={getattr(checkpoint, field)!r} current={expected[field]!r}"
        for field in IDENTITY_FIELDS
        if getattr(checkpoint, field) != expected[field]
    ]
    if mismatches:
        raise ValueError(f"checkpoint identity mismatch: {'; '.join(mismatches)}")
