"""
File: tests/unit/test_dataset_checkpoints.py
Purpose: Unit tests for the Level 16 resumable-build checkpoints: an exact save/load round trip,
         refusal of a checkpoint from different settings, ignoring of leftover temporary files,
         and the cutoff a resumed build derives from saved world ends.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from replay.dataset_builder import WorldExtract
from replay.runner import IST
from scripts.build_level_16_datasets import derive_dataset_cutoff, world_end_for_recorded
from scripts.dataset_checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    DatasetCheckpoint,
    checkpoint_path,
    checkpoint_temp_path,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint_identity,
)


def _extract() -> WorldExtract:
    """Returns an extract holding a datetime, a None and a nested dict, so the pickle is checked on all three."""
    return WorldExtract(
        world_id="test-0000",
        split="test",
        seed=101,
        run_id="7ee68d1b5a539657",
        gang_families=("family_a",),
        decisions_per_event_payment=True,
        registry_wired=True,
        registry_wired_reason="wired",
        payment_rows=[
            {
                "payment_id": "p-1",
                "decision_time": datetime(2026, 9, 2, 10, 30, tzinfo=IST),
                "receiver_account_id": None,
                "feature_values": {"velocity": {"1h": 3, "24h": 11}},
            }
        ],
        snapshot_rows=[
            {
                "account_id": "a-1",
                "snapshot_time": datetime(2026, 9, 3, 0, 0, tzinfo=IST),
                "feature_values": {"balance": None},
            }
        ],
        label_rows=[
            {
                "label_id": "l-1",
                "arrival_time": datetime(2026, 9, 4, 12, 0, tzinfo=IST),
                "payment_ids": ["p-1"],
            }
        ],
        label_status_rows=[
            {"label_id": "l-1", "change": "revoked", "arrival_time": datetime(2026, 9, 5, 8, 0, tzinfo=IST)}
        ],
    )


def _checkpoint(extract: WorldExtract, *, seed: int = 101) -> DatasetCheckpoint:
    """Returns a checkpoint wrapping extract, with the identity fields the round-trip tests compare."""
    return DatasetCheckpoint(
        format_version=CHECKPOINT_FORMAT_VERSION,
        split="test",
        world_id="test-0000",
        seed=seed,
        preset_name="tiny",
        sampler_seed=2,
        code_version="abc123",
        run_id=extract.run_id,
        world_end=datetime(2026, 10, 1, 0, 0, tzinfo=IST),
        record_seconds=12.5,
        replay_seconds=34.25,
        extract=extract,
    )


def test_save_and_load_round_trip_exactly(tmp_path: Path) -> None:
    """Verifies a saved checkpoint loads back with the same identity, world end and extract."""
    extract = _extract()
    checkpoint = _checkpoint(extract)
    path = checkpoint_path(tmp_path, "test", "test-0000")

    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)

    assert loaded == checkpoint
    assert loaded.extract == extract
    assert loaded.extract.payment_rows[0]["receiver_account_id"] is None
    assert loaded.extract.payment_rows[0]["decision_time"] == datetime(2026, 9, 2, 10, 30, tzinfo=IST)
    assert loaded.extract.payment_rows[0]["feature_values"] == {"velocity": {"1h": 3, "24h": 11}}


def test_identity_mismatch_names_the_field_and_both_values() -> None:
    """Verifies a checkpoint recorded with a different seed is refused, naming seed and both values."""
    checkpoint = _checkpoint(_extract(), seed=101)

    with pytest.raises(ValueError) as error:
        verify_checkpoint_identity(
            checkpoint,
            split="test",
            world_id="test-0000",
            seed=202,
            preset_name="tiny",
            sampler_seed=2,
            code_version="abc123",
        )

    message = str(error.value)
    assert "seed: checkpoint=101 current=202" in message
    assert "preset_name" not in message


def test_leftover_temp_file_is_ignored_and_never_loaded(tmp_path: Path) -> None:
    """Verifies a .pkl.tmp left by a crash is not the checkpoint: it is never read and a save replaces it."""
    path = checkpoint_path(tmp_path, "test", "test-0000")
    temp_path = checkpoint_temp_path(path)
    temp_path.parent.mkdir(parents=True)
    temp_path.write_bytes(b"half-written checkpoint")

    assert path.is_file() is False
    with pytest.raises(FileNotFoundError):
        load_checkpoint(path)

    save_checkpoint(path, _checkpoint(_extract()))
    assert path.is_file() is True
    assert temp_path.exists() is False
    assert load_checkpoint(path).extract == _extract()


def test_resumed_cutoff_equals_max_of_per_world_ends() -> None:
    """Verifies derive_dataset_cutoff stays the maximum world end, which is what a resumed build recomputes."""
    world_one = MagicMock()
    world_one.output.plan.clock.start_date = datetime(2026, 9, 1, 9, 0, 0, tzinfo=IST)
    world_one.output.plan.n_days = 61

    world_two = MagicMock()
    world_two.output.plan.clock.start_date = datetime(2026, 9, 1, 9, 0, 0, tzinfo=IST)
    world_two.output.plan.n_days = 120

    per_world_ends = [world_end_for_recorded(world_one), world_end_for_recorded(world_two)]
    assert derive_dataset_cutoff([world_one, world_two]) == max(per_world_ends)
    assert derive_dataset_cutoff([world_one, world_two]) == datetime(2026, 9, 1, 0, 0, 0, tzinfo=IST) + timedelta(
        days=120
    )
