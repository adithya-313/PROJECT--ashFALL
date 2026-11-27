"""
File: tests/unit/test_dataset_shards.py
Purpose: Unit tests for splitting a split's worlds across workers, refusing bad shard text, the
         STOP file, and assembling a dataset from checkpoints in plan order.

WHAT: Each test calls the real production helpers in scripts/dataset_shards.py or the real
build_split_dataset with only the record/replay/write steps replaced.

WHY: Worker completion order must never change the dataset, a missing checkpoint must never write a
partial dataset, and a STOP file must stop before a world starts.
"""

import asyncio
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import scripts.build_level_16_datasets as build_module
from replay.dataset_builder import WorldExtract
from replay.runner import IST
from scripts.build_level_16_datasets import BuildStopped, build_split_dataset
from scripts.dataset_checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    DatasetCheckpoint,
    checkpoint_path,
    key_values_fingerprint,
    save_checkpoint,
)
from scripts.dataset_shards import (
    MissingCheckpointsError,
    WorldShard,
    load_plan_checkpoints,
    parse_world_shard,
    shard_specs,
)
from world.world_sampler import WorldSpec

TEST_CODE_VERSION = "abc123"


def _specs(count: int) -> list[WorldSpec]:
    """Returns count training worlds with distinct ids and seeds, in plan order."""
    return [
        WorldSpec(
            world_id=f"training-{position:04d}",
            split="training",
            seed=1000000 + position,
            range_name="training",
            key_values=(),
        )
        for position in range(count)
    ]


def _extract(spec: WorldSpec, run_id: str) -> WorldExtract:
    """Returns a minimal extract whose world id and run id match its spec."""
    return WorldExtract(
        world_id=spec.world_id,
        split=spec.split,
        seed=spec.seed,
        run_id=run_id,
        gang_families=(),
        decisions_per_event_payment=True,
        registry_wired=False,
        registry_wired_reason="",
        payment_rows=[],
        snapshot_rows=[],
        label_rows=[],
    )


def _save_checkpoint(checkpoint_dir: Path, spec: WorldSpec, *, run_id: str | None = None) -> None:
    """Saves a valid checkpoint for spec, with the identity fields a resumed build checks."""
    identity_run_id = run_id if run_id is not None else f"run-{spec.world_id}"
    save_checkpoint(
        checkpoint_path(checkpoint_dir, spec.split, spec.world_id),
        DatasetCheckpoint(
            format_version=CHECKPOINT_FORMAT_VERSION,
            split=spec.split,
            world_id=spec.world_id,
            seed=spec.seed,
            preset_name="tiny",
            sampler_seed=1,
            code_version=TEST_CODE_VERSION,
            key_values_sha256=key_values_fingerprint(spec.key_values),
            run_id=identity_run_id,
            world_end=datetime(2026, 9, 1, tzinfo=IST),
            record_seconds=1.5,
            replay_seconds=2.5,
            extract=_extract(spec, identity_run_id),
        ),
    )


def _forbid_world_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replaces the recording step with one that fails the test if any world is started."""

    def forbidden_record(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no world may be recorded after a STOP file")

    monkeypatch.setattr(build_module, "record_world_for_spec", forbidden_record)


@pytest.mark.parametrize("shard_count", [1, 2, 3])
def test_shard_selection_covers_every_world_exactly_once(shard_count: int) -> None:
    """For N=1,2,3 over 7 worlds, the N shards together hold every world once and no world twice."""
    specs = _specs(7)

    selected = [
        spec.world_id
        for index in range(shard_count)
        for spec in shard_specs(specs, WorldShard(index, shard_count))
    ]

    assert sorted(selected) == sorted(spec.world_id for spec in specs)
    assert len(selected) == len(set(selected))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1/0", "at least 1"),
        ("2/2", "outside 0..1"),
        ("-1/2", "outside 0..1"),
        ("a/2", "not two integers"),
        ("1", "is not K/N"),
        ("1/2/3", "is not K/N"),
        ("", "is not K/N"),
    ],
)
def test_parse_world_shard_refuses_bad_text(text: str, message: str) -> None:
    """Every malformed shard string is refused with a message naming the problem."""
    with pytest.raises(ValueError, match=message):
        parse_world_shard(text)


def test_assemble_only_lists_the_missing_world_and_writes_nothing(tmp_path: Path) -> None:
    """With one of three checkpoints missing, assembly names that world and no dataset folder appears."""
    specs = _specs(3)
    checkpoint_dir = tmp_path / "checkpoints"
    datasets_root = tmp_path / "datasets"
    _save_checkpoint(checkpoint_dir, specs[0])
    _save_checkpoint(checkpoint_dir, specs[2])

    with pytest.raises(MissingCheckpointsError) as error:
        asyncio.run(
            build_split_dataset(
                split="training",
                specs=specs,
                preset_name="tiny",
                sampler_cfg={},
                code_version=TEST_CODE_VERSION,
                datasets_root=datasets_root,
                admin_pg="postgresql://unused",
                base_redis_url="redis://unused",
                checkpoint_dir=checkpoint_dir,
                assemble_only=True,
            )
        )

    message = str(error.value)
    assert "training-0001" in message
    assert "training-0000" not in message
    assert "training-0002" not in message
    assert datasets_root.exists() is False


def test_stop_file_before_a_world_means_that_world_is_not_started(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the first world still to build and a STOP file present, no world runs and no checkpoint is written."""
    specs = _specs(2)
    checkpoint_dir = tmp_path / "checkpoints"
    stop_file = checkpoint_dir / "training" / "STOP"
    stop_file.parent.mkdir(parents=True)
    stop_file.write_text("stop\n")
    _forbid_world_runs(monkeypatch)

    with pytest.raises(BuildStopped):
        asyncio.run(
            build_split_dataset(
                split="training",
                specs=specs,
                preset_name="tiny",
                sampler_cfg={},
                code_version=TEST_CODE_VERSION,
                admin_pg="postgresql://unused",
                base_redis_url="redis://unused",
                checkpoint_dir=checkpoint_dir,
            )
        )

    assert "STOP file found; not starting training-0000" in capsys.readouterr().out
    assert checkpoint_path(checkpoint_dir, "training", "training-0000").exists() is False


def test_stop_file_after_a_resumed_world_stops_before_the_next_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finished world resumes without a STOP check, and the next world that needs building is not started."""
    specs = _specs(2)
    checkpoint_dir = tmp_path / "checkpoints"
    _save_checkpoint(checkpoint_dir, specs[0])
    stop_file = checkpoint_dir / "training" / "STOP"
    stop_file.write_text("stop\n")
    _forbid_world_runs(monkeypatch)

    def forbidden_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a stopped build must not assemble a dataset")

    monkeypatch.setattr(build_module, "build_dataset", forbidden_build)

    with pytest.raises(BuildStopped):
        asyncio.run(
            build_split_dataset(
                split="training",
                specs=specs,
                preset_name="tiny",
                sampler_cfg={},
                code_version=TEST_CODE_VERSION,
                admin_pg="postgresql://unused",
                base_redis_url="redis://unused",
                checkpoint_dir=checkpoint_dir,
            )
        )

    out = capsys.readouterr().out
    assert "Resumed world training-0000" in out
    assert "STOP file found; not starting training-0001" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--world-shard", "0/1"],
        ["--assemble-only"],
        ["--checkpoint-dir", "cp", "--world-shard", "0/1", "--assemble-only"],
    ],
)
def test_new_flags_refuse_combinations_without_a_checkpoint_dir(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    """Each new flag needs --checkpoint-dir, and a shard worker cannot also assemble, so argparse exits 2."""
    monkeypatch.setattr(sys, "argv", ["build_level_16_datasets.py", *argv])

    with pytest.raises(SystemExit) as error:
        build_module.main()

    assert error.value.code == 2


def test_world_shard_flag_reaches_the_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--world-shard 1/3 with --checkpoint-dir reaches build_all as a parsed WorldShard."""
    calls: dict[str, Any] = {}

    async def fake_build(**kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(build_module, "build_all_level_16_datasets", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_level_16_datasets.py", "--checkpoint-dir", str(tmp_path), "--world-shard", "1/3"],
    )

    build_module.main()

    assert calls["world_shard"] == WorldShard(1, 3)
    assert calls["assemble_only"] is False


def test_assembly_follows_plan_order_not_completion_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checkpoints written in reverse order still feed build_dataset in plan order."""
    specs = _specs(2)
    checkpoint_dir = tmp_path / "checkpoints"
    _save_checkpoint(checkpoint_dir, specs[1])
    _save_checkpoint(checkpoint_dir, specs[0])

    captured: list[list[str]] = []

    def fake_build_dataset(
        extracts: list[WorldExtract], split: str, cutoff: datetime, code_version: str
    ) -> MagicMock:
        captured.append([extract.world_id for extract in extracts])
        return MagicMock()

    monkeypatch.setattr(build_module, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(build_module, "write_dataset", lambda dataset, root: tmp_path / "dataset")
    monkeypatch.setattr(build_module, "supersede_prior_datasets", lambda split, new_dir, datasets_root: [])

    asyncio.run(
        build_split_dataset(
            split="training",
            specs=specs,
            preset_name="tiny",
            sampler_cfg={},
            code_version=TEST_CODE_VERSION,
            datasets_root=tmp_path,
            admin_pg="postgresql://unused",
            base_redis_url="redis://unused",
            checkpoint_dir=checkpoint_dir,
            assemble_only=True,
        )
    )

    assert captured == [["training-0000", "training-0001"]]

    extracts, world_ends = load_plan_checkpoints(
        checkpoint_dir,
        "training",
        specs,
        preset_name="tiny",
        sampler_seed=1,
        code_version=TEST_CODE_VERSION,
    )
    assert [extract.world_id for extract in extracts] == ["training-0000", "training-0001"]
    assert world_ends == [datetime(2026, 9, 1, tzinfo=IST), datetime(2026, 9, 1, tzinfo=IST)]


def test_assembly_refuses_a_checkpoint_with_different_key_values(tmp_path: Path) -> None:
    """A checkpoint sampled with other key values than the plan's spec is refused, naming key_values_sha256."""
    spec = replace(_specs(1)[0], key_values=(("world_population", 60),))
    checkpoint_dir = tmp_path / "checkpoints"
    _save_checkpoint(checkpoint_dir, spec)
    planned = replace(spec, key_values=(("world_population", 61),))
    assert checkpoint_path(checkpoint_dir, "training", spec.world_id).is_file()

    with pytest.raises(ValueError) as error:
        load_plan_checkpoints(
            checkpoint_dir,
            "training",
            [planned],
            preset_name="tiny",
            sampler_seed=1,
            code_version=TEST_CODE_VERSION,
        )

    message = str(error.value)
    assert "key_values_sha256" in message
    assert f"checkpoint={key_values_fingerprint(spec.key_values)!r}" in message
    assert f"current={key_values_fingerprint(planned.key_values)!r}" in message
    assert "world_id" not in message
