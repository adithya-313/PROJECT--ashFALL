"""
File: tests/unit/test_parallel_workers.py
Purpose: Unit tests for the --parallel-workers launcher: the child command line, the no-shell start,
         assembly only after every child succeeded, the failure exit code, and the pool limit.

WHAT: Every test calls the real production functions; only the child runner and the dataset steps
are replaced, so no process is started and no replay runs.

WHY: A failed child must never be hidden by an assembly, and the worker count must follow the
configured replay workspace pool instead of a typed-in number.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import scripts.build_level_16_datasets as build_module
from scripts.build_level_16_datasets import (
    WorkerCommand,
    WorkerFailure,
    build_all_level_16_datasets,
    start_worker_process,
    validate_parallel_workers,
    worker_argv,
)
from scripts.dataset_shards import WorldShard
from world.replay_workspace import load_replay_settings
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


def _patch_build(monkeypatch: pytest.MonkeyPatch, codes: list[int], specs: list[WorldSpec]) -> dict[str, Any]:
    """Replaces planning, the child runner and the dataset steps, and returns the recorded calls."""
    calls: dict[str, Any] = {"assembled": [], "commands": []}

    def fake_plan(*args: Any, **kwargs: Any) -> dict[str, list[WorldSpec]]:
        return {"training": specs}

    def fake_run(commands: list[WorkerCommand]) -> list[int]:
        calls["commands"].append(commands)
        return codes

    async def fake_build_split(**kwargs: Any) -> tuple[MagicMock, Path, list[Path]]:
        calls["assembled"].append(kwargs)
        return MagicMock(), Path("dataset-dir"), []

    monkeypatch.setattr(build_module, "plan_datasets", fake_plan)
    monkeypatch.setattr(build_module, "run_worker_commands", fake_run)
    monkeypatch.setattr(build_module, "build_split_dataset", fake_build_split)
    monkeypatch.setattr(build_module, "summarize_labels_by_source_and_via", lambda dataset: {})
    return calls


def _run_parallel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int) -> dict[str, Any]:
    """Runs build_all for one 2-world training split with 2 children that both succeed."""
    specs = _specs(2)
    calls = _patch_build(monkeypatch, [0, 0], specs)
    asyncio.run(
        build_all_level_16_datasets(
            splits=["training"],
            worlds_per_split={"training": 2},
            checkpoint_dir=tmp_path / "checkpoints",
            datasets_root=tmp_path / "datasets",
            parallel_workers=workers,
            base_argv=["--checkpoint-dir", str(tmp_path / "checkpoints"), "--parallel-workers", str(workers)],
        )
    )
    return calls


def test_worker_argv_adds_the_shard_and_strips_the_parallel_flag() -> None:
    """The child command keeps the parent flags, drops --parallel-workers/--splits, and adds its own."""
    base = ["--checkpoint-dir", "cp", "--splits", "training,validation", "--parallel-workers", "4", "--preset", "tiny"]

    argv = worker_argv(base, WorldShard(2, 4), "training")

    assert argv[0] == sys.executable
    assert argv[1].endswith("build_level_16_datasets.py")
    assert "--parallel-workers" not in argv
    assert argv[argv.index("--splits") + 1] == "training"
    assert argv.count("--splits") == 1
    assert argv[argv.index("--world-shard") + 1] == "2/4"
    assert "--preset" in argv


def test_worker_argv_strips_the_equals_forms_too() -> None:
    """--parallel-workers=N and --splits=... written with '=' are dropped as well."""
    argv = worker_argv(["--parallel-workers=4", "--splits=training", "--preset", "tiny"], WorldShard(0, 2), "test")

    assert "--parallel-workers" not in argv
    assert all(not arg.startswith("--parallel-workers=") for arg in argv)
    assert all(not arg.startswith("--splits=") for arg in argv)
    assert argv[argv.index("--splits") + 1] == "test"
    assert argv[argv.index("--world-shard") + 1] == "0/2"


def test_start_worker_process_uses_an_argument_list_and_no_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The child is started from a list with stderr merged into its log and without a shell."""
    captured: dict[str, Any] = {}

    def fake_popen(argv: list[str], **kwargs: Any) -> MagicMock:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(build_module.subprocess, "Popen", fake_popen)
    log_path = tmp_path / "training" / "worker-0.log"

    start_worker_process(["py", "script.py", "--world-shard", "0/2"], log_path)

    assert captured["argv"] == ["py", "script.py", "--world-shard", "0/2"]
    assert captured["kwargs"].get("shell") is None
    assert captured["kwargs"]["stderr"] == subprocess.STDOUT
    assert log_path.is_file() is True


def test_all_children_succeed_and_the_split_is_assembled_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two children exit 0 and the parent assembles the split once from checkpoints."""
    calls = _run_parallel(tmp_path, monkeypatch, workers=2)

    assert len(calls["commands"]) == 1
    commands = calls["commands"][0]
    assert [command.index for command in commands] == [0, 1]
    assert commands[0].world_ids == ("training-0000",)
    assert commands[1].world_ids == ("training-0001",)
    assert len(calls["assembled"]) == 1
    assert calls["assembled"][0]["assemble_only"] is True

    out = capsys.readouterr().out
    assert "Worker 0: worlds training-0000" in out
    assert "Worker 1: worlds training-0001" in out


def test_one_parallel_worker_keeps_todays_single_process_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--parallel-workers 1 starts no child and builds the split directly."""
    calls = _run_parallel(tmp_path, monkeypatch, workers=1)

    assert calls["commands"] == []
    assert len(calls["assembled"]) == 1
    assert calls["assembled"][0]["assemble_only"] is False


@pytest.mark.parametrize(("codes", "expected"), [([0, 3], 3), ([0, 1], 1), ([3, 3], 3), ([3, 1], 1)])
def test_child_failure_blocks_assembly_and_sets_the_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    codes: list[int],
    expected: int,
) -> None:
    """A non-zero child means no assembly; the code is 3 only when every failed child stopped."""
    specs = _specs(2)
    calls = _patch_build(monkeypatch, codes, specs)
    for index, code in enumerate(codes):
        log_path = tmp_path / "checkpoints" / "training" / f"worker-{index}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(f"line {number}" for number in range(1, 7)) + "\n")

    with pytest.raises(WorkerFailure) as error:
        asyncio.run(
            build_all_level_16_datasets(
                splits=["training"],
                worlds_per_split={"training": 2},
                checkpoint_dir=tmp_path / "checkpoints",
                datasets_root=tmp_path / "datasets",
                parallel_workers=2,
                base_argv=["--parallel-workers", "2"],
            )
        )

    assert error.value.exit_code == expected
    assert calls["assembled"] == []
    out = capsys.readouterr().out
    assert "Worker 1 exited {}".format(codes[1]) in out
    for number in range(2, 7):
        assert f"line {number}" in out
    assert "line 1" not in out


def test_parallel_workers_above_the_workspace_pool_is_refused() -> None:
    """One worker more than the configured Redis workspace pool is refused, naming the pool."""
    settings = load_replay_settings()
    pool_size = settings.redis_workspace_db_max - settings.redis_workspace_db_min + 1

    with pytest.raises(ValueError, match=f"exceeds the replay workspace pool of {pool_size}"):
        validate_parallel_workers(pool_size + 1)

    assert validate_parallel_workers(pool_size) == pool_size


def test_parallel_workers_below_one_is_refused() -> None:
    """A worker count of zero is refused before any child can start."""
    with pytest.raises(ValueError, match="at least 1"):
        validate_parallel_workers(0)
