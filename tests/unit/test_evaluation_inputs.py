"""
WHAT: Unit tests for saving Level 17 evaluation inputs from the dataset builder.

WHY: The evaluation runner needs hidden truth and a base rate for every dataset world, written
outside the operational tables. These tests prove the files appear, a second save for the same
run is refused, and the build path stays untouched when the feature is off.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import orjson
import pytest

from core.config_validator import load_yaml
from replay.runner import IST
from scripts import build_level_16_datasets as builder
from scripts.build_level_16_datasets import build_split_dataset, plan_datasets, save_evaluation_inputs
from world.fast_mode import FastRunSettings
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.replay_recording import RecordedWorld
from world.run_identity import REPO_ROOT

FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
TEST_CODE_VERSION = "test-version-evaluation-inputs"

TRUTH_FILES = (
    "roles.jsonl",
    "device_control.jsonl",
    "payment_stories.jsonl",
    "gangs.jsonl",
    "personas.jsonl",
    "manifest.json",
)


def _tiny_recorded() -> RecordedWorld:
    """Runs the frozen tiny world with fraud off, as the truth-writer tests do."""
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    return RecordedWorld(output=output, calls=())


def test_save_evaluation_inputs_writes_truth_and_base_rate(tmp_path: Path) -> None:
    """The run truth folder holds all truth files plus personas and manifest, and a base rate file."""
    recorded = _tiny_recorded()
    run_id = recorded.output.result.run_id
    truth_root = tmp_path / "truth"
    world_cards_root = tmp_path / "world_cards"

    result = save_evaluation_inputs(recorded, truth_root, world_cards_root)

    truth_dir = truth_root / run_id
    for name in TRUTH_FILES:
        assert (truth_dir / name).is_file()

    base_rate_path = world_cards_root / run_id / "base_rate.json"
    assert base_rate_path.is_file()
    base_rate = orjson.loads(base_rate_path.read_bytes())
    assert "fraud_inflation_factor" in base_rate

    assert result == {
        "run_id": run_id,
        "truth_dir": str(truth_dir),
        "base_rate_path": str(base_rate_path),
        "fraud_inflation_factor": base_rate["fraud_inflation_factor"],
    }


def test_save_evaluation_inputs_refuses_to_overwrite_the_same_run(tmp_path: Path) -> None:
    """A second save for the same run id raises instead of replacing the first truth files."""
    recorded = _tiny_recorded()
    truth_root = tmp_path / "truth"
    world_cards_root = tmp_path / "world_cards"
    save_evaluation_inputs(recorded, truth_root, world_cards_root)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_evaluation_inputs(recorded, truth_root, world_cards_root)


def test_build_split_dataset_without_the_flag_never_saves_evaluation_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the flag off the build runs end to end without calling save_evaluation_inputs."""
    call_count = 0

    def forbidden_save(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        raise AssertionError("save_evaluation_inputs must not run when the flag is off")

    def fake_record_world(
        spec: Any,
        preset_name: str,
        sampler_cfg: dict[str, Any],
        code_version: str,
        loop: asyncio.AbstractEventLoop,
    ) -> MagicMock:
        recorded = MagicMock()
        recorded.output.plan.clock.start_date = datetime(2026, 9, 1, 9, 0, tzinfo=IST)
        recorded.output.plan.n_days = 1
        return recorded

    async def fake_replay_and_extract(recorded: MagicMock, admin_pg: str, base_redis_url: str) -> MagicMock:
        return MagicMock()

    def fake_build_dataset(
        extracts: list[MagicMock],
        split: str,
        cutoff: datetime,
        code_version: str,
    ) -> MagicMock:
        return MagicMock()

    def fake_write_dataset(dataset: MagicMock, root: Path) -> Path:
        dataset_dir = root / "training-fake"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "manifest.json").write_bytes(orjson.dumps({"split": "training"}))
        return dataset_dir

    monkeypatch.setattr(builder, "save_evaluation_inputs", forbidden_save)
    monkeypatch.setattr(builder, "record_world_for_spec", fake_record_world)
    monkeypatch.setattr(builder, "replay_and_extract", fake_replay_and_extract)
    monkeypatch.setattr(builder, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(builder, "write_dataset", fake_write_dataset)

    dataset, dataset_dir, superseded = asyncio.run(
        build_split_dataset(
            split="training",
            specs=plan_datasets()["training"][:1],
            preset_name="tiny",
            sampler_cfg={},
            code_version=TEST_CODE_VERSION,
            datasets_root=tmp_path,
            admin_pg="postgresql://unused",
            base_redis_url="redis://unused",
        )
    )

    assert call_count == 0
    assert dataset_dir == tmp_path / "training-fake"
    assert superseded == []
