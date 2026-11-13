# File: tests/unit/test_build_level_16_switches.py
# Purpose: Unit test for the replay-switch line printed by the Level 16 dataset build (Level 18a1).
"""
WHAT: Verifies that build_split_dataset prints one line naming the three replay switches before replay.

WHY: The build runs the same loader functions as run_replay, so a plain stdout line lets an unattended
build record which drain paths it used. The line must name all three switches.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

import scripts.build_level_16_datasets as build_module
from scripts.build_level_16_datasets import build_split_dataset

pytestmark = pytest.mark.anyio


async def test_build_prints_all_three_replay_switch_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies the stdout line names watermark_advance_skip_overlap, local_offset_tracking, registry_poll_on_change."""
    monkeypatch.setattr(build_module, "derive_dataset_cutoff", lambda worlds: datetime(2026, 1, 1, tzinfo=timezone.utc))
    monkeypatch.setattr(build_module, "build_dataset", lambda *a, **k: object())
    monkeypatch.setattr(build_module, "write_dataset", lambda *a, **k: Path("dataset-dir"))
    monkeypatch.setattr(build_module, "supersede_prior_datasets", lambda *a, **k: [])

    await build_split_dataset(
        split="test",
        specs=[],
        preset_name="tiny",
        sampler_cfg={},
        code_version="test",
        admin_pg="postgresql://ignored",
        base_redis_url="redis://ignored",
    )

    lines: list[str] = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Replay switches:")]
    assert len(lines) == 1
    line = lines[0]
    assert "watermark_advance_skip_overlap=" in line
    assert "local_offset_tracking=" in line
    assert "registry_poll_on_change=" in line
