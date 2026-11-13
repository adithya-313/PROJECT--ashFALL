"""
File: tests/unit/test_dataset_build_plan.py
Purpose: Fast unit tests for Level 16 dataset build planning, split sampling,
         prior version superseding, and cutoff calculation without running slow replays.
"""

import dataclasses
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import orjson
import pyarrow as pa
import pytest

import scripts.build_level_16_datasets as build_module
from replay.dataset_builder import Dataset
from replay.runner import IST
from scripts.build_level_16_datasets import (
    derive_dataset_cutoff,
    find_fitting_sampler_seed,
    load_datasets_config,
    plan_datasets,
    summarize_labels_by_source_and_via,
    supersede_prior_datasets,
)
from world.world_sampler import WorldSpec


def _patch_seed_scan(
    monkeypatch: pytest.MonkeyPatch,
    fits_from: int,
    attempted_seeds: list[int] | None = None,
) -> None:
    """
    Marks each planned spec with the sampler seed that produced it and replaces the gang resolver
    with one that refuses every seed below fits_from for too few eligible people. When
    attempted_seeds is given, every scanned seed is appended to it once, in scan order.
    """
    real_plan_datasets = build_module.plan_datasets

    def marked_plan_datasets(
        datasets_cfg: dict[str, int] | None = None,
        sampler_cfg: dict[str, Any] | None = None,
        world_ranges: dict[str, Any] | None = None,
        sampler_seed: int = 1,
    ) -> dict[str, list[WorldSpec]]:
        plan = real_plan_datasets(datasets_cfg, sampler_cfg, world_ranges, sampler_seed=sampler_seed)
        return {
            split: [dataclasses.replace(spec, world_id=f"{spec.world_id}-sampler-{sampler_seed}") for spec in specs]
            for split, specs in plan.items()
        }

    def resolver(
        spec: WorldSpec,
        preset_name: str,
        gang_configs: tuple[Any, ...] | None,
        sampler_cfg: dict[str, Any],
        configs: Any,
    ) -> tuple[Any, ...]:
        scan_seed = int(spec.world_id.rsplit("-", 1)[1])
        if attempted_seeds is not None and scan_seed not in attempted_seeds:
            attempted_seeds.append(scan_seed)
        if scan_seed < fits_from:
            raise ValueError(
                f"the gang mix of {spec.world_id} needs 63 people as mules and victims, "
                "but the world has only 60 eligible people"
            )
        return ()

    monkeypatch.setattr(build_module, "plan_datasets", marked_plan_datasets)
    monkeypatch.setattr(build_module, "resolve_sampled_world_gangs", resolver)


def test_load_datasets_config_reads_expected_splits() -> None:
    """Verifies that load_datasets_config reads config/datasets.yaml with all 4 splits and expected world counts."""
    cfg = load_datasets_config()
    assert isinstance(cfg, dict)
    assert set(cfg.keys()) == {"training", "validation", "test", "stress"}
    assert cfg["training"] == 2
    assert cfg["validation"] == 1
    assert cfg["test"] == 1
    assert cfg["stress"] == 1


def test_plan_datasets_generates_expected_world_specs() -> None:
    """Verifies that plan_datasets plans the exact world counts and seeds for each split."""
    plan = plan_datasets(sampler_seed=1)
    assert len(plan["training"]) == 2
    assert len(plan["validation"]) == 1
    assert len(plan["test"]) == 1
    assert len(plan["stress"]) == 1

    # Verify seeds conform to configured seed blocks
    assert plan["training"][0].seed == 1000000
    assert plan["training"][1].seed == 1000001
    assert plan["validation"][0].seed == 2000000
    assert plan["test"][0].seed == 3000000
    assert plan["stress"][0].seed == 4000000

    # Verify world_ids
    assert plan["training"][0].world_id == "training-0000"
    assert plan["training"][1].world_id == "training-0001"
    assert plan["validation"][0].world_id == "validation-0000"
    assert plan["test"][0].world_id == "test-0000"
    assert plan["stress"][0].world_id == "stress-0000"


def test_supersede_prior_datasets_updates_matching_split_only(tmp_path: Path) -> None:
    """Verifies that supersede_prior_datasets marks only older datasets of the same split."""
    # Create an old training dataset folder
    old_train = tmp_path / "training-20260901T0000-old111111111"
    old_train.mkdir()
    (old_train / "manifest.json").write_bytes(
        orjson.dumps({"split": "training", "content_hash": "oldhash1"})
    )

    # Create an old validation dataset folder (different split, must NOT be marked)
    old_val = tmp_path / "validation-20260901T0000-old222222222"
    old_val.mkdir()
    (old_val / "manifest.json").write_bytes(
        orjson.dumps({"split": "validation", "content_hash": "oldhash2"})
    )

    # Create the new training dataset folder
    new_train = tmp_path / "training-20261101T0000-new333333333"
    new_train.mkdir()
    (new_train / "manifest.json").write_bytes(
        orjson.dumps({"split": "training", "content_hash": "newhash3"})
    )

    superseded = supersede_prior_datasets("training", new_train, datasets_root=tmp_path)
    assert superseded == [old_train]

    # Verify old training manifest has superseded_by pointing to new_train.name
    updated_train = orjson.loads((old_train / "manifest.json").read_bytes())
    assert updated_train.get("superseded_by") == new_train.name

    # Verify validation manifest is completely untouched
    untouched_val = orjson.loads((old_val / "manifest.json").read_bytes())
    assert "superseded_by" not in untouched_val

    # Second run should find 0 newly superseded datasets
    superseded_again = supersede_prior_datasets("training", new_train, datasets_root=tmp_path)
    assert superseded_again == []


def test_supersede_prior_datasets_handles_empty_root(tmp_path: Path) -> None:
    """Verifies that supersede_prior_datasets returns empty list on non-existent or empty directory."""
    non_existent = tmp_path / "does_not_exist"
    assert supersede_prior_datasets("training", tmp_path / "new", datasets_root=non_existent) == []

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    new_dir = empty_dir / "new_dataset"
    new_dir.mkdir()
    (new_dir / "manifest.json").write_bytes(orjson.dumps({"split": "training"}))
    assert supersede_prior_datasets("training", new_dir, datasets_root=empty_dir) == []


def test_derive_dataset_cutoff_uses_max_world_end() -> None:
    """Verifies that derive_dataset_cutoff derives the maximum simulation end timestamp."""
    # World 1: starts 2026-09-01, 61 days -> ends 2026-11-01 00:00:00 IST
    w1 = MagicMock()
    w1.output.plan.clock.start_date = datetime(2026, 9, 1, 9, 0, 0, tzinfo=IST)
    w1.output.plan.n_days = 61

    # World 2: starts 2026-09-01, 120 days -> ends 2026-12-30 00:00:00 IST
    w2 = MagicMock()
    w2.output.plan.clock.start_date = datetime(2026, 9, 1, 9, 0, 0, tzinfo=IST)
    w2.output.plan.n_days = 120

    cutoff = derive_dataset_cutoff([w1, w2])
    expected = datetime(2026, 9, 1, 0, 0, 0, tzinfo=IST) + timedelta(days=120)
    assert cutoff == expected

    # Empty list must raise ValueError
    with pytest.raises(ValueError, match="empty recorded worlds"):
        derive_dataset_cutoff([])


def test_summarize_labels_by_source_and_via() -> None:
    """Verifies label aggregation breakdown across complaint direct, hold request, registry, police."""
    labels_table = pa.Table.from_arrays(
        [
            pa.array(["complaint", "complaint", "registry", "police_bulk"]),
            pa.array([None, "hold_request", None, None]),
        ],
        names=["source", "via"],
    )
    mock_dataset = Dataset(
        payments=pa.table({}),
        snapshots=pa.table({}),
        labels=labels_table,
        manifest={},
    )
    summary = summarize_labels_by_source_and_via(mock_dataset)
    assert summary == {
        "complaint": 1,
        "complaint (via hold_request)": 1,
        "registry": 1,
        "police_bulk": 1,
    }


def test_main_defaults_the_sampler_seed_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A build without --sampler-seed passes seed 1, so today's datasets stay reproducible."""
    calls: dict[str, Any] = {}

    async def fake_build(**kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(build_module, "build_all_level_16_datasets", fake_build)
    monkeypatch.setattr(sys, "argv", ["build_level_16_datasets.py"])

    build_module.main()

    assert calls["sampler_seed"] == 1


def test_find_fitting_sampler_seed_skips_the_seed_that_does_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed 1 needs more people than the world has, so the scan returns the next seed, 2."""
    _patch_seed_scan(monkeypatch, fits_from=2)

    seed = find_fitting_sampler_seed(["training"], "tiny", start_seed=1, max_tries=5)

    assert seed == 2


def test_find_fitting_sampler_seed_propagates_other_value_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal that is not about eligible people is a real failure, not a seed that does not fit."""
    _patch_seed_scan(monkeypatch, fits_from=1)

    def failing_resolver(
        spec: WorldSpec,
        preset_name: str,
        gang_configs: tuple[Any, ...] | None,
        sampler_cfg: dict[str, Any],
        configs: Any,
    ) -> tuple[Any, ...]:
        raise ValueError("world ranges are misconfigured")

    monkeypatch.setattr(build_module, "resolve_sampled_world_gangs", failing_resolver)

    with pytest.raises(ValueError, match="world ranges are misconfigured"):
        find_fitting_sampler_seed(["training"], "tiny", start_seed=1, max_tries=3)


def test_find_fitting_sampler_seed_refuses_when_every_try_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two unfitting seeds exhaust the scan and the error names the start seed, tries and split."""
    _patch_seed_scan(monkeypatch, fits_from=100)

    with pytest.raises(ValueError, match="no sampler seed from 3 fits training within 2 tries"):
        find_fitting_sampler_seed(["training"], "tiny", start_seed=3, max_tries=2)


def test_find_sampler_seed_cli_starts_at_the_sampler_seed_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With --sampler-seed 3 and a fit from seed 5, the CLI scans from 3 and prints exactly seed 5."""
    attempted_seeds: list[int] = []
    _patch_seed_scan(monkeypatch, fits_from=5, attempted_seeds=attempted_seeds)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_level_16_datasets.py", "--find-sampler-seed", "--sampler-seed", "3", "--splits", "training"],
    )

    build_module.main()

    assert capsys.readouterr().out == "fitting sampler seed: 5\n"
    assert attempted_seeds == [3, 4, 5]
