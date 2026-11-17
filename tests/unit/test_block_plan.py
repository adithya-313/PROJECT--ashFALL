"""
File: tests/unit/test_block_plan.py
Purpose: Unit tests for the committed Level 12f block plan: parsing --blocks, building each block
         with sample_block, the 40-world pin between the legacy plan and block 0, seed disjointness,
         and the legacy-count guard.

WHAT: Every test calls the real production functions; the sampler and the world ranges are the
repository's own config files.

WHY: A plan of 40 worlds per split must be block 0 of the committed plan, not one hypercube of 40,
and a legacy count above one block would silently build worlds that are not part of that plan.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from scripts.build_level_16_datasets import (
    build_all_level_16_datasets,
    parse_blocks,
    plan_blocks,
    plan_datasets,
    plan_run,
)
from world.config_reader import read_world_ranges
from world.world_sampler import WorldSpec, load_sampler_config, sample_block

SPLITS = ["training", "validation", "test", "stress"]


def _configs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns the repository sampler config and world ranges."""
    return load_sampler_config(), read_world_ranges()


def _key_values(specs: list[WorldSpec]) -> list[tuple[tuple[str, float | int], ...]]:
    """Returns the key values of each spec, in plan order."""
    return [spec.key_values for spec in specs]


def test_parse_blocks_reads_inclusive_ranges() -> None:
    """Each piece becomes the inclusive block list it names, in the order given."""
    assert parse_blocks("validation=1-2,test=0-0,training=1-1") == {
        "validation": [1, 2],
        "test": [0],
        "training": [1],
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("holdout=0-1", "unknown split 'holdout'"),
        ("training=1--2", "must be at least 0"),
        ("training=2-1", "starts at 2 after it ends at 1"),
        ("training=0-0,training=1-1", "appears more than once"),
        ("training=0-0,,test=0-0", "empty block piece"),
        ("", "empty block piece"),
        ("training=0", "is not start-end"),
        ("training=a-b", "is not two integers"),
    ],
)
def test_parse_blocks_refuses_a_bad_piece(text: str, message: str) -> None:
    """Every malformed piece is refused with a message naming it."""
    with pytest.raises(ValueError, match=message):
        parse_blocks(text)


def test_plan_blocks_concatenates_the_named_blocks_in_order() -> None:
    """Validation blocks [1, 2] are sample_block 1 followed by sample_block 2, ids and seeds included."""
    sampler_cfg, world_ranges = _configs()

    planned = plan_blocks({"validation": [2, 1]}, 2, sampler_cfg, world_ranges)

    expected = sample_block("validation", 1, 2, sampler_cfg, world_ranges) + sample_block(
        "validation", 2, 2, sampler_cfg, world_ranges
    )
    assert [spec.world_id for spec in planned["validation"]] == [spec.world_id for spec in expected]
    assert [spec.seed for spec in planned["validation"]] == [spec.seed for spec in expected]
    assert _key_values(planned["validation"]) == _key_values(expected)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("sampler_seed", [1, 2])
def test_forty_legacy_worlds_are_block_zero(split: str, sampler_seed: int) -> None:
    """PIN: a legacy plan of 40 worlds per split has block 0's key values, world for world."""
    sampler_cfg, world_ranges = _configs()

    legacy = plan_datasets({split: 40}, sampler_cfg, world_ranges, sampler_seed=sampler_seed)
    blocked = plan_blocks({split: [0]}, sampler_seed, sampler_cfg, world_ranges)

    assert _key_values(legacy[split]) == _key_values(blocked[split])


def test_blocks_have_no_repeated_seeds_and_validation_stays_apart_from_training() -> None:
    """Two blocks never repeat a seed, and no validation block shares a seed with a training block."""
    sampler_cfg, world_ranges = _configs()

    planned = plan_blocks(
        {"training": [0, 1], "validation": [0, 1, 2, 3, 4, 5, 6, 7]},
        2,
        sampler_cfg,
        world_ranges,
    )

    training_seeds = [spec.seed for spec in planned["training"]]
    validation_seeds = [spec.seed for spec in planned["validation"]]
    assert len(set(training_seeds)) == len(training_seeds)
    assert len(set(validation_seeds)) == len(validation_seeds)
    assert set(training_seeds).isdisjoint(validation_seeds)


def test_legacy_count_above_one_block_is_refused_and_block_size_is_accepted() -> None:
    """Without blocks, exactly block_size worlds is the last legacy plan allowed; one more is refused."""
    sampler_cfg, world_ranges = _configs()
    block_size = int(sampler_cfg["block_size"]["value"])

    accepted = plan_run({"training": block_size}, None, 2, sampler_cfg, world_ranges)
    assert len(accepted["training"]) == block_size
    with pytest.raises(ValueError, match=f"above the {block_size}-world block"):
        plan_run({"training": block_size + 1}, None, 2, sampler_cfg, world_ranges)


def test_blocks_and_worlds_per_split_cannot_be_combined(tmp_path: Path) -> None:
    """Naming blocks and overriding counts would describe two different plans, so the build refuses."""
    with pytest.raises(ValueError, match="--blocks and --worlds-per-split cannot be used together"):
        asyncio.run(
            build_all_level_16_datasets(
                splits=["training"],
                worlds_per_split={"training": 2},
                blocks={"training": [0]},
                checkpoint_dir=tmp_path / "checkpoints",
                datasets_root=tmp_path / "datasets",
            )
        )
