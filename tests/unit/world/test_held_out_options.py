"""
WHAT: Unit test suite verifying held-out tactic option validation, split guards, and configuration bounds.

WHY: Level 12g requires held-out tactics (slow_drip and device_rotation) to be impossible in training and
validation worlds. These tests verify fail-closed enforcement across direct checks, planner execution,
structural rule validators, and whole-world runner integration.

ISSUE IDs: 12-8
"""

from pathlib import Path

import pytest
import yaml

from core.config_validator import load_yaml
from world import tactic_config
from world.build_world import build_world
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig, validate_gang_config
from world.gang_planner import plan_gang
from world.held_out_options import (
    check_held_out_option_ranges,
    check_held_out_options_allowed,
    load_exempt_splits,
)
from world.interactive_runner import RunnerSettings, run_interactive_world
from world.tactic_config import (
    TACTIC_BLOCKS_PATH,
    get_drip_hold_minutes_range,
    get_drip_piece_max_inr_range,
    get_drip_spread_days_range,
    get_rotate_every_days_range,
    get_rotations_range,
)

# Standard testing budget and source fixtures modeled after tests/unit/world/test_gang_runs.py.
TEST_BUDGET = GangBudget(money_paise=10_000_000, time_minutes=10_000, max_risk=0.5)
TEST_SOURCE = FixtureSource("acc_gang_src_01", 0, 10_000_000, 1440)
TINY_SETTINGS = FastRunSettings(101, 60, 12, 61, "training")


def _make_drip_gang(gang_id: str = "gang_drip") -> GangConfig:
    """Constructs a valid slow-drip test gang config with layered routing and calibrated drip settings."""
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("accounts_per_layer", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    return GangConfig(gang_id, 1, TEST_BUDGET, TEST_SOURCE, steps)


def _make_rotation_gang(gang_id: str = "gang_rot") -> GangConfig:
    """Constructs a valid device-rotation test gang config with rotation interval and turnover counts."""
    steps = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep(
            "take_control",
            (
                ("rotate_every_days", 7),
                ("rotations", 2),
            ),
        ),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 1))),
    )
    return GangConfig(gang_id, 1, TEST_BUDGET, TEST_SOURCE, steps)


def test_drip_refused_for_training_split() -> None:
    """Verifies that slow drip options are strictly refused when executed under the training evaluation split."""
    gang = _make_drip_gang()
    with pytest.raises(ValueError) as excinfo:
        check_held_out_options_allowed(gang, "training")
    message = str(excinfo.value)
    assert "drip_piece_max_inr" in message
    assert "slow_drip" in message
    assert "training" in message


def test_drip_refused_for_validation_split() -> None:
    """Verifies that slow drip options are strictly refused when executed under the validation evaluation split."""
    gang = _make_drip_gang()
    with pytest.raises(ValueError) as excinfo:
        check_held_out_options_allowed(gang, "validation")
    message = str(excinfo.value)
    assert "drip_piece_max_inr" in message
    assert "slow_drip" in message
    assert "validation" in message


def test_drip_refused_when_split_is_none() -> None:
    """Verifies that slow drip options are rejected fail-closed when no explicit evaluation split is provided."""
    gang = _make_drip_gang()
    with pytest.raises(ValueError) as excinfo:
        check_held_out_options_allowed(gang, None)
    message = str(excinfo.value)
    assert "drip_piece_max_inr" in message
    assert "slow_drip" in message
    assert "no split was given" in message


def test_drip_permitted_for_test_and_stress_splits() -> None:
    """Verifies that slow drip options pass the split admissibility check cleanly for test and stress splits."""
    gang = _make_drip_gang()
    # Both test and stress splits must pass the split permission check without raising an exception.
    check_held_out_options_allowed(gang, "test")
    check_held_out_options_allowed(gang, "stress")


def test_rotation_refused_for_training_split() -> None:
    """Verifies that device rotation options are strictly refused when executed under the training split."""
    gang = _make_rotation_gang()
    with pytest.raises(ValueError) as excinfo:
        check_held_out_options_allowed(gang, "training")
    message = str(excinfo.value)
    assert "rotate_every_days" in message
    assert "device_rotation" in message
    assert "training" in message


def test_load_exempt_splits_returns_policy_values() -> None:
    """Verifies that load_exempt_splits parses the expected test and stress evaluation split names from config."""
    splits = load_exempt_splits()
    assert splits == ("test", "stress")


def test_check_held_out_option_ranges_real_config() -> None:
    """Verifies that the production configuration satisfies all mathematical separation constraints between splits."""
    # Production tactic_blocks.yaml has drip_piece_max_inr upper 1999 < 2000 and drip_hold_minutes lower 240 > 60.
    check_held_out_option_ranges()


def test_check_held_out_option_ranges_fails_on_overlapping_drip_amount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that check_held_out_option_ranges raises ValueError when drip piece max overlaps the dividing line."""
    # Pattern copied from test_take_control_devices_range_follows_config in tests/unit/world/test_gang_config.py.
    blocks_data = load_yaml(TACTIC_BLOCKS_PATH)
    route_params = blocks_data["tactic_blocks"]["blocks"]["route"]["parameters"]
    route_params["drip_piece_max_inr"]["range"] = [500, 2000]

    temp_blocks_path = tmp_path / "tactic_blocks.yaml"
    with open(temp_blocks_path, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(blocks_data, file_handle)

    monkeypatch.setattr(tactic_config, "TACTIC_BLOCKS_PATH", temp_blocks_path)
    with pytest.raises(ValueError, match=r"Held-out drip_piece_max_inr upper bound \(2000\) must be strictly below"):
        check_held_out_option_ranges(temp_blocks_path)


def test_check_held_out_option_ranges_fails_on_overlapping_drip_dwell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that check_held_out_option_ranges raises ValueError when drip hold minutes overlaps training dwell."""
    blocks_data = load_yaml(TACTIC_BLOCKS_PATH)
    route_params = blocks_data["tactic_blocks"]["blocks"]["route"]["parameters"]
    route_params["drip_hold_minutes"]["range"] = [60, 2880]

    temp_blocks_path = tmp_path / "tactic_blocks.yaml"
    with open(temp_blocks_path, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(blocks_data, file_handle)

    monkeypatch.setattr(tactic_config, "TACTIC_BLOCKS_PATH", temp_blocks_path)
    with pytest.raises(ValueError, match=r"Held-out drip_hold_minutes lower bound \(60\) must be strictly above"):
        check_held_out_option_ranges(temp_blocks_path)


def test_structural_drip_requires_all_three_parameters() -> None:
    """Verifies that route validation refuses incomplete slow drip configurations missing any of the 3 parameters."""
    incomplete_steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("accounts_per_layer", 2),
                ("drip_piece_max_inr", 1500),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"requires all three drip parameters"):
        validate_gang_config(GangConfig("gang_incomplete_drip", 1, TEST_BUDGET, TEST_SOURCE, incomplete_steps))


def test_structural_drip_forbids_dwell_minutes() -> None:
    """Verifies that route validation refuses slow drip parameters paired with legacy dwell_minutes."""
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("accounts_per_layer", 2),
                ("dwell_minutes", 30),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"cannot set dwell_minutes or amount_inr"):
        validate_gang_config(GangConfig("gang_dwell_drip", 1, TEST_BUDGET, TEST_SOURCE, steps))


def test_structural_drip_forbids_amount_inr() -> None:
    """Verifies that route validation refuses slow drip parameters paired with legacy amount_inr."""
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("accounts_per_layer", 2),
                ("amount_inr", 5000),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"cannot set dwell_minutes or amount_inr"):
        validate_gang_config(GangConfig("gang_amount_drip", 1, TEST_BUDGET, TEST_SOURCE, steps))


def test_structural_drip_requires_layers_at_least_two() -> None:
    """Verifies that route validation refuses slow drip configurations with layers less than two or missing."""
    steps_one_layer = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 1),
                ("accounts_per_layer", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"requires layers >= 2"):
        validate_gang_config(GangConfig("gang_drip_1_layer", 1, TEST_BUDGET, TEST_SOURCE, steps_one_layer))

    steps_no_layer = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("accounts_per_layer", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"requires layers parameter with value >= 2, got none"):
        validate_gang_config(GangConfig("gang_drip_no_layer", 1, TEST_BUDGET, TEST_SOURCE, steps_no_layer))


def test_structural_drip_range_checks() -> None:
    """Verifies that slow drip parameter values outside calibrated YAML ranges are strictly rejected."""
    min_piece, max_piece = get_drip_piece_max_inr_range()
    min_spread, max_spread = get_drip_spread_days_range()
    min_hold, max_hold = get_drip_hold_minutes_range()

    # Out of range piece max (above max).
    steps_bad_piece = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("drip_piece_max_inr", max_piece + 1),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"drip_piece_max_inr must be an integer in"):
        validate_gang_config(GangConfig("gang_bad_piece", 1, TEST_BUDGET, TEST_SOURCE, steps_bad_piece))

    # Out of range spread days (below min).
    steps_bad_spread = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", min_spread - 1),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"drip_spread_days must be an integer in"):
        validate_gang_config(GangConfig("gang_bad_spread", 1, TEST_BUDGET, TEST_SOURCE, steps_bad_spread))

    # Out of range hold minutes (below min).
    steps_bad_hold = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", min_hold - 1),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"drip_hold_minutes must be an integer in"):
        validate_gang_config(GangConfig("gang_bad_hold", 1, TEST_BUDGET, TEST_SOURCE, steps_bad_hold))


def test_structural_drip_forbids_disguise_and_split() -> None:
    """Verifies that slow drip routes cannot coexist with disguise or split_below_threshold tactic blocks."""
    steps_disguise = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", 300),
            ),
        ),
        BlockStep("disguise", (("jitter_minutes", 10),)),
    )
    with pytest.raises(ValueError, match=r"cannot contain disguise or split_below_threshold"):
        validate_gang_config(GangConfig("gang_disguise_drip", 2, TEST_BUDGET, TEST_SOURCE, steps_disguise))

    steps_split = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("split_below_threshold", (("threshold_inr", 50000),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 5),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"cannot contain disguise or split_below_threshold"):
        validate_gang_config(GangConfig("gang_split_drip", 1, TEST_BUDGET, TEST_SOURCE, steps_split))


def test_structural_rotation_requires_both_parameters() -> None:
    """Verifies that take_control device rotation requires rotate_every_days and rotations together."""
    steps_only_days = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep("take_control", (("rotate_every_days", 7),)),
    )
    with pytest.raises(ValueError, match=r"requires rotate_every_days and rotations to be set together"):
        validate_gang_config(GangConfig("gang_rot_days", 1, TEST_BUDGET, TEST_SOURCE, steps_only_days))

    steps_only_rot = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep("take_control", (("rotations", 2),)),
    )
    with pytest.raises(ValueError, match=r"requires rotate_every_days and rotations to be set together"):
        validate_gang_config(GangConfig("gang_rot_num", 1, TEST_BUDGET, TEST_SOURCE, steps_only_rot))


def test_structural_rotation_forbids_shared_devices() -> None:
    """Verifies that take_control device rotation cannot appear alongside the legacy shared devices parameter."""
    steps = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep(
            "take_control",
            (
                ("devices", 2),
                ("rotate_every_days", 7),
                ("rotations", 2),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"cannot specify devices together with rotation options"):
        validate_gang_config(GangConfig("gang_rot_devices", 1, TEST_BUDGET, TEST_SOURCE, steps))


def test_structural_rotation_range_checks() -> None:
    """Verifies that take_control rotation parameter values outside calibrated YAML ranges are refused."""
    min_days, max_days = get_rotate_every_days_range()
    min_rot, max_rot = get_rotations_range()

    steps_bad_days = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep("take_control", (("rotate_every_days", max_days + 1), ("rotations", 2))),
    )
    with pytest.raises(ValueError, match=r"rotate_every_days must be an integer in"):
        validate_gang_config(GangConfig("gang_bad_rot_days", 1, TEST_BUDGET, TEST_SOURCE, steps_bad_days))

    steps_bad_rot = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep("take_control", (("rotate_every_days", 7), ("rotations", min_rot - 1))),
    )
    with pytest.raises(ValueError, match=r"rotations must be an integer in"):
        validate_gang_config(GangConfig("gang_bad_rotations", 1, TEST_BUDGET, TEST_SOURCE, steps_bad_rot))


def test_plan_gang_refuses_rotation_held_out_option_before_c3() -> None:
    """Verifies that plan_gang refuses planning for rotation options because planning arrives in C3."""
    world = build_world(TINY_SETTINGS.seed, TINY_SETTINGS.n_people, TINY_SETTINGS.n_shops)
    import numpy as np
    rng = np.random.default_rng(42)

    gang = _make_rotation_gang()
    with pytest.raises(ValueError, match=r"held-out option planning is not built yet \(Level 12g C2/C3\)"):
        plan_gang(gang, world, rng, world_split="test")


def test_runner_refuses_drip_gang_in_training_world() -> None:
    """Verifies that run_interactive_world refuses a drip gang under training split before planning commences."""
    # Pattern copied from test_gang_runs.py (run_interactive_world with TINY, RunnerSettings, gang_configs).
    gang = _make_drip_gang()
    with pytest.raises(ValueError) as excinfo:
        run_interactive_world(
            TINY_SETTINGS,
            RunnerSettings(fraud_enabled=True),
            gang_configs=(gang,),
            world_split="training",
        )
    message = str(excinfo.value)
    assert "drip_piece_max_inr" in message
    assert "slow_drip" in message
    assert "training" in message
