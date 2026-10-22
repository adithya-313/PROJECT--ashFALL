"""
WHAT: Unit and integration tests for Level 12d gang rotation and amount-splitting reactions.
Validates configuration schema constraints for new keys, GangHolding depth propagation,
pure account rotation selection via choose_spare_account, and trouble telemetry updates.

WHY: Demonstrates that reactive criminal syndicates evaluate operational resilience
mechanisms without violating information isolation or corrupting random streams.

TERM: rotation rule: rerouting illicit payments away from compromised mules to viable spares.
TERM: split rule: structuring transfers exceeding a threshold into smaller equal pieces.
TERM: trouble telemetry: any delivered observation indicating friction on held infrastructure.
"""

from typing import Any

import numpy as np
import pytest

from world.gang_feed import (
    GangHolding,
    GangObservation,
    build_gang_holdings_from_roles,
    load_gang_reactions_config,
    validate_gang_reactions_config,
)
from world.gang_reactions import (
    GangReactionState,
    choose_spare_account,
    draw_rotation_flag,
    draw_split_flag,
    draw_split_threshold_paise,
    resolve_rotation_share_range,
    resolve_split_share_range,
    resolve_split_threshold_range,
    update_gang_reaction,
)
from world.truth_records import RoleSpan


def test_gang_reactions_config_validation_for_new_keys() -> None:
    """
    WHAT: Validates schema constraints on rotation share, split share, and split threshold.
    WHY: Enforces that both file configuration and test fixtures adhere to valid parameter bounds.
    """
    # 1. Base valid file configuration loaded directly from disk passes validation.
    base_cfg = load_gang_reactions_config()
    validate_gang_reactions_config(base_cfg)

    # 2. Non-fixture configuration missing new required keys must raise ValueError.
    policy_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "testing required keys",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [10, 60]},
            "gang_feed_miss_share": {"training": [0.0, 0.1]},
            "gang_stops_using_liened_account_share": {"training": [0.5, 0.9]},
        }
    }
    with pytest.raises(ValueError, match="missing required key 'gang_rotates_to_spare_account_share'"):
        validate_gang_reactions_config(policy_cfg)

    # 3. Invalid rotation share range (bounds outside [0, 1] or lower > upper) must raise.
    bad_rot_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "bad rotation bounds",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [10, 60]},
            "gang_feed_miss_share": {"training": [0.0, 0.1]},
            "gang_stops_using_liened_account_share": {"training": [0.5, 0.9]},
            "gang_rotates_to_spare_account_share": {"training": [0.9, 0.4]},
        }
    }
    with pytest.raises(ValueError, match="rotation share range"):
        validate_gang_reactions_config(bad_rot_cfg)

    # 4. Invalid split share range (negative lower bound) must raise.
    bad_split_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "bad split bounds",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [10, 60]},
            "gang_feed_miss_share": {"training": [0.0, 0.1]},
            "gang_stops_using_liened_account_share": {"training": [0.5, 0.9]},
            "gang_splits_amounts_share": {"training": [-0.1, 0.5]},
        }
    }
    with pytest.raises(ValueError, match="split share range"):
        validate_gang_reactions_config(bad_split_cfg)

    # 5. Invalid split threshold in INR (threshold <= 0 or lower > upper) must raise.
    bad_thresh_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "bad threshold bounds",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [10, 60]},
            "gang_feed_miss_share": {"training": [0.0, 0.1]},
            "gang_stops_using_liened_account_share": {"training": [0.5, 0.9]},
            "gang_split_threshold_inr": {"training": [50000, 10000]},
        }
    }
    with pytest.raises(ValueError, match="split threshold range"):
        validate_gang_reactions_config(bad_thresh_cfg)

    zero_thresh_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "zero threshold bounds",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [10, 60]},
            "gang_feed_miss_share": {"training": [0.0, 0.1]},
            "gang_stops_using_liened_account_share": {"training": [0.5, 0.9]},
            "gang_split_threshold_inr": {"training": [0, 10000]},
        }
    }
    with pytest.raises(ValueError, match="split threshold range"):
        validate_gang_reactions_config(zero_thresh_cfg)


def test_gang_holding_depth_and_role_propagation() -> None:
    """
    WHAT: Verifies GangHolding depth validation and copying from RoleSpan records.
    WHY: Rotation relies on operational depth matching between stopped accounts and spares.
    """
    # 1. Valid depth values (depth >= 1 or None) are accepted.
    h1 = GangHolding("g1", "acc_1", 100, 500, depth=1)
    assert h1.depth == 1

    h_default = GangHolding("g1", "acc_1", 100, 500)
    assert h_default.depth is None

    # 2. Depth < 1 raises ValueError.
    with pytest.raises(ValueError, match="depth .* must be >= 1"):
        GangHolding("g1", "acc_1", 100, 500, depth=0)

    with pytest.raises(ValueError, match="depth .* must be >= 1"):
        GangHolding("g1", "acc_1", 100, 500, depth=-2)

    # 3. build_gang_holdings_from_roles copies role depth.
    roles = (
        RoleSpan("acc_mule_1", "mule", "g1", depth=2, valid_from=100, valid_to=600),
        RoleSpan("acc_victim", "victim", "g1", depth=None, valid_from=100, valid_to=None),
        RoleSpan("acc_collector", "collector", "g1", depth=None, valid_from=50, valid_to=1000),
    )
    holdings = build_gang_holdings_from_roles(roles)
    # Victim is excluded; mule and collector are preserved.
    assert len(holdings) == 2
    assert holdings[0].account_id == "acc_mule_1"
    assert holdings[0].depth == 2
    assert holdings[1].account_id == "acc_collector"
    assert holdings[1].depth is None


def test_choose_spare_account_skips_stopped_and_returns_none_when_exhausted() -> None:
    """
    WHAT: Tests pure candidate selection under choose_spare_account.
    WHY: Confirms rotation prioritizes the first non-stopped candidate in gang_holdings order.
    """
    state = GangReactionState(gang_id="g1", follows_stop_rule=True, follows_rotation_rule=True)
    candidates = ("acc_mule_a", "acc_mule_b", "acc_mule_c")

    # 1. When no accounts are stopped, returns first candidate in order.
    assert choose_spare_account(state, candidates) == "acc_mule_a"

    # 2. When first candidate is stopped, skips it and returns second.
    state.stopped_accounts.add("acc_mule_a")
    assert choose_spare_account(state, candidates) == "acc_mule_b"

    # 3. When both first and second are stopped, returns third.
    state.stopped_accounts.add("acc_mule_b")
    assert choose_spare_account(state, candidates) == "acc_mule_c"

    # 4. When all candidates are stopped, returns None.
    state.stopped_accounts.add("acc_mule_c")
    assert choose_spare_account(state, candidates) is None

    # 5. Empty candidate list returns None.
    assert choose_spare_account(state, ()) is None


def test_has_seen_trouble_flips_on_first_observation() -> None:
    """
    WHAT: Verifies has_seen_trouble flips from False to True on ANY delivered observation kind.
    WHY: Trouble telemetry signals operational friction, arming reactive splitting behavior.
    """
    # 1. State initializes with has_seen_trouble == False.
    state = GangReactionState(gang_id="g1", follows_stop_rule=True, follows_split_rule=True)
    assert not state.has_seen_trouble

    # 2. Observation for another gang is ignored and leaves has_seen_trouble False.
    other_obs = GangObservation("g_other", "acc_1", "payment_declined", 100, 150)
    update_gang_reaction(state, other_obs)
    assert not state.has_seen_trouble

    # 3. Any observation for this gang flips has_seen_trouble to True.
    declined_obs = GangObservation("g1", "acc_1", "payment_declined", 100, 150)
    update_gang_reaction(state, declined_obs)
    assert state.has_seen_trouble

    # 4. Lien observation also flips has_seen_trouble and stops the account.
    fresh_state = GangReactionState(gang_id="g2", follows_stop_rule=True)
    assert not fresh_state.has_seen_trouble
    lien_obs = GangObservation("g2", "acc_2", "lien_placed", 200, 260)
    update_gang_reaction(fresh_state, lien_obs)
    assert fresh_state.has_seen_trouble
    assert "acc_2" in fresh_state.stopped_accounts


def test_rotation_and_split_draw_functions() -> None:
    """
    WHAT: Tests draw functions for rotation flag, split flag, and whole-rupee split threshold.
    WHY: Verifies parameter resolution across splits and deterministic drawing mechanics.
    """
    rng = np.random.default_rng(12345)

    # 1. Rotation share range resolution and drawing.
    rot_range = resolve_rotation_share_range("training")
    assert 0.0 <= rot_range[0] <= rot_range[1] <= 1.0
    rot_flag = draw_rotation_flag("training", rng)
    assert isinstance(rot_flag, bool)

    # 2. Split share range resolution and drawing.
    split_range = resolve_split_share_range("training")
    assert 0.0 <= split_range[0] <= split_range[1] <= 1.0
    split_flag = draw_split_flag("training", rng)
    assert isinstance(split_flag, bool)

    # 3. Split threshold resolution and drawing in whole rupees converted to paise.
    thresh_range = resolve_split_threshold_range("training")
    assert 0 < thresh_range[0] <= thresh_range[1]
    thresh_paise = draw_split_threshold_paise("training", rng)
    assert thresh_paise > 0
    # Must be whole rupees converted to paise (divisible by 100).
    assert thresh_paise % 100 == 0
    assert thresh_range[0] * 100 <= thresh_paise <= thresh_range[1] * 100
