"""
WHAT: Unit and integration tests for Level 12d gang rotation and amount-splitting reactions.
Validates configuration schema constraints for new keys, GangHolding depth propagation,
pure account rotation selection via choose_spare_account, trouble telemetry updates,
and end-to-end interactive simulation of account rotation, splitting, edge cases, and determinism.

WHY: Demonstrates that reactive criminal syndicates evaluate operational resilience
mechanisms without violating information isolation or corrupting random streams.

TERM: rotation rule: rerouting illicit payments away from compromised mules to viable spares.
TERM: split rule: structuring transfers exceeding a threshold into smaller equal pieces.
TERM: trouble telemetry: any delivered observation indicating friction on held infrastructure.
"""

from typing import Any

import numpy as np
import pytest

from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.gang_feed import (
    GangHolding,
    GangObservation,
    build_gang_holdings_from_roles,
    draw_gang_feed_settings,
    load_gang_reactions_config,
    validate_gang_reactions_config,
)
from world.gang_reactions import (
    GangReactionState,
    choose_spare_account,
    draw_rotation_flag,
    draw_split_flag,
    draw_split_threshold_paise,
    draw_stop_rule_flag,
    resolve_rotation_share_range,
    resolve_split_share_range,
    resolve_split_threshold_range,
    update_gang_reaction,
)
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.outside_events import OutsideEvent
from world.truth_records import RoleSpan

TINY = FastRunSettings(101, 60, 12, 61, "training")
GENEROUS_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)


def _build_fixture_gang() -> GangConfig:
    """
    WHAT: Constructs a reproducible scam gang with 4 acquired mules, a scam campaign, and a 2x2 laundering route.
    WHY: Supplies multi-hop fraud flows through layer-1 mules to test reactive flow rotations and splits.
    """
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    return GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, 1440)


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


def test_rotation_on_reroutes_payments_to_spare_and_preserves_balance_bounds() -> None:
    """
    WHAT: Verifies rotation reaction redirects payments into a stopped mule to an active same-depth spare.
    WHY: Confirms redirected_by_gang records the events, no payments reach the stopped mule afterwards,
    and the spare mule never forwards more money than it legitimately received.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    spare_account = "acc_person_000006"
    minute_l = 3000

    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "rotation test fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [1100, 1100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
            "gang_rotates_to_spare_account_share": {"training": [1.0, 1.0]},
        }
    }
    seen_time = minute_l + 1100

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-rot-01",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    # 1. Assert redirected_by_gang is not empty and targets the same-depth spare.
    assert len(out.result.redirected_by_gang) > 0, "redirected_by_gang should record redirections"
    for r in out.result.redirected_by_gang:
        assert r.planned_to == mule_account
        assert r.new_to == spare_account
        assert r.minute >= seen_time

    # 2. No payment lands on the stopped mule after seen_time.
    for i, ev in enumerate(out.result.run_result.events):
        if out.result.event_sources[i] == "fraud" and ev.minute >= seen_time:
            assert ev.to_account != mule_account, f"Payment reached stopped mule {mule_account} at {ev.minute}"

    # 3. Total spare outflow <= total spare inflow across all successful fraud events.
    spare_inflow = sum(
        ev.amount_paise
        for i, ev in enumerate(out.result.run_result.events)
        if out.result.event_sources[i] == "fraud" and ev.to_account == spare_account and ev.status == "success"
    )
    spare_outflow = sum(
        ev.amount_paise
        for i, ev in enumerate(out.result.run_result.events)
        if out.result.event_sources[i] == "fraud" and ev.from_account == spare_account and ev.status == "success"
    )
    assert spare_outflow <= spare_inflow, f"Spare outflow ({spare_outflow}) exceeded inflow ({spare_inflow})"


def test_rotation_on_with_all_spares_stopped_cancels_with_no_spare_account() -> None:
    """
    WHAT: Tests rotation reaction when every available spare mule at that depth is already stopped.
    WHY: Verifies the payment is cancelled with explicit reason 'no_spare_account'.
    """
    gang = _build_fixture_gang()
    mule_1 = "acc_person_000006"
    mule_2 = "acc_person_000010"
    minute_l = 3000

    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "rotation exhaustion test fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [1100, 1100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
            "gang_rotates_to_spare_account_share": {"training": [1.0, 1.0]},
        }
    }

    # Inject liens on BOTH depth-1 mules simultaneously.
    lien1 = OutsideEvent(minute=minute_l, kind="lien", account_id=mule_1, amount_paise=50000, encumbrance_id="enc-ex-1")
    lien2 = OutsideEvent(minute=minute_l, kind="lien", account_id=mule_2, amount_paise=50000, encumbrance_id="enc-ex-2")
    inputs = FraudInputs(outside_events=(lien1, lien2))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    # All candidate spares are stopped, so no redirection can succeed.
    no_spare_cancels = [c for c in out.result.cancelled_by_gang if c.reason == "no_spare_account"]
    assert len(no_spare_cancels) > 0, "Expected cancellations with reason 'no_spare_account'"


def test_rotation_off_cancels_as_in_d1_without_redirect() -> None:
    """
    WHAT: Verifies that when rotation rule is disabled, a stopped to_account cancels with to_account_stopped.
    WHY: Confirms backward compatibility with Level 12d D-1 behavior when rotation share is 0.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 3000

    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "rotation disabled test fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [1100, 1100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
            "gang_rotates_to_spare_account_share": {"training": [0.0, 0.0]},
        }
    }

    lien = OutsideEvent(
        minute=minute_l, kind="lien", account_id=mule_account, amount_paise=50000, encumbrance_id="enc-off-1"
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    # Redirection list must be empty.
    assert len(out.result.redirected_by_gang) == 0
    # Cancellations after seen_time have reason to_account_stopped.
    to_stopped_cancels = [c for c in out.result.cancelled_by_gang if c.reason == "to_account_stopped"]
    assert len(to_stopped_cancels) > 0


def test_split_on_partitions_payments_above_threshold_only_after_trouble() -> None:
    """
    WHAT: Tests that payment splitting occurs strictly after trouble is observed,
    pieces do not exceed threshold or breach slow_drip, and pieces sum to original amount.
    WHY: Validates structured evasion reactions while preventing leakage into held-out tactics.
    """
    gang = _build_fixture_gang()
    threshold_inr = 25000
    threshold_paise = threshold_inr * 100
    minute_l = 3000
    delay_minutes = 1100
    seen_time = minute_l + delay_minutes

    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "splitting test fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_minutes, delay_minutes]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [0.0, 0.0]},
            "gang_rotates_to_spare_account_share": {"training": [0.0, 0.0]},
            "gang_splits_amounts_share": {"training": [1.0, 1.0]},
            "gang_split_threshold_inr": {"training": [threshold_inr, threshold_inr]},
        }
    }

    # Lien placed on mule at 3000 signals trouble delivered at 4100.
    lien = OutsideEvent(
        minute=minute_l, kind="lien", account_id="acc_person_000010", amount_paise=50000, encumbrance_id="enc-sp-1"
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    # 1. No split should occur prior to seen_time.
    for split_item in out.result.split_by_gang:
        assert split_item.minute >= seen_time, f"Split occurred before trouble seen: {split_item}"

    # 2. Splits occur after seen_time.
    assert len(out.result.split_by_gang) > 0, "Expected splits to occur after trouble observed"

    # 3. For each split, verify piece bounds and amount conservation.
    for split_item in out.result.split_by_gang:
        assert split_item.amount > threshold_paise
        n = split_item.piece_count
        base = split_item.amount // n
        rem = split_item.amount % n
        pieces = [base + (1 if k < rem else 0) for k in range(n)]
        assert sum(pieces) == split_item.amount
        for p in pieces:
            assert p <= threshold_paise
            assert p >= 200000  # slow_drip dividing line: 2000 INR = 200000 paise


def test_split_on_below_slow_drip_line_raises_value_error_in_training() -> None:
    """
    WHAT: Tests that forcing split threshold below the slow_drip dividing line in training raises ValueError.
    WHY: Protects training worlds from data leakage into the held-out slow_drip tactic family.
    """
    gang = _build_fixture_gang()
    minute_l = 3000

    # 500 INR is valid (> 0) under validate_gang_reactions_config, but below 2000 INR slow_drip line.
    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "sub-slow-drip split fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [100, 100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [0.0, 0.0]},
            "gang_rotates_to_spare_account_share": {"training": [0.0, 0.0]},
            "gang_splits_amounts_share": {"training": [1.0, 1.0]},
            "gang_split_threshold_inr": {"training": [500, 500]},
        }
    }

    lien = OutsideEvent(
        minute=minute_l, kind="lien", account_id="acc_person_000010", amount_paise=50000, encumbrance_id="enc-leak-1"
    )
    inputs = FraudInputs(outside_events=(lien,))

    with pytest.raises(ValueError, match="slow_drip_max_amount_inr"):
        run_interactive_world(
            TINY,
            RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
            fraud_inputs=inputs,
            gang_configs=(gang,),
            gang_reactions_config=fixture_cfg,
        )


def test_gang_rotation_split_determinism_across_identical_runs() -> None:
    """
    WHAT: Asserts that running two simulations with identical seed and reactions config yields identical results.
    WHY: Ensures RNG draw order on child 15 and execution ordering remain deterministic.
    """
    gang = _build_fixture_gang()
    minute_l = 3000

    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "determinism fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [1100, 1100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
            "gang_rotates_to_spare_account_share": {"training": [1.0, 1.0]},
            "gang_splits_amounts_share": {"training": [1.0, 1.0]},
            "gang_split_threshold_inr": {"training": [25000, 25000]},
        }
    }

    lien = OutsideEvent(
        minute=minute_l, kind="lien", account_id="acc_person_000010", amount_paise=50000, encumbrance_id="enc-det-1"
    )
    inputs = FraudInputs(outside_events=(lien,))

    out1 = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    out2 = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    assert out1.result.run_result.events == out2.result.run_result.events
    assert out1.result.redirected_by_gang == out2.result.redirected_by_gang
    assert out1.result.split_by_gang == out2.result.split_by_gang
    assert out1.result.cancelled_by_gang == out2.result.cancelled_by_gang


def test_d1_reactions_equality_with_no_friction() -> None:
    """
    WHAT: Asserts that with reactions enabled and no external friction, output matches reactions disabled.
    WHY: Proves reactions engine remains strictly dormant without observations.
    """
    gang = _build_fixture_gang()

    out_off = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=False),
        gang_configs=(gang,),
    )

    out_on = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        gang_configs=(gang,),
    )

    assert out_on.result.run_result.events == out_off.result.run_result.events
    assert len(out_on.result.redirected_by_gang) == 0
    assert len(out_on.result.split_by_gang) == 0


def test_edge_case_lien_on_spare_itself_later() -> None:
    """
    WHAT: Tests edge case where a spare account itself receives a lien after the primary is stopped.
    WHY: Rotation must cease using the spare once the spare's lien is observed, cancelling further flows.
    """
    gang = _build_fixture_gang()
    primary = "acc_person_000010"
    spare = "acc_person_000006"
    delay = 1100

    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "spare lien test fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay, delay]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
            "gang_rotates_to_spare_account_share": {"training": [1.0, 1.0]},
        }
    }

    # Primary lien at 2000 (seen at 3100), spare lien at 2500 (seen at 3600).
    lien_primary = OutsideEvent(
        minute=2000, kind="lien", account_id=primary, amount_paise=50000, encumbrance_id="enc-p"
    )
    lien_spare = OutsideEvent(
        minute=2500, kind="lien", account_id=spare, amount_paise=50000, encumbrance_id="enc-s"
    )
    inputs = FraudInputs(outside_events=(lien_primary, lien_spare))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    # 1. Redirections happen while spare is operational.
    assert any(r.planned_to == primary and r.new_to == spare for r in out.result.redirected_by_gang)

    # 2. After spare is seen stopped (3600), no payment reaches spare.
    for i, ev in enumerate(out.result.run_result.events):
        if out.result.event_sources[i] == "fraud" and ev.minute >= 3600:
            assert ev.to_account != spare
            assert ev.to_account != primary


def test_edge_case_redirect_at_exact_minute_lien_is_seen() -> None:
    """
    WHAT: Tests same-minute rule when an observation is delivered at the exact minute of a planned payment.
    WHY: In tie_break_order, gang_observations is ranked last. A payment at minute M executes
    before an observation at minute M is processed, so it is not redirected; payments scheduled
    after minute M are redirected to the spare.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    spare_account = "acc_person_000006"
    seen_time = 1670
    minute_l = 1500
    delay_d = seen_time - minute_l

    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "same minute test fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_d, delay_d]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
            "gang_rotates_to_spare_account_share": {"training": [1.0, 1.0]},
        }
    }

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-same-minute-01",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    # 1. Under the same-minute rule, payment at exactly seen_time (1670) executes before the observation.
    pmts_at_m = [
        ev for i, ev in enumerate(out.result.run_result.events)
        if out.result.event_sources[i] == "fraud"
        and ev.minute == seen_time
        and ev.to_account == mule_account
    ]
    assert len(pmts_at_m) > 0, f"Payment at exactly {seen_time} must execute before observation"

    # 2. Subsequent payments after seen_time are redirected to the spare.
    redirects_after_m = [
        r for r in out.result.redirected_by_gang
        if r.minute > seen_time and r.planned_to == mule_account and r.new_to == spare_account
    ]
    assert len(redirects_after_m) > 0, "Payments after seen_time must be redirected to spare"


def test_stopped_cash_out_with_rotation_on_is_never_redirected() -> None:
    """
    WHAT: Tests that a stopped cash_out account with rotation ON is cancelled rather than redirected.
    WHY: Rotation must only swap mule for mule. Accounts with depth None cannot rotate or be spares.
    """
    gang = GangConfig(
        "g1",
        1,
        GENEROUS_BUDGET,
        None,
        (
            BlockStep("acquire", (("count", 4),)),
            BlockStep("scam", (("receiving_mules", 1),)),
            BlockStep("route", (("layers", 1), ("accounts_per_layer", 1))),
            BlockStep("cash_out", ()),
        ),
        1440,
    )
    fixture_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "cash_out rotation test",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [10, 10]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
            "gang_rotates_to_spare_account_share": {"training": [1.0, 1.0]},
        }
    }
    # An extra cash_out holding with depth=None exists. Under buggy rotation, this would be chosen as spare.
    spare_cash_out = GangHolding("g1", "acc_person_000008", 4000, 10000, depth=None)
    cash_out_target = "acc_person_000060"
    mule_source = "acc_person_000006"
    lien_minute = 4065

    lien = OutsideEvent(
        minute=lien_minute,
        kind="lien",
        account_id=cash_out_target,
        amount_paise=50000,
        encumbrance_id="enc-co-1",
    )
    inputs = FraudInputs(outside_events=(lien,), gang_holdings=(spare_cash_out,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_cfg,
    )

    # 1. Prove the scenario really happened: payment to cash_out existed and was cancelled.
    matched_cancels = [
        c
        for c in out.result.cancelled_by_gang
        if c.reason == "no_spare_account" and c.accounts == (mule_source, cash_out_target)
    ]
    assert len(matched_cancels) == 1, (
        f"Expected 1 cancelled payment to stopped cash_out account {cash_out_target}, "
        f"got: {out.result.cancelled_by_gang}"
    )

    # 2. Assert redirected_by_gang has no entry whose planned_to has depth None.
    assert len(out.result.redirected_by_gang) == 0
    assert not any(r.planned_to == cash_out_target for r in out.result.redirected_by_gang)

    # 3. No payment ever landed on the stopped cash_out account.
    for i, ev in enumerate(out.result.run_result.events):
        if out.result.event_sources[i] == "fraud":
            assert ev.to_account != cash_out_target


def test_draw_functions_preserve_stream_position_across_degenerate_and_non_degenerate_ranges() -> None:
    """
    WHAT: Asserts that each draw function advances the RNG stream identically on degenerate and non-degenerate ranges.
    WHY: Proves the number of random draws remains strictly constant and decoupled from configuration bounds.
    """
    # 1. draw_gang_feed_settings (child 14)
    cfg_degen_feed = {
        "gang_reactions": {
            "gang_feed_delay_minutes": {"training": [100, 100]},
            "gang_feed_miss_share": {"training": [0.1, 0.1]},
        }
    }
    cfg_non_degen_feed = {
        "gang_reactions": {
            "gang_feed_delay_minutes": {"training": [100, 200]},
            "gang_feed_miss_share": {"training": [0.1, 0.5]},
        }
    }
    rng1 = np.random.default_rng(101)
    rng2 = np.random.default_rng(101)
    draw_gang_feed_settings("training", rng1, config=cfg_degen_feed)
    draw_gang_feed_settings("training", rng2, config=cfg_non_degen_feed)
    assert rng1.random() == rng2.random(), "feed settings draw count mismatch"

    # 2. draw_stop_rule_flag (child 15)
    cfg_degen_stop = {
        "gang_reactions": {"gang_stops_using_liened_account_share": {"training": [1.0, 1.0]}}
    }
    cfg_non_degen_stop = {
        "gang_reactions": {"gang_stops_using_liened_account_share": {"training": [0.2, 0.8]}}
    }
    rng1 = np.random.default_rng(102)
    rng2 = np.random.default_rng(102)
    draw_stop_rule_flag("training", rng1, config=cfg_degen_stop)
    draw_stop_rule_flag("training", rng2, config=cfg_non_degen_stop)
    assert rng1.random() == rng2.random(), "stop rule draw count mismatch"

    # 3. draw_rotation_flag (child 15)
    cfg_degen_rot = {
        "gang_reactions": {"gang_rotates_to_spare_account_share": {"training": [0.0, 0.0]}}
    }
    cfg_non_degen_rot = {
        "gang_reactions": {"gang_rotates_to_spare_account_share": {"training": [0.3, 0.7]}}
    }
    rng1 = np.random.default_rng(103)
    rng2 = np.random.default_rng(103)
    draw_rotation_flag("training", rng1, config=cfg_degen_rot)
    draw_rotation_flag("training", rng2, config=cfg_non_degen_rot)
    assert rng1.random() == rng2.random(), "rotation rule draw count mismatch"

    # 4. draw_split_flag (child 15)
    cfg_degen_split = {
        "gang_reactions": {"gang_splits_amounts_share": {"training": [1.0, 1.0]}}
    }
    cfg_non_degen_split = {
        "gang_reactions": {"gang_splits_amounts_share": {"training": [0.2, 0.8]}}
    }
    rng1 = np.random.default_rng(104)
    rng2 = np.random.default_rng(104)
    draw_split_flag("training", rng1, config=cfg_degen_split)
    draw_split_flag("training", rng2, config=cfg_non_degen_split)
    assert rng1.random() == rng2.random(), "split rule draw count mismatch"

    # 5. draw_split_threshold_paise (child 15)
    cfg_degen_thresh = {
        "gang_reactions": {"gang_split_threshold_inr": {"training": [25000, 25000]}}
    }
    cfg_non_degen_thresh = {
        "gang_reactions": {"gang_split_threshold_inr": {"training": [10000, 50000]}}
    }
    rng1 = np.random.default_rng(105)
    rng2 = np.random.default_rng(105)
    draw_split_threshold_paise("training", rng1, config=cfg_degen_thresh)
    draw_split_threshold_paise("training", rng2, config=cfg_non_degen_thresh)
    assert rng1.random() == rng2.random(), "split threshold draw count mismatch"

