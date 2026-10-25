"""
WHAT: Unit tests for gang observation feed, holding records, and reactive decision rules.
Validates immutable representations of holdings and observations, parameter resolution
from empirical world ranges, feed settings drawing, holding lookup logic, role filtering,
and opening handle resolution.

WHY: Level 12d criminal syndicates act upon lossy, time-delayed observation feeds of external
friction touching their accounts, rather than inspecting ground truth or bank ledgers.
Rigorous unit tests verify isolation and correct mathematical behaviors across splits.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from world.gang_feed import (
    OBSERVATION_KINDS,
    GangFeedSettings,
    GangHolding,
    GangObservation,
    build_gang_holdings_from_roles,
    draw_gang_feed_settings,
    find_holding_gang,
    load_gang_reactions_config,
    resolve_feed_delay_range,
    resolve_feed_miss_share_range,
)
from world.gang_reactions import (
    GangReactionState,
    may_move_money,
    resolve_stop_rule_share_range,
    update_gang_reaction,
)
from world.interactive_records import FraudInputs
from world.opening_handles import resolve_opening_handles
from world.outside_events import OutsideEvent
from world.truth_records import RoleSpan


def test_gang_holding_record_immutability_and_coverage() -> None:
    """Verifies that GangHolding records are immutable and evaluate half-open temporal coverage correctly."""
    holding = GangHolding(gang_id="gang_alpha", account_id="acc_person_000001", valid_from=100, valid_to=200)
    assert holding.gang_id == "gang_alpha"
    assert holding.account_id == "acc_person_000001"
    assert holding.valid_from == 100
    assert holding.valid_to == 200

    # Test half-open temporal interval coverage: valid_from <= minute < valid_to.
    assert not holding.covers(99)
    assert holding.covers(100)
    assert holding.covers(150)
    assert not holding.covers(200)
    assert not holding.covers(250)

    # Test unending validity span where valid_to is None.
    unending = GangHolding(gang_id="gang_beta", account_id="acc_person_000002", valid_from=300, valid_to=None)
    assert not unending.covers(299)
    assert unending.covers(300)
    assert unending.covers(1000000)

    # Immutability check: modifying attributes must raise FrozenInstanceError.
    with pytest.raises(FrozenInstanceError):
        holding.account_id = "acc_person_000099"  # type: ignore[misc]


def test_gang_observation_validation_and_timing_invariants() -> None:
    """Asserts that GangObservation rejects invalid kinds and prevents seen minutes preceding happened minutes."""
    # Verify instantiation with authorized observation kinds.
    for kind in OBSERVATION_KINDS:
        obs = GangObservation(
            gang_id="gang_alpha",
            account_id="acc_person_000001",
            kind=kind,
            happened_minute=100,
            seen_minute=120,
        )
        assert obs.kind == kind

    # Reject uncatalogued observation categories.
    with pytest.raises(ValueError, match="is invalid; must be one of"):
        GangObservation(
            gang_id="gang_alpha",
            account_id="acc_person_000001",
            kind="fraud_detected",
            happened_minute=100,
            seen_minute=120,
        )

    # Reject impossible temporal ordering where an observation is seen before it occurs.
    with pytest.raises(ValueError, match="seen_minute.*cannot precede happened_minute"):
        GangObservation(
            gang_id="gang_alpha",
            account_id="acc_person_000001",
            kind="lien_placed",
            happened_minute=150,
            seen_minute=140,
        )


def test_gang_reactions_config_loads_and_has_required_metadata() -> None:
    """Validates that config/gang_reactions.yaml loads properly and contains policy metadata."""
    cfg = load_gang_reactions_config()
    assert "gang_reactions" in cfg
    sub = cfg["gang_reactions"]
    assert sub["label"] == "policy"
    assert sub["derived_from"] == "amendment A-017"
    assert "G4" in sub["issue_ids_v2"]
    assert "world_ranges_reference" in sub["gang_feed_delay_minutes"]
    assert "world_ranges_reference" in sub["gang_feed_miss_share"]
    assert "world_ranges_reference" in sub["gang_stops_using_liened_account_share"]


def test_resolve_feed_parameter_ranges_across_splits_and_fixtures() -> None:
    """Ensures parameter bounds resolve accurately across dataset splits and support fixture overrides."""
    # Dereference bounds from config/world_ranges.yaml.
    delay_train = resolve_feed_delay_range("training")
    assert isinstance(delay_train, tuple)
    assert len(delay_train) == 2
    assert delay_train[0] <= delay_train[1]

    miss_train = resolve_feed_miss_share_range("training")
    assert isinstance(miss_train, tuple)
    assert 0.0 <= miss_train[0] <= miss_train[1] <= 1.0

    stop_train = resolve_stop_rule_share_range("training")
    assert isinstance(stop_train, tuple)
    assert 0.0 <= stop_train[0] <= stop_train[1] <= 1.0

    # Verify fixture dictionary overrides for delay, miss share, and stop rule.
    fixture_config = {
        "parameters": {
            "gang_feed_delay_minutes": {"training": [45, 90]},
            "gang_feed_miss_share": {"training": [0.10, 0.25]},
            "gang_stops_using_liened_account_share": {"training": [0.75, 0.85]},
        }
    }
    assert resolve_feed_delay_range("training", fixture_config) == (45, 90)
    assert resolve_feed_miss_share_range("training", fixture_config) == (0.10, 0.25)
    assert resolve_stop_rule_share_range("training", fixture_config) == (0.75, 0.85)


def test_draw_gang_feed_settings_produces_consistent_parameters() -> None:
    """Verifies that draw_gang_feed_settings produces integer delay minutes and float miss shares in bounds."""
    rng = np.random.default_rng(12345)
    fixture_config = {
        "parameters": {
            "gang_feed_delay_minutes": {"training": [30, 60]},
            "gang_feed_miss_share": {"training": [0.05, 0.20]},
        }
    }
    for _ in range(20):
        settings = draw_gang_feed_settings("training", rng, fixture_config)
        assert isinstance(settings, GangFeedSettings)
        assert isinstance(settings.delay_minutes, int)
        assert 30 <= settings.delay_minutes <= 60
        assert isinstance(settings.miss_share, float)
        assert 0.05 <= settings.miss_share <= 0.20


def test_find_holding_gang_locates_controlling_syndicate() -> None:
    """Confirms that find_holding_gang identifies the operational controller during active temporal spans."""
    holdings = (
        GangHolding("gang_1", "acc_person_000001", 100, 300),
        GangHolding("gang_2", "acc_person_000002", 200, 400),
    )
    # Account 1 held by gang_1 between 100 and 300.
    assert find_holding_gang(holdings, "acc_person_000001", 99) is None
    assert find_holding_gang(holdings, "acc_person_000001", 100) == "gang_1"
    assert find_holding_gang(holdings, "acc_person_000001", 299) == "gang_1"
    assert find_holding_gang(holdings, "acc_person_000001", 300) is None

    # Account 2 held by gang_2 between 200 and 400.
    assert find_holding_gang(holdings, "acc_person_000002", 250) == "gang_2"

    # Non-existent account returns None.
    assert find_holding_gang(holdings, "acc_person_000999", 250) is None


def test_build_gang_holdings_from_roles_excludes_victims() -> None:
    """Ensures that build_gang_holdings_from_roles retains mules, collectors, and cash-outs while dropping victims."""
    roles = (
        RoleSpan(
            account_id="acc_victim", role="victim", gang_id="gang_alpha",
            depth=None, valid_from=0, valid_to=None,
        ),
        RoleSpan(
            account_id="acc_mule", role="mule", gang_id="gang_alpha",
            depth=1, valid_from=100, valid_to=500, origin="bought_account",
        ),
        RoleSpan(
            account_id="acc_collector", role="collector", gang_id="gang_alpha",
            depth=None, valid_from=200, valid_to=600,
        ),
        RoleSpan(
            account_id="acc_cash_out", role="cash_out", gang_id="gang_alpha",
            depth=None, valid_from=300, valid_to=700,
        ),
    )
    holdings = build_gang_holdings_from_roles(roles)
    # Only mule, collector, and cash_out roles should produce holdings.
    assert len(holdings) == 3
    account_ids = {h.account_id for h in holdings}
    assert "acc_victim" not in account_ids
    assert account_ids == {"acc_mule", "acc_collector", "acc_cash_out"}


def test_resolve_opening_handles_updates_gang_holdings() -> None:
    """
    Verifies that resolve_opening_handles rewrites opening handles to assigned account identifiers in gang holdings.
    """
    handle = "new:gang_test:0"
    inputs = FraudInputs(
        outside_events=(
            OutsideEvent(
                minute=100,
                kind="open_account",
                owner_kind="person",
                opening_stream="gang_account_openings",
                opening_handle=handle,
            ),
        ),
        gang_holdings=(
            GangHolding(gang_id="gang_test", account_id=handle, valid_from=100, valid_to=500),
        ),
    )

    resolved, assigned = resolve_opening_handles(inputs, 60, 12)
    assert assigned == {handle: "acc_person_000060"}
    assert len(resolved.gang_holdings) == 1
    assert resolved.gang_holdings[0].account_id == "acc_person_000060"
    assert resolved.gang_holdings[0].gang_id == "gang_test"
    assert resolved.gang_holdings[0].valid_from == 100
    assert resolved.gang_holdings[0].valid_to == 500


def test_gang_reaction_state_and_stop_rule_behavior() -> None:
    """Checks that update_gang_reaction marks liened accounts stopped only when following the stop rule."""
    state_adherent = GangReactionState(gang_id="gang_1", follows_stop_rule=True)
    state_non_adherent = GangReactionState(gang_id="gang_2", follows_stop_rule=False)

    obs_lien_1 = GangObservation("gang_1", "acc_mule_1", "lien_placed", 100, 130)
    obs_lien_2 = GangObservation("gang_2", "acc_mule_2", "lien_placed", 100, 130)
    obs_hold_1 = GangObservation("gang_1", "acc_mule_3", "debit_hold_placed", 100, 130)
    obs_foreign = GangObservation("gang_foreign", "acc_mule_4", "lien_placed", 100, 130)

    # Foreign telemetry ignored.
    update_gang_reaction(state_adherent, obs_foreign)
    assert len(state_adherent.stopped_accounts) == 0

    # Non-lien observations do not trigger the stop rule.
    update_gang_reaction(state_adherent, obs_hold_1)
    assert len(state_adherent.stopped_accounts) == 0
    assert may_move_money(state_adherent, "acc_mule_3")

    # Lien placed on adherent gang stops the account.
    update_gang_reaction(state_adherent, obs_lien_1)
    assert "acc_mule_1" in state_adherent.stopped_accounts
    assert not may_move_money(state_adherent, "acc_mule_1")

    # Non-adherent gang does not stop the account even when observing a lien.
    update_gang_reaction(state_non_adherent, obs_lien_2)
    assert len(state_non_adherent.stopped_accounts) == 0
    assert may_move_money(state_non_adherent, "acc_mule_2")
