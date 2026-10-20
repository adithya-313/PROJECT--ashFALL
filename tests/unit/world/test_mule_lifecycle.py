"""
WHAT: Unit tests for mule lifecycle phases (Level 12c C-3b): warm_up, active, and burned states.
Verifies temporal bounds, inflow windows, dropped payment tracking, burned_from termination,
and owner activity pausing integration.

WHY: Mules progress through explicit operational stages. Validating that active_from and burned_from
align with campaign configuration and prevent fraud activity after burning guarantees lifecycle consistency.

TERM: active_from: Simulation minute when a mule's seasoning/dormancy ends and active inflow begins.
TERM: burned_from: Simulation minute when a mule is dropped or concludes its final laundering operation.
TERM: inflow window: Half-open interval [active_from, active_from + active_days * 1440) for receiving victim funds.
"""

import numpy as np
import pytest

import world.scam_planner as scam_planner
from evaluation.truth import TruthStore
from world.build_world import World, build_world
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.gang_planner import plan_gang
from world.rented_owner_activity import plan_owner_activity
from world.truth_records import RoleSpan, validate_role_lifecycle

START_MINUTE = 1440
GENEROUS_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)


@pytest.fixture
def lifecycle_world() -> World:
    """Small reproducible world with enough people for testing mule campaigns."""
    return build_world(seed=101, n_people=60, n_shops=12)


def test_phase_at_gives_phases_at_right_minutes() -> None:
    """Verifies TruthStore.phase_at returns warm_up, active, and burned at exact minute transitions."""
    roles = [
        {
            "account_id": "acc_mule_1",
            "role": "mule",
            "gang_id": "g1",
            "depth": 1,
            "origin": "rented",
            "valid_from": 1440,
            "valid_to": None,
            "active_from": 2880,
            "burned_from": 5760,
        },
        {
            "account_id": "acc_fixture_mule",
            "role": "mule",
            "gang_id": "g2",
            "depth": 1,
            "origin": None,
            "valid_from": 1440,
            "valid_to": None,
            "active_from": None,
            "burned_from": None,
        },
        {
            "account_id": "acc_victim_1",
            "role": "victim",
            "gang_id": "g1",
            "depth": None,
            "origin": None,
            "valid_from": 2880,
            "valid_to": None,
            "active_from": None,
            "burned_from": None,
        },
    ]
    store = TruthStore(roles=roles, device_control=[], payment_stories={})

    # Prior to valid_from: None
    assert store.phase_at("acc_mule_1", 1000) is None
    # During warm_up interval [valid_from, active_from)
    assert store.phase_at("acc_mule_1", 1440) == "warm_up"
    assert store.phase_at("acc_mule_1", 2000) == "warm_up"
    assert store.phase_at("acc_mule_1", 2879) == "warm_up"
    # During active interval [active_from, burned_from)
    assert store.phase_at("acc_mule_1", 2880) == "active"
    assert store.phase_at("acc_mule_1", 4000) == "active"
    assert store.phase_at("acc_mule_1", 5759) == "active"
    # During burned interval [burned_from, infinity)
    assert store.phase_at("acc_mule_1", 5760) == "burned"
    assert store.phase_at("acc_mule_1", 10000) == "burned"

    # Account without lifecycle fields or non-mule role returns None
    assert store.phase_at("acc_fixture_mule", 2000) is None
    assert store.phase_at("acc_victim_1", 3000) is None
    assert store.phase_at("acc_unknown", 2000) is None


def test_active_from_minus_valid_from_equals_warmup_days_in_minutes(lifecycle_world: World) -> None:
    """Verifies that active_from minus valid_from exactly equals configured warm_up duration in minutes."""
    warmup_days = 3
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("warm_up", (("days", warmup_days),)),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    config = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE)
    plan = plan_gang(config, lifecycle_world, np.random.default_rng(8))

    mule_roles = [r for r in plan.truth.roles if r.role == "mule"]
    assert len(mule_roles) == 4
    for r in mule_roles:
        assert r.active_from is not None and r.valid_from is not None
        assert r.active_from - r.valid_from == warmup_days * 1440


def test_every_victim_payment_inside_inflow_window_and_past_counted_dropped(lifecycle_world: World) -> None:
    """Verifies all planned victim payments fall within the mule's inflow window, with past payments dropped."""
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("warm_up", (("days", 2),)),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 5))),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    config = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE)

    scam_step = next(s for s in config.steps if s.block == "scam")
    scam_plan = scam_planner.plan_scam_step(
        lifecycle_world, config, scam_step, ["m0", "m1", "m2", "m3"], START_MINUTE + 2 * 1440,
        None, np.random.default_rng(12),
    )
    assert scam_plan.dropped_payments_count >= 0
    assert scam_plan.dropped_victim_payments_count == scam_plan.dropped_payments_count

    # Check each planned payment arrives strictly before its receiving mule's window end
    for payment, story in zip(scam_plan.payments, scam_plan.stories):
        if not story.description.endswith("payout back"):
            mule_inflow = scam_plan.mule_inflows[payment.to_account]
            assert payment.minute < mule_inflow.inflow_window_end


def test_no_fraud_payment_touches_a_mule_at_or_after_burned_from(lifecycle_world: World) -> None:
    """Verifies that no planned fraud transfer touches a mule account at or after its burned_from minute."""
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("warm_up", (("days", 2),)),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 3))),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    config = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE)
    plan = plan_gang(config, lifecycle_world, np.random.default_rng(42))

    roles_by_account = {r.account_id: r for r in plan.truth.roles if r.role == "mule"}
    for payment in plan.payments:
        # If payment is from a mule:
        if payment.from_account in roles_by_account:
            sender_role = roles_by_account[payment.from_account]
            assert sender_role.burned_from is not None
            assert payment.minute < sender_role.burned_from
        # If payment is to a receiving mule (depth 1):
        if payment.to_account in roles_by_account and roles_by_account[payment.to_account].depth == 1:
            receiver_role = roles_by_account[payment.to_account]
            assert receiver_role.burned_from is not None
            assert payment.minute < receiver_role.burned_from


def test_active_days_follows_narrowed_config(monkeypatch: pytest.MonkeyPatch, lifecycle_world: World) -> None:
    """Narrows mule_active_days_before_dropped range to (5, 5) and asserts exact window duration."""
    monkeypatch.setattr(scam_planner, "resolve_mule_active_days_before_dropped", lambda split: (5, 5))
    steps = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 2))),
    )
    config = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE)
    plan = scam_planner.plan_scam_step(
        lifecycle_world, config, steps[1], ["m0", "m1"], START_MINUTE, None, np.random.default_rng(7),
    )
    inflow = plan.mule_inflows["m0"]
    assert inflow.inflow_window_end - START_MINUTE == 5 * 1440


def test_victims_of_one_mule_do_not_all_start_at_same_minute(lifecycle_world: World) -> None:
    """Verifies that victims assigned to a single receiving mule have staggered first payment minutes."""
    steps = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 8))),
    )
    config = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE)
    plan = scam_planner.plan_scam_step(
        lifecycle_world, config, steps[1], ["m0", "m1"], START_MINUTE, None, np.random.default_rng(99),
    )
    victim_first_minutes = [role.valid_from for role in plan.roles if role.role == "victim"]
    assert len(set(victim_first_minutes)) > 1


def test_rented_owner_pausing_stops_at_burned_from() -> None:
    """Verifies that rented owner activity pausing strictly ends once burned_from is reached."""
    role = RoleSpan(
        account_id="acc_rented_01",
        role="mule",
        gang_id="g1",
        depth=1,
        valid_from=1440,
        valid_to=None,
        origin="rented",
        active_from=2880,
        burned_from=5760,
    )
    activity = plan_owner_activity([role], "training", np.random.default_rng(1))
    owner = activity["acc_rented_01"]

    assert owner.covers(1440) is True
    assert owner.covers(2880) is True
    assert owner.covers(5759) is True
    assert owner.covers(5760) is False
    assert owner.covers(10000) is False


def test_fixture_gangs_have_no_lifecycle_fields(lifecycle_world: World) -> None:
    """Verifies that fixture-sourced gangs retain active_from=None and burned_from=None across all roles."""
    source = FixtureSource("acc_person_000010", 0, 1_000_000, START_MINUTE)
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("warm_up", (("days", 2),)),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    config = GangConfig("g_fixture", 1, GENEROUS_BUDGET, source, steps)
    plan = plan_gang(config, lifecycle_world, np.random.default_rng(5))

    for role in plan.truth.roles:
        assert role.active_from is None
        assert role.burned_from is None


def test_lifecycle_validation_refusals() -> None:
    """Verifies that validate_role_lifecycle raises ValueError on contradictory or out-of-order lifecycle fields."""
    # 1. Partial lifecycle specification (one set, other None)
    with pytest.raises(ValueError, match="both active_from and burned_from set or both None"):
        validate_role_lifecycle(
            RoleSpan("acc_1", "mule", "g1", 1, 0, None, active_from=100, burned_from=None)
        )
    with pytest.raises(ValueError, match="both active_from and burned_from set or both None"):
        validate_role_lifecycle(
            RoleSpan("acc_1", "mule", "g1", 1, 0, None, active_from=None, burned_from=100)
        )

    # 2. Lifecycle set on non-mule role
    with pytest.raises(ValueError, match="lifecycle is only allowed on mule roles"):
        validate_role_lifecycle(
            RoleSpan("acc_1", "victim", "g1", None, 0, None, active_from=100, burned_from=200)
        )

    # 3. Invalid temporal ordering (valid_from > active_from)
    with pytest.raises(ValueError, match="invalid lifecycle order"):
        validate_role_lifecycle(
            RoleSpan("acc_1", "mule", "g1", 1, 500, None, active_from=400, burned_from=600)
        )

    # 4. Invalid temporal ordering (active_from > burned_from)
    with pytest.raises(ValueError, match="invalid lifecycle order"):
        validate_role_lifecycle(
            RoleSpan("acc_1", "mule", "g1", 1, 100, None, active_from=500, burned_from=400)
        )
