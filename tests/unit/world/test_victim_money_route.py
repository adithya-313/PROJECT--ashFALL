"""
Tests that a gang's route forwards victim money and nothing else (Level 12c C-2): the planned route, then whole
worlds run through the interactive runner. Covers the forwarding timeline, a bank outage that stops a victim
payment, the payout back to a victim, and the hidden truth the scam gang writes.

TERM: receiving mule: an acquired mule account that victims pay; with a route it is the layer 1 mule.
TERM: victim money: illicit money that reached a mule through a successful victim payment.
TERM: payout back: a small payment from a mule back to a victim, funded only from illicit money the mule holds.
TERM: bank downtime: a window in which payments of one bank fail with some probability.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from showcase.level_12_support_b import get_showcase_b_settings
from world.build_world import World, build_world
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.gang_planner import _execute_victim_route_step, plan_gang
from world.interactive_records import RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world

BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
START_MINUTE = 1440

# Measured in the seed-101 world: bank 1 is down from minute 9145 for 82 minutes with success rate 0.3079. With this
# gang started at minute 8724, the only victim, acc_person_000033 of bank 1, makes one payment at minute 9145 and it
# fails.
OUTAGE_START_MINUTE = 9145
OUTAGE_GANG_START_MINUTE = 8724
OUTAGE_BANK_INDEX = 1
OUTAGE_ACQUIRE_COUNT = 8
OUTAGE_VICTIM = "acc_person_000033"
OUTAGE_MULE = "acc_person_000057"
OUTAGE_PLANNED_PAISE = 2785150
OUTAGE_STATUS = "failed"
OUTAGE_REASON = "bank_downtime"
OUTAGE_DOMAIN = "sender_bank"


@dataclass(frozen=True)
class FraudEvent:
    """One event of a fraud payment together with the story block and description of the payment it belongs to."""

    event: Any
    block: str
    description: str
    layer: int | None


def make_gang(
    acquire_count: int, receiving_mules: int, victims: int | None, layers: int, route_params: tuple[Any, ...] = (),
    start_minute: int = START_MINUTE, strength: int = 1, with_disguise: bool = False,
) -> GangConfig:
    """Builds a scam gang that acquires mules, scams victims into receiving_mules mules, and routes through layers."""
    scam_params: tuple[tuple[str, Any], ...] = (("receiving_mules", receiving_mules),)
    if victims is not None:
        scam_params += (("victims_per_receiving_mule", victims),)
    route_all = (("layers", layers), ("accounts_per_layer", receiving_mules)) + tuple(route_params)
    steps = [BlockStep("acquire", (("count", acquire_count),)), BlockStep("scam", scam_params)]
    steps.append(BlockStep("route", route_all))
    if with_disguise:
        steps.append(BlockStep("disguise", ()))
    return GangConfig("g1", strength, BUDGET, None, tuple(steps), start_minute)


def run_gang(gang: GangConfig) -> Any:
    """Runs the seed-101 showcase world with fraud on and the given gang, and returns the InteractiveWorldOutput."""
    settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        fast_settings=settings, runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,), code_version=code_version,
    )


def fraud_events_in_order(output: Any) -> list[FraudEvent]:
    """Lists every event of a gang payment in execution order, each with its story block, text and layer."""
    result = output.result
    listed: list[FraudEvent] = []
    for position, event in enumerate(result.run_result.events):
        fraud_index = result.event_fraud_input_index[position]
        if result.event_sources[position] != "fraud" or fraud_index is None:
            continue
        story = result.truth.payment_stories[fraud_index]
        listed.append(FraudEvent(event, story.block, story.description, story.layer))
    return listed


@pytest.fixture(scope="module")
def route_world() -> World:
    """Builds one reproducible world of 60 people and 12 shops, shared by the planning tests of this module."""
    return build_world(seed=101, n_people=60, n_shops=12)


@pytest.fixture(scope="module")
def chain_output() -> Any:
    """Runs one three-layer scam gang (2 receiving mules, 5 victims each) through the runner, shared by its tests."""
    return run_gang(make_gang(acquire_count=6, receiving_mules=2, victims=5, layers=3))


# ---- planned route --------------------------------------------------------------------------------------------


def plan_route_gang(world: World, gang: GangConfig, seed: int = 9) -> Any:
    """Plans a gang with plan_gang and a seeded generator."""
    return plan_gang(gang, world, np.random.default_rng(seed))


def split_plan(plan: Any) -> tuple[list[Any], list[Any]]:
    """Splits a plan's payments into the scam payments and the route payments, by their story block."""
    scam = [p for p, s in zip(plan.payments, plan.truth.payment_stories) if s.block == "scam"]
    route = [p for p, s in zip(plan.payments, plan.truth.payment_stories) if s.block == "route"]
    return scam, route


def test_route_amount_is_the_receiving_mules_net_victim_money(route_world: World) -> None:
    """Checks each layer 1 mule forwards victim payments minus payouts, and every later hop forwards the same."""
    plan = plan_route_gang(route_world, make_gang(6, 2, 4, 3))
    scam, route = split_plan(plan)

    first_hop_mules = {payment.from_account for payment in route} & {payment.to_account for payment in scam}
    assert len(first_hop_mules) == 2
    for mule in first_hop_mules:
        net = sum(p.amount_paise for p in scam if p.to_account == mule) - sum(
            p.amount_paise for p in scam if p.from_account == mule
        )
        forwards = [p for p in route if p.from_account == mule]
        assert [p.amount_paise for p in forwards] == [net]
        second_hop = [p for p in route if p.from_account == forwards[0].to_account]
        assert [p.amount_paise for p in second_hop] == [net]


def test_route_has_no_source_hop_and_no_collector(route_world: World) -> None:
    """Checks no route payment comes from a victim and the truth holds no collector role."""
    plan = plan_route_gang(route_world, make_gang(6, 2, 4, 3))
    scam, route = split_plan(plan)

    victims = {payment.from_account for payment in scam if payment.to_account.startswith("acc_")} - {
        payment.from_account for payment in route
    }
    assert victims
    victim_accounts = {role.account_id for role in plan.truth.roles if role.role == "victim"}
    assert not {payment.from_account for payment in route} & victim_accounts
    assert "collector" not in {role.role for role in plan.truth.roles}
    assert sorted({story.layer for story in plan.truth.payment_stories if story.block == "route"}) == [2, 3]


def test_first_forward_is_one_dwell_after_the_last_victim_payment(route_world: World) -> None:
    """Checks with a fixed dwell of 30 minutes that hop one is 30 minutes after the last victim payment, hop two 60."""
    plan = plan_route_gang(route_world, make_gang(6, 2, 4, 3, route_params=(("dwell_minutes", 30),)))
    scam, route = split_plan(plan)

    for mule in {payment.from_account for payment in route} & {payment.to_account for payment in scam}:
        last_victim_payment = max(p.minute for p in scam if p.to_account == mule)
        first_forward = next(p for p in route if p.from_account == mule)
        second_forward = next(p for p in route if p.from_account == first_forward.to_account)
        assert first_forward.minute == last_victim_payment + 30
        assert second_forward.minute == last_victim_payment + 60


def test_disguise_keeps_each_forward_inside_its_dwell_window(route_world: World) -> None:
    """Checks with disguise at strength 2 that a forward falls between the last victim payment and one dwell later."""
    gang = make_gang(6, 2, 4, 2, route_params=(("dwell_minutes", 40),), strength=2, with_disguise=True)
    plan = plan_route_gang(route_world, gang)
    scam, route = split_plan(plan)

    for mule in {payment.from_account for payment in route} & {payment.to_account for payment in scam}:
        last_victim_payment = max(p.minute for p in scam if p.to_account == mule)
        forward = next(p for p in route if p.from_account == mule)
        assert last_victim_payment <= forward.minute <= last_victim_payment + 40


def test_every_route_payment_spends_and_credits_illicit_money(route_world: World) -> None:
    """Checks route payments keep (spends, credits) = (True, True) and scam payments keep their own rules."""
    plan = plan_route_gang(route_world, make_gang(6, 2, 4, 3))

    for payment_index, (story, rule) in enumerate(zip(plan.truth.payment_stories, plan.money_rules)):
        assert rule.input_index == payment_index
        if story.block == "route":
            assert (rule.spends_illicit, rule.credits_illicit) == (True, True)
        elif "payout back" in story.description:
            assert (rule.spends_illicit, rule.credits_illicit) == (True, False)
        else:
            assert (rule.spends_illicit, rule.credits_illicit) == (False, True)


def test_one_layer_route_plans_no_forwards_but_still_costs(route_world: World) -> None:
    """Checks a one-layer route plans no payments: the layer 1 mules keep the money, and the route still has a cost."""
    plan = plan_route_gang(route_world, make_gang(4, 2, 3, 1))

    assert split_plan(plan)[1] == []
    assert len(plan.costs) == 3
    assert plan.costs[2].money_paise > 0
    assert {role.depth for role in plan.truth.roles if role.role == "mule"} == {1}


def test_route_amount_set_by_config_is_refused_for_a_scam_gang(route_world: World) -> None:
    """Checks plan_gang refuses a route amount_inr on a gang with a scam step, so config cannot disagree."""
    gang = make_gang(4, 2, 3, 2, route_params=(("amount_inr", 5000),))

    with pytest.raises(ValueError, match="must not set amount_inr"):
        plan_route_gang(route_world, gang)


def test_a_layer_one_mule_without_victim_money_is_refused(route_world: World) -> None:
    """Checks the route refuses a layer 1 mule that has no planned inflow, instead of inventing an amount."""
    gang = make_gang(4, 2, 3, 2)
    route_step = gang.steps[2]

    with pytest.raises(ValueError, match="receives no victim money to route"):
        _execute_victim_route_step(
            gang, route_step, ["m1", "m2", "m3", "m4"], START_MINUTE, {}, np.random.default_rng(1)
        )


def test_same_seed_gives_the_same_route(route_world: World) -> None:
    """Checks the planned payments of a scam gang with a route repeat exactly for the same seed."""
    gang = make_gang(6, 2, 4, 3)

    assert plan_route_gang(route_world, gang, 4).payments == plan_route_gang(route_world, gang, 4).payments


# ---- whole worlds ---------------------------------------------------------------------------------------------


def running_totals(events: list[FraudEvent], mule: str) -> list[tuple[int, int, int]]:
    """Returns (received, forwarded, paid back) after each successful event touching a mule, in execution order."""
    received = forwarded = paid_back = 0
    totals: list[tuple[int, int, int]] = []
    for item in events:
        event = item.event
        if event.status != "success" or mule not in (event.from_account, event.to_account):
            continue
        if event.to_account == mule:
            received += event.amount_paise
        elif item.block == "route":
            forwarded += event.amount_paise
        else:
            paid_back += event.amount_paise
        totals.append((received, forwarded, paid_back))
    return totals


def test_victim_to_layer_one_to_layer_two_never_forwards_more_than_received(chain_output: Any) -> None:
    """Follows victim to layer 1 to layer 2 to layer 3 in the events; no mule ever pays out more than it received."""
    events = fraud_events_in_order(chain_output)
    mules = {role.account_id: role.depth for role in chain_output.result.truth.roles if role.role == "mule"}

    victim_payments = [item for item in events if item.block == "scam" and "payout back" not in item.description]
    assert any(item.event.status == "success" for item in victim_payments)
    for mule in mules:
        for received, forwarded, paid_back in running_totals(events, mule):
            assert forwarded + paid_back <= received
    forwards_by_layer = {
        item.layer for item in events if item.block == "route" and item.event.status == "success"
    }
    assert forwards_by_layer == {2, 3}


def test_each_hop_receives_only_what_the_previous_mule_forwarded(chain_output: Any) -> None:
    """Checks a layer 2 mule receives only layer 1 forwards and a layer 3 mule only layer 2 forwards."""
    events = fraud_events_in_order(chain_output)
    depth_of = {role.account_id: role.depth for role in chain_output.result.truth.roles if role.role == "mule"}

    for item in events:
        if item.block != "route":
            continue
        assert depth_of[item.event.from_account] == item.layer - 1
        assert depth_of[item.event.to_account] == item.layer


def test_payout_reaches_the_victim_and_is_funded_only_by_held_illicit_money(chain_output: Any) -> None:
    """Checks every successful payout goes to a victim and never exceeds the illicit money the mule held."""
    events = fraud_events_in_order(chain_output)
    victims = {role.account_id for role in chain_output.result.truth.roles if role.role == "victim"}
    payouts = [item for item in events if "payout back" in item.description and item.event.status == "success"]
    assert payouts, "this gang is expected to include at least one successful payout back"

    for payout in payouts:
        mule, victim = payout.event.from_account, payout.event.to_account
        assert victim in victims
        position = events.index(payout)
        earlier = events[:position]
        held = 0
        for item in earlier:
            event = item.event
            if event.status != "success":
                continue
            if event.to_account == mule:
                held += event.amount_paise
            elif event.from_account == mule:
                held -= event.amount_paise
        assert payout.event.amount_paise <= held


def test_victims_have_a_role_from_their_first_payment_and_mules_keep_their_depths(chain_output: Any) -> None:
    """Checks victim roles start at the first payment, mules hold depths 1 to 3 two each, and no account holds two."""
    events = fraud_events_in_order(chain_output)
    roles = chain_output.result.truth.roles

    victim_roles = [role for role in roles if role.role == "victim"]
    assert len(victim_roles) == 10
    for role in victim_roles:
        first_payment = min(
            item.event.minute for item in events if item.event.from_account == role.account_id and item.block == "scam"
        )
        assert role.valid_from == first_payment
        assert role.valid_to is None and role.depth is None and role.gang_id == "g1"
    mule_depths = sorted(role.depth for role in roles if role.role == "mule")
    assert mule_depths == [1, 1, 2, 2, 3, 3]
    assert len({role.account_id for role in roles}) == len(roles)


def test_outage_stops_a_victim_payment_and_the_mule_forwards_nothing(route_world: World) -> None:
    """Places the only victim payment in the measured bank 1 outage and checks it fails and nothing is forwarded."""
    settings, _ = get_showcase_b_settings()
    outage = next(
        incident for incident in prepare_world(settings, None).plan.incidents
        if incident.kind == "bank_downtime" and incident.start_minute == OUTAGE_START_MINUTE
    )
    assert outage.bank_index == OUTAGE_BANK_INDEX
    assert outage.start_minute <= OUTAGE_START_MINUTE < outage.start_minute + outage.duration_minutes

    gang = make_gang(OUTAGE_ACQUIRE_COUNT, 1, 1, 2, start_minute=OUTAGE_GANG_START_MINUTE)
    output = run_gang(gang)
    events = fraud_events_in_order(output)

    victim_events = [item for item in events if item.block == "scam"]
    assert len(victim_events) == 1
    failed = victim_events[0].event
    assert (failed.from_account, failed.to_account, failed.amount_paise) == (
        OUTAGE_VICTIM, OUTAGE_MULE, OUTAGE_PLANNED_PAISE
    )
    assert (failed.minute, failed.status, failed.reason, failed.failure_domain) == (
        OUTAGE_START_MINUTE, OUTAGE_STATUS, OUTAGE_REASON, OUTAGE_DOMAIN
    )
    assert not [item for item in events if item.block == "route"]
    skipped = output.result.skipped_forwards
    assert [(s.from_account, s.planned_paise, s.available_paise, s.reason) for s in skipped] == [
        (OUTAGE_MULE, OUTAGE_PLANNED_PAISE, 0, "no_illicit_money_arrived")
    ]


def test_a_victim_who_runs_out_of_balance_fails_and_the_mule_forwards_only_what_arrived() -> None:
    """Checks failed victim payments of any reason leave the mules without a forward larger than what arrived."""
    output = run_gang(make_gang(7, 1, 2, 2, start_minute=7600))
    events = fraud_events_in_order(output)

    reasons = {item.event.reason for item in events if item.block == "scam" and item.event.status == "failed"}
    assert reasons
    receiving_mules = {
        item.event.to_account for item in events if item.block == "scam" and "payout back" not in item.description
    }
    for mule in receiving_mules:
        for received, forwarded, paid_back in running_totals(events, mule):
            assert forwarded + paid_back <= received
