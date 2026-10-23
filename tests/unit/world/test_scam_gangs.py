"""
Tests for gangs whose money comes from scam victims (world/scam_planner.py, world/gang_config.py and
world/gang_planner.py).
Covers who the victims are, the payment schedule of each scam type, the money rules and truth the step writes, the
refusals for a wrong money source or step order, and the proof that a gang with a fixture source plans exactly as
before.

TERM: receiving mule: an acquired mule account that victims pay.
TERM: victim: an ordinary person who pays a receiving mule because of a scam.
TERM: payout back: a small payment from the receiving mule back to the victim, after the first payment and before
      the last.
TERM: fixture source: a pre-funded collector account that is a gang's money source when it has no scam step.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import world.scam_planner as scam_planner
from showcase.level_12_support_b import get_showcase_b_settings, load_showcase_b_fixture
from world.build_world import World, build_world
from world.gang_config import (
    BlockStep,
    FixtureSource,
    GangBudget,
    GangConfig,
    gang_config_from_plain,
    gang_config_to_plain,
    gang_start_minute,
    validate_gang_config,
)
from world.gang_planner import plan_gang
from world.interactive_records import RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.scam_planner import (
    TypeChances,
    choose_victims,
    plan_scam_step,
    plan_victim_schedule,
)

GENEROUS_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
START_MINUTE = 1440
SPLIT = "training"
INVESTMENT_CHANCES = TypeChances(investment_or_job=1.0, investment_within=1.0)
JOB_CHANCES = TypeChances(investment_or_job=1.0, investment_within=0.0)
DEBIT_CHANCES = TypeChances(investment_or_job=0.0, investment_within=0.5)

# Values the 12b fixture gang produced at commit a103b48, before the scam step existed. The plan value is a
# projection: role spans reduced to their six original fields (the origin field came later), then payments,
# stories, costs, totals and money rules. It was computed at a103b48 and still matches, so no draw changed.
# Story fields are reduced to their four original fields (scam_type and scam_step came later, Level 12e).
FIXTURE_PLAIN_SHA256 = "28615d91d9a0b0f0ab09d79c55d5241d051638fc8df4bf03cc5e59ee1252f30e"
FIXTURE_RUN_ID = "f3dc93d7013b2eba"
FIXTURE_PLAN_SHA256 = "ade10fcfd89f329ba9b8fa67712da4e6253e05c09ff884ac4caa68338b27fc07"


@dataclass(frozen=True)
class OriginalStoryFields:
    """Stand-in for PaymentStory as it was at a103b48 (four fields), used only to keep the pinned plan hash.

    Its repr reproduces the four-field PaymentStory repr exactly, so FIXTURE_PLAN_SHA256 keeps proving that no
    plan value or random draw changed. scam_type and scam_step (Level 12e) are checked separately.
    """

    gang_id: str
    block: str
    layer: int | None
    description: str

    def __repr__(self) -> str:
        """Prints the four fields in the exact text the four-field PaymentStory printed at a103b48."""
        return (
            f"PaymentStory(gang_id={self.gang_id!r}, block={self.block!r}, "
            f"layer={self.layer!r}, description={self.description!r})"
        )


@pytest.fixture
def scam_world() -> World:
    """Builds a small reproducible world of 60 people and 12 shops, enough people for victims and mules."""
    return build_world(seed=101, n_people=60, n_shops=12)


def make_scam_config(
    receiving_mules: int = 2, acquire_count: int = 4, extra_scam_params: tuple[tuple[str, Any], ...] = (),
    strength: int = 1, route: bool = False,
) -> GangConfig:
    """Builds a scam-sourced gang: acquire, scam, and optionally a route sized to the receiving mules."""
    scam_params = (("receiving_mules", receiving_mules),) + extra_scam_params
    steps = [BlockStep("acquire", (("count", acquire_count),)), BlockStep("scam", scam_params)]
    if route:
        steps.append(BlockStep("route", (("layers", 2), ("accounts_per_layer", receiving_mules))))
    return GangConfig("g1", strength, GENEROUS_BUDGET, None, tuple(steps), START_MINUTE)


def plan_scam(world: World, config: GangConfig, seed: int = 5) -> Any:
    """Runs plan_scam_step for a config's scam step with fixed mules and a seeded generator."""
    mules = [f"mule_{index}" for index in range(config.steps[0].params_dict["count"])]
    scam_step = next(step for step in config.steps if step.block == "scam")
    return plan_scam_step(world, config, scam_step, mules, START_MINUTE, None, np.random.default_rng(seed))


def victim_payments_only(schedule_payments: tuple[Any, ...]) -> list[Any]:
    """Returns the victim payments of a schedule, leaving out the payout back."""
    return [payment for payment in schedule_payments if not payment.is_payout]


def test_victims_are_ordinary_people_never_mules_sources_or_reserved(scam_world: World) -> None:
    """Checks every victim is a person account, distinct, and outside the mules, source and reserved accounts."""
    people = {person.account_id for person in scam_world.people}
    shops = {shop.account_id for shop in scam_world.shops}
    sorted_people = sorted(people)
    mules = sorted_people[:4]
    reserved = {sorted_people[4], sorted_people[5]}
    source_account = sorted_people[6]
    excluded = set(mules) | reserved | {source_account}

    for seed in range(20):
        victims = choose_victims(scam_world, 30, excluded, np.random.default_rng(seed))
        assert len(set(victims)) == 30
        assert set(victims) <= people
        assert not set(victims) & shops
        assert not set(victims) & excluded


def test_plan_excludes_source_reserved_and_mules_from_victims(scam_world: World) -> None:
    """Checks plan_scam_step never uses the receiving mules, the reserved accounts or a source as victims."""
    config = make_scam_config()
    mules = ["acc_person_000000", "acc_person_000001", "acc_person_000002", "acc_person_000003"]
    reserved = {"acc_person_000004", "acc_person_000005"}
    scam_step = config.steps[1]

    for seed in range(20):
        plan = plan_scam_step(scam_world, config, scam_step, mules, START_MINUTE, reserved, np.random.default_rng(seed))
        victim_accounts = {role.account_id for role in plan.roles}
        assert not victim_accounts & (set(mules) | reserved)
        assert all(role.role == "victim" for role in plan.roles)


def test_fixture_source_account_is_never_a_victim(scam_world: World) -> None:
    """Checks that the account of a fixture source named on the config is excluded from the victim pool."""
    source = FixtureSource("acc_person_000010", 0, 1_000_000, START_MINUTE)
    steps = (BlockStep("acquire", (("count", 4),)), BlockStep("scam", (("receiving_mules", 2),)))
    config = GangConfig("g1", 1, GENEROUS_BUDGET, source, steps)
    mules = ["acc_person_000000", "acc_person_000001", "acc_person_000002", "acc_person_000003"]

    # A direct call bypasses validation on purpose: the exclusion itself is under test, not the refusal.
    plan = plan_scam_step(scam_world, config, steps[1], mules, START_MINUTE, None, np.random.default_rng(1))

    assert "acc_person_000010" not in {role.account_id for role in plan.roles}


def test_each_victim_pays_one_receiving_mule_and_mules_split_the_victims(scam_world: World) -> None:
    """Checks each victim pays exactly one mule and only the first receiving_mules acquired mules are paid."""
    config = make_scam_config(receiving_mules=3, acquire_count=5)
    plan = plan_scam(scam_world, config)

    victim_to_mules: dict[str, set[str]] = {}
    for payment in plan.payments:
        if payment.to_account.startswith("mule_"):
            victim_to_mules.setdefault(payment.from_account, set()).add(payment.to_account)
    assert all(len(mule_set) == 1 for mule_set in victim_to_mules.values())
    paid_mules = set().union(*victim_to_mules.values())
    assert paid_mules <= {"mule_0", "mule_1", "mule_2"}
    assert set(plan.mule_inflows) == paid_mules


# Each seed checks the same property of one drawn schedule; 8 seeds cover it, so the suite stays fast.
@pytest.mark.parametrize("seed", range(8))
def test_investment_schedule_follows_config(seed: int) -> None:
    """Checks count, first amount, escalation, minimum gap, spread and payout of an investment victim."""
    rng = np.random.default_rng(seed)
    schedule = plan_victim_schedule("victim", "mule", INVESTMENT_CHANCES, START_MINUTE, SPLIT, rng)
    assert schedule.scam_type == "investment"
    victim_payments = victim_payments_only(schedule.payments)
    amounts = [payment.amount_paise for payment in victim_payments]

    assert 1 <= len(victim_payments) <= 6
    assert 200_000 <= amounts[0] <= 5_000_000
    assert victim_payments[0].minute == START_MINUTE
    for previous, current in zip(amounts, amounts[1:]):
        assert previous <= current <= previous * 2
    minutes = [payment.minute for payment in victim_payments]
    assert all(later - earlier >= 2 for earlier, later in zip(minutes, minutes[1:]))
    # The raw offsets stay inside 10 days; the minimum gap can only push the last payment out by (count - 1) * 2.
    assert minutes[-1] - minutes[0] <= 10 * 1440 + 2 * (len(minutes) - 1)
    assert_payout_follows_config(schedule.payments)


def assert_payout_follows_config(payments: tuple[Any, ...]) -> None:
    """Checks the payout of a schedule: absent for one payment, otherwise strictly between first and last."""
    victim_payments = victim_payments_only(payments)
    payouts = [payment for payment in payments if payment.is_payout]
    if len(victim_payments) == 1:
        assert payouts == []
        return
    assert len(payouts) <= 1
    for payout in payouts:
        assert victim_payments[0].minute < payout.minute < victim_payments[-1].minute
        paid_before = sum(payment.amount_paise for payment in victim_payments if payment.minute < payout.minute)
        assert int(0.05 * paid_before) <= payout.amount_paise <= int(0.3 * paid_before)
    assert [payment.minute for payment in payments] == sorted(payment.minute for payment in payments)


def test_job_schedule_has_the_same_shape_as_investment() -> None:
    """Checks a job victim gets the escalating schedule with a job label, and never a debit schedule."""
    for seed in range(20):
        schedule = plan_victim_schedule("victim", "mule", JOB_CHANCES, START_MINUTE, SPLIT, np.random.default_rng(seed))
        assert schedule.scam_type == "job"
        amounts = [payment.amount_paise for payment in victim_payments_only(schedule.payments)]
        assert amounts == sorted(amounts)


# Each seed checks the same property of one drawn schedule; 8 seeds cover it, so the suite stays fast.
@pytest.mark.parametrize("seed", range(8))
def test_unauthorised_debit_has_no_payout_and_stays_inside_the_spread(seed: int) -> None:
    """Checks debits: one to three of them inside 60 minutes, drawn amounts, and no payout back."""
    schedule = plan_victim_schedule("victim", "mule", DEBIT_CHANCES, START_MINUTE, SPLIT, np.random.default_rng(seed))
    assert schedule.scam_type == "unauthorised_debit"

    assert not any(payment.is_payout for payment in schedule.payments)
    assert 1 <= len(schedule.payments) <= 3
    assert schedule.payments[0].minute == START_MINUTE
    assert max(payment.minute for payment in schedule.payments) - START_MINUTE <= 60
    assert all(200_000 <= payment.amount_paise <= 5_000_000 for payment in schedule.payments)


def test_type_mix_follows_the_drawn_chances() -> None:
    """Checks the draw: certain chances give the matching types and a mixed chance gives all three."""
    rng = np.random.default_rng(3)
    seen = {
        plan_victim_schedule("v", "m", TypeChances(0.6, 0.5), START_MINUTE, SPLIT, rng).scam_type for _ in range(200)
    }
    assert seen == {"investment", "job", "unauthorised_debit"}


def test_narrowed_config_values_reach_the_investment_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Narrows every investment setting to one value through the resolvers and checks the schedule uses each."""
    monkeypatch.setattr(scam_planner, "resolve_payments_per_victim", lambda split: (3, 3))
    monkeypatch.setattr(scam_planner, "resolve_payment_spread_days", lambda split: (0, 0))
    monkeypatch.setattr(scam_planner, "resolve_first_payment_amount_inr", lambda split: (5000.0, 5000.0))
    monkeypatch.setattr(scam_planner, "resolve_payment_escalation_multiple", lambda split: (1.5, 1.5))
    monkeypatch.setattr(scam_planner, "resolve_min_minutes_between_victim_payments", lambda: 10)
    monkeypatch.setattr(scam_planner, "resolve_payout_back_share", lambda split: (0.2, 0.2))

    schedule = plan_victim_schedule("victim", "mule", INVESTMENT_CHANCES, 100, SPLIT, np.random.default_rng(1))

    payments = victim_payments_only(schedule.payments)
    assert [payment.amount_paise for payment in payments] == [500_000, 750_000, 1_125_000]
    assert [payment.minute for payment in payments] == [100, 110, 120]
    payouts = [payment for payment in schedule.payments if payment.is_payout]
    assert len(payouts) == 1
    paid_before = sum(payment.amount_paise for payment in payments if payment.minute < payouts[0].minute)
    assert payouts[0].amount_paise == int(0.2 * paid_before)


def test_narrowed_config_values_reach_the_debit_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Narrows every debit setting and checks count, spread, amount, the payout flag and the remainder type name."""
    monkeypatch.setattr(scam_planner, "resolve_debit_payment_count", lambda split: (2, 2))
    monkeypatch.setattr(scam_planner, "resolve_debit_spread_minutes", lambda split: (0, 0))
    monkeypatch.setattr(scam_planner, "resolve_debit_amount_inr", lambda split: (3000.0, 3000.0))
    monkeypatch.setattr(scam_planner, "resolve_remainder_scam_type", lambda: "takeover_debit")

    schedule = plan_victim_schedule("victim", "mule", DEBIT_CHANCES, 100, SPLIT, np.random.default_rng(1))

    assert schedule.scam_type == "takeover_debit"
    assert [(payment.minute, payment.amount_paise) for payment in schedule.payments] == [(100, 300_000), (100, 300_000)]


def test_debit_payout_follows_the_config_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switches the debit payout flag on and checks a payout appears strictly between first and last debit."""
    monkeypatch.setattr(scam_planner, "resolve_debit_pays_back", lambda: True)
    monkeypatch.setattr(scam_planner, "resolve_debit_payment_count", lambda split: (3, 3))
    monkeypatch.setattr(scam_planner, "resolve_debit_spread_minutes", lambda split: (60, 60))

    payouts_seen = 0
    for seed in range(30):
        schedule = plan_victim_schedule("v", "m", DEBIT_CHANCES, 100, SPLIT, np.random.default_rng(seed))
        assert_payout_follows_config(schedule.payments)
        payouts_seen += sum(1 for payment in schedule.payments if payment.is_payout)
    assert payouts_seen > 0


def test_narrowed_type_chances_and_counts_reach_the_plan(monkeypatch: pytest.MonkeyPatch, scam_world: World) -> None:
    """Narrows the chances, victim range, risk range and purpose, and checks the plan carries each value."""
    monkeypatch.setattr(scam_planner, "resolve_investment_or_job_share", lambda split: (1.0, 1.0))
    monkeypatch.setattr(scam_planner, "resolve_investment_share_of_investment_or_job", lambda split: (1.0, 1.0))
    monkeypatch.setattr(scam_planner, "get_block_risk_range", lambda block: (0.5, 0.5))
    monkeypatch.setattr(scam_planner, "get_gang_payment_purpose", lambda: "rent")
    config = make_scam_config(receiving_mules=2, extra_scam_params=(("victims_per_receiving_mule", 2),))

    plan = plan_scam(scam_world, config)

    assert len(plan.roles) == 4
    assert plan.risk == 0.5
    assert all(payment.purpose == "rent" for payment in plan.payments)
    assert all(story.description.split()[2] == "investment" for story in plan.stories)


def test_explicit_victims_per_receiving_mule_is_used(scam_world: World) -> None:
    """Checks that a pinned victims_per_receiving_mule gives every receiving mule exactly that many victims."""
    config = make_scam_config(receiving_mules=2, extra_scam_params=(("victims_per_receiving_mule", 3),))

    plan = plan_scam(scam_world, config)

    assert len(plan.roles) == 6


def test_money_rules_stories_and_roles_of_the_scam_step(scam_world: World) -> None:
    """Checks rules, stories and roles line up with payments, and the payout spends where victim payments credit."""
    config = make_scam_config(receiving_mules=2)
    plan = plan_scam(scam_world, config, seed=11)
    payout_seen = False

    assert len(plan.payments) == len(plan.stories) == len(plan.rules)
    for index, (payment, story, rule) in enumerate(zip(plan.payments, plan.stories, plan.rules)):
        assert rule.input_index == index
        assert story.gang_id == "g1" and story.block == "scam" and story.layer is None
        if payment.from_account.startswith("mule_"):
            payout_seen = True
            assert (rule.spends_illicit, rule.credits_illicit) == (True, False)
            assert "payout back" in story.description
        else:
            assert (rule.spends_illicit, rule.credits_illicit) == (False, True)
            assert "scam payment" in story.description and " of " in story.description
    first_minutes = {role.account_id: role.valid_from for role in plan.roles}
    for payment in plan.payments:
        if not payment.from_account.startswith("mule_"):
            assert first_minutes[payment.from_account] <= payment.minute
    assert all(role.role == "victim" and role.depth is None and role.valid_to is None for role in plan.roles)
    assert payout_seen, "seed 11 is expected to include at least one payout back"


def test_payment_story_numbers_count_only_victim_payments(scam_world: World) -> None:
    """Checks the 'payment k of n' text counts a victim's own payments in order and leaves out the payout."""
    config = make_scam_config(receiving_mules=1, extra_scam_params=(("victims_per_receiving_mule", 1),))
    for seed in range(30):
        plan = plan_scam(scam_world, config, seed=seed)
        texts = [story.description for story in plan.stories if "payment " in story.description]
        total = len(texts)
        victim_texts = [text for text in texts if "scam payment" in text]
        if len(victim_texts) == total and "investment" in victim_texts[0] and total >= 2:
            for number, text in enumerate(victim_texts, start=1):
                assert text.endswith(f"payment {number} of {total}")
            return
    pytest.fail("no multi-payment investment victim found in 30 seeds")


def test_mule_inflow_is_victim_payments_minus_payouts(scam_world: World) -> None:
    """Checks each mule's planned inflow equals what its victims pay minus what it pays back, and its last minute."""
    config = make_scam_config(receiving_mules=2)
    plan = plan_scam(scam_world, config, seed=11)

    for mule, inflow in plan.mule_inflows.items():
        received = [p for p in plan.payments if p.to_account == mule]
        paid_back = [p for p in plan.payments if p.from_account == mule]
        assert inflow.net_paise == sum(p.amount_paise for p in received) - sum(p.amount_paise for p in paid_back)
        assert inflow.last_victim_payment_minute == max(p.minute for p in received)


def test_same_seed_gives_the_same_plan_and_other_seed_differs(scam_world: World) -> None:
    """Checks determinism: the same seed repeats the plan exactly and another seed changes it."""
    config = make_scam_config()

    assert plan_scam(scam_world, config, seed=4).payments == plan_scam(scam_world, config, seed=4).payments
    assert plan_scam(scam_world, config, seed=4).payments != plan_scam(scam_world, config, seed=5).payments


def test_plan_gang_costs_scam_step_by_span_and_risk(scam_world: World) -> None:
    """Checks plan_gang gives the scam step zero money, its span as time, and a risk from the scam risk range."""
    config = make_scam_config()

    plan = plan_gang(config, scam_world, np.random.default_rng(8))

    scam_cost = plan.costs[1]
    assert scam_cost.money_paise == 0
    assert scam_cost.time_minutes == max(p.minute for p in plan.payments) - START_MINUTE
    assert 0.001 <= scam_cost.risk <= 0.02
    assert [rule.input_index for rule in plan.money_rules] == list(range(len(plan.payments)))
    assert len(plan.truth.payment_stories) == len(plan.payments)


def test_scam_step_after_warm_up_starts_later(scam_world: World) -> None:
    """Checks the scam step starts at the gang clock, which a warm_up before it has moved forward."""
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("warm_up", (("days", 2),)),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    config = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE)

    plan = plan_gang(config, scam_world, np.random.default_rng(8))

    mule_roles = [role for role in plan.truth.roles if role.role == "mule"]
    assert mule_roles
    assert all(role.active_from == START_MINUTE + 2 * 1440 for role in mule_roles)


def test_scam_needs_enough_acquired_mules_and_people(scam_world: World) -> None:
    """Checks plan_gang refuses a scam naming more receiving mules than were acquired, and too many victims."""
    config = make_scam_config(receiving_mules=3, acquire_count=2)
    with pytest.raises(ValueError, match="needs 3 receiving mules but acquired 2"):
        plan_gang(config, scam_world, np.random.default_rng(8))

    with pytest.raises(ValueError, match="eligible people are available"):
        choose_victims(scam_world, 1000, set(), np.random.default_rng(1))


def test_a_mule_with_nothing_left_after_payouts_is_refused(monkeypatch: pytest.MonkeyPatch, scam_world: World) -> None:
    """Forces payouts above the payments with a share over 1 and checks the plan refuses a mule with no money."""
    monkeypatch.setattr(scam_planner, "resolve_payout_back_share", lambda split: (2.0, 2.0))
    monkeypatch.setattr(scam_planner, "resolve_payments_per_victim", lambda split: (2, 2))
    monkeypatch.setattr(scam_planner, "resolve_payment_escalation_multiple", lambda split: (1.0, 1.0))
    monkeypatch.setattr(scam_planner, "resolve_payment_spread_days", lambda split: (1, 1))
    monkeypatch.setattr(scam_planner, "resolve_investment_or_job_share", lambda split: (1.0, 1.0))
    config = make_scam_config(receiving_mules=1, extra_scam_params=(("victims_per_receiving_mule", 1),))

    with pytest.raises(ValueError, match="nothing to forward"):
        for seed in range(30):
            plan_scam(scam_world, config, seed=seed)


def test_zero_victims_per_mule_is_refused(monkeypatch: pytest.MonkeyPatch, scam_world: World) -> None:
    """Narrows the victims range to zero and checks the step refuses a receiving mule without victims."""
    monkeypatch.setattr(scam_planner, "resolve_victims_per_mule_largest", lambda split: (0, 0))

    with pytest.raises(ValueError, match="at least 1 victim"):
        plan_scam(scam_world, make_scam_config())


# ---- gang configuration: money source, step order, parameters -------------------------------------------------


def test_scam_gang_validates_and_reports_its_start_minute() -> None:
    """Checks a well-formed scam gang passes validation and gang_start_minute returns its own start_minute."""
    config = make_scam_config(route=True)

    validate_gang_config(config)

    assert gang_start_minute(config) == START_MINUTE


def test_fixture_gang_start_minute_comes_from_its_source() -> None:
    """Checks a fixture gang reports its source's start_minute."""
    gang, _ = load_showcase_b_fixture()

    assert gang_start_minute(gang) == gang.source.start_minute


def test_gang_with_neither_money_source_is_refused() -> None:
    """Checks a gang with no fixture source and no scam step is refused with one plain sentence."""
    steps = (BlockStep("acquire", (("count", 2),)),)

    with pytest.raises(ValueError, match="exactly one money source but has neither"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE))


def test_gang_with_both_money_sources_is_refused() -> None:
    """Checks a gang with a fixture source and a scam step is refused."""
    source = FixtureSource("acc_collector_01", 0, 1_000_000, START_MINUTE)
    steps = (BlockStep("acquire", (("count", 2),)), BlockStep("scam", (("receiving_mules", 1),)))

    with pytest.raises(ValueError, match="exactly one money source but has both"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, source, steps))


def test_gang_with_two_scam_steps_is_refused() -> None:
    """Checks that two scam steps are refused, because a gang has one money source."""
    scam = BlockStep("scam", (("receiving_mules", 1),))
    steps = (BlockStep("acquire", (("count", 2),)), scam, scam)

    with pytest.raises(ValueError, match="only one scam step"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE))


@pytest.mark.parametrize("start_minute", [None, -1, True, 1.5])
def test_scam_gang_needs_a_valid_start_minute(start_minute: Any) -> None:
    """Checks a scam gang without a usable start_minute is refused."""
    steps = (BlockStep("acquire", (("count", 2),)), BlockStep("scam", (("receiving_mules", 1),)))

    with pytest.raises(ValueError, match="needs a start_minute of 0 or more"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, start_minute))


def test_start_minute_with_a_fixture_source_is_refused() -> None:
    """Checks that start_minute beside a fixture source is refused, since the source already carries one."""
    source = FixtureSource("acc_collector_01", 0, 1_000_000, START_MINUTE)
    steps = (BlockStep("acquire", (("count", 2),)),)

    with pytest.raises(ValueError, match="already carries one"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, source, steps, START_MINUTE))


def test_scam_after_route_and_before_acquire_are_refused() -> None:
    """Checks the step order: the scam step needs acquire before it and route after it."""
    scam = BlockStep("scam", (("receiving_mules", 1),))
    acquire = BlockStep("acquire", (("count", 2),))
    route = BlockStep("route", (("layers", 1), ("accounts_per_layer", 1)))

    with pytest.raises(ValueError, match="must scam before routing"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, (acquire, route, scam), START_MINUTE))
    with pytest.raises(ValueError, match="must acquire accounts before a scam step"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, (scam, acquire), START_MINUTE))


def test_route_accounts_per_layer_must_equal_receiving_mules() -> None:
    """Checks a route whose layer 1 is not exactly the receiving mules is refused, including a missing value."""
    acquire = BlockStep("acquire", (("count", 4),))
    scam = BlockStep("scam", (("receiving_mules", 2),))
    for route_params in ((("layers", 2), ("accounts_per_layer", 3)), (("layers", 2),)):
        steps = (acquire, scam, BlockStep("route", route_params))
        with pytest.raises(ValueError, match="accounts_per_layer must equal the scam step's receiving_mules of 2"):
            validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE))


def test_route_amount_with_a_scam_source_is_refused() -> None:
    """Checks that a configured route amount_inr is refused when the money comes from victims."""
    route = BlockStep("route", (("layers", 2), ("accounts_per_layer", 2), ("amount_inr", 5000)))
    steps = (BlockStep("acquire", (("count", 4),)), BlockStep("scam", (("receiving_mules", 2),)), route)

    with pytest.raises(ValueError, match="must not set amount_inr"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ((), "scam step needs receiving_mules"),
        ((("receiving_mules", 0),), "receiving_mules must be an integer in"),
        ((("receiving_mules", 11),), "receiving_mules must be an integer in"),
        ((("receiving_mules", True),), "receiving_mules must be an integer in"),
        ((("receiving_mules", 1.5),), "receiving_mules must be an integer in"),
        ((("receiving_mules", 1), ("victims_per_receiving_mule", 0)), "victims_per_receiving_mule must be an integer"),
        ((("receiving_mules", 1), ("victims_per_receiving_mule", 51)), "victims_per_receiving_mule must be an integer"),
    ],
)
def test_scam_parameters_are_validated(params: tuple[tuple[str, Any], ...], message: str) -> None:
    """Checks a missing, out-of-range, boolean or fractional scam parameter is refused with a plain sentence."""
    steps = (BlockStep("acquire", (("count", 2),)), BlockStep("scam", params))

    with pytest.raises(ValueError, match=message):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE))


def test_scam_block_accepts_both_strengths() -> None:
    """Checks the scam block is allowed at strength 1 and 2 and refused for an unknown parameter."""
    validate_gang_config(make_scam_config(strength=1))
    validate_gang_config(make_scam_config(strength=2))
    steps = (BlockStep("acquire", (("count", 2),)), BlockStep("scam", (("receiving_mules", 1), ("bogus", 1))))
    with pytest.raises(ValueError, match="unknown parameter 'bogus'"):
        validate_gang_config(GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, START_MINUTE))


# ---- plain form, run id and plan of the fixture gang ----------------------------------------------------------


def test_scam_gang_plain_form_has_start_minute_and_no_source_and_round_trips() -> None:
    """Checks the plain dict of a scam gang omits source, writes start_minute, and converts back to an equal config."""
    config = make_scam_config(route=True)

    plain = gang_config_to_plain(config)

    assert "source" not in plain
    assert plain["start_minute"] == START_MINUTE
    assert gang_config_from_plain(json.loads(json.dumps(plain))) == config


def test_fixture_gang_plain_form_has_source_and_no_start_minute() -> None:
    """Checks the 12b fixture gang's plain dict has its source and no start_minute key."""
    gang, _ = load_showcase_b_fixture()

    plain = gang_config_to_plain(gang)

    assert "start_minute" not in plain
    assert plain["source"]["start_minute"] == gang.source.start_minute


def test_fixture_gang_plain_form_run_id_and_plan_are_unchanged() -> None:
    """Compares the 12b fixture's plain dict, run id and plan with the values computed at commit a103b48."""
    gang, _ = load_showcase_b_fixture()
    settings, code_version = get_showcase_b_settings()

    plain_text = json.dumps(gang_config_to_plain(gang), sort_keys=True)
    assert hashlib.sha256(plain_text.encode("utf-8")).hexdigest() == FIXTURE_PLAIN_SHA256

    output = run_interactive_world(
        fast_settings=settings, runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,), code_version=code_version,
    )
    assert output.result.run_id == FIXTURE_RUN_ID

    children = np.random.SeedSequence(settings.seed).spawn(10)
    dev_rng, plan_rng = np.random.default_rng(children[6]), np.random.default_rng(children[7])
    world = prepare_world(settings, None).world
    plan = plan_gang(gang, world, plan_rng, reserved_accounts={gang.source.account_id}, gang_devices_rng=dev_rng)
    original_role_fields = tuple(
        (role.account_id, role.role, role.gang_id, role.depth, role.valid_from, role.valid_to)
        for role in plan.truth.roles
    )
    original_money_rules = tuple(
        (rule.input_index, rule.spends_illicit, rule.credits_illicit)
        for rule in plan.money_rules
    )
    original_story_fields = tuple(
        OriginalStoryFields(story.gang_id, story.block, story.layer, story.description)
        for story in plan.truth.payment_stories
    )
    # This 12b fixture is not a scam gang, so the later scam fields must be empty on every story.
    assert all(story.scam_type is None and story.scam_step is None for story in plan.truth.payment_stories)
    plan_text = repr(
        (
            plan.payments,
            original_role_fields,
            original_story_fields,
            plan.costs,
            plan.totals,
            original_money_rules,
        )
    )
    assert hashlib.sha256(plan_text.encode("utf-8")).hexdigest() == FIXTURE_PLAN_SHA256


def test_scam_gang_runs_through_the_interactive_runner() -> None:
    """Runs a scam-only gang end to end: victims pay mules, no source is registered, and a payment succeeds."""
    settings, code_version = get_showcase_b_settings()
    config = make_scam_config(receiving_mules=2)

    output = run_interactive_world(
        fast_settings=settings, runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(config,), code_version=code_version,
    )

    assert output.result.money_created_paise == 0
    fraud_events = [
        event for event, source in zip(output.result.run_result.events, output.result.event_sources)
        if source == "fraud"
    ]
    assert fraud_events
    assert any(event.status == "success" for event in fraud_events)
    assert {role.role for role in output.result.truth.roles} == {"victim"}
