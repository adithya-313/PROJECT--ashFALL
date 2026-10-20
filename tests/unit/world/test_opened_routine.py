"""
Tests that new genuine people get a normal routine life (world/opened_routine.py and its use in the fraud-on loop).
Covers the idle share and its config, the planned routine of one person, and whole worlds that compare runs with and
without routines: nothing before the opening minute, salary for a new salaried person, no change to any existing
genuine payment's draws, and no routine for gang-opened accounts.

TERM: new genuine person: a person account opened mid-run by the account openings switch.
TERM: idle person: a new genuine person with no routine payments at all.
TERM: routine: salary, rent, bills, everyday shop payments and, for lookalike personas, inbound peer transfers.
TERM: baseline run: the same openings with every new person idle, so no routine payment exists.
"""

import copy
import dataclasses
from typing import Any

import numpy as np
import pytest

import world.mule_origins as mule_origins
from showcase.level_12_support_b import get_showcase_b_settings
from world.account_openings_config import load_account_openings_config
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import load_runner_config, prepare_world, run_interactive_world
from world.opened_routine import (
    draw_no_routine_share,
    is_idle,
    plan_opened_routine,
    resolve_no_routine_share_range,
)
from world.outside_events import OutsideEvent
from world.personas import get_persona

OPENING_SETTINGS = RunnerSettings(fraud_enabled=True, account_openings_enabled=True)
DAY = 1440


def openings_config(idle_share: tuple[float, float]) -> dict[str, Any]:
    """Builds an account openings config with many person openings and the given idle share for every split."""
    config = copy.deepcopy(load_account_openings_config())
    parameters = config["parameters"]
    for split in ("training", "test", "stress"):
        parameters["genuine_account_openings_per_1000_people_per_30_days"][split] = [300.0, 300.0]
        parameters["share_of_openings_that_are_shops"][split] = [0.0, 0.0]
        parameters["share_of_new_genuine_people_with_no_routine"][split] = list(idle_share)
    return config


def run_openings(idle_share: tuple[float, float], gang: GangConfig | None = None) -> Any:
    """Runs the seed-101 showcase world with many genuine person openings and the given idle share."""
    settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        fast_settings=settings, runner_settings=OPENING_SETTINGS, code_version=code_version,
        gang_configs=(gang,) if gang is not None else (), account_openings_config=openings_config(idle_share),
    )


def planned_events(output: Any) -> list[Any]:
    """Lists the events of genuine payments (existing and new people alike) in execution order."""
    result = output.result
    return [event for event, source in zip(result.run_result.events, result.event_sources) if source == "planned"]


def opened_person_ids(output: Any) -> set[str]:
    """Returns the account ids of the person accounts opened during the run."""
    return {row[1] for row in output.result.opened_accounts if row[2] == "person"}


def touches(event: Any, accounts: set[str]) -> bool:
    """Tells whether an event pays from or to one of the accounts."""
    return event.from_account in accounts or event.to_account in accounts


def drawn_outcome(event: Any) -> tuple[Any, ...]:
    """
    Reduces an event to what the random failure draws decided. Running out of balance follows from money flow, not
    from a draw, so it counts as no drawn failure, like a success.
    """
    drawn_failure = event.status == "failed" and event.reason != "insufficient_balance"
    reason = (event.reason, event.failure_domain) if drawn_failure else None
    return (event.minute, event.from_account, event.to_account, event.amount_paise, reason)


@pytest.fixture(scope="module")
def idle_output() -> Any:
    """Runs the openings with every new person idle: the baseline that has no routine payments."""
    return run_openings((1.0, 1.0))


@pytest.fixture(scope="module")
def routine_output() -> Any:
    """Runs the same openings with every new person given a routine."""
    return run_openings((0.0, 0.0))


# ---- the idle share and its config ----------------------------------------------------------------------------


def test_idle_share_is_read_from_the_committed_config() -> None:
    """Checks the committed ranges: training and test 0.2 to 0.6, stress 0 to 1."""
    config = load_account_openings_config()

    assert resolve_no_routine_share_range("training", config) == (0.2, 0.6)
    assert resolve_no_routine_share_range("test", config) == (0.2, 0.6)
    assert resolve_no_routine_share_range("stress", config) == (0.0, 1.0)


def test_idle_share_follows_a_narrowed_config() -> None:
    """Narrows the range and checks the resolver and the draw return the narrowed values."""
    config = openings_config((0.7, 0.7))

    assert resolve_no_routine_share_range("training", config) == (0.7, 0.7)
    assert draw_no_routine_share("training", config, np.random.default_rng(1)) == 0.7


def test_idle_share_config_refusals() -> None:
    """Checks a missing setting, a missing split and a malformed range are each refused in one sentence."""
    config = copy.deepcopy(load_account_openings_config())
    entry = config["parameters"]["share_of_new_genuine_people_with_no_routine"]
    entry["training"] = [0.6, 0.2]
    with pytest.raises(ValueError, match="two ordered shares in \\[0, 1\\]"):
        resolve_no_routine_share_range("training", config)
    entry["training"] = [0.2, 1.5]
    with pytest.raises(ValueError, match="two ordered shares in \\[0, 1\\]"):
        resolve_no_routine_share_range("training", config)
    del entry["training"]
    with pytest.raises(ValueError, match="no range for split 'training'"):
        resolve_no_routine_share_range("training", config)
    del config["parameters"]["share_of_new_genuine_people_with_no_routine"]
    with pytest.raises(ValueError, match="has no setting"):
        resolve_no_routine_share_range("training", config)


def test_idle_decision_follows_the_share() -> None:
    """Checks a share of 1 makes every draw idle, a share of 0 makes none idle, and each decision reads one number."""
    rng = np.random.default_rng(2)
    assert all(is_idle(1.0, rng) for _ in range(20))
    assert not any(is_idle(0.0, rng) for _ in range(20))

    used = np.random.default_rng(9)
    first_number = float(np.random.default_rng(9).uniform(0.0, 1.0))
    assert is_idle(0.5, used) == (first_number < 0.5)
    # Exactly one number was read: the next number is the generator's second one.
    second_number = float(np.random.default_rng(9).uniform(0.0, 1.0, size=2)[1])
    assert float(used.uniform(0.0, 1.0)) == second_number


def test_runner_config_has_the_new_streams_and_tie_break_rank() -> None:
    """Checks children 10, 11 and 12 are configured with reasons and the routine rank is last."""
    runner_config = load_runner_config()

    children = runner_config["random_stream_children"]
    assert (children["opened_account_routine"], children["opened_routine_failures"]) == (10, 11)
    assert children["rented_owner_activity"] == 12
    assert runner_config["tie_break_order"][-1] == "opened_routine_payments"
    assert runner_config["tie_break_order"][:5] == [
        "outside_events", "planned_payments", "fraud_payments", "resumed_payments", "gang_device_events",
    ]


# ---- one person's routine -------------------------------------------------------------------------------------


def persons_by_persona() -> dict[str, Any]:
    """Returns the first person of each persona in the seed-101 showcase world."""
    settings, _ = get_showcase_b_settings()
    found: dict[str, Any] = {}
    for person in prepare_world(settings, None).world.people:
        found.setdefault(person.persona_name, person)
    return found


def test_routine_of_one_person_starts_at_the_opening_minute_and_uses_level_11_generators() -> None:
    """Plans a salaried person's routine and checks timing, ownership, salary from the employer and determinism."""
    settings, _ = get_showcase_b_settings()
    prepared = prepare_world(settings, None)
    person = persons_by_persona()["salaried_employee"]
    open_minute = 20 * DAY

    payments, next_index = plan_opened_routine(
        prepared.world, prepared.plan, person, open_minute, np.random.default_rng(4), 500
    )

    assert payments and next_index > 500
    assert all(p.minute >= open_minute for p in payments)
    assert all(person.account_id in (p.from_account, p.to_account) for p in payments)
    assert {p.purpose for p in payments} >= {"everyday", "salary"}
    employer = next(shop for shop in prepared.world.shops if shop.shop_id == person.employer_shop_id)
    assert any(p.purpose == "salary" and p.from_account == employer.account_id for p in payments)
    again, _ = plan_opened_routine(prepared.world, prepared.plan, person, open_minute, np.random.default_rng(4), 500)
    assert payments == again


def test_lookalike_person_also_receives_peer_transfers() -> None:
    """Checks a lookalike persona gets small inbound peer transfers from other people on top of everyday payments."""
    settings, _ = get_showcase_b_settings()
    prepared = prepare_world(settings, None)
    people = persons_by_persona()
    lookalike_persona = next(name for name in people if get_persona(name).looks_like_mule_in_normal_life)
    # A copy with its own account, like a real new person who is not yet in world.people.
    person = dataclasses.replace(people[lookalike_persona], person_id="person_new", account_id="acc_person_new")

    payments, _ = plan_opened_routine(prepared.world, prepared.plan, person, 0, np.random.default_rng(4), 0)

    inbound = [p for p in payments if p.purpose == "lookalike_in"]
    assert inbound and all(p.to_account == person.account_id and p.from_account != person.account_id for p in inbound)


# ---- whole worlds ---------------------------------------------------------------------------------------------


def test_idle_people_have_no_payments(idle_output: Any) -> None:
    """Checks that with every new person idle no genuine event involves a newly opened person."""
    new_people = opened_person_ids(idle_output)

    assert len(new_people) >= 10
    assert not [e for e in planned_events(idle_output) if touches(e, new_people)]


def test_routine_people_pay_only_from_their_opening_minute(routine_output: Any) -> None:
    """Checks every new person with enough days left has payments, and none is dated before their opening."""
    opened_at = {row[1]: row[0] for row in routine_output.result.opened_accounts if row[2] == "person"}
    events = planned_events(routine_output)

    for account, minute in opened_at.items():
        mine = [e for e in events if touches(e, {account})]
        assert all(e.minute >= minute for e in mine)
        if minute < 40 * DAY:
            assert mine, f"{account} opened at minute {minute} has no payments"


def test_a_new_salaried_person_gets_salary_from_their_employer_shop(routine_output: Any) -> None:
    """Checks some new person receives a salary payment from a shop account."""
    new_people = opened_person_ids(routine_output)
    result = routine_output.result
    run_result = result.run_result
    salaries = []
    for event, source, purpose in zip(run_result.events, result.event_sources, run_result.event_purposes):
        if source == "planned" and purpose == "salary" and event.to_account in new_people:
            salaries.append(event)

    assert salaries
    assert all(event.from_account.startswith("acc_shop_") for event in salaries)


def test_routines_do_not_shift_any_existing_genuine_payments_draws(idle_output: Any, routine_output: Any) -> None:
    """
    Compares the drawn failures of every genuine payment that does not involve a new person with the all-idle baseline.
    They must be identical, so the routine payments' own streams (children 10 and 11) took nothing from Level 11's.
    Insufficient balance counts as money flow because lookalike senders now pay out money they did not before.
    """
    new_people = opened_person_ids(routine_output)

    existing_with_routines = [drawn_outcome(e) for e in planned_events(routine_output) if not touches(e, new_people)]
    existing_in_baseline = [drawn_outcome(e) for e in planned_events(idle_output) if not touches(e, new_people)]

    assert existing_with_routines == existing_in_baseline
    assert len(existing_in_baseline) > 1000


def test_routine_payments_are_recorded_as_genuine(routine_output: Any) -> None:
    """Checks routine payments carry the genuine source and use Level 11's payment purposes only."""
    new_people = opened_person_ids(routine_output)
    result = routine_output.result
    sources = {
        source for event, source in zip(result.run_result.events, result.event_sources) if touches(event, new_people)
    }
    purposes = {
        purpose for event, purpose in zip(result.run_result.events, result.run_result.event_purposes)
        if touches(event, new_people)
    }

    assert sources == {"planned"}
    assert purposes <= {"salary", "rent", "bill", "everyday", "lookalike_in"}


def test_gang_opened_accounts_never_get_routine_payments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs a gang whose mules are all new accounts beside routines and checks only fraud payments touch them."""
    monkeypatch.setattr(
        mule_origins, "get_mule_origin_share_ranges",
        lambda split: {"rented": (0.0, 0.0), "borrowed_kyc": (1.0, 1.0), "business": (1.0, 1.0)},
    )
    steps = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "world_ranges"))),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    gang = GangConfig("g1", 1, GangBudget(50_000_000, 500_000, 0.99), None, steps, DAY)

    output = run_openings((0.0, 0.0), gang)

    gang_accounts = {role.account_id for role in output.result.truth.roles if role.role == "mule"}
    assert len(gang_accounts) == 4
    result = output.result
    sources = {
        source for event, source in zip(result.run_result.events, result.event_sources) if touches(event, gang_accounts)
    }
    assert sources == {"fraud"}


def test_switch_off_gives_a_genuine_opening_no_routine() -> None:
    """With the switch off, an opening supplied as an outside event still gets no routine payments."""
    settings, code_version = get_showcase_b_settings()
    opening = OutsideEvent(DAY, "open_account", owner_kind="person", opening_stream="genuine_account_openings")

    output = run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True), fraud_inputs=FraudInputs(outside_events=(opening,)),
        code_version=code_version,
    )

    new_person = {row[1] for row in output.result.opened_accounts}
    assert len(new_person) == 1
    assert not [e for e in planned_events(output) if touches(e, new_person)]
