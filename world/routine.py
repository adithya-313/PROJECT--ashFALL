"""
WHAT: Scheduled routine life generation (salaries, bills, rent, daily spend) and minimal month runner.
WHY: Generates recurring genuine transactions so accounts develop authentic financial histories,
     and look-alike personas accumulate high-velocity incoming transfers from diverse senders.
ISSUE IDs: E2 (spending amounts), A2 (merchant payments).
TERM: routine item: A planned financial payment schedule entry representing routine life activities.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.amounts import draw_amount_paise, lookalike_small_amount_paise, split_into_legal_payments
from world.payments import find_ledger_for_account, make_payment
from world.people import Person
from world.personas import get_persona
from world.placeholders import (
    BILL_DAY_WINDOW,
    BILL_MULTIPLE,
    BILLS_PER_MONTH,
    EVERYDAY_PAYMENTS_PER_MONTH,
    LOOKALIKE_SMALL_INCOMING_PER_MONTH,
    RENT_DAY_WINDOW,
    RENT_MULTIPLE,
    ROUTINE_MINUTE_OF_DAY,
    SALARY_DAY_WINDOW,
    SALARY_MULTIPLE,
)
from world.shops import Shop


@dataclass
class RoutineItem:
    """
    WHAT: Represents a scheduled payment obligation or income event in a person's monthly routine.
    WHY: Queues recurring transactions to be ordered by time and executed through the payment engine.
    """

    person_id: str
    kind: str
    day_of_month: int
    amount_paise: int
    counterparty_account_id: str
    is_inbound: bool


def build_bill_items(person: Person, utilities: list[Shop], rng: np.random.Generator) -> list[RoutineItem]:
    """
    WHAT: Creates routine bill payment items for utility expenses.
    WHY: Separates recurring monthly household utility bills into a distinct helper.
    """
    items: list[RoutineItem] = []
    med = person.amount_profile.median_paise
    for _ in range(BILLS_PER_MONTH):
        if utilities:
            u_shop = rng.choice(utilities)
            b_day = int(rng.integers(BILL_DAY_WINDOW[0], BILL_DAY_WINDOW[1], endpoint=True))
            items.append(
                RoutineItem(person.person_id, "bill_out", b_day, BILL_MULTIPLE * med, u_shop.account_id, False)
            )
    return items


def build_salaried_items(person: Person, shops: list[Shop], rng: np.random.Generator) -> list[RoutineItem]:
    """
    WHAT: Creates salary deposit, residential rent, and utility bill routine items for salaried employees.
    WHY: Implements typical monthly fixed cash flow obligations for working households.
    """
    items: list[RoutineItem] = []
    med = person.amount_profile.median_paise

    if person.employer_shop_id:
        employer = next(s for s in shops if s.shop_id == person.employer_shop_id)
        s_day = int(rng.integers(SALARY_DAY_WINDOW[0], SALARY_DAY_WINDOW[1], endpoint=True))
        items.append(
            RoutineItem(person.person_id, "salary_in", s_day, SALARY_MULTIPLE * med, employer.account_id, True)
        )

    landlords = [s for s in shops if s.kind == "landlord"]
    if landlords:
        landlord = rng.choice(landlords)
        r_day = int(rng.integers(RENT_DAY_WINDOW[0], RENT_DAY_WINDOW[1], endpoint=True))
        items.append(
            RoutineItem(person.person_id, "rent_out", r_day, RENT_MULTIPLE * med, landlord.account_id, False)
        )

    utilities = [s for s in shops if s.kind == "utility"]
    items.extend(build_bill_items(person, utilities, rng))
    return items


def build_casual_and_lookalike_items(
    person: Person, shops: list[Shop], all_people: list[Person], rng: np.random.Generator
) -> list[RoutineItem]:
    """
    WHAT: Generates everyday store spending and small incoming transfers for mule look-alikes.
    WHY: Implements daily consumer payments and provides genuine camouflage for look-alike accounts.
    """
    items: list[RoutineItem] = []
    med = person.amount_profile.median_paise

    for _ in range(EVERYDAY_PAYMENTS_PER_MONTH):
        shop = rng.choice(shops)
        amt = draw_amount_paise(person.amount_profile, rng)
        day = int(rng.integers(1, 28, endpoint=True))
        items.append(RoutineItem(person.person_id, "everyday_spend", day, amt, shop.account_id, False))

    persona = get_persona(person.persona_name)
    if persona.looks_like_mule_in_normal_life:
        small_amt = lookalike_small_amount_paise(med)
        others = [p for p in all_people if p.person_id != person.person_id]
        if others:
            senders = rng.choice(others, size=LOOKALIKE_SMALL_INCOMING_PER_MONTH, replace=True)
            for s in senders:
                day = int(rng.integers(1, 28, endpoint=True))
                items.append(RoutineItem(person.person_id, "lookalike_in", day, small_amt, s.account_id, True))

    return items


def build_routine(
    person: Person, shops: list[Shop], all_people: list[Person], rng: np.random.Generator
) -> list[RoutineItem]:
    """
    WHAT: Compiles all scheduled income, fixed expenses, and store payments for one person.
    WHY: Assembles a complete monthly lifestyle agenda for the individual.
    """
    items: list[RoutineItem] = []
    if person.persona_name == "salaried_employee":
        items.extend(build_salaried_items(person, shops, rng))
    items.extend(build_casual_and_lookalike_items(person, shops, all_people, rng))
    return items


def execute_single_item(
    world: Any, it: RoutineItem, person_acc: str, minute: int, results: dict[str, int]
) -> None:
    """
    WHAT: Splits and executes a single routine payment leg, tallying the outcome.
    WHY: Handles directional routing between personal accounts and counterparties.
    """
    from_acc = it.counterparty_account_id if it.is_inbound else person_acc
    to_acc = person_acc if it.is_inbound else it.counterparty_account_id

    sender_l = find_ledger_for_account(world.ledgers, from_acc)
    limit = sender_l.upi_limit_paise if sender_l else 10000000
    legs = split_into_legal_payments(it.amount_paise, limit)

    for leg in legs:
        res = make_payment(world, from_acc, to_acc, leg, minute)
        results[res.reason] = results.get(res.reason, 0) + 1


def run_one_month(world: Any, rng: np.random.Generator) -> dict[str, int]:
    """
    WHAT: Runs a minimal one-month cycle of routine payments across all simulated citizens.
    WHY: Exercises all ledgers with realistic concurrent transaction traffic to test invariants.
    """
    timed_items: list[tuple[int, RoutineItem, str]] = []

    for person in world.people:
        items = build_routine(person, world.shops, world.people, rng)
        for it in items:
            if it.kind in ("salary_in", "rent_out", "bill_out"):
                m = (it.day_of_month - 1) * 1440 + ROUTINE_MINUTE_OF_DAY
            else:
                m = (it.day_of_month - 1) * 1440 + int(rng.integers(0, 1440))
            timed_items.append((m, it, person.account_id))

    # Sort strictly by timestamp so payments occur chronologically.
    timed_items.sort(key=lambda x: x[0])
    results: dict[str, int] = {}
    for minute, item, acc in timed_items:
        execute_single_item(world, item, acc, minute, results)

    return results
