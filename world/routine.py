"""
WHAT: Scheduled routine life generation (salaries, bills, rent, daily spend) and minimal month runner.
It generates recurring genuine transactions so accounts develop authentic financial histories,
and look-alike personas accumulate high-velocity incoming transfers from diverse senders.
It also provides a chronological month runner executing these scheduled payments across bank ledgers.

WHY: Realistic transaction histories are essential for testing fraud detection models under realistic noise.
Simulating monthly economic activity verifies that balance invariants and regulatory limits hold over time.
Providing authentic financial flows establishes a clear behavioral contrast between genuine citizens and mules.

ISSUE IDs: E2 (spending amounts), A2 (merchant payments).
TERM: routine item: A planned financial payment schedule entry representing routine life activities. For example, a scheduled salary deposit or monthly utility bill.
TERM: chronological scheduling: Ordering events strictly by simulated minute timestamps so payments process in true temporal order. For example, processing a 9:00 AM transfer before a 10:00 AM transfer.
TERM: mule look-alike camouflage: Inbound low-value transfers from varied accounts mimicking regular consumer interactions. For example, receiving 25 small peer-to-peer transfers across the month.
TERM: salary window: The calendar day interval (days 1 to 5) during which employers disburse monthly wages. For example, paying employees on day 3 of the month.
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
    It holds metadata regarding the participant, transaction kind, calendar day, amount, counterparty, and direction.
    The simulation engine collates these items across all citizens to execute chronological payments.

    WHY: Queues recurring transactions to be ordered by time and executed through the payment engine.
    Decoupling routine plan generation from immediate execution allows global chronological sorting.
    Encapsulating transaction attributes into a single dataclass standardizes payment processing logic.

    Args:
        person_id: String identifier of the person executing or receiving the transaction.
        kind: Category descriptor string ('salary_in', 'rent_out', 'bill_out', 'everyday_spend', 'lookalike_in').
        day_of_month: Calendar day (1 through 28) on which the transaction is scheduled to occur.
        amount_paise: Positive integer currency amount in paise to be transferred.
        counterparty_account_id: Account identifier of the merchant, employer, or peer counterparty.
        is_inbound: Boolean flag where True indicates money entering the person's account, and False indicates outbound expense.

    Returns:
        Initialized RoutineItem instance.

    Example:
        # Create a routine item for monthly electric bill:
        # item = RoutineItem("person_000000", "bill_out", 5, 2000, "acc_shop_0001", False)
        # assert item.kind == "bill_out" and item.amount_paise == 2000
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
    It samples utility shops and scheduled billing days within the configured calendar window.
    Each bill payment is scaled using the configured BILL_MULTIPLE against the person's median spend.

    WHY: Separates recurring monthly household utility bills into a distinct helper.
    Isolating utility obligations allows different demographic personas to customize fixed billing frequencies.
    Using configured day windows mirrors real utility invoicing practices where bills arrive mid-month.

    Args:
        person: Person instance whose median transaction profile scales the utility amount.
        utilities: List of commercial Shop instances classified under the 'utility' category.
        rng: Initialized NumPy random generator for deterministic counterparty and date selection.

    Returns:
        List of RoutineItem objects representing monthly utility payments.

    Example:
        # Generate bill items for a citizen:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # utils = [s for s in w.shops if s.kind == "utility"]
        # items = build_bill_items(w.people[0], utils, np.random.default_rng(42))
        # assert len(items) == 2
    """
    items: list[RoutineItem] = []
    # Scale utility bill amount using configured multiple of median spending.
    med = person.amount_profile.median_paise
    for _ in range(BILLS_PER_MONTH):
        if utilities:
            # Sample utility merchant and billing day within configured monthly window.
            u_shop = rng.choice(utilities)
            b_day = int(rng.integers(BILL_DAY_WINDOW[0], BILL_DAY_WINDOW[1], endpoint=True))
            items.append(
                RoutineItem(person.person_id, "bill_out", b_day, BILL_MULTIPLE * med, u_shop.account_id, False)
            )
    return items


def build_salaried_items(person: Person, shops: list[Shop], rng: np.random.Generator) -> list[RoutineItem]:
    """
    WHAT: Creates salary deposit, residential rent, and utility bill routine items for salaried employees.
    It links employment income to the person's assigned employer shop and schedules monthly rent to landlords.
    It also appends recurring household utility obligations to complete the employee's fixed budget.

    WHY: Implements typical monthly fixed cash flow obligations for working households.
    Salaried workers exhibit structured monthly cash flows characterized by regular payroll and living costs.
    Modeling fixed cash flows provides a realistic baseline against which unexpected fraud anomalies stand out.

    Args:
        person: Person instance representing the salaried employee.
        shops: List of all commercial Shop instances in the simulated town.
        rng: Initialized NumPy random generator for deterministic scheduling.

    Returns:
        List of RoutineItem objects capturing the employee's fixed monthly cash flows.

    Example:
        # Generate fixed items for a citizen:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # items = build_salaried_items(w.people[0], w.shops, np.random.default_rng(42))
        # assert len(items) == 3
    """
    items: list[RoutineItem] = []
    med = person.amount_profile.median_paise

    # Schedule monthly salary disbursement from employer shop if employed.
    if person.employer_shop_id:
        employer = next(s for s in shops if s.shop_id == person.employer_shop_id)
        s_day = int(rng.integers(SALARY_DAY_WINDOW[0], SALARY_DAY_WINDOW[1], endpoint=True))
        items.append(
            RoutineItem(person.person_id, "salary_in", s_day, SALARY_MULTIPLE * med, employer.account_id, True)
        )

    # Schedule monthly residential rent payment to a randomly chosen landlord.
    landlords = [s for s in shops if s.kind == "landlord"]
    if landlords:
        landlord = rng.choice(landlords)
        r_day = int(rng.integers(RENT_DAY_WINDOW[0], RENT_DAY_WINDOW[1], endpoint=True))
        items.append(
            RoutineItem(person.person_id, "rent_out", r_day, RENT_MULTIPLE * med, landlord.account_id, False)
        )

    # Schedule recurring monthly utility bills.
    utilities = [s for s in shops if s.kind == "utility"]
    items.extend(build_bill_items(person, utilities, rng))
    return items


def build_casual_and_lookalike_items(
    person: Person, shops: list[Shop], all_people: list[Person], rng: np.random.Generator
) -> list[RoutineItem]:
    """
    WHAT: Generates everyday store spending and small incoming transfers for mule look-alikes.
    It samples random merchant purchases across the month according to the person's spend profile.
    For personas designated as mule look-alikes, it schedules frequent small inbound transfers from diverse citizens.

    WHY: Implements daily consumer payments and provides genuine camouflage for look-alike accounts.
    Authentic payment networks contain high-volume casual shopping that introduces realistic transactional noise.
    Mule look-alike camouflage challenges fraud detection models by testing whether they can separate genuine high-volume accounts from illicit mules.

    Args:
        person: Person instance whose profile governs purchase amounts and look-alike status.
        shops: List of all Shop entities available in the town for casual retail spending.
        all_people: List of all Person entities in the world available as counterparty senders.
        rng: Initialized NumPy random generator for reproducible transaction sampling.

    Returns:
        List of RoutineItem objects containing casual expenditures and camouflage receipts.

    Example:
        # Generate casual items for a citizen:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # items = build_casual_and_lookalike_items(w.people[0], w.shops, w.people, np.random.default_rng(42))
        # assert len(items) == 15
    """
    items: list[RoutineItem] = []
    med = person.amount_profile.median_paise

    # Generate routine daily consumer retail purchases across local shops.
    for _ in range(EVERYDAY_PAYMENTS_PER_MONTH):
        shop = rng.choice(shops)
        amt = draw_amount_paise(person.amount_profile, rng)
        day = int(rng.integers(1, 28, endpoint=True))
        items.append(RoutineItem(person.person_id, "everyday_spend", day, amt, shop.account_id, False))

    # Generate camouflage incoming peer transfers for mule look-alike accounts.
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
    It checks the person's demographic persona and aggregates their appropriate lifestyle agenda.
    The returned list contains all routine financial events scheduled across the month.

    WHY: Assembles a complete monthly lifestyle agenda for the individual.
    Consolidating payment generation into one routine builder guarantees that each citizen receives a coherent schedule.
    It provides a modular structure allowing varied personas to generate specialized financial routines.

    Args:
        person: Person instance whose complete monthly agenda is being compiled.
        shops: Master list of all commercial Shop entities in the simulated world.
        all_people: Master list of all citizen Person entities in the simulated world.
        rng: Initialized NumPy random generator ensuring reproducible schedule generation.

    Returns:
        List of all scheduled RoutineItem objects for the person's monthly routine.

    Example:
        # Compile complete routine for a citizen:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # routine = build_routine(w.people[0], w.shops, w.people, np.random.default_rng(42))
        # assert len(routine) == 15
    """
    items: list[RoutineItem] = []
    # Append fixed payroll and living expenses for salaried employee personas.
    if person.persona_name == "salaried_employee":
        items.extend(build_salaried_items(person, shops, rng))
    # Append casual retail spending and any applicable mule camouflage transfers.
    items.extend(build_casual_and_lookalike_items(person, shops, all_people, rng))
    return items


def execute_single_item(
    world: Any, it: RoutineItem, person_acc: str, minute: int, results: dict[str, int]
) -> None:
    """
    WHAT: Splits and executes a single routine payment leg, tallying the outcome.
    It resolves directional account routing between personal accounts and external counterparties.
    It splits oversized transfers into compliant legal legs and submits each leg to the payment engine.

    WHY: Handles directional routing between personal accounts and counterparties.
    Ensuring payments respect the sending bank's regulatory UPI limit maintains legal compliance.
    Updating a results tally mapping provides operational statistics on transaction success and failure rates.

    Args:
        world: Root World container holding all bank ledgers and financial state.
        it: RoutineItem instance specifying the amount, counterparty, and transfer direction.
        person_acc: Account identifier of the person executing the routine item.
        minute: Simulation minute timestamp when the transfer takes place.
        results: Mutable dictionary tallying transaction outcome reason counts.

    Returns:
        None.

    Example:
        # Execute an individual routine payment item:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # item = RoutineItem("p0", "everyday_spend", 1, 1000, w.shops[0].account_id, False)
        # res = {}; execute_single_item(w, item, w.people[0].account_id, 600, res)
        # assert res == {"ok": 1}
    """
    # Determine source and destination accounts based on transfer direction.
    from_acc = it.counterparty_account_id if it.is_inbound else person_acc
    to_acc = person_acc if it.is_inbound else it.counterparty_account_id

    # Retrieve the sending bank's UPI limit and split transfer into legal legs.
    sender_l = find_ledger_for_account(world.ledgers, from_acc)
    limit = sender_l.upi_limit_paise if sender_l else 10000000
    legs = split_into_legal_payments(it.amount_paise, limit)

    # Submit each payment leg to the atomic payment engine and tally outcome.
    for leg in legs:
        res = make_payment(world, from_acc, to_acc, leg, minute)
        results[res.reason] = results.get(res.reason, 0) + 1


def run_one_month(world: Any, rng: np.random.Generator) -> dict[str, int]:
    """
    WHAT: Runs a minimal one-month cycle of routine payments across all simulated citizens.
    It builds monthly schedules for every person, assigns timestamps, and sorts them chronologically.
    It executes all scheduled payments sequentially and returns the aggregate summary of payment outcomes.

    WHY: Exercises all ledgers with realistic concurrent transaction traffic to test invariants.
    Simulating an entire calendar cycle verifies that accounts remain solvent and ledgers balance under load.
    Chronological execution guarantees that income deposits precede subsequent consumption expenses realistically.

    Args:
        world: Complete simulated World container populated with banks, shops, and citizens.
        rng: Initialized NumPy random generator for timestamp jitter and execution randomization.

    Returns:
        Dictionary mapping payment outcome reason strings ('ok', 'insufficient_balance', etc.) to occurrence counts.

    Example:
        # Simulate one month of routine transactions across a 10-person world:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # results = run_one_month(w, np.random.default_rng(42))
        # assert results.get("ok", 0) == 266
    """
    timed_items: list[tuple[int, RoutineItem, str]] = []

    # Compile and timestamp routine payment items across all citizens.
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
    # Execute each scheduled routine item in chronological sequence.
    for minute, item, acc in timed_items:
        execute_single_item(world, item, acc, minute, results)

    return results

