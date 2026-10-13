"""
WHAT: Calendar-driven payment simulation engine coordinating temporal rhythms, failure routing, and event auditing.
WHY: Replaces 11A fixed-day payment runner with an authentic astronomical timeline that records payment attempts.
TERM: calendar runner: An execution engine that schedules and runs financial transactions across simulated time.
TERM: payment event: An immutable record documenting transaction outcome, financial routing, and failure domains.
TERM: failure domain: The operational zone responsible for a payment rejection (central, banks, app, or user).
TERM: partial degradation: A temporary drop in transaction success probability without complete service blackout.
TERM: diurnal rhythm: Periodic fluctuations in human activity recurring over a 24-hour cycle.
TERM: vectorised: Mathematical operations applied across entire arrays simultaneously using optimized library routines.
TERM: Poisson: A discrete probability distribution expressing the count of events occurring in a fixed interval of time.
TERM: stable sort: A sorting algorithm that preserves the relative input order of records with equal keys.
"""

# Import Python standard library calendar utilities for Gregorian month lengths.
import calendar
from dataclasses import dataclass
from typing import Any

# Import high-performance numerical library for array operations and random distributions.
import numpy as np

# Import domain modules for currency limits, failures, ledgers, and temporal schedules.
from world.amounts import draw_amount_paise, lookalike_small_amount_paise, split_into_legal_payments
from world.failure_decision import decide_failure
from world.failures import PaymentEvent, make_event
from world.ledger import Ledger
from world.payments import find_ledger_for_account, make_payment
from world.personas import get_persona
from world.placeholders import (
    EVERYDAY_PAYMENTS_PER_MONTH,
    LOOKALIKE_SMALL_INCOMING_PER_MONTH,
    ROUTINE_MINUTE_OF_DAY,
)
from world.rhythms import daily_multiplier, hour_weight
from world.routine import build_salaried_items
from world.world_plan import WorldPlan

PAYMENT_PURPOSES: tuple[str, ...] = ("salary", "rent", "bill", "everyday", "lookalike_in")
ROUTINE_TYPE_TO_PURPOSE: dict[str, str] = {
    "salary_in": "salary",
    "rent_out": "rent",
    "bill_out": "bill",
}


@dataclass(frozen=True)
class PlannedPayment:
    """
    WHAT: Represents a scheduled payment attempt before execution and failure evaluation.
    WHY: Queues chronological candidate payments with stable sequence ordering across the town.
    """

    minute: int
    from_account: str
    to_account: str
    amount_paise: int
    creation_index: int
    purpose: str


@dataclass(frozen=True)
class RunResult:
    """
    WHAT: Encapsulates all generated payment events and diagnostic count aggregates from a simulation run.
    WHY: Delivers an audit trail of successful and failed attempts for downstream evaluation.
    """

    events: list[PaymentEvent]
    counts: dict[str, int]
    event_purposes: list[str]


def _build_account_bank_map(ledgers: list[Ledger]) -> dict[str, int]:
    """
    WHAT: Maps every registered account identifier to its owning bank institution index.
    WHY: Provides constant-time bank resolution for payment routing and incident checks across the world.
    Args: ledgers: List of active bank ledgers in the simulated world containing registered accounts.
    Returns: Dictionary mapping account identifier strings to integer bank indices.
    Example: bank_map = _build_account_bank_map(world.ledgers); assert bank_map['acc_person_1'] >= 0
    """
    # Allocate mapping table for fast O(1) bank index resolution.
    account_map: dict[str, int] = {}
    # Iterate across each bank ledger to map managed account identifiers.
    for ledger in ledgers:
        # Extract the owning bank index for the current ledger.
        b_idx = ledger.bank_index
        # Assign bank index to each account registered in this ledger.
        for account_id in ledger.accounts:
            account_map[account_id] = b_idx
    # Return the completed constant-time lookup dictionary.
    return account_map


def _collect_monthly_windows(plan: WorldPlan) -> list[tuple[int, int, int]]:
    """
    WHAT: Identifies all distinct calendar months and their starting day offsets within the world horizon.
    WHY: Aligns recurring monthly obligations such as salaries and utility bills to true Gregorian months.
    Args: plan: The active WorldPlan specifying horizon duration and simulation clock start date.
    Returns: List of tuples containing calendar year, month number, and starting world day offset.
    Example: windows = _collect_monthly_windows(plan); assert len(windows) >= 1 and windows[0][2] == 0
    """
    # Storage list for chronological calendar month records.
    windows: list[tuple[int, int, int]] = []
    # Set of observed year-month tuples to detect boundary transitions.
    seen_months: set[tuple[int, int]] = set()
    # Step through every day of the simulation horizon to find unique months.
    for day in range(plan.n_days):
        # Translate day index into a calendar datetime using the simulation clock.
        dt = plan.clock.to_datetime(day * 1440)
        ym = (dt.year, dt.month)
        # Register new month occurrence and track its initial world day offset.
        if ym not in seen_months:
            seen_months.add(ym)
            windows.append((dt.year, dt.month, day))
    # Return chronological month start positions.
    return windows


def _fixed_for_person(
    person: Any, shops: list[Any], rng: np.random.Generator, first_day: int, n_days: int, start_idx: int
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Generates salary deposit, rent, and utility bill routine items for one salaried person in one month.
    WHY: Converts monthly routine obligations into timestamped candidate payments bounded by simulation horizon.
    Args: person: Citizen entity; shops: merchants; rng: generator; first_day: day offset; n_days: cap; start_idx: idx.
    Returns: Tuple of generated PlannedPayment records and the updated sequence counter.
    Example: items, n_idx = _fixed_for_person(p, shops, rng, 0, 61, 0); assert isinstance(items, list)
    """
    # Output list collecting scheduled routine payments for this employee.
    items: list[PlannedPayment] = []
    idx = start_idx
    # Call existing routine builder to obtain monthly living expense items.
    for item in build_salaried_items(person, shops, rng):
        # Calculate absolute simulation minute using month day offset.
        minute = (first_day + item.day_of_month - 1) * 1440 + ROUTINE_MINUTE_OF_DAY
        # Verify that the generated timestamp falls within the simulation horizon.
        if minute < n_days * 1440:
            # Map sender and receiver accounts according to inbound cash flow flag.
            from_acc = item.counterparty_account_id if item.is_inbound else person.account_id
            to_acc = person.account_id if item.is_inbound else item.counterparty_account_id
            purpose = ROUTINE_TYPE_TO_PURPOSE[item.kind]
            # Append planned payment and increment stable sequence counter.
            items.append(PlannedPayment(minute, from_acc, to_acc, item.amount_paise, idx, purpose))
            idx += 1
    return items, idx


def _generate_fixed_items(
    world: Any, plan: WorldPlan, rng: np.random.Generator, start_idx: int
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Generates scheduled salary, rent, and utility bill payments for all salaried citizens.
    WHY: Ensures fixed monthly living expenses execute on calendar days regardless of festival multipliers.
    Args: world: Container of citizens and shops; plan: WorldPlan; rng: generator; start_idx: counter.
    Returns: Tuple containing list of PlannedPayment records and the updated sequence counter.
    Example: items, n_idx = _generate_fixed_items(world, plan, rng, 0); assert len(items) >= 0
    """
    # Aggregate container holding all fixed scheduled items.
    items: list[PlannedPayment] = []
    idx = start_idx
    # Find all distinct Gregorian months that overlap the world horizon.
    windows = _collect_monthly_windows(plan)
    # Filter demographic population down to salaried employees.
    salaried = [p for p in world.people if p.persona_name == "salaried_employee"]
    # Iterate across each calendar month window and each salaried citizen.
    for _year, _month, first_day in windows:
        for person in salaried:
            # Generate items for this person and advance stable counter.
            new_items, idx = _fixed_for_person(person, world.shops, rng, first_day, plan.n_days, idx)
            items.extend(new_items)
    return items, idx


def _precompute_persona_weights(plan: WorldPlan) -> dict[str, np.ndarray]:
    """
    WHAT: Precomputes normalized 24-hour diurnal activity probabilities for each persona.
    WHY: Optimizes payment timestamp sampling by vectorizing hour selection across days.
    Args: plan: WorldPlan supplying temporal rhythm settings and persona peak hours.
    Returns: Dictionary mapping persona string names to normalized 24-element probability arrays.
    Example: weights = _precompute_persona_weights(plan); assert len(weights['student']) == 24
    """
    # Lookup table storing discrete hourly probability density functions.
    weights: dict[str, np.ndarray] = {}
    personas = ("salaried_employee", "student", "retiree", "small_shopkeeper", "gig_worker")
    # Evaluate hour weights at the midpoint of each hour (h + 0.5) per specification.
    for persona in personas:
        # DECISION: evaluating at h + 0.5 means the middle of the hour.
        raw = [hour_weight(h + 0.5, persona, plan.rhythm_settings) for h in range(24)]
        # Normalize weights so probabilities sum strictly to 1.0.
        weights[persona] = np.array(raw, dtype=float) / sum(raw)
    return weights


def _sample_single_everyday(
    person: Any, shops: list[Any], day: int, probs: dict[str, np.ndarray], rng: np.random.Generator, idx: int
) -> PlannedPayment:
    """
    WHAT: Samples an individual everyday store payment for a person on a specific day.
    WHY: Selects payment hour via persona diurnal weights, picks random shop, and draws transaction amount.
    Args: person: Citizen spending; shops: catalog; day: day index; probs: weights; rng: generator; idx: counter.
    Returns: PlannedPayment instance representing the scheduled commercial purchase.
    Example: pmt = _sample_single_everyday(person, shops, 0, probs, rng, 1); assert pmt.amount_paise > 0
    """
    # Sample payment hour using persona diurnal curve.
    hour = int(rng.choice(24, p=probs[person.persona_name]))
    # Pick a random minute uniformly within the selected hour.
    minute = day * 1440 + hour * 60 + int(rng.integers(0, 60))
    # Choose destination merchant establishment uniformly from catalog.
    shop = rng.choice(shops)
    # Draw purchase amount from person log-normal spending distribution.
    amt = draw_amount_paise(person.amount_profile, rng)
    return PlannedPayment(minute, person.account_id, shop.account_id, amt, idx, "everyday")


def _generate_day_everyday_payments(
    world: Any, plan: WorldPlan, day: int, probs: dict[str, np.ndarray], rng: np.random.Generator, start_idx: int
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Samples casual everyday commercial store transactions for all citizens on one day.
    WHY: Drives everyday commerce using calendar, festival, weekend, and diurnal multipliers.
    Args: world: World container; plan: WorldPlan; day: index; probs: weights; rng: generator; start_idx: counter.
    Returns: Tuple containing generated PlannedPayment records and advanced sequence counter.
    Example: pmts, n_idx = _generate_day_everyday_payments(world, plan, 0, probs, rng, 0); assert n_idx >= 0
    """
    # Container collecting commercial payments generated for this day.
    payments: list[PlannedPayment] = []
    idx = start_idx
    # Resolve Gregorian date to determine month duration.
    dt = plan.clock.to_datetime(day * 1440)
    days_in_month = calendar.monthrange(dt.year, dt.month)[1]
    # Calculate baseline frequency and apply composite calendar rhythm multiplier.
    base_rate = EVERYDAY_PAYMENTS_PER_MONTH / float(days_in_month)
    boost = daily_multiplier(day * 1440, plan.clock, plan.rhythm_settings, plan.festivals)
    # Vectorized Poisson draw for transaction counts across all citizens.
    counts = rng.poisson(base_rate * boost, size=len(world.people))
    for i, count in enumerate(counts):
        # Sample each individual commercial purchase for this person.
        for _ in range(count):
            payments.append(_sample_single_everyday(world.people[i], world.shops, day, probs, rng, idx))
            idx += 1
    return payments, idx


def _sample_lookalike_inbounds(
    person: Any, others: list[Any], day: int, probs: dict[str, np.ndarray],
    count: int, rng: np.random.Generator, start_idx: int
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Samples small incoming peer transfers for one lookalike person on a simulation day.
    WHY: Generates camouflage inbound traffic where transaction timing follows sender diurnal peak.
    Args: person: Recipient; others: senders; day: day index; probs: weights; count: draws; rng: rng; start_idx: idx.
    Returns: Tuple of generated PlannedPayment records and advanced sequence counter.
    Example: inbounds, n_idx = _sample_lookalike_inbounds(p, others, 0, probs, 2, rng, 0); assert len(inbounds) == 2
    """
    # Container collecting peer transfers into lookalike account.
    payments: list[PlannedPayment] = []
    idx = start_idx
    # Scale small inbound credit amount using the shared lookalike rule.
    amt = lookalike_small_amount_paise(person.amount_profile.median_paise)
    for _ in range(count):
        # Sample a distinct sender from the community.
        sender = rng.choice(others)
        # SENDER rhythm is used because sender is the active actor initiating transfer.
        hour = int(rng.choice(24, p=probs[sender.persona_name]))
        minute = day * 1440 + hour * 60 + int(rng.integers(0, 60))
        # Record scheduled peer-to-peer inbound transfer.
        payments.append(PlannedPayment(minute, sender.account_id, person.account_id, amt, idx, "lookalike_in"))
        idx += 1
    return payments, idx


def _generate_day_lookalike_payments(
    world: Any, plan: WorldPlan, day: int, probs: dict[str, np.ndarray], rng: np.random.Generator, start_idx: int
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Samples small inbound credits for citizens who mimic mule account velocities.
    WHY: Simulates authentic high-velocity camouflaging transactions across genuine peers.
    Args: world: World container; plan: WorldPlan; day: index; probs: weights; rng: generator; start_idx: counter.
    Returns: Tuple containing generated PlannedPayment records and advanced sequence counter.
    Example: la_pmts, n_idx = _generate_day_lookalike_payments(world, plan, 0, probs, rng, 0); assert n_idx >= 0
    """
    # Container collecting lookalike camouflage payments.
    payments: list[PlannedPayment] = []
    idx = start_idx
    # Calculate daily intensity for lookalike inbound camouflage transfers.
    dt = plan.clock.to_datetime(day * 1440)
    days_in_month = calendar.monthrange(dt.year, dt.month)[1]
    base_rate = LOOKALIKE_SMALL_INCOMING_PER_MONTH / float(days_in_month)
    boost = daily_multiplier(day * 1440, plan.clock, plan.rhythm_settings, plan.festivals)
    # Filter lookalike personas and collect candidate external senders.
    for person in world.people:
        if get_persona(person.persona_name).looks_like_mule_in_normal_life:
            # Sample number of incoming transfers arriving today.
            count = int(rng.poisson(base_rate * boost))
            others = [p for p in world.people if p.person_id != person.person_id]
            # Generate transfers if candidate senders are present.
            if others and count > 0:
                new_pmts, idx = _sample_lookalike_inbounds(person, others, day, probs, count, rng, idx)
                payments.extend(new_pmts)
    return payments, idx


def _collect_all_planned_payments(
    world: Any, plan: WorldPlan, rng: np.random.Generator
) -> list[PlannedPayment]:
    """
    WHAT: Aggregates and chronologically sorts all fixed, commercial, and lookalike planned payments.
    WHY: Establishes a unified timeline ordered stably by minute and creation sequence.
    Args: world: Container of people, shops, ledgers; plan: WorldPlan; rng: generator.
    Returns: List of PlannedPayment records sorted strictly by timestamp and creation order.
    Example: planned = _collect_all_planned_payments(world, plan, rng); assert isinstance(planned, list)
    """
    # Unified list collecting all candidate transactions.
    all_payments: list[PlannedPayment] = []
    # 1. Generate recurring monthly fixed expenses and salaries.
    fixed, next_idx = _generate_fixed_items(world, plan, rng, 0)
    all_payments.extend(fixed)
    # 2. Precompute hourly diurnal probability profiles.
    probs = _precompute_persona_weights(plan)
    # 3. Iterate across horizon days to sample everyday and lookalike payments.
    for day in range(plan.n_days):
        everyday, next_idx = _generate_day_everyday_payments(world, plan, day, probs, rng, next_idx)
        all_payments.extend(everyday)
        lookalike, next_idx = _generate_day_lookalike_payments(world, plan, day, probs, rng, next_idx)
        all_payments.extend(lookalike)
    # STABLE SORT: Sorting by (minute, creation_index) guarantees identical output for identical seeds.
    all_payments.sort(key=lambda p: (p.minute, p.creation_index))
    return all_payments


def _handle_live_payment(
    world: Any, payment: PlannedPayment, leg: int, f_bank: int, t_bank: int
) -> PaymentEvent:
    """
    WHAT: Executes an approved payment leg on the bank ledgers and tags insufficient funds as customer failure.
    WHY: Preserves double-entry accounting laws and maps balance refusals directly into PaymentEvent records.
    Args: world: World container; payment: planned item; leg: amount paise; f_bank: sender bank; t_bank: receiver bank.
    Returns: PaymentEvent documenting success or customer insufficient balance decline.
    Example: ev = _handle_live_payment(world, pmt, 500, 0, 1); assert ev.status in ('success', 'failed')
    """
    # Call atomic payment transfer engine to execute double-entry update.
    res = make_payment(world, payment.from_account, payment.to_account, leg, payment.minute)
    # Return success event when funds were transferred cleanly.
    if res.success:
        return make_event(payment.minute, payment.from_account, payment.to_account, leg, f_bank, t_bank, None, "ok")
    # Tag customer failure domain when sender balance is insufficient.
    if res.reason == "insufficient_balance":
        return make_event(
            payment.minute, payment.from_account, payment.to_account, leg, f_bank, t_bank, "customer", res.reason
        )
    # Internal validation errors indicate bugs that must not be hidden.
    raise RuntimeError(f"Unexpected payment failure: {res.reason}")


def _execute_payment_leg(
    world: Any, plan: WorldPlan, payment: PlannedPayment, leg: int, bank_map: dict[str, int], rng: np.random.Generator
) -> PaymentEvent:
    """
    WHAT: Evaluates outages and executes a single UPI payment leg across bank ledgers.
    WHY: Implements rule R7 by transforming failed payments into events without balance mutations.
    Args: world: World container; plan: WorldPlan; payment: planned item; leg: amount; bank_map: lookup; rng: generator.
    Returns: PaymentEvent recording outcome, status, diagnostic reason, and failure domain.
    Example: ev = _execute_payment_leg(world, plan, pmt, 1000, bank_map, rng); assert isinstance(ev, PaymentEvent)
    """
    # Resolve owning financial institutions for both counterparties.
    f_bank = bank_map[payment.from_account]
    t_bank = bank_map[payment.to_account]
    # Check if external outage or background dropout declines the transaction.
    failure = decide_failure(payment.minute, f_bank, t_bank, plan.incidents, plan.failure_settings, rng)
    if failure is not None:
        domain, reason = failure
        # Record failed payment event immediately; no money moves and make_payment is NOT called.
        return make_event(
            payment.minute, payment.from_account, payment.to_account, leg, f_bank, t_bank, domain, reason
        )
    # Execute atomic transfer if transaction was not declined by failure engine.
    return _handle_live_payment(world, payment, leg, f_bank, t_bank)


def _record_event_counts(ev: PaymentEvent, counts: dict[str, int]) -> None:
    """
    WHAT: Updates aggregate simulation attempt counters, successes, and domain-specific failure totals.
    WHY: Produces diagnostic summary metrics for reporting and verification without post-processing logs.
    Args: ev: Processed PaymentEvent; counts: mutable tracking dictionary.
    Example: counts = {'attempts': 0, 'successes': 0, 'failures': 0}; _record_event_counts(ev, counts)
    """
    # Increment total transaction attempt counter.
    counts["attempts"] += 1
    # Tally outcomes by terminal operational status.
    if ev.status == "success":
        counts["successes"] += 1
    else:
        # Tally failures and track attribution by specific operational domain.
        counts["failures"] += 1
        counts[f"domain_{ev.failure_domain}"] = counts.get(f"domain_{ev.failure_domain}", 0) + 1


def run_world(world: Any, plan: WorldPlan, rng: np.random.Generator) -> RunResult:
    """
    WHAT: Executes calendar-driven routine life, evaluating outages and logging payment events.
    WHY: Orchestrates Level 11B-3 payment execution with strict R7 ledger invariant adherence.
    Args: world: Container of people, shops, ledgers; plan: WorldPlan; rng: generator.
    Returns: RunResult dataclass containing chronological event audit log and diagnostic tallies.
    Example: res = run_world(world, plan, rng); assert len(res.events) >= 0 and res.counts['attempts'] >= 0
    """
    # Build account to bank mapping table for fast routing resolution.
    bank_map = _build_account_bank_map(world.ledgers)
    # Assemble and sort all planned transactions across the horizon.
    planned = _collect_all_planned_payments(world, plan, rng)
    events: list[PaymentEvent] = []
    event_purposes: list[str] = []
    counts: dict[str, int] = {"attempts": 0, "successes": 0, "failures": 0}
    # Execute each planned payment, splitting by UPI transaction limit caps.
    for payment in planned:
        sender_l = find_ledger_for_account(world.ledgers, payment.from_account)
        limit = sender_l.upi_limit_paise if sender_l else 10000000
        legs = split_into_legal_payments(payment.amount_paise, limit)
        # DECISION: later legs still run if an earlier leg failed. Retries are not modelled in this piece.
        for leg in legs:
            ev = _execute_payment_leg(world, plan, payment, leg, bank_map, rng)
            events.append(ev)
            event_purposes.append(payment.purpose)
            _record_event_counts(ev, counts)
    if len(event_purposes) != len(events) or any(p not in PAYMENT_PURPOSES for p in event_purposes):
        raise RuntimeError("Validation error: event_purposes length mismatch or invalid purpose.")
    return RunResult(events, counts, event_purposes)
