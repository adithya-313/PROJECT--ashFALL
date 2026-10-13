"""
WHAT: Calendar-driven payment simulation engine coordinating temporal rhythms, failure routing, and event auditing.
WHY: Replaces 11A fixed-day payment runner with an authentic astronomical timeline that records payment attempts.
TERM: calendar runner: a program running simulated town payments minute by minute,
      e.g. paying salaries on the first and rent on the third.
TERM: payment event: a saved note recording what happened to a payment,
      e.g. whether money moved or a bank declined it.
TERM: failure domain: who caused a payment to fail,
      e.g. the customer running out of money versus a bank server crashing.
TERM: partial degradation: an outage where some payments fail while others still go through,
      e.g. a bank declining half of its transfers while approving the rest.
TERM: daily rhythm: 24-hour cycles of human activity,
      e.g. people sleeping at night and shopping during the afternoon.
TERM: vectorised: doing math on whole lists of numbers at once,
      e.g. multiplying 24 hours of weights in a single fast step.
TERM: Poisson: a way to pick a random count when you only know the average,
      e.g. a shop averaging 3 customers an hour might get 1 one hour and 5 the next.
TERM: stable sort: sorting items so ties keep their starting order,
      e.g. two payments at minute 60 stay in the order they were created.
TERM: purpose: the real-life reason for a payment,
      e.g. paying monthly rent or buying groceries at a shop.
"""

# Import Python standard library calendar utilities for Gregorian month lengths.
import calendar
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# Import high-performance numerical library for array operations and random distributions.
import numpy as np

# Import domain modules for currency limits, failures, ledgers, and temporal schedules.
from world.amounts import draw_amount_paise, lookalike_small_amount_paise, split_into_legal_payments
from world.device_events import DeviceEvent
from world.device_runtime import (
    DeviceRuntimeState,
    DeviceStamp,
    apply_device_event,
    stamp_sender_device,
    start_device_runtime,
)
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

PAYMENT_PURPOSES: tuple[str, ...] = (
    "salary", "rent", "bill", "everyday", "lookalike_in", "episode_in", "episode_out"
)
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
    WHAT: Encapsulates generated payment events, device telemetry stamps, and diagnostic count aggregates.
    WHY: Delivers an audit trail of successful and failed attempts for downstream evaluation.
    """

    events: list[PaymentEvent]
    counts: dict[str, int]
    event_purposes: list[str]
    event_sender_devices: list[DeviceStamp]
    device_counts: dict[str, int]


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


def persona_hour_probabilities(plan: WorldPlan) -> dict[str, np.ndarray]:
    """
    WHAT: Precomputes normalized 24-hour diurnal activity probabilities for each persona.
    WHY: Optimizes payment timestamp sampling by vectorizing hour selection across days.
    Args: plan: WorldPlan supplying temporal rhythm settings and persona peak hours.
    Returns: Dictionary mapping persona string names to normalized 24-element probability arrays.
    Example: weights = persona_hour_probabilities(plan); assert len(weights['student']) == 24
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
    probs = persona_hour_probabilities(plan)
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
    WHY: Implements issue R7 (failed payments are recorded but move no money)
         by creating failure events without balance mutations.
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


def _apply_due_device_events(
    device_events: Sequence[DeviceEvent], dev_idx: int, up_to_minute: int,
    p_by_id: dict[str, Any], dev_state: DeviceRuntimeState,
) -> int:
    """
    WHAT: Advances device event schedule up to specified simulation minute.
    WHY: Enforces strict chronological order where device updates precede payment execution.
    Args:
        device_events: Schedule; dev_idx: Offset; up_to_minute: Minute;
        p_by_id: Person lookup; dev_state: Runtime state tracker.
    Returns: Updated offset integer index in the device_events sequence.
    Example:
        # new_idx = _apply_due_device_events(events, 0, 100, p_map, state); assert new_idx >= 0
    """
    n_dev = len(device_events)
    # Advance pointer and apply each pending event to the target person.
    while dev_idx < n_dev and device_events[dev_idx].minute <= up_to_minute:
        dev_ev = device_events[dev_idx]
        apply_device_event(p_by_id[dev_ev.person_id], dev_ev, dev_state)
        dev_idx += 1
    return dev_idx


def _execute_single_planned(
    world: Any, plan: WorldPlan, payment: PlannedPayment, stamp: DeviceStamp,
    bank_map: dict[str, int], rng: np.random.Generator,
    events: list[PaymentEvent], purposes: list[str], stamps: list[DeviceStamp], counts: dict[str, int],
) -> None:
    """
    WHAT: Splits a planned transaction into legal UPI payment legs and records telemetry for each leg.
    WHY: Handles double-entry ledger execution and appends aligned event purposes and device stamps.
    Args:
        world: World; plan: Plan; payment: Item; stamp: DeviceStamp; bank_map: Map;
        rng: Generator; events: List; purposes: List; stamps: List; counts: Dict.
    Returns: None. Updates output lists and count dictionaries in place.
    Example:
        # _execute_single_planned(w, plan, pmt, stamp, bmap, rng, evs, ps, ss, counts)
    """
    # 1. Determine ledger transaction limit for sender institution.
    sender_l = find_ledger_for_account(world.ledgers, payment.from_account)
    limit = sender_l.upi_limit_paise if sender_l else 10000000
    # 2. Execute each payment leg and record aligned event telemetry.
    for leg in split_into_legal_payments(payment.amount_paise, limit):
        ev = _execute_payment_leg(world, plan, payment, leg, bank_map, rng)
        events.append(ev)
        purposes.append(payment.purpose)
        stamps.append(stamp)
        _record_event_counts(ev, counts)


def _merge_planned_payments(
    normal: list[PlannedPayment], episode_payments: Sequence[PlannedPayment]
) -> list[PlannedPayment]:
    """
    WHAT: Reindexes scheduled episode payments after routine payments and sorts by minute and sequence.
    WHY: Preserves deterministic sequence ordering where routine payments precede episode payments on ties.
         An empty sequence for episode_payments means no burst episodes are active in this run.
    Args: normal: Routine payments list; episode_payments: Sequence of candidate episode payment records.
    Returns: Combined and chronologically sorted list of all candidate payments across the simulated horizon.
    Example:
        # all_pmts = _merge_planned_payments(normal, episodes); assert len(all_pmts) >= len(normal)
    """
    # An empty sequence for episode_payments means no burst episodes are active in this run.
    base_len = len(normal)
    # Reindex episode payments so sequence indices strictly follow routine payments.
    reindexed = [dataclasses.replace(p, creation_index=base_len + i) for i, p in enumerate(episode_payments)]
    # Merge routine and episode candidates into unified candidate transaction list.
    all_payments = normal + reindexed
    # Perform stable sort by simulation minute followed by sequential creation index.
    all_payments.sort(key=lambda p: (p.minute, p.creation_index))
    return all_payments


def run_world(
    world: Any, plan: WorldPlan, rng: np.random.Generator,
    device_events: Sequence[DeviceEvent] = (), episode_payments: Sequence[PlannedPayment] = ()
) -> RunResult:
    """
    WHAT: Executes calendar-driven routine life, evaluating outages and logging payment events.
    WHY: Orchestrates calendar payment execution following issue R7, R5 device tracking, and A1 episodes.
         Empty device_events means devices never change; empty episode_payments means no episodes.
    Args: world: World; plan: Plan; rng: RNG; device_events: Events; episode_payments: Episodes.
    Returns: RunResult containing chronological event logs, device stamps, and tallies.
    Example: # res = run_world(world, plan, rng); assert len(res.events) >= 0
    """
    # 1. Build lookup tables and initialize device history tracking.
    bank_map, p_by_id = _build_account_bank_map(world.ledgers), {p.person_id: p for p in world.people}
    p_by_acc, dev_state = {p.account_id: p for p in world.people}, start_device_runtime(world.people)
    events, purposes, stamps, counts = [], [], [], {"attempts": 0, "successes": 0, "failures": 0}
    dev_idx, new_devs = 0, 0
    # 2. Collect routine payments and merge with reindexed scheduled episode payments.
    all_planned = _merge_planned_payments(_collect_all_planned_payments(world, plan, rng), episode_payments)
    # 3. Iterate planned payments, applying due device events and stamping sender devices.
    for pmt in all_planned:
        dev_idx = _apply_due_device_events(device_events, dev_idx, pmt.minute, p_by_id, dev_state)
        stamp = stamp_sender_device(p_by_acc.get(pmt.from_account), dev_state)
        new_devs += int(stamp.is_new_device)
        _execute_single_planned(world, plan, pmt, stamp, bank_map, rng, events, purposes, stamps, counts)
    # 4. Apply any remaining device events after the final transaction has executed.
    dev_idx = _apply_due_device_events(device_events, dev_idx, 100000000, p_by_id, dev_state)
    if len(purposes) != len(events) or len(stamps) != len(events) or any(p not in PAYMENT_PURPOSES for p in purposes):
        raise RuntimeError("Validation error: event list length mismatch or invalid purpose.")
    return RunResult(events, counts, purposes, stamps, {"events_applied": dev_idx, "new_device_payments": new_devs})

