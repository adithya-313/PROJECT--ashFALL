"""
WHAT: The fraud-on loop. Every planned payment, fraud payment and outside event is put on one ordered queue and taken
off minute by minute. Before each payment a bank hook answers allow, hold or decline; liens and debit holds from the
encumbrance book can block a debit; held payments wait until an outside event resumes them.

WHY: Fraud needs a world that can react: a bank holds a suspicious payment, the police place a lien, a gang's payment
is refused. Doing this in one queue, in a fixed order, keeps every run repeatable. Genuine payments always read their
failure draws from the run stream at their planned place, so fraud can never change a genuine payment's luck (see the
module notes in world/interactive_runner.py).

TERM: queue: the waiting line of items, always handing out the earliest item first; same-minute ties are settled by
      the tie-break order in config (outside events, then planned, then fraud, then resumed payments).
TERM: priority queue (heap): a list kept so the smallest key can be taken off quickly; Python's heapq module.
TERM: hook: a function standing in for a bank; it sees a PaymentView and answers "allow", "hold" or "decline".
TERM: piece (leg): one part of a payment split at the per-payment limit.
TERM: draw: one random number taken from a stream; each piece takes its failure draws before anything else happens.
TERM: run stream: Level 11's child-1 stream; planned payments draw from it, always at their planned place.
TERM: fraud stream / resumed stream: children 4 and 5; fraud and resumed payments draw from these instead.
TERM: stamp: the record of the sender's phone for one payment; taken once per payment, shared by all its pieces.
TERM: funds_on_hold: the failure reason when a lien or debit hold blocks a debit the balance could otherwise pay.
TERM: declined_by_bank: the failure reason when the hook declined the payment.
TERM: event source: whether an event truly came from the world's plan ("planned") or fraud inputs ("fraud").
TERM: resumed payment: a held payment released by an outside event; draws from child 5, with events_resumed True.
"""

import dataclasses
import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from world.amounts import split_into_legal_payments
from world.calendar_runner import (
    PAYMENT_PURPOSES,
    PlannedPayment,
    RunResult,
    _apply_due_device_events,
    _collect_all_planned_payments,
    _handle_live_payment,
    _merge_planned_payments,
    _record_event_counts,
)
from world.cash_withdrawals import (
    CashWithdrawal,
    CashWithdrawalRecord,
    apply_illicit_cap,
    process_cash_withdrawal,
    validate_cash_withdrawal,
)
from world.device_events import DeviceEvent
from world.device_runtime import DeviceStamp, stamp_sender_device
from world.encumbrance import EncumbranceBook
from world.failure_decision import decide_failure
from world.failures import PaymentEvent, make_event
from world.gang_devices import GangDeviceEvent
from world.gang_feed import GangFeedSettings, GangHolding, GangObservation, draw_gang_feed_settings, find_holding_gang
from world.gang_reactions import (
    GangReactionState,
    choose_spare_account,
    draw_rotation_flag,
    draw_split_flag,
    draw_split_threshold_paise,
    draw_stop_rule_flag,
    may_move_money,
    update_gang_reaction,
)
from world.interactive_records import (
    HOOK_DECISIONS,
    ForwardAdjustment,
    FraudInputs,
    GangCancellation,
    GangRedirect,
    GangSplit,
    HookDecision,
    IllicitMoneyRule,
    InteractiveRunResult,
    PaymentView,
    Resale,
    RunnerSettings,
)
from world.interactive_state import RunnerState
from world.opened_routine import draw_no_routine_share, is_idle, plan_opened_routine
from world.opening_handles import resolve_opening_handles
from world.outside_events import OutsideEvent, validate_outside_event
from world.rented_owner_activity import OwnerActivity, PausedOwnerPayment, plan_owner_activity
from world.run_identity import make_payment_id
from world.tactic_config import resolve_dividing_line
from world.truth_records import RoleSpan, TruthFacts, validate_role_lifecycle, validate_truth_facts
from world.world_plan import WorldPlan

# The failure reasons this runner adds; both are failures of the sender's bank.
REASON_DECLINED_BY_BANK: str = "declined_by_bank"
REASON_FUNDS_ON_HOLD: str = "funds_on_hold"
SENDER_BANK_DOMAIN: str = "sender_bank"


@dataclass(frozen=True)
class QueuedPayment:
    """
    WHAT: One payment waiting on the queue: the payment itself, its stream source ("planned", "fraud", "resumed" or
    "opened_routine"),
    where it truly came from ("planned" or "fraud"), its fraud input index, and, once it has been seen the first time,
    its planned payment id and its sender stamp.

    WHY: A held payment that is later resumed must keep its first name, its first stamp, and its fraud input index, so
    the phone it was made from is not marked "new" twice and truth records correlate the resumed event correctly.

    Args:
        payment: The PlannedPayment (minute, accounts, amount, order number, purpose).
        source: "planned", "fraud", "resumed" or "opened_routine" (the routine of a newly opened genuine person).
        origin: "planned" or "fraud": where the payment truly came from.
        planned_payment_id: Its name after the first time it is seen, else None.
        stamp: Its sender stamp after the first time it is seen, else None.
        input_index: The 0-based position in FraudInputs.payments; None for planned payments.

    Returns:
        A frozen QueuedPayment.

    Example:
        QueuedPayment(payment, "fraud", "fraud", input_index=0) before it is first seen;
        QueuedPayment(payment, "resumed", "fraud", "run-p-000004", stamp, input_index=0) when resumed.
    """

    payment: PlannedPayment
    source: str
    origin: str
    planned_payment_id: str | None = None
    stamp: DeviceStamp | None = None
    input_index: int | None = None


@dataclass(frozen=True)
class MultiSpanOwnerActivity:
    """
    WHAT: Owner activity tracking across multiple disjoint role spans for a resold rented mule.
    Tracks the drawn kept share and the collection of half-open active intervals.

    WHY: Ground rule G5 resells rented mules across successive syndicates, resulting in
    distinct active periods punctuated by inactive intervals between resale handoffs.

    Args:
        kept_share: Probability in [0, 1] that an outgoing owner transaction inside active spans is kept.
        intervals: Sequence of (valid_from, span_end) half-open minute intervals.

    Returns:
        A frozen MultiSpanOwnerActivity instance.
    """

    kept_share: float
    intervals: tuple[tuple[int, int | None], ...]

    def covers(self, minute: int) -> bool:
        """
        WHAT: Checks whether the given minute falls into any active rented mule interval.

        WHY: Routine owner payments occurring during any gang's active hold are subject to pausing.

        Args:
            minute: A payment's minute.

        Returns:
            True when minute lies within at least one half-open active span.
        """
        return any(
            start <= minute and (end is None or minute < end)
            for start, end in self.intervals
        )


@dataclass
class LoopContext:
    """
    WHAT: Everything one fraud-on run changes as it goes: the queue, the encumbrance book, held payments, the device
    position, counters for names, and the growing output lists (events, purposes, stamps, counts, ids, sources,
    and fraud input indices).

    WHY: Passing one context to small functions keeps each function short and named after one job, while all of
    them share the same run.

    Args:
        world: The World being run.
        plan: The WorldPlan.
        run_rng: The run stream (child 1).
        streams: The runner's own streams by config name (children 4 and 5).
        hook: The bank hook.
        run_id: The run id.
        digits: Width of padded numbers in ids, from config (6).
        ranks: Tie-break rank of each queue item kind, from config.
        state: The runner's lookup tables.
        device_events: The device schedule.

    Returns:
        A LoopContext with empty outputs.

    Example:
        A new context has events == [] and counts == {"attempts": 0, "successes": 0, "failures": 0}.
    """

    world: Any
    plan: WorldPlan
    run_rng: np.random.Generator
    streams: dict[str, np.random.Generator]
    hook: Callable[[PaymentView], HookDecision]
    run_id: str
    digits: int
    ranks: dict[str, int]
    state: RunnerState
    device_events: Sequence[DeviceEvent]
    book: EncumbranceBook = field(default_factory=EncumbranceBook)
    held: dict[str, QueuedPayment] = field(default_factory=dict)
    queue: list[tuple[int, int, int, int, Any]] = field(default_factory=list)
    device_index: int = 0
    new_devices: int = 0
    insertion_sequence: int = 0
    payment_sequence: int = 0
    encumbrance_sequence: int = 0
    events: list[PaymentEvent] = field(default_factory=list)
    purposes: list[str] = field(default_factory=list)
    stamps: list[DeviceStamp] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {"attempts": 0, "successes": 0, "failures": 0})
    payment_ids: list[str] = field(default_factory=list)
    planned_payment_ids: list[str] = field(default_factory=list)
    event_sources: list[str] = field(default_factory=list)
    events_resumed: list[bool] = field(default_factory=list)
    applied_device_events: list[DeviceEvent] = field(default_factory=list)
    applied_gang_device_events: list[GangDeviceEvent] = field(default_factory=list)
    event_fraud_input_index: list[int | None] = field(default_factory=list)
    truth: TruthFacts = field(default_factory=TruthFacts)
    illicit_balances: dict[str, int] = field(default_factory=dict)
    pair_credits: dict[tuple[str, str], int] = field(default_factory=dict)
    money_rules_by_index: dict[int, IllicitMoneyRule] = field(default_factory=dict)
    original_fraud_amounts: list[int] = field(default_factory=list)
    skipped_forwards: list[ForwardAdjustment] = field(default_factory=list)
    capped_forwards: list[ForwardAdjustment] = field(default_factory=list)
    opening_ids: dict[str, str] = field(default_factory=dict)
    owner_activity: dict[str, Any] = field(default_factory=dict)
    no_routine_share: float | None = None
    opened_routine_index: int = 0
    paused_owner_payments: list[PausedOwnerPayment] = field(default_factory=list)
    cash_withdrawals: list[CashWithdrawalRecord] = field(default_factory=list)
    money_withdrawn_paise: int = 0
    withdrawal_sequence: int = 0
    daily_atm_withdrawn: dict[tuple[str, int], int] = field(default_factory=dict)
    gang_reactions_enabled: bool = False
    gang_feed_settings: dict[str, GangFeedSettings] = field(default_factory=dict)
    gang_reaction_states: dict[str, GangReactionState] = field(default_factory=dict)
    gang_holdings: tuple[GangHolding, ...] = ()
    cancelled_by_gang: list[GangCancellation] = field(default_factory=list)
    redirected_by_gang: list[GangRedirect] = field(default_factory=list)
    split_by_gang: list[GangSplit] = field(default_factory=list)
    account_resales: list[Resale] = field(default_factory=list)


def build_component_streams(seed: int, children: dict[str, int]) -> dict[str, np.random.Generator]:
    """
    WHAT: Splits the world's seed into numbered child streams and returns one generator per configured component,
    e.g. {"fraud_payment_failures": child 4, "resumed_payment_failures": child 5}.

    WHY: Splitting a fresh copy of the seed into more children leaves children 0 to 3 exactly as Level 11 gets them
    (checked earlier), and each new component drawing from its own child cannot shift any other component's draws.

    Args:
        seed: The world's seed, e.g. 101.
        children: Component name -> child number, from config.

    Returns:
        Component name -> its own random generator.

    Example:
        build_component_streams(101, {"fraud_payment_failures": 4})["fraud_payment_failures"] is child 4 of seed 101.
    """
    # Enough children to reach the highest configured number.
    spawned = np.random.SeedSequence(seed).spawn(max(children.values()) + 1)
    return {name: np.random.default_rng(spawned[index]) for name, index in children.items()}


def push(context: LoopContext, minute: int, rank_name: str, order: int, item: Any) -> None:
    """
    WHAT: Adds one item to the queue under the key (minute, rank, order within rank, insertion number).

    WHY: The key fully decides the order. The insertion number makes every key unique, so two items never need to be
    compared directly and the order never depends on chance.

    Args:
        context: The loop context.
        minute: When the item is due, e.g. 1440.
        rank_name: Its kind's name in the tie-break order, e.g. "planned_payments".
        order: Its order within that kind (creation index for planned payments, input order for fraud payments).
        item: A QueuedPayment or an OutsideEvent.

    Returns:
        None.

    Example:
        push(context, 1440, "outside_events", 0, lien) puts the lien ahead of every payment at minute 1440.
    """
    # Count insertions so every key is different.
    context.insertion_sequence += 1
    key = (minute, context.ranks[rank_name], order, context.insertion_sequence, item)
    heapq.heappush(context.queue, key)


def validate_illicit_money_inputs(fraud_inputs: FraudInputs, state: RunnerState) -> None:
    """
    Validates illicit money rules and seeds before simulation starts.

    Ensures each rule references a valid fraud payment index without duplicates,
    each seed specifies a positive amount in paise, and seed accounts are registered.
    """
    num_payments = len(fraud_inputs.payments)
    seen_indices: set[int] = set()
    for rule in fraud_inputs.money_rules:
        if rule.input_index < 0 or rule.input_index >= num_payments:
            raise ValueError(
                f"illicit money rule input index {rule.input_index} is out of range for {num_payments} fraud payments"
            )
        if rule.input_index in seen_indices:
            raise ValueError(f"duplicate illicit money rule for fraud payment index {rule.input_index}")
        if rule.requires_credit_from is not None and not rule.spends_illicit:
            raise ValueError(
                f"rule at index {rule.input_index} has requires_credit_from but spends_illicit is False"
            )
        seen_indices.add(rule.input_index)

    for seed in fraud_inputs.illicit_seeds:
        if seed.amount_paise <= 0:
            raise ValueError(
                f"illicit seed amount for account {seed.account_id} must be positive, got {seed.amount_paise}"
            )
        if seed.account_id not in state.bank_map:
            raise ValueError(f"illicit seed account {seed.account_id} is not registered with the runner")


def queue_initial_items(context: LoopContext, planned: Sequence[PlannedPayment], fraud_inputs: FraudInputs) -> None:
    """
    WHAT: Puts every planned payment, fraud payment and outside event on the queue before the run starts. Planned
    payments keep their creation index as their order; fraud payments and outside events keep their input order.
    Each outside event is validated, and each fraud payment must use one of Level 11's payment purposes.

    WHY: Using Level 11's creation index keeps genuine payments in exactly run_world's order. Checking inputs now
    stops a malformed event before any money moves.

    Args:
        context: The loop context.
        planned: The genuine and episode payments in run_world's order.
        fraud_inputs: The fraud payments and outside events.

    Returns:
        None.

    Example:
        For the seed-101 world with no fraud, 2,915 planned payments are queued.
    """
    # Genuine and episode payments, in Level 11's order.
    for payment in planned:
        push(
            context, payment.minute, "planned_payments", payment.creation_index,
            QueuedPayment(payment=payment, source="planned", origin="planned"),
        )
    # Fraud payments, in the order given.
    for order, payment in enumerate(fraud_inputs.payments):
        if payment.purpose not in PAYMENT_PURPOSES:
            raise ValueError(f"fraud payment purpose '{payment.purpose}' is not one of {', '.join(PAYMENT_PURPOSES)}")
        push(
            context, payment.minute, "fraud_payments", order,
            QueuedPayment(payment=payment, source="fraud", origin="fraud", input_index=order),
        )
    # Outside events, checked as they are queued.
    for order, event in enumerate(fraud_inputs.outside_events):
        validate_outside_event(event)
        push(context, event.minute, "outside_events", order, event)
    # Gang device events, queued under the gang_device_events tie-break rank.
    for order, dev_event in enumerate(fraud_inputs.gang_device_events):
        push(context, dev_event.minute, "gang_device_events", order, dev_event)
    # Hidden truth facts, validated against planned fraud payments before the loop begins.
    validate_truth_facts(fraud_inputs.truth, len(fraud_inputs.payments))
    context.truth = fraud_inputs.truth
    # Illicit money rules and seeds, validated and registered before simulation proceeds.
    validate_illicit_money_inputs(fraud_inputs, context.state)
    for seed in fraud_inputs.illicit_seeds:
        context.illicit_balances[seed.account_id] = (
            context.illicit_balances.get(seed.account_id, 0) + seed.amount_paise
        )
    context.money_rules_by_index = {rule.input_index: rule for rule in fraud_inputs.money_rules}
    context.original_fraud_amounts = [p.amount_paise for p in fraud_inputs.payments]
    # Cash withdrawals, validated and queued under the cash_withdrawals tie-break rank.
    for order, withdrawal in enumerate(fraud_inputs.withdrawals):
        validate_cash_withdrawal(withdrawal, context.state)
        push(context, withdrawal.minute, "cash_withdrawals", order, withdrawal)
    context.account_resales = list(fraud_inputs.account_resales)


def _maybe_emit_gang_observation(
    context: LoopContext,
    gang_id: str,
    account_id: str,
    kind: str,
    happened_minute: int,
) -> None:
    """
    WHAT: Evaluates feed miss probability and delivery latency for telemetry on a gang-held account.
    If kept by the loss draw on child 14, queues a GangObservation at happened_minute + delay_minutes.

    WHY: Amendment A-017 models operational friction: syndicates receive delayed, lossy telemetry
    concerning friction on their accounts rather than directly observing banking ledger ground truth.

    Args:
        context: Active simulation LoopContext.
        gang_id: Identifier of the syndicate holding the account.
        account_id: Identifier of the account experiencing friction.
        kind: External friction category, one of OBSERVATION_KINDS.
        happened_minute: Simulation minute when the external friction occurred.

    Returns:
        None.
    """
    gang_settings = context.gang_feed_settings.get(gang_id)
    if gang_settings is None:
        raise ValueError(f"gang {gang_id} has no feed settings")
    # The miss draw comes first on child 14 (gang_observation_feed).
    stream = context.streams["gang_observation_feed"]
    miss_draw = float(stream.uniform(0.0, 1.0))
    if miss_draw < gang_settings.miss_share:
        return
    # Packet delivery is delayed by the syndicate's operational response latency in minutes.
    seen_minute = happened_minute + gang_settings.delay_minutes
    obs = GangObservation(
        gang_id=gang_id,
        account_id=account_id,
        kind=kind,
        happened_minute=happened_minute,
        seen_minute=seen_minute,
    )
    push(context, seen_minute, "gang_observations", 0, obs)


def apply_gang_observation(context: LoopContext, obs: GangObservation) -> None:
    """
    WHAT: Delivers a popped GangObservation to the recipient syndicate's isolated mental decision state.
    Updates stopped accounts under the stop rule, and curtails mule RoleSpan lifecycles in context.truth.

    WHY: Syndicates reactively abandon compromised accounts to protect funds; when a mule is stopped,
    its ground-truth operational lifespan terminates at the minute the syndicate learned of the lien.

    Args:
        context: Active simulation LoopContext.
        obs: Delivered GangObservation telemetry packet.

    Returns:
        None.
    """
    reaction_state = context.gang_reaction_states.get(obs.gang_id)
    if reaction_state is None:
        raise ValueError(f"gang {obs.gang_id} has no reaction state")
    was_stopped = obs.account_id in reaction_state.stopped_accounts
    update_gang_reaction(reaction_state, obs)
    is_stopped = obs.account_id in reaction_state.stopped_accounts

    # Under amendment A-017, stopping a mule replaces its RoleSpan so burned_from is updated.
    if not was_stopped and is_stopped:
        new_roles: list[RoleSpan] = []
        for role_span in context.truth.roles:
            if (
                role_span.role == "mule"
                and role_span.gang_id == obs.gang_id
                and role_span.account_id == obs.account_id
            ):
                old_burned = role_span.burned_from
                new_burned = obs.seen_minute if old_burned is None else min(old_burned, obs.seen_minute)
                old_active = role_span.active_from
                new_active = min(old_active, new_burned) if old_active is not None else None
                if new_active is None:
                    new_active = min(role_span.valid_from, new_burned)
                updated_role = dataclasses.replace(
                    role_span,
                    burned_from=new_burned,
                    active_from=new_active,
                )
                validate_role_lifecycle(updated_role)
                new_roles.append(updated_role)
            else:
                new_roles.append(role_span)
        context.truth = dataclasses.replace(context.truth, roles=tuple(new_roles))


def apply_outside_event(context: LoopContext, event: OutsideEvent) -> None:
    """
    WHAT: Applies one outside event: a lien or debit hold is placed in the encumbrance book (under its given id or a
    new "enc-000000"-style id), a release ends one, and a resume puts a held payment back on the queue at this minute
    as a "resumed" payment with its first name, stamp, and fraud input index.

    WHY: Outside events act on the book and the held list only; they never move money. A resume for a payment that is
    not held means the inputs contradict each other, so it raises instead of doing nothing.

    Args:
        context: The loop context.
        event: A validated OutsideEvent.

    Returns:
        None.

    Example:
        OutsideEvent(1440, "lien", "acc_a", 50000) with no id is placed as "enc-000000", the first encumbrance.
    """
    # Place a lien or debit hold, naming it if the event did not.
    if event.kind in ("lien", "debit_hold"):
        encumbrance_id = event.encumbrance_id or make_payment_id("enc", context.encumbrance_sequence, context.digits)
        context.encumbrance_sequence += 1
        context.book.place(event, encumbrance_id)
        if context.gang_reactions_enabled and event.account_id is not None:
            holding_gang = find_holding_gang(context.gang_holdings, event.account_id, event.minute)
            if holding_gang is not None:
                kind = "lien_placed" if event.kind == "lien" else "debit_hold_placed"
                _maybe_emit_gang_observation(context, holding_gang, event.account_id, kind, event.minute)
    elif event.kind == "release":
        released_enc = context.book.release(str(event.encumbrance_id), event.minute)
        if context.gang_reactions_enabled and released_enc.account_id is not None:
            holding_gang = find_holding_gang(context.gang_holdings, released_enc.account_id, event.minute)
            if holding_gang is not None:
                _maybe_emit_gang_observation(context, holding_gang, released_enc.account_id, "released", event.minute)
    elif event.kind == "open_account":
        stream_name = str(event.opening_stream)
        if stream_name not in context.streams:
            raise ValueError(f"unknown opening stream '{stream_name}'")
        rng = context.streams[stream_name]
        opened_id = context.state.open_account(
            owner_kind=str(event.owner_kind),
            opening_stream=stream_name,
            minute=event.minute,
            rng=rng,
        )
        # A gang opening was given its id before the run; a different id means the replay and the queue disagree.
        if event.opening_handle is not None:
            assigned_id = context.opening_ids[event.opening_handle]
            if opened_id != assigned_id:
                raise RuntimeError(
                    f"open_account for handle '{event.opening_handle}' produced '{opened_id}' "
                    f"but '{assigned_id}' was assigned before the run"
                )
        # A new genuine person may get a routine life; gang openings and shops never do.
        if (
            event.opening_stream == "genuine_account_openings"
            and event.owner_kind == "person"
            and context.no_routine_share is not None
        ):
            schedule_opened_routine(context, opened_id, event.minute)
    else:
        # A resume: the named payment must be waiting in the held list.
        planned_payment_id = str(event.planned_payment_id)
        if planned_payment_id not in context.held:
            raise ValueError(f"payment {planned_payment_id} is not held, so it cannot be resumed")
        held = context.held.pop(planned_payment_id)
        resumed_payment = dataclasses.replace(held.payment, minute=event.minute)
        resumed = QueuedPayment(
            payment=resumed_payment, source="resumed", origin=held.origin,
            planned_payment_id=held.planned_payment_id, stamp=held.stamp,
            input_index=held.input_index,
        )
        push(context, event.minute, "resumed_payments", context.insertion_sequence, resumed)


def schedule_opened_routine(context: LoopContext, account_id: str, open_minute: int) -> None:
    """
    WHAT: Decides whether a newly opened genuine person is idle, with one draw from the opened account routine stream
    against this run's no-routine share. A person with a routine gets Level 11-style payments planned from the opening
    minute on, and each is queued as an "opened_routine" payment under its own tie-break rank.

    WHY: These payments have their own stream for draws (child 10) and failures (child 11), so nothing they read can
    shift a draw of an existing genuine payment.

    Args:
        context: The loop context; context.no_routine_share must be set.
        account_id: The new person's account id.
        open_minute: Minute at which the account was opened.

    Returns:
        None. Pushes the routine payments onto the queue; an idle person pushes nothing.

    Example:
        With a no-routine share of 0 every new person has payments; with 1 none has.
    """
    stream = context.streams["opened_account_routine"]
    if is_idle(float(context.no_routine_share), stream):
        return
    person = context.state.people_by_account[account_id]
    payments, context.opened_routine_index = plan_opened_routine(
        context.world, context.plan, person, open_minute, stream, context.opened_routine_index
    )
    for payment in payments:
        push(
            context, payment.minute, "opened_routine_payments", payment.creation_index,
            QueuedPayment(payment=payment, source="opened_routine", origin="planned"),
        )


def apply_due_device_events(context: LoopContext, minute: int) -> None:
    """
    WHAT: Applies every device event due at or before this minute, with Level 11's own _apply_due_device_events and
    the same arguments run_world passes, and records each applied event in order.

    WHY: Phones must change at exactly the same points as in Level 11, or "new device" marks would differ. Recording
    them gives later parts the device event output Level 11 does not keep.

    Args:
        context: The loop context.
        minute: The minute of the payment about to be processed.

    Returns:
        None.

    Example:
        In the seed-101 world all 32 device events are applied by the end of the run.
    """
    # Level 11's helper returns the new position in the schedule.
    previous_index = context.device_index
    context.device_index = _apply_due_device_events(
        context.device_events, previous_index, minute, context.state.people_by_id, context.state.device_state
    )
    context.applied_device_events.extend(context.device_events[previous_index:context.device_index])


def banks_for(context: LoopContext, payment: PlannedPayment) -> tuple[int, int]:
    """
    WHAT: Returns the bank numbers of the paying and the receiving account from the runner's bank map, raising
    ValueError("account X is not registered with the runner") when either is unknown.

    WHY: An account opened mid-run without register_account would otherwise fail deep inside a lookup with a bare
    KeyError. A plain message points straight at the missing registration.

    Args:
        context: The loop context.
        payment: The payment about to be processed.

    Returns:
        (sender bank number, receiver bank number), e.g. (2, 0).

    Example:
        The seed-101 world's first payment, acc_person_000055 to acc_shop_0011, gives (0, 1).
    """
    # Both ends of the payment must be known to the runner.
    for account_id in (payment.from_account, payment.to_account):
        if account_id not in context.state.bank_map:
            raise ValueError(f"account {account_id} is not registered with the runner")
    return context.state.bank_map[payment.from_account], context.state.bank_map[payment.to_account]


def first_look(context: LoopContext, queued: QueuedPayment) -> QueuedPayment:
    """
    WHAT: Gives a payment its planned payment id and its sender stamp the first time it is processed, counting a new
    phone if the stamp says so. A payment that already has both (a resumed one) is returned unchanged.

    WHY: run_world stamps each planned payment exactly once, before its pieces run, and the stamp changes the device
    state. Doing it once here, and never again on resume, keeps device marks identical to Level 11.

    Args:
        context: The loop context.
        queued: The payment as it came off the queue.

    Returns:
        The payment with its planned payment id and stamp filled in.

    Example:
        The first payment processed in a run gets planned payment id "<run id>-p-000000".
    """
    # A payment seen before keeps its first name and stamp.
    if queued.planned_payment_id is not None and queued.stamp is not None:
        return queued
    # Name it with a neutral running number; the name says nothing about where it came from.
    planned_payment_id = make_payment_id(f"{context.run_id}-p", context.payment_sequence, context.digits)
    context.payment_sequence += 1
    # Stamp the sender's phone once, exactly as run_world does.
    owner = context.state.people_by_account.get(queued.payment.from_account)
    stamp = stamp_sender_device(owner, context.state.device_state)
    context.new_devices += int(stamp.is_new_device)
    return dataclasses.replace(queued, planned_payment_id=planned_payment_id, stamp=stamp)


def ask_hook(context: LoopContext, queued: QueuedPayment) -> HookDecision:
    """
    WHAT: Shows the hook the payment's PaymentView (id, minute, accounts, amount, nothing about its source) and
    returns its answer, raising ValueError for anything other than "allow", "hold" or "decline".

    WHY: The hook stands in for a bank and may be written by anyone; an unexpected answer must stop the run with a
    clear message instead of being treated as one of the three.

    Args:
        context: The loop context.
        queued: The payment, already named and stamped.

    Returns:
        "allow", "hold" or "decline".

    Example:
        With allow_all every payment returns "allow"; a hook returning "maybe" raises ValueError.
    """
    # Show only what a bank could see.
    payment = queued.payment
    view = PaymentView(
        str(queued.planned_payment_id), payment.minute, payment.from_account, payment.to_account, payment.amount_paise
    )
    decision = context.hook(view)
    if decision not in HOOK_DECISIONS:
        raise ValueError(f"the hook answered {decision!r}; it must answer allow, hold or decline")
    return decision


def stream_for(context: LoopContext, source: str) -> np.random.Generator:
    """
    WHAT: Picks the random stream for a payment's failure draws: the run stream for a planned payment at its planned
    place, child 4 for a fraud payment, child 5 for a resumed payment.

    WHY: This choice is the run-stream guarantee: genuine payments read exactly the numbers they read in Level 11,
    and nothing fraud adds can take a number meant for them.

    Args:
        context: The loop context.
        source: "planned", "fraud" or "resumed".

    Returns:
        The generator to draw from.

    Example:
        stream_for(context, "fraud") is the "fraud_payment_failures" stream, child 4 of the seed.
    """
    # Planned payments keep Level 11's stream; the others have their own.
    if source == "planned":
        return context.run_rng
    if source == "fraud":
        return context.streams["fraud_payment_failures"]
    if source == "opened_routine":
        return context.streams["opened_routine_failures"]
    return context.streams["resumed_payment_failures"]


def partition_equal_pieces(amount_paise: int, threshold_paise: int) -> list[int]:
    """
    WHAT: Partitions amount_paise into equal integer pieces strictly below threshold_paise.
    WHY: Implements the exact equal-piece partition rule from apply_split_below_threshold
    for interactive online payments where full offline Campaign records are not present.

    Args:
        amount_paise: Total payment amount to partition in integer paise.
        threshold_paise: Maximum threshold ceiling in integer paise.

    Returns:
        List of integer piece amounts in paise summing to amount_paise.
    """
    n = (amount_paise // threshold_paise) + 1
    base = amount_paise // n
    remainder = amount_paise % n
    return [base + (1 if k < remainder else 0) for k in range(n)]


def pieces_of(context: LoopContext, payment: PlannedPayment, sender_bank: int) -> list[int]:
    """
    WHAT: Splits a payment into pieces at its sending bank's per-payment limit, the way run_world does.

    WHY: Each piece takes its own failure draws, so the number of pieces decides how many draws a payment reads. Using
    the sender's ledger limit keeps that number exactly Level 11's.

    Args:
        context: The loop context.
        payment: The payment.
        sender_bank: The sending bank's number.

    Returns:
        The piece amounts in paise.

    Example:
        15,306,555 paise against a 6,122,622 limit gives [6122622, 6122622, 3061311].
    """
    # The limit lives on the sending bank's ledger.
    limit = context.world.ledgers[sender_bank].upi_limit_paise
    return split_into_legal_payments(payment.amount_paise, limit)


def decide_piece(
    context: LoopContext, payment: PlannedPayment, piece: int, banks: tuple[int, int],
    decision: HookDecision, stream: np.random.Generator,
) -> PaymentEvent:
    """
    WHAT: Decides one piece. It ALWAYS takes the failure draws first. Then: a declined payment fails as
    sender_bank/declined_by_bank; a drawn failure is recorded as Level 11 records it; a debit blocked by a lien or hold
    that the balance could pay fails as sender_bank/funds_on_hold without touching the ledger; otherwise Level 11's own
    _handle_live_payment moves the money or records insufficient_balance.

    WHY: Drawing first, every time, keeps the stream position independent of the bank's answer and of any block. The
    last step is Level 11's own code, so an allowed, unblocked piece behaves exactly as in run_world.

    Args:
        context: The loop context.
        payment: The payment the piece belongs to.
        piece: The piece amount in paise.
        banks: (sender bank, receiver bank).
        decision: The hook's answer, "allow" or "decline".
        stream: The stream for this payment's draws.

    Returns:
        The PaymentEvent for this piece.

    Example:
        Seed-101 world, event 1,000: acc_person_000006 has 2,220,327 paise and a lien of 2,202,320, so 18,007 are
        available; its 18,008-paise piece fails as sender_bank / funds_on_hold.
    """
    # The failure draws come first, whatever happens next.
    sender_bank, receiver_bank = banks
    failure = decide_failure(
        payment.minute, sender_bank, receiver_bank, context.plan.incidents, context.plan.failure_settings, stream
    )
    base = (payment.minute, payment.from_account, payment.to_account, piece, sender_bank, receiver_bank)
    # The bank refused the payment outright.
    if decision == "decline":
        return make_event(*base, SENDER_BANK_DOMAIN, REASON_DECLINED_BY_BANK)
    # A drawn failure, recorded exactly as Level 11 records it.
    if failure is not None:
        return make_event(*base, failure[0], failure[1])
    # A lien or hold blocks money the account actually has.
    balance = context.world.ledgers[sender_bank].balance_of(payment.from_account)
    if context.book.blocks_debit(payment.from_account, balance, piece) and balance >= piece:
        return make_event(*base, SENDER_BANK_DOMAIN, REASON_FUNDS_ON_HOLD)
    # Level 11's own payment step: success, or insufficient balance.
    return _handle_live_payment(context.world, payment, piece, sender_bank, receiver_bank)


def record_event(context: LoopContext, event: PaymentEvent, queued: QueuedPayment) -> None:
    """
    WHAT: Appends one event and everything that lines up with it: its purpose, the payment's shared stamp, the counts
    (with Level 11's _record_event_counts), its payment id from its position, its planned payment id and its source.

    WHY: All these lists must stay the same length as the event list, or ids would attach to the wrong event. Doing it
    in one place guarantees they grow together.

    Args:
        context: The loop context.
        event: The piece's PaymentEvent.
        queued: The payment it belongs to, already named and stamped.

    Returns:
        None.

    Example:
        The first event recorded gets payment id "<run id>-000000".
    """
    # The payment id is the event's position in the run.
    position = len(context.events)
    context.events.append(event)
    context.purposes.append(queued.payment.purpose)
    context.stamps.append(queued.stamp)  # type: ignore[arg-type]
    _record_event_counts(event, context.counts)
    context.payment_ids.append(make_payment_id(context.run_id, position, context.digits))
    context.planned_payment_ids.append(str(queued.planned_payment_id))
    context.event_sources.append(queued.origin)
    context.events_resumed.append(queued.source == "resumed")
    context.event_fraud_input_index.append(queued.input_index)

    # Telemetry trigger (2): failed events on held accounts notify the controlling syndicate.
    if context.gang_reactions_enabled and event.failure_domain == SENDER_BANK_DOMAIN:
        if event.reason == REASON_DECLINED_BY_BANK:
            holding_gang = find_holding_gang(context.gang_holdings, event.from_account, event.minute)
            if holding_gang is not None:
                _maybe_emit_gang_observation(
                    context, holding_gang, event.from_account, "payment_declined", event.minute
                )
        elif event.reason == REASON_FUNDS_ON_HOLD:
            holding_gang = find_holding_gang(context.gang_holdings, event.from_account, event.minute)
            if holding_gang is not None:
                _maybe_emit_gang_observation(
                    context, holding_gang, event.from_account, "payment_blocked_by_hold", event.minute
                )


def cap_forward_to_illicit_balance(context: LoopContext, queued: QueuedPayment) -> QueuedPayment | None:
    """
    WHAT: For a payment whose rule spends illicit money, limits the amount to the illicit money the sender holds now,
    and if requires_credit_from is set, to the illicit credit received specifically from that sender account.
    Nothing held: the payment is skipped and logged in skipped_forwards. Less held than planned: the amount is
    lowered to what is held and the change is logged in capped_forwards. Enough held: the payment is unchanged.

    WHY: A mule must pass on only money that reached it. When a payout is tied to a victim, funds paid in by other
    victims must never subsidize that victim's payout.

    Args:
        context: The loop context; its illicit balances and pair credits are read and its adjustment logs may grow.
        queued: A payment whose money rule has spends_illicit set.

    Returns:
        The payment to run (the same one, or a copy with a lower amount), or None when it must be skipped.

    Example:
        A mule planned to forward 5,000 paise that holds 3,000 paise of illicit money returns a copy of the payment
        for 3,000 paise and adds one "only_part_arrived" entry to capped_forwards.
    """
    payment = queued.payment
    planned_paise = context.original_fraud_amounts[queued.input_index]
    available_paise = max(0, context.illicit_balances.get(payment.from_account, 0))

    rule = context.money_rules_by_index.get(queued.input_index) if queued.input_index is not None else None
    if rule is not None and rule.requires_credit_from is not None:
        victim_acct = rule.requires_credit_from
        pair = (payment.from_account, victim_acct)
        pair_credit = max(0, context.pair_credits.get(pair, 0))
        effective_paise = min(planned_paise, available_paise, pair_credit)
        if effective_paise == 0:
            context.skipped_forwards.append(
                ForwardAdjustment(
                    planned_payment_id=str(queued.planned_payment_id),
                    minute=payment.minute,
                    from_account=payment.from_account,
                    to_account=payment.to_account,
                    planned_paise=planned_paise,
                    available_paise=0,
                    reason="victim_paid_nothing",
                )
            )
            return None
        if effective_paise < planned_paise:
            context.capped_forwards.append(
                ForwardAdjustment(
                    planned_payment_id=str(queued.planned_payment_id),
                    minute=payment.minute,
                    from_account=payment.from_account,
                    to_account=payment.to_account,
                    planned_paise=planned_paise,
                    available_paise=effective_paise,
                    reason="only_part_arrived",
                )
            )
        capped_payment = dataclasses.replace(payment, amount_paise=effective_paise)
        return dataclasses.replace(queued, payment=capped_payment)

    cap_res = apply_illicit_cap(
        context,
        str(queued.planned_payment_id),
        payment.minute,
        payment.from_account,
        payment.to_account,
        planned_paise,
        available_paise,
    )
    if cap_res is None:
        return None
    capped_payment = dataclasses.replace(payment, amount_paise=cap_res)
    return dataclasses.replace(queued, payment=capped_payment)


def pause_owner_payment(context: LoopContext, queued: QueuedPayment, banks: tuple[int, int]) -> bool:
    """
    WHAT: Decides whether a genuine payment of a rented mule's owner is paused. Only a planned payment sent from a
    rented mule account inside its mule span gets one draw from the rented owner activity stream; a draw at or above
    the mule's kept share pauses it. A paused payment reads the failure draws a held planned payment reads, records
    no event and moves no money, and is added to paused_owner_payments.

    WHY: The failure draws keep the run stream where it would have been, so every other genuine payment reads the
    numbers it read before. Incoming payments are never paused because only the sender is looked at.

    Args:
        context: The loop context.
        queued: The payment as it came off the queue, before it is named or stamped.
        banks: (sender bank, receiver bank) of the payment.

    Returns:
        True when the payment was paused and the caller must stop processing it.

    Example:
        With a kept share of 0 every outgoing payment of the rented mule inside its span is paused.
    """
    payment = queued.payment
    activity = context.owner_activity.get(payment.from_account)
    if queued.source != "planned" or activity is None or not activity.covers(payment.minute):
        return False
    draw = float(context.streams["rented_owner_activity"].uniform(0.0, 1.0))
    if draw < activity.kept_share:
        return False
    # Take the same failure draws a held planned payment takes, so later genuine payments read unchanged numbers.
    for _ in pieces_of(context, payment, banks[0]):
        decide_failure(payment.minute, banks[0], banks[1], context.plan.incidents,
                       context.plan.failure_settings, context.run_rng)
    context.paused_owner_payments.append(
        PausedOwnerPayment(payment.minute, payment.from_account, payment.to_account, payment.amount_paise)
    )
    return True


def process_payment(context: LoopContext, queued: QueuedPayment) -> None:
    """
    WHAT: Handles one payment taken off the queue: applies due device events, names and stamps it the first time, asks
    the hook, and then either holds it (a planned payment still takes its pieces' draws) or runs every piece.

    WHY: This is run_world's per-payment order with the bank's question inserted after the stamp. A held planned
    payment still reads its draws so the next genuine payment reads exactly the number it read in Level 11.

    Args:
        context: The loop context.
        queued: The payment as it came off the queue.

    Returns:
        None.

    Example:
        A planned payment of one piece held by the hook reads one number from the run stream and records no event.
    """
    # Unknown accounts stop the run with a plain message before anything changes.
    payment = queued.payment
    banks = banks_for(context, payment)
    # Phones change first, then the payment is named and stamped once.
    apply_due_device_events(context, payment.minute)

    # For a fraud-origin payment, cancel before stamping/drawing if either account is stopped by the gang mind.
    if context.gang_reactions_enabled and queued.origin == "fraud":
        if queued.input_index is None or queued.input_index >= len(context.truth.payment_stories):
            raise ValueError(
                f"Missing payment story for fraud payment from {payment.from_account} "
                f"to {payment.to_account} at minute {payment.minute}."
            )
        story = context.truth.payment_stories[queued.input_index]
        gang_id = story.gang_id
        if gang_id not in context.gang_reaction_states:
            raise ValueError(f"Unknown gang id '{gang_id}' for fraud payment at index {queued.input_index}.")
        reaction_state = context.gang_reaction_states[gang_id]
        from_stopped = not may_move_money(reaction_state, payment.from_account)
        to_stopped = not may_move_money(reaction_state, payment.to_account)

        if from_stopped:
            context.cancelled_by_gang.append(
                GangCancellation(
                    minute=payment.minute,
                    gang=gang_id,
                    accounts=(payment.from_account, payment.to_account),
                    amount=payment.amount_paise,
                    reason="from_account_stopped",
                )
            )
            return

        if to_stopped:
            if not reaction_state.follows_rotation_rule:
                context.cancelled_by_gang.append(
                    GangCancellation(
                        minute=payment.minute,
                        gang=gang_id,
                        accounts=(payment.from_account, payment.to_account),
                        amount=payment.amount_paise,
                        reason="to_account_stopped",
                    )
                )
                return

            # Find operational holding of this gang covering the stopped account at this minute.
            stopped_holding: GangHolding | None = None
            for h in context.gang_holdings:
                if h.gang_id == gang_id and h.account_id == payment.to_account and h.covers(payment.minute):
                    stopped_holding = h
                    break
            if stopped_holding is None:
                raise ValueError(
                    f"No holding of gang '{gang_id}' covers stopped account '{payment.to_account}' "
                    f"at minute {payment.minute}."
                )

            # Rotation applies only when the stopped to_account's holding at this minute has depth not None.
            # Otherwise cancel with reason "no_spare_account".
            if stopped_holding.depth is None:
                context.cancelled_by_gang.append(
                    GangCancellation(
                        minute=payment.minute,
                        gang=gang_id,
                        accounts=(payment.from_account, payment.to_account),
                        amount=payment.amount_paise,
                        reason="no_spare_account",
                    )
                )
                return

            stopped_depth = stopped_holding.depth
            # Candidate spare accounts: same gang, same depth (not None), held at this minute,
            # not equal to from_account.
            candidates: list[str] = []
            for h in context.gang_holdings:
                if (
                    h.gang_id == gang_id
                    and h.depth is not None
                    and h.depth == stopped_depth
                    and h.covers(payment.minute)
                    and h.account_id != payment.from_account
                    and h.account_id not in candidates
                ):
                    candidates.append(h.account_id)

            spare = choose_spare_account(reaction_state, candidates)
            if spare is not None:
                planned_to = payment.to_account
                payment = dataclasses.replace(payment, to_account=spare)
                queued = dataclasses.replace(queued, payment=payment)
                banks = banks_for(context, payment)
                context.redirected_by_gang.append(
                    GangRedirect(
                        minute=payment.minute,
                        gang=gang_id,
                        planned_to=planned_to,
                        new_to=spare,
                        amount=payment.amount_paise,
                    )
                )
            else:
                context.cancelled_by_gang.append(
                    GangCancellation(
                        minute=payment.minute,
                        gang=gang_id,
                        accounts=(payment.from_account, payment.to_account),
                        amount=payment.amount_paise,
                        reason="no_spare_account",
                    )
                )
                return

    if pause_owner_payment(context, queued, banks):
        return
    queued = first_look(context, queued)

    rule = (
        context.money_rules_by_index.get(queued.input_index)
        if queued.input_index is not None
        else None
    )
    if rule is not None and rule.spends_illicit:
        capped = cap_forward_to_illicit_balance(context, queued)
        if capped is None:
            return
        queued = capped
        payment = queued.payment

    decision = ask_hook(context, queued)
    stream = stream_for(context, queued.source)

    pieces: list[int]
    is_split_by_gang = False
    if context.gang_reactions_enabled and queued.origin == "fraud":
        if queued.input_index is None or queued.input_index >= len(context.truth.payment_stories):
            raise ValueError(
                f"Missing payment story for fraud payment from {payment.from_account} "
                f"to {payment.to_account} at minute {payment.minute}."
            )
        story = context.truth.payment_stories[queued.input_index]
        gang_id = story.gang_id
        if gang_id not in context.gang_reaction_states:
            raise ValueError(f"Unknown gang id '{gang_id}' for fraud payment at index {queued.input_index}.")
        reaction_state = context.gang_reaction_states[gang_id]
        if reaction_state.split_threshold_paise is None:
            raise ValueError(f"Gang '{gang_id}' has split_threshold_paise as None.")
        if (
            reaction_state.follows_split_rule
            and reaction_state.has_seen_trouble
            and payment.amount_paise > reaction_state.split_threshold_paise
        ):
            is_split_by_gang = True
            threshold = reaction_state.split_threshold_paise
            equal_pieces = partition_equal_pieces(payment.amount_paise, threshold)

            if context.plan.range_name == "training":
                slow_drip_min_inr = float(resolve_dividing_line("slow_drip_max_amount_inr"))
                slow_drip_min_paise = int(round(slow_drip_min_inr * 100))
                min_piece = min(equal_pieces)
                if min_piece < slow_drip_min_paise:
                    raise ValueError(
                        f"Gang '{gang_id}' split piece amount {min_piece} paise is below "
                        f"slow_drip_max_amount_inr dividing line of {slow_drip_min_paise} paise."
                    )

            context.split_by_gang.append(
                GangSplit(
                    minute=payment.minute,
                    gang=gang_id,
                    amount=payment.amount_paise,
                    piece_count=len(equal_pieces),
                )
            )

            limit = context.world.ledgers[banks[0]].upi_limit_paise
            pieces = []
            for eq_piece in equal_pieces:
                pieces.extend(split_into_legal_payments(eq_piece, limit))

    if not is_split_by_gang:
        pieces = pieces_of(context, payment, banks[0])
    if decision == "hold":
        if context.gang_reactions_enabled:
            holding_gang = find_holding_gang(context.gang_holdings, payment.from_account, payment.minute)
            if holding_gang is not None:
                _maybe_emit_gang_observation(
                    context, holding_gang, payment.from_account, "payment_held", payment.minute
                )
        # A planned payment still uses up its draws at its planned place; fraud and resumed ones use nothing.
        if queued.source in ("planned", "opened_routine"):
            for _ in pieces:
                decide_failure(payment.minute, banks[0], banks[1], context.plan.incidents,
                               context.plan.failure_settings, stream)
        context.held[str(queued.planned_payment_id)] = queued
        return
    # Allowed or declined: every piece is decided and recorded, with no early stop.
    for piece in pieces:
        event = decide_piece(context, payment, piece, banks, decision, stream)
        record_event(context, event, queued)
        if event.status == "success" and rule is not None:
            if rule.credits_illicit:
                context.illicit_balances[payment.to_account] = (
                    context.illicit_balances.get(payment.to_account, 0) + event.amount_paise
                )
                pair = (payment.to_account, payment.from_account)
                context.pair_credits[pair] = context.pair_credits.get(pair, 0) + event.amount_paise
            if rule.spends_illicit:
                context.illicit_balances[payment.from_account] = (
                    context.illicit_balances.get(payment.from_account, 0) - event.amount_paise
                )
                if rule.requires_credit_from is not None:
                    pair = (payment.from_account, rule.requires_credit_from)
                    context.pair_credits[pair] = max(0, context.pair_credits.get(pair, 0) - event.amount_paise)


def finish_run(context: LoopContext, money_created_paise: int) -> InteractiveRunResult:
    """
    WHAT: Applies the remaining device events (as run_world does after its last payment), checks the output lists line
    up, and builds the result: Level 11's RunResult plus the ids, sources, device events, held payments, the lien and
    hold log, the run id and the money created.

    WHY: The same final steps as run_world make the Level 11-shaped part identical for an empty fraud plan. The length
    and purpose check repeats Level 11's own guard, with Level 11's message.

    Args:
        context: The loop context after the queue is empty.
        money_created_paise: Opening balances of accounts registered during the run.

    Returns:
        The InteractiveRunResult.

    Example:
        For the seed-101 world with no fraud: 2,915 events, {"events_applied": 32, "new_device_payments": 7}.
    """
    # Apply every device event still waiting; the schedule is sorted, so its last minute covers them all.
    if context.device_events:
        apply_due_device_events(context, context.device_events[-1].minute)
    # Level 11's own guard on the output lists.
    events, purposes, stamps = context.events, context.purposes, context.stamps
    if len(purposes) != len(events) or len(stamps) != len(events) or any(p not in PAYMENT_PURPOSES for p in purposes):
        raise RuntimeError("Validation error: event list length mismatch or invalid purpose.")
    device_counts = {"events_applied": context.device_index, "new_device_payments": context.new_devices}
    held = [PaymentView(str(item.planned_payment_id), item.payment.minute, item.payment.from_account,
                        item.payment.to_account, item.payment.amount_paise) for item in context.held.values()]
    return InteractiveRunResult(
        run_result=RunResult(events, context.counts, purposes, stamps, device_counts),
        payment_ids=context.payment_ids,
        planned_payment_ids=context.planned_payment_ids,
        event_sources=context.event_sources,
        events_resumed=context.events_resumed,
        applied_device_events=context.applied_device_events,
        applied_gang_device_events=context.applied_gang_device_events,
        event_fraud_input_index=context.event_fraud_input_index,
        held_payments=held,
        encumbrance_log=list(context.book.log),
        run_id=context.run_id,
        money_created_paise=money_created_paise,
        truth=context.truth,
        skipped_forwards=context.skipped_forwards,
        capped_forwards=context.capped_forwards,
        opened_accounts=list(context.state.opened_accounts),
        paused_owner_payments=context.paused_owner_payments,
        cash_withdrawals=context.cash_withdrawals,
        money_withdrawn_paise=context.money_withdrawn_paise,
        cancelled_by_gang=context.cancelled_by_gang,
        redirected_by_gang=context.redirected_by_gang,
        split_by_gang=context.split_by_gang,
        account_resales=context.account_resales,
    )


def run_fraud_on(
    world: Any, plan: WorldPlan, rng: np.random.Generator, device_events: Sequence[DeviceEvent],
    episode_payments: Sequence[PlannedPayment], *, fraud_inputs: FraudInputs,
    hook: Callable[[PaymentView], HookDecision], run_id: str, runner_config: dict[str, Any], state: RunnerState,
    settings: RunnerSettings = RunnerSettings(fraud_enabled=True),
    account_openings_config: dict[str, Any] | None = None,
    gang_reactions_config: dict[str, Any] | None = None,
) -> InteractiveRunResult:
    """
    WHAT: Runs a world with the fraud switch on. It plans genuine payments exactly as run_world does (on the run stream,
    before anything else), schedules genuine account openings if enabled, queues them with fraud payments and outside
    events, then takes items off the queue in order until it is empty, and finishes as run_world does.

    WHY: Planning first on the same stream gives the identical genuine plan. Taking everything from one ordered queue
    makes every run with the same inputs repeat exactly. When gang_reactions_enabled is True, draws per-gang feed
    settings on child 14 and stop rule flags on child 15 in the order gangs first appear in gang_holdings, then in truth
    roles (not alphabetical).

    Args:
        world: The prepared World.
        plan: The WorldPlan.
        rng: The run stream, unused so far.
        device_events: The device schedule.
        episode_payments: The episodes' planned payments.
        fraud_inputs: Fraud payments and outside events.
        hook: The bank hook.
        run_id: The run id.
        runner_config: The interactive_runner config block.
        state: The runner's lookup tables.
        settings: RunnerSettings governing feature flags like account_openings_enabled and gang_reactions_enabled.
        account_openings_config: Optional account openings configuration dictionary.
        gang_reactions_config: Optional gang reactions configuration dictionary.

    Returns:
        The InteractiveRunResult.

    Example:
        With no fraud inputs and allow_all on the seed-101 world, the RunResult equals run_world's exactly.
    """
    # 1. Plan genuine and episode payments exactly as run_world does, on the same stream, first.
    planned = _merge_planned_payments(_collect_all_planned_payments(world, plan, rng), episode_payments)
    # 2-3. The runner's own streams and the context for this run.
    ranks = {name: rank for rank, name in enumerate(runner_config["tie_break_order"])}
    streams = build_component_streams(world.seed, runner_config["random_stream_children"])
    context = LoopContext(world, plan, rng, streams, hook, run_id, runner_config["position_digits"], ranks, state,
                          device_events)
    # 4. Schedule genuine account openings if enabled, then queue everything on one queue.
    effective_fraud_inputs = fraud_inputs
    if settings.account_openings_enabled:
        from world.account_openings_config import schedule_genuine_account_openings

        genuine_openings = schedule_genuine_account_openings(
            len(world.people), plan.n_days, plan.range_name,
            streams["genuine_account_openings"], config=account_openings_config,
        )
        if genuine_openings:
            effective_fraud_inputs = dataclasses.replace(
                fraud_inputs,
                outside_events=fraud_inputs.outside_events + genuine_openings,
            )
    if settings.account_openings_enabled:
        from world.account_openings_config import load_account_openings_config

        openings_config = (
            account_openings_config if account_openings_config is not None else load_account_openings_config()
        )
        context.no_routine_share = draw_no_routine_share(
            plan.range_name, openings_config, streams["opened_account_routine"]
        )
    # Gang openings are named by handles in the plan; give them their real ids now that every opening is known.
    effective_fraud_inputs, context.opening_ids = resolve_opening_handles(
        effective_fraud_inputs, state.next_person_index, state.next_shop_index
    )
    queue_initial_items(context, planned, effective_fraud_inputs)
    # Rented mules keep part of their owner's life: draw each one's kept share once, before the loop starts.
    rented_roles = [
        role for role in context.truth.roles
        if role.role == "mule" and role.origin == "rented"
    ]
    rented_accounts = {role.account_id for role in rented_roles}
    if len(rented_accounts) == len(rented_roles):
        context.owner_activity = plan_owner_activity(
            context.truth.roles, plan.range_name, streams["rented_owner_activity"]
        )
    else:
        from world.tactic_config import get_rented_owner_kept_share_range

        low_share, high_share = get_rented_owner_kept_share_range(plan.range_name)
        roles_by_acc: dict[str, list[RoleSpan]] = {}
        for role in rented_roles:
            roles_by_acc.setdefault(role.account_id, []).append(role)
        multi_activity: dict[str, OwnerActivity | MultiSpanOwnerActivity] = {}
        for acc_id in sorted(roles_by_acc.keys()):
            kept = float(streams["rented_owner_activity"].uniform(low_share, high_share))
            spans_list: list[tuple[int, int | None]] = []
            for r in roles_by_acc[acc_id]:
                span_end = r.burned_from if r.burned_from is not None else r.valid_to
                spans_list.append((r.valid_from, span_end))
            multi_activity[acc_id] = MultiSpanOwnerActivity(kept_share=kept, intervals=tuple(spans_list))
        context.owner_activity = multi_activity

    # Reactive syndicates (Level 12d, A-017): draw feed settings on child 14, then stop rules on child 15.
    if settings.gang_reactions_enabled:
        # Collect gangs in order: first appearance in gang_holdings, then truth roles (not alphabetical).
        gang_ids: list[str] = []
        for h in effective_fraud_inputs.gang_holdings:
            if h.gang_id not in gang_ids:
                gang_ids.append(h.gang_id)
        for r in effective_fraud_inputs.truth.roles:
            if r.gang_id not in gang_ids:
                gang_ids.append(r.gang_id)

        # Draw feed delay and miss share per gang on child 14.
        feed_stream = streams["gang_observation_feed"]
        for gang_id in gang_ids:
            context.gang_feed_settings[gang_id] = draw_gang_feed_settings(
                plan.range_name, feed_stream, config=gang_reactions_config
            )

        # Draw stop rule adherence flag per gang on child 15.
        reactions_stream = streams["gang_reactions"]
        for gang_id in gang_ids:
            follows_stop = draw_stop_rule_flag(
                plan.range_name, reactions_stream, config=gang_reactions_config
            )
            context.gang_reaction_states[gang_id] = GangReactionState(
                gang_id=gang_id, follows_stop_rule=follows_stop
            )

        # Draw rotation rule adherence flag per gang on child 15.
        for gang_id in gang_ids:
            follows_rotation = draw_rotation_flag(
                plan.range_name, reactions_stream, config=gang_reactions_config
            )
            context.gang_reaction_states[gang_id].follows_rotation_rule = follows_rotation

        # Draw split rule adherence flag and threshold per gang on child 15.
        for gang_id in gang_ids:
            follows_split = draw_split_flag(
                plan.range_name, reactions_stream, config=gang_reactions_config
            )
            split_threshold = draw_split_threshold_paise(
                plan.range_name, reactions_stream, config=gang_reactions_config
            )
            context.gang_reaction_states[gang_id].follows_split_rule = follows_split
            context.gang_reaction_states[gang_id].split_threshold_paise = split_threshold
        context.gang_reactions_enabled = True
        context.gang_holdings = effective_fraud_inputs.gang_holdings

    # 5. Take items off in order: outside events, gang device events, cash withdrawals, observations, or payments.
    while context.queue:
        item: (
            OutsideEvent | GangDeviceEvent | QueuedPayment | CashWithdrawal | GangObservation
        ) = heapq.heappop(context.queue)[-1]
        if isinstance(item, OutsideEvent):
            apply_outside_event(context, item)
        elif isinstance(item, GangDeviceEvent):
            context.applied_gang_device_events.append(item)
        elif isinstance(item, CashWithdrawal):
            process_cash_withdrawal(context, item)
        elif isinstance(item, GangObservation):
            apply_gang_observation(context, item)
        else:
            process_payment(context, item)
    # 6. Final device events, Level 11's guard, and the result.
    return finish_run(context, state.money_created_paise)
