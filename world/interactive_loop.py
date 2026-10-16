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
from world.device_events import DeviceEvent
from world.device_runtime import DeviceStamp, stamp_sender_device
from world.encumbrance import EncumbranceBook
from world.failure_decision import decide_failure
from world.failures import PaymentEvent, make_event
from world.interactive_records import (
    HOOK_DECISIONS,
    FraudInputs,
    HookDecision,
    InteractiveRunResult,
    PaymentView,
)
from world.interactive_state import RunnerState
from world.outside_events import OutsideEvent, validate_outside_event
from world.run_identity import make_payment_id
from world.world_plan import WorldPlan

# The failure reasons this runner adds; both are failures of the sender's bank.
REASON_DECLINED_BY_BANK: str = "declined_by_bank"
REASON_FUNDS_ON_HOLD: str = "funds_on_hold"
SENDER_BANK_DOMAIN: str = "sender_bank"


@dataclass(frozen=True)
class QueuedPayment:
    """
    WHAT: One payment waiting on the queue: the payment itself, where it came from ("planned", "fraud" or "resumed"),
    and, once it has been seen the first time, its planned payment id and its sender stamp.

    WHY: A held payment that is later resumed must keep its first name and its first stamp, so the phone it was made
    from is not marked "new" twice. Carrying both on the queued item makes that automatic.

    Args:
        payment: The PlannedPayment (minute, accounts, amount, order number, purpose).
        source: "planned", "fraud" or "resumed".
        planned_payment_id: Its name after the first time it is seen, else None.
        stamp: Its sender stamp after the first time it is seen, else None.

    Returns:
        A frozen QueuedPayment.

    Example:
        QueuedPayment(payment, "fraud") before it is first seen; QueuedPayment(payment, "resumed", "run-p-000004",
        stamp) when a held payment comes back.
    """

    payment: PlannedPayment
    source: str
    planned_payment_id: str | None = None
    stamp: DeviceStamp | None = None


@dataclass
class LoopContext:
    """
    WHAT: Everything one fraud-on run changes as it goes: the queue, the encumbrance book, held payments, the device
    position, counters for names, and the growing output lists (events, purposes, stamps, counts, ids, sources).

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
    applied_device_events: list[DeviceEvent] = field(default_factory=list)


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
        push(context, payment.minute, "planned_payments", payment.creation_index, QueuedPayment(payment, "planned"))
    # Fraud payments, in the order given.
    for order, payment in enumerate(fraud_inputs.payments):
        if payment.purpose not in PAYMENT_PURPOSES:
            raise ValueError(f"fraud payment purpose '{payment.purpose}' is not one of {', '.join(PAYMENT_PURPOSES)}")
        push(context, payment.minute, "fraud_payments", order, QueuedPayment(payment, "fraud"))
    # Outside events, checked as they are queued.
    for order, event in enumerate(fraud_inputs.outside_events):
        validate_outside_event(event)
        push(context, event.minute, "outside_events", order, event)


def apply_outside_event(context: LoopContext, event: OutsideEvent) -> None:
    """
    WHAT: Applies one outside event: a lien or debit hold is placed in the encumbrance book (under its given id or a
    new "enc-000000"-style id), a release ends one, and a resume puts a held payment back on the queue at this minute
    as a "resumed" payment with its first name and stamp.

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
    elif event.kind == "release":
        context.book.release(str(event.encumbrance_id), event.minute)
    else:
        # A resume: the named payment must be waiting in the held list.
        planned_payment_id = str(event.planned_payment_id)
        if planned_payment_id not in context.held:
            raise ValueError(f"payment {planned_payment_id} is not held, so it cannot be resumed")
        held = context.held.pop(planned_payment_id)
        resumed_payment = dataclasses.replace(held.payment, minute=event.minute)
        resumed = QueuedPayment(resumed_payment, "resumed", held.planned_payment_id, held.stamp)
        push(context, event.minute, "resumed_payments", context.insertion_sequence, resumed)


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
    return QueuedPayment(queued.payment, queued.source, planned_payment_id, stamp)


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
    return context.streams["resumed_payment_failures"]


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
    context.event_sources.append(queued.source)


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
    queued = first_look(context, queued)
    decision = ask_hook(context, queued)
    stream = stream_for(context, queued.source)
    pieces = pieces_of(context, payment, banks[0])
    if decision == "hold":
        # A planned payment still uses up its draws at its planned place; fraud and resumed ones use nothing.
        if queued.source == "planned":
            for _ in pieces:
                decide_failure(payment.minute, banks[0], banks[1], context.plan.incidents,
                               context.plan.failure_settings, stream)
        context.held[str(queued.planned_payment_id)] = queued
        return
    # Allowed or declined: every piece is decided and recorded, with no early stop.
    for piece in pieces:
        record_event(context, decide_piece(context, payment, piece, banks, decision, stream), queued)


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
        RunResult(events, context.counts, purposes, stamps, device_counts), context.payment_ids,
        context.planned_payment_ids, context.event_sources, context.applied_device_events, held,
        list(context.book.log), context.run_id, money_created_paise,
    )


def run_fraud_on(
    world: Any, plan: WorldPlan, rng: np.random.Generator, device_events: Sequence[DeviceEvent],
    episode_payments: Sequence[PlannedPayment], *, fraud_inputs: FraudInputs,
    hook: Callable[[PaymentView], HookDecision], run_id: str, runner_config: dict[str, Any], state: RunnerState,
) -> InteractiveRunResult:
    """
    WHAT: Runs a world with the fraud switch on. It plans genuine payments exactly as run_world does (on the run stream,
    before anything else), queues them with the fraud payments and outside events, then takes items off the queue in
    order until it is empty, and finishes as run_world does.

    WHY: Planning first on the same stream gives the identical genuine plan. Taking everything from one ordered queue
    makes every run with the same inputs repeat exactly.

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
    # 4. Everything goes on one queue.
    queue_initial_items(context, planned, fraud_inputs)
    # 5. Take items off in order: outside events change the book or the held list; payments run.
    while context.queue:
        item = heapq.heappop(context.queue)[-1]
        if isinstance(item, OutsideEvent):
            apply_outside_event(context, item)
        else:
            process_payment(context, item)
    # 6. Final device events, Level 11's guard, and the result.
    return finish_run(context, state.money_created_paise)
