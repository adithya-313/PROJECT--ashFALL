# File: replay/ordering.py
# Purpose: Pure functions and data structures defining world-time replay step interleaving.
"""
WHAT: Interleaves recorded bank decision calls and bank feed projection events into a causal,
strictly-ordered replay step sequence.

WHY: Architect decisions 16b Q2, P1, S1, S2, and 16c-1 condition-2 guard:
Replay evaluation evaluates decisions at the exact point in simulation time when they occurred,
ensuring all prior bank feed events have been ingested into Redis and Postgres before scoring
the decision, while preventing future events or decision outcomes from leaking into the evaluation.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.event_clock import world_minute_to_ist
from world.bank_connectors import DecisionCall
from world.bank_feed import BankFeed
from world.replay_recording import RecordedWorld

KIND_ORDER: dict[str, int] = {
    "account_event": 0,
    "transaction": 1,
    "withdrawal": 2,
    "complaint": 3,
}


@dataclass(frozen=True)
class FeedStep:
    """
    WHAT: Represents a projected bank feed record scheduled for delivery during replay.
    WHY: Encapsulates world time, tenant bank, entity kind, identifier, and raw feed record.
    """

    world_time: datetime
    kind: str
    bank_id: str
    record_id: str
    record: Any


@dataclass(frozen=True)
class DecisionStep:
    """
    WHAT: Represents a recorded bank decision evaluation scheduled during replay.
    WHY: Encapsulates decision request payload, idempotency metadata, and target planned payment identifier.
    """

    world_time: datetime
    planned_payment_id: str
    call: DecisionCall


def check_no_resumed_events(output: Any) -> None:
    """
    WHAT: Validates that no payment events in the simulation output were held and subsequently resumed.
    WHY: Replay assumes zero held payments because the pass-1 hook always allows; a resumed event would
    carry its resume minute, not its attempt minute, and leak hold-period information into its decision.

    Args:
        output: InteractiveWorldOutput instance from the pass-1 simulation.

    Raises:
        ValueError: If any event carries events_resumed[i] == True, identifying the payment id and minute.
    """
    events_resumed = output.result.events_resumed
    for i, resumed in enumerate(events_resumed):
        if resumed:
            payment_id = output.result.payment_ids[i]
            minute = output.result.run_result.events[i].minute
            raise ValueError(
                f"Resumed event encountered in replay output: payment_id {payment_id} at minute {minute}"
            )


def _resolve_record_id(kind: str, record: Any) -> str:
    """Extracts a stable record identifier string from a projected feed record instance."""
    if hasattr(record, "record_id"):
        return str(record.record_id)
    if kind == "transaction":
        side = getattr(record, "side", "")
        pid = getattr(record, "payment_id", "")
        return f"{pid}:{side}" if side else str(pid)
    if kind == "account_event":
        return str(getattr(record, "event_id", ""))
    if kind == "withdrawal":
        return str(getattr(record, "withdrawal_id", ""))
    if kind == "complaint":
        return str(getattr(record, "complaint_id", ""))
    return str(getattr(record, "id", ""))


def build_replay_steps(
    recorded_world: RecordedWorld | Any,
    feed: BankFeed,
    decisions_per_event_payment: bool = False,
) -> list[FeedStep | DecisionStep]:
    """
    WHAT: Builds a causally ordered sequence of FeedStep and DecisionStep items for replay.

    WHY: Enforces deterministic world-clock interleaving so decisions execute against historically
    accurate stream state without future information leakage. Supports both planned-payment decisions
    (legacy mode, decisions_per_event_payment=False) and per-event decisions (Decision D5).

    Rules:
    1. Every feed record whose world time is at or before a decision's time is placed before that decision.
    2. EXCEPT a payment's own sides: in default mode, sides whose planned id equals the decision's planned id;
       in per-event mode, sides whose payment_id equals the decision's event payment id. They are placed
       immediately after their decision (decide first, then post).
    3. Decisions with the same world minute keep order; each is followed by its own sides before the next decision.
    4. Remaining ties between feed records: by (kind order: account_event, transaction, withdrawal, complaint),
       then bank_id, then record id.

    Args:
        recorded_world: RecordedWorld instance containing simulation output and recorded decision calls.
        feed: BankFeed containing projected transaction sides, withdrawals, account events, and complaints.
        decisions_per_event_payment: If True, derives one DecisionStep per executed event payment attempt (Decision D5).

    Returns:
        List of interleaved FeedStep and DecisionStep items ready for sequential replay.

    Raises:
        ValueError: If check_no_resumed_events detects any held-and-resumed payments.
    """
    check_no_resumed_events(recorded_world.output)

    clock_start = recorded_world.output.plan.clock.start_date
    general_feed_steps: list[FeedStep] = []
    own_sides: dict[str, list[FeedStep]] = defaultdict(list)

    if decisions_per_event_payment:
        from replay.event_decisions import build_event_decision_calls
        from world.bank_connectors import MockBankConnectors
        from world.bank_feed import resolve_bank_id
        from world.bank_feed_config import load_bank_feed_config

        feed_cfg = load_bank_feed_config()
        account_bank_map: dict[str, str] = {}

        if hasattr(recorded_world.output, "world") and recorded_world.output.world is not None:
            for ledger in recorded_world.output.world.ledgers:
                b_id = resolve_bank_id(ledger.bank_index, feed_cfg)
                for acc in ledger.accounts:
                    account_bank_map[acc] = b_id

        if hasattr(recorded_world.output.result, "opened_accounts"):
            for opened in recorded_world.output.result.opened_accounts:
                account_bank_map[opened[1]] = resolve_bank_id(opened[4], feed_cfg)

        for b_id, p_sides in feed.payments_by_bank.items():
            for p_side in p_sides:
                account_bank_map[p_side.account_id] = b_id
                if p_side.counterparty_vpa:
                    account_bank_map[p_side.counterparty_vpa] = p_side.counterparty_bank_id

        for ev in recorded_world.output.result.run_result.events:
            if ev.from_bank_index in feed_cfg.bank_ids_by_index:
                account_bank_map[ev.from_account] = resolve_bank_id(ev.from_bank_index, feed_cfg)
            if ev.to_bank_index in feed_cfg.bank_ids_by_index:
                account_bank_map[ev.to_account] = resolve_bank_id(ev.to_bank_index, feed_cfg)

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        connectors = MockBankConnectors(
            transports={},
            modes={},
            account_bank_map=account_bank_map,
            clock_start=clock_start,
            feed_config=feed_cfg,
            decision_timeout_seconds=feed_cfg.decision_timeout_seconds,
            loop=loop,
        )
        event_decision_calls = build_event_decision_calls(recorded_world, connectors)
        event_pids = set(recorded_world.output.result.payment_ids)

        # Map transaction sides by event payment_id
        for bank_id, sides in feed.payments_by_bank.items():
            for side in sides:
                rec_id = _resolve_record_id("transaction", side)
                step = FeedStep(
                    world_time=side.event_time,
                    kind="transaction",
                    bank_id=bank_id,
                    record_id=rec_id,
                    record=side,
                )
                if side.payment_id in event_pids:
                    own_sides[side.payment_id].append(step)
                else:
                    general_feed_steps.append(step)

    else:
        # Legacy mode: Map planned_payment_id -> event_ids and lookup by planned_id
        event_to_planned_id: dict[str, str] = {}
        for event_pid, planned_pid in zip(
            recorded_world.output.result.payment_ids,
            recorded_world.output.result.planned_payment_ids,
        ):
            event_to_planned_id[event_pid] = planned_pid

        for bank_id, sides in feed.payments_by_bank.items():
            for side in sides:
                rec_id = _resolve_record_id("transaction", side)
                step = FeedStep(
                    world_time=side.event_time,
                    kind="transaction",
                    bank_id=bank_id,
                    record_id=rec_id,
                    record=side,
                )
                planned_id = event_to_planned_id.get(side.payment_id)
                if planned_id is not None:
                    own_sides[planned_id].append(step)
                else:
                    general_feed_steps.append(step)

    # Account events
    for bank_id, acct_events in feed.account_events_by_bank.items():
        for acct_event in acct_events:
            rec_id = _resolve_record_id("account_event", acct_event)
            general_feed_steps.append(
                FeedStep(
                    world_time=acct_event.event_time,
                    kind="account_event",
                    bank_id=bank_id,
                    record_id=rec_id,
                    record=acct_event,
                )
            )

    # Cash withdrawals
    for bank_id, withdrawals in feed.withdrawals_by_bank.items():
        for withdrawal in withdrawals:
            rec_id = _resolve_record_id("withdrawal", withdrawal)
            general_feed_steps.append(
                FeedStep(
                    world_time=withdrawal.event_time,
                    kind="withdrawal",
                    bank_id=bank_id,
                    record_id=rec_id,
                    record=withdrawal,
                )
            )

    # Fraud complaints
    for bank_id, complaints in feed.complaints_by_bank.items():
        for complaint in complaints:
            rec_id = _resolve_record_id("complaint", complaint)
            general_feed_steps.append(
                FeedStep(
                    world_time=complaint.received_at,
                    kind="complaint",
                    bank_id=bank_id,
                    record_id=rec_id,
                    record=complaint,
                )
            )

    # Sort general feed records by (world_time, kind_order, bank_id, record_id)
    general_feed_steps.sort(
        key=lambda s: (s.world_time, KIND_ORDER[s.kind], s.bank_id, s.record_id)
    )

    # Sort own sides within each decision key by Rule 4
    for key in own_sides:
        own_sides[key].sort(
            key=lambda s: (s.world_time, KIND_ORDER[s.kind], s.bank_id, s.record_id)
        )

    # 3. Interleave decisions and feed steps
    steps: list[FeedStep | DecisionStep] = []
    feed_idx = 0
    total_feed = len(general_feed_steps)

    if decisions_per_event_payment:
        for event_pid, view, call in event_decision_calls:
            dec_world_time = world_minute_to_ist(view.minute, clock_start)
            decision_step = DecisionStep(
                world_time=dec_world_time,
                planned_payment_id=event_pid,
                call=call,
            )

            # Rule 1: Every feed record whose world time is at or before a decision's time is placed before it
            while feed_idx < total_feed and general_feed_steps[feed_idx].world_time <= dec_world_time:
                steps.append(general_feed_steps[feed_idx])
                feed_idx += 1

            # Place the decision
            steps.append(decision_step)

            # Rule 2 & 3: Place own sides immediately after their decision
            if event_pid in own_sides:
                steps.extend(own_sides[event_pid])

    else:
        for payment, call in recorded_world.calls:
            dec_world_time = world_minute_to_ist(payment.minute, clock_start)
            decision_step = DecisionStep(
                world_time=dec_world_time,
                planned_payment_id=payment.planned_payment_id,
                call=call,
            )

            # Rule 1: Every feed record whose world time is at or before a decision's time is placed before it
            while feed_idx < total_feed and general_feed_steps[feed_idx].world_time <= dec_world_time:
                steps.append(general_feed_steps[feed_idx])
                feed_idx += 1

            # Place the decision
            steps.append(decision_step)

            # Rule 2 & 3: Place own sides immediately after their decision
            if payment.planned_payment_id in own_sides:
                steps.extend(own_sides[payment.planned_payment_id])

    # Append any remaining general feed steps occurring after the last decision
    while feed_idx < total_feed:
        steps.append(general_feed_steps[feed_idx])
        feed_idx += 1

    return steps
