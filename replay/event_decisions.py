# File: replay/event_decisions.py
# Purpose: Builds per-event decision calls for replay evaluation (Architect Decisions P2(b), 16c-1).
"""
WHAT: Derives per-executed-attempt DecisionCall records from recorded simulation events.
WHY: In replay and Level 16c datasets, each executed payment attempt (including individual pieces
of split payments) evaluates against the decision engine independently with its own receiver_vpa.
Ensures zero leakage of post-attempt facts (status, reason, failure_domain, balance_after_paise)
per Architect condition 1.
"""
from typing import Any

from replay.ordering import check_no_resumed_events
from world.bank_connectors import DecisionCall, MockBankConnectors
from world.interactive_records import PaymentView


def build_event_decision_calls(
    recorded_world: Any,
    connectors: MockBankConnectors,
) -> tuple[tuple[str, PaymentView, DecisionCall], ...]:
    """
    WHAT: Derives one DecisionCall per executed payment event in world event order.
    WHY: Implements architect decisions P2(b) and 16c-1 conditions 1-3. Each executed
    attempt generates a decision request using only attempt-time information:
    - planned_payment_id is set to the event's unique payment_id
    - receiver_vpa is included via connectors.build_decision_request(view, include_receiver_vpa=True)
    - never includes status, reason, failure_domain, or balance_after_paise (condition 1).

    Args:
        recorded_world: Recorded simulation world containing output results.
        connectors: MockBankConnectors instance providing build_decision_request.

    Returns:
        Tuple of (event_payment_id, PaymentView, DecisionCall) for every executed payment event.

    Raises:
        ValueError: If check_no_resumed_events detects any held-and-resumed payments.
    """
    # 1. Reject resumed payments to prevent leakage of hold-period information.
    check_no_resumed_events(recorded_world.output)

    # 2. Iterate through events in execution order, constructing attempt-time PaymentViews.
    calls: list[tuple[str, PaymentView, DecisionCall]] = []
    events = recorded_world.output.result.run_result.events
    payment_ids = recorded_world.output.result.payment_ids

    for event_pid, event in zip(payment_ids, events):
        # Per Architect condition 1: use only attempt-time facts.
        # planned_payment_id is set to the event's payment id so the decision payload carries it.
        view = PaymentView(
            planned_payment_id=event_pid,
            minute=event.minute,
            from_account=event.from_account,
            to_account=event.to_account,
            amount_paise=event.amount_paise,
        )
        call = connectors.build_decision_request(view, include_receiver_vpa=True)
        calls.append((event_pid, view, call))

    return tuple(calls)
