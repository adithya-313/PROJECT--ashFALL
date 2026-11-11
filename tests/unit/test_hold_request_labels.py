"""
Tests for hold_request labels (replay/hold_requests.py and the hold branch of features/registry_label_consumer.py).

WHY: A cross-bank complaint reaches only the victim's bank A; the receiving bank B learns of the first-layer receiver
through the 1930 hold request, after the Level 13 latency, unless the hold is missed. Same-bank complaints already label
their bank directly, so they get no hold label.

TERM: hold request: the 1930 request that the receiving bank freeze the named account.
TERM: missed hold: a hold the bank never acts on (hold_missed_share); no label.
"""

import copy
from datetime import date, timedelta

import numpy as np

from core.event_clock import world_minute_to_ist
from features.registry_label_consumer import build_own_identifiers, match_registry_event, registry_label_rows
from replay.hold_requests import HOLD_REQUEST, plan_hold_requests
from replay.registry_replay import replay_dpip_key
from tests.unit.world.test_phone_book_recording import record
from world.hold_replay import load_hold_configuration
from world.label_records import LabelEvent

RUN = "0123456789abcdef"
KEY = replay_dpip_key(RUN)
START = date(2026, 9, 1)
RANGES = record().output.effective_config["world_configs"]["world_ranges"]
HOLD_CFG = load_hold_configuration()
BANK_FACTORS = [1.0, 1.0, 1.0]


def complaint(label_id: str, complainant_bank: int, reported_bank: int, arrival: int = 2000) -> LabelEvent:
    """A victim complaint at the complainant's bank naming an account at the reported bank."""
    return LabelEvent(
        label_id=label_id, kind="complaint", arrival_minute=arrival, complainant_account=f"victim_{label_id}",
        complainant_bank_index=complainant_bank, reported_account=f"mule_{label_id}", reported_bank_index=reported_bank,
        payment_ids=(f"{RUN}-p{label_id}",), claimed_type="investment",
    )


def plan(complaints: list[LabelEvent], ranges: dict | None = None, seed: int = 5):  # type: ignore[no-untyped-def,type-arg]
    """Plans hold requests for the complaints with the real N3 ranges (or the given ranges)."""
    return plan_hold_requests(
        complaints, ranges or RANGES, "training", HOLD_CFG, BANK_FACTORS, np.random.default_rng(seed), RUN, KEY, START
    )


def with_missed_share(share: float) -> dict:  # type: ignore[type-arg]
    """The world ranges with hold_missed_share forced to one value in the training split."""
    ranges = copy.deepcopy(RANGES)
    ranges["parameters"]["hold_missed_share"]["training"] = [share, share]
    return ranges


def test_same_bank_complaint_gets_no_hold_label() -> None:
    """Checks a complaint whose victim and receiver share a bank produces no hold event."""
    events, missed = plan([complaint(f"{RUN}-L0001", 1, 1)], with_missed_share(0.0))
    assert events == [] and missed == 0


def test_cross_bank_complaint_gives_one_event_at_b_with_latency_in_range() -> None:
    """Checks a cross-bank complaint gives one event at the receiving bank B, landing after the drawn latency."""
    events, _ = plan([complaint(f"{RUN}-L0001", 0, 2)], with_missed_share(0.0))
    (event,) = events
    assert event.item.bank_id == "bank_2" and event.label_id == f"{RUN}-L0001-hold"
    low, high = RANGES["parameters"]["hold_latency_minutes"]["training"]
    _, extra_high = RANGES["parameters"]["unreachable_1930_extra_delay_minutes"]["training"]
    # latency = base * bank factor (1.0) * night * weekend multipliers + the unreachable extra delay
    multipliers = HOLD_CFG["night_multiplier"] * HOLD_CFG["weekend_multiplier"]
    assert low <= event.latency_minutes <= high * multipliers + extra_high
    landing = world_minute_to_ist(2000, START) + timedelta(minutes=event.latency_minutes)
    assert event.visible_time == landing


def test_missed_hold_gives_no_event_and_is_counted() -> None:
    """Checks hold_missed_share 1 misses every cross-bank hold: no events, all counted."""
    events, missed = plan([complaint(f"{RUN}-L0001", 0, 2), complaint(f"{RUN}-L0002", 1, 0)], with_missed_share(1.0))
    assert events == [] and missed == 2


def test_same_seed_gives_the_same_events() -> None:
    """Checks the plan is a pure function of its stream."""
    complaints = [complaint(f"{RUN}-L000{i}", i % 3, (i + 1) % 3) for i in range(1, 6)]
    assert plan(complaints, with_missed_share(0.0)) == plan(complaints, with_missed_share(0.0))


def test_hold_label_is_a_complaint_label_via_hold_request_at_the_receiving_bank_only() -> None:
    """Checks B's label: source complaint, via hold_request, the deterministic id; A and other banks get none."""
    events, _ = plan([complaint(f"{RUN}-L0001", 0, 2)], with_missed_share(0.0))
    (event,) = events
    payload = event.item.to_dict()
    payload.update(event_type=HOLD_REQUEST, label_id=event.label_id, payment_ids=list(event.payment_ids),
                   claimed_type=event.claimed_type)
    holder = build_own_identifiers("bank_2", 2, False, [f"mule_{RUN}-L0001"], {}, {}, KEY, RUN)
    accounts = match_registry_event(payload, holder)
    assert accounts == [f"mule_{RUN}-L0001"]
    (row,) = registry_label_rows(payload, accounts, holder, RUN, event.visible_time)
    got = (row.source, row.via, row.label_id, row.bank_id)
    assert got == ("complaint", "hold_request", f"{RUN}-L0001-hold", "bank_2")
    victim_bank = build_own_identifiers("bank_0", 0, True, [f"mule_{RUN}-L0001"], {}, {}, KEY, RUN)
    assert match_registry_event(payload, victim_bank) == []


def test_the_hold_stream_is_its_own_child_and_moves_no_other_draw() -> None:
    """Checks hold_request has a child number no other stream uses and adding it leaves every other child unchanged."""
    from world.interactive_loop import build_component_streams
    from world.run_identity import load_runner_config

    children = load_runner_config()["random_stream_children"]
    assert list(children.values()).count(children["hold_request"]) == 1
    others = {name: n for name, n in children.items() if name != "hold_request"}
    before = build_component_streams(101, others)
    after = build_component_streams(101, children)
    for name in others:
        assert before[name].uniform() == after[name].uniform()
