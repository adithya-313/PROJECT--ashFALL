"""
WHAT: Plans the hold requests a replay sends to receiving banks: for each cross-bank victim complaint, when the
1930 hold request reaches the bank that holds the named first-layer receiver (complaint arrival plus the Level 13
hold latency), unless the hold is missed.

WHY: A cross-bank complaint reaches only the victim's bank, so the receiving bank B learns of the mule only through the
hold request. B writes its own label for the named account; there is no forward tracing.

TERM: hold request: the 1930 administrative request that a receiving bank freeze the named account.
TERM: first-layer receiver: the account the complaint names as the receiver of the bad payments.
TERM: missed hold: a hold request the bank never acts on (hold_missed_share); it produces no label.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from core.event_clock import world_minute_to_ist
from nationalsim.adapter import IngestionItem
from nationalsim.hold_flow import compute_hold_latency
from nationalsim.keyed_hash import compute_identifier_hash
from world.config_reader import draw_in_range, get_parameter_entry
from world.label_records import LabelEvent

__all__ = ["HoldEvent", "HOLD_REQUEST", "plan_hold_requests"]

HOLD_REQUEST = "hold_request"


@dataclass(frozen=True)
class HoldEvent:
    """
    WHAT: One hold request that reaches the bank holding the reported account.
    WHY: The registry driver stages it as a registry.events event of type hold_request at visible_time.
    Args:
        item: Ingestion item (account type, the account's own-bank hash, bank_id = the receiving bank B).
        visible_time: Aware IST time the hold request reaches B (complaint arrival + latency).
        complaint_id: The victim's complaint id.
        label_id: Deterministic label id, "<complaint_id>-hold".
        payment_ids: Payments the complaint names.
        latency_minutes: The drawn hold latency.
        claimed_type: The complaint's claimed scam type.
    Example:
        HoldEvent(item, when, "r-L0001", "r-L0001-hold", ("p1",), 42, None)
    """

    item: IngestionItem
    visible_time: datetime
    complaint_id: str
    label_id: str
    payment_ids: tuple[str, ...]
    latency_minutes: int
    claimed_type: str | None


def plan_hold_requests(
    complaints: Sequence[LabelEvent],
    world_ranges: dict[str, Any],
    split: str,
    hold_cfg: dict[str, Any],
    bank_factors: Sequence[float],
    rng: np.random.Generator,
    run_id: str,
    key: str,
    world_start: date,
) -> tuple[list[HoldEvent], int]:
    """
    WHAT: For every complaint that names payments, makes the same six draws Level 13 makes (hold latency, missed share,
    unreachable share, unreachable delay, two uniforms), then keeps only cross-bank complaints whose hold is not
    missed. Returns the hold events in complaint order and the number of missed cross-bank holds.
    WHY: A fixed draw count per complaint keeps the stream aligned however the complaint banks fall; same-bank
    complaints already label their bank directly, so they get no hold label.
    Args:
        complaints: The world's victim complaints (LabelEvent).
        world_ranges: The run's world_ranges dictionary.
        split: Range split name.
        hold_cfg: night_hours, night_multiplier and weekend_multiplier from config/nationalsim.yaml.
        bank_factors: hold_delay_factor of each bank, by bank index.
        rng: The hold_request stream.
        run_id: The run id.
        key: The DPIP key (hash key).
        world_start: World start date.
    Returns:
        (hold events, missed cross-bank hold count).
    Example:
        events, missed = plan_hold_requests(complaints, ranges, "training", cfg, [1.0] * 3, rng, run_id, key, start)
    """
    entry_latency = get_parameter_entry(world_ranges, "hold_latency_minutes")
    entry_missed = get_parameter_entry(world_ranges, "hold_missed_share")
    entry_unreach_share = get_parameter_entry(world_ranges, "share_of_victims_unable_to_reach_1930_first_try")
    entry_unreach_delay = get_parameter_entry(world_ranges, "unreachable_1930_extra_delay_minutes")
    events: list[HoldEvent] = []
    missed = 0
    for complaint in complaints:
        if not complaint.payment_ids:
            continue
        base = draw_in_range(entry_latency, rng, split, "hold_latency_minutes")
        missed_share = draw_in_range(entry_missed, rng, split, "hold_missed_share")
        unreachable_share = draw_in_range(
            entry_unreach_share, rng, split, "share_of_victims_unable_to_reach_1930_first_try"
        )
        unreachable_delay = draw_in_range(entry_unreach_delay, rng, split, "unreachable_1930_extra_delay_minutes")
        u_unreachable = float(rng.uniform(0.0, 1.0))
        u_missed = float(rng.uniform(0.0, 1.0))
        if complaint.complainant_bank_index == complaint.reported_bank_index:
            continue
        if u_missed < missed_share:
            missed += 1
            continue
        arrival = world_minute_to_ist(complaint.arrival_minute, world_start)
        latency = compute_hold_latency(
            base_latency_minutes=base,
            bank_hold_delay_factor=bank_factors[complaint.reported_bank_index],
            arrival_dt=arrival,
            night_hours=hold_cfg["night_hours"],
            night_multiplier=hold_cfg["night_multiplier"],
            weekend_multiplier=hold_cfg["weekend_multiplier"],
            u_unreachable=u_unreachable,
            unreachable_share=unreachable_share,
            unreachable_extra_delay_minutes=unreachable_delay,
        )
        lands = arrival + timedelta(minutes=latency)
        bank_index = complaint.reported_bank_index
        account_hash = compute_identifier_hash(
            key, run_id, "account", complaint.reported_account, bank_index=bank_index
        )
        item = IngestionItem(
            bank_id=f"bank_{bank_index}",
            report_id=f"{complaint.label_id}-hold",
            identifier_hash=account_hash,
            identifier_type="account",
            reason_code=HOLD_REQUEST,
            reported_at=lands,
            run_id=run_id,
            linked_account_hash=None,
            reported_minute=complaint.arrival_minute + latency,
        )
        events.append(
            HoldEvent(
                item, lands, complaint.label_id, f"{complaint.label_id}-hold", tuple(complaint.payment_ids),
                latency, complaint.claimed_type,
            )
        )
    return events, missed
