"""
WHAT: Pure planners for complaints as label events (Level 12e, round E-1, amendment A-020): true complaints from
scam victims, false complaints from ordinary payers, and the numbering of the finished list.

WHY: Planning is separated from reading a finished run so each rule can be tested on synthetic inputs. Every
planner draws a fixed number of random values that never depends on configuration, so one world's complaints cannot
shift the draws of another setting.

TERM: true complaint: a scammed victim reports money lost to the account it paid.
TERM: false complaint: an ordinary payer reports an ordinary payment as fraud.
TERM: claimed type: the scam type a complaint says happened; a false complaint claims the remainder type.
TERM: merchant weight: how many times likelier a payment to a shop is to draw a false complaint than a person payment.
"""

import dataclasses
import math
from typing import Any, Sequence

import numpy as np

from world.label_records import (
    LABEL_KIND_COMPLAINT,
    GenuinePayment,
    LabelEvent,
    LabelWorldSettings,
    VictimLoss,
    resolve_delay_bounds_minutes,
)
from world.scam_config import resolve_remainder_scam_type

PAYMENTS_PER_RATE_UNIT: int = 100_000


def delay_from_uniform(unit_draw: float, low_minutes: int, high_minutes: int) -> int:
    """
    Maps one uniform draw in [0, 1) to a whole-minute delay in [low_minutes, high_minutes]. Raises ValueError if
    rounding ever pushes the result past the high bound (it cannot for a draw below 1 except by float error).
    """
    span = high_minutes - low_minutes + 1
    offset = int(math.floor(unit_draw * span))
    if offset >= span:
        raise ValueError(f"Delay draw {unit_draw} mapped outside [{low_minutes}, {high_minutes}]")
    return low_minutes + offset


def plan_true_complaints(
    losses: Sequence[VictimLoss],
    settings: LabelWorldSettings,
    world_ranges: dict[str, Any],
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[LabelEvent]:
    """
    WHAT: Plans the true complaints of a world. Each loss complains with the world's victim complaint share; a
    complaint arrives last_paid_minute plus a delay drawn inside the range of the loss's scam type, in whole
    minutes. The complainant is the victim, the reported account is the receiving account, and the claimed type is
    the loss's scam type. label_id is left empty for assign_label_ids.

    WHY: Delays differ by scam type (issue L1) and not every victim complains (issue L2). Several victims may report
    the same receiving account; that stacking is kept, not merged.

    Draws: always two arrays of len(losses) uniforms, first "complains?" and then the delay position, for every loss
    whether or not it complains, so the draw count depends only on how many losses there are. With no losses nothing
    is drawn and [] is returned. Arrivals after the world's last minute are KEPT; whoever reads labels cuts them by
    minute (ground rule 14).

    Args:
        losses: The victim losses of the world, in a fixed order.
        settings: The world's drawn LabelWorldSettings.
        world_ranges: The run's (possibly layered) world_ranges dictionary.
        split: Range split name.
        rng: Generator for true_complaints.
        config: The `labels` block.

    Returns:
        The complaints as LabelEvents in loss order, each with an empty label_id.

    Raises:
        ValueError: If a loss has an unknown scam type or its delay bounds are invalid.

    Example:
        plan_true_complaints([loss], LabelWorldSettings(1.0, 1.0, 5.0), ranges, "training", rng, config)
    """
    loss_count = len(losses)
    complain_draws = rng.random(loss_count)
    delay_draws = rng.random(loss_count)

    bounds_by_type: dict[str, tuple[int, int]] = {}
    events: list[LabelEvent] = []
    for position, loss in enumerate(losses):
        # Resolved for every loss, even one that will not complain, so a wrong scam type always fails loudly.
        if loss.scam_type not in bounds_by_type:
            bounds_by_type[loss.scam_type] = resolve_delay_bounds_minutes(world_ranges, split, loss.scam_type, config)
        if complain_draws[position] >= settings.victim_complaint_share:
            continue
        low_minutes, high_minutes = bounds_by_type[loss.scam_type]
        delay_minutes = delay_from_uniform(float(delay_draws[position]), low_minutes, high_minutes)
        events.append(
            LabelEvent(
                label_id="",
                kind=LABEL_KIND_COMPLAINT,
                arrival_minute=loss.last_paid_minute + delay_minutes,
                complainant_account=loss.victim_account,
                complainant_bank_index=loss.victim_bank_index,
                reported_account=loss.mule_account,
                reported_bank_index=loss.mule_bank_index,
                payment_ids=loss.payment_ids,
                claimed_type=loss.scam_type,
            )
        )
    return events


def plan_false_complaints(
    payments: Sequence[GenuinePayment],
    settings: LabelWorldSettings,
    world_ranges: dict[str, Any],
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[LabelEvent]:
    """
    WHAT: Plans the false complaints of a world. The expected total is rate x payments / 100,000. A payment to a
    person draws a complaint with person_chance and a payment to a shop with shop_chance = merchant_weight x
    person_chance, where person_chance = expected_total / (person_count + weight x shop_count), so the expected
    total matches the rate. The complainant is the payer, the reported account the payee, the claimed type the
    remainder scam type, and the delay comes from the remainder type's range, counted from the payment minute.

    WHY: False complaints are the only source of wrong registry entries later (X8), and shops draw more of them
    (issue A3). Nothing is clipped: a chance above 1 means the settings cannot be met and is an error.

    Draws: always two arrays of len(payments) uniforms, first "complains?" and then the delay position. With no
    payments nothing is drawn and [] is returned.

    Args:
        payments: Successful genuine payments of the world, in a fixed order.
        settings: The world's drawn LabelWorldSettings.
        world_ranges: The run's (possibly layered) world_ranges dictionary.
        split: Range split name.
        rng: Generator for false_complaints.
        config: The `labels` block.

    Returns:
        The complaints as LabelEvents in payment order, each with an empty label_id and one payment id.

    Raises:
        ValueError: If person_chance or shop_chance exceeds 1 (the message carries the numbers) or the remainder
            type's delay bounds are invalid.

    Example:
        plan_false_complaints(payments, LabelWorldSettings(0.5, 2.0, 5.0), ranges, "training", rng, config)
    """
    payment_count = len(payments)
    complain_draws = rng.random(payment_count)
    delay_draws = rng.random(payment_count)
    if payment_count == 0:
        return []

    shop_count = sum(1 for payment in payments if payment.payee_is_shop)
    person_count = payment_count - shop_count
    weight = settings.false_complaint_merchant_weight
    expected_total = settings.false_complaint_rate_per_100000 * payment_count / PAYMENTS_PER_RATE_UNIT
    person_chance = expected_total / (person_count + weight * shop_count)
    shop_chance = weight * person_chance
    if person_chance > 1.0 or shop_chance > 1.0:
        raise ValueError(
            f"False complaint chances cannot exceed 1: person_chance={person_chance}, shop_chance={shop_chance} "
            f"(expected_total={expected_total}, persons={person_count}, shops={shop_count}, weight={weight})"
        )

    claimed_type = resolve_remainder_scam_type()
    low_minutes, high_minutes = resolve_delay_bounds_minutes(world_ranges, split, claimed_type, config)
    events: list[LabelEvent] = []
    for position, payment in enumerate(payments):
        chance = shop_chance if payment.payee_is_shop else person_chance
        if complain_draws[position] >= chance:
            continue
        delay_minutes = delay_from_uniform(float(delay_draws[position]), low_minutes, high_minutes)
        events.append(
            LabelEvent(
                label_id="",
                kind=LABEL_KIND_COMPLAINT,
                arrival_minute=payment.minute + delay_minutes,
                complainant_account=payment.payer_account,
                complainant_bank_index=payment.payer_bank_index,
                reported_account=payment.payee_account,
                reported_bank_index=payment.payee_bank_index,
                payment_ids=(payment.payment_id,),
                claimed_type=claimed_type,
            )
        )
    return events


def assign_label_ids(events: Sequence[LabelEvent], run_id: str, digits: int) -> list[LabelEvent]:
    """
    WHAT: Sorts the events by (arrival_minute, complainant_account, first payment id, claimed_type) and numbers them
    from 1 as "<run_id>-L<position>", zero-padded to `digits`. Returns new events; the inputs are not changed.

    WHY: A fixed order makes ids reproducible. Two events equal on all four sort keys would make the order
    arbitrary, so that is refused instead of guessed.

    Args:
        events: Events with any label_id; each needs at least one payment id.
        run_id: The run id used as the id prefix.
        digits: Width of the zero-padded position; at least 1.

    Returns:
        The events in sorted order with label_id set.

    Raises:
        ValueError: If digits is below 1, the run id is empty, an event has no payment ids, two events tie on all
            four sort keys, or the count does not fit in `digits` digits.

    Example:
        assign_label_ids([event], "abc123", 4)[0].label_id == "abc123-L0001"
    """
    if digits < 1:
        raise ValueError(f"digits must be at least 1, got {digits}")
    if run_id == "":
        raise ValueError("run_id must not be empty")
    if len(events) >= 10**digits:
        raise ValueError(f"{len(events)} events do not fit in {digits} digits")

    def sort_key(event: LabelEvent) -> tuple[int, str, str, str]:
        """Builds the four-part sort key of one event."""
        return (event.arrival_minute, event.complainant_account, event.payment_ids[0], event.claimed_type)

    ordered = sorted(events, key=sort_key)
    for previous, current in zip(ordered, ordered[1:]):
        if sort_key(previous) == sort_key(current):
            raise ValueError(
                f"Two label events tie on every sort key {sort_key(current)}; their order would be unstable"
            )
    numbered: list[LabelEvent] = []
    for position, event in enumerate(ordered, start=1):
        numbered.append(dataclasses.replace(event, label_id=f"{run_id}-L{position:0{digits}d}"))
    return numbered
