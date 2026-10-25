"""
WHAT: Planner for bulk confirmations, the police crackdowns that confirm many mule-side accounts of one region at
once (Level 12e, round E-2, amendment A-021, issue L5). It draws the number of crackdowns from a Poisson count and,
for each one, a cut-off minute, a region, a size and a delay, and picks the accounts it confirms.

WHY: Crackdowns confirm accounts long after the money moved, and short worlds rarely contain one, so the planner is
a pure function over plain candidates that tests can feed directly. Requested and confirmed counts are both kept and
nothing is capped silently.

TERM: crackdown: one bulk confirmation event with a cut-off minute, a region, a requested size and a delay.
TERM: cut-off minute: only accounts whose mule-side role began at or before it can be confirmed by the crackdown.
TERM: candidate: a mule-side account with a known region, as (account_id, bank_index, region, role_valid_from).
"""

from typing import Any, Sequence

import numpy as np

from world.complaints import delay_from_uniform
from world.config_reader import draw_in_range, get_parameter_entry
from world.device_schedule import MINUTES_PER_DAY
from world.label_records import BULK_CONFIRMATION_KIND, BulkConfirmation

DAYS_PER_YEAR: int = 365

# One candidate: (account_id, bank_index, region, role_valid_from).
BulkCandidate = tuple[str, int, int, int]


def _range_bounds(
    world_ranges: dict[str, Any], split: str, reference: str, factor: int
) -> tuple[int, int]:
    """
    Returns the (low, high) of a world_ranges entry for a split, each multiplied by `factor` and rounded to a whole
    number. Raises ValueError if the split or its pair is missing or malformed, or low is negative or above high.
    """
    entry = get_parameter_entry(world_ranges, reference)
    if split not in entry:
        raise ValueError(f"world_ranges entry '{reference}' has no split '{split}'")
    bounds = entry[split]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"world_ranges entry '{reference}' split '{split}' must be a pair, got {bounds!r}")
    low = int(round(float(bounds[0]) * factor))
    high = int(round(float(bounds[1]) * factor))
    if low < 0 or low > high:
        raise ValueError(f"world_ranges entry '{reference}' split '{split}' gives invalid bounds {low}..{high}")
    return low, high


def plan_bulk_confirmations(
    candidates: Sequence[BulkCandidate],
    n_days: int,
    n_regions: int,
    world_ranges: dict[str, Any],
    split: str,
    labels_config: dict[str, Any],
    rng: np.random.Generator,
    run_id: str,
    digits: int,
) -> list[BulkConfirmation]:
    """
    WHAT: Plans the crackdowns of a world. In this order it draws (1) the yearly crackdown rate, one uniform inside
    its range; (2) the crackdown count, a Poisson draw with mean rate x n_days / 365; (3) for each crackdown a block
    of four uniforms (cut-off minute, region, size, delay) and then a permutation of the candidates.
    A crackdown confirms the first `size` candidates, in permutation order, that are in its region and whose role
    began at or before its cut-off. It arrives at cut-off plus a delay of whole minutes converted from days. Batches
    are sorted by (arrival, cut-off, region) with ties kept in drawing order, and numbered "<run_id>-B<position>".

    WHY: The rate and the size, delay and region choices come from the run's own ranges, so a layer can change
    them. A crackdown that finds fewer candidates than its size is kept with the smaller list; the requested size is
    stored beside it, so nothing is capped silently.

    Draw count: it varies with the drawn Poisson count and with the number of candidates. That is acceptable here
    because the count is itself a draw, not a config value, and child 21 serves nothing else and is used after the
    run, so a different consumption cannot shift any other stream.

    Args:
        candidates: Mule-side accounts with a known region, in a fixed order.
        n_days: Length of the world in days; at least 1.
        n_regions: Number of regions to choose from; at least 1.
        world_ranges: The run's (possibly layered) world_ranges dictionary.
        split: Range split name.
        labels_config: The `labels` block.
        rng: Generator for bulk_confirmations.
        run_id: The run id for batch ids.
        digits: Width of the zero-padded position; at least 1.

    Returns:
        The crackdowns, possibly none.

    Raises:
        ValueError: If n_days, n_regions or digits is below 1, a range is invalid, or the batch count does not fit in
            `digits` digits.

    Example:
        plan_bulk_confirmations([("acc_a", 0, 1, 1440)], 61, 4, ranges, "training", config, rng, "run1", 6)
    """
    if n_days < 1 or n_regions < 1 or digits < 1:
        raise ValueError(f"n_days {n_days}, n_regions {n_regions} and digits {digits} must each be at least 1")
    rate_name = labels_config["bulk_confirmation_events_per_year"]["world_ranges_reference"]
    size_name = labels_config["bulk_confirmation_size_accounts"]["world_ranges_reference"]
    delay_name = labels_config["bulk_confirmation_delay_days"]["world_ranges_reference"]
    size_low, size_high = _range_bounds(world_ranges, split, size_name, 1)
    delay_low, delay_high = _range_bounds(world_ranges, split, delay_name, MINUTES_PER_DAY)

    rate_per_year = draw_in_range(get_parameter_entry(world_ranges, rate_name), rng, split, rate_name)
    crackdown_count = int(rng.poisson(rate_per_year * n_days / DAYS_PER_YEAR))
    world_minutes = n_days * MINUTES_PER_DAY

    unnumbered: list[tuple[int, int, int, int, tuple[BulkCandidate, ...]]] = []
    for _crackdown in range(crackdown_count):
        cutoff_draw, region_draw, size_draw, delay_draw = (float(value) for value in rng.random(4))
        order = rng.permutation(len(candidates))
        cutoff_minute = int(cutoff_draw * world_minutes)
        region = int(region_draw * n_regions)
        requested_size = delay_from_uniform(size_draw, size_low, size_high)
        delay_minutes = delay_from_uniform(delay_draw, delay_low, delay_high)
        eligible = [
            candidates[int(index)] for index in order
            if candidates[int(index)][2] == region and candidates[int(index)][3] <= cutoff_minute
        ]
        unnumbered.append(
            (cutoff_minute + delay_minutes, cutoff_minute, region, requested_size, tuple(eligible[:requested_size]))
        )

    # sorted() is stable, so crackdowns tied on all three keys keep the order they were drawn in.
    ordered = sorted(unnumbered, key=lambda row: (row[0], row[1], row[2]))
    if len(ordered) >= 10**digits:
        raise ValueError(f"{len(ordered)} bulk confirmations do not fit in {digits} digits")
    batches: list[BulkConfirmation] = []
    for position, (arrival, cutoff, region, requested, chosen) in enumerate(ordered, start=1):
        batches.append(
            BulkConfirmation(
                batch_id=f"{run_id}-B{position:0{digits}d}",
                kind=BULK_CONFIRMATION_KIND,
                arrival_minute=arrival,
                cutoff_minute=cutoff,
                region=region,
                requested_size=requested,
                account_ids=tuple(candidate[0] for candidate in chosen),
                account_bank_indexes=tuple(candidate[1] for candidate in chosen),
            )
        )
    return batches
