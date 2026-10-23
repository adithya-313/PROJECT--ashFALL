"""
WHAT: Registry reports from confirmed accounts (Level 12e, round E-2, amendment A-021). A bank treats an account as
confirmed when the first complaint naming it arrives; each world draws one reporting share, and each confirmed
account is reported with that chance after a drawn delay, as a RegistryReport event for Level 13.

WHY: Registry reports are the only place a wrong registry entry can appear, and they come from false complaints. The
reporting share is drawn from its own random stream, apart from the victim complaint share, so changing one never
changes the other's outcomes (issue L2).

TERM: confirmed account: an account whose first complaint has arrived; its confirmed minute is that arrival minute.
TERM: reporting share: the chance that a bank reports a confirmed account to the registry, drawn once per world.
TERM: holder fingerprint: the device fingerprint of the person who holds the account, when the holder is a person.
"""

from typing import Any, Sequence

import numpy as np

from world.complaints import delay_from_uniform
from world.config_reader import draw_in_range, get_parameter_entry
from world.label_records import (
    REGISTRY_REPORT_KIND,
    LabelEvent,
    RegistryReport,
    resolve_registry_delay_bounds_minutes,
)

# One confirmed account: (account_id, bank_index, confirmed_minute, complaint_ids).
ConfirmedAccount = tuple[str, int, int, tuple[str, ...]]


def confirm_accounts(complaints: Sequence[LabelEvent]) -> list[ConfirmedAccount]:
    """
    WHAT: Groups complaints by the account they report, in first-seen order after sorting by (arrival_minute,
    label_id). The confirmed minute of an account is the arrival of its first complaint, and its complaint ids are
    every complaint about it that arrived at or before that minute.

    WHY: Until bank decisions exist, the first complaint is when a bank treats an account as confirmed. Sorting by
    arrival first makes "first" independent of the order the complaints were planned in.

    Args:
        complaints: Numbered label events (every label_id must be set).

    Returns:
        One (account_id, bank_index, confirmed_minute, complaint_ids) per reported account; the bank index is that
        of the first complaint.

    Raises:
        ValueError: If a complaint has no label id.

    Example:
        Complaints at minutes 10 and 20 about one account give confirmed_minute 10 and only the first complaint id.
    """
    for complaint in complaints:
        if complaint.label_id == "":
            raise ValueError("confirm_accounts needs numbered complaints; a label_id is empty")
    ordered = sorted(complaints, key=lambda complaint: (complaint.arrival_minute, complaint.label_id))

    first_by_account: dict[str, LabelEvent] = {}
    for complaint in ordered:
        if complaint.reported_account not in first_by_account:
            first_by_account[complaint.reported_account] = complaint

    confirmed: list[ConfirmedAccount] = []
    for account_id, first in first_by_account.items():
        confirmed_minute = first.arrival_minute
        ids_at_confirmation = tuple(
            complaint.label_id for complaint in ordered
            if complaint.reported_account == account_id and complaint.arrival_minute <= confirmed_minute
        )
        confirmed.append((account_id, first.reported_bank_index, confirmed_minute, ids_at_confirmation))
    return confirmed


def draw_bank_registry_report_share(
    world_ranges: dict[str, Any], split: str, rng: np.random.Generator, labels_config: dict[str, Any]
) -> float:
    """
    WHAT: Draws the world's bank reporting share uniformly inside the split's range of the entry named by
    bank_registry_report_share (share_reported_to_shared_database). It makes exactly one draw.

    WHY: Drawing it once per world, first, from the registry stream keeps the share independent of the victim
    complaint share, which comes from another stream.

    Args:
        world_ranges: The run's (possibly layered) world_ranges dictionary.
        split: Range split name.
        rng: Generator for registry_reports; advanced by one draw.
        labels_config: The `labels` block.

    Returns:
        The drawn share, between 0 and 1.

    Raises:
        ValueError: If the entry or split is missing, or the drawn share is outside [0, 1].

    Example:
        draw_bank_registry_report_share(ranges, "training", np.random.default_rng(1), config)
    """
    reference = labels_config["bank_registry_report_share"]["world_ranges_reference"]
    share = draw_in_range(get_parameter_entry(world_ranges, reference), rng, split, reference)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"bank registry report share {share} must be between 0 and 1")
    return share


def plan_registry_reports(
    confirmed: Sequence[ConfirmedAccount],
    holder_fingerprints: dict[str, str],
    world_ranges: dict[str, Any],
    split: str,
    labels_config: dict[str, Any],
    rng: np.random.Generator,
    run_id: str,
    digits: int,
) -> list[RegistryReport]:
    """
    WHAT: Plans the registry reports of a world. It first draws the bank reporting share, then two arrays of
    len(confirmed) uniforms: "reported?" and the delay position inside the registry delay bounds of the labels
    config. A confirmed account is reported when its uniform is below the share, at confirmed_minute plus the drawn
    whole-minute delay. Reports are sorted by (reported_minute, account_id) and numbered
    "<run_id>-R<position>".

    WHY: A fixed draw count (one, then two arrays) keeps the stream aligned whatever the share is. The delay range
    comes from the labels config because it is defined there, not in world_ranges. Holders that are not people
    (shops, accounts opened mid-run) are absent from holder_fingerprints and get None.

    Args:
        confirmed: Output of confirm_accounts.
        holder_fingerprints: person account id -> device fingerprint.
        world_ranges: The run's (possibly layered) world_ranges dictionary.
        split: Range split name.
        labels_config: The `labels` block.
        rng: Generator for registry_reports.
        run_id: The run id for report ids.
        digits: Width of the zero-padded position (at least 1).

    Returns:
        The reports in report order; empty when nothing is confirmed or reported.

    Raises:
        ValueError: If digits is below 1, two reports tie on (reported_minute, account_id), the count does not fit in
            `digits` digits, or a range is invalid.

    Example:
        plan_registry_reports(confirmed, {}, ranges, "training", config, rng, "run1", 6)
    """
    if digits < 1:
        raise ValueError(f"digits must be at least 1, got {digits}")
    share = draw_bank_registry_report_share(world_ranges, split, rng, labels_config)
    confirmed_count = len(confirmed)
    reported_draws = rng.random(confirmed_count)
    delay_draws = rng.random(confirmed_count)
    low_minutes, high_minutes = resolve_registry_delay_bounds_minutes(split, labels_config)

    unnumbered: list[tuple[int, str, int, str | None, int, tuple[str, ...]]] = []
    for position, (account_id, bank_index, confirmed_minute, complaint_ids) in enumerate(confirmed):
        if reported_draws[position] >= share:
            continue
        delay_minutes = delay_from_uniform(float(delay_draws[position]), low_minutes, high_minutes)
        fingerprint = holder_fingerprints[account_id] if account_id in holder_fingerprints else None
        unnumbered.append(
            (confirmed_minute + delay_minutes, account_id, bank_index, fingerprint, confirmed_minute, complaint_ids)
        )

    unnumbered.sort(key=lambda row: (row[0], row[1]))
    for previous, current in zip(unnumbered, unnumbered[1:]):
        if (previous[0], previous[1]) == (current[0], current[1]):
            raise ValueError(f"Two registry reports tie on (reported_minute, account_id) {(current[0], current[1])}")
    if len(unnumbered) >= 10**digits:
        raise ValueError(f"{len(unnumbered)} registry reports do not fit in {digits} digits")

    reports: list[RegistryReport] = []
    for position, row in enumerate(unnumbered, start=1):
        reported_minute, account_id, bank_index, fingerprint, confirmed_minute, complaint_ids = row
        reports.append(
            RegistryReport(
                report_id=f"{run_id}-R{position:0{digits}d}",
                kind=REGISTRY_REPORT_KIND,
                reported_minute=reported_minute,
                reporting_bank_index=bank_index,
                account_id=account_id,
                device_fingerprint=fingerprint,
                confirmed_minute=confirmed_minute,
                complaint_ids=complaint_ids,
            )
        )
    return reports
