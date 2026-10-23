"""
WHAT: Builds label events (complaints) from a finished interactive run and writes them as a JSON Lines file
(Level 12e, round E-1, amendment A-020). It collects what scam victims lost and which ordinary payments succeeded,
draws the world's complaint settings and complaints from three private random streams, and numbers the result.

WHY: Labels are made after the run, the same way truth files are, so the runner, run_world, the events file and
random children 0-16 stay exactly as they were. Reading ranges from the run's own layered world ranges lets stress
layers such as complaint-backlog change label delays with no extra code.

TERM: label event: a complaint a bank receives about a payment, with no field saying whether it is true.
TERM: victim loss: the successful victim payments one scam victim made to one receiving account.
TERM: genuine payment: a successful payment from the planned world, which can draw a false complaint.
TERM: layered world ranges: the run's world_ranges after stress layers were applied; stored in effective_config.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from world.bulk_confirmations import BulkCandidate, plan_bulk_confirmations
from world.complaints import assign_label_ids, plan_false_complaints, plan_true_complaints
from world.interactive_loop import build_component_streams
from world.interactive_records import InteractiveWorldOutput
from world.label_records import (
    BulkConfirmationResult,
    GenuinePayment,
    LabelEvent,
    RegistryReport,
    VictimLoss,
    draw_label_world_settings,
    load_labels_config,
)
from world.registry_reports import confirm_accounts, plan_registry_reports
from world.run_identity import load_runner_config
from world.truth_records import SCAM_STEP_VICTIM_PAYMENT, PaymentStory

EVENT_SOURCE_FRAUD: str = "fraud"
EVENT_SOURCE_PLANNED: str = "planned"
EVENT_STATUS_SUCCESS: str = "success"
OWNER_KIND_SHOP: str = "shop"
LABEL_STREAM_NAMES: tuple[str, str, str] = ("label_world_settings", "true_complaints", "false_complaints")
REGISTRY_STREAM_NAME: str = "registry_reports"
BULK_STREAM_NAME: str = "bulk_confirmations"


def _story_of_fraud_event(output: InteractiveWorldOutput, position: int) -> PaymentStory:
    """
    Returns the payment story of the fraud event at `position`. Raises ValueError if the event has no valid
    fraud input index or the index points outside the run's payment stories.
    """
    result = output.result
    input_index = result.event_fraud_input_index[position]
    stories = result.truth.payment_stories
    if input_index is None or input_index < 0 or input_index >= len(stories):
        raise ValueError(
            f"Fraud event '{result.payment_ids[position]}' has invalid event_fraud_input_index {input_index} "
            f"(expected 0 <= index < {len(stories)})"
        )
    return stories[input_index]


def collect_victim_losses(output: InteractiveWorldOutput) -> list[VictimLoss]:
    """
    WHAT: Groups the successful victim payments of a run by (victim account, receiving account), in first-seen
    order, into VictimLoss records. Only fraud-source events with status success whose story step is
    victim_payment count; payouts back and failed payments are left out. Banks come from the first event of a
    group and last_paid_minute is the latest minute in the group.

    WHY: A victim can complain only about money actually lost, and only about what it paid, never about a payout
    that came back. Reading the step from the story avoids parsing description text.

    Args:
        output: The finished InteractiveWorldOutput; it is only read.

    Returns:
        One VictimLoss per victim and receiving account that had at least one successful victim payment.

    Raises:
        ValueError: If a fraud event has an invalid story index or two stories of one group disagree on the scam type.

    Example:
        A victim whose only payment failed in an outage has no group, so the result has no loss for that victim.
    """
    result = output.result
    positions_by_pair: dict[tuple[str, str], list[int]] = {}
    for position, event in enumerate(result.run_result.events):
        if result.event_sources[position] != EVENT_SOURCE_FRAUD:
            continue
        if event.status != EVENT_STATUS_SUCCESS:
            continue
        story = _story_of_fraud_event(output, position)
        if story.scam_step != SCAM_STEP_VICTIM_PAYMENT:
            continue
        positions_by_pair.setdefault((event.from_account, event.to_account), []).append(position)

    losses: list[VictimLoss] = []
    for (victim_account, mule_account), positions in positions_by_pair.items():
        events = [result.run_result.events[position] for position in positions]
        scam_types = {_story_of_fraud_event(output, position).scam_type for position in positions}
        if len(scam_types) != 1:
            raise ValueError(
                f"Stories of victim '{victim_account}' paying '{mule_account}' disagree on the scam type: "
                f"{sorted(str(item) for item in scam_types)}"
            )
        scam_type = scam_types.pop()
        if scam_type is None:
            raise ValueError(f"Victim payment of '{victim_account}' has a story without a scam type")
        losses.append(
            VictimLoss(
                victim_account=victim_account,
                victim_bank_index=events[0].from_bank_index,
                mule_account=mule_account,
                mule_bank_index=events[0].to_bank_index,
                scam_type=scam_type,
                payment_ids=tuple(result.payment_ids[position] for position in positions),
                last_paid_minute=max(event.minute for event in events),
            )
        )
    return losses


def collect_genuine_payments(output: InteractiveWorldOutput) -> list[GenuinePayment]:
    """
    WHAT: Lists the successful planned-source events of a run as GenuinePayment records, in event order. A payee
    counts as a shop when it is one of the world's shops or an account opened during the run with owner kind shop.

    WHY: False complaints come from ordinary payments, and shops draw more of them (issue A3), so the planner needs
    to know which payees are shops, including shops that opened mid-run.

    Args:
        output: The finished InteractiveWorldOutput; it is only read.

    Returns:
        One GenuinePayment per successful planned event.

    Example:
        With fraud off, every successful event of the run appears once.
    """
    result = output.result
    shop_accounts = {shop.account_id for shop in output.world.shops}
    for opened in result.opened_accounts:
        if opened[2] == OWNER_KIND_SHOP:
            shop_accounts.add(opened[1])

    payments: list[GenuinePayment] = []
    for position, event in enumerate(result.run_result.events):
        if result.event_sources[position] != EVENT_SOURCE_PLANNED:
            continue
        if event.status != EVENT_STATUS_SUCCESS:
            continue
        payments.append(
            GenuinePayment(
                payment_id=result.payment_ids[position],
                minute=event.minute,
                payer_account=event.from_account,
                payer_bank_index=event.from_bank_index,
                payee_account=event.to_account,
                payee_bank_index=event.to_bank_index,
                payee_is_shop=event.to_account in shop_accounts,
            )
        )
    return payments


def build_label_events(
    output: InteractiveWorldOutput,
    runner_config: dict[str, Any] | None = None,
    labels_config: dict[str, Any] | None = None,
) -> list[LabelEvent]:
    """
    WHAT: Builds the numbered label events of a finished run. It reads the layered world ranges, the split, the
    seed and the run id from the output; makes generators for children label_world_settings, true_complaints and
    false_complaints; draws the world's settings, plans true then false complaints, and numbers all events.

    WHY: Using only three new children leaves children 0-16 untouched, so the run is unchanged. Nothing in the
    output is modified, and with fraud off only false complaints exist because there are no victim losses.

    Args:
        output: The finished InteractiveWorldOutput; it is only read.
        runner_config: Optional interactive_runner block; defaults to config/interactive_runner.yaml.
        labels_config: Optional validated labels block; defaults to config/labels.yaml.

    Returns:
        The label events sorted by arrival with ids "<run_id>-L<position>".

    Raises:
        ValueError: If a label stream is missing from the runner config or any planner refuses its input.

    Example:
        build_label_events(output)[0].label_id == output.result.run_id + "-L000001"
    """
    runner = runner_config if runner_config is not None else load_runner_config()
    labels = labels_config if labels_config is not None else load_labels_config()
    effective = output.effective_config
    world_ranges = effective["world_configs"]["world_ranges"]
    split = effective["settings"]["range_name"]
    seed = effective["settings"]["seed"]
    run_id = output.result.run_id

    children = runner["random_stream_children"]
    missing = [name for name in LABEL_STREAM_NAMES if name not in children]
    if missing:
        raise ValueError(f"random_stream_children is missing the label streams {missing}")
    streams = build_component_streams(seed, {name: children[name] for name in LABEL_STREAM_NAMES})

    settings = draw_label_world_settings(world_ranges, split, streams["label_world_settings"], labels)
    true_events = plan_true_complaints(
        collect_victim_losses(output), settings, world_ranges, split, streams["true_complaints"], labels
    )
    false_events = plan_false_complaints(
        collect_genuine_payments(output), settings, world_ranges, split, streams["false_complaints"], labels
    )
    return assign_label_ids(true_events + false_events, run_id, runner["position_digits"])


def write_label_events_file(events: list[LabelEvent], path: Path) -> str:
    """
    WHAT: Writes one compact JSON line per label event, with sorted keys and payment_ids as a list, and returns the
    SHA-256 of the written bytes. Uses LF newlines and UTF-8 like the truth files.

    WHY: Byte-identical files make repeatability checkable with one hash.

    Args:
        events: The label events to write, in order.
        path: Destination file; it is overwritten if it exists.

    Returns:
        The hexadecimal SHA-256 digest of the file.

    Raises:
        OSError: If the file cannot be written.

    Example:
        write_label_events_file(events, Path("labels.jsonl")) returns a 64-character hex string.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as label_file:
        for event in events:
            row = dataclasses.asdict(event)
            row["payment_ids"] = list(event.payment_ids)
            label_file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_registry_reports(
    output: InteractiveWorldOutput,
    complaints: list[LabelEvent],
    runner_config: dict[str, Any] | None = None,
    labels_config: dict[str, Any] | None = None,
) -> list[RegistryReport]:
    """
    WHAT: Builds the numbered registry reports of a finished run from its label events. Accounts are confirmed by
    their first complaint, the reporting share and the reports are drawn from the registry_reports stream, and the
    holder fingerprint is looked up for person accounts only.

    WHY: Using its own child leaves the victim complaint draws and children 0-19 untouched, so the two shares of
    issue L2 stay independent. Shops and accounts opened mid-run have no device record and get None.

    Args:
        output: The finished InteractiveWorldOutput; it is only read.
        complaints: The numbered label events from build_label_events.
        runner_config: Optional interactive_runner block; defaults to config/interactive_runner.yaml.
        labels_config: Optional validated labels block; defaults to config/labels.yaml.

    Returns:
        The registry reports sorted by report minute, with ids "<run_id>-R<position>".

    Raises:
        ValueError: If the registry stream is missing from the runner config or a planner refuses its input.

    Example:
        build_registry_reports(output, build_label_events(output))[0].report_id ends with "-R000001"
    """
    runner = runner_config if runner_config is not None else load_runner_config()
    labels = labels_config if labels_config is not None else load_labels_config()
    effective = output.effective_config
    children = runner["random_stream_children"]
    if REGISTRY_STREAM_NAME not in children:
        raise ValueError(f"random_stream_children is missing the stream '{REGISTRY_STREAM_NAME}'")
    streams = build_component_streams(
        effective["settings"]["seed"], {REGISTRY_STREAM_NAME: children[REGISTRY_STREAM_NAME]}
    )
    holder_fingerprints = {person.account_id: person.device.fingerprint for person in output.world.people}
    return plan_registry_reports(
        confirm_accounts(complaints), holder_fingerprints, effective["world_configs"]["world_ranges"],
        effective["settings"]["range_name"], labels, streams[REGISTRY_STREAM_NAME],
        output.result.run_id, runner["position_digits"],
    )


def collect_bulk_candidates(
    output: InteractiveWorldOutput, labels_config: dict[str, Any]
) -> tuple[list[BulkCandidate], list[str]]:
    """
    WHAT: Lists the mule-side accounts a crackdown could sweep and the mule-side accounts it could not. A role span
    counts when its role is in mule_side_roles. An account with a region (a person or a shop of the world) becomes a
    candidate (account_id, bank_index, region, earliest role start); an account with no region, which means one
    opened during the run, is returned in a separate list.

    WHY: Opened accounts have no region record, so they can never be swept, and the gap must be visible instead of
    silent. An account that held several mule-side spans (for example a resold mule) appears once, at its earliest
    start.

    Args:
        output: The finished InteractiveWorldOutput; it is only read.
        labels_config: The validated labels block, for mule_side_roles.

    Returns:
        (candidates, accounts_without_region), both in first-seen order of the truth roles.

    Example:
        A mule that opened its account mid-run is in accounts_without_region and in no candidate.
    """
    regions: dict[str, tuple[int, int]] = {}
    for person in output.world.people:
        regions[person.account_id] = (person.region, person.bank_index)
    for shop in output.world.shops:
        regions[shop.account_id] = (shop.region, shop.bank_index)

    mule_side_roles = labels_config["mule_side_roles"]
    earliest_start: dict[str, int] = {}
    for span in output.result.truth.roles:
        if span.role not in mule_side_roles:
            continue
        if span.account_id not in earliest_start or span.valid_from < earliest_start[span.account_id]:
            earliest_start[span.account_id] = span.valid_from

    candidates: list[BulkCandidate] = []
    without_region: list[str] = []
    for account_id, role_start in earliest_start.items():
        if account_id in regions:
            region, bank_index = regions[account_id]
            candidates.append((account_id, bank_index, region, role_start))
        else:
            without_region.append(account_id)
    return candidates, without_region


def build_bulk_confirmations(
    output: InteractiveWorldOutput,
    runner_config: dict[str, Any] | None = None,
    labels_config: dict[str, Any] | None = None,
) -> BulkConfirmationResult:
    """
    WHAT: Builds the numbered bulk confirmations of a finished run from child bulk_confirmations. Candidates come
    from the truth roles and the world's people and shops; the number of regions is one more than the highest
    region of any person or shop; the world length comes from the run settings.

    WHY: A separate child leaves every other stream untouched. Mule-side accounts without a region are returned with
    the crackdowns instead of being dropped silently.

    Args:
        output: The finished InteractiveWorldOutput; it is only read.
        runner_config: Optional interactive_runner block; defaults to config/interactive_runner.yaml.
        labels_config: Optional validated labels block; defaults to config/labels.yaml.

    Returns:
        A BulkConfirmationResult with the crackdowns and the accounts that had no region.

    Raises:
        ValueError: If the bulk stream is missing from the runner config or the planner refuses its input.

    Example:
        build_bulk_confirmations(output).confirmations may be empty in a short world.
    """
    runner = runner_config if runner_config is not None else load_runner_config()
    labels = labels_config if labels_config is not None else load_labels_config()
    effective = output.effective_config
    children = runner["random_stream_children"]
    if BULK_STREAM_NAME not in children:
        raise ValueError(f"random_stream_children is missing the stream '{BULK_STREAM_NAME}'")
    streams = build_component_streams(effective["settings"]["seed"], {BULK_STREAM_NAME: children[BULK_STREAM_NAME]})

    candidates, without_region = collect_bulk_candidates(output, labels)
    region_numbers = [person.region for person in output.world.people] + [shop.region for shop in output.world.shops]
    batches = plan_bulk_confirmations(
        candidates, effective["settings"]["n_days"], max(region_numbers) + 1,
        effective["world_configs"]["world_ranges"], effective["settings"]["range_name"], labels,
        streams[BULK_STREAM_NAME], output.result.run_id, runner["position_digits"],
    )
    return BulkConfirmationResult(tuple(batches), tuple(without_region))
