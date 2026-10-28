# File: world/bank_feed.py
# Purpose: Projects simulation runtime records into isolated, bank-visible payment feed records.
"""
WHAT: Provides boundary guards and transformation functions that project raw simulation world
records into sanitized, tenant-visible bank feed events without leaking internal simulation state.
It transforms PaymentEvent instances into BankFeedPaymentSide records following R7 visibility rules.

WHY: Commercial bank participants in Level 14 only observe information that an authentic bank switch
or core banking system would provide. Simulation-internal fields like fraud ground truth, internal
episode purposes, switch outage details unseen by receiving banks, and device flags must be scrubbed.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Sequence

from core.event_clock import world_minute_to_ist
from world.bank_feed_config import BankFeedConfig, load_bank_feed_config
from world.device_runtime import DeviceStamp
from world.failures import PaymentEvent
from world.interactive_records import InteractiveWorldOutput
from world.label_records import LabelEvent


class ProjectionLeakError(ValueError):
    """
    WHAT: Exception raised when a projected record contains fields outside the permitted bank-visible schema.

    WHY: Serves as an active boundary guard ensuring no simulation-internal attributes leak across the projection.
    """


@dataclass(frozen=True)
class BankFeedPaymentSide:
    """
    WHAT: Represents a single tenant bank's ledger perspective (debit or credit) of a UPI payment.
    It encapsulates the executing bank identifier, unique payment ID, posting side, account,
    counterparty bank, routing metadata, paise amount, aware event time, and client device telemetry.

    WHY: Enforces strict tenant data boundaries where each bank observes only its own transaction side
    with sanitized counterparty references and without access to the other party's client hardware.
    """

    bank_id: str
    payment_id: str
    side: str
    account_id: str
    counterparty_bank_id: str
    sender_bank_id: str
    receiver_bank_id: str
    amount_paise: int
    event_time: datetime
    status: str
    failure_domain: str | None
    payee_type: str
    channel: str
    counterparty_vpa: str | None = None
    sender_device_fingerprint: str | None = None
    sender_os_generation: int | None = None


BANK_VISIBLE_PAYMENT_FIELDS: frozenset[str] = frozenset(
    {
        "bank_id",
        "payment_id",
        "side",
        "account_id",
        "counterparty_bank_id",
        "sender_bank_id",
        "receiver_bank_id",
        "amount_paise",
        "event_time",
        "status",
        "failure_domain",
        "payee_type",
        "channel",
        "counterparty_vpa",
        "sender_device_fingerprint",
        "sender_os_generation",
    }
)

BANK_VISIBLE_ACCOUNT_EVENT_FIELDS: frozenset[str] = frozenset(
    {
        "bank_id",
        "event_id",
        "account_id",
        "kind",
        "event_time",
        "details",
    }
)

BANK_VISIBLE_COMPLAINT_FIELDS: frozenset[str] = frozenset(
    {
        "bank_id",
        "complaint_id",
        "complainant_account",
        "reported_account",
        "reported_bank_id",
        "payment_ids",
        "claimed_type",
        "received_at",
    }
)


@dataclass(frozen=True)
class BankFeedAccountEvent:
    """
    WHAT: Bank-visible account security and lifecycle event, such as a device binding change or opening.
    It encapsulates bank, event ID, account, event kind, aware timestamp, and sanitized hardware details.

    WHY: Ground rule 29 and issue 14-10 require that banks receive device updates through an authentic
    account-event feed without internal simulation novelty flags or internal device identifiers.
    """

    bank_id: str
    event_id: str
    account_id: str
    kind: str
    event_time: datetime
    details: dict[str, Any]


@dataclass(frozen=True)
class BankFeedComplaint:
    """
    WHAT: Bank-visible customer fraud complaint filed against a suspected mule or merchant account.
    It encapsulates the complainant bank, complaint identifier, involved accounts, counterparty bank,
    referenced payment identifiers, claimed scam classification, and receipt timestamp.

    WHY: Banks only observe complaints initiated by their own customers or reported against counterparties,
    with simulation-internal label classifications stripped to match production banking interfaces.
    """

    bank_id: str
    complaint_id: str
    complainant_account: str
    reported_account: str
    reported_bank_id: str
    payment_ids: tuple[str, ...]
    claimed_type: str
    received_at: datetime


@dataclass(frozen=True)
class BankFeed:
    """
    WHAT: Container holding projected, tenant-isolated feeds for all commercial banks in a simulation run.
    It organizes payment sides, account lifecycle events, and fraud complaints by bank identifier,
    alongside counters of unmapped or dropped device events.

    WHY: Provides the complete bank-side external data interface for Level 14c without exposing
    internal simulation models or cross-bank confidential account records.
    """

    payments_by_bank: dict[str, tuple[BankFeedPaymentSide, ...]]
    account_events_by_bank: dict[str, tuple[BankFeedAccountEvent, ...]]
    complaints_by_bank: dict[str, tuple[BankFeedComplaint, ...]]
    dropped_device_events_by_kind: dict[str, int]


def check_bank_visible(record: dict[str, Any], allowed: frozenset[str]) -> None:
    """
    WHAT: Validates that a dictionary record contains only approved bank-visible attribute keys.
    It identifies any unexpected keys and raises a ProjectionLeakError specifying each violating key.

    WHY: Acts as a strict architectural barrier preventing accidental serialization of sensitive
    simulation attributes such as fraud flags, episode reasons, or internal simulator timestamps.

    Args:
        record: Dictionary representation of a projected record to audit.
        allowed: Frozenset of approved bank-visible attribute names.

    Raises:
        ProjectionLeakError: If any key in record is not present in the allowed set.
    """
    unexpected_keys = sorted(set(record.keys()) - allowed)
    if unexpected_keys:
        raise ProjectionLeakError(f"Record contains bank-invisible fields: {unexpected_keys}")


def project_payment(
    event: PaymentEvent,
    payment_id: str,
    stamp: DeviceStamp | None,
    clock_start: date,
    config: BankFeedConfig,
) -> list[BankFeedPaymentSide]:
    """
    WHAT: Transforms a raw simulation PaymentEvent into one or two bank-visible BankFeedPaymentSide records.
    It always constructs a debit side for the originating bank, and constructs a credit side for the
    beneficiary bank only if the payment succeeded or the failure domain is visible per R7 rules.

    WHY: Ground rule 29 and R7 visibility require that switch-level or sender-side payment failures
    never appear on the receiving bank's feed because the transaction request never reached that bank.

    Args:
        event: Raw simulation PaymentEvent generated by the world runner.
        payment_id: Stable deterministic payment identifier string.
        stamp: Optional client hardware telemetry snapshot captured at payment execution.
        clock_start: Aware simulation start datetime used for minute timestamp resolution.
        config: Loaded BankFeedConfig providing bank mappings, visibility tables, and prefixes.

    Returns:
        List containing one (debit only) or two (debit and credit) BankFeedPaymentSide instances.

    Raises:
        ValueError: If a bank index is unknown, an account prefix is unrecognized, or status is invalid.
    """
    # 1. Resolve commercial bank identifiers from integer simulation bank indexes.
    if event.from_bank_index not in config.bank_ids_by_index:
        raise ValueError(f"Unknown sender bank index: {event.from_bank_index}")
    if event.to_bank_index not in config.bank_ids_by_index:
        raise ValueError(f"Unknown receiver bank index: {event.to_bank_index}")

    sender_bank_id = config.bank_ids_by_index[event.from_bank_index]
    receiver_bank_id = config.bank_ids_by_index[event.to_bank_index]

    # 2. Derive bank-visible payee classification from beneficiary account prefix.
    resolved_payee_type: str | None = None
    for prefix, ptype in config.payee_type_by_account_prefix.items():
        if event.to_account.startswith(prefix):
            resolved_payee_type = ptype
            break

    if resolved_payee_type is None:
        raise ValueError(
            f"to_account '{event.to_account}' has unknown prefix; "
            f"expected one of {sorted(config.payee_type_by_account_prefix.keys())}"
        )

    # 3. Compute timezone-aware IST event timestamp from simulation minute.
    event_time = world_minute_to_ist(event.minute, clock_start)

    # 4. Construct sender bank debit record (sender always observes their own attempt).
    debit_record = BankFeedPaymentSide(
        bank_id=sender_bank_id,
        payment_id=payment_id,
        side="debit",
        account_id=event.from_account,
        counterparty_bank_id=receiver_bank_id,
        sender_bank_id=sender_bank_id,
        receiver_bank_id=receiver_bank_id,
        amount_paise=event.amount_paise,
        event_time=event_time,
        status=event.status,
        failure_domain=event.failure_domain,
        payee_type=resolved_payee_type,
        channel=config.channel_value,
        counterparty_vpa=None,
        sender_device_fingerprint=stamp.fingerprint if stamp is not None else None,
        sender_os_generation=stamp.os_generation if stamp is not None else None,
    )
    check_bank_visible(asdict(debit_record), BANK_VISIBLE_PAYMENT_FIELDS)
    sides: list[BankFeedPaymentSide] = [debit_record]

    # 5. Evaluate R7 visibility rule to determine if beneficiary bank receives a credit record.
    include_credit = False
    if event.status == "success":
        include_credit = True
    elif event.status == "failed":
        if event.failure_domain is None:
            raise ValueError(f"Payment {payment_id} marked failed but missing failure_domain")
        if event.failure_domain not in config.failed_payment_visibility:
            raise ValueError(f"Unknown failure_domain '{event.failure_domain}' on payment {payment_id}")
        include_credit = config.failed_payment_visibility[event.failure_domain]["receiver"]
    else:
        raise ValueError(f"Unknown payment status '{event.status}' on payment {payment_id}")

    if include_credit:
        credit_record = BankFeedPaymentSide(
            bank_id=receiver_bank_id,
            payment_id=payment_id,
            side="credit",
            account_id=event.to_account,
            counterparty_bank_id=sender_bank_id,
            sender_bank_id=sender_bank_id,
            receiver_bank_id=receiver_bank_id,
            amount_paise=event.amount_paise,
            event_time=event_time,
            status=event.status,
            failure_domain=event.failure_domain,
            payee_type=resolved_payee_type,
            channel=config.channel_value,
            counterparty_vpa=None,
            sender_device_fingerprint=None,
            sender_os_generation=None,
        )
        check_bank_visible(asdict(credit_record), BANK_VISIBLE_PAYMENT_FIELDS)
        sides.append(credit_record)

    return sides


def project_run(
    output: InteractiveWorldOutput,
    label_events: Sequence[LabelEvent] = (),
) -> BankFeed:
    """
    WHAT: Projects an entire interactive simulation run into tenant-isolated BankFeed event streams.
    It compiles payment sides, account security lifecycle events, mid-run account openings, and
    external fraud complaints, grouping them by commercial bank identifier and ordering by time.

    WHY: Serves as the primary batch feed adapter converting world simulation outputs into authentic
    per-bank telemetry feeds, strictly verifying that every record conforms to bank-visible schemas.

    Chronological Ordering and Tie-Break Rules:
        - Payment sides: Sorted by (event_time, payment_id, side), placing 'debit' before 'credit'.
        - Account events: Sorted by (event_time, event_id), breaking timestamp ties by event_id.
        - Complaints: Sorted by (received_at, complaint_id), breaking ties by complaint_id.

    Args:
        output: Complete InteractiveWorldOutput containing world ledgers, plan, and run results.
        label_events: Optional sequence of LabelEvent complaints filed during or after the simulation run.

    Returns:
        Immutable BankFeed instance containing per-bank event tuples and unmapped device event counts.

    Raises:
        ValueError: If an account, bank index, or unmapped event kind violates validation rules.
    """
    config = load_bank_feed_config()
    clock_start = output.plan.clock.start_date
    run_id = output.result.run_id

    # 1. Initialize per-bank accumulator lists and unmapped kind counters.
    all_bank_ids = set(config.bank_ids_by_index.values())
    payments_by_bank: dict[str, list[BankFeedPaymentSide]] = {b: [] for b in all_bank_ids}
    account_events_by_bank: dict[str, list[BankFeedAccountEvent]] = {b: [] for b in all_bank_ids}
    complaints_by_bank: dict[str, list[BankFeedComplaint]] = {b: [] for b in all_bank_ids}
    dropped_device_events_by_kind: dict[str, int] = {k: 0 for k in config.unmapped_gang_device_kinds}

    # 2. Build account-to-bank and person-to-account lookups for rapid O(1) resolution.
    account_to_bank: dict[str, str] = {}
    for ledger in output.world.ledgers:
        b_id = config.bank_ids_by_index[ledger.bank_index]
        for acc in ledger.accounts:
            account_to_bank[acc] = b_id

    for opened in output.result.opened_accounts:
        op_acc = opened[1]
        op_b_idx = opened[4]
        account_to_bank[op_acc] = config.bank_ids_by_index[op_b_idx]

    people_by_id = {p.person_id: p for p in output.world.people}

    # 3. Project all payment events into debit and credit sides across banks.
    for idx, event in enumerate(output.result.run_result.events):
        pid = output.result.payment_ids[idx]
        stamp = output.result.run_result.event_sender_devices[idx]
        sides = project_payment(event, pid, stamp, clock_start, config)
        for side in sides:
            payments_by_bank[side.bank_id].append(side)

    # 4. Project genuine citizen device events.
    device_position = 0
    for dev_ev in output.result.applied_device_events:
        if dev_ev.person_id not in people_by_id:
            raise ValueError(f"Device event references unknown person '{dev_ev.person_id}'")
        person = people_by_id[dev_ev.person_id]
        account_id = person.account_id
        bank_id = config.bank_ids_by_index[person.bank_index]

        if dev_ev.kind not in config.device_event_kind_map:
            dropped_device_events_by_kind[dev_ev.kind] = dropped_device_events_by_kind.get(dev_ev.kind, 0) + 1
            continue

        target_kind = config.device_event_kind_map[dev_ev.kind]
        event_time = world_minute_to_ist(dev_ev.minute, clock_start)
        event_id = f"{run_id}-dev-{device_position:06d}"
        device_position += 1

        details = {
            "platform": dev_ev.platform,
            "fingerprint_before": dev_ev.fingerprint_before,
            "fingerprint_after": dev_ev.fingerprint_after,
            "os_generation_before": dev_ev.os_generation_before,
            "os_generation_after": dev_ev.os_generation_after,
        }
        acct_ev = BankFeedAccountEvent(
            bank_id=bank_id,
            event_id=event_id,
            account_id=account_id,
            kind=target_kind,
            event_time=event_time,
            details=details,
        )
        check_bank_visible(asdict(acct_ev), BANK_VISIBLE_ACCOUNT_EVENT_FIELDS)
        account_events_by_bank[bank_id].append(acct_ev)

    # 5. Project applied gang device events (only mapped kinds with an associated account).
    for gang_ev in output.result.applied_gang_device_events:
        if gang_ev.account_id is None or gang_ev.kind not in config.device_event_kind_map:
            dropped_device_events_by_kind[gang_ev.kind] = dropped_device_events_by_kind.get(gang_ev.kind, 0) + 1
            continue

        if gang_ev.account_id not in account_to_bank:
            raise ValueError(f"Gang device event references unknown account '{gang_ev.account_id}'")
        bank_id = account_to_bank[gang_ev.account_id]
        target_kind = config.device_event_kind_map[gang_ev.kind]
        event_time = world_minute_to_ist(gang_ev.minute, clock_start)
        event_id = f"{run_id}-dev-{device_position:06d}"
        device_position += 1

        details = {
            "platform": gang_ev.platform,
            "fingerprint": gang_ev.fingerprint,
            "os_generation": gang_ev.os_generation,
        }
        acct_ev = BankFeedAccountEvent(
            bank_id=bank_id,
            event_id=event_id,
            account_id=gang_ev.account_id,
            kind=target_kind,
            event_time=event_time,
            details=details,
        )
        check_bank_visible(asdict(acct_ev), BANK_VISIBLE_ACCOUNT_EVENT_FIELDS)
        account_events_by_bank[bank_id].append(acct_ev)

    # 6. Project mid-run opened accounts into account_opened events.
    for opened in output.result.opened_accounts:
        op_minute = opened[0]
        op_account_id = opened[1]
        op_bank_index = opened[4]
        bank_id = config.bank_ids_by_index[op_bank_index]
        event_time = world_minute_to_ist(op_minute, clock_start)
        event_id = f"{run_id}-open-{op_account_id}"

        acct_ev = BankFeedAccountEvent(
            bank_id=bank_id,
            event_id=event_id,
            account_id=op_account_id,
            kind="account_opened",
            event_time=event_time,
            details={},
        )
        check_bank_visible(asdict(acct_ev), BANK_VISIBLE_ACCOUNT_EVENT_FIELDS)
        account_events_by_bank[bank_id].append(acct_ev)

    # 7. Project customer fraud complaints into complainant bank records.
    for label_ev in label_events:
        complainant_bank_id = config.bank_ids_by_index[label_ev.complainant_bank_index]
        reported_bank_id = config.bank_ids_by_index[label_ev.reported_bank_index]
        received_at = world_minute_to_ist(label_ev.arrival_minute, clock_start)

        complaint = BankFeedComplaint(
            bank_id=complainant_bank_id,
            complaint_id=label_ev.label_id,
            complainant_account=label_ev.complainant_account,
            reported_account=label_ev.reported_account,
            reported_bank_id=reported_bank_id,
            payment_ids=tuple(label_ev.payment_ids),
            claimed_type=label_ev.claimed_type,
            received_at=received_at,
        )
        check_bank_visible(asdict(complaint), BANK_VISIBLE_COMPLAINT_FIELDS)
        complaints_by_bank[complainant_bank_id].append(complaint)

    # 8. Sort per-bank collections chronologically with stable tie-breaks and freeze into tuples.
    sorted_payments = {
        b: tuple(sorted(payments_by_bank[b], key=lambda s: (s.event_time, s.payment_id, s.side)))
        for b in sorted(all_bank_ids)
    }
    sorted_account_events = {
        b: tuple(sorted(account_events_by_bank[b], key=lambda e: (e.event_time, e.event_id)))
        for b in sorted(all_bank_ids)
    }
    sorted_complaints = {
        b: tuple(sorted(complaints_by_bank[b], key=lambda c: (c.received_at, c.complaint_id)))
        for b in sorted(all_bank_ids)
    }

    return BankFeed(
        payments_by_bank=sorted_payments,
        account_events_by_bank=sorted_account_events,
        complaints_by_bank=sorted_complaints,
        dropped_device_events_by_kind=dropped_device_events_by_kind,
    )

