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

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Sequence

from core.event_clock import world_minute_to_ist
from world.bank_feed_config import BankFeedConfig, load_bank_feed_config
from world.build_world import World
from world.cash_withdrawals import ALLOWED_CHANNELS
from world.device_runtime import DeviceStamp
from world.failures import FAILURE_DOMAINS, PaymentEvent
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
    balance_after_paise: int | None = None


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
        "balance_after_paise",
    }
)

ACCOUNT_OPENED_DETAIL_FIELDS: frozenset[str] = frozenset({"home_region"})


@dataclass(frozen=True)
class BankFeedWithdrawal:
    """
    WHAT: Represents a cash withdrawal (ATM or branch) visible to a commercial bank.
    WHY: Commercial banks observe cash extractions made on their accounts with running balances.
    """

    withdrawal_id: str
    bank_id: str
    account_id: str
    channel: str
    amount_paise: int
    status: str
    failure_domain: str | None
    event_time: datetime
    balance_after_paise: int | None


BANK_VISIBLE_WITHDRAWAL_FIELDS: frozenset[str] = frozenset(
    {
        "withdrawal_id",
        "bank_id",
        "account_id",
        "channel",
        "amount_paise",
        "status",
        "failure_domain",
        "event_time",
        "balance_after_paise",
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

GENUINE_DEVICE_DETAIL_FIELDS: frozenset[str] = frozenset(
    {
        "platform",
        "fingerprint_before",
        "fingerprint_after",
        "os_generation_before",
        "os_generation_after",
    }
)

GANG_DEVICE_DETAIL_FIELDS: frozenset[str] = frozenset(
    {
        "platform",
        "fingerprint",
        "os_generation",
    }
)


def resolve_bank_id(bank_index: int, config: BankFeedConfig) -> str:
    """
    WHAT: Resolves an integer simulation bank index to its registered commercial bank identifier.
    It verifies that bank_index exists in config.bank_ids_by_index and returns the mapped bank ID.

    WHY: Prevents bare KeyErrors and provides descriptive error messages identifying both the
    violating bank index and all valid recognized bank indexes across all projection pathways.

    Args:
        bank_index: Integer simulation bank index to resolve.
        config: Loaded BankFeedConfig providing the bank index mapping table.

    Returns:
        Mapped commercial bank identifier string (e.g. 'bank_0').

    Raises:
        ValueError: If bank_index is not found, naming the index and all known indexes.
    """
    if bank_index not in config.bank_ids_by_index:
        known_indexes = sorted(config.bank_ids_by_index.keys())
        raise ValueError(f"Unknown bank index {bank_index}; expected one of {known_indexes}")
    return config.bank_ids_by_index[bank_index]


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
    withdrawals_by_bank: dict[str, tuple[BankFeedWithdrawal, ...]] = field(default_factory=dict)



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


class BalanceReplay:
    """
    WHAT: Replayed running balances from bank ledgers.
    WHY: Associates each ledger entry with the account balance immediately after that entry posted.
    """

    def __init__(
        self,
        balances_by_transfer: dict[tuple[str, str], list[int]],
        balances_by_account: dict[str, int],
    ) -> None:
        """
        Initialises balance replay state tracking chronological account balances and occurrences.

        Args:
            balances_by_transfer: Mapping of (account_id, transfer_id) to list of posted balances.
            balances_by_account: Authoritative final balances for each account.
        """
        self._balances = balances_by_transfer
        self._cursors: dict[tuple[str, str], int] = {}
        self._balances_by_account = balances_by_account

    def get_balance_after(self, account_id: str, transfer_id: str) -> int:
        """
        Returns the running balance of account_id after the specific occurrence of transfer_id.
        Advances cursor so identical payments in the same minute get monotonic balances.
        """
        key = (account_id, transfer_id)
        if key not in self._balances:
            raise ValueError(f"No ledger entry found for account {account_id!r} and transfer {transfer_id!r}")
        idx = self._cursors.get(key, 0)
        lst = self._balances[key]
        if idx >= len(lst):
            raise ValueError(f"Exhausted occurrences for account {account_id!r} and transfer {transfer_id!r}")
        bal = lst[idx]
        self._cursors[key] = idx + 1
        return bal

    def get_withdrawal_balance_after(self, account_id: str, transfer_id: str) -> int:
        """
        Returns the final balance of account_id after all legs of the withdrawal transfer_id have posted.
        """
        key = (account_id, transfer_id)
        if key not in self._balances:
            raise ValueError(f"No ledger entry found for account {account_id!r} and withdrawal {transfer_id!r}")
        lst = self._balances[key]
        self._cursors[key] = len(lst)
        return lst[-1]

    def get_final_balance(self, account_id: str) -> int:
        """Returns the final balance of the account at simulation end."""
        if account_id not in self._balances_by_account:
            raise ValueError(f"Account {account_id!r} not found in balance replay")
        return self._balances_by_account[account_id]


def replay_balances(world: World) -> BalanceReplay:
    """
    WHAT: Walks ledger transactions in entry order, tracking per-account running balances.
    CRITICAL: Maps by (account_id, transfer_id) occurrence. At the end of the walk, asserts
    `running_balance[account] == ledger.balance_of(account)` for every account seen; if any mismatch,
    raises ValueError (self-check against silent divergence).
    """
    balances_by_transfer: dict[tuple[str, str], list[int]] = {}
    running_balances: dict[str, int] = {}

    for ledger in world.ledgers:
        for acc in ledger.accounts:
            running_balances[acc] = 0

        for entry in ledger.entries:
            acc = entry.account_id
            running_balances[acc] = running_balances[acc] + entry.amount_paise
            key = (acc, entry.transfer_id)
            balances_by_transfer.setdefault(key, []).append(running_balances[acc])

        for acc in ledger.accounts:
            actual = ledger.balance_of(acc)
            if running_balances[acc] != actual:
                raise ValueError(
                    f"Balance replay self-check divergence for account {acc!r}: "
                    f"calculated {running_balances[acc]} != actual {actual}"
                )

    return BalanceReplay(balances_by_transfer, dict(running_balances))


def project_payment(
    event: PaymentEvent,
    payment_id: str,
    stamp: DeviceStamp | None,
    clock_start: date,
    config: BankFeedConfig,
    balance_replay: BalanceReplay | None = None,
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
    sender_bank_id = resolve_bank_id(event.from_bank_index, config)
    receiver_bank_id = resolve_bank_id(event.to_bank_index, config)

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

    tx_id = f"tx_{event.minute}_{event.from_account}_{event.to_account}_{event.amount_paise}"
    if event.status == "success" and balance_replay is not None:
        debit_balance: int | None = balance_replay.get_balance_after(event.from_account, tx_id)
        credit_balance: int | None = balance_replay.get_balance_after(event.to_account, tx_id)
    else:
        debit_balance = None
        credit_balance = None

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
        balance_after_paise=debit_balance,
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
            balance_after_paise=credit_balance,
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
    withdrawals_by_bank: dict[str, list[BankFeedWithdrawal]] = {b: [] for b in all_bank_ids}
    dropped_device_events_by_kind: dict[str, int] = {k: 0 for k in config.unmapped_gang_device_kinds}

    # 2. Build account-to-bank and person-to-account lookups for rapid O(1) resolution.
    account_to_bank: dict[str, str] = {}
    for ledger in output.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, config)
        for acc in ledger.accounts:
            account_to_bank[acc] = b_id

    for opened in output.result.opened_accounts:
        op_acc = opened[1]
        op_b_idx = opened[4]
        account_to_bank[op_acc] = resolve_bank_id(op_b_idx, config)

    account_to_region: dict[str, int] = {}
    for p in output.world.people:
        account_to_region[p.account_id] = p.region
    for s in output.world.shops:
        account_to_region[s.account_id] = s.region

    balance_replay = replay_balances(output.world)
    people_by_id = {p.person_id: p for p in output.world.people}

    # 3. Project all payment events into debit and credit sides across banks.
    for idx, event in enumerate(output.result.run_result.events):
        pid = output.result.payment_ids[idx]
        stamp = output.result.run_result.event_sender_devices[idx]
        sides = project_payment(event, pid, stamp, clock_start, config, balance_replay)
        for side in sides:
            payments_by_bank[side.bank_id].append(side)

    # 4. Project genuine citizen device events.
    device_position = 0
    for dev_ev in output.result.applied_device_events:
        if dev_ev.person_id not in people_by_id:
            raise ValueError(f"Device event references unknown person '{dev_ev.person_id}'")
        person = people_by_id[dev_ev.person_id]
        account_id = person.account_id
        bank_id = resolve_bank_id(person.bank_index, config)

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
        check_bank_visible(details, GENUINE_DEVICE_DETAIL_FIELDS)
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
        check_bank_visible(details, GANG_DEVICE_DETAIL_FIELDS)
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
        bank_id = resolve_bank_id(op_bank_index, config)
        event_time = world_minute_to_ist(op_minute, clock_start)
        event_id = f"{run_id}-open-{op_account_id}"

        if op_account_id not in account_to_region:
            raise ValueError(f"Account {op_account_id!r} has no home region")
        region = account_to_region[op_account_id]
        details = {"home_region": region}
        check_bank_visible(details, ACCOUNT_OPENED_DETAIL_FIELDS)

        acct_ev = BankFeedAccountEvent(
            bank_id=bank_id,
            event_id=event_id,
            account_id=op_account_id,
            kind="account_opened",
            event_time=event_time,
            details=details,
        )
        check_bank_visible(asdict(acct_ev), BANK_VISIBLE_ACCOUNT_EVENT_FIELDS)
        account_events_by_bank[bank_id].append(acct_ev)

    # 6b. Emit account_opened events for existing accounts at start of run if configured.
    if config.emit_opening_for_existing_accounts:
        opened_ids = {opened[1] for opened in output.result.opened_accounts}
        existing_accounts = sorted({acc for ledger in output.world.ledgers for acc in ledger.accounts})

        start_time = world_minute_to_ist(0, clock_start)
        for acc_id in existing_accounts:
            if acc_id not in opened_ids:
                if acc_id not in account_to_bank:
                    raise ValueError(f"Account {acc_id!r} not found in account_to_bank mapping")
                b_id = account_to_bank[acc_id]
                if acc_id not in account_to_region:
                    raise ValueError(f"Account {acc_id!r} has no home region")
                if acc_id not in output.account_opening_days:
                    raise ValueError(f"Account {acc_id!r} missing from account_opening_days")
                days_open = output.account_opening_days[acc_id]
                event_time = start_time - timedelta(days=days_open)
                r = account_to_region[acc_id]
                open_details = {"home_region": r}
                check_bank_visible(open_details, ACCOUNT_OPENED_DETAIL_FIELDS)
                open_ev = BankFeedAccountEvent(
                    bank_id=b_id,
                    event_id=f"{run_id}-open-{acc_id}",
                    account_id=acc_id,
                    kind="account_opened",
                    event_time=event_time,
                    details=open_details,
                )
                check_bank_visible(asdict(open_ev), BANK_VISIBLE_ACCOUNT_EVENT_FIELDS)
                account_events_by_bank[b_id].append(open_ev)

    # 6c. Project cash withdrawals.
    for w in output.result.cash_withdrawals:
        if w.status == "skipped":
            continue
        if w.account not in account_to_bank:
            raise ValueError(f"Cash withdrawal {w.id} references unknown account {w.account!r}")
        b_id = account_to_bank[w.account]
        event_time = world_minute_to_ist(w.minute, clock_start)
        if w.channel not in ALLOWED_CHANNELS:
            raise ValueError(f"Withdrawal {w.id} has invalid channel {w.channel!r}; expected one of {ALLOWED_CHANNELS}")

        amount_paise = w.withdrawn if w.status == "success" else w.requested

        if w.status == "success":
            failure_domain = None
            balance_after_paise = balance_replay.get_withdrawal_balance_after(w.account, w.id)
        elif w.status == "failed":
            # R7: a technical failure must never be counted against the customer,
            # so an unknown domain is an error, not a customer failure.
            if w.domain is None or w.domain not in FAILURE_DOMAINS:
                raise ValueError(
                    f"Withdrawal {w.id} marked failed but has invalid or missing domain: {w.domain}"
                )
            failure_domain = w.domain
            balance_after_paise = None
        else:
            raise ValueError(f"Unknown withdrawal status: {w.status!r}")

        withdrawal_rec = BankFeedWithdrawal(
            withdrawal_id=w.id,
            bank_id=b_id,
            account_id=w.account,
            channel=w.channel,
            amount_paise=amount_paise,
            status=w.status,
            failure_domain=failure_domain,
            event_time=event_time,
            balance_after_paise=balance_after_paise,
        )
        check_bank_visible(asdict(withdrawal_rec), BANK_VISIBLE_WITHDRAWAL_FIELDS)
        withdrawals_by_bank[b_id].append(withdrawal_rec)

    # 7. Project customer fraud complaints into complainant bank records.
    for label_ev in label_events:
        complainant_bank_id = resolve_bank_id(label_ev.complainant_bank_index, config)
        reported_bank_id = resolve_bank_id(label_ev.reported_bank_index, config)
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
    sorted_withdrawals = {
        b: tuple(sorted(withdrawals_by_bank[b], key=lambda w: (w.event_time, w.withdrawal_id)))
        for b in sorted(all_bank_ids)
    }

    return BankFeed(
        payments_by_bank=sorted_payments,
        account_events_by_bank=sorted_account_events,
        complaints_by_bank=sorted_complaints,
        dropped_device_events_by_kind=dropped_device_events_by_kind,
        withdrawals_by_bank=sorted_withdrawals,
    )

