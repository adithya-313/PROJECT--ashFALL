# File: world/bank_connectors.py
# Purpose: Connectors and hook dispatching planned payments and feed projections to commercial bank transports.
"""
WHAT: Connectors and hook dispatching planned payments and feed projections to commercial bank transports.
It bridges the world simulation with commercial bank REST API transports (in-process or HTTP).

WHY: Translates PaymentView hooks into real-time decision requests and delivers post-simulation
BankFeed projections to per-bank transports while tracking operational metrics and enforcing policy.
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import orjson

from core.config_validator import load_yaml
from core.event_clock import world_minute_to_ist
from world.bank_feed import BankFeed, project_run, resolve_bank_id
from world.bank_feed_config import (
    DEFAULT_BANK_API_CONFIG_PATH,
    BankFeedConfig,
    load_bank_feed_config,
)
from world.interactive_records import (
    HookDecision,
    InteractiveWorldOutput,
    PaymentView,
)
from world.interactive_runner import (
    FastRunSettings,
    RunnerSettings,
    prepare_world,
    run_interactive_world,
)
from world.label_records import LabelEvent

VALID_BANK_MODES: frozenset[str] = frozenset({"shadow", "live"})


@runtime_checkable
class TransportOutcome(Protocol):
    """
    WHAT: Structural protocol defining the outcome contract returned by bank transport post operations.
    WHY: Allows bank connectors to process outcomes from both InProcessBankClient and SignedHttpBankClient
    without importing api/ packages into world/.
    """

    status: int
    body: dict[str, Any]
    replayed: bool


@runtime_checkable
class BankTransport(Protocol):
    """
    WHAT: Structural protocol defining the transport interface for submitting bank API requests.
    WHY: Decouples the world simulation connectors from specific wire or database transport implementations.
    """

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> TransportOutcome:
        """Submits a POST request to the bank endpoint."""
        ...


def load_bank_modes(path: Path | str | None = None) -> dict[str, str]:
    """
    WHAT: Reads bank operating modes directly from config/bank_api.yaml without importing the api package.
    WHY: Enforces that every registered bank in bank_api.banks is assigned a verified mode ('shadow' or 'live').

    Args:
        path: Optional path to bank_api.yaml; defaults to config/bank_api.yaml.

    Returns:
        Mapping of bank identifier to operating mode string.

    Raises:
        ValueError: If config is missing, malformed, any bank lacks a mode, or an unknown mode is encountered.
    """
    target_path = Path(path) if path is not None else DEFAULT_BANK_API_CONFIG_PATH
    if not target_path.is_file():
        raise ValueError(f"Bank API configuration file not found: {target_path}")

    raw_data: Any = load_yaml(target_path)
    if not isinstance(raw_data, dict) or "bank_api" not in raw_data:
        raise ValueError(f"Bank API configuration '{target_path}' must contain a top-level 'bank_api' key")

    bank_api_data = raw_data["bank_api"]
    if "banks" not in bank_api_data or not isinstance(bank_api_data["banks"], list):
        raise ValueError("Bank API configuration missing 'banks' list")
    banks = bank_api_data["banks"]

    if "bank_modes" not in bank_api_data or not isinstance(bank_api_data["bank_modes"], dict):
        raise ValueError("Bank API configuration missing 'bank_modes' section")

    modes_raw = bank_api_data["bank_modes"].get("value")
    if not isinstance(modes_raw, dict):
        raise ValueError("bank_modes must have a 'value' mapping")

    bank_modes: dict[str, str] = {}
    for b in banks:
        bank_str = str(b)
        if bank_str not in modes_raw:
            raise ValueError(f"Missing bank mode for configured bank {bank_str!r}")
        mode_val = str(modes_raw[bank_str])
        if mode_val not in VALID_BANK_MODES:
            raise ValueError(
                f"Invalid mode {mode_val!r} for bank {bank_str!r}; expected one of {sorted(VALID_BANK_MODES)}"
            )
        bank_modes[bank_str] = mode_val

    return bank_modes


def load_max_identifier_length(path: Path | str | None = None) -> int:
    """
    WHAT: Reads max_identifier_length directly from config/bank_api.yaml without importing the api package.
    WHY: Used to bound idempotency key lengths across connector operations.

    Args:
        path: Optional path to bank_api.yaml; defaults to config/bank_api.yaml.

    Returns:
        Maximum allowed identifier length integer.

    Raises:
        ValueError: If the configuration file or setting is missing or not a positive integer.
    """
    target_path = Path(path) if path is not None else DEFAULT_BANK_API_CONFIG_PATH
    if not target_path.is_file():
        raise ValueError(f"Bank API configuration file not found: {target_path}")

    raw_data: Any = load_yaml(target_path)
    if not isinstance(raw_data, dict) or "bank_api" not in raw_data:
        raise ValueError(f"Bank API configuration '{target_path}' must contain a top-level 'bank_api' key")

    bank_api_data = raw_data["bank_api"]
    raw_val = bank_api_data.get("max_identifier_length", {}).get("value")
    if not isinstance(raw_val, int) or raw_val <= 0:
        raise ValueError("max_identifier_length.value must be a positive integer")
    return raw_val


def make_decision_idempotency_key(planned_payment_id: str, max_length: int) -> str:
    """
    WHAT: Formats a deterministic idempotency key for payment decision requests.
    WHY: Ensures identical planned payment evaluations yield consistent idempotency identity across retries.

    Args:
        planned_payment_id: Identifier of the planned payment.
        max_length: Upper bound on key length from bank API policy.

    Returns:
        Formatted idempotency key string 'dec:<planned_payment_id>'.

    Raises:
        ValueError: If key length exceeds max_length.
    """
    key = f"dec:{planned_payment_id}"
    if len(key) > max_length:
        raise ValueError(f"Decision idempotency key {key!r} exceeds max length {max_length}")
    return key


def make_transaction_idempotency_key(payment_id: str, side: str, max_length: int) -> str:
    """
    WHAT: Formats a deterministic idempotency key for transaction side records.
    WHY: Dispatches debit and credit legs of a payment with distinct, reproducible idempotency keys.

    Args:
        payment_id: Base payment identifier.
        side: Transaction side ('debit' or 'credit').
        max_length: Upper bound on key length from bank API policy.

    Returns:
        Formatted idempotency key string 'tx:<payment_id>:<side>'.

    Raises:
        ValueError: If key length exceeds max_length.
    """
    key = f"tx:{payment_id}:{side}"
    if len(key) > max_length:
        raise ValueError(f"Transaction idempotency key {key!r} exceeds max length {max_length}")
    return key


def make_account_event_idempotency_key(event_id: str, max_length: int) -> str:
    """
    WHAT: Formats a deterministic idempotency key for account lifecycle events.
    WHY: Protects account security and lifecycle event submissions against duplicate dispatch.

    Args:
        event_id: Unique event identifier.
        max_length: Upper bound on key length from bank API policy.

    Returns:
        Formatted idempotency key string 'acct:<event_id>'.

    Raises:
        ValueError: If key length exceeds max_length.
    """
    key = f"acct:{event_id}"
    if len(key) > max_length:
        raise ValueError(f"Account event idempotency key {key!r} exceeds max length {max_length}")
    return key


def make_complaint_idempotency_key(complaint_id: str, max_length: int) -> str:
    """
    WHAT: Formats a deterministic idempotency key for fraud complaints.
    WHY: Enforces idempotency when reporting customer complaints to receiving banks.

    Args:
        complaint_id: Unique complaint identifier.
        max_length: Upper bound on key length from bank API policy.

    Returns:
        Formatted idempotency key string 'cmp:<complaint_id>'.

    Raises:
        ValueError: If key length exceeds max_length.
    """
    key = f"cmp:{complaint_id}"
    if len(key) > max_length:
        raise ValueError(f"Complaint idempotency key {key!r} exceeds max length {max_length}")
    return key


class MockBankConnectors:
    """
    WHAT: Bridges the simulated world and commercial bank API transports.
    WHY: Translates PaymentView hooks into real-time decision requests and delivers
    post-simulation BankFeed projections to per-bank transports while tracking metrics.
    """

    def __init__(
        self,
        transports: dict[str, BankTransport],
        modes: dict[str, str],
        account_bank_map: dict[str, str],
        clock_start: date,
        feed_config: BankFeedConfig,
        decision_timeout_seconds: float,
        loop: asyncio.AbstractEventLoop,
        max_identifier_length: int | None = None,
    ) -> None:
        """
        Initialises mock bank connectors with required transports and policy configuration.

        Args:
            transports: Mapping of bank identifier to BankTransport implementation.
            modes: Mapping of bank identifier to active operating mode ('shadow' or 'live').
            account_bank_map: Mapping of account identifiers to their owning bank identifier.
            clock_start: Calendar start date of the simulation plan.
            feed_config: Validated BankFeedConfig instance.
            decision_timeout_seconds: Timeout for decision calls before falling back to allow.
            loop: Active asyncio event loop used for cross-thread coroutine scheduling.
            max_identifier_length: Optional key length limit; read from config if None.
        """
        self.transports = transports
        self.modes = modes
        self.account_bank_map = account_bank_map
        self.clock_start = clock_start
        self.feed_config = feed_config
        self.decision_timeout_seconds = decision_timeout_seconds
        self.loop = loop
        self.max_identifier_length = (
            max_identifier_length if max_identifier_length is not None else load_max_identifier_length()
        )

        self.decisions_sent: int = 0
        self.decisions_replayed: int = 0
        self.decision_failures: list[tuple[str, str, str]] = []
        self.live_fallbacks: int = 0
        self.sent_by_kind: dict[str, int] = {"transaction": 0, "account_event": 0, "complaint": 0}
        self.replayed_by_kind: dict[str, int] = {"transaction": 0, "account_event": 0, "complaint": 0}
        self.seen_accounts: set[str] = set()

    def make_hook(self) -> Callable[[PaymentView], HookDecision]:
        """
        WHAT: Constructs a synchronous hook callable by run_interactive_world for each planned payment.
        WHY: Intercepts planned payments, maps sender/receiver banks, builds decision requests, and
        dispatches them across the bank transport while respecting shadow vs live mode policies.

        Returns:
            Synchronous function accepting PaymentView and returning 'allow', 'hold', or 'decline'.
        """

        def hook(payment: PaymentView) -> HookDecision:
            """
            Synchronously evaluates a planned payment by dispatching a decision request to the bank transport.
            """
            # 1. Resolve sending and receiving bank identifiers from account mappings.
            if payment.from_account not in self.account_bank_map:
                raise ValueError(f"Sender account {payment.from_account!r} is not in account_bank_map")
            sender_bank_id = self.account_bank_map[payment.from_account]

            if payment.to_account not in self.account_bank_map:
                raise ValueError(f"Receiver account {payment.to_account!r} is not in account_bank_map")
            receiver_bank_id = self.account_bank_map[payment.to_account]

            self.seen_accounts.add(payment.from_account)
            self.seen_accounts.add(payment.to_account)

            # 2. Derive bank-visible payee classification from account prefix.
            resolved_payee_type: str | None = None
            for prefix, ptype in self.feed_config.payee_type_by_account_prefix.items():
                if payment.to_account.startswith(prefix):
                    resolved_payee_type = ptype
                    break
            if resolved_payee_type is None:
                raise ValueError(
                    f"Receiver account {payment.to_account!r} has unrecognized prefix; "
                    f"expected one of {sorted(self.feed_config.payee_type_by_account_prefix.keys())}"
                )

            # 3. Build DecisionRequest payload and deterministic idempotency key.
            event_time_dt = world_minute_to_ist(payment.minute, self.clock_start)
            body_dict: dict[str, Any] = {
                "payment_id": payment.planned_payment_id,
                "sender_account_id": payment.from_account,
                "receiver_bank_id": receiver_bank_id,
                "amount_paise": payment.amount_paise,
                "event_time": event_time_dt.isoformat(),
                "channel": self.feed_config.channel_value,
                "payee_type": resolved_payee_type,
            }
            body_bytes = orjson.dumps(body_dict)
            idempotency_key = make_decision_idempotency_key(
                payment.planned_payment_id, self.max_identifier_length
            )

            if sender_bank_id not in self.transports:
                raise ValueError(f"No transport configured for sender bank {sender_bank_id!r}")
            transport = self.transports[sender_bank_id]

            # 4. Dispatch decision request to event loop with timeout.
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    transport.post("/v1/decisions", body_bytes, idempotency_key),
                    self.loop,
                )
                outcome = fut.result(timeout=self.decision_timeout_seconds)
            except (TimeoutError, concurrent.futures.TimeoutError) as exc:
                fut.cancel()
                self.decision_failures.append((payment.planned_payment_id, type(exc).__name__, str(exc)))
                return "allow"
            except Exception as exc:
                fut.cancel()
                self.decision_failures.append((payment.planned_payment_id, type(exc).__name__, str(exc)))
                return "allow"

            # 5. Record delivery metric.
            if outcome.replayed:
                self.decisions_replayed += 1
            else:
                self.decisions_sent += 1

            # 6. Verify mode response against configured bank mode.
            if not isinstance(outcome.body, dict):
                raise ValueError(f"Decision response body must be a dictionary, got: {type(outcome.body)}")
            if "mode" not in outcome.body:
                raise ValueError(f"Decision response body missing 'mode': {outcome.body!r}")
            resp_mode = str(outcome.body["mode"])

            if sender_bank_id not in self.modes:
                raise ValueError(f"No mode configured for sender bank {sender_bank_id!r}")
            expected_mode = self.modes[sender_bank_id]

            if resp_mode != expected_mode:
                raise ValueError(
                    f"Mode mismatch for bank {sender_bank_id!r}: configured mode is {expected_mode!r}, "
                    f"but response reported mode {resp_mode!r}"
                )

            # 7. Evaluate outcome based on operating mode.
            if expected_mode == "shadow":
                return "allow"

            if "decision" not in outcome.body:
                raise ValueError(f"Decision response body missing 'decision': {outcome.body!r}")
            decision_val = str(outcome.body["decision"])
            if decision_val == "allow":
                return "allow"

            # Issue 14-6: Live mode falls back to 'allow' until Level 19.
            self.live_fallbacks += 1
            return "allow"

        return hook

    async def send_feed(self, feed: BankFeed) -> None:
        """
        WHAT: Sends every transaction side, account event, and complaint across all banks in one global order.
        WHY: Delivers projected bank records to per-bank transports strictly ordered by (event_time, bank_id,
        record_kind, original_index) to preserve causality and avoid out-of-order race conditions.

        Args:
            feed: Validated BankFeed containing payment sides, account events, and complaints.

        Raises:
            Exception: If any transport call fails, terminating the feed transfer immediately.
        """
        items: list[tuple[datetime, str, int, int, str, Any]] = []

        # Gather payment sides.
        for bank_id, payment_sides in feed.payments_by_bank.items():
            for idx, p in enumerate(payment_sides):
                items.append((p.event_time, p.bank_id, 0, idx, "transaction", p))

        # Gather account lifecycle and device events.
        for bank_id, acct_events in feed.account_events_by_bank.items():
            for idx, a in enumerate(acct_events):
                items.append((a.event_time, a.bank_id, 1, idx, "account_event", a))

        # Gather fraud complaints.
        for bank_id, complaints in feed.complaints_by_bank.items():
            for idx, c in enumerate(complaints):
                items.append((c.received_at, c.bank_id, 2, idx, "complaint", c))

        # Sort strictly by (event_time, bank_id, record_kind, original_index).
        sorted_items = sorted(items, key=lambda x: (x[0], x[1], x[2], x[3]))

        for item in sorted_items:
            kind = item[4]
            rec = item[5]
            target_bank_id = item[1]

            if target_bank_id not in self.transports:
                raise ValueError(f"No transport configured for target bank {target_bank_id!r}")
            transport = self.transports[target_bank_id]

            if kind == "transaction":
                body_dict: dict[str, Any] = {
                    "payment_id": rec.payment_id,
                    "side": rec.side,
                    "account_id": rec.account_id,
                    "counterparty_vpa": rec.counterparty_vpa,
                    "counterparty_bank_id": rec.counterparty_bank_id,
                    "sender_bank_id": rec.sender_bank_id,
                    "receiver_bank_id": rec.receiver_bank_id,
                    "amount_paise": rec.amount_paise,
                    "event_time": rec.event_time.isoformat(),
                    "status": rec.status,
                    "failure_domain": rec.failure_domain,
                    "payee_type": rec.payee_type,
                    "channel": rec.channel,
                    "sender_device_fingerprint": rec.sender_device_fingerprint,
                    "sender_os_generation": rec.sender_os_generation,
                }
                body_bytes = orjson.dumps(body_dict)
                idempotency_key = make_transaction_idempotency_key(
                    rec.payment_id, rec.side, self.max_identifier_length
                )
                outcome = await transport.post("/v1/transactions", body_bytes, idempotency_key)
                if outcome.replayed:
                    self.replayed_by_kind["transaction"] += 1
                else:
                    self.sent_by_kind["transaction"] += 1

            elif kind == "account_event":
                body_dict = {
                    "event_id": rec.event_id,
                    "account_id": rec.account_id,
                    "kind": rec.kind,
                    "event_time": rec.event_time.isoformat(),
                    "details": rec.details,
                }
                body_bytes = orjson.dumps(body_dict)
                idempotency_key = make_account_event_idempotency_key(
                    rec.event_id, self.max_identifier_length
                )
                outcome = await transport.post("/v1/account-events", body_bytes, idempotency_key)
                if outcome.replayed:
                    self.replayed_by_kind["account_event"] += 1
                else:
                    self.sent_by_kind["account_event"] += 1

            elif kind == "complaint":
                body_dict = {
                    "complaint_id": rec.complaint_id,
                    "complainant_account": rec.complainant_account,
                    "reported_account": rec.reported_account,
                    "reported_bank_id": rec.reported_bank_id,
                    "payment_ids": list(rec.payment_ids),
                    "claimed_type": rec.claimed_type,
                    "received_at": rec.received_at.isoformat(),
                }
                body_bytes = orjson.dumps(body_dict)
                idempotency_key = make_complaint_idempotency_key(
                    rec.complaint_id, self.max_identifier_length
                )
                outcome = await transport.post("/v1/complaints", body_bytes, idempotency_key)
                if outcome.replayed:
                    self.replayed_by_kind["complaint"] += 1
                else:
                    self.sent_by_kind["complaint"] += 1


@dataclass(frozen=True)
class ConnectorRunReport:
    """
    WHAT: Immutable summary report produced by running a world simulation through bank connectors.
    WHY: Aggregates counts of decisions sent/replayed/failed, live mode fallbacks, feed records delivered,
    unmapped hardware events dropped, active bank operating modes, and raw world simulation output.
    """

    decisions_sent: int
    decisions_replayed: int
    decision_failures: tuple[tuple[str, str, str], ...]
    live_fallbacks: int
    sent_by_kind: dict[str, int]
    replayed_by_kind: dict[str, int]
    dropped_device_events_by_kind: dict[str, int]
    modes: dict[str, str]
    output: InteractiveWorldOutput


async def run_world_through_banks(
    fast_settings: FastRunSettings,
    runner_settings: RunnerSettings,
    transports: dict[str, BankTransport],
    label_events: tuple[LabelEvent, ...] = (),
    feed_config: BankFeedConfig | None = None,
    bank_modes: dict[str, str] | None = None,
    decision_timeout_seconds: float | None = None,
) -> ConnectorRunReport:
    """
    WHAT: Executes a complete interactive world simulation dispatched through bank connectors.
    Prepares account-to-bank mappings, runs the simulation in a worker thread with the connector hook,
    verifies account consistency post-run, projects the bank feed, and dispatches feed events to transports.

    WHY: Serves as the high-level orchestration interface for end-to-end integration runs and parity tests.

    Args:
        fast_settings: Sizing and seeds for the simulation run.
        runner_settings: Runner settings including fraud enabled flag.
        transports: Mapping of bank identifier to BankTransport instance.
        label_events: Optional tuple of fraud complaint LabelEvent records.
        feed_config: Optional BankFeedConfig; defaults to loading config/bank_feed.yaml.
        bank_modes: Optional bank modes mapping; defaults to loading config/bank_api.yaml.
        decision_timeout_seconds: Optional decision timeout float override.

    Returns:
        ConnectorRunReport with complete metrics, delivery counts, and simulation output.

    Raises:
        ValueError: If account mappings mismatch between hook and post-run world, or transport calls fail.
    """
    cfg = feed_config if feed_config is not None else load_bank_feed_config()
    modes = bank_modes if bank_modes is not None else load_bank_modes()
    timeout = decision_timeout_seconds if decision_timeout_seconds is not None else cfg.decision_timeout_seconds

    prepared = prepare_world(fast_settings, None)
    clock_start = prepared.plan.clock.start_date

    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

    loop = asyncio.get_running_loop()
    connectors = MockBankConnectors(
        transports=transports,
        modes=modes,
        account_bank_map=account_bank_map,
        clock_start=clock_start,
        feed_config=cfg,
        decision_timeout_seconds=timeout,
        loop=loop,
    )
    hook = connectors.make_hook()

    output: InteractiveWorldOutput = await asyncio.to_thread(
        run_interactive_world,
        fast_settings,
        runner_settings,
        hook=hook,
    )

    post_world_accounts: dict[str, str] = {}
    for ledger in output.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, cfg)
        for acc in ledger.accounts:
            post_world_accounts[acc] = b_id

    for opened in output.result.opened_accounts:
        post_world_accounts[opened[1]] = resolve_bank_id(opened[4], cfg)

    for seen_acc in connectors.seen_accounts:
        if seen_acc not in post_world_accounts:
            raise ValueError(f"Account {seen_acc!r} seen by connector hook is missing from output.world")
        if post_world_accounts[seen_acc] != account_bank_map[seen_acc]:
            raise ValueError(
                f"Account {seen_acc!r} mapped to bank {account_bank_map[seen_acc]!r} during hook "
                f"but mapped to {post_world_accounts[seen_acc]!r} in output.world"
            )

    feed = project_run(output, label_events=label_events)
    await connectors.send_feed(feed)

    return ConnectorRunReport(
        decisions_sent=connectors.decisions_sent,
        decisions_replayed=connectors.decisions_replayed,
        decision_failures=tuple(connectors.decision_failures),
        live_fallbacks=connectors.live_fallbacks,
        sent_by_kind=dict(connectors.sent_by_kind),
        replayed_by_kind=dict(connectors.replayed_by_kind),
        dropped_device_events_by_kind=dict(feed.dropped_device_events_by_kind),
        modes=dict(modes),
        output=output,
    )
