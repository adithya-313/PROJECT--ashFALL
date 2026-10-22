"""
WHAT: The small records the interactive runner takes in and hands back: the fraud switch (RunnerSettings), the
fraud inputs (extra payments and outside events), the view of a payment a bank is shown (PaymentView), the hook's
possible answers, and the results of a run (InteractiveRunResult and InteractiveWorldOutput).

WHY: Keeping the records apart from the loop means tests, the showcase and later levels can build and read them
without importing the loop. Every record is frozen, so a result cannot be changed after the run that made it.

TERM: fraud switch: RunnerSettings.fraud_enabled; False (the default) runs Level 11's world exactly as it was.
TERM: hook: a function the runner asks before each payment, standing in for a bank; it answers allow, hold or
      decline, e.g. allow_all, which always answers "allow".
TERM: hold: the bank keeps the payment waiting; nothing happens until an outside event resumes it.
TERM: decline: the bank refuses the payment; it is recorded as failed and no money moves.
TERM: planned payment id: the name of one payment as the world planned it, shared by all its pieces when it is split
      over the per-payment limit, e.g. "66db9aa439b092ee-p-000017".
TERM: payment id: the name of one recorded event, e.g. "66db9aa439b092ee-000042" for the 43rd event.
TERM: event source: whether an event's true origin was the world's own plan ("planned") or fraud inputs ("fraud");
      known to the world only, never shown to a bank.
TERM: resumed: an event whose execution was held by the bank and resumed later by an outside event;
      recorded in events_resumed.
TERM: fraud input index: the 0-based position of a fraud payment in FraudInputs.payments; None for planned payments.
TERM: frozen: a record whose fields cannot be changed after it is made.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from world.build_world import World
from world.calendar_runner import PlannedPayment, RunResult
from world.cash_withdrawals import CashWithdrawal, CashWithdrawalRecord
from world.device_events import DeviceEvent
from world.episodes import Episode
from world.gang_devices import GangDeviceEvent
from world.gang_feed import GangHolding
from world.outside_events import OutsideEvent
from world.rented_owner_activity import PausedOwnerPayment
from world.truth_records import TruthFacts
from world.world_plan import WorldPlan

# The three answers a hook may give, and the same three as a tuple for checking answers at run time.
HookDecision = Literal["allow", "hold", "decline"]
HOOK_DECISIONS: tuple[str, ...] = ("allow", "hold", "decline")


@dataclass(frozen=True)
class RunnerSettings:
    """
    WHAT: The fraud switch. With fraud_enabled False (the default) the runner hands the world to Level 11's run_world
    unchanged; with True it runs the ordered event queue that can hold, decline and block payments.

    WHY: Ground rule 13 asks for a switch with a fallback. Off must be the safe default, so nothing changes for anyone
    who does not ask for fraud.

    Args:
        fraud_enabled: True to run the fraud-on loop, False to fall back to Level 11.
        account_openings_enabled: True to schedule genuine mid-run account openings, False by default.

    Returns:
        A frozen RunnerSettings.

    Example:
        RunnerSettings() has fraud_enabled False; RunnerSettings(fraud_enabled=True) turns the loop on.
    """

    fraud_enabled: bool = False
    account_openings_enabled: bool = False
    gang_reactions_enabled: bool = False


@dataclass(frozen=True)
class IllicitMoneyRule:
    """
    Controls how illicit money flows through one planned fraud payment.

    input_index identifies the target payment in FraudInputs.payments.
    spends_illicit limits the payment to the sender's current illicit balance.
    credits_illicit credits successful transfer amounts to the receiver's illicit balance.
    requires_credit_from ties spending to funds credited specifically from that sender account.
    """

    input_index: int
    spends_illicit: bool
    credits_illicit: bool
    requires_credit_from: str | None = None

    def __post_init__(self) -> None:
        """Enforces that requires_credit_from is only set when spends_illicit is True."""
        if self.requires_credit_from is not None and not self.spends_illicit:
            raise ValueError("requires_credit_from is only allowed on a rule with spends_illicit=True.")


@dataclass(frozen=True)
class IllicitSeed:
    """
    Initial illicit money balance allocated to an account before simulation begins.

    account_id specifies the target account identifier.
    amount_paise specifies the starting illicit balance in paise.
    """

    account_id: str
    amount_paise: int


@dataclass(frozen=True)
class ForwardAdjustment:
    """
    Audit record for a fraud forward payment capped or skipped by the illicit money guard.

    Records planned versus available paise, accounts involved, and the adjustment reason.
    """

    planned_payment_id: str
    minute: int
    from_account: str
    to_account: str
    planned_paise: int
    available_paise: int
    reason: str


@dataclass(frozen=True)
class FraudInputs:
    """
    WHAT: Inputs for a world run with the fraud switch on: fraud payments to interleave with genuine ones, outside
    events to apply, hidden truth facts, synthesized gang device events, and illicit money rules and seeds.

    WHY: Fraud payments use the same PlannedPayment record as genuine ones and a purpose from Level 11's list, so the
    events they create look like any other event. Which payments are fraud is kept by the world, not in the purpose.

    Args:
        payments: Extra payments, in the order they should be tried at equal minutes.
        outside_events: Liens, debit holds, releases and resumes to apply during the run.
        truth: Hidden truth facts (roles, device control spans, payment stories) associated with the run.
        gang_device_events: Synthesized gang device lifecycle events to queue and apply.
        money_rules: Rules governing illicit funds movement for specific fraud payments.
        illicit_seeds: Initial illicit balances allocated to designated source accounts.

    Returns:
        A frozen FraudInputs.

    Example:
        FraudInputs() is empty; FraudInputs(outside_events=(OutsideEvent(1440, "debit_hold", "acc_a"),)) blocks
        acc_a from the start of day 2.
    """

    payments: tuple[PlannedPayment, ...] = ()
    outside_events: tuple[OutsideEvent, ...] = ()
    truth: TruthFacts = field(default_factory=TruthFacts)
    gang_device_events: tuple[GangDeviceEvent, ...] = ()
    money_rules: tuple[IllicitMoneyRule, ...] = ()
    illicit_seeds: tuple[IllicitSeed, ...] = ()
    withdrawals: tuple[CashWithdrawal, ...] = ()
    gang_holdings: tuple[GangHolding, ...] = ()


@dataclass(frozen=True)
class PaymentView:
    """
    WHAT: What the hook is shown about one payment: its planned payment id, minute, both accounts and the amount.
    It has no field saying where the payment came from.

    WHY: The hook stands in for a bank, and a bank never knows which payments are fraud. Leaving that field out of the
    view makes it impossible for any hook to cheat by reading the answer.

    Args:
        planned_payment_id: The payment's name, e.g. "66db9aa439b092ee-p-000017".
        minute: When it is being tried, e.g. 1630.
        from_account: The paying account.
        to_account: The receiving account.
        amount_paise: The full amount before any split, in paise.

    Returns:
        A frozen PaymentView.

    Example:
        PaymentView("66db9aa439b092ee-p-000000", 211, "acc_person_000001", "acc_person_000003", 4844).
    """

    planned_payment_id: str
    minute: int
    from_account: str
    to_account: str
    amount_paise: int


def allow_all(payment: PaymentView) -> HookDecision:
    """
    WHAT: The default hook: it answers "allow" for every payment, whatever the view says.

    WHY: With this hook and no fraud inputs, the fraud-on loop must give exactly Level 11's run. That equality is the
    test that proves the loop adds nothing on its own.

    Args:
        payment: The PaymentView being asked about; not used.

    Returns:
        Always "allow".

    Example:
        allow_all(PaymentView("run-p-000000", 211, "acc_a", "acc_b", 4844)) returns "allow".
    """
    # A bank that never intervenes.
    return "allow"


@dataclass(frozen=True)
class GangCancellation:
    """
    WHAT: Audit record for a payment or withdrawal cancelled by a gang's reactive decision engine.
    WHY: Amendment A-017 requires tracking all actions prevented by a gang mind upon observing account friction.

    Args:
        minute: Simulation minute when the cancellation occurred.
        gang: Identifier of the syndicate deciding to cancel.
        accounts: Tuple of account identifiers involved in the cancelled transfer or withdrawal.
        amount: Integer amount of the cancelled action in paise.
        reason: Explanation of the cancellation decision.
    """

    minute: int
    gang: str
    accounts: tuple[str, ...]
    amount: int
    reason: str

    @property
    def gang_id(self) -> str:
        """Exposes the syndicate identifier alias for downstream report compatibility."""
        return self.gang

    @property
    def amount_paise(self) -> int:
        """Returns the cancelled transaction monetary amount in integer paise units."""
        return self.amount


@dataclass(frozen=True)
class InteractiveRunResult:
    """
    WHAT: Everything one interactive run produced. run_result is shaped exactly like Level 11's RunResult; the other
    lists line up with run_result.events one to one (payment ids, planned payment ids, event sources, events resumed,
    and fraud input indices), followed by the device events that were applied, the payments still held at the end,
    the lien and hold log, the run id and the money created by accounts opened during the run.

    WHY: Keeping Level 11's part unchanged lets every Level 11 check and hash read it as before. The extra lists sit
    beside it, so nothing Level 11 reads gains a new field.

    Args:
        run_result: Level 11-shaped result: events, counts, purposes, sender device stamps, device counts.
        payment_ids: One per event, e.g. "66db9aa439b092ee-000000" for the first.
        planned_payment_ids: One per event; all pieces of one split payment share it.
        event_sources: One per event: "planned" or "fraud". World side only.
        events_resumed: One boolean per event: True if the payment was held and resumed, False otherwise.
        applied_device_events: The device events applied during the run, in order.
        applied_gang_device_events: The gang device events applied during the run, in order.
        event_fraud_input_index: One per event: index in FraudInputs.payments for fraud events,
            None for planned payments.
        held_payments: Payments held by the hook and never resumed.
        encumbrance_log: (minute, action, encumbrance id) for every lien or hold placed or released.
        run_id: The run's name, e.g. "66db9aa439b092ee".
        money_created_paise: Opening balances of accounts registered during the run, in paise.
        truth: Hidden truth facts (roles, device control spans, payment stories) associated with the run.
        skipped_forwards: Adjustments where forwards were skipped due to lack of received illicit money.
        capped_forwards: Adjustments where forwards were capped to the illicit balance received.
        opened_accounts: (minute, account_id, owner_kind, opening_stream, bank_index, opening_balance_paise)
            for accounts registered mid-run.
        paused_owner_payments: Genuine payments of rented mules' owners that were paused (no event, no money moved).

    Returns:
        A frozen InteractiveRunResult.

    Example:
        A fraud-off tiny-world run (seed 101) has 2,915 events, so payment_ids, planned_payment_ids,
        event_sources, events_resumed and event_fraud_input_index each hold 2,915 entries.
    """

    run_result: RunResult
    payment_ids: list[str]
    planned_payment_ids: list[str]
    event_sources: list[str]
    events_resumed: list[bool]
    applied_device_events: list[DeviceEvent]
    applied_gang_device_events: list[GangDeviceEvent] = field(default_factory=list)
    event_fraud_input_index: list[int | None] = field(default_factory=list)
    held_payments: list[PaymentView] = field(default_factory=list)
    encumbrance_log: list[tuple[int, str, str]] = field(default_factory=list)
    run_id: str = ""
    money_created_paise: int = 0
    truth: TruthFacts = field(default_factory=TruthFacts)
    skipped_forwards: list[ForwardAdjustment] = field(default_factory=list)
    capped_forwards: list[ForwardAdjustment] = field(default_factory=list)
    opened_accounts: list[tuple[int, str, str, str, int, int]] = field(default_factory=list)
    paused_owner_payments: list[PausedOwnerPayment] = field(default_factory=list)
    cash_withdrawals: list[CashWithdrawalRecord] = field(default_factory=list)
    money_withdrawn_paise: int = 0
    cancelled_by_gang: list[GangCancellation] = field(default_factory=list)


@dataclass(frozen=True)
class InteractiveWorldOutput:
    """
    WHAT: A whole interactive world: the built world, its plan, the episodes and their report, the device schedule,
    the run result, the effective configuration the run id was made from, and the code version.

    WHY: Level 11's FastRunOutput keeps no device schedule. Later parts need it for their own device event output, and
    they need the effective configuration to rebuild or audit the run, so this record keeps both.

    Args:
        world: The World after the run (balances and devices as they ended).
        plan: The WorldPlan (calendar, festivals, incidents, failure settings).
        episodes: The life episodes drawn for ordinary people.
        episode_report: Counters from planning those episodes' payments.
        device_events: The full device schedule, in order.
        result: The InteractiveRunResult.
        effective_config: {"settings": ..., "world_configs": ...} as plain data.
        code_version: The version recorded for this run, e.g. "d41b176" or "unknown".

    Returns:
        A frozen InteractiveWorldOutput.

    Example:
        run_interactive_world(FastRunSettings(101, 60, 12, 61, "training")).result.run_result.events has 2,915 events.
    """

    world: World
    plan: WorldPlan
    episodes: list[Episode]
    episode_report: dict[str, Any]
    device_events: tuple[DeviceEvent, ...]
    result: InteractiveRunResult
    effective_config: dict[str, Any]
    code_version: str
