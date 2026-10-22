"""
WHAT: Cash withdrawals module for interactive runner (Level 12c C-4, D2).
Provides CashWithdrawal request and CashWithdrawalRecord result data structures,
channel validation (ATM and branch), ATM card daily limits, and execution
order: illicit money capping, ATM daily limit enforcement, encumbrance blocking,
failure decision evaluation on child 13, and ledger debiting split by UPI limit.

WHY: Mules extract stolen illicit balances into physical cash via ATMs or branch
counters. Tracking withdrawals maintains closed-loop money conservation
(opening + created - withdrawn) without polluting payments events or bank-facing truth.

ISSUE IDs: G2 (cash-out channels), 12-17 (split payments).
"""

from dataclasses import dataclass
from typing import Any

from world.amounts import split_into_legal_payments
from world.failure_decision import decide_failure

CHANNEL_ATM: str = "atm"
CHANNEL_BRANCH: str = "branch"
ALLOWED_CHANNELS: tuple[str, ...] = (CHANNEL_ATM, CHANNEL_BRANCH)


@dataclass(frozen=True)
class CashWithdrawal:
    """
    WHAT: A planned cash withdrawal from an account via ATM or branch channel.
    WHY: Represents money leaving the simulated banking system into physical currency.

    Args:
        minute: Simulation timestamp minute when withdrawal occurs.
        account_id: Account identifier being debited.
        amount_paise: Positive integer withdrawal amount requested in paise.
        channel: Cash withdrawal channel: 'atm' or 'branch'.
        card_daily_limit_paise: Optional daily limit ceiling for ATM debit cards.
    """

    minute: int
    account_id: str
    amount_paise: int
    channel: str
    card_daily_limit_paise: int | None = None


@dataclass(frozen=True)
class CashWithdrawalRecord:
    """
    WHAT: Audit record of one cash withdrawal attempt and its outcome.
    WHY: Tracks cash extraction across accounts and channels without modifying payment events.

    Args:
        id: Unique identifier for the withdrawal attempt, e.g. '<run_id>-w-000000'.
        minute: Simulation minute timestamp.
        account: Account identifier that attempted the withdrawal.
        channel: Channel used ('atm' or 'branch').
        requested: Requested withdrawal amount in paise.
        withdrawn: Actual amount debited and dispensed in paise.
        status: Execution status ('success', 'failed', or 'skipped').
        reason: Diagnostic failure reason string if not successful.
        domain: Failure attribution domain if failed.
    """

    id: str
    minute: int
    account: str
    channel: str
    requested: int
    withdrawn: int
    status: str
    reason: str | None = None
    domain: str | None = None

    @property
    def account_id(self) -> str:
        """Alias for account to match account_id naming conventions."""
        return self.account

    @property
    def requested_paise(self) -> int:
        """Alias for requested paise."""
        return self.requested

    @property
    def withdrawn_paise(self) -> int:
        """Alias for withdrawn paise."""
        return self.withdrawn

    @property
    def failure_domain(self) -> str | None:
        """Alias for failure domain."""
        return self.domain


def validate_cash_withdrawal(withdrawal: CashWithdrawal, state: Any = None) -> None:
    """
    WHAT: Validates attributes of a CashWithdrawal record.
    WHY: Prevents negative minutes, invalid channels, or non-positive amounts before simulation.

    Args:
        withdrawal: The CashWithdrawal to validate.
        state: Optional RunnerState to verify account registration.

    Raises:
        ValueError: If any attribute fails validation constraints.
    """
    if isinstance(withdrawal.minute, bool) or not isinstance(withdrawal.minute, int) or withdrawal.minute < 0:
        raise ValueError(f"Withdrawal minute must be a non-negative integer, got {withdrawal.minute!r}.")
    if not isinstance(withdrawal.account_id, str) or not withdrawal.account_id:
        raise ValueError("Withdrawal account_id must be a non-empty string.")
    if (
        isinstance(withdrawal.amount_paise, bool)
        or not isinstance(withdrawal.amount_paise, int)
        or withdrawal.amount_paise <= 0
    ):
        raise ValueError(f"Withdrawal amount_paise must be a positive integer, got {withdrawal.amount_paise!r}.")
    if withdrawal.channel not in ALLOWED_CHANNELS:
        raise ValueError(
            f"Withdrawal channel must be one of {ALLOWED_CHANNELS}, got {withdrawal.channel!r}."
        )
    if withdrawal.channel == CHANNEL_ATM:
        if (
            withdrawal.card_daily_limit_paise is None
            or isinstance(withdrawal.card_daily_limit_paise, bool)
            or not isinstance(withdrawal.card_daily_limit_paise, int)
            or withdrawal.card_daily_limit_paise <= 0
        ):
            limit_val = withdrawal.card_daily_limit_paise
            raise ValueError(
                f"channel 'atm' requires card_daily_limit_paise to be a positive integer, got {limit_val!r}."
            )
    elif withdrawal.channel == CHANNEL_BRANCH:
        if withdrawal.card_daily_limit_paise is not None:
            limit_val = withdrawal.card_daily_limit_paise
            raise ValueError(
                f"channel 'branch' requires card_daily_limit_paise to be None, got {limit_val!r}."
            )
    if state is not None and withdrawal.account_id not in state.bank_map:
        raise ValueError(f"Account '{withdrawal.account_id}' is not registered with the runner.")


def apply_illicit_cap(
    context: Any,
    planned_payment_id: str,
    minute: int,
    from_account: str,
    to_account: str,
    planned_paise: int,
    available_paise: int,
) -> int | None:
    """
    WHAT: Caps a planned amount to available illicit balance, recording adjustments in context logs.
    WHY: Shared helper path used by both forward payments and cash withdrawals without duplication.

    Args:
        context: The LoopContext holding skipped_forwards and capped_forwards logs.
        planned_payment_id: Identifier of the transfer or withdrawal being evaluated.
        minute: Simulation minute timestamp.
        from_account: Debited account holding illicit funds.
        to_account: Credited destination account (empty for cash withdrawals).
        planned_paise: Original amount requested in paise.
        available_paise: Current illicit balance held by the account.

    Returns:
        Effective paise to execute (<= available_paise), or None if 0 available (skipped).
    """
    from world.interactive_records import ForwardAdjustment

    effective_paise = min(planned_paise, available_paise)
    if effective_paise == 0:
        context.skipped_forwards.append(
            ForwardAdjustment(
                planned_payment_id=planned_payment_id,
                minute=minute,
                from_account=from_account,
                to_account=to_account,
                planned_paise=planned_paise,
                available_paise=0,
                reason="no_illicit_money_arrived",
            )
        )
        return None
    if effective_paise < planned_paise:
        context.capped_forwards.append(
            ForwardAdjustment(
                planned_payment_id=planned_payment_id,
                minute=minute,
                from_account=from_account,
                to_account=to_account,
                planned_paise=planned_paise,
                available_paise=effective_paise,
                reason="only_part_arrived",
            )
        )
    return effective_paise


def process_cash_withdrawal(context: Any, withdrawal: CashWithdrawal) -> None:
    """
    WHAT: Executes a single cash withdrawal attempt following Decision D1 processing order.
    WHY: Enforces illicit capping, ATM daily limit ceilings, encumbrances, bank outage failures,
         ledger balance verification, and UPI leg splitting for debits.

    Processing order:
        1. Cap to account's illicit balance (record skipped/capped adjustment).
        2. ATM only: Fail with domain 'sender_bank', reason 'atm_daily_limit' if daily ATM sum exceeds card limit.
        3. Encumbrance: Fail with domain 'sender_bank', reason 'funds_on_hold' if lien or debit hold blocks debit.
        4. Failure draw: Evaluate decide_failure on child 13 ('cash_withdrawal_failures').
        5. Balance check: Compare real ledger balance with effective amount; fail with domain 'customer',
           reason 'insufficient_balance' if short, posting no legs.
        6. Ledger post_debit: Split amount into legal legs at bank's upi_limit_paise under single transfer id.
    """
    if getattr(context, "gang_reactions_enabled", False):
        from world.gang_feed import find_holding_gang
        from world.gang_reactions import may_move_money
        from world.interactive_records import GangCancellation

        holding_gang = find_holding_gang(context.gang_holdings, withdrawal.account_id, withdrawal.minute)
        if holding_gang is not None:
            reaction_state = context.gang_reaction_states.get(holding_gang)
            if reaction_state is not None and not may_move_money(reaction_state, withdrawal.account_id):
                context.cancelled_by_gang.append(
                    GangCancellation(
                        minute=withdrawal.minute,
                        gang=holding_gang,
                        accounts=(withdrawal.account_id,),
                        amount=withdrawal.amount_paise,
                        reason="account_stopped",
                    )
                )
                return

    # Generate canonical withdrawal identifier: <run_id>-w-NNNNNN
    w_id = f"{context.run_id}-w-{context.withdrawal_sequence:06d}"
    context.withdrawal_sequence += 1

    # Step 1: Cap to illicit balance using shared helper path
    planned_paise = withdrawal.amount_paise
    available_paise = max(0, context.illicit_balances.get(withdrawal.account_id, 0))
    effective_paise = apply_illicit_cap(
        context, w_id, withdrawal.minute, withdrawal.account_id, "", planned_paise, available_paise
    )
    if effective_paise is None:
        rec = CashWithdrawalRecord(
            id=w_id,
            minute=withdrawal.minute,
            account=withdrawal.account_id,
            channel=withdrawal.channel,
            requested=planned_paise,
            withdrawn=0,
            status="skipped",
            reason="no_illicit_money_arrived",
            domain=None,
        )
        context.cash_withdrawals.append(rec)
        return

    # Step 2: ATM only daily limit verification
    day = withdrawal.minute // 1440
    prior_atm = 0
    if withdrawal.channel == CHANNEL_ATM and withdrawal.card_daily_limit_paise is not None:
        prior_atm = context.daily_atm_withdrawn.get((withdrawal.account_id, day), 0)
        if prior_atm + effective_paise > withdrawal.card_daily_limit_paise:
            rec = CashWithdrawalRecord(
                id=w_id,
                minute=withdrawal.minute,
                account=withdrawal.account_id,
                channel=withdrawal.channel,
                requested=planned_paise,
                withdrawn=0,
                status="failed",
                reason="atm_daily_limit",
                domain="sender_bank",
            )
            context.cash_withdrawals.append(rec)
            return

    # Step 3: Encumbrance block (lien or hold)
    bank_idx = context.state.bank_map[withdrawal.account_id]
    ledger = context.world.ledgers[bank_idx]
    balance = ledger.balance_of(withdrawal.account_id)
    if context.book.blocks_debit(withdrawal.account_id, balance, effective_paise) and balance >= effective_paise:
        rec = CashWithdrawalRecord(
            id=w_id,
            minute=withdrawal.minute,
            account=withdrawal.account_id,
            channel=withdrawal.channel,
            requested=planned_paise,
            withdrawn=0,
            status="failed",
            reason="funds_on_hold",
            domain="sender_bank",
        )
        context.cash_withdrawals.append(rec)
        return

    # Step 4: Failure draw via child 13 (cash_withdrawal_failures)
    stream = context.streams["cash_withdrawal_failures"]
    failure = decide_failure(
        withdrawal.minute, bank_idx, bank_idx, context.plan.incidents, context.plan.failure_settings, stream
    )
    if failure is not None:
        rec = CashWithdrawalRecord(
            id=w_id,
            minute=withdrawal.minute,
            account=withdrawal.account_id,
            channel=withdrawal.channel,
            requested=planned_paise,
            withdrawn=0,
            status="failed",
            reason=failure[1],
            domain=failure[0],
        )
        context.cash_withdrawals.append(rec)
        return

    # Step 5: Ledger balance verification before posting any legs
    balance = ledger.balance_of(withdrawal.account_id)
    if balance < effective_paise:
        rec = CashWithdrawalRecord(
            id=w_id,
            minute=withdrawal.minute,
            account=withdrawal.account_id,
            channel=withdrawal.channel,
            requested=planned_paise,
            withdrawn=0,
            status="failed",
            reason="insufficient_balance",
            domain="customer",
        )
        context.cash_withdrawals.append(rec)
        return

    # Step 6: Post debit in legal UPI legs and update monetary tracking
    legs = split_into_legal_payments(effective_paise, ledger.upi_limit_paise)
    for leg in legs:
        ledger.post_debit(withdrawal.account_id, leg, withdrawal.minute, w_id)

    context.money_withdrawn_paise += effective_paise
    context.illicit_balances[withdrawal.account_id] -= effective_paise
    if withdrawal.channel == CHANNEL_ATM:
        context.daily_atm_withdrawn[(withdrawal.account_id, day)] = prior_atm + effective_paise

    rec = CashWithdrawalRecord(
        id=w_id,
        minute=withdrawal.minute,
        account=withdrawal.account_id,
        channel=withdrawal.channel,
        requested=planned_paise,
        withdrawn=effective_paise,
        status="success",
        reason=None,
        domain=None,
    )
    context.cash_withdrawals.append(rec)
