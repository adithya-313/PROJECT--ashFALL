"""
WHAT: Atomic money transfer engine executing payments across bank ledgers.
WHY: Guarantees that payment legs succeed completely or abort with zero side-effects.
TERM: transfer: An atomic payment from a source account to a destination account consisting of a debit and credit.
"""

from dataclasses import dataclass
from typing import Any
from world.ledger import Ledger


@dataclass
class PaymentResult:
    """
    WHAT: Holds the outcome and diagnostic reason for an attempted payment.
    WHY: Informs calling systems whether money moved and why it failed if rejected.
    """

    success: bool
    reason: str
    transfer_id: str | None = None


def find_ledger_for_account(ledgers: list[Ledger], account_id: str) -> Ledger | None:
    """
    WHAT: Locates the bank ledger responsible for a specific account identifier.
    WHY: Resolves which bank manages the source or destination of a transaction.
    """
    for ledger in ledgers:
        if account_id in ledger.accounts:
            return ledger
    return None


def check_payment_validity(
    amount_paise: int,
    sender_ledger: Ledger | None,
    receiver_ledger: Ledger | None,
    from_account: str,
) -> str | None:
    """
    WHAT: Validates amount, account existence, UPI limit compliance, and sender solvency.
    WHY: Centralizes all pre-execution checks so failure causes an immediate clean abort.
    """
    if amount_paise <= 0:
        return "invalid_amount"
    if sender_ledger is None or receiver_ledger is None:
        return "unknown_account"
    if amount_paise > sender_ledger.upi_limit_paise:
        return "over_upi_limit"
    if sender_ledger.balance_of(from_account) < amount_paise:
        return "insufficient_balance"
    return None


def make_payment(
    world: Any,
    from_account: str,
    to_account: str,
    amount_paise: int,
    minute: int,
) -> PaymentResult:
    """
    WHAT: Atomically transfers funds from one account to another, potentially across banks.
    WHY: The ONLY function authorized to move money; guarantees no money is lost or created.
    """
    sender_l = find_ledger_for_account(world.ledgers, from_account)
    receiver_l = find_ledger_for_account(world.ledgers, to_account)

    failure_reason = check_payment_validity(amount_paise, sender_l, receiver_l, from_account)
    if failure_reason is not None:
        return PaymentResult(success=False, reason=failure_reason)

    # All checks passed: execute both legs atomically.
    tx_id = f"tx_{minute}_{from_account}_{to_account}_{amount_paise}"
    assert sender_l is not None and receiver_l is not None
    sender_l.post_debit(from_account, amount_paise, minute, tx_id)
    receiver_l.post_credit(to_account, amount_paise, minute, tx_id)
    return PaymentResult(success=True, reason="ok", transfer_id=tx_id)
