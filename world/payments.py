"""
WHAT: Atomic money transfer engine executing payments across bank ledgers.
It provides atomic multi-bank payment routing, validates financial constraints, and logs matching ledger entries.
It serves as the singular authoritative gateway through which all currency movements take place in the simulation.

WHY: Guarantees that payment legs succeed completely or abort with zero side-effects.
Financial integrity requires that money cannot vanish from a payer without appearing in a payee account.
Centralizing all transfers in one atomic engine ensures complete auditability and invariant preservation.

TERM: atomic payment: A financial operation where both debit and credit succeed together or the entire transfer aborts without state change. For example, moving 500 rupees from Alice to Bob where either both accounts update or neither does.
TERM: ledger resolution: Locating which specific bank holds custody of a given account identifier. For example, determining that account 'acc_01' is registered at Bank 2.
TERM: payment validity check: Pre-execution verification confirming positive funds, non-null accounts, regulatory limits, and solvency. For example, checking that a payer has at least 1000 paise before executing a transfer.
"""

from dataclasses import dataclass
from typing import Any

from world.ledger import Ledger


@dataclass
class PaymentResult:
    """
    WHAT: Holds the outcome and diagnostic reason for an attempted payment.
    It captures whether funds moved successfully, a machine-readable reason code, and the transaction ID.
    External runners and analytics pipelines inspect this result to monitor simulation cash flows.

    WHY: Informs calling systems whether money moved and why it failed if rejected.
    Providing granular failure codes allows callers to distinguish between solvency issues and limits.
    Recording the transfer ID gives callers a handle to trace transactions across ledgers.

    Args:
        success: Boolean flag indicating if the transfer completed and posted to both ledgers.
        reason: Machine-readable status string ('ok', 'insufficient_balance', 'over_upi_limit', etc.).
        transfer_id: Optional string identifier assigned to the transfer if executed.

    Returns:
        Initialized PaymentResult instance.

    Example:
        # PaymentResult instance for a successful payment:
        # res = PaymentResult(success=True, reason="ok", transfer_id="tx_1")
        # assert res.success is True and res.reason == "ok"
    """

    success: bool
    reason: str
    transfer_id: str | None = None


def find_ledger_for_account(ledgers: list[Ledger], account_id: str) -> Ledger | None:
    """
    WHAT: Locates the bank ledger responsible for a specific account identifier.
    It iterates across all bank ledgers in the world to find which ledger contains the target account.
    If no matching ledger is found, it returns None to signal an unknown account.

    WHY: Resolves which bank manages the source or destination of a transaction.
    Decentralized banking models require dynamic account-to-bank resolution for cross-bank transfers.
    Returning None on missing accounts allows the caller to gracefully reject invalid counterparties.

    Args:
        ledgers: List of Ledger instances representing all banking institutions in the world.
        account_id: String identifier of the account being queried.

    Returns:
        The matching Ledger instance that owns the account, or None if not found.

    Example:
        # Locate the ledger owning 'acc_1':
        # l = Ledger(0, 10000000)
        # l.add_account("acc_1")
        # found = find_ledger_for_account([l], "acc_1")
        # assert found is not None and found.bank_index == 0
    """
    # Iterate through ledgers to find the institution managing this account.
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
    It performs pre-execution audits to ensure all prerequisite banking invariants are satisfied.
    If any check fails, it returns a descriptive error string; otherwise it returns None.

    WHY: Centralizes all pre-execution checks so failure causes an immediate clean abort.
    Validating before debiting prevents partial transfers where money leaves a payer but cannot credit.
    A single validation function guarantees consistent enforcement of banking rules across all transactions.

    Args:
        amount_paise: Proposed transfer amount in integer paise.
        sender_ledger: Ledger managing the payer's account, or None if unknown.
        receiver_ledger: Ledger managing the recipient's account, or None if unknown.
        from_account: Account identifier of the proposed payer.

    Returns:
        String error code ('invalid_amount', 'unknown_account', 'over_upi_limit', 'insufficient_balance') or None.

    Example:
        # Validate a legal transfer of 1000 paise from an account holding 5000 paise:
        # l = Ledger(0, 10000000)
        # l.add_account("acc_1", 5000)
        # err = check_payment_validity(1000, l, l, "acc_1")
        # assert err is None
    """
    # Validate that the proposed transfer amount is strictly positive.
    if amount_paise <= 0:
        return "invalid_amount"
    # Verify that both sender and receiver accounts belong to recognized bank ledgers.
    if sender_ledger is None or receiver_ledger is None:
        return "unknown_account"
    # Enforce that transfer amount does not exceed the sender bank's regulatory UPI limit.
    if amount_paise > sender_ledger.upi_limit_paise:
        return "over_upi_limit"
    # Check that the paying account has sufficient balance to cover the transfer.
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
    It resolves ledgers, validates solvency and limits, and executes matching debit and credit legs.
    If any validation rule fails, it immediately returns a failed PaymentResult with zero state changes.

    WHY: The ONLY function authorized to move money; guarantees no money is lost or created.
    Atomic execution guarantees double-entry conservation across arbitrary multi-bank transfer topologies.
    Restricting monetary movements to a single choke-point guarantees auditability across the simulation.

    Args:
        world: World container holding ledgers, banks, and simulation state.
        from_account: Source account identifier from which funds will be deducted.
        to_account: Destination account identifier into which funds will be credited.
        amount_paise: Positive integer currency amount in paise to be transferred.
        minute: Simulation minute timestamp when the payment occurs.

    Returns:
        PaymentResult dataclass instance indicating success or specific failure reason.

    Example:
        # Execute payment of 1000 paise between two citizens:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # p1 = w.people[0]; p2 = w.people[1]
        # res = make_payment(w, p1.account_id, p2.account_id, 1000, 30)
        # assert res.success is True and res.reason == "ok"
    """
    # Resolve the custodial bank ledgers for both sender and receiver accounts.
    sender_l = find_ledger_for_account(world.ledgers, from_account)
    receiver_l = find_ledger_for_account(world.ledgers, to_account)

    # Validate solvency, accounts existence, and regulatory UPI compliance.
    failure_reason = check_payment_validity(amount_paise, sender_l, receiver_l, from_account)
    if failure_reason is not None:
        return PaymentResult(success=False, reason=failure_reason)

    # All checks passed: execute both legs atomically.
    # Generate a unique correlation transaction identifier for both ledger legs.
    tx_id = f"tx_{minute}_{from_account}_{to_account}_{amount_paise}"
    assert sender_l is not None and receiver_l is not None
    # Execute debit on sender ledger and credit on receiver ledger atomically.
    sender_l.post_debit(from_account, amount_paise, minute, tx_id)
    receiver_l.post_credit(to_account, amount_paise, minute, tx_id)
    return PaymentResult(success=True, reason="ok", transfer_id=tx_id)

