"""
WHAT: Pure audit checker functions verifying financial ledger integrity.
WHY: Programmatically verifies that money is conserved, balances never go negative, and entries match state.
TERM: conservation of money: The accounting invariant that the total money across all accounts remains constant.
"""

from world.ledger import Ledger


def check_no_negative_balances(ledgers: list[Ledger]) -> list[str]:
    """
    WHAT: Checks that every account across all bank ledgers has a balance of at least 0 paise.
    WHY: In real banking, deposit accounts cannot silently drop into unapproved negative balances.
    """
    problems: list[str] = []
    for ledger in ledgers:
        for acc_id, balance in ledger.balances.items():
            if balance < 0:
                problems.append(
                    f"Bank {ledger.bank_index} account '{acc_id}' has negative balance: {balance} paise"
                )
    return problems


def check_balances_equal_entry_sums(ledgers: list[Ledger]) -> list[str]:
    """
    WHAT: Recomputes each account balance from historical entries and verifies equality with stored balance.
    WHY: Catches any state corruption or unrecorded ledger updates.
    """
    problems: list[str] = []
    for ledger in ledgers:
        for acc_id in ledger.accounts:
            stored = ledger.balance_of(acc_id)
            recomputed = ledger.recompute_balance(acc_id)
            if stored != recomputed:
                problems.append(
                    f"Bank {ledger.bank_index} account '{acc_id}' balance mismatch: stored {stored} != sum {recomputed}"
                )
    return problems


def check_money_conserved(ledgers: list[Ledger], expected_total_paise: int) -> list[str]:
    """
    WHAT: Sums all balances across all banks and compares against the expected opening total.
    WHY: Proves mathematically that transactions neither manufacture nor destroy simulated currency.
    """
    problems: list[str] = []
    actual_total = sum(ledger.balance_of(acc) for ledger in ledgers for acc in ledger.accounts)
    if actual_total != expected_total_paise:
        problems.append(
            f"Money conservation violated: actual total {actual_total} != expected total {expected_total_paise} paise"
        )
    return problems
