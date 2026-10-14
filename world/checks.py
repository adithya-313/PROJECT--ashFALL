"""
WHAT: Pure audit checker functions verifying financial ledger integrity.
It inspects all bank ledgers across the world to assert non-negative balances, entry consistency, and money conservation.
These verification routines act as automated auditors confirming that the financial engine operates without flaws.

WHY: Programmatically verifies that money is conserved, balances never go negative, and entries match state.
Closed-loop macroeconomic simulations require mathematical proof that currency is neither lost nor minted.
Automated audit functions provide fast invariant checking before, during, and after simulation runs.

TERM: conservation of money: The fundamental accounting invariant that the total currency across all accounts remains constant unless explicitly deposited or withdrawn. For example, verifying that total paise across all banks matches opening balances.
TERM: negative balance: An illegal financial state where an account holds less than zero currency units. For example, an account showing -500 paise due to an unauthorized overdraft.
TERM: audit check: A pure programmatic verification verifying that simulated financial ledgers adhere strictly to banking rules. For example, asserting that stored account balances match their summed ledger entry history.
"""

from world.ledger import Ledger


def check_no_negative_balances(ledgers: list[Ledger]) -> list[str]:
    """
    WHAT: Checks that every account across all bank ledgers maintains a non-negative balance.
    It iterates across each bank ledger and inspects all stored account balances.
    If any account balance is less than zero, it appends a diagnostic violation message to the returned list.

    WHY: In real banking, deposit accounts cannot silently drop into unapproved negative balances.
    Enforcing non-negative funds validates that debit authorization routines prevent illegal overdrafts.
    Returning a list of problem strings provides actionable debugging details without crashing the audit sweep.

    Args:
        ledgers: List of bank Ledger instances to inspect for negative balances.

    Returns:
        List of diagnostic error strings describing any accounts with balances below zero.

    Example:
        # Check ledgers from a newly built world:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # problems = check_no_negative_balances(w.ledgers)
        # assert problems == []
    """
    problems: list[str] = []
    # Iterate through every bank ledger and inspect each registered account balance.
    for ledger in ledgers:
        for acc_id, balance in ledger.balances.items():
            # Flag any account whose stored balance has fallen below zero paise.
            if balance < 0:
                problems.append(
                    f"Bank {ledger.bank_index} account '{acc_id}' has negative balance: {balance} paise"
                )
    return problems


def check_balances_equal_entry_sums(ledgers: list[Ledger]) -> list[str]:
    """
    WHAT: Recomputes each account balance from historical entries and verifies equality with stored balance.
    It calculates the independent sum of all debits and credits recorded in the ledger entry history.
    It flags any discrepancy where the fast cached balance deviates from the true historical sum.

    WHY: Catches any state corruption or unrecorded ledger updates.
    Direct balance manipulation without corresponding ledger entries violates double-entry audit standards.
    Comparing cached state against raw event history proves the completeness and consistency of ledger logs.

    Args:
        ledgers: List of bank Ledger instances whose accounts will be audited.

    Returns:
        List of diagnostic error strings identifying any accounts where stored balance differs from entry sum.

    Example:
        # Audit account balance consistency against entry histories:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # problems = check_balances_equal_entry_sums(w.ledgers)
        # assert problems == []
    """
    problems: list[str] = []
    # Inspect each bank ledger and verify all constituent accounts.
    for ledger in ledgers:
        for acc_id in ledger.accounts:
            # Compare stored running balance against independent entry history sum.
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
    It computes the total currency in integer paise currently circulating across every bank account.
    If the actual total does not match expected_total_paise, it returns a conservation error message.

    WHY: Proves mathematically that transactions neither manufacture nor destroy simulated currency.
    Closed-loop simulations must strictly conserve total monetary supply during routine operations.
    Early detection of monetary leakage prevents inaccurate econometric and fraud model training.

    Args:
        ledgers: List of bank Ledger instances representing the entire municipal economy.
        expected_total_paise: Exact expected total currency in integer paise seeded at world creation.

    Returns:
        List of diagnostic error strings if money conservation is violated, or empty list if conserved.

    Example:
        # Verify money conservation against initial total:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # tot = sum(l.balance_of(a) for l in w.ledgers for a in l.accounts)
        # problems = check_money_conserved(w.ledgers, tot)
        # assert problems == []
    """
    problems: list[str] = []
    # Aggregate the current balance across every account in all bank ledgers.
    actual_total = sum(ledger.balance_of(acc) for ledger in ledgers for acc in ledger.accounts)
    # Verify that aggregate currency exactly equals expected opening total.
    if actual_total != expected_total_paise:
        problems.append(
            f"Money conservation violated: actual total {actual_total} != expected total {expected_total_paise} paise"
        )
    return problems

