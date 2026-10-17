"""
WHAT: Support utilities and domain assertions for Level 12 showcase verification checks.
Provides configuration loaders, financial conservation validators, lien debit target selectors,
and targeted hold hooks for reactive simulation evaluations.

WHY: Isolates simulation inspection helpers from test suites so showcase checks remain fully decoupled
from tests/ while keeping all check runners modular and well within the 30-line code ceiling.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: encumbrance: A lien or debit hold placed on an account that freezes funds or blocks outgoing debits.
TERM: planned payment id: A canonical identifier shared by all split pieces of a single planned transfer.
TERM: money conservation: Invariant requiring total world money to equal opening balances plus created funds.
"""

from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.interactive_records import InteractiveWorldOutput, PaymentView

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SHOWCASE_12_CONFIG_PATH: Path = REPO_ROOT / "config" / "showcase_level_12.yaml"


def load_level_12_config(path: Path = SHOWCASE_12_CONFIG_PATH) -> dict[str, Any]:
    """
    WHAT: Loads config/showcase_level_12.yaml containing policy parameters and targets for Level 12 showcase checks.
    Extracts policy blocks for split world benchmarks, showcase run versions, lien targets, and hold indices.

    WHY: Decouples check execution from hardcoded magic numbers, ensuring all test bounds, search offsets,
    and seed parameters originate from validated configuration files.

    Args:
        path: Filesystem path to the Level 12 showcase YAML configuration file.

    Returns:
        Dictionary mapping configuration block names to their validated parameters.

    Example:
        config = load_level_12_config(); split_seed = config["split_world"]["seed"]
    """
    # Load and parse the Level 12 showcase YAML specification.
    return load_yaml(path)


def compute_opening_total(ledgers: list[Any]) -> int:
    """
    WHAT: Computes the aggregate starting balance in paise across all bank accounts before simulation execution begins.
    Iterates through each ledger and sums the initial balance of every registered account.

    WHY: Establishes the authoritative financial baseline needed to verify strict double-entry money conservation
    throughout subsequent simulation events.

    Args:
        ledgers: Sequence of bank Ledger objects holding citizen and commercial shop accounts.

    Returns:
        Integer representing the total initial funds across all ledgers in paise.

    Example:
        total_paise = compute_opening_total(prepared_world.world.ledgers)
    """
    # Sum opening account balances across every bank ledger in the world.
    return sum(ledger.balance_of(account) for ledger in ledgers for account in ledger.accounts)


def verify_money_rules(output: InteractiveWorldOutput, expected_total: int) -> None:
    """
    WHAT: Validates Level 11's core financial invariants across all bank ledgers following a simulation run.
    Asserts no negative balances, balance equality with ledger entry sums, and exact money conservation.

    WHY: Proves that outside interventions (such as liens and holds) never create, destroy, or leak funds,
    confirming double-entry bookkeeping integrity holds under reactive interventions.

    Args:
        output: InteractiveWorldOutput container holding post-run simulation state and ledgers.
        expected_total: Target total paise expected across all accounts (opening total plus created funds).

    Returns:
        None. Raises AssertionError if any financial invariant check fails.

    Example:
        verify_money_rules(run_output, expected_total_paise)
    """
    # Extract ledgers from the simulation output and evaluate each financial invariant rule.
    ledgers = output.world.ledgers
    negative_errors = check_no_negative_balances(ledgers)
    assert not negative_errors, f"Negative balance invariant broken: {negative_errors}"
    sum_errors = check_balances_equal_entry_sums(ledgers)
    assert not sum_errors, f"Balance entry sum invariant broken: {sum_errors}"
    conservation_errors = check_money_conserved(ledgers, expected_total)
    assert not conservation_errors, f"Money conservation invariant broken: {conservation_errors}"


def find_lien_candidate(output: InteractiveWorldOutput, search_start: int) -> tuple[int, Any, int]:
    """
    WHAT: Scans baseline events starting at search_start to locate a successful person debit eligible for a lien test.
    Computes the sender account's exact balance immediately prior to the candidate debit.

    WHY: Requires an outgoing transfer that succeeds under baseline conditions and possesses no conflicting
    transactions in the same minute, ensuring that a lien is the sole cause of subsequent debit failure.

    Args:
        output: Baseline allow-all InteractiveWorldOutput to inspect for candidate debits.
        search_start: Minimum event index to begin candidate selection, ensuring realistic prior history.

    Returns:
        Tuple of (event index, candidate PaymentEvent, sender balance before debit in paise).

    Example:
        pos, debit_event, balance_before = find_lien_candidate(baseline_output, 1000)
    """
    # Scan events sequentially to find candidate debit and recompute prior balance from entry history.
    ledgers = output.world.ledgers
    events = output.result.run_result.events
    for position, event in enumerate(events):
        earlier_same_minute = any(
            other.from_account == event.from_account and other.minute == event.minute
            for other in events[:position]
        )
        is_suitable = event.status == "success" and event.from_account.startswith("acc_person")
        if position >= search_start and is_suitable and not earlier_same_minute:
            balance_before = sum(
                entry.amount_paise
                for ledger in ledgers
                if event.from_account in ledger.accounts
                for entry in ledger.entries
                if entry.account_id == event.from_account and entry.minute < event.minute
            )
            return position, event, balance_before
    raise AssertionError("No suitable debit found matching search criteria in baseline run")


class HoldOneHook:
    """
    WHAT: A bank decision hook callable that places holds on transfers matching a specific planned payment id.
    Permits all non-matching transfers to execute normally while capturing inspection records of all viewed payments.

    WHY: Simulates precise compliance-driven bank hold interventions targeting a specific payment during
    interactive simulation without altering any surrounding transactions.

    Args:
        target_planned_id: The planned payment id string that should be placed on hold upon bank inquiry.

    Returns:
        A configured HoldOneHook instance.

    Example:
        hook = HoldOneHook("2628a156073034da-p-000100"); decision = hook(payment_view)
    """

    def __init__(self, target_planned_id: str) -> None:
        """
        WHAT: Initializes the hold hook with the target planned payment id and an empty record of seen payments.

        WHY: Captures the specific transaction target to isolate during interactive bank decision evaluations.

        Args:
            target_planned_id: The specific planned payment id string to intercept and hold.

        Returns:
            None.

        Example:
            hook = HoldOneHook("2628a156073034da-p-000100")
        """
        # Store target id to hold and initialize empty audit log of presented payment views.
        self.target_planned_id: str = target_planned_id
        self.seen: list[PaymentView] = []

    def __call__(self, payment: PaymentView) -> str:
        """
        WHAT: Inspects incoming PaymentView metadata and returns 'hold' if matching the target id, otherwise 'allow'.
        Appends the evaluated payment view to the internal audit record.

        WHY: Implements the bank callback interface expected by run_interactive for per-transaction decisions.

        Args:
            payment: PaymentView container providing high-level payment parameters without leaking origin.

        Returns:
            String decision: 'hold' for matching target planned payment id, 'allow' for all others.

        Example:
            decision = hook(payment_view)  # Returns "hold" if payment matches target id
        """
        # Log view and return hold decision for matching target planned id.
        self.seen.append(payment)
        if payment.planned_payment_id == self.target_planned_id:
            return "hold"
        return "allow"
