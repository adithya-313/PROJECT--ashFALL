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

import hashlib
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.fast_mode import event_lines
from world.interactive_records import HookDecision, InteractiveWorldOutput, PaymentView
from world.interactive_runner import interactive_fast_output
from world.outside_events import OutsideEvent
from world.run_hashes import (
    hash_device_stamps,
    hash_ledger_entries,
    hash_people_devices,
)

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
        position, debit_event, balance_before = find_lien_candidate(baseline_output, 1000)
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
        held_planned_id: The planned payment id string that should be placed on hold upon bank inquiry.

    Returns:
        A configured HoldOneHook instance.

    Example:
        hook = HoldOneHook("2628a156073034da-p-000100"); decision = hook(payment_view)
    """

    def __init__(self, held_planned_id: str) -> None:
        """
        WHAT: Initializes the hold hook with the target planned payment id and an empty record of seen payments.

        WHY: Captures the specific transaction target to isolate during interactive bank decision evaluations.

        Args:
            held_planned_id: The specific planned payment id string to intercept and hold.

        Returns:
            None.

        Example:
            hook = HoldOneHook("2628a156073034da-p-000100")
        """
        # Store planned payment id to hold and initialize empty audit log of presented views.
        self.held_planned_id: str = held_planned_id
        self.seen: list[PaymentView] = []

    def __call__(self, payment: PaymentView) -> HookDecision:
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
        if payment.planned_payment_id == self.held_planned_id:
            return "hold"
        return "allow"


def verify_lien_blocked_debits(
    baseline_output: InteractiveWorldOutput,
    lien_output: InteractiveWorldOutput,
    position: int,
    debit: Any,
    lien: OutsideEvent,
    opening_total: int,
) -> int:
    """
    WHAT: Asserts that placing an outside lien blocks the candidate debit and subsequent debits on the target account.
    Verifies event failure metadata, prior event immutability, encumbrance audit log entries,
    and double-entry money conservation.

    WHY: Decouples check assertions from fixed expected counts like 30, proving that encumbrance holds
    strictly enforce funds availability without leaking currency or corrupting untargeted accounts.

    Args:
        baseline_output: Simulation output from allow-all baseline execution without outside encumbrances.
        lien_output: Simulation output from execution with the targeted lien outside event applied.
        position: Event index of the candidate debit expected to fail due to insufficient unencumbered funds.
        debit: The PaymentEvent chosen as the candidate outgoing transfer target.
        lien: The OutsideEvent specifying the placed encumbrance amount, timestamp, and target account.
        opening_total: Expected aggregate starting balance in paise across all world accounts.

    Returns:
        Integer count of all debits blocked with reason 'funds_on_hold' during the simulation.

    Example:
        blocked = verify_lien_blocked_debits(base_out, lien_out, 1000, debit, lien, opening_paise)
    """
    # 1. Verify candidate event failed with sender_bank funds_on_hold failure classification.
    events = lien_output.result.run_result.events
    baseline_events = baseline_output.result.run_result.events
    target_event = events[position]
    assert (target_event.status, target_event.failure_domain, target_event.reason) == (
        "failed", "sender_bank", "funds_on_hold"
    )

    # 2. Assert at least one debit was blocked and all blocked debits belong to the liened account
    # and postdate the lien.
    blocked_events = [event for event in events if event.reason == "funds_on_hold"]
    assert len(blocked_events) >= 1
    assert all(
        event.from_account == debit.from_account and event.minute >= debit.minute
        for event in blocked_events
    )

    # 3. Assert prior events are bit-identical and encumbrance log contains the expected entry.
    assert events[:position] == baseline_events[:position]
    assert lien_output.result.encumbrance_log == [(debit.minute, "place_lien", lien.encumbrance_id)]

    # 4. Verify double-entry money conservation invariants across all world bank ledgers.
    verify_money_rules(lien_output, opening_total)
    return len(blocked_events)


def compare_runner_against_level_11(on_output: InteractiveWorldOutput, level_11: Any) -> tuple[int, int]:
    """
    WHAT: Compares an interactive fraud-on allow-all run against Level 11 fast-mode output across five core outputs.
    Evaluates RunResult, event-lines hash, ledger entries hash, citizen device hash, and device stamps hash.

    WHY: Re-implements Test T2 comparison logic in a modular helper so showcase checks remain well under
    the 30-line code ceiling while verifying full execution fidelity against baseline mechanics.

    Args:
        on_output: InteractiveWorldOutput container holding post-run simulation state and ledgers.
        level_11: FastRunOutput container returned by run_fast_world for the identical world configuration.

    Returns:
        Tuple of (matches_count, total_comparisons_count).

    Example:
        matches, total = compare_runner_against_level_11(on_out, level_11_out); assert matches == total
    """
    on_fast = interactive_fast_output(on_output)
    on_lines = hashlib.sha256("".join(event_lines(on_fast)).encode("utf-8")).hexdigest()
    fast_lines = hashlib.sha256("".join(event_lines(level_11)).encode("utf-8")).hexdigest()
    comparisons = [
        ("run_result", on_output.result.run_result == level_11.result),
        ("event_lines_hash", on_lines == fast_lines),
        ("ledger_hash", hash_ledger_entries(on_output.world.ledgers) == hash_ledger_entries(level_11.world.ledgers)),
        (
            "people_devices_hash",
            hash_people_devices(on_output.world.people) == hash_people_devices(level_11.world.people),
        ),
        (
            "device_stamps_hash",
            hash_device_stamps(on_output.result.run_result.event_sender_devices)
            == hash_device_stamps(level_11.result.event_sender_devices),
        ),
    ]
    matches = sum(1 for _, matched in comparisons if matched)
    return matches, len(comparisons)


def verify_held_payment_isolation(
    baseline_output: InteractiveWorldOutput,
    held_output: InteractiveWorldOutput,
    held_index: int,
    held_planned_id: str,
    target_event: Any,
    opening_total: int,
) -> tuple[int, int, int]:
    """
    WHAT: Asserts that holding a planned transaction produces 0 events and 0 ledger entries while preserving
    all other simulation events and maintaining total double-entry ledger currency conservation across all banks.

    WHY: Modular helper ensuring check functions adhere strictly to the 30-line code ceiling while verifying
    that transactional holds isolate the target payment without side effects or leaking funds.

    Args:
        baseline_output: InteractiveWorldOutput container holding baseline simulation run without any holds.
        held_output: InteractiveWorldOutput container holding simulation run where the target payment was held.
        held_index: Integer index of the held payment within the baseline sequence of planned payments.
        held_planned_id: String unique identifier of the planned payment intercepted by the hold hook.
        target_event: PaymentEvent container of the intercepted transaction from the baseline run.
        opening_total: Expected aggregate starting balance in paise across all world accounts.

    Returns:
        Tuple of (baseline_entries_count, held_entries_count, remaining_events_count).

    Example:
        counts = verify_held_payment_isolation(base_out, held_out, held_index, held_planned_id, event, total)
    """
    # 1. Assert held payment is removed from executed planned payments and recorded in held_payments container.
    assert held_planned_id not in held_output.result.planned_payment_ids
    assert len(held_output.result.held_payments) == 1
    assert held_output.result.held_payments[0].planned_payment_id == held_planned_id

    # 2. Assert held payment produces 0 ledger entries compared to exactly 2 entries in baseline run.
    transfer_id = (
        f"tx_{target_event.minute}_{target_event.from_account}_{target_event.to_account}_{target_event.amount_paise}"
    )
    baseline_entries = [
        entry for ledger in baseline_output.world.ledgers
        for entry in ledger.entries if entry.transfer_id == transfer_id
    ]
    held_entries = [
        entry for ledger in held_output.world.ledgers
        for entry in ledger.entries if entry.transfer_id == transfer_id
    ]
    assert len(baseline_entries) == 2
    assert len(held_entries) == 0

    # 3. Assert all non-held events match baseline bit-for-bit and money invariants hold.
    baseline_events = [
        event for index, event in enumerate(baseline_output.result.run_result.events) if index != held_index
    ]
    assert held_output.result.run_result.events == baseline_events
    verify_money_rules(held_output, opening_total)
    return len(baseline_entries), len(held_entries), len(held_output.result.run_result.events)

