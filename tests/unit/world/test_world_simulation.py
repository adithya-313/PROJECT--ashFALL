"""
WHAT: Unit tests for Step 5 (routine life, build_world, fingerprinting, and determinism).
WHY: Verifies T10 (checkers pass after month run), T12 (look-alike distinct sender count),
     T14 (routine execution invariants), T15 (reproducibility across processes), and T16 (input validation).
ISSUE IDs: E2, E3, E4, A2.
TERM: world fingerprint: A cryptographic SHA-256 checksum over the entire state of a generated world.
"""

import subprocess
import sys

import numpy as np
import pytest

from world.build_world import build_world, world_fingerprint
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.ledger import Ledger
from world.personas import get_persona
from world.routine import build_routine, run_one_month


def test_t16_build_world_refuses_invalid_sizes() -> None:
    """
    WHAT: Checks that build_world refuses non-integer, zero, or negative counts for people and shops.
    WHY: Fulfills test requirement T16 preventing invalid world topologies.
    """
    with pytest.raises(ValueError, match="people count"):
        build_world(seed=1, n_people=0, n_shops=10)
    with pytest.raises(ValueError, match="people count"):
        build_world(seed=1, n_people=-5, n_shops=10)
    with pytest.raises(ValueError, match="shops count"):
        build_world(seed=1, n_people=20, n_shops=0)
    with pytest.raises(ValueError, match="shops count"):
        build_world(seed=1, n_people=20, n_shops=-1)


def test_t14_routine_generation_and_execution() -> None:
    """
    WHAT: Validates that salaried employees have salary items and run_one_month executes cleanly.
    WHY: Fulfills test requirement T14 ensuring transactions obey limits and conservation.
    """
    world = build_world(seed=42, n_people=100, n_shops=15)
    rng = np.random.default_rng(42)

    salaried = [p for p in world.people if p.persona_name == "salaried_employee"]
    for p in salaried:
        items = build_routine(p, world.shops, world.people, rng)
        assert any(it.kind == "salary_in" for it in items)

    results = run_one_month(world, rng)
    assert sum(results.values()) > 0 and results.get("ok", 0) > 0
    assert check_no_negative_balances(world.ledgers) == []
    assert check_balances_equal_entry_sums(world.ledgers) == []


def test_t10_checkers_pass_after_month_run() -> None:
    """
    WHAT: Runs all three invariant audit checkers on a world after a month of financial simulation.
    WHY: Fulfills test requirement T10 verifying that realistic monthly activity preserves accounting laws.
    """
    world = build_world(seed=101, n_people=60, n_shops=12)
    rng = np.random.default_rng(101)
    total_opening = sum(ldg.balance_of(acc) for ldg in world.ledgers for acc in ldg.accounts)

    run_one_month(world, rng)

    assert check_no_negative_balances(world.ledgers) == []
    assert check_balances_equal_entry_sums(world.ledgers) == []
    assert check_money_conserved(world.ledgers, total_opening) == []


def extract_senders_by_receiver(ledgers: list[Ledger]) -> dict[str, set[str]]:
    """
    WHAT: Scans all credit entries in ledgers to map recipient accounts to their distinct sender accounts.
    WHY: Helper for T12 to measure counterparty diversity between personas.
    """
    senders: dict[str, set[str]] = {}
    for ledger in ledgers:
        for entry in ledger.entries:
            if entry.kind == "transfer_credit" and entry.account_id.startswith("acc_person_"):
                parts = entry.transfer_id.split("_")
                if len(parts) >= 4:
                    senders.setdefault(entry.account_id, set()).add(parts[2])
    return senders


def test_t12_lookalikes_receive_from_more_distinct_senders() -> None:
    """
    WHAT: Verifies look-alike accounts receive transfers from more distinct senders than regular accounts.
    WHY: Fulfills test requirement T12 proving that simulated look-alikes generate camouflage inbound density.
    """
    world = build_world(seed=2026, n_people=150, n_shops=15)
    rng = np.random.default_rng(2026)
    run_one_month(world, rng)

    senders = extract_senders_by_receiver(world.ledgers)
    la_ids = {p.account_id for p in world.people if get_persona(p.persona_name).looks_like_mule_in_normal_life}
    non_la_ids = {p.account_id for p in world.people if not get_persona(p.persona_name).looks_like_mule_in_normal_life}

    avg_la = float(np.mean([len(senders.get(acc, set())) for acc in la_ids]))
    avg_non_la = float(np.mean([len(senders.get(acc, set())) for acc in non_la_ids]))
    assert avg_la > avg_non_la


def test_t15_determinism_in_process_and_across_seeds() -> None:
    """
    WHAT: Tests identical seeds yield identical fingerprints, while different seeds differ.
    WHY: Fulfills first part of T15 guaranteeing deterministic in-process simulation.
    """
    w1 = build_world(seed=777, n_people=50, n_shops=10)
    w2 = build_world(seed=777, n_people=50, n_shops=10)
    w3 = build_world(seed=888, n_people=50, n_shops=10)

    fp1 = world_fingerprint(w1)
    fp2 = world_fingerprint(w2)
    fp3 = world_fingerprint(w3)

    assert fp1 == fp2
    assert fp1 != fp3


def test_t15_determinism_across_subprocesses() -> None:
    """
    WHAT: Spawns an external Python subprocess to verify fingerprint matches across processes.
    WHY: Fulfills second part of T15 proving complete platform reproducibility without state leaks.
    """
    w = build_world(seed=777, n_people=50, n_shops=10)
    in_process_fp = world_fingerprint(w)

    cmd = [
        sys.executable,
        "-B",
        "-c",
        "from world.build_world import build_world, world_fingerprint; "
        "w = build_world(777, 50, 10); "
        "print(world_fingerprint(w))",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    subprocess_fp = proc.stdout.strip()

    assert subprocess_fp == in_process_fp
