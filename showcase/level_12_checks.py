"""
WHAT: Implementation of Level 12 showcase verification checks (Part A).
Orchestrates automated evaluation of engine consistency, split-world equivalence, lien debit blocking,
held payment isolation, and issue register schema validation.

WHY: Verifies the Level 12 reactive world engine against frozen anchors and policy requirements
while demonstrating deterministic parity between fraud-off and fraud-on allow-all simulation modes.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: equivalence: Fraud-off and fraud-on allow-all execution produces bit-for-bit identical outputs.
TERM: lien: A legal encumbrance that freezes account funds and blocks debits exceeding free balance.
TERM: hold: A temporary suspension of payment execution pending explicit administrative resumption.
TERM: issue register: Architectural ledger tracking requirements, evidence, and operational statuses.
"""

import hashlib
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from core import issue_register, issue_register_v2
from core.config_validator import load_yaml
from scripts.verify_world import check_fp, check_g1, check_g2, check_g3
from showcase.framework import CheckOutcome
from showcase.level_12_support import (
    HoldOneHook,
    compare_runner_against_level_11,
    compute_opening_total,
    find_lien_candidate,
    load_level_12_config,
    verify_held_payment_isolation,
    verify_lien_blocked_debits,
)
from world.fast_mode import FastRunSettings, event_lines, run_fast_world
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import (
    interactive_fast_output,
    prepare_world,
    run_interactive_world,
)
from world.outside_events import OutsideEvent

FRAUD_ON = RunnerSettings(fraud_enabled=True)


REPO_ROOT: Path = Path(__file__).resolve().parent.parent
FROZEN_ANCHORS_PATH: Path = REPO_ROOT / "config" / "frozen_anchors.yaml"
ISSUES_V2_PATH: Path = REPO_ROOT / "config" / "issues_v2.yaml"
ISSUES_V1_PATH: Path = REPO_ROOT / "config" / "issues.yaml"
GOLDEN_VALUES_PATH: Path = REPO_ROOT / "scripts" / "golden_values.yaml"


async def run_consistency_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the canonical seed-101 world through both the fraud-off interactive runner and fast mode.
    Verifies that all 2,915 event lines match the golden SHA-256 digest on both paths, and verifies
    that golden fingerprints FP, G1, G2, and G3 match stored baselines.

    WHY: Proves that baseline world generation remains completely unchanged and deterministic, confirming that
    Level 12 interactive runner abstractions introduced zero drift into core simulation mechanics.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting the verified SHA-256 digest and golden fingerprint parity.

    Example:
        outcome = asyncio.run(run_consistency_check(None))
    """
    # 1. Execute the fraud-off world using pinned frozen anchor parameters through both runner and fast paths.
    anchors = load_yaml(FROZEN_ANCHORS_PATH)["frozen_anchors"]["fraud_off_events"]
    settings = FastRunSettings(
        seed=anchors["seed"], n_people=anchors["n_people"], n_shops=anchors["n_shops"],
        n_days=anchors["n_days"], range_name=anchors["range_name"],
    )
    code_version = load_level_12_config()["showcase_run"]["code_version"]
    runner_output = run_interactive_world(settings, code_version=code_version)
    runner_lines = list(event_lines(interactive_fast_output(runner_output)))
    runner_hash = hashlib.sha256("".join(runner_lines).encode("utf-8")).hexdigest()

    fast_lines = list(event_lines(run_fast_world(settings)))
    fast_hash = hashlib.sha256("".join(fast_lines).encode("utf-8")).hexdigest()

    assert runner_hash == anchors["sha256"], f"Runner hash mismatch: {runner_hash} != {anchors['sha256']}"
    assert fast_hash == anchors["sha256"], f"Fast-world hash mismatch: {fast_hash} != {anchors['sha256']}"
    assert runner_lines == fast_lines

    # 2. Verify golden fingerprints FP, G1, G2, and G3 from golden_values.yaml.
    golden_cfg = load_yaml(GOLDEN_VALUES_PATH)
    ok_fp, _, _ = check_fp(golden_cfg)
    ok_g1, _, _ = check_g1(golden_cfg)
    ok_g2, _, _ = check_g2(golden_cfg)
    ok_g3, _, _ = check_g3(golden_cfg)
    assert ok_fp and ok_g1 and ok_g2 and ok_g3, "Golden fingerprint check failed"

    key_number = f"SHA-256 {runner_hash[:8]}... ({len(runner_lines):,} events, FP/G1/G2/G3 PASS)"
    return CheckOutcome(key_number=key_number, evidence={"sha256": runner_hash, "lines": len(runner_lines)})


async def run_split_world_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the benchmark split world (seed 1 at 120 people, 20 shops, 61 days) under fraud-on allow-all
    and Level 11 fast mode. Compares all five outputs against Level 11, validates planned payment id equality
    across the switch, and counts split payments.

    WHY: Confirms requirement F-b: payment splitting over per-payment UPI limits operates identically across the
    fraud switch boundary without shifting planned payment identities or altering child random stream draws.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting split payment counts, total event counts, and measured output parity ratio.

    Example:
        outcome = asyncio.run(run_split_world_check(None))
    """
    # 1. Load split world parameters and fixed showcase code version from configuration.
    config = load_level_12_config()
    split_world = config["split_world"]
    code_version = config["showcase_run"]["code_version"]
    settings = FastRunSettings(
        seed=split_world["seed"], n_people=split_world["n_people"], n_shops=split_world["n_shops"],
        n_days=split_world["n_days"], range_name=split_world["range_name"],
    )

    # 2. Execute Level 11 fast world alongside runner fraud-off and fraud-on allow-all paths.
    level_11 = run_fast_world(settings)
    off_output = run_interactive_world(settings, code_version=code_version)
    on_output = run_interactive_world(settings, FRAUD_ON, code_version=code_version)

    # 3. Compare the fraud-on allow-all run against Level 11 across all five outputs.
    matches, total_comparisons = compare_runner_against_level_11(on_output, level_11)
    assert matches == total_comparisons
    assert on_output.result.planned_payment_ids == off_output.result.planned_payment_ids

    # 4. Count split payments where multiple events share a single planned payment id.
    counts = Counter(on_output.result.planned_payment_ids)
    split_payments = [planned_id for planned_id, piece_count in counts.items() if piece_count > 1]
    assert len(split_payments) >= split_world["min_split_payments"]

    event_count = len(on_output.result.run_result.events)
    parity_str = f"{matches}/{total_comparisons}"
    key_number = f"{len(split_payments)} split payments ({event_count:,} events, {parity_str} match)"
    evidence = {"splits": len(split_payments), "events": event_count, "parity": parity_str}
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_lien_blocks_debit_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Places a targeted lien on a person's account after search_start events, leaving one paise less than needed
    for a chosen debit. Verifies that subsequent debits on that account are blocked with reason 'funds_on_hold'.

    WHY: Validates outside event handling and encumbrance enforcement: liens must freeze funds and block debits
    without mutating unencumbered accounts or violating double-entry financial conservation rules.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting the count of blocked debits, target account, and financial integrity.

    Example:
        outcome = asyncio.run(run_lien_blocks_debit_check(None))
    """
    # 1. Locate a candidate debit in the baseline run past the configured search_start position.
    config = load_level_12_config()
    code_version = config["showcase_run"]["code_version"]
    anchors = load_yaml(FROZEN_ANCHORS_PATH)["frozen_anchors"]["fraud_off_events"]
    settings = FastRunSettings(
        seed=anchors["seed"], n_people=anchors["n_people"], n_shops=anchors["n_shops"],
        n_days=anchors["n_days"], range_name=anchors["range_name"],
    )
    baseline_output = run_interactive_world(settings, FRAUD_ON, code_version=code_version)
    position, debit, balance_before = find_lien_candidate(baseline_output, config["check_3_lien"]["search_start"])

    # 2. Place a lien encumbrance for balance_before - debit.amount_paise + 1.
    lien_amount = balance_before - debit.amount_paise + 1
    lien = OutsideEvent(debit.minute, "lien", debit.from_account, lien_amount, encumbrance_id="enc-000000")
    lien_output = run_interactive_world(
        settings, FRAUD_ON, FraudInputs(outside_events=(lien,)), code_version=code_version
    )

    # 3. Assert all lien blocking criteria hold and return measured blocked count via helper.
    opening_total = compute_opening_total(prepare_world(settings, None).world.ledgers)
    blocked_count = verify_lien_blocked_debits(baseline_output, lien_output, position, debit, lien, opening_total)

    rupees_str = f"Rs {lien_amount / 100:,.2f}"
    key_number = (
        f"{blocked_count} debits from {debit.from_account} blocked after a {rupees_str} lien "
        "(lien never released); ledger balances"
    )
    evidence = {"blocked_debits": blocked_count, "account": debit.from_account, "lien_amount_paise": lien_amount}
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_held_payment_isolation_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Intercepts and holds a planned payment via a bank hook. Asserts that the payment produces 0 events,
    leaves 0 ledger entries, appears in held_payments, and preserves all surrounding events and balances intact.

    WHY: Proves that held transactions freeze execution cleanly without advancing streams or moving funds,
    allowing pending transfers to await subsequent administrative resumption in later levels.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting 0 events for the held payment and total preservation of remaining events.

    Example:
        outcome = asyncio.run(run_held_payment_isolation_check(None))
    """
    # 1. Identify target planned payment at the configured index in the baseline run.
    config = load_level_12_config()
    code_version = config["showcase_run"]["code_version"]
    anchors = load_yaml(FROZEN_ANCHORS_PATH)["frozen_anchors"]["fraud_off_events"]
    settings = FastRunSettings(
        seed=anchors["seed"], n_people=anchors["n_people"], n_shops=anchors["n_shops"],
        n_days=anchors["n_days"], range_name=anchors["range_name"],
    )
    baseline_output = run_interactive_world(settings, FRAUD_ON, code_version=code_version)
    held_index = config["check_4_hold"]["target_index"]
    held_planned_id = baseline_output.result.planned_payment_ids[held_index]
    target_event = baseline_output.result.run_result.events[held_index]

    # 2. Execute interactive world with a bank hook holding only the target planned id.
    held_output = run_interactive_world(
        settings, FRAUD_ON, FraudInputs(), HoldOneHook(held_planned_id), code_version=code_version
    )

    # 3. Assert held payment produces 0 events and 0 ledger entries and verify money rules.
    opening_total = compute_opening_total(prepare_world(settings, None).world.ledgers)
    baseline_entry_count, held_entry_count, remaining_count = verify_held_payment_isolation(
        baseline_output, held_output, held_index, held_planned_id, target_event, opening_total
    )

    key_number = (
        f"0 events and {held_entry_count} ledger entries for held payment "
        f"({baseline_entry_count} in baseline; {remaining_count:,} remaining events intact)"
    )
    evidence = {
        "held_id": held_planned_id,
        "baseline_entries": baseline_entry_count,
        "held_entries": held_entry_count,
        "remaining_events": remaining_count,
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_v2_register_negative_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Validates that core.issue_register_v2 strictly rejects an invalid v2 issue register in an isolated directory.
    Asserts that the real issues_v2.yaml file remains untouched and that the v1 register invariant (28/2/1) holds.

    WHY: Guarantees that issue register validation rules are enforced unconditionally on corrupted files while
    ensuring that tests never mutate repository configuration files during automated showcase evaluations.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting rejection of invalid register and real register hash invariance.

    Example:
        outcome = asyncio.run(run_v2_register_negative_check(None))
    """
    # 1. Record the original hash of config/issues_v2.yaml before negative testing.
    real_content = ISSUES_V2_PATH.read_bytes()
    initial_hash = hashlib.sha256(real_content).hexdigest()

    # 2. Mutate a temporary copy by stripping evidence from built issue K1 and verify rejection.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_v2_path = Path(temp_dir) / "issues_v2.yaml"
        temp_data = issue_register_v2.load_v2_register(ISSUES_V2_PATH)
        del temp_data["issues"]["K1"]["evidence"]
        temp_v2_path.write_text(yaml.safe_dump(temp_data), encoding="utf-8")
        reloaded = issue_register_v2.load_v2_register(temp_v2_path)
        errors = issue_register_v2.check_register(reloaded, REPO_ROOT)
        assert errors == ["K1: built needs at least one evidence item"]
        main_exit_code = issue_register_v2.main([str(temp_v2_path)])
        assert main_exit_code == 1

    # 3. Assert real file hash is identical before and after temp validation.
    final_hash = hashlib.sha256(ISSUES_V2_PATH.read_bytes()).hexdigest()
    assert initial_hash == final_hash, "Real issues_v2.yaml file was modified"

    # 4. Verify v1 issue register remains at the frozen 28 built, 2 planned, 1 deferred invariant.
    v1_data = issue_register.load_register(ISSUES_V1_PATH)
    v1_errors = issue_register.check_register(v1_data, REPO_ROOT)
    assert not v1_errors, f"v1 issue register check failed: {v1_errors}"
    v1_counts = Counter(
        entry.get("status") for entry in v1_data.values() if isinstance(entry, dict)
    )
    assert dict(v1_counts) == {"built": 28, "planned": 2, "deferred": 1}
    v1_ratio = f"{v1_counts['built']}/{v1_counts['planned']}/{v1_counts['deferred']}"

    key_number = f"broken register rejected; real v2 hash and v1 {v1_ratio} invariant intact"
    evidence = {"errors_caught": len(errors), "real_hash_unchanged": True, "v1_status": v1_ratio}
    return CheckOutcome(key_number=key_number, evidence=evidence)

