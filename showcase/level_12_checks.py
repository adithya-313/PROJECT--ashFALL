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
    compute_opening_total,
    find_lien_candidate,
    load_level_12_config,
    verify_money_rules,
)
from world.fast_mode import FastRunSettings, event_lines, run_fast_world
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.outside_events import OutsideEvent
from world.run_hashes import hash_device_stamps, hash_ledger_entries, hash_people_devices

FRAUD_ON = RunnerSettings(fraud_enabled=True)


REPO_ROOT: Path = Path(__file__).resolve().parent.parent
FROZEN_ANCHORS_PATH: Path = REPO_ROOT / "config" / "frozen_anchors.yaml"
ISSUES_V2_PATH: Path = REPO_ROOT / "config" / "issues_v2.yaml"
ISSUES_V1_PATH: Path = REPO_ROOT / "config" / "issues.yaml"
GOLDEN_VALUES_PATH: Path = REPO_ROOT / "scripts" / "golden_values.yaml"


async def run_consistency_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the canonical seed-101 world in fast mode and verifies that all 2,915 event lines match the golden
    SHA-256 hash. Also validates that golden fingerprints FP, G1, G2, and G3 match stored baselines.

    WHY: Proves that baseline world generation remains completely unchanged and deterministic, confirming that
    Level 12 interactive runner abstractions introduced zero drift into core simulation mechanics.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting the verified SHA-256 digest and golden fingerprint parity.

    Example:
        outcome = asyncio.run(run_consistency_check(None))
    """
    # 1. Execute the fast mode world using pinned frozen anchor parameters.
    anchors = load_yaml(FROZEN_ANCHORS_PATH)["frozen_anchors"]["fraud_off_events"]
    settings = FastRunSettings(
        seed=anchors["seed"], n_people=anchors["n_people"], n_shops=anchors["n_shops"],
        n_days=anchors["n_days"], range_name=anchors["range_name"],
    )
    lines = list(event_lines(run_fast_world(settings)))
    actual_hash = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    assert actual_hash == anchors["sha256"], f"Event hash mismatch: {actual_hash} != {anchors['sha256']}"

    # 2. Verify golden fingerprints FP, G1, G2, and G3 from golden_values.yaml.
    golden_cfg = load_yaml(GOLDEN_VALUES_PATH)
    ok_fp, _, _ = check_fp(golden_cfg)
    ok_g1, _, _ = check_g1(golden_cfg)
    ok_g2, _, _ = check_g2(golden_cfg)
    ok_g3, _, _ = check_g3(golden_cfg)
    assert ok_fp and ok_g1 and ok_g2 and ok_g3, "Golden fingerprint check failed"

    key_number = f"SHA-256 {actual_hash[:8]}... ({len(lines):,} events, FP/G1/G2/G3 PASS)"
    return CheckOutcome(key_number=key_number, evidence={"sha256": actual_hash, "lines": len(lines)})


async def run_split_world_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the benchmark split world (seed 1 at 120 people, 20 shops, 61 days) under both fraud-off and fraud-on
    paths. Verifies 5 matching outputs, identical planned payment ids, and genuine split payment counts.

    WHY: Confirms requirement F-b: payment splitting over per-payment UPI limits operates identically across the
    fraud switch boundary without shifting planned payment identities or altering child random stream draws.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting split payment counts, total event counts, and 5-way output parity.

    Example:
        outcome = asyncio.run(run_split_world_check(None))
    """
    # 1. Load split world parameters and fixed showcase code version from configuration.
    config = load_level_12_config()
    split_cfg = config["split_world"]
    code_ver = config["showcase_run"]["code_version"]
    settings = FastRunSettings(
        seed=split_cfg["seed"], n_people=split_cfg["n_people"], n_shops=split_cfg["n_shops"],
        n_days=split_cfg["n_days"], range_name=split_cfg["range_name"],
    )

    # 2. Execute world under both fraud-off and fraud-on allow-all execution paths.
    off_out = run_interactive_world(settings, code_version=code_ver)
    on_out = run_interactive_world(settings, FRAUD_ON, code_version=code_ver)

    # 3. Assert all 5 core outputs and planned payment ids match identically.
    assert on_out.result.run_result == off_out.result.run_result
    assert hash_ledger_entries(on_out.world.ledgers) == hash_ledger_entries(off_out.world.ledgers)
    assert hash_people_devices(on_out.world.people) == hash_people_devices(off_out.world.people)
    assert hash_device_stamps(on_out.result.run_result.event_sender_devices) == hash_device_stamps(
        off_out.result.run_result.event_sender_devices)
    assert on_out.result.planned_payment_ids == off_out.result.planned_payment_ids

    # 4. Count split payments where multiple events share a single planned payment id.
    counts = Counter(on_out.result.planned_payment_ids)
    splits = [pid for pid, count in counts.items() if count > 1]
    assert len(splits) >= split_cfg["min_split_payments"]

    key_number = f"{len(splits)} split payments ({len(off_out.result.run_result.events):,} events, 5/5 match)"
    evidence = {"splits": len(splits), "events": len(off_out.result.run_result.events), "parity": "5/5"}
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_lien_blocks_debit_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Places a targeted lien on a person's account after search_start events, leaving one rupee less than needed
    for a chosen debit. Verifies that all 30 subsequent debits are blocked with reason 'funds_on_hold'.

    WHY: Validates outside event handling and encumbrance enforcement: liens must freeze funds and block debits
    without mutating unencumbered accounts or violating double-entry financial conservation rules.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting the total count of 30 blocked debits and financial integrity.

    Example:
        outcome = asyncio.run(run_lien_blocks_debit_check(None))
    """
    # 1. Locate a candidate debit in the baseline run past the configured search_start position.
    config = load_level_12_config()
    code_ver = config["showcase_run"]["code_version"]
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    baseline_out = run_interactive_world(settings, FRAUD_ON, code_version=code_ver)
    pos, debit, balance_before = find_lien_candidate(baseline_out, config["check_3_lien"]["search_start"])

    # 2. Place a lien encumbrance for balance_before - debit.amount_paise + 1.
    lien_amount = balance_before - debit.amount_paise + 1
    lien = OutsideEvent(debit.minute, "lien", debit.from_account, lien_amount, encumbrance_id="enc-000000")
    lien_out = run_interactive_world(settings, FRAUD_ON, FraudInputs(outside_events=(lien,)), code_version=code_ver)

    # 3. Assert exactly 30 debits were blocked as funds_on_hold and prior events remain identical.
    blocked = [e for e in lien_out.result.run_result.events if e.reason == "funds_on_hold"]
    assert len(blocked) == 30, f"Expected 30 blocked debits, got {len(blocked)}"
    assert lien_out.result.run_result.events[:pos] == baseline_out.result.run_result.events[:pos]
    assert lien_out.result.encumbrance_log == [(debit.minute, "place_lien", "enc-000000")]

    # 4. Verify that Level 11 double-entry money conservation rules hold.
    verify_money_rules(lien_out, compute_opening_total(prepare_world(settings, None).world.ledgers))

    key_number = f"30 debits blocked as funds_on_hold (total 30 debits prevented by Rs {lien_amount/100:,.2f} lien)"
    evidence = {"blocked_debits": len(blocked), "account": debit.from_account, "lien_amount_paise": lien_amount}
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_held_payment_isolation_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Intercepts and holds planned payment index 100 via a bank hook. Asserts that the payment produces
    0 events, leaves 0 ledger entries, is recorded in held_payments, and leaves other 2,914 events intact.

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
    code_ver = config["showcase_run"]["code_version"]
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    baseline_out = run_interactive_world(settings, FRAUD_ON, code_version=code_ver)
    target_idx = config["check_4_hold"]["target_index"]
    target_pid = baseline_out.result.planned_payment_ids[target_idx]
    target_event = baseline_out.result.run_result.events[target_idx]

    # 2. Execute interactive world with a bank hook holding only the target planned id.
    hook = HoldOneHook(target_pid)
    held_out = run_interactive_world(settings, FRAUD_ON, FraudInputs(), hook, code_version=code_ver)

    # 3. Assert held payment produces 0 events and 0 ledger entries while appearing in held_payments.
    assert target_pid not in held_out.result.planned_payment_ids
    assert len(held_out.result.held_payments) == 1 and held_out.result.held_payments[0].planned_payment_id == target_pid
    minute, sender, receiver, amount = (
        target_event.minute, target_event.from_account, target_event.to_account, target_event.amount_paise
    )
    transfer_id = f"tx_{minute}_{sender}_{receiver}_{amount}"
    assert all(entry.transfer_id != transfer_id for ldg in held_out.world.ledgers for entry in ldg.entries)

    # 4. Assert remaining 2,914 events are identical to baseline and money rules hold.
    baseline_events = [e for i, e in enumerate(baseline_out.result.run_result.events) if i != target_idx]
    assert held_out.result.run_result.events == baseline_events
    verify_money_rules(held_out, compute_opening_total(prepare_world(settings, None).world.ledgers))

    remaining_count = len(held_out.result.run_result.events)
    key_number = f"0 events and 0 ledger entries for held payment ({remaining_count:,} remaining events intact)"
    evidence = {"held_id": target_pid, "remaining_events": remaining_count}
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

    # 2. Mutate a temporary copy by stripping evidence from built issue K1.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_v2_path = Path(temp_dir) / "issues_v2.yaml"
        temp_data = issue_register_v2.load_v2_register(ISSUES_V2_PATH)
        del temp_data["issues"]["K1"]["evidence"]
        temp_v2_path.write_text(yaml.safe_dump(temp_data), encoding="utf-8")
        reloaded = issue_register_v2.load_v2_register(temp_v2_path)
        errors = issue_register_v2.check_register(reloaded, REPO_ROOT)
        assert errors == ["K1: built needs at least one evidence item"]

    # 3. Assert real file hash is identical before and after temp validation.
    final_hash = hashlib.sha256(ISSUES_V2_PATH.read_bytes()).hexdigest()
    assert initial_hash == final_hash, "Real issues_v2.yaml file was modified"

    # 4. Verify v1 issue register remains at the frozen 28 built, 2 partial, 1 deferred invariant.
    v1_data = issue_register.load_register(ISSUES_V1_PATH)
    v1_errors = issue_register.check_register(v1_data, REPO_ROOT)
    assert not v1_errors, f"v1 issue register check failed: {v1_errors}"

    key_number = "broken register rejected; real v2 hash and v1 28/2/1 invariant intact"
    evidence = {"errors_caught": len(errors), "real_hash_unchanged": True, "v1_status": "28/2/1"}
    return CheckOutcome(key_number=key_number, evidence=evidence)
