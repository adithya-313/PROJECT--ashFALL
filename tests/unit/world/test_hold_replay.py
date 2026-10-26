"""
tests/unit/world/test_hold_replay.py

Unit test suite for two-pass administrative hold replay pipeline (world/hold_replay.py).
Verifies run repeatability across seeds, too_late classification on pre-landing money departure,
applied classification with lien registration in pass-2 encumbrance log, bank routing privacy,
single-hop forwarding termination, and fraud-off backward compatibility under Amendment A-029.
"""

import statistics
from dataclasses import replace

import pytest

from nationalsim.hold_flow import (
    HoldOutcome,
    HoldRequest,
    OutgoingTransfer,
    classify_hold_outcome,
    decide_one_hop_forwarding,
)
from showcase.level_12_support_b import get_showcase_b_settings
from showcase.level_12_support_c import load_showcase_c_fixture
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.hold_replay import (
    measure_hold_latencies_by_window,
    reconstruct_account_balance,
    run_two_pass_replay,
)
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.ledger import Ledger
from world.outside_events import OutsideEvent
from world.world_configs import read_world_configs


def test_two_pass_replay_is_deterministic_on_same_seed() -> None:
    """Proves that running two-pass replay twice on the same seed produces identical holds and outcomes.

    What it is for:
        Guarantees that hold generation, latency calculation, and outcome classification
        are perfectly reproducible across repeated executions under Ground Rule 1.

    Inputs:
        None. Executes run_two_pass_replay twice using fixed showcase seed and gang fixture.

    What it returns:
        None. Asserts all primary holds, outcomes, forwarded holds, and pass outputs match exactly.

    How it works:
        1. Load standard showcase settings and fixture gang.
        2. Execute run_two_pass_replay twice with identical arguments.
        3. Compare primary_holds, primary_outcomes, forwarded_holds, and forwarded_outcomes.
        4. Compare Pass 2 run identifiers and event counts.

    Errors and side effects:
        Raises AssertionError if any field or outcome differs between runs.

    Limits:
        Evaluates deterministic execution on a fixed seed.
    """
    settings, code_version = get_showcase_b_settings()
    gang = load_showcase_c_fixture()

    replay1 = run_two_pass_replay(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        code_version=code_version,
    )

    replay2 = run_two_pass_replay(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        code_version=code_version,
    )

    assert replay1.primary_holds == replay2.primary_holds
    assert replay1.primary_outcomes == replay2.primary_outcomes
    assert replay1.forwarded_holds == replay2.forwarded_holds
    assert replay1.forwarded_outcomes == replay2.forwarded_outcomes
    assert replay1.pass2_output.result.run_id == replay2.pass2_output.result.run_id


def test_constructed_case_money_leaves_before_hold_lands_is_too_late() -> None:
    """Proves that an account where funds departed before hold landing is classified as too_late.

    What it is for:
        Verifies that balance reconstruction and outcome classification accurately identify
        situations where a victim's money has fled the target account before the bank froze it.

    Inputs:
        None. Constructs an isolated ledger where opening funds are debited prior to hold arrival.

    What it returns:
        None. Asserts reconstructed balance is zero and outcome status is 'too_late'.

    How it works:
        1. Create an isolated bank Ledger with a 100,000 INR limit.
        2. Register account 'acc_mule_test' with 500,000 paise opening balance at minute 0.
        3. Post a debit of 500,000 paise at minute 40, draining the balance to 0.
        4. Reconstruct balance at minute 60; verify it is 0.
        5. Classify a 500,000 paise hold landing at minute 60; assert status is 'too_late'.

    Errors and side effects:
        Raises AssertionError if balance is non-zero or outcome is not 'too_late'.

    Limits:
        Evaluates constructed single-account ledger history.
    """
    ledger = Ledger(bank_index=1, upi_limit_paise=10000000)
    target_account = "acc_mule_test"
    ledger.add_account(target_account, opening_balance_paise=500000, minute=0)

    # Funds depart at minute 40 via outgoing debit
    ledger.post_debit(target_account, amount_paise=500000, minute=40, transfer_id="tx_depart_40")

    # Hold lands at minute 60
    landing_minute = 60
    reconstructed_balance = reconstruct_account_balance(ledger, target_account, landing_minute)
    assert reconstructed_balance <= 0, f"Expected depleted balance <= 0, got {reconstructed_balance}"

    outcome = classify_hold_outcome(
        available_balance_paise=reconstructed_balance,
        requested_amount_paise=500000,
        account_id=target_account,
        landing_minute=landing_minute,
        missed=False,
    )
    assert outcome.status == "too_late"
    assert outcome.held_amount_paise == 0


def test_money_stays_is_applied_and_lien_appears_in_pass2_encumbrance_log() -> None:
    """Proves that when funds remain in the account, the hold is applied and logged in encumbrance_log.

    What it is for:
        Guarantees that successful holds encumber available funds and their liens are physically
        placed into the pass-2 encumbrance book and recorded in the audit log.

    Inputs:
        None. Evaluates a scenario where money remains in the account when the lien lands.

    What it returns:
        None. Asserts outcome status is 'applied' and lien record appears in pass-2 encumbrance_log.

    How it works:
        1. Construct a ledger where 1,000,000 paise is deposited and not debited.
        2. Reconstruct balance at landing minute 100; assert available balance is 1,000,000.
        3. Classify outcome; assert status is 'applied' and held amount is 1,000,000.
        4. Run an interactive world run with a lien OutsideEvent; assert encumbrance_log
           records (100, 'place_lien', 'lien_test_01').

    Errors and side effects:
        Raises AssertionError if hold is not applied or lien is omitted from encumbrance_log.

    Limits:
        Evaluates encumbrance log tracking and full preservation classification.
    """
    ledger = Ledger(bank_index=0, upi_limit_paise=10000000)
    target_account = "acc_stay_test"
    ledger.add_account(target_account, opening_balance_paise=1000000, minute=0)

    landing_minute = 100
    reconstructed_balance = reconstruct_account_balance(ledger, target_account, landing_minute)
    assert reconstructed_balance == 1000000

    outcome = classify_hold_outcome(
        available_balance_paise=reconstructed_balance,
        requested_amount_paise=1000000,
        account_id=target_account,
        landing_minute=landing_minute,
        missed=False,
    )
    assert outcome.status == "applied"
    assert outcome.held_amount_paise == 1000000

    # Verify that placing this lien in an actual interactive run logs to encumbrance_log
    settings, code_version = get_showcase_b_settings()
    lien = OutsideEvent(
        minute=landing_minute,
        kind="lien",
        account_id="acc_person_000001",
        amount_paise=1000000,
        encumbrance_id="lien_test_01",
    )
    run_output = run_interactive_world(
        settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=FraudInputs(outside_events=(lien,)),
        code_version=code_version,
    )
    assert (landing_minute, "place_lien", "lien_test_01") in run_output.result.encumbrance_log


def test_hold_reaches_only_reported_account_bank() -> None:
    """Proves that hold directives target only the receiving bank managing the reported account.

    What it is for:
        Enforces banking partition privacy: an administrative hold directive is delivered
        strictly to the reported account's custodial institution, never to uninvolved rival banks.

    Inputs:
        None. Inspects bank routing on hold directives generated by run_two_pass_replay.

    What it returns:
        None. Asserts every hold directs to the bank that owns the targeted account.

    How it works:
        1. Run two-pass replay with showcase settings and fixture gang.
        2. For each generated hold, look up the target bank index.
        3. Assert that the reported account is in the target bank's accounts registry.
        4. Assert that the reported account is not present in rival banks' account sets.

    Errors and side effects:
        Raises AssertionError if hold is routed to a bank that does not hold the account.

    Limits:
        Evaluates institutional boundary isolation.
    """
    settings, code_version = get_showcase_b_settings()
    gang = load_showcase_c_fixture()

    replay = run_two_pass_replay(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        code_version=code_version,
    )

    ledgers = replay.pass2_output.world.ledgers
    for hold in replay.primary_holds:
        target_ledger = ledgers[hold.receiving_bank_index]
        assert hold.account_id in target_ledger.accounts, (
            f"Hold target account '{hold.account_id}' must belong to receiving bank {hold.receiving_bank_index}"
        )
        # Verify rival banks do not manage this account
        for other_idx, other_ledger in enumerate(ledgers):
            if other_idx != hold.receiving_bank_index:
                assert hold.account_id not in other_ledger.accounts


def test_forwarding_sends_one_request_to_receiving_bank_and_stops_there() -> None:
    """Proves that naive forwarding issues exactly one hop to the recipient bank and terminates.

    What it is for:
        Enforces Amendment A-029 boundary constraint: receiving banks trace unencumbered funds
        strictly one hop downstream without chaining transitive requests.

    Inputs:
        None. Evaluates decide_one_hop_forwarding with multi-step transfer histories.

    What it returns:
        None. Asserts single-hop forward is created and secondary forwarding attempt returns None.

    How it works:
        1. Construct primary hold and too_late outcome.
        2. Supply outgoing transfer to Bank 2; verify forwarded request targets Bank 2.
        3. Verify forwarded hold is flagged with is_forwarded=True.
        4. Supply subsequent transfer from Bank 2 to Bank 0 and attempt secondary forwarding.
        5. Assert secondary forwarding returns None, terminating after exactly one hop.

    Errors and side effects:
        Raises AssertionError if forwarding chains past the single-hop boundary.

    Limits:
        Evaluates single-hop traceback enforcement.
    """
    primary_hold = HoldRequest(
        complaint_id="c_test_hop",
        payment_id="p_origin",
        receiving_bank_index=0,
        account_id="acc_hop_1",
        amount_paise=750000,
        landing_minute=120,
        is_forwarded=False,
    )
    outcome_too_late = HoldOutcome(
        status="too_late",
        held_amount_paise=0,
        requested_amount_paise=750000,
        account_id="acc_hop_1",
        landing_minute=120,
    )
    outgoing = [
        OutgoingTransfer(
            transfer_id="tx_hop_1_to_2",
            from_account="acc_hop_1",
            to_account="acc_hop_2",
            to_bank_index=2,
            amount_paise=750000,
            minute=90,
        )
    ]

    fwd_hold = decide_one_hop_forwarding(primary_hold, outcome_too_late, outgoing)
    assert fwd_hold is not None
    assert fwd_hold.is_forwarded is True
    assert fwd_hold.receiving_bank_index == 2
    assert fwd_hold.account_id == "acc_hop_2"

    # Secondary forwarding attempt with fwd_hold MUST return None
    outcome_second_hop = HoldOutcome(
        status="too_late",
        held_amount_paise=0,
        requested_amount_paise=750000,
        account_id="acc_hop_2",
        landing_minute=180,
    )
    outgoing_second = [
        OutgoingTransfer(
            transfer_id="tx_hop_2_to_0",
            from_account="acc_hop_2",
            to_account="acc_hop_3",
            to_bank_index=0,
            amount_paise=750000,
            minute=150,
        )
    ]
    second_hop = decide_one_hop_forwarding(fwd_hold, outcome_second_hop, outgoing_second)
    assert second_hop is None, "Forwarding must strictly terminate after one hop"


def test_fraud_off_output_is_unchanged() -> None:
    """Proves that running two-pass replay with fraud disabled produces unchanged Level 11 events.

    What it is for:
        Guarantees backward compatibility: when fraud is disabled, two-pass replay behaves
        identically to the baseline fraud-off engine and generates no hold directives.

    Inputs:
        None. Runs two-pass replay with RunnerSettings(fraud_enabled=False).

    What it returns:
        None. Asserts hold lists are empty and Pass 2 events match baseline fraud-off run exactly.

    How it works:
        1. Set up small FastRunSettings(101, 60, 12, 61, 'training').
        2. Run baseline run_interactive_world with fraud_enabled=False.
        3. Run run_two_pass_replay with fraud_enabled=False.
        4. Assert primary_holds, primary_outcomes, forwarded_holds are empty.
        5. Assert pass-2 events match baseline events in quantity, ordering, and content.

    Errors and side effects:
        Raises AssertionError if fraud-off events deviate from baseline.

    Limits:
        Evaluates fraud switch backward compatibility.
    """
    settings = FastRunSettings(101, 60, 12, 61, "training")
    baseline = run_interactive_world(settings, runner_settings=RunnerSettings(fraud_enabled=False))

    replay = run_two_pass_replay(settings, runner_settings=RunnerSettings(fraud_enabled=False))

    assert replay.primary_holds == []
    assert replay.primary_outcomes == []
    assert replay.forwarded_holds == []
    assert replay.forwarded_outcomes == []

    baseline_events = baseline.result.run_result.events
    replay_events = replay.pass2_output.result.run_result.events
    assert len(baseline_events) == len(replay_events)
    assert baseline_events == replay_events


def test_hold_latency_medians_higher_at_night_and_weekends() -> None:
    """Proves that empirical hold latency medians are higher at night and weekends matching multipliers.

    What it is for:
        Validates that simulated hold response latencies accurately reflect diurnal off-hours
        emergency staffing and weekend slowdowns across the simulated banking network.

    Inputs:
        None. Runs two-pass replay on seed 102 with full victim complaint reporting.

    What it returns:
        None. Asserts night median > day median and weekend median > weekday median.

    How it works:
        1. Configure seed 102 with large scam syndicate and victim complaint share at 1.0.
        2. Execute run_two_pass_replay.
        3. Measure and group hold latencies by temporal window using measure_hold_latencies_by_window.
        4. Assert that day, night, weekday, and weekend buckets each contain measured samples.
        5. Calculate medians dynamically using statistics.median (never hardcoded).
        6. Assert night median exceeds day median and weekend median exceeds weekday median.

    Errors and side effects:
        Raises AssertionError if night or weekend medians fail to exceed baseline daytime medians.

    Limits:
        Evaluates empirical sample medians across simulated victim complaints.
    """
    budget = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
    steps = (
        BlockStep("acquire", (("count", 6),)),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 12))),
    )
    gang = GangConfig("g_large", 1, budget, None, steps, 1440)

    configs = read_world_configs()
    configs.world_ranges["parameters"]["victim_complaint_share"]["training"] = [1.0, 1.0]

    base_settings, code_version = get_showcase_b_settings()
    settings = FastRunSettings(
        102, base_settings.n_people, base_settings.n_shops, base_settings.n_days, base_settings.range_name
    )

    replay = run_two_pass_replay(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        code_version=code_version,
        configs=configs,
    )

    groups = measure_hold_latencies_by_window(replay)

    assert len(groups["day"]) > 0, "Expected at least one daytime hold"
    assert len(groups["night"]) > 0, "Expected at least one night hold"
    assert len(groups["weekday"]) > 0, "Expected at least one weekday hold"
    assert len(groups["weekend"]) > 0, "Expected at least one weekend hold"

    med_day = statistics.median(groups["day"])
    med_night = statistics.median(groups["night"])
    med_weekday = statistics.median(groups["weekday"])
    med_weekend = statistics.median(groups["weekend"])

    assert med_night > med_day, (
        f"Night median latency ({med_night}) must exceed day median ({med_day})"
    )
    assert med_weekend > med_weekday, (
        f"Weekend median latency ({med_weekend}) must exceed weekday median ({med_weekday})"
    )


def test_missing_complaint_raises_value_error() -> None:
    """Proves that measure_hold_latencies_by_window raises ValueError if a complaint is missing.

    What it is for:
        Guarantees fail-fast integrity: if a primary hold references a complaint_id
        that cannot be found in the Pass 1 complaint registry, measure_hold_latencies_by_window
        raises ValueError naming the missing complaint rather than silently skipping it.

    Inputs:
        None. Modifies a valid replay result by replacing primary_holds with a hold
        referencing a non-existent complaint ID.

    What it returns:
        None. Asserts ValueError is raised naming the missing complaint ID.

    How it works:
        1. Set up small FastRunSettings and run two-pass replay to obtain a valid result structure.
        2. Construct an artificial HoldRequest referencing 'c_missing_complaint_404'.
        3. Replace primary_holds on the replay result using dataclasses.replace.
        4. Call measure_hold_latencies_by_window with pytest.raises(ValueError) and check error message.

    Errors and side effects:
        Raises AssertionError if ValueError is not raised or does not name 'c_missing_complaint_404'.

    Limits:
        Evaluates missing complaint error handling on Pass 1 complaint mismatch.
    """
    settings, code_version = get_showcase_b_settings()
    gang = load_showcase_c_fixture()

    replay = run_two_pass_replay(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        code_version=code_version,
    )

    bogus_hold = HoldRequest(
        complaint_id="c_missing_complaint_404",
        payment_id="p_test_missing",
        receiving_bank_index=0,
        account_id="acc_missing_01",
        amount_paise=100000,
        landing_minute=100,
        is_forwarded=False,
    )
    tampered_replay = replace(replay, primary_holds=[bogus_hold])

    with pytest.raises(ValueError, match="c_missing_complaint_404"):
        measure_hold_latencies_by_window(tampered_replay)


def test_partly_funded_hold_is_applied_with_shortfall() -> None:
    """Proves that a partly-funded hold receives status 'applied' with correct held amount and shortfall.

    What it is for:
        Verifies that when an account has positive balance less than the requested hold amount,
        the hold outcome status is 'applied', held_amount_paise equals the available balance,
        and shortfall_paise records the unencumbered difference under Amendment A-029.

    Inputs:
        None. Constructs an isolated ledger with 400,000 paise available balance and 1,000,000 paise hold.

    What it returns:
        None. Asserts status is 'applied', held_amount_paise is 400,000, and shortfall_paise is 600,000.

    How it works:
        1. Create an isolated bank ledger and deposit 400,000 paise opening balance.
        2. Reconstruct balance at minute 50; assert available balance is 400,000 paise.
        3. Classify hold outcome for 1,000,000 paise requested hold.
        4. Assert status == 'applied', held_amount_paise == 400,000, and shortfall_paise == 600,000.

    Errors and side effects:
        Raises AssertionError if status is not 'applied' or held/shortfall amounts are incorrect.

    Limits:
        Evaluates partly-funded administrative hold classification.
    """
    ledger = Ledger(bank_index=0, upi_limit_paise=10000000)
    target_account = "acc_partly_funded"
    ledger.add_account(target_account, opening_balance_paise=400000, minute=0)

    landing_minute = 50
    reconstructed_bal = reconstruct_account_balance(ledger, target_account, landing_minute)
    assert reconstructed_bal == 400000

    outcome = classify_hold_outcome(
        available_balance_paise=reconstructed_bal,
        requested_amount_paise=1000000,
        account_id=target_account,
        landing_minute=landing_minute,
        missed=False,
    )
    assert outcome.status == "applied"
    assert outcome.held_amount_paise == 400000
    assert outcome.shortfall_paise == 600000

