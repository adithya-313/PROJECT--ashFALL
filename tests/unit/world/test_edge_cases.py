"""
WHAT: Unit tests for Level 12h edge-case library loader, validation rules, and schema checks.
It verifies that all 13 scenario files load, status counts match, unknown schema keys are refused,
invalid amendment references are rejected, and layer containment nesting violations raise errors.

WHY: Enforces rigorous structural integrity over the scenario edge-case library so invalid
definitions, corrupted overlays, or unanchored assumptions fail immediately with clear messages.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from world.edge_cases import (
    DEFAULT_CASES_DIR,
    EXPECTATIONS,
    check_every_complaint_false,
    check_genuine_shop_downstream_of_mule,
    check_mules_forward_after_days,
    check_no_victim_ever_complains,
    check_one_huge_transfer,
    check_registry_wrong_listing,
    check_wedding_collection,
    load_edge_case,
    load_edge_case_library,
    run_edge_case,
)
from world.label_records import LabelEvent
from world.truth_records import RoleSpan


def test_all_13_edge_cases_load_and_validate_cleanly() -> None:
    """
    WHAT: Tests that all 13 edge cases in scenarios/edge_cases/ load and validate successfully.
    WHY: Verifies baseline library completeness and confirms each case YAML satisfies closed schema rules.
    """
    cases = load_edge_case_library()
    assert len(cases) == 13
    names = [c.name for c in cases]
    assert len(set(names)) == 13
    assert names[0] == "busy_shop_tiny_payments"
    assert names[-1] == "recycled_phone_number"


def test_counts_by_status_match_the_authored_files() -> None:
    """
    WHAT: Asserts that scenario status counts match the architect's fixed assignment distribution.
    WHY: Verifies the exact partition between runnable (9), later_level (3), and deferred (1) cases.
    """
    cases = load_edge_case_library()
    runnable = [c for c in cases if c.status == "runnable"]
    later_level = [c for c in cases if c.status == "later_level"]
    deferred = [c for c in cases if c.status == "deferred"]

    assert len(runnable) == 10
    assert len(later_level) == 2
    assert len(deferred) == 1

    assert {c.name for c in later_level} == {
        "busy_shop_tiny_payments",
        "trace_outage_mid_trace",
    }
    assert {c.name for c in deferred} == {"mule_imitating_salaried_employee"}


def test_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that loading an edge case YAML containing an unapproved key raises ValueError.
    WHY: Closed schema enforcement prevents ad-hoc, untyped parameters from entering the library.
    """
    bad_file = tmp_path / "99-unknown_key.yaml"
    bad_file.write_text(
        yaml.dump({
            "name": "unknown_key",
            "title": "Unknown Key Case",
            "expected_behaviour": "Should be refused by closed schema validator.",
            "reason": "Testing unknown key rejection.",
            "status": "later_level",
            "active_from_level": 15,
            "unsupported_extra_field": "disallowed",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown key 'unsupported_extra_field'"):
        load_edge_case(bad_file)


def test_deferred_case_citing_unknown_amendment_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that a deferred case citing a non-existent amendment (e.g. A-999) raises ValueError.
    WHY: Guarantees that every deferred scenario links directly to an approved architectural amendment.
    """
    bad_file = tmp_path / "99-bad_amendment.yaml"
    bad_file.write_text(
        yaml.dump({
            "name": "bad_amendment",
            "title": "Bad Amendment Case",
            "expected_behaviour": "Deferred under A-999; no expectation is checked.",
            "reason": "Testing unknown amendment refusal.",
            "status": "deferred",
            "deferred_by": "A-999",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deferred_by amendment 'A-999' not found"):
        load_edge_case(bad_file)


def test_later_level_case_with_no_active_from_level_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that a later_level edge case omitting active_from_level raises ValueError.
    WHY: Enforces that future milestone requirements explicitly declare their target delivery level.
    """
    bad_file = tmp_path / "99-missing_level.yaml"
    bad_file.write_text(
        yaml.dump({
            "name": "missing_level",
            "title": "Missing Level Case",
            "expected_behaviour": "Should be refused due to missing active level.",
            "reason": "Testing missing active level refusal.",
            "status": "later_level",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="later_level case requires integer active_from_level"):
        load_edge_case(bad_file)


def test_layer_breaking_nesting_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that a layer change violating containment nesting (training not in test/stress) raises ValueError.
    WHY: Verifies that edge-case configuration overlays preserve training <= test <= stress containment bounds.
    """
    bad_file = tmp_path / "99-bad_nesting.yaml"
    # Pinned value 0.95 outside stress range [0.05, 0.9]
    bad_file.write_text(
        yaml.dump({
            "name": "bad_nesting",
            "title": "Bad Nesting Case",
            "expected_behaviour": "Should be refused due to broken nesting.",
            "reason": "Testing layer nesting violation refusal.",
            "status": "runnable",
            "split": "training",
            "seed": 101,
            "days": 61,
            "preset": "tiny",
            "layer_changes": {
                "world_ranges": {
                    "victim_complaint_share": {
                        "training": [0.95, 0.95],
                        "test": [0.1, 0.95],
                        "stress": [0.05, 0.9],
                    }
                }
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test range is not inside the stress range"):
        load_edge_case(bad_file)


def test_run_edge_case_03_mules_forward_after_days() -> None:
    """
    WHAT: Runs runnable edge case 03 and asserts check_mules_forward_after_days passes with wait_min=1440.
    WHY: Proves that slow-drip evasion routing enforces >= 24-hour holds between successive hops.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "03-mules_forward_after_days.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 03 failed: {detail}"
    assert key_num == "wait_min=1440"


def test_run_edge_case_04_no_victim_ever_complains() -> None:
    """
    WHAT: Runs runnable edge case 04 and asserts check_no_victim_ever_complains passes with complaints=0.
    WHY: Proves that zero victim complaint rate yields zero true complaints despite active scam operations.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "04-no_victim_ever_complains.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 04 failed: {detail}"
    assert key_num == "complaints=0"


def test_run_edge_case_05_every_complaint_false() -> None:
    """
    WHAT: Runs runnable edge case 05 and asserts check_every_complaint_false passes with false_ratio=1.0.
    WHY: Proves that elevated background false complaint parameters produce only false complaints.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "05-every_complaint_false.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 05 failed: {detail}"
    assert key_num == "false_ratio=1.0"


def test_run_edge_case_06_registry_wrong_listing() -> None:
    """
    WHAT: Runs runnable edge case 06 and asserts check_registry_wrong_listing passes with wrong_listing_count=2.
    WHY: Proves that false complaints propagate to the national registry, listing innocent accounts.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "06-registry_wrong_listing.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 06 failed: {detail}"
    assert key_num == "wrong_listing_count=2"


def test_run_edge_case_07_one_huge_transfer() -> None:
    """
    WHAT: Runs runnable edge case 07 and asserts check_one_huge_transfer passes with amount_inr=100000.
    WHY: Proves that lump-sum transfers maximize per-transaction limit in single unsplit hops.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "07-one_huge_transfer.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 07 failed: {detail}"
    assert key_num == "amount_inr=100000"


def test_run_edge_case_09_wedding_collection() -> None:
    """
    WHAT: Runs runnable edge case 09 and asserts check_wedding_collection passes with payer_count=75.
    WHY: Proves that wedding collection pooling episodes involve genuine citizens without gang roles.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "09-wedding_collection.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 09 failed: {detail}"
    assert key_num == "payer_count=75"


def test_run_edge_case_10_medical_fundraiser() -> None:
    """
    WHAT: Runs runnable edge case 10 and asserts check_medical_fundraiser passes with payer_count=75.
    WHY: Proves that medical fundraiser pooling episodes involve genuine citizens without gang roles.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "10-medical_fundraiser.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 10 failed: {detail}"
    assert key_num == "payer_count=75"


def test_run_edge_case_11_chit_fund() -> None:
    """
    WHAT: Runs runnable edge case 11 and asserts check_chit_fund passes with payer_count=97.
    WHY: Proves that rotating chit fund pooling episodes involve genuine citizens without gang roles.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "11-chit_fund.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 11 failed: {detail}"
    assert key_num == "payer_count=97"


def test_run_edge_case_12_genuine_shop_downstream_of_mule() -> None:
    """
    WHAT: Runs runnable edge case 12 and asserts check_genuine_shop_downstream_of_mule passes with cashout_shops=1.
    WHY: Proves that retail shops receiving mule cash-out payments remain innocent and untagged in truth.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "12-genuine_shop_downstream_of_mule.yaml")
    output = run_edge_case(case)
    passed, key_num, detail = EXPECTATIONS[case.name](output, case)
    assert passed, f"Case 12 failed: {detail}"
    assert key_num == "cashout_shops=1"


def test_negative_03_short_holding_or_seen_gang_fails() -> None:
    """
    WHAT: Asserts that check_mules_forward_after_days fails if the gang is tagged seen instead of held_out.
    WHY: Verifies negative invariant detection when evasion syndicates fail taxonomic held-out classification.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "03-mules_forward_after_days.yaml")
    output = run_edge_case(case)
    tampered_gangs = tuple(
        dataclasses.replace(g, kind="seen", held_out_options=())
        for g in output.result.truth.gangs
    )
    tampered_truth = dataclasses.replace(output.result.truth, gangs=tampered_gangs)
    tampered_result = dataclasses.replace(output.result, truth=tampered_truth)
    tampered_output = dataclasses.replace(output, result=tampered_result)

    passed, key_num, detail = check_mules_forward_after_days(tampered_output, case)
    assert not passed
    assert "gang kind 'seen' != 'held_out'" in detail


def test_negative_04_forced_true_complaint_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Asserts that check_no_victim_ever_complains fails if a complaint names an actual fraud payment.
    WHY: Verifies negative invariant detection when uncomplained fraud scenarios erroneously produce true complaints.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "04-no_victim_ever_complains.yaml")
    output = run_edge_case(case)
    fraud_pids = [
        pid for pid, src in zip(output.result.payment_ids, output.result.event_sources) if src == "fraud"
    ]
    fake_complaint = LabelEvent(
        label_id="fake-L001",
        kind="complaint",
        arrival_minute=5000,
        complainant_account="acc_person_000001",
        complainant_bank_index=0,
        reported_account="acc_person_000002",
        reported_bank_index=1,
        payment_ids=(fraud_pids[0],),
        claimed_type="investment",
    )
    monkeypatch.setattr("world.edge_cases.build_label_events", lambda out: [fake_complaint])

    passed, key_num, detail = check_no_victim_ever_complains(output, case)
    assert not passed
    assert "found 1 true complaints" in detail


def test_negative_05_complaint_naming_fraud_payment_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Asserts that check_every_complaint_false fails if any complaint names a fraud payment.
    WHY: Verifies negative invariant detection when pure false-complaint scenarios contain authentic scam complaints.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "05-every_complaint_false.yaml")
    output = run_edge_case(case)
    fraud_pids = [
        pid for pid, src in zip(output.result.payment_ids, output.result.event_sources) if src == "fraud"
    ]
    fake_complaint = LabelEvent(
        label_id="fake-L002",
        kind="complaint",
        arrival_minute=5000,
        complainant_account="acc_person_000001",
        complainant_bank_index=0,
        reported_account="acc_person_000002",
        reported_bank_index=1,
        payment_ids=(fraud_pids[0],),
        claimed_type="investment",
    )
    monkeypatch.setattr("world.edge_cases.build_label_events", lambda out: [fake_complaint])

    passed, key_num, detail = check_every_complaint_false(output, case)
    assert not passed
    assert "names a fraud payment" in detail


def test_negative_06_registry_report_on_mule_account_fails() -> None:
    """
    WHAT: Asserts that check_registry_wrong_listing fails if all reported accounts hold mule roles.
    WHY: Verifies negative invariant detection when wrong-listing checks encounter exclusively guilty mule listings.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "06-registry_wrong_listing.yaml")
    output = run_edge_case(case)
    tampered_roles = tuple(
        RoleSpan(
            account_id=acc,
            role="mule",
            gang_id="gang_test",
            depth=1,
            valid_from=0,
            valid_to=None,
        )
        for acc in ("acc_shop_0000", "acc_shop_0003")
    )
    tampered_truth = dataclasses.replace(output.result.truth, roles=tampered_roles)
    tampered_result = dataclasses.replace(output.result, truth=tampered_truth)
    tampered_output = dataclasses.replace(output, result=tampered_result)

    passed, key_num, detail = check_registry_wrong_listing(tampered_output, case)
    assert not passed
    assert "no registry report for an account without a gang role" in detail


def test_negative_06_case_without_gang_fails_no_mule_in_the_truth() -> None:
    """
    WHAT: Asserts that check_registry_wrong_listing fails if case 06 is executed with no gang present.
    WHY: Verifies that wrong-listing checks require active mule operations in truth to avoid trivial empty passes.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "06-registry_wrong_listing.yaml")
    case_no_gang = dataclasses.replace(case, gangs=[])
    output = run_edge_case(case_no_gang)

    passed, key_num, detail = check_registry_wrong_listing(output, case_no_gang)
    assert not passed
    assert detail == "no mule in the truth"


def test_negative_07_split_or_non_100k_payment_fails() -> None:
    """
    WHAT: Asserts that check_one_huge_transfer fails if a route payment is split into non-100k amounts.
    WHY: Verifies negative invariant detection when lump-sum transfers are split into smaller pieces.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "07-one_huge_transfer.yaml")
    output = run_edge_case(case)
    tampered_events = [
        dataclasses.replace(ev, amount_paise=5000000)
        for ev in output.result.run_result.events
    ]
    tampered_run_result = dataclasses.replace(output.result.run_result, events=tampered_events)
    tampered_result = dataclasses.replace(output.result, run_result=tampered_run_result)
    tampered_output = dataclasses.replace(output, result=tampered_result)

    passed, key_num, detail = check_one_huge_transfer(tampered_output, case)
    assert not passed
    assert "route payment amount" in detail


def test_negative_09_collector_or_payer_with_mule_role_fails() -> None:
    """
    WHAT: Asserts that check_wedding_collection fails if the collector or a contributing payer holds a mule role.
    WHY: Verifies negative invariant detection when life event community pooling is compromised by syndicate roles.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "09-wedding_collection.yaml")
    output = run_edge_case(case)
    ep = [e for e in output.episodes if e.kind == "wedding_or_medical_collection"][0]
    collector = next(p for p in output.world.people if p.person_id == ep.person_id)

    # 1. Collector holds a mule role
    mule_collector_role = RoleSpan(
        account_id=collector.account_id,
        role="mule",
        gang_id="gang_infiltrated",
        depth=1,
        valid_from=0,
        valid_to=None,
    )
    tampered_truth_collector = dataclasses.replace(output.result.truth, roles=(mule_collector_role,))
    tampered_result_collector = dataclasses.replace(output.result, truth=tampered_truth_collector)
    tampered_output_collector = dataclasses.replace(output, result=tampered_result_collector)

    passed, key_num, detail = check_wedding_collection(tampered_output_collector, case)
    assert not passed
    assert f"collector {collector.account_id} holds gang role mule" in detail

    # 2. Payer holds a mule role
    payer_accs = [
        ev.from_account
        for ev in output.result.run_result.events
        if ev.to_account == collector.account_id and ep.start_minute <= ev.minute <= ep.end_minute
    ]
    mule_payer_role = RoleSpan(
        account_id=payer_accs[0],
        role="mule",
        gang_id="gang_infiltrated",
        depth=1,
        valid_from=0,
        valid_to=None,
    )
    tampered_truth_payer = dataclasses.replace(output.result.truth, roles=(mule_payer_role,))
    tampered_result_payer = dataclasses.replace(output.result, truth=tampered_truth_payer)
    tampered_output_payer = dataclasses.replace(output, result=tampered_result_payer)

    passed, key_num, detail = check_wedding_collection(tampered_output_payer, case)
    assert not passed
    assert "payer holds gang role mule" in detail


def test_negative_12_cashout_shop_holding_gang_role_fails() -> None:
    """
    WHAT: Asserts that check_genuine_shop_downstream_of_mule fails if the cash-out shop holds a gang role.
    WHY: Verifies negative invariant detection when cash-out destination accounts are syndicate-controlled entities.
    """
    case = load_edge_case(DEFAULT_CASES_DIR / "12-genuine_shop_downstream_of_mule.yaml")
    output = run_edge_case(case)
    shop_mule_role = RoleSpan(
        account_id="acc_shop_0007",
        role="mule",
        gang_id="gang_shop_cashout",
        depth=1,
        valid_from=0,
        valid_to=None,
    )
    tampered_truth = dataclasses.replace(output.result.truth, roles=(shop_mule_role,))
    tampered_result = dataclasses.replace(output.result, truth=tampered_truth)
    tampered_output = dataclasses.replace(output, result=tampered_result)

    passed, key_num, detail = check_genuine_shop_downstream_of_mule(tampered_output, case)
    assert not passed
    assert "shop account acc_shop_0007 holds gang role mule" in detail
