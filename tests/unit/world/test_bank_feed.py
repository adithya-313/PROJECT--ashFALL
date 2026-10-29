# File: tests/unit/world/test_bank_feed.py
# Purpose: Unit tests validating bank feed payment projection, R7 visibility, and boundary leakage guards.
"""
WHAT: Tests the payment projection transformations in world/bank_feed.py.
It verifies strict R7 domain visibility rules, dual-side projection on successful and same-bank
transfers, payee classification by prefix, IST timezone preservation, boundary leakage guards,
and complete schema conformance across the 2,915-event tiny world simulation run.

WHY: Guarantees that tenant commercial banks receive only approved transaction sides and fields,
protecting against simulation state leakage while ensuring exact ledger balances across banks.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any

import pytest

from core.event_clock import IST, world_minute_to_ist
from world.bank_feed import (
    BANK_VISIBLE_ACCOUNT_EVENT_FIELDS,
    BANK_VISIBLE_COMPLAINT_FIELDS,
    BANK_VISIBLE_PAYMENT_FIELDS,
    GANG_DEVICE_DETAIL_FIELDS,
    GENUINE_DEVICE_DETAIL_FIELDS,
    BankFeed,
    ProjectionLeakError,
    check_bank_visible,
    project_payment,
    project_run,
    resolve_bank_id,
)
from world.bank_feed_config import load_bank_feed_config
from world.device_runtime import DeviceStamp
from world.failures import FAILURE_DOMAINS, PaymentEvent
from world.fast_mode import FastRunSettings
from world.gang_devices import GangDeviceEvent
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.label_records import LabelEvent

CLOCK_START = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def bank_config() -> Any:
    """Provides a verified BankFeedConfig instance for the test module."""
    return load_bank_feed_config()


@pytest.fixture(scope="module")
def tiny_world_output() -> InteractiveWorldOutput:
    """Executes the standard tiny world simulation run with fraud enabled."""
    return run_interactive_world(
        FastRunSettings(101, 60, 12, 61, "training"),
        runner_settings=RunnerSettings(fraud_enabled=True),
    )


def test_r7_visibility_rules_per_domain(bank_config: Any) -> None:
    """Verifies that failed payments emit credit sides only for failure domains permitted by R7 rules."""
    stamp = DeviceStamp("fp_android_v3_12345678", 3, False)
    for domain in FAILURE_DOMAINS:
        event = PaymentEvent(
            minute=100,
            from_account="acc_person_000001",
            to_account="acc_person_000002",
            amount_paise=50000,
            from_bank_index=0,
            to_bank_index=1,
            status="failed",
            reason="simulated failure",
            failure_domain=domain,
        )
        sides = project_payment(event, "p-001", stamp, CLOCK_START, bank_config)
        assert len(sides) >= 1
        assert sides[0].side == "debit"
        assert sides[0].bank_id == "bank_0"
        assert sides[0].failure_domain == domain

        # Under R7, only receiver_bank failures reach the beneficiary bank.
        expected_credit = bank_config.failed_payment_visibility[domain]["receiver"]
        if expected_credit:
            assert len(sides) == 2
            assert sides[1].side == "credit"
            assert sides[1].bank_id == "bank_1"
        else:
            assert len(sides) == 1


def test_successful_payment_dual_sides_and_same_bank(bank_config: Any) -> None:
    """Verifies that successful payments create two sides, including when originating and terminating at one bank."""
    stamp = DeviceStamp("fp_android_v3_12345678", 3, False)

    # 1. Inter-bank successful payment.
    inter_bank_event = PaymentEvent(
        minute=120,
        from_account="acc_person_000001",
        to_account="acc_person_000002",
        amount_paise=75000,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason=None,
        failure_domain=None,
    )
    inter_sides = project_payment(inter_bank_event, "p-002", stamp, CLOCK_START, bank_config)
    assert len(inter_sides) == 2
    assert inter_sides[0].side == "debit" and inter_sides[0].bank_id == "bank_0"
    assert inter_sides[1].side == "credit" and inter_sides[1].bank_id == "bank_1"

    # 2. Intra-bank (same bank) successful payment.
    intra_bank_event = PaymentEvent(
        minute=130,
        from_account="acc_person_000001",
        to_account="acc_person_000003",
        amount_paise=25000,
        from_bank_index=0,
        to_bank_index=0,
        status="success",
        reason=None,
        failure_domain=None,
    )
    intra_sides = project_payment(intra_bank_event, "p-003", stamp, CLOCK_START, bank_config)
    assert len(intra_sides) == 2
    assert intra_sides[0].side == "debit" and intra_sides[0].bank_id == "bank_0"
    assert intra_sides[1].side == "credit" and intra_sides[1].bank_id == "bank_0"


def test_payee_type_resolution_and_invalid_prefix(bank_config: Any) -> None:
    """Verifies payee classification for person and shop accounts, rejecting unknown account prefixes."""
    stamp = DeviceStamp("fp_android_v3_12345678", 3, False)

    # Person payee.
    person_event = PaymentEvent(
        minute=10,
        from_account="acc_person_000001",
        to_account="acc_person_000002",
        amount_paise=1000,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason=None,
        failure_domain=None,
    )
    person_sides = project_payment(person_event, "p-004", stamp, CLOCK_START, bank_config)
    assert person_sides[0].payee_type == "person"
    assert person_sides[1].payee_type == "person"

    # Shop payee.
    shop_event = PaymentEvent(
        minute=10,
        from_account="acc_person_000001",
        to_account="acc_shop_0001",
        amount_paise=2000,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason=None,
        failure_domain=None,
    )
    shop_sides = project_payment(shop_event, "p-005", stamp, CLOCK_START, bank_config)
    assert shop_sides[0].payee_type == "merchant"
    assert shop_sides[1].payee_type == "merchant"

    # Invalid account prefix.
    invalid_event = PaymentEvent(
        minute=10,
        from_account="acc_person_000001",
        to_account="unknown_account_9999",
        amount_paise=1000,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason=None,
        failure_domain=None,
    )
    with pytest.raises(ValueError, match="has unknown prefix"):
        project_payment(invalid_event, "p-006", stamp, CLOCK_START, bank_config)


def test_event_time_has_ist_offset_and_matches_helper(bank_config: Any) -> None:
    """Verifies that generated event timestamps maintain a +05:30 offset and equal world_minute_to_ist."""
    stamp = DeviceStamp("fp_android_v3_12345678", 3, False)
    test_minute = 1440
    event = PaymentEvent(
        minute=test_minute,
        from_account="acc_person_000001",
        to_account="acc_person_000002",
        amount_paise=1000,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason=None,
        failure_domain=None,
    )
    sides = project_payment(event, "p-007", stamp, CLOCK_START, bank_config)
    expected_time = world_minute_to_ist(test_minute, CLOCK_START)
    for side in sides:
        assert side.event_time == expected_time
        assert side.event_time.tzinfo == IST
        assert side.event_time.utcoffset() is not None
        assert side.event_time.utcoffset().total_seconds() == 19800


def test_check_bank_visible_rejects_purpose_field() -> None:
    """Verifies that unexpected keys such as 'purpose' are rejected with a descriptive ProjectionLeakError."""
    leaked_record = {
        "bank_id": "bank_0",
        "payment_id": "p-001",
        "purpose": "rent_payment",
    }
    with pytest.raises(ProjectionLeakError) as exc_info:
        check_bank_visible(leaked_record, BANK_VISIBLE_PAYMENT_FIELDS)
    assert "purpose" in str(exc_info.value)


def test_full_world_schema_and_counts(tiny_world_output: InteractiveWorldOutput, bank_config: Any) -> None:
    """Projects every event from the tiny world, verifying strict schema compliance and computed counts."""
    result = tiny_world_output.result
    clock_start = tiny_world_output.plan.clock.start_date
    events = result.run_result.events
    stamps = result.run_result.event_sender_devices
    payment_ids = result.payment_ids

    total_debits = 0
    total_credits = 0

    expected_debits = len(events)
    expected_credits = sum(
        1 for ev in events if ev.status == "success" or ev.failure_domain == "receiver_bank"
    )

    for idx, event in enumerate(events):
        pid = payment_ids[idx]
        stamp = stamps[idx]
        sides = project_payment(event, pid, stamp, clock_start, bank_config)
        for side in sides:
            side_dict = asdict(side)
            assert set(side_dict.keys()) == BANK_VISIBLE_PAYMENT_FIELDS
            if side.side == "debit":
                total_debits += 1
            elif side.side == "credit":
                total_credits += 1

    assert total_debits == expected_debits
    assert total_credits == expected_credits


def test_device_event_mapping_and_counts_on_tiny_world(tiny_world_output: InteractiveWorldOutput) -> None:
    """Verifies mapping of device events including phone_change and checks total projected plus dropped counts."""
    feed = project_run(tiny_world_output)
    assert isinstance(feed, BankFeed)

    total_projected = sum(len(events) for events in feed.account_events_by_bank.values())
    total_dropped = sum(feed.dropped_device_events_by_kind.values())
    total_raw_device_events = len(tiny_world_output.result.applied_device_events)

    assert total_raw_device_events == 32
    # 32 device events + 72 existing accounts = 104 total projected and dropped account events
    assert total_projected + total_dropped == 104

    # Verify that at least one phone_change was mapped to device_binding_change with clean details.
    all_account_events = [ev for bank_events in feed.account_events_by_bank.values() for ev in bank_events]
    binding_events = [ev for ev in all_account_events if ev.kind == "device_binding_change"]
    assert len(binding_events) >= 1

    for ev in binding_events:
        assert set(ev.details.keys()) == GENUINE_DEVICE_DETAIL_FIELDS
        assert "platform" in ev.details
        assert "fingerprint_before" in ev.details
        assert "fingerprint_after" in ev.details
        assert "os_generation_before" in ev.details
        assert "os_generation_after" in ev.details
        assert "counts_as_new_device" not in ev.details
        assert "device_id" not in ev.details


def test_opened_account_drops_owner_kind(tiny_world_output: InteractiveWorldOutput) -> None:
    """Verifies that mid-run opened accounts become account_opened events with owner_kind and stream dropped."""
    from dataclasses import replace

    target_person = next(p for p in tiny_world_output.world.people if p.bank_index == 1)
    synthetic_opened = (300, target_person.account_id, "person", "new_stream", 1, 500000)
    modified_result = replace(tiny_world_output.result, opened_accounts=[synthetic_opened])
    modified_output = replace(tiny_world_output, result=modified_result)

    feed = project_run(modified_output)
    bank_1_events = feed.account_events_by_bank["bank_1"]
    open_events = [ev for ev in bank_1_events if ev.account_id == target_person.account_id]

    assert len(open_events) == 1
    ev = open_events[0]
    assert ev.kind == "account_opened"
    assert ev.bank_id == "bank_1"
    assert ev.event_id == f"{tiny_world_output.result.run_id}-open-{target_person.account_id}"
    assert ev.details == {"home_region": target_person.region}

    ev_dict = asdict(ev)
    assert "owner_kind" not in ev_dict
    assert "opening_stream" not in ev_dict
    check_bank_visible(ev_dict, BANK_VISIBLE_ACCOUNT_EVENT_FIELDS)


def test_constructed_label_event_becomes_complaint_for_complainant_bank(
    tiny_world_output: InteractiveWorldOutput,
) -> None:
    """Verifies that a LabelEvent transforms into a BankFeedComplaint for complainant bank with kind dropped."""
    label_ev = LabelEvent(
        label_id="test-run-L0001",
        kind="complaint",
        arrival_minute=5000,
        complainant_account="acc_person_000001",
        complainant_bank_index=0,
        reported_account="acc_shop_0001",
        reported_bank_index=1,
        payment_ids=("test-run-000001", "test-run-000002"),
        claimed_type="investment",
    )

    feed = project_run(tiny_world_output, label_events=[label_ev])

    # The complaint must appear strictly at complainant bank_0, never reported bank_1.
    assert len(feed.complaints_by_bank["bank_0"]) == 1
    assert len(feed.complaints_by_bank["bank_1"]) == 0

    complaint = feed.complaints_by_bank["bank_0"][0]
    assert complaint.bank_id == "bank_0"
    assert complaint.complaint_id == "test-run-L0001"
    assert complaint.complainant_account == "acc_person_000001"
    assert complaint.reported_account == "acc_shop_0001"
    assert complaint.reported_bank_id == "bank_1"
    assert complaint.payment_ids == ("test-run-000001", "test-run-000002")
    assert complaint.claimed_type == "investment"

    c_dict = asdict(complaint)
    assert "kind" not in c_dict
    check_bank_visible(c_dict, BANK_VISIBLE_COMPLAINT_FIELDS)


def test_project_run_is_deterministic(tiny_world_output: InteractiveWorldOutput) -> None:
    """Verifies that repeated invocations of project_run yield bitwise identical BankFeed structures."""
    feed1 = project_run(tiny_world_output)
    feed2 = project_run(tiny_world_output)
    assert feed1 == feed2


def test_no_record_in_any_bank_list_belongs_to_another_bank(tiny_world_output: InteractiveWorldOutput) -> None:
    """Ensures that no record inside any bank-specific tuple contains a mismatched foreign bank_id."""
    label_ev = LabelEvent(
        label_id="test-run-L0002",
        kind="complaint",
        arrival_minute=6000,
        complainant_account="acc_person_000001",
        complainant_bank_index=0,
        reported_account="acc_shop_0001",
        reported_bank_index=1,
        payment_ids=("test-run-000001",),
        claimed_type="phishing",
    )
    feed = project_run(tiny_world_output, label_events=[label_ev])

    for bank_id, payments in feed.payments_by_bank.items():
        for payment in payments:
            assert payment.bank_id == bank_id

    for bank_id, account_events in feed.account_events_by_bank.items():
        for acct_ev in account_events:
            assert acct_ev.bank_id == bank_id

    for bank_id, complaints in feed.complaints_by_bank.items():
        for complaint in complaints:
            assert complaint.bank_id == bank_id


def test_debit_and_credit_side_device_stamp_projection(bank_config: Any) -> None:
    """Verifies that the debit side carries the client device stamp while the credit side always sets both None."""
    stamp = DeviceStamp("fp_android_v3_abcdef12", 4, False)
    event = PaymentEvent(
        minute=200,
        from_account="acc_person_000001",
        to_account="acc_person_000002",
        amount_paise=30000,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason=None,
        failure_domain=None,
    )

    # 1. Payment with active device stamp: debit has stamp fields, credit has None.
    sides = project_payment(event, "p-stamp-1", stamp, CLOCK_START, bank_config)
    assert len(sides) == 2
    debit, credit = sides[0], sides[1]
    assert debit.sender_device_fingerprint == "fp_android_v3_abcdef12"
    assert debit.sender_os_generation == 4
    assert credit.sender_device_fingerprint is None
    assert credit.sender_os_generation is None

    # 2. Payment with None device stamp: debit side sets fingerprint and generation to None.
    sides_none = project_payment(event, "p-stamp-2", None, CLOCK_START, bank_config)
    assert sides_none[0].sender_device_fingerprint is None
    assert sides_none[0].sender_os_generation is None


def test_counterparty_bank_id_cross_bank_direction(bank_config: Any) -> None:
    """Verifies counterparty_bank_id orientation: beneficiary on debit side and originator on credit side."""
    event = PaymentEvent(
        minute=210,
        from_account="acc_person_000001",
        to_account="acc_person_000002",
        amount_paise=45000,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason=None,
        failure_domain=None,
    )
    sides = project_payment(event, "p-cp-1", None, CLOCK_START, bank_config)
    assert len(sides) == 2
    debit, credit = sides[0], sides[1]

    # Debit side (posted at sender bank_0) references receiver bank_1 as counterparty.
    assert debit.bank_id == "bank_0"
    assert debit.counterparty_bank_id == "bank_1"
    assert debit.sender_bank_id == "bank_0"
    assert debit.receiver_bank_id == "bank_1"

    # Credit side (posted at receiver bank_1) references sender bank_0 as counterparty.
    assert credit.bank_id == "bank_1"
    assert credit.counterparty_bank_id == "bank_0"
    assert credit.sender_bank_id == "bank_0"
    assert credit.receiver_bank_id == "bank_1"


def test_gang_device_events_projection_and_filtering(tiny_world_output: InteractiveWorldOutput) -> None:
    """Verifies projection of mapped gang events, dropped counter tracking, and unknown account validation."""
    gang_sim_swap = GangDeviceEvent(
        minute=200,
        device_id="dev_ext_000001",
        account_id="acc_person_000001",
        kind="sim_swap",
        platform="android",
        os_generation=3,
        fingerprint="fp_android_v3_gang1234",
    )
    gang_acquired = GangDeviceEvent(
        minute=150,
        device_id="dev_ext_000002",
        account_id=None,
        kind="device_acquired",
        platform="android",
        os_generation=3,
        fingerprint="fp_android_v3_gang5678",
    )
    gang_emulator = GangDeviceEvent(
        minute=250,
        device_id="dev_ext_000001",
        account_id="acc_person_000001",
        kind="emulator_enabled",
        platform="android",
        os_generation=3,
        fingerprint="fp_android_v3_gang1234",
    )

    modified_result = replace(
        tiny_world_output.result,
        applied_gang_device_events=[gang_sim_swap, gang_acquired, gang_emulator],
    )
    modified_output = replace(tiny_world_output, result=modified_result)
    feed = project_run(modified_output)

    # 1. Verify sim_swap on known account is projected at the account's bank (bank_0).
    sim_swap_events = [ev for ev in feed.account_events_by_bank["bank_0"] if ev.kind == "sim_swap"]
    assert len(sim_swap_events) == 1
    ev = sim_swap_events[0]
    assert ev.account_id == "acc_person_000001"
    assert set(ev.details.keys()) == GANG_DEVICE_DETAIL_FIELDS
    assert ev.details["platform"] == "android"
    assert ev.details["fingerprint"] == "fp_android_v3_gang1234"
    assert ev.details["os_generation"] == 3

    # 2. Verify device_acquired and emulator_enabled are not projected but counted in dropped events.
    assert feed.dropped_device_events_by_kind["device_acquired"] == 1
    assert feed.dropped_device_events_by_kind["emulator_enabled"] == 1

    # 3. Verify sim_swap on unknown account raises a descriptive ValueError.
    bad_gang_ev = GangDeviceEvent(
        minute=200,
        device_id="dev_ext_000001",
        account_id="acc_unknown_999999",
        kind="sim_swap",
        platform="android",
        os_generation=3,
        fingerprint="fp_android_v3_gang1234",
    )
    bad_output = replace(
        tiny_world_output,
        result=replace(tiny_world_output.result, applied_gang_device_events=[bad_gang_ev]),
    )
    with pytest.raises(ValueError, match="Gang device event references unknown account 'acc_unknown_999999'"):
        project_run(bad_output)


def test_details_dict_extra_key_raises_projection_leak_error() -> None:
    """Verifies that unauthorized keys such as counts_as_new_device in account event details raise a leak error."""
    leaked_details = {
        "platform": "android",
        "fingerprint": "fp_android_v3_12345678",
        "os_generation": 3,
        "counts_as_new_device": True,
    }
    with pytest.raises(ProjectionLeakError) as exc_info:
        check_bank_visible(leaked_details, GANG_DEVICE_DETAIL_FIELDS)
    assert "counts_as_new_device" in str(exc_info.value)


def test_unknown_bank_index_raises_value_error_not_key_error(bank_config: Any) -> None:
    """Verifies that out-of-range bank indexes raise a descriptive ValueError rather than a bare KeyError."""
    event = PaymentEvent(
        minute=100,
        from_account="acc_person_000001",
        to_account="acc_person_000002",
        amount_paise=1000,
        from_bank_index=999,
        to_bank_index=0,
        status="success",
        reason=None,
        failure_domain=None,
    )
    with pytest.raises(ValueError) as exc_info:
        project_payment(event, "p-err", None, CLOCK_START, bank_config)
    # Check that error is not a KeyError and names the violating index and valid options.
    assert not issubclass(exc_info.type, KeyError)
    assert "Unknown bank index 999; expected one of [0, 1, 2]" in str(exc_info.value)

    # Also verify direct resolve_bank_id helper behavior.
    with pytest.raises(ValueError) as helper_exc:
        resolve_bank_id(42, bank_config)
    assert not issubclass(helper_exc.type, KeyError)
    assert "Unknown bank index 42; expected one of [0, 1, 2]" in str(helper_exc.value)


