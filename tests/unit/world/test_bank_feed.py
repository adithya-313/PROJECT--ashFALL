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

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import pytest

from core.event_clock import IST, world_minute_to_ist
from world.bank_feed import (
    BANK_VISIBLE_ACCOUNT_EVENT_FIELDS,
    BANK_VISIBLE_COMPLAINT_FIELDS,
    BANK_VISIBLE_PAYMENT_FIELDS,
    BankFeed,
    ProjectionLeakError,
    check_bank_visible,
    project_payment,
    project_run,
)
from world.bank_feed_config import load_bank_feed_config
from world.device_runtime import DeviceStamp
from world.failures import FAILURE_DOMAINS, PaymentEvent
from world.fast_mode import FastRunSettings
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
    assert total_projected + total_dropped == 32

    # Verify that at least one phone_change was mapped to device_binding_change with clean details.
    all_account_events = [ev for bank_events in feed.account_events_by_bank.values() for ev in bank_events]
    binding_events = [ev for ev in all_account_events if ev.kind == "device_binding_change"]
    assert len(binding_events) >= 1

    for ev in binding_events:
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

    synthetic_opened = (300, "acc_person_000999", "person", "new_stream", 1, 500000)
    modified_result = replace(tiny_world_output.result, opened_accounts=[synthetic_opened])
    modified_output = replace(tiny_world_output, result=modified_result)

    feed = project_run(modified_output)
    bank_1_events = feed.account_events_by_bank["bank_1"]
    open_events = [ev for ev in bank_1_events if ev.account_id == "acc_person_000999"]

    assert len(open_events) == 1
    ev = open_events[0]
    assert ev.kind == "account_opened"
    assert ev.bank_id == "bank_1"
    assert ev.event_id == f"{tiny_world_output.result.run_id}-open-acc_person_000999"
    assert ev.details == {}

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

