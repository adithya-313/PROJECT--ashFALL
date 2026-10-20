"""
Unit tests for mid-run account openings in existing Level 11 person and shop ID formats.
Verifies ID sequence continuation, format regex conformance, payment timing barriers,
money conservation tracking, and stream isolation between genuine and gang openings.
"""

import re

import numpy as np
import pytest

from world.calendar_runner import PlannedPayment
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.fast_mode import FastRunSettings
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.interactive_state import start_runner_state
from world.outside_events import OutsideEvent, validate_outside_event

TINY = FastRunSettings(101, 60, 12, 61, "training")


def test_mid_run_account_ids_continue_numbering_and_match_regex() -> None:
    """
    Verifies that mid-run accounts continue Level 11 sequence numbering and match canonical regex patterns.
    Tests interleaved opening of persons and shops, asserting sequential numbering and identifier formats.
    """
    events = (
        OutsideEvent(minute=100, kind="open_account", owner_kind="person", opening_stream="genuine_account_openings"),
        OutsideEvent(minute=200, kind="open_account", owner_kind="shop", opening_stream="genuine_account_openings"),
        OutsideEvent(minute=300, kind="open_account", owner_kind="person", opening_stream="genuine_account_openings"),
        OutsideEvent(minute=400, kind="open_account", owner_kind="shop", opening_stream="genuine_account_openings"),
    )
    output = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True),
        fraud_inputs=FraudInputs(outside_events=events),
        code_version="test-mid-run-accounts",
    )

    opened = output.result.opened_accounts
    assert len(opened) == 4

    expected_ids = ["acc_person_000060", "acc_shop_0012", "acc_person_000061", "acc_shop_0013"]
    actual_ids = [row[1] for row in opened]
    assert actual_ids == expected_ids

    # Assert regex formats continue Level 11 structure.
    for acc in ["acc_person_000060", "acc_person_000061"]:
        assert re.match(r"^acc_person_\d{6}$", acc) is not None
    for acc in ["acc_shop_0012", "acc_shop_0013"]:
        assert re.match(r"^acc_shop_\d{4}$", acc) is not None

    # Verify owner kinds and streams match logged metadata.
    assert [row[2] for row in opened] == ["person", "shop", "person", "shop"]
    assert all(row[3] == "genuine_account_openings" for row in opened)


def test_payment_before_opening_fails_and_succeeds_after_opening() -> None:
    """
    Verifies that payments targeting an account before its opening minute fail while later payments succeed.
    Ensures the runner lookup tables strictly reject unknown destination accounts until opened.
    """
    open_event = OutsideEvent(
        minute=100, kind="open_account", owner_kind="person", opening_stream="genuine_account_openings"
    )
    pmt_early = PlannedPayment(
        minute=50, from_account="acc_person_000001", to_account="acc_person_000060",
        amount_paise=5000, creation_index=0, purpose="everyday",
    )
    pmt_late = PlannedPayment(
        minute=120, from_account="acc_person_000001", to_account="acc_person_000060",
        amount_paise=5000, creation_index=1, purpose="everyday",
    )

    # Early payment must fail because acc_person_000060 is not yet in the bank map at minute 50.
    with pytest.raises(ValueError, match="account acc_person_000060 is not registered with the runner"):
        run_interactive_world(
            TINY,
            RunnerSettings(fraud_enabled=True),
            fraud_inputs=FraudInputs(payments=(pmt_early,), outside_events=(open_event,)),
            code_version="test-mid-run-accounts",
        )

    # Late payment must succeed because account is registered at minute 100.
    output = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True),
        fraud_inputs=FraudInputs(payments=(pmt_late,), outside_events=(open_event,)),
        code_version="test-mid-run-accounts",
    )
    target_events = [
        event for event in output.result.run_result.events
        if event.to_account == "acc_person_000060"
    ]
    assert len(target_events) == 1
    assert target_events[0].status == "success"
    assert target_events[0].amount_paise == 5000


def test_money_conservation_includes_mid_run_opening_balances() -> None:
    """
    Verifies that ledger money conservation checks account for opening balances of accounts created mid-run.
    Checks that total ledger balances match initial world balances plus newly created money.
    """
    events = (
        OutsideEvent(minute=100, kind="open_account", owner_kind="person", opening_stream="genuine_account_openings"),
        OutsideEvent(minute=200, kind="open_account", owner_kind="shop", opening_stream="genuine_account_openings"),
    )
    output = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True),
        fraud_inputs=FraudInputs(outside_events=events),
        code_version="test-mid-run-accounts",
    )

    fresh_world = prepare_world(TINY, None).world
    opening_total = sum(ledger.balance_of(acc) for ledger in fresh_world.ledgers for acc in ledger.accounts)
    expected_total = opening_total + output.result.money_created_paise

    ledgers = output.world.ledgers
    assert output.result.money_created_paise > 0
    assert check_no_negative_balances(ledgers) == []
    assert check_balances_equal_entry_sums(ledgers) == []
    assert check_money_conserved(ledgers, expected_total) == []


def test_stream_isolation_gang_opening_does_not_shift_genuine_openings() -> None:
    """
    Verifies that draws on gang_account_openings (stream 9) do not shift genuine_account_openings (stream 8).
    Executes baseline genuine openings and asserts identical attributes when a gang opening is interleaved.
    """
    genuine_events = (
        OutsideEvent(minute=100, kind="open_account", owner_kind="person", opening_stream="genuine_account_openings"),
        OutsideEvent(minute=200, kind="open_account", owner_kind="shop", opening_stream="genuine_account_openings"),
    )
    output_a = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True),
        fraud_inputs=FraudInputs(outside_events=genuine_events),
        code_version="test-mid-run-accounts",
    )

    # Interleave a gang opening on stream 9 before genuine openings on stream 8.
    interleaved_events = (
        OutsideEvent(minute=50, kind="open_account", owner_kind="person", opening_stream="gang_account_openings"),
        OutsideEvent(minute=100, kind="open_account", owner_kind="person", opening_stream="genuine_account_openings"),
        OutsideEvent(minute=200, kind="open_account", owner_kind="shop", opening_stream="genuine_account_openings"),
    )
    output_b = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True),
        fraud_inputs=FraudInputs(outside_events=interleaved_events),
        code_version="test-mid-run-accounts",
    )

    # Output A genuine person is row 0; Output B genuine person is row 1.
    gen_person_a = output_a.result.opened_accounts[0]
    gen_person_b = output_b.result.opened_accounts[1]
    # Bank index and opening balance drawn on stream 8 must match identically.
    assert gen_person_a[4] == gen_person_b[4]
    assert gen_person_a[5] == gen_person_b[5]

    # Output A genuine shop is row 1; Output B genuine shop is row 2.
    gen_shop_a = output_a.result.opened_accounts[1]
    gen_shop_b = output_b.result.opened_accounts[2]
    assert gen_shop_a[4] == gen_shop_b[4]
    assert gen_shop_a[5] == gen_shop_b[5]


def test_open_account_validation_refusals() -> None:
    """
    Verifies schema validation refusals for malformed open_account outside events.
    Checks that missing owner_kind, invalid owner_kind, unknown stream, or supplied account_id raise ValueError.
    """
    # 1. Missing owner_kind
    with pytest.raises(ValueError, match="needs owner_kind to be 'person' or 'shop'"):
        validate_outside_event(OutsideEvent(100, "open_account", opening_stream="genuine_account_openings"))

    # 2. Invalid owner_kind
    with pytest.raises(ValueError, match="needs owner_kind to be 'person' or 'shop'"):
        validate_outside_event(
            OutsideEvent(100, "open_account", owner_kind="robot", opening_stream="genuine_account_openings")
        )

    # 3. Unknown opening_stream
    with pytest.raises(ValueError, match="needs opening_stream to be one of"):
        validate_outside_event(
            OutsideEvent(100, "open_account", owner_kind="person", opening_stream="magic_stream")
        )

    # 4. Account_id supplied
    with pytest.raises(ValueError, match="must not specify an account_id"):
        validate_outside_event(
            OutsideEvent(
                100, "open_account", account_id="acc_custom_01", owner_kind="person",
                opening_stream="genuine_account_openings",
            )
        )


def test_fraud_off_output_byte_identical_and_opened_accounts_empty() -> None:
    """
    Verifies that running with fraud_enabled=False leaves opened_accounts empty and preserves baseline equality.
    """
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version="test-mid-run-accounts")
    assert output.result.opened_accounts == []


def test_open_account_requires_range_name_in_drawn_parameters() -> None:
    """
    Verifies that open_account does not guess the range name: a world whose drawn parameters lack range_name
    raises KeyError instead of silently using the training range.
    """
    state = start_runner_state(prepare_world(TINY, None).world)
    del state.world.parameters_drawn["range_name"]

    with pytest.raises(KeyError, match="range_name"):
        state.open_account("person", "genuine_account_openings", 100, np.random.default_rng(1))
