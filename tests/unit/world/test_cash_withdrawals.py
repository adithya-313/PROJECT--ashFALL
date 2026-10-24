"""
WHAT: Unit tests for Cash Withdrawals module (Level 12c C-4, D2).
Verifies tie-break ordering, stream child 13 configuration, validation constraints,
ATM card daily limit enforcement and next-day reset, encumbrance blocking (funds_on_hold),
illicit money balance capping and skipping, legal UPI leg splitting on debits, bank outage
failure evaluation, world isolation from payment events and truth, and closed-loop money conservation.

WHY: Cash withdrawals simulate the extraction of criminal proceeds into physical currency.
Rigorous testing guarantees that ledger debits strictly balance money conservation
(opening + created - withdrawn) without leaking into payment events or bank-facing truth.

ISSUE IDs: G2 (cash-out channels), 12-17 (split payments).
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

from core.config_validator import load_yaml
from world.amounts import split_into_legal_payments
from world.cash_withdrawals import (
    CHANNEL_ATM,
    CHANNEL_BRANCH,
    CashWithdrawal,
    process_cash_withdrawal,
    validate_cash_withdrawal,
)
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.fast_mode import FastRunSettings
from world.incidents import Incident
from world.interactive_records import FraudInputs, IllicitSeed, RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.interactive_state import start_runner_state
from world.outside_events import OutsideEvent
from world.run_identity import REPO_ROOT, load_runner_config

FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])


def opening_total(settings: FastRunSettings) -> int:
    """Calculates total opening balance across all bank ledgers for a fresh world."""
    world = prepare_world(settings, None).world
    return sum(ledger.balance_of(account) for ledger in world.ledgers for account in ledger.accounts)


def test_runner_config_has_cash_withdrawals_stream_and_tie_break_rank() -> None:
    """
    Checks that child 13 is cash_withdrawal_failures and the first seven tie_break_order entries
    end with cash_withdrawals, appearing exactly once.
    """
    runner_config = load_runner_config()

    children = runner_config["random_stream_children"]
    assert children["cash_withdrawal_failures"] == 13

    assert runner_config["tie_break_order"][:7] == [
        "outside_events",
        "planned_payments",
        "fraud_payments",
        "resumed_payments",
        "gang_device_events",
        "opened_routine_payments",
        "cash_withdrawals",
    ]
    assert runner_config["tie_break_order"].count("cash_withdrawals") == 1


def test_validate_cash_withdrawal_constraints() -> None:
    """Checks input validation for CashWithdrawal fields."""
    valid_atm = CashWithdrawal(
        minute=100, account_id="acc_person_000000", amount_paise=10000, channel="atm", card_daily_limit_paise=50000
    )
    validate_cash_withdrawal(valid_atm)

    valid_branch = CashWithdrawal(
        minute=100, account_id="acc_person_000000", amount_paise=10000, channel="branch", card_daily_limit_paise=None
    )
    validate_cash_withdrawal(valid_branch)

    # Negative minute
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_cash_withdrawal(
            CashWithdrawal(minute=-1, account_id="acc_person_000000", amount_paise=10000, channel="branch")
        )

    # Empty account id
    with pytest.raises(ValueError, match="non-empty string"):
        validate_cash_withdrawal(
            CashWithdrawal(minute=100, account_id="", amount_paise=10000, channel="branch")
        )

    # Non-positive amount
    with pytest.raises(ValueError, match="positive integer"):
        validate_cash_withdrawal(
            CashWithdrawal(minute=100, account_id="acc_person_000000", amount_paise=0, channel="branch")
        )

    # Invalid channel
    with pytest.raises(ValueError, match="Withdrawal channel must be one of"):
        validate_cash_withdrawal(
            CashWithdrawal(minute=100, account_id="acc_person_000000", amount_paise=10000, channel="crypto")
        )

    # Invalid card daily limit for ATM (zero or negative)
    with pytest.raises(ValueError, match="card_daily_limit_paise to be a positive integer"):
        validate_cash_withdrawal(
            CashWithdrawal(
                minute=100,
                account_id="acc_person_000000",
                amount_paise=10000,
                channel="atm",
                card_daily_limit_paise=0,
            )
        )

    # ATM without card daily limit is refused
    with pytest.raises(ValueError, match="channel 'atm' requires card_daily_limit_paise"):
        validate_cash_withdrawal(
            CashWithdrawal(
                minute=100,
                account_id="acc_person_000000",
                amount_paise=10000,
                channel="atm",
                card_daily_limit_paise=None,
            )
        )

    # Branch with card daily limit is refused
    with pytest.raises(ValueError, match="channel 'branch' requires card_daily_limit_paise to be None"):
        validate_cash_withdrawal(
            CashWithdrawal(
                minute=100,
                account_id="acc_person_000000",
                amount_paise=10000,
                channel="branch",
                card_daily_limit_paise=50000,
            )
        )

    # Unregistered account in state
    prepared = prepare_world(TINY, None)
    state = start_runner_state(prepared.world)
    with pytest.raises(ValueError, match="is not registered with the runner"):
        validate_cash_withdrawal(
            CashWithdrawal(minute=100, account_id="acc_unregistered", amount_paise=10000, channel="branch"),
            state=state,
        )


def test_cash_withdrawal_money_conservation_and_record() -> None:
    """
    Verifies that a successful cash withdrawal debits the ledger, updates money_withdrawn_paise,
    records a CashWithdrawalRecord with status 'success', and maintains total money conservation.
    """
    account = "acc_person_000000"
    withdrawal_amount = 250000  # 2,500 INR in paise

    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=withdrawal_amount),),
        withdrawals=(
            CashWithdrawal(
                minute=100,
                account_id=account,
                amount_paise=withdrawal_amount,
                channel=CHANNEL_BRANCH,
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
    )

    assert len(output.result.cash_withdrawals) == 1
    rec = output.result.cash_withdrawals[0]
    assert rec.status == "success"
    assert rec.requested == withdrawal_amount
    assert rec.withdrawn == withdrawal_amount
    assert rec.channel == CHANNEL_BRANCH
    assert rec.account == account
    assert rec.reason is None
    assert rec.domain is None
    assert rec.id.endswith("-w-000000")

    assert output.result.money_withdrawn_paise == withdrawal_amount

    # Money conservation check: opening + created - withdrawn
    ledgers = output.world.ledgers
    expected = opening_total(TINY) + output.result.money_created_paise - output.result.money_withdrawn_paise
    assert check_no_negative_balances(ledgers) == []
    assert check_balances_equal_entry_sums(ledgers) == []
    assert check_money_conserved(ledgers, expected) == []


def test_withdrawal_from_an_account_opened_during_the_run_completes() -> None:
    """A gang-opened account gets money after its opening and is withdrawn from at an ATM; no registration error."""
    from world.calendar_runner import PlannedPayment
    from world.interactive_records import IllicitMoneyRule

    handle = "new:g:0"
    opening = OutsideEvent(
        100, "open_account", owner_kind="person", opening_stream="gang_account_openings", opening_handle=handle
    )
    inputs = FraudInputs(
        payments=(PlannedPayment(300, "acc_person_000001", handle, 20_000, 0, "everyday"),),
        money_rules=(IllicitMoneyRule(0, False, True),),
        outside_events=(opening,),
        withdrawals=(CashWithdrawal(5_000, handle, 10_000, CHANNEL_ATM, 500_000),),
    )

    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), fraud_inputs=inputs)

    # With 60 people the first opening is person number 60, written with six digits.
    assert len(output.result.cash_withdrawals) == 1
    record = output.result.cash_withdrawals[0]
    assert record.account == "acc_person_000060"
    assert record.status == "success"
    assert record.withdrawn == 10_000


def test_withdrawal_on_an_account_that_is_never_opened_is_refused_at_queue_time() -> None:
    """An account that is neither registered nor opened later is refused before the run, with the old message."""
    inputs = FraudInputs(withdrawals=(CashWithdrawal(100, "acc_person_999999", 10_000, CHANNEL_BRANCH),))
    with pytest.raises(ValueError, match="Account 'acc_person_999999' is not registered with the runner"):
        run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), fraud_inputs=inputs)


def test_processing_a_withdrawal_on_an_unregistered_account_raises_a_plain_error() -> None:
    """If a withdrawal runs before its account is registered, it stops with a message naming account and minute."""
    from types import SimpleNamespace

    context = SimpleNamespace(state=SimpleNamespace(bank_map={}), gang_reactions_enabled=False)
    with pytest.raises(ValueError, match="cash withdrawal at minute 42 uses account acc_new, which is not registered"):
        process_cash_withdrawal(context, CashWithdrawal(42, "acc_new", 10_000, CHANNEL_BRANCH))


def test_atm_daily_limit_exceeded_fails_and_balance_untouched() -> None:
    """
    Tests that ATM withdrawal exceeding card daily limit fails with reason 'atm_daily_limit'
    and domain 'sender_bank', leaving ledger balance completely untouched.
    """
    account = "acc_person_000000"
    daily_limit = 200000  # 2,000 INR
    requested = 300000    # 3,000 INR (exceeds limit)

    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=requested),),
        withdrawals=(
            CashWithdrawal(
                minute=100,
                account_id=account,
                amount_paise=requested,
                channel=CHANNEL_ATM,
                card_daily_limit_paise=daily_limit,
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
    )

    assert len(output.result.cash_withdrawals) == 1
    rec = output.result.cash_withdrawals[0]
    assert rec.status == "failed"
    assert rec.reason == "atm_daily_limit"
    assert rec.domain == "sender_bank"
    assert rec.withdrawn == 0
    assert rec.requested == requested

    assert output.result.money_withdrawn_paise == 0

    # Conservation must equal opening total exactly since no money was withdrawn
    ledgers = output.world.ledgers
    expected = opening_total(TINY) + output.result.money_created_paise
    assert check_money_conserved(ledgers, expected) == []


def test_atm_daily_limit_resets_next_day() -> None:
    """
    Tests that ATM daily withdrawals accumulate within the same day (minute // 1440)
    and reset on subsequent days.
    """
    account = "acc_person_000000"
    daily_limit = 200000  # 2,000 INR limit per day
    day0_w1 = 150000      # 1,500 INR at minute 100 (day 0) -> succeeds
    day0_w2 = 100000      # 1,000 INR at minute 200 (day 0) -> fails (1500 + 1000 > 2000)
    day1_w1 = 150000      # 1,500 INR at minute 1500 (day 1) -> succeeds (limit reset)

    total_illicit = day0_w1 + day0_w2 + day1_w1
    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=total_illicit),),
        withdrawals=(
            CashWithdrawal(
                minute=100,
                account_id=account,
                amount_paise=day0_w1,
                channel=CHANNEL_ATM,
                card_daily_limit_paise=daily_limit,
            ),
            CashWithdrawal(
                minute=200,
                account_id=account,
                amount_paise=day0_w2,
                channel=CHANNEL_ATM,
                card_daily_limit_paise=daily_limit,
            ),
            CashWithdrawal(
                minute=1500,
                account_id=account,
                amount_paise=day1_w1,
                channel=CHANNEL_ATM,
                card_daily_limit_paise=daily_limit,
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
    )

    records = output.result.cash_withdrawals
    assert len(records) == 3
    assert records[0].status == "success"
    assert records[0].withdrawn == day0_w1

    assert records[1].status == "failed"
    assert records[1].reason == "atm_daily_limit"
    assert records[1].domain == "sender_bank"
    assert records[1].withdrawn == 0

    assert records[2].status == "success"
    assert records[2].withdrawn == day1_w1

    assert output.result.money_withdrawn_paise == day0_w1 + day1_w1


def test_funds_on_hold_blocks_cash_withdrawal() -> None:
    """
    Tests that an encumbered account cannot withdraw cash when unencumbered funds are insufficient,
    failing with reason 'funds_on_hold' and domain 'sender_bank'.
    """
    account = "acc_person_000000"
    fresh_world = prepare_world(TINY, None).world
    ledger = next(item for item in fresh_world.ledgers if account in item.accounts)
    balance = ledger.balance_of(account)

    # Place lien covering almost entire balance
    lien_amount = balance - 5000  # only 50 INR free
    requested = 10000             # 100 INR requested (> 50 INR free)

    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=requested),),
        outside_events=(
            OutsideEvent(minute=50, kind="lien", account_id=account, amount_paise=lien_amount),
        ),
        withdrawals=(
            CashWithdrawal(
                minute=100,
                account_id=account,
                amount_paise=requested,
                channel=CHANNEL_BRANCH,
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
    )

    assert len(output.result.cash_withdrawals) == 1
    rec = output.result.cash_withdrawals[0]
    assert rec.status == "failed"
    assert rec.reason == "funds_on_hold"
    assert rec.domain == "sender_bank"
    assert rec.withdrawn == 0
    assert output.result.money_withdrawn_paise == 0


def test_withdrawal_capped_and_skipped_to_illicit_balance() -> None:
    """
    Tests that a withdrawal is capped to available illicit balance (recorded in capped_forwards),
    and when zero illicit balance remains, subsequent attempts are skipped (recorded in skipped_forwards).
    """
    account = "acc_person_000000"
    seeded_illicit = 300000  # 3,000 INR

    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=seeded_illicit),),
        withdrawals=(
            CashWithdrawal(
                minute=100,
                account_id=account,
                amount_paise=500000,  # 5,000 INR requested, only 3,000 available
                channel=CHANNEL_BRANCH,
            ),
            CashWithdrawal(
                minute=200,
                account_id=account,
                amount_paise=200000,  # 2,000 INR requested, 0 available
                channel=CHANNEL_BRANCH,
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
    )

    recs = output.result.cash_withdrawals
    assert len(recs) == 2

    # First withdrawal capped to 3,000 INR
    assert recs[0].status == "success"
    assert recs[0].requested == 500000
    assert recs[0].withdrawn == 300000
    assert len(output.result.capped_forwards) == 1
    assert output.result.capped_forwards[0].reason == "only_part_arrived"
    assert output.result.capped_forwards[0].available_paise == 300000

    # Second withdrawal skipped due to 0 illicit balance
    assert recs[1].status == "skipped"
    assert recs[1].requested == 200000
    assert recs[1].withdrawn == 0
    assert recs[1].reason == "no_illicit_money_arrived"
    assert len(output.result.skipped_forwards) == 1
    assert output.result.skipped_forwards[0].reason == "no_illicit_money_arrived"

    assert output.result.money_withdrawn_paise == 300000


def test_withdrawal_above_upi_limit_splits_into_legal_legs() -> None:
    """
    Tests that a cash withdrawal larger than the bank's UPI per-payment limit
    is debited across multiple legal legs sharing the same withdrawal id.
    """
    account = "acc_test_high_balance"
    fresh_world = prepare_world(TINY, None).world
    bank_idx = 0
    ledger = fresh_world.ledgers[bank_idx]
    upi_limit = ledger.upi_limit_paise

    large_amount = upi_limit + 500000  # exceeds upi_limit by 5,000 INR
    opening_balance = upi_limit + 1000000
    expected_legs = split_into_legal_payments(large_amount, upi_limit)
    assert len(expected_legs) == 2

    def state_factory(prepared: Any) -> Any:
        state = start_runner_state(prepared.world)
        state.register_account(account, bank_idx, opening_balance_paise=opening_balance, minute=0)
        return state

    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=large_amount),),
        withdrawals=(
            CashWithdrawal(
                minute=100,
                account_id=account,
                amount_paise=large_amount,
                channel=CHANNEL_BRANCH,
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
        state_factory=state_factory,
    )

    rec = output.result.cash_withdrawals[0]
    assert rec.status == "success"
    assert rec.withdrawn == large_amount

    # Verify debit entries on ledger
    runner_ledger = output.world.ledgers[bank_idx]
    debit_entries = [
        entry for entry in runner_ledger.entries
        if entry.account_id == account and entry.transfer_id == rec.id and entry.kind == "transfer_debit"
    ]
    assert len(debit_entries) == len(expected_legs)
    assert [abs(e.amount_paise) for e in debit_entries] == expected_legs


def test_cash_withdrawal_inside_bank_outage_fails() -> None:
    """
    Tests that a cash withdrawal during a bank outage fails with reason 'bank_downtime'
    via failure evaluation on child 13 (cash_withdrawal_failures).
    """
    account = "acc_person_000000"
    prepared = prepare_world(TINY, None)
    bank_idx = next(i for i, item in enumerate(prepared.world.ledgers) if account in item.accounts)

    # Inject an outage incident covering minute 100-200 for this bank
    outage_incident = Incident(
        kind="bank_downtime",
        start_minute=100,
        duration_minutes=100,
        bank_index=bank_idx,
        success_rate=0.0,
    )
    plan_with_incident = dataclasses.replace(prepared.plan, incidents=[outage_incident])
    prepared_modified = prepare_world(TINY, None)

    # Run directly using loop with modified incident plan
    from world.interactive_loop import run_fraud_on
    from world.run_identity import make_run_id

    runner_config = load_runner_config()
    run_id = make_run_id(TINY.seed, {"range_name": TINY.range_name}, "test", runner_config["run_id_hex_length"])
    state = start_runner_state(prepared_modified.world)

    withdrawal = CashWithdrawal(
        minute=150, account_id=account, amount_paise=100000, channel=CHANNEL_ATM, card_daily_limit_paise=200000
    )
    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=100000),),
        withdrawals=(withdrawal,),
    )

    result = run_fraud_on(
        prepared_modified.world,
        plan_with_incident,
        prepared_modified.run_rng,
        prepared_modified.device_events,
        prepared_modified.episode_payments,
        fraud_inputs=inputs,
        hook=lambda _: "allow",
        run_id=run_id,
        runner_config=runner_config,
        state=state,
    )

    assert len(result.cash_withdrawals) == 1
    rec = result.cash_withdrawals[0]
    assert rec.status == "failed"
    assert rec.reason == "bank_downtime"
    assert rec.domain == "sender_bank"
    assert rec.withdrawn == 0


def test_cash_withdrawals_never_in_events_or_truth() -> None:
    """
    Verifies isolation: cash withdrawals never generate PaymentEvents in run_result.events
    and are never included in bank-facing truth facts (roles, payment stories, device control).
    """
    account = "acc_person_000000"
    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=100000),),
        withdrawals=(
            CashWithdrawal(
                minute=100, account_id=account, amount_paise=100000, channel=CHANNEL_ATM, card_daily_limit_paise=200000
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
    )

    # Check that withdrawal id never appears in payment_ids
    w_id = output.result.cash_withdrawals[0].id
    assert w_id not in output.result.payment_ids
    assert not any(e.from_account == account and e.minute == 100 for e in output.result.run_result.events)

    # Truth records must not reference the withdrawal
    assert not any(role.account_id == w_id for role in output.result.truth.roles)
    assert not any(span.account_id == w_id for span in output.result.truth.device_control)


def test_empty_withdrawals_preserves_exact_behavior() -> None:
    """
    Checks that a run with empty withdrawals produces identical output to default fraud_inputs.
    """
    baseline_output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=FraudInputs(),
    )

    explicit_empty_output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=FraudInputs(withdrawals=()),
    )

    assert baseline_output.result.money_withdrawn_paise == 0
    assert explicit_empty_output.result.money_withdrawn_paise == 0
    assert baseline_output.result.cash_withdrawals == []
    assert explicit_empty_output.result.cash_withdrawals == []
    assert len(baseline_output.result.run_result.events) == len(explicit_empty_output.result.run_result.events)


def test_insufficient_ledger_balance_fails_cleanly_and_leaves_balance_untouched() -> None:
    """
    F1 reproduction: verifies that a withdrawal whose effective illicit amount exceeds the real ledger
    balance fails cleanly with status 'failed', reason 'insufficient_balance', domain 'customer',
    and leaves the account's ledger balance and entries completely untouched.
    """
    account = "acc_person_009999"
    bank_idx = 0
    opening_balance = 5_000_000
    illicit_amount = 9_000_000

    def state_factory(prepared: Any) -> Any:
        state = start_runner_state(prepared.world)
        state.register_account(account, bank_idx, opening_balance_paise=opening_balance, minute=0)
        return state

    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=illicit_amount),),
        withdrawals=(
            CashWithdrawal(
                minute=100,
                account_id=account,
                amount_paise=illicit_amount,
                channel=CHANNEL_BRANCH,
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
        state_factory=state_factory,
    )

    assert len(output.result.cash_withdrawals) == 1
    rec = output.result.cash_withdrawals[0]
    assert rec.status == "failed"
    assert rec.reason == "insufficient_balance"
    assert rec.domain == "customer"
    assert rec.requested == illicit_amount
    assert rec.withdrawn == 0
    assert output.result.money_withdrawn_paise == 0

    # Real ledger balance must be unchanged
    ledger = output.world.ledgers[bank_idx]
    assert ledger.balance_of(account) == opening_balance
    account_debits = [e for e in ledger.entries if e.account_id == account and e.kind == "transfer_debit"]
    assert account_debits == []


def test_two_leg_withdrawal_with_short_balance_posts_no_legs() -> None:
    """
    Verifies that a multi-leg withdrawal (amount exceeding upi_limit_paise) where the account has
    sufficient balance for the first leg but insufficient for the total effective amount fails cleanly
    and posts zero debit legs to the ledger (all-or-nothing).
    """
    fresh_world = prepare_world(TINY, None).world
    bank_idx = 0
    ledger = fresh_world.ledgers[bank_idx]
    upi_limit = ledger.upi_limit_paise

    account = "acc_person_009998"
    # Total withdrawal needs 2 legs
    withdrawal_amount = upi_limit + 2_000_000
    # Opening balance is enough for the first leg (upi_limit), but short of total withdrawal_amount
    opening_balance = upi_limit + 500_000

    def state_factory(prepared: Any) -> Any:
        state = start_runner_state(prepared.world)
        state.register_account(account, bank_idx, opening_balance_paise=opening_balance, minute=0)
        return state

    inputs = FraudInputs(
        illicit_seeds=(IllicitSeed(account_id=account, amount_paise=withdrawal_amount),),
        withdrawals=(
            CashWithdrawal(
                minute=100,
                account_id=account,
                amount_paise=withdrawal_amount,
                channel=CHANNEL_BRANCH,
            ),
        ),
    )

    output = run_interactive_world(
        fast_settings=TINY,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
        state_factory=state_factory,
    )

    rec = output.result.cash_withdrawals[0]
    assert rec.status == "failed"
    assert rec.reason == "insufficient_balance"
    assert rec.domain == "customer"
    assert rec.withdrawn == 0

    # Ensure no leg was posted to the ledger
    runner_ledger = output.world.ledgers[bank_idx]
    assert runner_ledger.balance_of(account) == opening_balance
    debit_entries = [e for e in runner_ledger.entries if e.account_id == account and e.kind == "transfer_debit"]
    assert len(debit_entries) == 0


def test_withdrawal_channel_card_limit_validation() -> None:
    """
    Verifies channel card limit constraints:
    - channel 'atm' requires card_daily_limit_paise > 0 (None or non-positive refused).
    - channel 'branch' requires card_daily_limit_paise is None (any limit refused).
    """
    # ATM without limit is refused
    with pytest.raises(ValueError, match="channel 'atm' requires card_daily_limit_paise"):
        validate_cash_withdrawal(
            CashWithdrawal(
                minute=100,
                account_id="acc_person_000000",
                amount_paise=10000,
                channel="atm",
                card_daily_limit_paise=None,
            )
        )

    # Branch with limit is refused
    with pytest.raises(ValueError, match="channel 'branch' requires card_daily_limit_paise to be None"):
        validate_cash_withdrawal(
            CashWithdrawal(
                minute=100,
                account_id="acc_person_000000",
                amount_paise=10000,
                channel="branch",
                card_daily_limit_paise=50000,
            )
        )


def test_stream_position_after_short_balance_equals_successful_withdrawal() -> None:
    """
    Verifies that the failure draw (child 13: cash_withdrawal_failures) is evaluated before the
    balance check so the stream position after a short-balance withdrawal equals the position
    after a successful one (same draws taken).
    """
    from types import SimpleNamespace

    from world.encumbrance import EncumbranceBook

    account = "acc_person_000000"
    withdrawal_amount = 50000

    # Build two generators initialized from identical seed
    seed = 42
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    prep1 = prepare_world(TINY, None)
    prep2 = prepare_world(TINY, None)

    state1 = start_runner_state(prep1.world)
    state2 = start_runner_state(prep2.world)

    bank_idx = state1.bank_map[account]

    # In context 1, account has ample balance
    prep1.world.ledgers[bank_idx].post_credit(account, withdrawal_amount * 2, 0, "test-init-1")

    # In context 2, account balance is 0 (short)
    curr_bal = prep2.world.ledgers[bank_idx].balance_of(account)
    if curr_bal > 0:
        prep2.world.ledgers[bank_idx].post_debit(account, curr_bal, 0, "test-clear-2")

    book1 = EncumbranceBook()
    book2 = EncumbranceBook()

    # The real LoopContext has this field; the fake context must mirror it.
    ctx1 = SimpleNamespace(
        run_id="run-1",
        withdrawal_sequence=0,
        illicit_balances={account: withdrawal_amount},
        skipped_forwards=[],
        capped_forwards=[],
        cash_withdrawals=[],
        daily_atm_withdrawn={},
        state=state1,
        world=prep1.world,
        book=book1,
        streams={"cash_withdrawal_failures": rng1},
        plan=prep1.plan,
        money_withdrawn_paise=0,
        gang_reactions_enabled=False,
    )

    ctx2 = SimpleNamespace(
        run_id="run-2",
        withdrawal_sequence=0,
        illicit_balances={account: withdrawal_amount},
        skipped_forwards=[],
        capped_forwards=[],
        cash_withdrawals=[],
        daily_atm_withdrawn={},
        state=state2,
        world=prep2.world,
        book=book2,
        streams={"cash_withdrawal_failures": rng2},
        plan=prep2.plan,
        money_withdrawn_paise=0,
        gang_reactions_enabled=False,
    )

    w = CashWithdrawal(minute=100, account_id=account, amount_paise=withdrawal_amount, channel=CHANNEL_BRANCH)

    process_cash_withdrawal(ctx1, w)
    process_cash_withdrawal(ctx2, w)

    # ctx1 succeeded, ctx2 failed due to insufficient balance
    assert ctx1.cash_withdrawals[0].status == "success"
    assert ctx2.cash_withdrawals[0].status == "failed"
    assert ctx2.cash_withdrawals[0].reason == "insufficient_balance"

    # Both generators took the exact same failure draw(s) from child 13
    assert rng1.bit_generator.state == rng2.bit_generator.state
