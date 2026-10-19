"""
Illicit money guard verification ensuring money conservation and valid forwarding rules.
Verifies that mule accounts forward only illicit funds actually received, preventing personal
balance drainage during routing hop failures, and tests rule validation and audit logs.
"""

import pytest

from showcase.level_12_support_b import get_showcase_b_settings, load_showcase_b_fixture
from world.calendar_runner import PlannedPayment
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.fast_mode import FastRunSettings
from world.interactive_records import (
    FraudInputs,
    HookDecision,
    IllicitMoneyRule,
    IllicitSeed,
    PaymentView,
    RunnerSettings,
)
from world.interactive_runner import prepare_world, run_interactive_world
from world.interactive_state import start_runner_state
from world.outside_events import OutsideEvent


def test_fact_2_regression_forward_only_what_arrived() -> None:
    """
    Verifies that layer-1 mule accounts skip forwarding when initial collector payments are declined.
    Tests regression of Fact 2 where mules previously drained authentic funds, asserting two skipped
    forwards for 676,300 and 4,731,000 paise with total ledger money conserved.
    """
    gang, _ = load_showcase_b_fixture()
    settings, code_version = get_showcase_b_settings()

    def decline_collector_hook(view: PaymentView) -> HookDecision:
        """Declines payments from the gang collector account while allowing all other transfers."""
        if view.from_account == "acc_collector_01":
            return "decline"
        return "allow"

    rules = (
        IllicitMoneyRule(input_index=0, spends_illicit=True, credits_illicit=True),
        IllicitMoneyRule(input_index=1, spends_illicit=True, credits_illicit=True),
        IllicitMoneyRule(input_index=2, spends_illicit=True, credits_illicit=True),
        IllicitMoneyRule(input_index=3, spends_illicit=True, credits_illicit=True),
    )
    seeds = (
        IllicitSeed(account_id=gang.source.account_id, amount_paise=gang.source.opening_balance_paise),
    )

    inputs = FraudInputs(money_rules=rules, illicit_seeds=seeds)
    output = run_interactive_world(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        fraud_inputs=inputs,
        hook=decline_collector_hook,
        code_version=code_version,
    )

    layer_1_mules = {"acc_person_000006", "acc_person_000010"}
    mule_successes = [
        event for event, source in zip(output.result.run_result.events, output.result.event_sources)
        if source == "fraud" and event.from_account in layer_1_mules and event.status == "success"
    ]
    assert len(mule_successes) == 0

    assert len(output.result.skipped_forwards) == 2
    assert [skipped.planned_paise for skipped in output.result.skipped_forwards] == [676300, 4731000]
    assert [skipped.available_paise for skipped in output.result.skipped_forwards] == [0, 0]
    assert [skipped.from_account for skipped in output.result.skipped_forwards] == [
        "acc_person_000006",
        "acc_person_000010",
    ]
    assert all(skipped.reason == "no_illicit_money_arrived" for skipped in output.result.skipped_forwards)

    ledgers = output.world.ledgers
    fresh_world = prepare_world(settings, None).world
    opening_sum = sum(ledger.balance_of(account) for ledger in fresh_world.ledgers for account in ledger.accounts)
    # Conservation holds against initial balances plus newly registered source balance.
    assert check_no_negative_balances(ledgers) == []
    assert check_balances_equal_entry_sums(ledgers) == []
    assert check_money_conserved(ledgers, opening_sum + output.result.money_created_paise) == []


def test_partial_arrival_caps_forward_amount() -> None:
    """
    Verifies that a payment receipt split across UPI limit pieces where one piece fails caps the forward.
    Ensures the forward payment amount is reduced to match the successfully credited piece amount
    and logs the adjustment in the capped_forwards audit collection.
    """
    settings = FastRunSettings(101, 60, 12, 61, "training")
    upi_limit = 6122622

    def custom_state_factory(prepared):
        """Initializes test runner state with source, mule, and destination accounts."""
        state = start_runner_state(prepared.world)
        # acc_source has 7,000,000 paise (sufficient for one piece of 6,122,622, but not two).
        state.register_account("acc_source", 0, opening_balance_paise=7000000, minute=0)
        state.register_account("acc_mule", 0, opening_balance_paise=0, minute=0)
        state.register_account("acc_sink", 0, opening_balance_paise=0, minute=0)
        return state

    pmt0 = PlannedPayment(
        minute=100, from_account="acc_source", to_account="acc_mule",
        amount_paise=2 * upi_limit, creation_index=0, purpose="everyday",
    )
    pmt1 = PlannedPayment(
        minute=120, from_account="acc_mule", to_account="acc_sink",
        amount_paise=2 * upi_limit, creation_index=1, purpose="everyday",
    )
    rules = (
        IllicitMoneyRule(input_index=0, spends_illicit=False, credits_illicit=True),
        IllicitMoneyRule(input_index=1, spends_illicit=True, credits_illicit=True),
    )
    inputs = FraudInputs(payments=(pmt0, pmt1), money_rules=rules)

    output = run_interactive_world(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=inputs,
        state_factory=custom_state_factory,
    )

    assert len(output.result.skipped_forwards) == 0
    assert len(output.result.capped_forwards) == 1
    capped = output.result.capped_forwards[0]
    assert capped.from_account == "acc_mule"
    assert capped.to_account == "acc_sink"
    assert capped.planned_paise == 2 * upi_limit
    assert capped.available_paise == upi_limit
    assert capped.reason == "only_part_arrived"

    fraud_events = [
        event for event, source in zip(output.result.run_result.events, output.result.event_sources)
        if source == "fraud"
    ]
    # First payment: piece 1 succeeds with 6,122,622; piece 2 fails due to insufficient balance.
    assert fraud_events[0].amount_paise == upi_limit and fraud_events[0].status == "success"
    assert fraud_events[1].amount_paise == upi_limit and fraud_events[1].status == "failed"
    # Second payment: executed with capped amount of 6,122,622 and succeeds.
    assert fraud_events[2].amount_paise == upi_limit and fraud_events[2].status == "success"


def test_forwarded_less_than_or_equal_received_invariant() -> None:
    """
    Verifies that total forwarded money never exceeds total received money for any mule account.
    Computes total inward and outward sums directly from executed transaction events across
    governed mule entities to guarantee the core invariant holds over full simulation runs.
    """
    gang, _ = load_showcase_b_fixture()
    settings, code_version = get_showcase_b_settings()

    rules = (
        IllicitMoneyRule(input_index=0, spends_illicit=True, credits_illicit=True),
        IllicitMoneyRule(input_index=1, spends_illicit=True, credits_illicit=True),
        IllicitMoneyRule(input_index=2, spends_illicit=True, credits_illicit=True),
        IllicitMoneyRule(input_index=3, spends_illicit=True, credits_illicit=True),
    )
    seeds = (
        IllicitSeed(account_id=gang.source.account_id, amount_paise=gang.source.opening_balance_paise),
    )

    inputs = FraudInputs(money_rules=rules, illicit_seeds=seeds)
    output = run_interactive_world(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        fraud_inputs=inputs,
        code_version=code_version,
    )

    mule_accounts = {"acc_person_000006", "acc_person_000010"}
    for mule in mule_accounts:
        total_received = sum(
            event.amount_paise
            for event, source in zip(output.result.run_result.events, output.result.event_sources)
            if source == "fraud" and event.to_account == mule and event.status == "success"
        )
        total_forwarded = sum(
            event.amount_paise
            for event, source in zip(output.result.run_result.events, output.result.event_sources)
            if source == "fraud" and event.from_account == mule and event.status == "success"
        )
        assert total_forwarded <= total_received


def test_hold_then_resume_rechecks_available_illicit_funds() -> None:
    """
    Verifies that a payment held by the hook re-evaluates available funds when resumed.
    Confirms that if additional funds arrive while on hold, the resumed attempt re-checks
    from original planned amount and allows the full transfer to proceed successfully.
    """
    settings = FastRunSettings(101, 60, 12, 61, "training")

    def custom_state_factory(prepared):
        """Initializes test runner state with two source accounts, a mule, and a destination account."""
        state = start_runner_state(prepared.world)
        state.register_account("acc_src1", 0, opening_balance_paise=5000, minute=0)
        state.register_account("acc_src2", 0, opening_balance_paise=5000, minute=0)
        state.register_account("acc_mule", 0, opening_balance_paise=0, minute=0)
        state.register_account("acc_sink", 0, opening_balance_paise=0, minute=0)
        return state

    pmt0 = PlannedPayment(
        minute=50, from_account="acc_src1", to_account="acc_mule",
        amount_paise=5000, creation_index=0, purpose="everyday",
    )
    pmt1 = PlannedPayment(
        minute=100, from_account="acc_mule", to_account="acc_sink",
        amount_paise=10000, creation_index=1, purpose="everyday",
    )
    pmt2 = PlannedPayment(
        minute=105, from_account="acc_src2", to_account="acc_mule",
        amount_paise=5000, creation_index=2, purpose="everyday",
    )
    rules = (
        IllicitMoneyRule(input_index=0, spends_illicit=False, credits_illicit=True),
        IllicitMoneyRule(input_index=1, spends_illicit=True, credits_illicit=True),
        IllicitMoneyRule(input_index=2, spends_illicit=False, credits_illicit=True),
    )

    def hold_mule_hook(view: PaymentView) -> HookDecision:
        """Holds outward transfers attempted from the intermediate mule account."""
        if view.from_account == "acc_mule":
            return "hold"
        return "allow"

    initial_inputs = FraudInputs(payments=(pmt0, pmt1, pmt2), money_rules=rules)
    initial_output = run_interactive_world(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=initial_inputs,
        hook=hold_mule_hook,
        state_factory=custom_state_factory,
    )

    held_id = initial_output.result.held_payments[0].planned_payment_id

    def resume_hook(view: PaymentView) -> HookDecision:
        """Holds mule transfers at minute 100 while allowing subsequent resumed attempts."""
        if view.from_account == "acc_mule" and view.minute == 100:
            return "hold"
        return "allow"

    resume_event = OutsideEvent(minute=110, kind="resume_payment", planned_payment_id=held_id)
    resumed_inputs = FraudInputs(payments=(pmt0, pmt1, pmt2), outside_events=(resume_event,), money_rules=rules)
    resumed_output = run_interactive_world(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=resumed_inputs,
        hook=resume_hook,
        state_factory=custom_state_factory,
    )

    assert len(resumed_output.result.held_payments) == 0
    resumed_events = [
        event for event, source in zip(resumed_output.result.run_result.events, resumed_output.result.event_sources)
        if event.from_account == "acc_mule" and event.status == "success"
    ]
    assert len(resumed_events) == 1
    assert resumed_events[0].amount_paise == 10000
    assert resumed_events[0].minute == 110


def test_no_rules_preserves_baseline_behavior() -> None:
    """
    Verifies that hand-made FraudInputs with no rules behave identically to existing baseline.
    Asserts skipped_forwards and capped_forwards remain empty and payments without rules proceed.
    """
    gang, _ = load_showcase_b_fixture()
    settings, code_version = get_showcase_b_settings()

    def decline_collector_hook(view: PaymentView) -> HookDecision:
        """Declines payments from the gang collector account while allowing all other transfers."""
        if view.from_account == "acc_collector_01":
            return "decline"
        return "allow"

    # No money rules or seeds supplied.
    output = run_interactive_world(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        fraud_inputs=FraudInputs(),
        hook=decline_collector_hook,
        code_version=code_version,
    )

    assert output.result.skipped_forwards == []
    assert output.result.capped_forwards == []

    # Without rules, mule 000006 forwards 676,300 using authentic funds as in Fact 2.
    mule_successes = [
        event for event, source in zip(output.result.run_result.events, output.result.event_sources)
        if source == "fraud" and event.from_account == "acc_person_000006" and event.status == "success"
    ]
    assert len(mule_successes) == 1
    assert mule_successes[0].amount_paise == 676300


def test_illicit_money_rule_validation_refusals() -> None:
    """
    Verifies input validation refusals for invalid illicit money rules and seeds.
    Ensures clear ValueErrors are raised for out-of-range indices, duplicate rules,
    unregistered seed accounts, and non-positive seed amounts.
    """
    settings = FastRunSettings(101, 60, 12, 61, "training")
    payment = PlannedPayment(
        minute=100, from_account="acc_person_000001", to_account="acc_person_000002",
        amount_paise=5000, creation_index=0, purpose="everyday",
    )

    # 1. Out-of-range rule index.
    bad_rule_inputs = FraudInputs(
        payments=(payment,),
        money_rules=(IllicitMoneyRule(input_index=5, spends_illicit=True, credits_illicit=True),),
    )
    with pytest.raises(ValueError, match="out of range"):
        run_interactive_world(
            fast_settings=settings,
            runner_settings=RunnerSettings(fraud_enabled=True),
            fraud_inputs=bad_rule_inputs,
        )

    # 2. Duplicate rule index.
    dup_rule_inputs = FraudInputs(
        payments=(payment,),
        money_rules=(
            IllicitMoneyRule(input_index=0, spends_illicit=True, credits_illicit=True),
            IllicitMoneyRule(input_index=0, spends_illicit=False, credits_illicit=True),
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        run_interactive_world(
            fast_settings=settings,
            runner_settings=RunnerSettings(fraud_enabled=True),
            fraud_inputs=dup_rule_inputs,
        )

    # 3. Seed on unregistered account.
    unregistered_seed_inputs = FraudInputs(
        payments=(payment,),
        illicit_seeds=(IllicitSeed(account_id="acc_unregistered_ghost", amount_paise=10000),),
    )
    with pytest.raises(ValueError, match="not registered"):
        run_interactive_world(
            fast_settings=settings,
            runner_settings=RunnerSettings(fraud_enabled=True),
            fraud_inputs=unregistered_seed_inputs,
        )

    # 4. Non-positive seed amount.
    zero_seed_inputs = FraudInputs(
        payments=(payment,),
        illicit_seeds=(IllicitSeed(account_id="acc_person_000001", amount_paise=0),),
    )
    with pytest.raises(ValueError, match="must be positive"):
        run_interactive_world(
            fast_settings=settings,
            runner_settings=RunnerSettings(fraud_enabled=True),
            fraud_inputs=zero_seed_inputs,
        )
