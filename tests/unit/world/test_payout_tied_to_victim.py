"""
WHAT: Unit tests for tying scam payouts to their specific victim's illicit credit (Level 12c C-3b, D4).
Verifies that payouts only spend money credited by their designated victim, failing payouts when the victim
paid nothing even if other victims' money is present, capping payouts when only partial funds arrived, refusing
requires_credit_from on non-spending rules, and maintaining backwards compatibility for standard rules.

WHY: In investment or task scam scenarios, a payout back to a victim is a psychological trick funded by that
victim's own deposits. A victim whose payment failed must never receive a payout subsidized by another victim.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from showcase.level_12_support_b import get_showcase_b_settings
from world.calendar_runner import PlannedPayment
from world.interactive_loop import QueuedPayment, cap_forward_to_illicit_balance
from world.interactive_records import (
    FraudInputs,
    HookDecision,
    IllicitMoneyRule,
    PaymentView,
    RunnerSettings,
)
from world.interactive_runner import run_interactive_world
from world.interactive_state import start_runner_state

START_MINUTE = 1440


def test_refusal_of_requires_credit_from_on_non_spending_rule() -> None:
    """Checks that IllicitMoneyRule and validation refuse requires_credit_from when spends_illicit is False."""
    with pytest.raises(ValueError, match="requires_credit_from is only allowed on a rule with spends_illicit=True"):
        IllicitMoneyRule(input_index=0, spends_illicit=False, credits_illicit=True, requires_credit_from="acc_v1")

    # Verify a valid rule with spends_illicit=True retains requires_credit_from
    valid_rule = IllicitMoneyRule(
        input_index=0, spends_illicit=True, credits_illicit=False, requires_credit_from="acc_v1"
    )
    assert valid_rule.requires_credit_from == "acc_v1"


def test_rules_without_requires_credit_from_behave_as_before() -> None:
    """Checks that rules without requires_credit_from draw from sender illicit balance and report standard reasons."""
    rule = IllicitMoneyRule(input_index=0, spends_illicit=True, credits_illicit=False, requires_credit_from=None)
    context = MagicMock()
    context.money_rules_by_index = {0: rule}
    context.original_fraud_amounts = [50_000]
    context.illicit_balances = {"acc_mule": 0}
    context.skipped_forwards = []
    context.capped_forwards = []

    payment = PlannedPayment(START_MINUTE, "acc_mule", "acc_other", 50_000, 0, "scam")
    queued = QueuedPayment(
        payment=payment, source="fraud", origin="fraud", planned_payment_id="pmt_1", stamp=None, input_index=0
    )

    # When available balance is 0, skips with no_illicit_money_arrived
    result = cap_forward_to_illicit_balance(context, queued)
    assert result is None
    assert len(context.skipped_forwards) == 1
    assert context.skipped_forwards[0].reason == "no_illicit_money_arrived"
    assert context.skipped_forwards[0].available_paise == 0

    # When balance is partial, caps with only_part_arrived
    context.illicit_balances["acc_mule"] = 20_000
    result = cap_forward_to_illicit_balance(context, queued)
    assert result is not None
    assert result.payment.amount_paise == 20_000
    assert len(context.capped_forwards) == 1
    assert context.capped_forwards[0].reason == "only_part_arrived"
    assert context.capped_forwards[0].available_paise == 20_000


def test_victim_whose_payment_failed_gets_no_payout_even_with_other_victim_funds() -> None:
    """
    Checks that when Victim 1's payment fails and Victim 2's payment succeeds into the same mule,
    Victim 1's payout is skipped with victim_paid_nothing despite the mule having sufficient illicit funds.
    """
    context = MagicMock()
    context.illicit_balances = {"acc_mule": 50_000}  # Mule has 50,000 from victim 2
    context.pair_credits = {
        ("acc_mule", "acc_victim_2"): 50_000,
        ("acc_mule", "acc_victim_1"): 0,  # Victim 1 paid nothing
    }
    context.skipped_forwards = []
    context.capped_forwards = []

    rule_payout_v1 = IllicitMoneyRule(
        input_index=0, spends_illicit=True, credits_illicit=False, requires_credit_from="acc_victim_1"
    )
    context.money_rules_by_index = {0: rule_payout_v1}
    context.original_fraud_amounts = [15_000]

    payment_v1 = PlannedPayment(START_MINUTE + 60, "acc_mule", "acc_victim_1", 15_000, 0, "scam")
    queued_v1 = QueuedPayment(
        payment=payment_v1, source="fraud", origin="fraud", planned_payment_id="pmt_payout_1", stamp=None, input_index=0
    )

    # Attempt to process Victim 1's payout
    result_v1 = cap_forward_to_illicit_balance(context, queued_v1)
    assert result_v1 is None
    assert len(context.skipped_forwards) == 1
    adj = context.skipped_forwards[0]
    assert adj.reason == "victim_paid_nothing"
    assert adj.from_account == "acc_mule"
    assert adj.to_account == "acc_victim_1"
    assert adj.planned_paise == 15_000
    assert adj.available_paise == 0


def test_payout_is_capped_to_what_that_victim_paid() -> None:
    """
    Checks that when a victim paid less than the planned payout amount, the payout is capped
    to that victim's specific credit with reason only_part_arrived.
    """
    context = MagicMock()
    context.illicit_balances = {"acc_mule": 100_000}  # Plenty of illicit funds in total
    context.pair_credits = {("acc_mule", "acc_victim_1"): 8_000}  # Only 8,000 arrived from victim 1
    context.skipped_forwards = []
    context.capped_forwards = []

    rule_payout = IllicitMoneyRule(
        input_index=0, spends_illicit=True, credits_illicit=False, requires_credit_from="acc_victim_1"
    )
    context.money_rules_by_index = {0: rule_payout}
    context.original_fraud_amounts = [20_000]

    payment = PlannedPayment(START_MINUTE + 120, "acc_mule", "acc_victim_1", 20_000, 0, "scam")
    queued = QueuedPayment(
        payment=payment, source="fraud", origin="fraud", planned_payment_id="pmt_payout_2", stamp=None, input_index=0
    )

    result = cap_forward_to_illicit_balance(context, queued)
    assert result is not None
    assert result.payment.amount_paise == 8_000
    assert len(context.capped_forwards) == 1
    adj = context.capped_forwards[0]
    assert adj.reason == "only_part_arrived"
    assert adj.planned_paise == 20_000
    assert adj.available_paise == 8_000


def test_failed_victim_payment_skips_payout_in_interactive_world() -> None:
    """
    Runs an interactive world where Victim 1's payment is declined by hook, Victim 2's succeeds,
    and asserts that Victim 1's payout is skipped with victim_paid_nothing while Victim 2's payout succeeds.
    """
    settings, code_version = get_showcase_b_settings()

    def custom_state_factory(prepared: Any) -> Any:
        state = start_runner_state(prepared.world)
        state.register_account("acc_v1", 0, opening_balance_paise=50_000, minute=0)
        state.register_account("acc_v2", 0, opening_balance_paise=50_000, minute=0)
        state.register_account("acc_mule", 0, opening_balance_paise=0, minute=0)
        return state

    p_v1 = PlannedPayment(
        minute=100, from_account="acc_v1", to_account="acc_mule",
        amount_paise=20_000, creation_index=0, purpose="everyday",
    )
    p_v2 = PlannedPayment(
        minute=110, from_account="acc_v2", to_account="acc_mule",
        amount_paise=30_000, creation_index=1, purpose="everyday",
    )
    payout_v1 = PlannedPayment(
        minute=150, from_account="acc_mule", to_account="acc_v1",
        amount_paise=5_000, creation_index=2, purpose="everyday",
    )
    payout_v2 = PlannedPayment(
        minute=160, from_account="acc_mule", to_account="acc_v2",
        amount_paise=5_000, creation_index=3, purpose="everyday",
    )

    rules = (
        IllicitMoneyRule(input_index=0, spends_illicit=False, credits_illicit=True),
        IllicitMoneyRule(input_index=1, spends_illicit=False, credits_illicit=True),
        IllicitMoneyRule(input_index=2, spends_illicit=True, credits_illicit=False, requires_credit_from="acc_v1"),
        IllicitMoneyRule(input_index=3, spends_illicit=True, credits_illicit=False, requires_credit_from="acc_v2"),
    )
    fraud_inputs = FraudInputs(payments=(p_v1, p_v2, payout_v1, payout_v2), money_rules=rules)

    def hook_decline_v1(view: PaymentView) -> HookDecision:
        if view.from_account == "acc_v1":
            return "decline"
        return "allow"

    output = run_interactive_world(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        fraud_inputs=fraud_inputs,
        hook=hook_decline_v1,
        state_factory=custom_state_factory,
        code_version=code_version,
    )

    # v1's payout was skipped with victim_paid_nothing
    skipped = [adj for adj in output.result.skipped_forwards if adj.to_account == "acc_v1"]
    assert len(skipped) == 1
    assert skipped[0].reason == "victim_paid_nothing"

    # v2's payout was made successfully because v2 paid in
    v2_payout_events = [
        e for e, src in zip(output.result.run_result.events, output.result.event_sources)
        if src == "fraud" and e.from_account == "acc_mule" and e.to_account == "acc_v2"
    ]
    assert len(v2_payout_events) == 1
    assert v2_payout_events[0].status == "success"
    assert v2_payout_events[0].amount_paise == 5_000
