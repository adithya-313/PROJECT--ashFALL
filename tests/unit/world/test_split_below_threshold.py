"""
WHAT: Unit tests for the split_below_threshold tactic building block (Level 12c C-4, issue 12-17).
Verifies:
1. Planned payments above threshold are partitioned into equal pieces below threshold at the same minute,
   each with its own IllicitMoneyRule and PaymentStory ("split piece i of n"), maintaining exact money total.
2. Training worlds refuse splits that produce pieces below the slow_drip dividing line (2,000 INR) with a plain message.
3. Stress worlds permit splits below the slow_drip dividing line.
4. Threshold parameters obey calibrated ranges across training and stress splits and respect narrowed configuration.

WHY: Issue 12-17 and amendment A-016 require structured payment splitting below reporting thresholds while preventing
leakage into the held-out slow_drip family in training worlds.
"""

import numpy as np
import pytest

from core.config_validator import load_yaml
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.gang_planner import plan_gang
from world.interactive_records import FraudInputs, IllicitSeed, RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.run_identity import REPO_ROOT
from world.split_below_threshold import draw_threshold_paise, get_threshold_inr_range

FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])


def test_pieces_below_threshold_same_total_same_minute_own_rules_and_stories() -> None:
    """
    WHAT: Tests that a route payment above threshold is replaced by equal pieces below threshold at the same minute,
    each with its own IllicitMoneyRule and PaymentStory ("split piece i of n"), conserving exact total paise.

    WHY: Issue 12-17 models structuring payments below detection thresholds without disturbing money conservation.
    """
    prepared = prepare_world(TINY, None)
    world = prepared.world
    world.parameters_drawn["range_name"] = "training"
    source_acct = world.people[0].account_id
    rng = np.random.default_rng(42)

    config = GangConfig(
        gang_id="gang_split_test",
        strength=1,
        source=FixtureSource(
            account_id=source_acct, bank_index=0, opening_balance_paise=10_000_000, start_minute=100
        ),
        budget=GangBudget(money_paise=10_000_000, time_minutes=10_000, max_risk=1.0),
        steps=(
            BlockStep("acquire", (("count", 2),)),
            BlockStep("split_below_threshold", (("threshold_inr", 25000),)),
            BlockStep(
                "route",
                (("layers", 1), ("accounts_per_layer", 1), ("amount_inr", 60000), ("dwell_minutes", 10)),
            ),
        ),
    )

    plan = plan_gang(config, world, rng)

    # Original amount 60,000 INR (6,000,000 paise) with threshold 25,000 INR (2,500,000 paise)
    # yields n = (6000000 // 2500000) + 1 = 3 pieces of 2,000,000 paise (20,000 INR) each
    assert len(plan.payments) == 3
    assert len(plan.truth.payment_stories) == 3
    assert len(plan.money_rules) == 3

    # All pieces strictly below threshold ceiling
    threshold_paise = 25000 * 100
    for p in plan.payments:
        assert p.amount_paise < threshold_paise
        assert p.amount_paise == 2_000_000

    # Total paise exactly conserved
    assert sum(p.amount_paise for p in plan.payments) == 6_000_000

    # All pieces scheduled at the same minute
    minutes = {p.minute for p in plan.payments}
    assert len(minutes) == 1

    # Individual stories formatted as "split piece i of n"
    expected_stories = ["split piece 1 of 3", "split piece 2 of 3", "split piece 3 of 3"]
    descriptions = [s.description for s in plan.truth.payment_stories]
    assert descriptions == expected_stories

    # Rules align with input_index and inherit spends_illicit/credits_illicit
    for i, rule in enumerate(plan.money_rules):
        assert rule.input_index == i
        assert rule.spends_illicit is True
        assert rule.credits_illicit is True

    # Execute interactive runner to ensure closed-loop ledger execution succeeds
    runner_settings = RunnerSettings(fraud_enabled=True)
    inputs = FraudInputs(
        payments=plan.payments,
        outside_events=plan.outside_events,
        truth=plan.truth,
        gang_device_events=plan.gang_device_events,
        money_rules=plan.money_rules,
        illicit_seeds=(IllicitSeed(account_id=source_acct, amount_paise=6_000_000),),
    )
    output = run_interactive_world(fast_settings=TINY, runner_settings=runner_settings, fraud_inputs=inputs)
    assert output.result.event_sources.count("fraud") == 3


def test_training_refusal_below_slow_drip_line() -> None:
    """
    WHAT: Tests that a training world refuses a split that would result in pieces below the slow_drip dividing
    line (2,000 INR = 200,000 paise) with a plain message.

    WHY: Held-out tactic boundary protection: training tactics cannot generate transfers below 2,000 INR.
    """
    prepared = prepare_world(TINY, None)
    world = prepared.world
    world.parameters_drawn["range_name"] = "training"
    source_acct = world.people[0].account_id
    rng = np.random.default_rng(42)

    # 3,000 INR routed with threshold 2,500 INR -> n = 2 pieces of 1,500 INR
    # 1,500 INR is below 2,000 INR slow_drip dividing line
    config = GangConfig(
        gang_id="gang_slow_drip_training",
        strength=1,
        source=FixtureSource(account_id=source_acct, bank_index=0, opening_balance_paise=10_000_000, start_minute=100),
        budget=GangBudget(money_paise=10_000_000, time_minutes=10_000, max_risk=1.0),
        steps=(
            BlockStep("acquire", (("count", 2),)),
            BlockStep("split_below_threshold", (("threshold_inr", 2500),)),
            BlockStep("route", (("layers", 1), ("accounts_per_layer", 1), ("amount_inr", 3000), ("dwell_minutes", 10))),
        ),
    )

    with pytest.raises(ValueError, match="training slow drip dividing line"):
        plan_gang(config, world, rng)


def test_stress_world_allows_split_below_slow_drip_line() -> None:
    """
    WHAT: Tests that a stress world allows a split that produces pieces below the slow_drip dividing line.

    WHY: Test and stress worlds test held-out evasion families, so splitting below 2,000 INR is valid.
    """
    prepared = prepare_world(TINY, None)
    world = prepared.world
    world.parameters_drawn["range_name"] = "stress"
    source_acct = world.people[0].account_id
    rng = np.random.default_rng(42)

    config = GangConfig(
        gang_id="gang_slow_drip_stress",
        strength=1,
        source=FixtureSource(account_id=source_acct, bank_index=0, opening_balance_paise=10_000_000, start_minute=100),
        budget=GangBudget(money_paise=10_000_000, time_minutes=10_000, max_risk=1.0),
        steps=(
            BlockStep("acquire", (("count", 2),)),
            BlockStep("split_below_threshold", (("threshold_inr", 2500),)),
            BlockStep("route", (("layers", 1), ("accounts_per_layer", 1), ("amount_inr", 3000), ("dwell_minutes", 10))),
        ),
    )

    plan = plan_gang(config, world, rng)
    assert len(plan.payments) == 2
    for p in plan.payments:
        assert p.amount_paise == 150000  # 1,500 INR allowed in stress world
    assert sum(p.amount_paise for p in plan.payments) == 300000


def test_narrowed_threshold_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Tests that split_below_threshold parameters resolve calibrated ranges across training and stress splits,
    and dynamically respect narrowed threshold configuration bounds.

    WHY: Ground rule 15 requires range bounds to be dynamically resolved without hardcoded drift.
    """
    # 1. Calibrated ranges from config
    training_bounds = get_threshold_inr_range("training")
    assert training_bounds == (10000, 50000)

    stress_bounds = get_threshold_inr_range("stress")
    assert stress_bounds == (2000, 100000)

    # 2. Narrowed threshold configuration dynamically respected
    mock_definition = {
        "parameters": {
            "threshold_inr": {
                "training": [15000, 30000],
                "stress": [5000, 80000],
            }
        }
    }
    monkeypatch.setattr(
        "world.split_below_threshold.get_block_definition",
        lambda block_name, config_path=None: mock_definition,
    )

    narrowed_training = get_threshold_inr_range("training")
    assert narrowed_training == (15000, 30000)

    narrowed_stress = get_threshold_inr_range("stress")
    assert narrowed_stress == (5000, 80000)

    rng = np.random.default_rng(123)
    drawn_paise = draw_threshold_paise("training", None, rng)
    assert 1500000 <= drawn_paise <= 3000000
