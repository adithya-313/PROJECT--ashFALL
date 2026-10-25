"""
Tests for the scam_type and scam_step fields of PaymentStory (world/truth_records.py), their planning in
world/scam_planner.py and their truth rows in world/truth_writer.py.
Covers that scam stories carry the schedule's scam type and the right step, that other gang stories carry neither
field, that an inconsistent pair is refused, and that truth rows carry both keys with None when fraud is off.

TERM: victim payment: a scam payment from a victim to a receiving mule.
TERM: payout back: a small payment from the receiving mule back to the victim.
"""

from typing import Any

import numpy as np
import pytest

from showcase.level_12_support_b import get_showcase_b_settings, load_showcase_b_fixture
from world.build_world import build_world
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.gang_planner import plan_gang
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.scam_planner import TypeChances, _build_schedule_records, plan_victim_schedule
from world.truth_records import SCAM_STEP_PAYOUT_BACK, SCAM_STEP_VICTIM_PAYMENT, PaymentStory
from world.truth_writer import build_truth_rows

START_MINUTE = 1440
SPLIT = "training"
BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
INVESTMENT_CHANCES = TypeChances(investment_or_job=1.0, investment_within=1.0)
JOB_CHANCES = TypeChances(investment_or_job=1.0, investment_within=0.0)
DEBIT_CHANCES = TypeChances(investment_or_job=0.0, investment_within=0.5)
STORY_KEYS = ("scam_type", "scam_step")


def make_scam_gang() -> GangConfig:
    """Builds a small scam gang: acquire 4 mules, scam through 2 receiving mules, no route."""
    steps = (BlockStep("acquire", (("count", 4),)), BlockStep("scam", (("receiving_mules", 2),)))
    return GangConfig("g1", 1, BUDGET, None, steps, START_MINUTE)


def run_world(gang: GangConfig, fraud_enabled: bool) -> Any:
    """Runs the seed-101 showcase world; the runner refuses gangs when fraud is off, so then none is passed."""
    settings, code_version = get_showcase_b_settings()
    gangs = (gang,) if fraud_enabled else ()
    return run_interactive_world(
        fast_settings=settings, runner_settings=RunnerSettings(fraud_enabled=fraud_enabled),
        gang_configs=gangs, code_version=code_version,
    )


# Each chance set draws one scam type, so the three cases cover investment, job and the remainder type.
@pytest.mark.parametrize("chances", [INVESTMENT_CHANCES, JOB_CHANCES, DEBIT_CHANCES])
def test_scam_stories_carry_the_schedule_type_and_payout_back_exactly_on_mule_to_victim(chances: TypeChances) -> None:
    """Checks every story has the schedule's scam type, and payout_back sits exactly on mule-to-victim payments."""
    saw_payout = False
    for seed in range(20):
        rng = np.random.default_rng(seed)
        schedule = plan_victim_schedule("victim", "mule", chances, START_MINUTE, SPLIT, rng)
        payments, stories, _rules = _build_schedule_records("g1", schedule, "scam")
        assert len(stories) == len(schedule.payments)
        for payment, story in zip(payments, stories):
            assert story.scam_type == schedule.scam_type
            goes_mule_to_victim = payment.from_account == "mule" and payment.to_account == "victim"
            expected_step = SCAM_STEP_PAYOUT_BACK if goes_mule_to_victim else SCAM_STEP_VICTIM_PAYMENT
            assert story.scam_step == expected_step
            saw_payout = saw_payout or goes_mule_to_victim
    # Investment and job schedules draw a payout in some seeds; without one the payout branch would be untested.
    if chances is not DEBIT_CHANCES:
        assert saw_payout


def test_non_scam_gang_stories_have_no_scam_fields() -> None:
    """Checks the gang_two_layers fixture, which has no scam step, leaves both fields None on every story."""
    gang, _seed = load_showcase_b_fixture()
    world = build_world(seed=101, n_people=60, n_shops=12)

    plan = plan_gang(gang, world, np.random.default_rng(3))

    assert len(plan.truth.payment_stories) > 0
    for story in plan.truth.payment_stories:
        assert story.scam_type is None
        assert story.scam_step is None


def test_scam_step_without_scam_type_or_with_unknown_value_is_refused() -> None:
    """Checks PaymentStory raises ValueError for a step with no type and for a step outside the two constants."""
    with pytest.raises(ValueError, match="scam_type is None"):
        PaymentStory("g1", "scam", None, "text", scam_type=None, scam_step=SCAM_STEP_VICTIM_PAYMENT)
    with pytest.raises(ValueError, match="must be one of"):
        PaymentStory("g1", "scam", None, "text", scam_type="investment", scam_step="refund")


def test_truth_rows_carry_both_keys_and_fraud_off_rows_carry_none() -> None:
    """Checks fraud-on rows hold the scam fields of scam payments and fraud-off rows hold None for both keys."""
    gang = make_scam_gang()

    fraud_on_rows = build_truth_rows(run_world(gang, fraud_enabled=True))["payment_stories"]
    fraud_off_rows = build_truth_rows(run_world(gang, fraud_enabled=False))["payment_stories"]

    assert all(key in row for row in fraud_on_rows for key in STORY_KEYS)
    scam_rows = [row for row in fraud_on_rows if row["block"] == "scam"]
    assert len(scam_rows) > 0
    assert all(row["scam_type"] is not None and row["scam_step"] is not None for row in scam_rows)
    non_scam_rows = [row for row in fraud_on_rows if row["block"] != "scam"]
    assert all(row["scam_type"] is None and row["scam_step"] is None for row in non_scam_rows)
    assert len(fraud_off_rows) > 0
    assert all(key in row for row in fraud_off_rows for key in STORY_KEYS)
    assert all(row[key] is None for row in fraud_off_rows for key in STORY_KEYS)
