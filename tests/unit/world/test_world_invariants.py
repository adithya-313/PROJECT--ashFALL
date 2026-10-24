"""
WHAT: Unit tests for the world invariant checks. A cheap fraud-on run with a gang-opened account and a cash withdrawal
is built once; a clean copy must give no problems, and each test breaks exactly one thing in a deep copy and checks
that only the intended check reports it, in a sentence that names the object. One generated small world is also
checked clean. Expected values are written in the tests or computed from the run's own records, never from the check.

WHY: A sweep trusts these checks to notice a broken rule. A check that cannot fail, or that reports under the wrong
name, would let a bad world through while looking sound.

TERM: break: a deliberate change to a copy of a finished run that violates exactly one invariant.
"""

import copy
import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from world.calendar_runner import PlannedPayment
from world.cash_withdrawals import CHANNEL_ATM, CashWithdrawal, CashWithdrawalRecord
from world.fast_mode import FastRunSettings
from world.interactive_records import FraudInputs, IllicitMoneyRule, InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.label_builder import build_label_events
from world.label_records import LabelEvent
from world.outside_events import OutsideEvent
from world.sampled_world import run_sampled_world
from world.truth_records import RoleSpan, TruthFacts
from world.world_configs import read_world_configs
from world.world_invariants import check_atm_daily_limits, check_world, expected_total_paise
from world.world_sampler import load_sampler_config, sample_worlds

TINY = FastRunSettings(101, 60, 12, 61, "training")
HANDLE = "new:g:0"
OPENED = "acc_person_000060"
LIMIT = 500_000


@pytest.fixture(scope="module")
def run() -> InteractiveWorldOutput:
    """Runs the cheapest fraud-on world with a gang-opened account: it opens, gets a payment, and pays out by ATM."""
    opening = OutsideEvent(
        100, "open_account", owner_kind="person", opening_stream="gang_account_openings", opening_handle=HANDLE
    )
    inputs = FraudInputs(
        payments=(PlannedPayment(300, "acc_person_000001", HANDLE, 20_000, 0, "everyday"),),
        money_rules=(IllicitMoneyRule(0, False, True),),
        outside_events=(opening,),
        withdrawals=(CashWithdrawal(5_000, HANDLE, 10_000, CHANNEL_ATM, LIMIT),),
    )
    return run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), fraud_inputs=inputs)


def _fraud_payment(output: InteractiveWorldOutput) -> tuple[str, int]:
    """Returns the id and minute of the one fraud payment of the cheap run."""
    result = output.result
    for payment_id, source, event in zip(result.payment_ids, result.event_sources, result.run_result.events):
        if source == "fraud":
            return payment_id, event.minute
    raise AssertionError("the cheap run has no fraud payment")


def _only(problems: list[str], tag: str) -> None:
    """Asserts that there is at least one problem and that every problem is tagged with the given check."""
    assert problems, f"expected a [{tag}] problem, got none"
    assert all(problem.startswith(f"[{tag}]") for problem in problems), problems


def test_a_clean_run_has_no_problems_and_the_total_matches_the_old_formula(run: InteractiveWorldOutput) -> None:
    """The clean run passes every check, and the formula equals opening total + money created - cash withdrawn."""
    payment_id, minute = _fraud_payment(run)
    label = LabelEvent("L1", "complaint", minute + 10, "acc_person_000001", 0, OPENED, 0, (payment_id,), "investment")
    assert check_world(run, [label], {OPENED: LIMIT}) == []
    fresh_total = sum(
        ledger.balance_of(account) for ledger in prepare_world(TINY, None).world.ledgers for account in ledger.accounts
    )
    assert expected_total_paise(run) == fresh_total + run.result.money_created_paise - run.result.money_withdrawn_paise


def test_a_move_before_the_opening_is_reported_by_the_opening_check_only(run: InteractiveWorldOutput) -> None:
    """Moving the credit into the opened account to minute 50 (it opens at 100) is reported by [opening_time] alone."""
    broken = copy.deepcopy(run)
    moved = 0
    for ledger in broken.world.ledgers:
        for entry in ledger.entries:
            if entry.account_id == OPENED and entry.kind == "transfer_credit" and moved == 0:
                entry.minute = 50
                moved += 1
    assert moved == 1
    problems = check_world(broken)
    _only(problems, "opening_time")
    assert OPENED in problems[0] and "minute 50" in problems[0] and "minute 100" in problems[0]


def test_a_repeated_payment_id_is_reported_by_the_id_check_only(run: InteractiveWorldOutput) -> None:
    """Giving the second event the first event's payment id is reported by [payment_ids] alone, naming the id."""
    broken = copy.deepcopy(run)
    first_id = broken.result.payment_ids[0]
    broken.result.payment_ids[1] = first_id
    problems = check_world(broken)
    _only(problems, "payment_ids")
    assert first_id in problems[0] and "2 times" in problems[0]


def test_an_atm_withdrawal_over_the_limit_is_reported_by_the_limit_check_only(run: InteractiveWorldOutput) -> None:
    """A successful ATM record of limit + 1 paise on one day is reported by [atm_daily_limit] alone."""
    broken = copy.deepcopy(run)
    broken.result.cash_withdrawals.append(
        CashWithdrawalRecord("extra", 5_100, OPENED, CHANNEL_ATM, LIMIT + 1, LIMIT + 1, "success")
    )
    problems = check_world(broken, atm_limits={OPENED: LIMIT})
    _only(problems, "atm_daily_limit")
    assert OPENED in problems[0] and f"above its daily limit of {LIMIT}" in problems[0]


def test_the_atm_check_counts_checked_and_skipped_withdrawals() -> None:
    """Two successful ATM withdrawals on accounts with a limit and one without give checked 2 and skipped 1."""
    records = [
        CashWithdrawalRecord("w1", 100, "acc_a", CHANNEL_ATM, 1_000, 1_000, "success"),
        CashWithdrawalRecord("w2", 200, "acc_b", CHANNEL_ATM, 2_000, 2_000, "success"),
        CashWithdrawalRecord("w3", 300, "acc_c", CHANNEL_ATM, 3_000, 3_000, "success"),
        CashWithdrawalRecord("w4", 400, "acc_c", CHANNEL_ATM, 4_000, 0, "failed"),
    ]
    fake = SimpleNamespace(result=SimpleNamespace(cash_withdrawals=records))
    outcome = check_atm_daily_limits(fake, {"acc_a": 5_000, "acc_b": 5_000})  # type: ignore[arg-type]
    assert (outcome.checked, outcome.skipped, outcome.problems) == (2, 1, [])
    no_limits = check_atm_daily_limits(fake, None)  # type: ignore[arg-type]
    assert (no_limits.checked, no_limits.skipped) == (0, 3)


def test_labels_before_their_payment_or_naming_a_missing_payment_are_reported_by_the_label_check_only(
    run: InteractiveWorldOutput,
) -> None:
    """A complaint arriving one minute before its payment, and one naming an unknown id, are [labels] problems."""
    payment_id, minute = _fraud_payment(run)
    early = LabelEvent("L1", "complaint", minute - 1, "acc_person_000001", 0, OPENED, 0, (payment_id,), "investment")
    ghost = LabelEvent("L2", "complaint", minute + 5, "acc_person_000001", 0, OPENED, 0, ("no-such-payment",), "job")
    problems = check_world(run, [early, ghost])
    _only(problems, "labels")
    assert any("L1" in p and payment_id in p and f"minute {minute - 1}" in p for p in problems)
    assert any("L2" in p and "no-such-payment" in p for p in problems)


def test_a_deleted_ledger_entry_is_reported_by_the_entry_sum_check_only(run: InteractiveWorldOutput) -> None:
    """Deleting one transfer entry keeps the stored balance but not the entry sum: [balances_equal_entry_sums]."""
    broken = copy.deepcopy(run)
    deleted: Any = None
    for ledger in broken.world.ledgers:
        for entry in list(ledger.entries):
            if entry.kind == "transfer_credit" and deleted is None:
                ledger.entries.remove(entry)
                deleted = entry
    assert deleted is not None
    problems = check_world(broken)
    _only(problems, "balances_equal_entry_sums")
    assert deleted.account_id in " ".join(problems)


def test_money_that_leaves_without_a_withdrawal_is_reported_by_the_conservation_check_only(
    run: InteractiveWorldOutput,
) -> None:
    """Claiming 1 paise more was withdrawn than the ledgers show breaks only [money_conserved]."""
    broken = dataclasses.replace(
        run, result=dataclasses.replace(run.result, money_withdrawn_paise=run.result.money_withdrawn_paise + 1)
    )
    problems = check_world(broken)
    _only(problems, "money_conserved")
    assert "expected total" in problems[0]


def test_a_truth_role_on_a_missing_account_is_reported_by_the_truth_check_only(run: InteractiveWorldOutput) -> None:
    """A role naming an account that is in no ledger is reported by [truth_accounts] alone, naming the account."""
    ghost_role = RoleSpan("acc_ghost", "mule", "g", 1, 100, None, "rented")
    truth = TruthFacts(
        roles=run.result.truth.roles + (ghost_role,), device_control=run.result.truth.device_control,
        payment_stories=run.result.truth.payment_stories,
    )
    broken = dataclasses.replace(run, result=dataclasses.replace(run.result, truth=truth))
    problems = check_world(broken)
    _only(problems, "truth_accounts")
    assert "acc_ghost" in problems[0]


def test_one_generated_small_world_has_no_problems() -> None:
    """A generated small training world, with its labels, passes every check that needs no planned limits."""
    cfg = load_sampler_config()
    spec = sample_worlds("training", 1, 12, cfg, read_world_configs().world_ranges)[0]
    output = run_sampled_world(spec, "small", None, cfg, "test-invariants")
    assert check_world(output, build_label_events(output)) == []
