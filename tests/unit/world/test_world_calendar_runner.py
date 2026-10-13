"""
WHAT: Unit tests C1-C10 verifying determinism, ledger conservation, failure attribution, and calendar rhythms.
WHY: Guarantees that the calendar runner conforms to issue R7 (failed payments are recorded but move no money),
     preserves accounting invariants, and models outages.
TERM: calendar runner: a program running simulated town payments minute by minute,
      e.g. paying salaries on the first and rent on the third.
TERM: payment event: a saved note recording what happened to a payment,
      e.g. whether money moved or a bank declined it.
TERM: failure domain: who caused a payment to fail,
      e.g. the customer running out of money versus a bank server crashing.
TERM: partial degradation: a service running slowly or dropping some requests rather than completely shutting down,
      e.g. a bank failing half of its transfers.
TERM: determinism: getting the exact same outcome every time you start with the same seed,
      e.g. running seed 7 twice produces identical numbers.
TERM: purpose: the real-life reason for a payment,
      e.g. paying monthly rent or buying groceries at a shop.
TERM: golden hash: a fixed fingerprint saved from an earlier known-good run,
      e.g. checking that tomorrow's run gives the exact same fingerprint so we know nothing broke.
"""

import calendar
import dataclasses
import hashlib
from typing import Any

import numpy as np

from world.build_world import build_world
from world.calendar_runner import (
    PlannedPayment,
    RunResult,
    _build_account_bank_map,
    _execute_payment_leg,
    run_world,
)
from world.checks import check_balances_equal_entry_sums, check_money_conserved, check_no_negative_balances
from world.failures import FAILURE_DOMAINS, FailureSettings, PaymentEvent
from world.festivals import Festival
from world.incidents import Incident
from world.payments import find_ledger_for_account, make_payment
from world.placeholders import EVERYDAY_PAYMENTS_PER_MONTH, SALARY_DAY_WINDOW
from world.rhythms import daily_multiplier, hour_weight
from world.routine import run_one_month
from world.world_plan import WorldPlan, draw_world_plan, spawn_world_streams


def _hash_event_list(events: list[PaymentEvent]) -> str:
    """
    WHAT: Serializes payment events into a canonical string and returns its SHA-256 digest.
    WHY: Provides a deterministic digest for comparing event sequences across simulation runs.
    Args: events: Chronological list of PaymentEvent instances.
    Returns: 64-character hexadecimal SHA-256 digest string.
    Example: h = _hash_event_list(res.events); assert len(h) == 64
    """
    lines: list[str] = []
    # Turn every field of each payment event into plain text so we can compare them.
    for ev in events:
        lines.append(
            f"{ev.minute}:{ev.from_account}:{ev.to_account}:{ev.amount_paise}:"
            f"{ev.from_bank_index}:{ev.to_bank_index}:{ev.status}:{ev.reason}:{ev.failure_domain}"
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def test_c1_determinism_identical_and_differing_seeds() -> None:
    """
    WHAT: Asserts identical seeds yield identical SHA-256 event digests, while different seeds differ.
    WHY: If the same seed ever gave different payments, no bug could be reproduced.
    Example:
        # Runs with seed 101 produce identical event lists; seed 202 produces different events
        # h1 = _hash_event_list(run_world(w1, p1, r1).events)
        # assert h1 == _hash_event_list(run_world(w2, p2, r2).events)
    """
    # Build the first town and run it with seed 101.
    w1 = build_world(seed=101, n_people=60, n_shops=12)
    p_rng1, r_rng1 = spawn_world_streams(101)
    res1 = run_world(w1, draw_world_plan(p_rng1, n_days=61), r_rng1)
    # Build an identical second town and run it with the exact same seed 101.
    w2 = build_world(seed=101, n_people=60, n_shops=12)
    p_rng2, r_rng2 = spawn_world_streams(101)
    res2 = run_world(w2, draw_world_plan(p_rng2, n_days=61), r_rng2)
    # Build a third town and run it with a completely different seed 202.
    w3 = build_world(seed=202, n_people=60, n_shops=12)
    p_rng3, r_rng3 = spawn_world_streams(202)
    res3 = run_world(w3, draw_world_plan(p_rng3, n_days=61), r_rng3)
    # Check that runs with the same seed match, and runs with different seeds do not.
    assert _hash_event_list(res1.events) == _hash_event_list(res2.events)
    assert _hash_event_list(res1.events) != _hash_event_list(res3.events)


def test_c2_ledger_invariants_pass_after_sixty_one_days() -> None:
    """
    WHAT: Evaluates non-negative balance, entry sum, and conservation checkers after 61 days.
    WHY: Protects against money disappearing or being created out of thin air across the banking system.
    Example:
        # All three checkers return empty lists indicating zero accounting violations
        # assert check_no_negative_balances(world.ledgers) == []
    """
    # Create the simulated world and add up the opening balance of all accounts.
    world = build_world(seed=42, n_people=80, n_shops=12)
    total_opening = sum(ldg.balance_of(acc) for ldg in world.ledgers for acc in ldg.accounts)
    # Split the random seed into independent streams for the plan and the simulation run.
    p_rng, r_rng = spawn_world_streams(42)
    plan = draw_world_plan(p_rng, n_days=61)
    # Run the town simulation for 61 calendar days across all three banks.
    run_world(world, plan, r_rng)
    # Make sure no person or shop ever ended up with a negative account balance.
    assert check_no_negative_balances(world.ledgers) == []
    # Make sure every balance matches the history of credits minus debits.
    assert check_balances_equal_entry_sums(world.ledgers) == []
    # Make sure the total amount of money in the town stayed exactly constant.
    assert check_money_conserved(world.ledgers, total_opening) == []


def test_c3_failed_payments_move_no_money() -> None:
    """
    WHAT: Asserts ledger entries match successful events exactly and failed payments mutate no funds.
    WHY: Protects issue R7 (failed payments are recorded but move no money) so declines never touch ledger balances.
    Example:
        # Successful event count equals transfer debit and credit pairs
        # assert len(debits) == len(successes) and sum(debit_paise) == sum(success_paise)
    """
    # Build a town and run the 61-day calendar simulation.
    world = build_world(seed=88, n_people=80, n_shops=12)
    p_rng, r_rng = spawn_world_streams(88)
    plan = draw_world_plan(p_rng, n_days=61)
    res = run_world(world, plan, r_rng)
    # Collect all transfer entries written to bank ledgers during the run.
    debits = [e for ldg in world.ledgers for e in ldg.entries if e.kind == "transfer_debit"]
    credits = [e for ldg in world.ledgers for e in ldg.entries if e.kind == "transfer_credit"]
    successes = [e for e in res.events if e.status == "success"]
    # Verify that only successful payments created ledger entries, exactly one debit and credit each.
    assert len(debits) == len(credits) == len(successes) == res.counts["successes"]
    # Verify that the total amount moved on ledgers equals the sum of successful payments.
    assert sum(abs(e.amount_paise) for e in debits) == sum(e.amount_paise for e in successes)


def test_c4_event_shape_and_chronological_ordering() -> None:
    """
    WHAT: Checks that every event adheres to domain and reason invariants and events are sorted.
    WHY: Protects the event log format and ensures simulated time never travels backward.
    Example:
        # Every failed event has an approved failure domain and minutes are non-decreasing
        # assert all(e.failure_domain in FAILURE_DOMAINS for e in failed_events)
    """
    # Build the town and run the 61-day calendar simulation.
    world = build_world(seed=77, n_people=60, n_shops=10)
    p_rng, r_rng = spawn_world_streams(77)
    plan = draw_world_plan(p_rng, n_days=61)
    res = run_world(world, plan, r_rng)
    assert len(res.events) > 0
    # Check that events appear in non-decreasing time order so time never moves backward.
    for i in range(len(res.events) - 1):
        assert res.events[i].minute <= res.events[i + 1].minute
    # Check that successful and failed payments carry the expected domain tags.
    for ev in res.events:
        if ev.status == "success":
            assert ev.failure_domain is None and ev.reason == "ok"
        else:
            assert ev.failure_domain in FAILURE_DOMAINS and ev.reason != "ok"


def _events_for_purpose(res: RunResult, purpose: str) -> list[PaymentEvent]:
    """
    WHAT: Filters simulated payment events by their recorded execution purpose label string.
    WHY: Enables purpose-driven evaluation across test assertions comparing like transactions with like.
    Args: res: RunResult containing generated payment event log and matching purpose labels; purpose: label.
    Returns: List of PaymentEvent records corresponding strictly to the requested purpose.
    Example:
        # everyday_events = _events_for_purpose(res, "everyday")
        # assert isinstance(everyday_events, list)
    """
    return [ev for ev, p in zip(res.events, res.event_purposes) if p == purpose]


def _measure_c5_degradation(
    events: list[PaymentEvent],
) -> tuple[int, int, float, float, int]:
    """
    WHAT: Measures payment counts, degraded bank failures, theoretical expectations, and unaffected failures for C5.
    WHY: Encapsulates statistical measurement calculations under 30 lines for bank downtime degradation checks.
    Args: events: Complete list of PaymentEvent records emitted during the calendar simulation.
    Returns: Tuple of (n0, b0_fails, exp, sd, unaffected) tracking transaction volume and failure bounds.
    Example:
        # n0, b0_fails, exp, sd, unaffected = _measure_c5_degradation(res.events)
        # assert n0 >= 0 and b0_fails >= 0
    """
    # Filter transactions occurring strictly within the bank downtime incident interval.
    w_events = [e for e in events if 1000 <= e.minute < 20000]
    # Isolate payments originating from the degraded financial institution.
    b0_events = [e for e in w_events if e.from_bank_index == 0]
    b0_fails = sum(1 for e in b0_events if e.failure_domain == "sender_bank")
    n0 = len(b0_events)
    # Expected failures is 50% with binomial standard deviation sqrt(n0 * 0.25).
    exp = n0 * 0.5
    sd = float(np.sqrt(n0 * 0.25))
    # Running out of money is not outage spillover, so count only non-customer failures.
    unaffected = sum(
        1 for e in w_events
        if e.from_bank_index != 0 and e.to_bank_index != 0 and e.status == "failed" and e.failure_domain != "customer"
    )
    return n0, b0_fails, exp, sd, unaffected


def test_c5_partial_degradation_under_bank_downtime() -> None:
    """
    WHAT: Measures failure share for bank downtime incident with rate 0.5 within 4 standard deviations.
    WHY: Protects realistic outage behavior where an affected bank slows down instead of completely crashing.
    Example:
        # Downtime at success rate 0.5 causes expected 50% failures: |k - 0.5*N| <= 4*sqrt(0.25*N)
        # assert abs(measured_fails - expected) <= 4 * sd
    """
    # Build a town and plan, then inject a temporary half-speed slowdown on bank 0.
    world = build_world(seed=55, n_people=100, n_shops=15)
    p_rng, r_rng = spawn_world_streams(55)
    base_plan = draw_world_plan(p_rng, n_days=61)
    inc = Incident("bank_downtime", 1000, 19000, 0, 0.5)
    clean_fails = FailureSettings(0.0, 0.0, 0.0)
    plan = dataclasses.replace(base_plan, incidents=[inc], failure_settings=clean_fails)
    res = run_world(world, plan, r_rng)
    # Compute downtime metrics and failure counts using measurement helper.
    n0, b0_fails, exp, sd, unaffected = _measure_c5_degradation(res.events)
    print(f"MEASURED C5: n={n0}, k={b0_fails}, expected={exp}, sd={sd}, 4sd={4.0*sd}, unaffected={unaffected}")
    # Make sure failures are within 4 standard deviations and bank 0 was partially degraded.
    assert abs(b0_fails - exp) <= 4.0 * sd, f"Measured {b0_fails} not within 4 SD ({4*sd}) of {exp}"
    assert 0 < b0_fails < n0, "Failure rate must be partial, not 0% or 100%"
    # Make sure transactions between completely unaffected banks had zero non-customer failures.
    assert unaffected == 0, f"Expected 0 failures on unaffected banks, got {unaffected}"


def _calc_poisson_expected_everyday(plan: Any, n_people: int) -> float:
    """
    WHAT: Computes expected total Poisson everyday transactions over the world timeline.
    WHY: Helper for test C6 to calculate analytical mean transaction volume across days.
    Args: plan: WorldPlan instance; n_people: total count of citizens in simulated town.
    Returns: Float expected transaction count summed across all simulation days.
    Example: exp = _calc_poisson_expected_everyday(plan, 100); assert exp > 0.0
    """
    total = 0.0
    # Walk through each day of the simulation to compute expected shopping trips.
    for d in range(plan.n_days):
        # Find how many days are in the current month to calculate daily base rate.
        dt = plan.clock.to_datetime(d * 1440)
        dim = calendar.monthrange(dt.year, dt.month)[1]
        base_rate = EVERYDAY_PAYMENTS_PER_MONTH / float(dim)
        # Multiply by the festival and weekly shopping multiplier for this day.
        boost = daily_multiplier(d * 1440, plan.clock, plan.rhythm_settings, plan.festivals)
        total += base_rate * boost * n_people
    return total


def _measure_c6_festival_volume(
    res: RunResult, plan: Any, n_people: int
) -> tuple[int, float, float, int, float]:
    """
    WHAT: Calculates total everyday purchases, expected Poisson mean, standard deviation, peak, and median.
    WHY: Supplies statistical measurement figures under 30 lines for festival surge and Poisson bound tests.
    Args: res: RunResult instance; plan: WorldPlan with festival multipliers; n_people: Citizen count in world.
    Returns: Tuple of (total, exp_total, sd, peak, med) containing volume metrics for C6.
    Example:
        # total, exp, sd, peak, med = _measure_c6_festival_volume(res, plan, 100)
        # assert total >= 0 and exp > 0.0
    """
    # Extract commercial everyday payments using purpose-based filtering.
    everyday = _events_for_purpose(res, "everyday")
    daily_counts = [0] * plan.n_days
    # Aggregate transaction attempts across each simulation day.
    for e in everyday:
        daily_counts[e.minute // 1440] += 1
    # Determine the peak volume on day 35 and the median volume across all days.
    peak = daily_counts[35]
    med = float(np.median(daily_counts))
    # Calculate Poisson expectation and standard deviation across the simulation.
    exp_total = _calc_poisson_expected_everyday(plan, n_people)
    sd = float(np.sqrt(exp_total))
    return len(everyday), exp_total, sd, peak, med


def test_c6_calendar_festival_volume_and_poisson_bound() -> None:
    """
    WHAT: Asserts peak festival attempts exceed 2x median daily volume and total matches Poisson within 4 SD.
    WHY: Protects festival surge realism ensuring commercial purchases spike noticeably during celebrations.
    Example:
        # Exaggerated festival bump at day 35 drives daily transactions > 2x median
        # assert peak_count > 2 * median_count and abs(total - expected) <= 4 * sqrt(expected)
    """
    # Build a town and plan with a large festival surge on day 35.
    world = build_world(seed=66, n_people=100, n_shops=15)
    p_rng, r_rng = spawn_world_streams(66)
    base_plan = draw_world_plan(p_rng, n_days=61)
    fest = Festival("diwali", 35, 5, 5, 4.0)
    plan = dataclasses.replace(base_plan, festivals=[fest])
    res = run_world(world, plan, r_rng)
    # Measure everyday volume metrics across the simulation horizon.
    total, exp_total, sd, peak, med = _measure_c6_festival_volume(res, plan, len(world.people))
    print(f"MEASURED C6: total={total}, expected={exp_total}, sd={sd}, 4sd={4.0*sd}, peak={peak}, median={med}")
    # Check that day 35 saw more than double the median daily shopping volume.
    assert peak > 2.0 * med, f"Peak {peak} not > 2x median {med}"
    # Check that the 61-day total is within 4 standard deviations of the expected Poisson sum.
    diff = abs(total - exp_total)
    assert diff <= 4.0 * sd, f"Measured {total} diff {diff} exceeds 4 SD ({4*sd}) from {exp_total}"


def _measure_c7_diurnal_night(
    res: RunResult, world: Any, plan: WorldPlan, persona: str
) -> tuple[int, int, float, float]:
    """
    WHAT: Measures sample size, night payments (hours 0-5), expected count, and standard deviation for C7.
    WHY: Supplies statistical measurement figures under 30 lines for persona diurnal rhythm tests.
    Args: res: RunResult instance; world: Simulated town; plan: WorldPlan instance; persona: Target demographic.
    Returns: Tuple of (n, k, exp, sd) tracking night payment counts against theoretical expectations.
    Example:
        # n, k, exp, sd = _measure_c7_diurnal_night(res, world, plan, "salaried_employee")
        # assert n >= 0 and k >= 0
    """
    # Isolate everyday payments initiated by people belonging to the target persona.
    accs = {p.account_id for p in world.people if p.persona_name == persona}
    everyday = [
        e for e in _events_for_purpose(res, "everyday")
        if e.from_account in accs and e.to_account.startswith("acc_shop_")
    ]
    # Compute the theoretical fraction of activity occurring between hours 0 and 5.
    raw = [hour_weight(h + 0.5, persona, plan.rhythm_settings) for h in range(24)]
    p_0_5 = sum(raw[:6]) / sum(raw)
    n = len(everyday)
    # Count the observed number of everyday payments falling within midnight to 6 AM.
    k = sum(1 for e in everyday if (e.minute % 1440) // 60 < 6)
    exp = n * p_0_5
    sd = float(np.sqrt(n * p_0_5 * (1.0 - p_0_5)))
    return n, k, exp, sd


def test_c7_persona_daily_diurnal_rhythm() -> None:
    """
    WHAT: Tests that persona everyday payments in hours 0-5 match diurnal curve within 4 standard deviations.
    WHY: Protects realistic daily human rhythms so people make payments during the day rather than late night.
    Example:
        # Binomial distribution for payments in hours 0-5: |k - n*p| <= 4 * sqrt(n*p*(1-p))
        # assert abs(measured_k - exp_k) <= 4 * sd
    """
    # Build a town and run the 61-day calendar simulation.
    world = build_world(seed=33, n_people=100, n_shops=15)
    p_rng, r_rng = spawn_world_streams(33)
    plan = draw_world_plan(p_rng, n_days=61)
    res = run_world(world, plan, r_rng)
    persona = "salaried_employee"
    # Measure night-time shopping frequency using diurnal measurement helper.
    n, k, exp, sd = _measure_c7_diurnal_night(res, world, plan, persona)
    print(f"MEASURED C7: n={n}, k={k}, expected={exp}, sd={sd}, 4sd={4.0*sd}")
    # Check that night payments match expected count within 4 standard deviations.
    diff = abs(k - exp)
    assert diff <= 4.0 * sd, f"Measured {k} diff {diff} exceeds 4 SD ({4*sd}) from {exp}"


def test_c8_insufficient_balance_produces_customer_failure() -> None:
    """
    WHAT: Drains citizen balance via valid payment, verifies subsequent attempt logs customer failure.
    WHY: Protects against overdrafts so accounts with zero balance fail payments safely without negative balances.
    Example:
        # Account drained to 0 paise rejects 500 paise with status 'failed' and domain 'customer'
        # assert ev.status == 'failed' and ev.failure_domain == 'customer'
    """
    # Build a town and find the first citizen and their bank ledger.
    world = build_world(seed=22, n_people=50, n_shops=10)
    p_rng, r_rng = spawn_world_streams(22)
    plan = draw_world_plan(p_rng, n_days=61)
    bank_map = _build_account_bank_map(world.ledgers)
    person, shop = world.people[0], world.shops[0]
    ldg = find_ledger_for_account(world.ledgers, person.account_id)
    assert ldg is not None
    # Spend the entire balance legitimately so the account is completely empty.
    bal = ldg.balance_of(person.account_id)
    make_payment(world, person.account_id, shop.account_id, bal, 0)
    assert ldg.balance_of(person.account_id) == 0
    entries_before = len(ldg.entries)
    # Attempt to send 500 paise from the empty account.
    pmt = PlannedPayment(10, person.account_id, shop.account_id, 500, 0, purpose="everyday")
    ev = _execute_payment_leg(world, plan, pmt, 500, bank_map, r_rng)
    # Check that payment failed with customer domain and bank ledger did not change.
    assert ev.status == "failed" and ev.failure_domain == "customer" and ev.reason == "insufficient_balance"
    assert ldg.balance_of(person.account_id) == 0 and len(ldg.entries) == entries_before


def test_c9_fixed_items_follow_calendar_months() -> None:
    """
    WHAT: Tests that salaried employees receive exactly two salary attempts inside configured windows.
    WHY: Protects monthly recurring living costs so salaries and rent execute on actual calendar dates.
    Example:
        # In 61 days covering Sep and Oct, salaried employees receive exactly 1 salary credit per month
        # assert len(salary_events) == 2 and 1 <= dt.day <= 5
    """
    # Build a town and run the 61-day simulation across September and October.
    world = build_world(seed=12, n_people=80, n_shops=12)
    p_rng, r_rng = spawn_world_streams(12)
    plan = draw_world_plan(p_rng, n_days=61)
    res = run_world(world, plan, r_rng)
    # Find all salaried workers and employer store accounts.
    salaried = [p for p in world.people if p.persona_name == "salaried_employee"]
    employer_accs = {s.account_id for s in world.shops if s.kind == "employer"}
    salary_events = _events_for_purpose(res, "salary")
    # Check that each salaried worker got paid twice, once in September and once in October.
    for p in salaried:
        sal_events = [e for e in salary_events if e.to_account == p.account_id]
        assert len(sal_events) == 2, f"Expected exactly 2 salary events, got {len(sal_events)}"
        for ev in sal_events:
            assert ev.from_account in employer_accs, f"Expected employer sender account, got {ev.from_account}"
            dt = plan.clock.to_datetime(ev.minute)
            assert dt.month in (9, 10), f"Unexpected month {dt.month}"
            assert SALARY_DAY_WINDOW[0] <= dt.day <= SALARY_DAY_WINDOW[1], f"Day {dt.day} outside {SALARY_DAY_WINDOW}"


def test_c10_legacy_runner_and_tests_untouched() -> None:
    """
    WHAT: Asserts 11A routine runner executes on fresh world and preserves ledger consistency.
    WHY: Protects backward compatibility ensuring earlier simulation engines and tests continue working.
    Example:
        # Running 11A run_one_month produces positive success counts and clean ledger invariant checks
        # results = run_one_month(world, rng); assert results.get('ok', 0) > 0
    """
    # Create a fresh town and a single random number generator.
    world = build_world(seed=99, n_people=50, n_shops=10)
    rng = np.random.default_rng(99)
    # Run the original 11A one-month routine cycle.
    results = run_one_month(world, rng)
    # Check that payments completed and all ledgers preserved accounting rules.
    assert sum(results.values()) > 0 and results.get("ok", 0) > 0
    assert check_no_negative_balances(world.ledgers) == []
    assert check_balances_equal_entry_sums(world.ledgers) == []
