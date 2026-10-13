"""
WHAT: Unit tests C1-C10 verifying determinism, ledger conservation, failure attribution, and calendar rhythms.
WHY: Guarantees that the calendar runner conforms to rule R7, preserves accounting invariants, and models outages.
TERM: calendar runner: An execution engine that schedules and runs financial transactions across simulated time.
TERM: payment event: An immutable record documenting transaction outcome, financial routing, and failure domains.
TERM: failure domain: The operational zone responsible for a payment rejection (central, banks, app, or user).
TERM: partial degradation: A temporary drop in transaction success probability without complete service blackout.
TERM: determinism: The guarantee that identical pseudo-random seeds yield bitwise identical simulation objects.
"""

import calendar
import dataclasses
import hashlib
from typing import Any

import numpy as np

from world.build_world import build_world
from world.calendar_runner import PlannedPayment, _build_account_bank_map, _execute_payment_leg, run_world
from world.checks import check_balances_equal_entry_sums, check_money_conserved, check_no_negative_balances
from world.failures import FAILURE_DOMAINS, FailureSettings, PaymentEvent
from world.festivals import Festival
from world.incidents import Incident
from world.payments import find_ledger_for_account, make_payment
from world.placeholders import EVERYDAY_PAYMENTS_PER_MONTH, SALARY_DAY_WINDOW
from world.rhythms import daily_multiplier, hour_weight
from world.routine import run_one_month
from world.world_plan import draw_world_plan


# =============================================================================
# CALENDAR RUNNER VERIFICATION SUITE: R7 AND 11B-3A INVARIANTS
# =============================================================================
# This test module exercises the discrete-event calendar simulation runner.
# It proves that the schedule conforms to all double-entry accounting rules,
# guarantees that degraded banking infrastructure produces realistic dropouts,
# and verifies that festive seasons and diurnal rhythms match statistical targets.
# All statistical bounds are evaluated at k = 4 standard deviations (99.993% confidence).
def _hash_event_list(events: list[PaymentEvent]) -> str:
    """
    WHAT: Serializes payment events into a canonical string and returns its SHA-256 digest.
    WHY: Provides a deterministic digest for comparing event sequences across simulation runs.
    Args: events: Chronological list of PaymentEvent instances.
    Returns: 64-character hexadecimal SHA-256 digest string.
    Example: h = _hash_event_list(res.events); assert len(h) == 64
    """
    lines: list[str] = []
    # Build unambiguous deterministic string lines for each event record.
    for ev in events:
        lines.append(
            f"{ev.minute}:{ev.from_account}:{ev.to_account}:{ev.amount_paise}:"
            f"{ev.from_bank_index}:{ev.to_bank_index}:{ev.status}:{ev.reason}:{ev.failure_domain}"
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# C1: Determinism and PRNG Stream Reproducibility
# -----------------------------------------------------------------------------
# Evaluates that identical random seed sequences produce identical transaction event
# logs byte-for-byte, while distinct random seeds yield distinct event sequences.
def test_c1_determinism_identical_and_differing_seeds() -> None:
    """
    WHAT: Asserts identical seeds yield identical SHA-256 event digests, while different seeds differ.
    WHY: Verifies requirement C1 proving complete reproducibility of payments and failure events.
    Example:
        # Runs with seed 101 produce identical event lists; seed 202 produces different events
        # h1 = _hash_event_list(run_world(w1, p1, r1).events)
        # assert h1 == _hash_event_list(run_world(w2, p2, r2).events)
    """
    # Build two worlds and plans using identical seed 101.
    w1 = build_world(seed=101, n_people=60, n_shops=12)
    p1 = draw_world_plan(np.random.default_rng(101), n_days=61)
    res1 = run_world(w1, p1, np.random.default_rng(101))
    w2 = build_world(seed=101, n_people=60, n_shops=12)
    p2 = draw_world_plan(np.random.default_rng(101), n_days=61)
    res2 = run_world(w2, p2, np.random.default_rng(101))
    # Build third run using distinct seed 202.
    w3 = build_world(seed=202, n_people=60, n_shops=12)
    p3 = draw_world_plan(np.random.default_rng(202), n_days=61)
    res3 = run_world(w3, p3, np.random.default_rng(202))
    # Assert identical digests for seed 101 and differing digest for seed 202.
    assert _hash_event_list(res1.events) == _hash_event_list(res2.events)
    assert _hash_event_list(res1.events) != _hash_event_list(res3.events)


# -----------------------------------------------------------------------------
# C2: Multi-Bank Double-Entry Accounting Invariants
# -----------------------------------------------------------------------------
# Runs a full 61-day town simulation across all banks and asserts that no balances
# become negative, entry sums match current balances, and total money is conserved.
def test_c2_ledger_invariants_pass_after_sixty_one_days() -> None:
    """
    WHAT: Evaluates non-negative balance, entry sum, and conservation checkers after 61 days.
    WHY: Verifies requirement C2 proving accounting invariants hold across all banks.
    Example:
        # All three checkers return empty lists indicating zero accounting violations
        # assert check_no_negative_balances(world.ledgers) == []
    """
    world = build_world(seed=42, n_people=80, n_shops=12)
    total_opening = sum(ldg.balance_of(acc) for ldg in world.ledgers for acc in ldg.accounts)
    plan = draw_world_plan(np.random.default_rng(42), n_days=61)
    # Execute full 61-day simulation run across all institutions.
    run_world(world, plan, np.random.default_rng(42))
    # Verify no account balance drops below zero paise.
    assert check_no_negative_balances(world.ledgers) == []
    # Verify balances equal sum of debits and credits.
    assert check_balances_equal_entry_sums(world.ledgers) == []
    # Verify total money is strictly conserved.
    assert check_money_conserved(world.ledgers, total_opening) == []


# -----------------------------------------------------------------------------
# C3: Money Non-Movement on Transaction Failures (Rule R7)
# -----------------------------------------------------------------------------
# Asserts that failed transactions generate audit events but mutate zero ledger state.
# Total posted debit and credit entries must exactly equal successful events.
def test_c3_failed_payments_move_no_money() -> None:
    """
    WHAT: Asserts ledger entries match successful events exactly and failed payments mutate no funds.
    WHY: Verifies requirement C3 proving rule R7: failed payments post zero ledger entries.
    Example:
        # Successful event count equals transfer debit and credit pairs
        # assert len(debits) == len(successes) and sum(debit_paise) == sum(success_paise)
    """
    world = build_world(seed=88, n_people=80, n_shops=12)
    plan = draw_world_plan(np.random.default_rng(88), n_days=61)
    res = run_world(world, plan, np.random.default_rng(88))
    # Extract transfer debits and credits posted during the run.
    debits = [e for ldg in world.ledgers for e in ldg.entries if e.kind == "transfer_debit"]
    credits = [e for ldg in world.ledgers for e in ldg.entries if e.kind == "transfer_credit"]
    successes = [e for e in res.events if e.status == "success"]
    # Check that transfer debits and credits match successful events count.
    assert len(debits) == len(credits) == len(successes) == res.counts["successes"]
    # Check that absolute sum of debits matches sum of successful event amounts.
    assert sum(abs(e.amount_paise) for e in debits) == sum(e.amount_paise for e in successes)


# -----------------------------------------------------------------------------
# C4: Event Record Schema Invariants and Monotonic Ordering
# -----------------------------------------------------------------------------
# Verifies that all emitted PaymentEvents carry valid failure domains and reasons,
# and that the timeline proceeds strictly monotonically in non-decreasing minutes.
def test_c4_event_shape_and_chronological_ordering() -> None:
    """
    WHAT: Checks that every event adheres to domain and reason invariants and events are sorted.
    WHY: Verifies requirement C4 guaranteeing schema validity and monotonic minute progression.
    Example:
        # Every failed event has an approved failure domain and minutes are non-decreasing
        # assert all(e.failure_domain in FAILURE_DOMAINS for e in failed_events)
    """
    world = build_world(seed=77, n_people=60, n_shops=10)
    plan = draw_world_plan(np.random.default_rng(77), n_days=61)
    res = run_world(world, plan, np.random.default_rng(77))
    assert len(res.events) > 0
    # Check that events advance chronologically.
    for i in range(len(res.events) - 1):
        assert res.events[i].minute <= res.events[i + 1].minute
    for ev in res.events:
        if ev.status == "success":
            assert ev.failure_domain is None and ev.reason == "ok"
        else:
            assert ev.failure_domain in FAILURE_DOMAINS and ev.reason != "ok"


# -----------------------------------------------------------------------------
# C5: Banking Infrastructure Downtime and Partial Degradation
# -----------------------------------------------------------------------------
# Injects a partial outage on bank 0 and proves that failure rates match the configured
# degradation probability within 4 standard deviations, without causing a total blackout.
def test_c5_partial_degradation_under_bank_downtime() -> None:
    """
    WHAT: Measures failure share for bank downtime incident with rate 0.5 within 4 standard deviations.
    WHY: Verifies requirement C5 proving degraded operation without full network blackouts.
    Example:
        # Downtime at success rate 0.5 causes expected 50% failures: |k - 0.5*N| <= 4*sqrt(0.25*N)
        # assert abs(measured_fails - expected) <= 4 * sd
    """
    world = build_world(seed=55, n_people=100, n_shops=15)
    base_plan = draw_world_plan(np.random.default_rng(55), n_days=61)
    # Inject bank downtime on bank 0 from minute 1000 to 20000 with 0.5 success rate.
    inc = Incident("bank_downtime", 1000, 19000, 0, 0.5)
    clean_fails = FailureSettings(0.0, 0.0, 0.0)
    plan = dataclasses.replace(base_plan, incidents=[inc], failure_settings=clean_fails)
    res = run_world(world, plan, np.random.default_rng(55))
    w_events = [e for e in res.events if 1000 <= e.minute < 20000]
    b0_events = [e for e in w_events if e.from_bank_index == 0]
    b0_fails = sum(1 for e in b0_events if e.failure_domain == "sender_bank")
    n0 = len(b0_events)
    # Arithmetic: binomial draw with p = 0.5, expected = n0 * 0.5, SD = sqrt(n0 * 0.5 * 0.5).
    exp = n0 * 0.5
    sd = float(np.sqrt(n0 * 0.25))
    assert abs(b0_fails - exp) <= 4.0 * sd, f"Measured {b0_fails} not within 4 SD ({4*sd}) of {exp}"
    assert 0 < b0_fails < n0, "Failure rate must be partial, not 0% or 100%"
    # Unaffected institutions experience zero failures since background rate is 0.
    unaffected = sum(1 for e in w_events if e.from_bank_index != 0 and e.to_bank_index != 0 and e.status == "failed")
    assert unaffected == 0, f"Expected 0 failures on unaffected banks, got {unaffected}"


# -----------------------------------------------------------------------------
# C6: Calendar Festival Multipliers and Poisson Volume Surges
# -----------------------------------------------------------------------------
# Verifies that major calendar festivals drive commerce surges > 2x median daily volume,
# and that total commercial transaction count matches analytical Poisson expectations.
def _calc_poisson_expected_everyday(plan: Any, n_people: int) -> float:
    """
    WHAT: Computes expected total Poisson everyday transactions over the world timeline.
    WHY: Helper for test C6 to calculate analytical mean transaction volume across days.
    Args: plan: WorldPlan instance; n_people: total count of citizens in simulated town.
    Returns: Float expected transaction count summed across all simulation days.
    Example: exp = _calc_poisson_expected_everyday(plan, 100); assert exp > 0.0
    """
    total = 0.0
    # Sum daily Poisson intensities modulated by calendar length and activity boost.
    for d in range(plan.n_days):
        dt = plan.clock.to_datetime(d * 1440)
        dim = calendar.monthrange(dt.year, dt.month)[1]
        base_rate = EVERYDAY_PAYMENTS_PER_MONTH / float(dim)
        boost = daily_multiplier(d * 1440, plan.clock, plan.rhythm_settings, plan.festivals)
        total += base_rate * boost * n_people
    return total


def test_c6_calendar_festival_volume_and_poisson_bound() -> None:
    """
    WHAT: Asserts peak festival attempts exceed 2x median daily volume and total matches Poisson within 4 SD.
    WHY: Verifies requirement C6 proving calendar multipliers drive transactional volume surges.
    Example:
        # Exaggerated festival bump at day 35 drives daily transactions > 2x median
        # assert peak_count > 2 * median_count and abs(total - expected) <= 4 * sqrt(expected)
    """
    world = build_world(seed=66, n_people=100, n_shops=15)
    base_plan = draw_world_plan(np.random.default_rng(66), n_days=61)
    fest = Festival("diwali", 35, 5, 5, 4.0)
    plan = dataclasses.replace(base_plan, festivals=[fest])
    res = run_world(world, plan, np.random.default_rng(66))
    everyday = [e for e in res.events if e.to_account.startswith("acc_shop_") and e.minute % 1440 != 600]
    daily_counts = [0] * 61
    for e in everyday:
        daily_counts[e.minute // 1440] += 1
    med = float(np.median(daily_counts))
    assert daily_counts[35] > 2.0 * med, f"Peak {daily_counts[35]} not > 2x median {med}"
    # Arithmetic: Poisson variance = mean, so SD = sqrt(expected).
    exp_total = _calc_poisson_expected_everyday(plan, len(world.people))
    sd = float(np.sqrt(exp_total))
    diff = abs(len(everyday) - exp_total)
    assert diff <= 4.0 * sd, f"Measured {len(everyday)} diff {diff} exceeds 4 SD ({4*sd}) from {exp_total}"


# -----------------------------------------------------------------------------
# C7: Persona Diurnal Rhythms and Hourly Spending Distributions
# -----------------------------------------------------------------------------
# Asserts that transaction timestamps follow hourly diurnal activity curves within 4 SD,
# validating that night hours (0-5) experience appropriate low transaction frequencies.
def test_c7_persona_daily_diurnal_rhythm() -> None:
    """
    WHAT: Tests that persona everyday payments in hours 0-5 match diurnal curve within 4 standard deviations.
    WHY: Verifies requirement C7 proving diurnal hourly curve shapes transaction timestamps.
    Example:
        # Binomial distribution for payments in hours 0-5: |k - n*p| <= 4 * sqrt(n*p*(1-p))
        # assert abs(measured_k - exp_k) <= 4 * sd
    """
    world = build_world(seed=33, n_people=100, n_shops=15)
    plan = draw_world_plan(np.random.default_rng(33), n_days=61)
    res = run_world(world, plan, np.random.default_rng(33))
    persona = "salaried_employee"
    accs = {p.account_id for p in world.people if p.persona_name == persona}
    everyday = [
        e for e in res.events
        if e.from_account in accs and e.to_account.startswith("acc_shop_") and e.minute % 1440 != 600
    ]
    raw = [hour_weight(h + 0.5, persona, plan.rhythm_settings) for h in range(24)]
    p_0_5 = sum(raw[:6]) / sum(raw)
    n = len(everyday)
    k = sum(1 for e in everyday if (e.minute % 1440) // 60 < 6)
    # Arithmetic: binomial draw with probability p_0_5, expected = n * p_0_5, SD = sqrt(n * p * (1-p)).
    exp = n * p_0_5
    sd = float(np.sqrt(n * p_0_5 * (1.0 - p_0_5)))
    diff = abs(k - exp)
    assert diff <= 4.0 * sd, f"Measured {k} diff {diff} exceeds 4 SD ({4*sd}) from {exp}"


# -----------------------------------------------------------------------------
# C8: Customer Insolvency and Insufficient Balance Failures
# -----------------------------------------------------------------------------
# Drains a citizen account to zero paise and verifies that subsequent payments fail
# with the customer failure domain and insufficient_balance reason code.
def test_c8_insufficient_balance_produces_customer_failure() -> None:
    """
    WHAT: Drains citizen balance via valid payment, verifies subsequent attempt logs customer failure.
    WHY: Verifies requirement C8 proving insolvency triggers failure events without ledger corruption.
    Example:
        # Account drained to 0 paise rejects 500 paise with status 'failed' and domain 'customer'
        # assert ev.status == 'failed' and ev.failure_domain == 'customer'
    """
    world = build_world(seed=22, n_people=50, n_shops=10)
    plan = draw_world_plan(np.random.default_rng(22), n_days=61)
    bank_map = _build_account_bank_map(world.ledgers)
    person, shop = world.people[0], world.shops[0]
    ldg = find_ledger_for_account(world.ledgers, person.account_id)
    assert ldg is not None
    bal = ldg.balance_of(person.account_id)
    # Drain citizen balance completely using legitimate payment engine transfer.
    make_payment(world, person.account_id, shop.account_id, bal, 0)
    assert ldg.balance_of(person.account_id) == 0
    entries_before = len(ldg.entries)
    pmt = PlannedPayment(10, person.account_id, shop.account_id, 500, 0)
    ev = _execute_payment_leg(world, plan, pmt, 500, bank_map, np.random.default_rng(22))
    assert ev.status == "failed" and ev.failure_domain == "customer" and ev.reason == "insufficient_balance"
    assert ldg.balance_of(person.account_id) == 0 and len(ldg.entries) == entries_before


# -----------------------------------------------------------------------------
# C9: Multi-Month Recurring Obligations and Payroll Windows
# -----------------------------------------------------------------------------
# Verifies that salaried employees receive exactly one salary payment per calendar month,
# occurring within the configured day-of-month window (days 1 to 5).
def test_c9_fixed_items_follow_calendar_months() -> None:
    """
    WHAT: Tests that salaried employees receive exactly two salary attempts inside configured windows.
    WHY: Verifies requirement C9 proving recurring fixed items follow true calendar dates.
    Example:
        # In 61 days covering Sep and Oct, salaried employees receive exactly 1 salary credit per month
        # assert len(salary_events) == 2 and 1 <= dt.day <= 5
    """
    world = build_world(seed=12, n_people=80, n_shops=12)
    plan = draw_world_plan(np.random.default_rng(12), n_days=61)
    res = run_world(world, plan, np.random.default_rng(12))
    salaried = [p for p in world.people if p.persona_name == "salaried_employee"]
    employer_accs = {s.account_id for s in world.shops if s.kind == "employer"}
    for p in salaried:
        sal_events = [e for e in res.events if e.to_account == p.account_id and e.from_account in employer_accs]
        assert len(sal_events) == 2, f"Expected exactly 2 salary events, got {len(sal_events)}"
        for ev in sal_events:
            dt = plan.clock.to_datetime(ev.minute)
            assert dt.month in (9, 10), f"Unexpected month {dt.month}"
            assert SALARY_DAY_WINDOW[0] <= dt.day <= SALARY_DAY_WINDOW[1], f"Day {dt.day} outside {SALARY_DAY_WINDOW}"


# -----------------------------------------------------------------------------
# C10: Backward Compatibility with Legacy 11A Routine Simulation
# -----------------------------------------------------------------------------
# Executes run_one_month to prove that legacy 11A routine simulation behavior is preserved
# without breaking existing ledger checks or baseline assumptions.
def test_c10_legacy_runner_and_tests_untouched() -> None:
    """
    WHAT: Asserts 11A routine runner executes on fresh world and preserves ledger consistency.
    WHY: Verifies requirement C10 proving backward compatibility with original 11A simulation.
    Example:
        # Running 11A run_one_month produces positive success counts and clean ledger invariant checks
        # results = run_one_month(world, rng); assert results.get('ok', 0) > 0
    """
    world = build_world(seed=99, n_people=50, n_shops=10)
    rng = np.random.default_rng(99)
    # Execute legacy 11A routine life runner across citizens.
    results = run_one_month(world, rng)
    assert sum(results.values()) > 0 and results.get("ok", 0) > 0
    assert check_no_negative_balances(world.ledgers) == []
    assert check_balances_equal_entry_sums(world.ledgers) == []
