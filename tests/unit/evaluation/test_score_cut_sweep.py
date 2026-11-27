"""
WHAT: Equivalence and timing tests for the fast anomaly score-cut sweep.

WHY: tune_anomaly_threshold must pick exactly the same cut, rate, recall and budget verdict as the
old per-cut loop while running in O(rows + cuts log cuts) instead of O(cuts x rows). The old loop is
kept here as a private reference so the equivalence claim is measured against the real metric
helpers, not against a second copy of their arithmetic.
"""

import random
import time
from datetime import date

from evaluation.level17_scoring import GENUINE_ROLES, ScoredDay, anomaly_restricted_days
from evaluation.rules_tuning import (
    AnomalyTuningResult,
    TuningConfig,
    _check_budget,
    _check_split,
    _validation_metrics,
    load_tuning_config,
    tune_anomaly_threshold,
)

ROLE_CHOICES = ["genuine", "victim", "mule", "collector", "cash_out"]
SCORE_CHOICES = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
BUDGET_CHOICES = [0.0, 0.5, 1.0, 2.5, 10.0]


def _reference_tune_anomaly_threshold(
    scored: list[ScoredDay],
    tuning: TuningConfig,
    split: str,
    budget_pct: float,
) -> AnomalyTuningResult:
    """
    WHAT: The old per-cut implementation of tune_anomaly_threshold, kept only for equivalence tests.

    Inputs and outputs match the production function. Selection uses the same keys: highest recall,
    then lowest rate, then earliest candidate among budget-meeting cuts; otherwise lowest rate, then
    highest recall, then earliest candidate with meets_budget False.

    Limits:
        Duplicates the production selection loop on purpose; the metric arithmetic itself comes from
        the real anomaly_restricted_days and _validation_metrics helpers.
    """
    _check_split(tuning, split)
    _check_budget(budget_pct)

    genuine_days = [scored_day for scored_day in scored if scored_day.role in GENUINE_ROLES]
    if not genuine_days:
        raise ValueError("Cannot tune an anomaly threshold without genuine account-days")

    candidate_thresholds: list[float] = []
    for scored_day in genuine_days:
        if scored_day.anomaly_score is None:
            raise ValueError("anomaly scores are required on every genuine account-day for tuning")
        candidate_thresholds.append(scored_day.anomaly_score)
    thresholds = sorted(set(candidate_thresholds))

    best_meeting: tuple[tuple[float, float, int], float, float, float] | None = None
    best_any: tuple[tuple[float, float, int], float, float, float] | None = None
    for index, threshold in enumerate(thresholds):
        restricted = anomaly_restricted_days(scored, threshold)
        validation_rate_pct, validation_recall = _validation_metrics(scored, restricted)

        if validation_rate_pct <= budget_pct:
            meeting_key = (-validation_recall, validation_rate_pct, index)
            if best_meeting is None or meeting_key < best_meeting[0]:
                best_meeting = (meeting_key, threshold, validation_rate_pct, validation_recall)
        any_key = (validation_rate_pct, -validation_recall, index)
        if best_any is None or any_key < best_any[0]:
            best_any = (any_key, threshold, validation_rate_pct, validation_recall)

    chosen = best_meeting if best_meeting is not None else best_any
    if chosen is None:
        raise ValueError("No anomaly score candidates available; cannot select a threshold")
    _, threshold, validation_rate_pct, validation_recall = chosen

    return AnomalyTuningResult(
        threshold=threshold,
        validation_rate_pct=validation_rate_pct,
        validation_recall=validation_recall,
        meets_budget=validation_rate_pct <= budget_pct,
    )


def _random_scored_case(rng: random.Random) -> list[ScoredDay]:
    """
    WHAT: Builds one random scored-day list with ties, mixed roles on one account and no-mule cases.

    The first account-day is always genuine so the tuner has at least one candidate cut. Roles are
    drawn per day, so an account can be genuine on some days and a mule on others.
    """
    account_count = rng.randint(1, 12)
    day_count = rng.randint(1, 6)
    rows: list[ScoredDay] = []
    for account_index in range(account_count):
        for day_index in range(day_count):
            role = "genuine" if account_index == 0 and day_index == 0 else rng.choice(ROLE_CHOICES)
            depth = rng.randint(1, 3) if role == "mule" else None
            rows.append(
                ScoredDay(
                    world_id="world_a",
                    bank_id=f"bank_{account_index % 2}",
                    account_id=f"account_{account_index}",
                    day=date(2026, 1, 1 + day_index),
                    rules_score=0,
                    rules_fired=(),
                    anomaly_score=rng.choice(SCORE_CHOICES),
                    role=role,
                    depth=depth,
                )
            )
    rng.shuffle(rows)
    return rows


def test_sweep_matches_the_old_loop_on_two_hundred_random_cases() -> None:
    """
    WHAT: Proves the fast sweep returns the same threshold, rate, recall and verdict as the old loop.

    The run uses fixed seeds 1000..1199 (200 cases) with score ties, accounts that are genuine on
    some days and mules on others, no-mule cases and budgets no cut can meet. Counters assert those
    scenarios really occurred, so the test cannot pass on a uniform run.
    """
    tuning = load_tuning_config()
    saw_no_mules = False
    saw_budget_miss = False
    saw_mixed_role = False

    for seed in range(1000, 1200):
        rng = random.Random(seed)
        scored = _random_scored_case(rng)
        budget_pct = rng.choice(BUDGET_CHOICES)

        expected = _reference_tune_anomaly_threshold(scored, tuning, "validation", budget_pct)
        actual = tune_anomaly_threshold(scored, tuning, "validation", budget_pct)

        assert actual == expected, f"seed {seed} budget {budget_pct}"

        if not any(row.role in {"mule", "collector", "cash_out"} for row in scored):
            saw_no_mules = True
        if not expected.meets_budget:
            saw_budget_miss = True
        roles_by_account: dict[str, set[str]] = {}
        for row in scored:
            roles_by_account.setdefault(row.account_id, set()).add(row.role)
        for roles in roles_by_account.values():
            has_genuine_day = bool(roles & {"genuine", "victim"})
            has_mule_day = bool(roles & {"mule", "collector", "cash_out"})
            if has_genuine_day and has_mule_day:
                saw_mixed_role = True
                break

    assert saw_no_mules
    assert saw_budget_miss
    assert saw_mixed_role


def test_sweep_handles_two_hundred_thousand_rows_well_under_ten_seconds() -> None:
    """
    WHAT: Proves the sweep scales to full-pilot size.

    Builds 200,000 synthetic account-days and times one tune_anomaly_threshold call at the main
    budget. The assertion is the under-ten-second guard; the measured seconds are printed by pytest
    only on failure.
    """
    rng = random.Random(20261118)
    days = [date(2026, 1, 1 + index % 10) for index in range(10)]
    rows: list[ScoredDay] = []
    for account_index in range(20000):
        role = "mule" if account_index % 50 == 0 else "genuine"
        for day in days:
            rows.append(
                ScoredDay(
                    world_id="world_a",
                    bank_id="bank_0",
                    account_id=f"account_{account_index}",
                    day=day,
                    rules_score=0,
                    rules_fired=(),
                    anomaly_score=rng.random(),
                    role=role,
                    depth=1 if role == "mule" else None,
                )
            )

    tuning = load_tuning_config()
    started = time.perf_counter()
    result = tune_anomaly_threshold(rows, tuning, "validation", 0.5)
    elapsed_seconds = time.perf_counter() - started

    assert len(rows) == 200000
    assert 0.0 <= result.validation_rate_pct <= 100.0
    assert elapsed_seconds < 10.0, f"sweep took {elapsed_seconds:.2f}s"
