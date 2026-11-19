"""
WHAT: Equivalence and timing tests for the vectorised rules grid in evaluation/rules_tuning.py.

WHY: The tuner must return exactly the thresholds, rate, recall, budget verdict, per-rule numbers
and collected grid points the per-combination loop returned, because the pilot's reported rules
operating point comes from it. The old loop is kept here as a private reference so the comparison
is measured against the real scoring and metric helpers, not against a second copy of their
arithmetic. The timing test proves one full 2,916-point grid over 200,000 account-days stays well
inside the run budget.
"""

import random
import time
from datetime import datetime
from itertools import product
from typing import Any

import pytest

from core.event_clock import IST
from decisions.rules_config import (
    RulesConfig,
    load_rules_config,
    load_rules_config_with_labels,
    with_thresholds,
)
from evaluation.level17_scoring import one_row_per_account_day, rules_restricted_days, score_account_days
from evaluation.rules_tuning import (
    RulesGridPoint,
    RuleStandalone,
    RulesTuningResult,
    TuningConfig,
    _check_budget,
    _check_split,
    _validation_metrics,
    load_labels_tuning_config,
    load_tuning_config,
    select_rules_grid_point,
    tune_rules,
)

FEATURE_NAMES = (
    "largest_credit_paise_24h",
    "pass_through_ratio_24h",
    "median_credit_to_debit_seconds_7d",
    "account_age_days",
    "distinct_payers_24h",
    "current_balance_paise",
    "new_device_r5",
    "registry_match_count",
    "hops_to_complained_account",
)

# Every list contains None (a missing snapshot value) and the exact untuned_start values, so the
# random cases hit boundary comparisons with both a present and a missing feature.
FEATURE_CHOICES: dict[str, tuple[Any, ...]] = {
    "largest_credit_paise_24h": (None, 0, 500000, 1000000, 2000000),
    "pass_through_ratio_24h": (None, 0.0, 0.4, 0.8, 0.9),
    "median_credit_to_debit_seconds_7d": (None, 0, 1800, 3600, 7200),
    "account_age_days": (None, 0, 14, 30, 60),
    "distinct_payers_24h": (None, 0, 5, 10, 20, 40),
    "current_balance_paise": (None, 0, 5000, 10000, 50000),
    "new_device_r5": (None, 0, 1),
    "registry_match_count": (None, 0, 1, 2),
    "hops_to_complained_account": (None, 0, 1, 2, 3),
}

ROLE_CHOICES = ("genuine", "victim", "mule", "collector", "cash_out")
BUDGET_CHOICES = (0.0, 0.05, 0.5, 1.0, 5.0)


def _row(
    bank_id: str,
    account_id: str,
    day_index: int,
    role: str,
    depth: int | None,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Builds one validation snapshot row on 2026-01-01 plus day_index IST days."""
    return {
        "bank_id": bank_id,
        "account_id": account_id,
        "snapshot_time": datetime(2026, 1, 1 + day_index, 12, 0, tzinfo=IST),
        "snapshot_kind": "daily",
        "truth_role": role,
        "truth_depth": depth,
        **features,
    }


def _random_rows(rng: random.Random) -> list[dict[str, Any]]:
    """
    WHAT: Builds one random validation row set with missing features, mixed roles and repeats.

    The first row is forced to genuine so every case has a genuine-restriction rate, and rows are
    drawn with replacement over bank, account and day, so the same (account, IST day) pair also
    appears under two banks and the last-wins role map is exercised.
    """
    rows: list[dict[str, Any]] = []
    for position in range(rng.randint(10, 40)):
        role = "genuine" if position == 0 else rng.choice(ROLE_CHOICES)
        depth = rng.randint(1, 3) if role == "mule" else None
        features = {name: rng.choice(FEATURE_CHOICES[name]) for name in FEATURE_NAMES}
        rows.append(
            _row(
                bank_id=f"bank_{rng.randint(0, 1)}",
                account_id=f"account_{rng.randint(0, 7)}",
                day_index=rng.randint(0, 3),
                role=role,
                depth=depth,
                features=features,
            )
        )
    return rows


def _small_tuning(config: RulesConfig, tuning: TuningConfig) -> TuningConfig:
    """
    WHAT: Shrinks every candidate list to the untuned_start value plus its first alternative.

    The full committed grid is checked once on a fixture; the 120 random cases only need enough
    candidate values to make threshold changes matter, and a smaller grid keeps the reference loop
    fast. A threshold with a single candidate keeps it.
    """
    candidates: dict[str, tuple[int | float, ...]] = {}
    for key, values in tuning.candidates.items():
        untuned = config.thresholds[key].value
        alternatives = tuple(value for value in values if value != untuned)
        candidates[key] = (untuned, alternatives[0]) if alternatives else (untuned,)
    return TuningConfig(
        candidates=candidates,
        allowed_tuning_splits=tuning.allowed_tuning_splits,
        label=tuning.label,
        reason=tuning.reason,
        derived_from=tuning.derived_from,
    )


def _reference_per_rule_standalone(scored: list[Any], config: RulesConfig) -> tuple[RuleStandalone, ...]:
    """WHAT: The previous per-rule loop, kept only as the equivalence reference."""
    results: list[RuleStandalone] = []
    for rule in config.rules:
        restricted: dict[str, set[Any]] = {}
        for scored_day in scored:
            if rule.reason_code in scored_day.rules_fired:
                restricted.setdefault(scored_day.account_id, set()).add(scored_day.day)
        rate_pct, recall = _validation_metrics(scored, restricted)
        results.append(
            RuleStandalone(
                reason_code=rule.reason_code,
                validation_rate_pct=rate_pct,
                validation_recall=recall,
            )
        )
    return tuple(results)


def _reference_tune_rules(
    rows: list[dict[str, Any]],
    config: RulesConfig,
    tuning: TuningConfig,
    split: str,
    budget_pct: float,
    *,
    collect_grid_points: bool = False,
) -> RulesTuningResult:
    """
    WHAT: The previous per-combination tune_rules loop, kept only as the equivalence reference.

    It scores every grid point through the real score_account_days, rules_restricted_days and
    _validation_metrics helpers, chooses with the unchanged select_rules_grid_point, and measures
    the chosen point's rules individually, so it is the exact behaviour the vectorised tuner must
    reproduce.
    """
    _check_split(tuning, split)
    _check_budget(budget_pct)

    day_rows = one_row_per_account_day(rows)
    threshold_keys = list(config.thresholds)
    candidate_lists = [tuning.candidates[key] for key in threshold_keys]

    points: list[RulesGridPoint] = []
    for index, combination in enumerate(product(*candidate_lists)):
        new_values: dict[str, int | float] = dict(zip(threshold_keys, combination, strict=True))
        tuned_config = with_thresholds(config, new_values, tuning.derived_from)
        scored = score_account_days(day_rows, tuned_config, None)
        restricted = rules_restricted_days(scored)
        validation_rate_pct, validation_recall = _validation_metrics(scored, restricted)
        changed = sum(1 for key in threshold_keys if new_values[key] != config.thresholds[key].value)
        points.append(
            RulesGridPoint(
                thresholds=new_values,
                validation_rate_pct=validation_rate_pct,
                validation_recall=validation_recall,
                changed_count=changed,
                grid_index=index,
            )
        )

    chosen = select_rules_grid_point(points, budget_pct)
    chosen_config = with_thresholds(config, chosen.thresholds, tuning.derived_from)
    chosen_scored = score_account_days(day_rows, chosen_config, None)

    return RulesTuningResult(
        thresholds=dict(chosen.thresholds),
        validation_rate_pct=chosen.validation_rate_pct,
        validation_recall=chosen.validation_recall,
        meets_budget=chosen.validation_rate_pct <= budget_pct,
        combinations_evaluated=len(points),
        per_rule_standalone=_reference_per_rule_standalone(chosen_scored, chosen_config),
        grid_points=tuple(points) if collect_grid_points else (),
    )


def test_vectorised_grid_matches_the_old_loop_on_one_hundred_twenty_random_cases() -> None:
    """
    WHAT: Proves the vectorised tuner returns exactly the old loop's result on 120 seeded cases.

    Seeds are 5000..5119. Even seeds tune the baseline table, odd seeds the labels table, so the
    'any' rule_6 is exercised. Rows carry missing features, boundary values equal to the untuned
    thresholds, accounts that are genuine on some days and mules on others, the same account-day
    under two banks, and budgets no point can meet. The counters assert those scenarios really
    occurred, so the test cannot pass on a uniform run.
    """
    baseline_config = load_rules_config()
    baseline_tuning = _small_tuning(baseline_config, load_tuning_config())
    labels_config = load_rules_config_with_labels()
    labels_tuning = _small_tuning(labels_config, load_labels_tuning_config())

    saw_missing_feature = False
    saw_mixed_role = False
    saw_duplicate_account_day = False
    saw_budget_miss = False
    saw_collected_points = False
    any_rule_cases = 0

    for seed in range(5000, 5120):
        rng = random.Random(seed)
        rows = _random_rows(rng)
        if seed % 2 == 0:
            config, tuning = baseline_config, baseline_tuning
        else:
            config, tuning = labels_config, labels_tuning
            any_rule_cases += 1
        budget_pct = rng.choice(BUDGET_CHOICES)
        collect = rng.random() < 0.3

        expected = _reference_tune_rules(rows, config, tuning, "validation", budget_pct, collect_grid_points=collect)
        actual = tune_rules(rows, config, tuning, "validation", budget_pct, collect_grid_points=collect)
        assert actual == expected, f"seed {seed} budget {budget_pct}"

        if any(value is None for row in rows for name in FEATURE_NAMES for value in (row.get(name),)):
            saw_missing_feature = True
        roles_by_account: dict[str, set[str]] = {}
        for row in rows:
            roles_by_account.setdefault(row["account_id"], set()).add(row["truth_role"])
        if any(
            roles & {"genuine", "victim"} and roles & {"mule", "collector", "cash_out"}
            for roles in roles_by_account.values()
        ):
            saw_mixed_role = True
        pairs = {(row["account_id"], row["snapshot_time"].date()) for row in rows}
        if len(pairs) < len(rows):
            saw_duplicate_account_day = True
        if not actual.meets_budget:
            saw_budget_miss = True
        if collect:
            saw_collected_points = True

    assert any_rule_cases == 60
    assert saw_missing_feature
    assert saw_mixed_role
    assert saw_duplicate_account_day
    assert saw_budget_miss
    assert saw_collected_points


def test_committed_grids_match_the_old_loop_exactly() -> None:
    """
    WHAT: Proves both real committed grids (972 baseline, 2,916 with labels) match the old loop.

    The fixture mixes missing features, boundary values and mixed roles, and the comparison covers
    every number in the result including every collected grid point and every per-rule standalone.
    """
    rows = _random_rows(random.Random(777))
    cases = (
        (load_rules_config(), load_tuning_config(), 972),
        (load_rules_config_with_labels(), load_labels_tuning_config(), 2916),
    )

    for config, tuning, expected_combinations in cases:
        expected = _reference_tune_rules(rows, config, tuning, "validation", 0.5, collect_grid_points=True)
        actual = tune_rules(rows, config, tuning, "validation", 0.5, collect_grid_points=True)

        assert actual == expected
        assert actual.combinations_evaluated == expected_combinations
        assert len(actual.grid_points) == expected_combinations
        assert len(actual.per_rule_standalone) == len(config.rules)


def test_missing_feature_never_fires_an_all_or_an_any_rule() -> None:
    """
    WHAT: Proves a None feature cannot pass a comparison, for both 'all' rules and the 'any' rule.

    Ten genuine accounts have one payer each and no other feature; one mule has every feature
    missing. No point fires, so the rate and recall are both zero and every standalone is zero.
    If a missing value were treated as passing an ordered comparison, the mule's <= rules or the
    any rule_6 would fire and recall would be 1.0.
    """
    rows = [
        _row(
            "bank_a",
            f"genuine_{index}",
            0,
            "genuine",
            None,
            {name: (1 if name == "distinct_payers_24h" else None) for name in FEATURE_NAMES},
        )
        for index in range(10)
    ]
    rows.append(_row("bank_a", "mule_0", 0, "mule", 1, {name: None for name in FEATURE_NAMES}))
    config = load_rules_config_with_labels()

    result = tune_rules(rows, config, load_labels_tuning_config(), "validation", 0.5)

    assert result.validation_rate_pct == 0.0
    assert result.validation_recall == 0.0
    assert all(standalone.validation_rate_pct == 0.0 for standalone in result.per_rule_standalone)
    assert all(standalone.validation_recall == 0.0 for standalone in result.per_rule_standalone)


def test_full_grid_on_two_hundred_thousand_rows_is_well_under_120_seconds() -> None:
    """
    WHAT: Proves one committed 2,916-point grid over 200,000 account-days stays under 120 s.

    Twenty thousand accounts over ten days, one in fifty a mule, with random feature values that
    include missing ones. The assertion is the time guard; the measured seconds appear only on
    failure.
    """
    rng = random.Random(20261119)
    rows: list[dict[str, Any]] = []
    for account_index in range(20000):
        role = "mule" if account_index % 50 == 0 else "genuine"
        depth = 1 if role == "mule" else None
        for day_index in range(10):
            features = {name: rng.choice(FEATURE_CHOICES[name]) for name in FEATURE_NAMES}
            rows.append(
                _row(
                    bank_id="bank_0",
                    account_id=f"account_{account_index:05d}",
                    day_index=day_index,
                    role=role,
                    depth=depth,
                    features=features,
                )
            )

    config = load_rules_config_with_labels()
    tuning = load_labels_tuning_config()

    started = time.perf_counter()
    result = tune_rules(rows, config, tuning, "validation", 0.5)
    elapsed_seconds = time.perf_counter() - started

    assert len(rows) == 200000
    assert result.combinations_evaluated == 2916
    assert 0.0 <= result.validation_rate_pct <= 100.0
    assert elapsed_seconds < 120.0, f"vectorised grid took {elapsed_seconds:.2f}s"


def test_test_and_stress_splits_are_still_refused() -> None:
    """A split that is not declared for tuning raises before any row is read or scored."""
    rows = _random_rows(random.Random(31))
    tuning = _small_tuning(load_rules_config(), load_tuning_config())

    for split in ("test", "stress"):
        with pytest.raises(ValueError, match="is not allowed for tuning"):
            tune_rules(rows, load_rules_config(), tuning, split, 0.5)
