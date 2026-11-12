"""
WHAT: Unit tests for evaluation/rules_tuning.py, the validation-only threshold search.

WHY: Tuning decides the operating point of the whole rules baseline, so each test proves the
chosen point from hand-computed counts, that test and stress splits are refused, that the search
is deterministic over the full 972-point grid, and that a candidate grid missing its
untuned_start value is rejected.
"""

import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from core.event_clock import IST
from decisions.rules_config import load_rules_config
from evaluation.level17_scoring import ScoredDay
from evaluation.rules_tuning import (
    load_tuning_config,
    tune_anomaly_threshold,
    tune_rules,
)

TUNING_DAY = date(2025, 1, 1)
BASELINE_VALUES = {
    "large_credit_amount_paise": 1000000,
    "outflow_share": 0.8,
    "time_window_seconds": 3600,
    "new_account_age_days": 30,
    "payer_count": 10,
    "near_zero_balance_paise": 10000,
}


def _row(account_id: str, distinct_payers: int, role: str = "genuine", depth: int | None = None) -> dict[str, Any]:
    """Builds one validation snapshot row carrying only distinct_payers_24h as a feature."""
    return {
        "bank_id": "bank_a",
        "account_id": account_id,
        "snapshot_time": datetime(2025, 1, 1, 12, 0, tzinfo=IST),
        "snapshot_kind": "daily",
        "truth_role": role,
        "truth_depth": depth,
        "distinct_payers_24h": distinct_payers,
    }


def _validation_rows() -> list[dict[str, Any]]:
    """
    Builds 200 genuine accounts and 4 mules with hand-checkable payer counts:
    197 genuine at 1, one each at 10, 20 and 40; three mules at 40 and one at 5.
    """
    rows = [_row(f"genuine_{index:03d}", 1) for index in range(197)]
    rows.append(_row("genuine_10", 10))
    rows.append(_row("genuine_20", 20))
    rows.append(_row("genuine_40", 40))
    rows.append(_row("mule_40_a", 40, role="mule", depth=1))
    rows.append(_row("mule_40_b", 40, role="mule", depth=1))
    rows.append(_row("mule_40_c", 40, role="mule", depth=1))
    rows.append(_row("mule_05", 5, role="mule", depth=1))
    return rows


def _scored_day(account_id: str, anomaly_score: float, role: str) -> ScoredDay:
    """Builds one scored validation day; rules fields are unused by the anomaly tuner."""
    return ScoredDay(
        bank_id="bank_a",
        account_id=account_id,
        day=TUNING_DAY,
        rules_score=0,
        rules_fired=(),
        anomaly_score=anomaly_score,
        role=role,
        depth=1 if role == "mule" else None,
    )


# (a) The grid search chooses the payer_count that spends the 0.5% budget exactly.
def test_tune_rules_chooses_payer_count_40_for_the_budget() -> None:
    """
    At payer_count 40: 1 of 200 genuine is restricted (0.5%) and 3 of 4 mules are caught (0.75).
    At 10 or 20 the genuine rate exceeds the 0.5% budget, and at 5 recall would be higher but the
    price in genuine restrictions is higher too.
    """
    result = tune_rules(_validation_rows(), load_rules_config(), load_tuning_config(), "validation", 0.5)

    assert result.thresholds == {**BASELINE_VALUES, "payer_count": 40}
    assert result.validation_rate_pct == 0.5
    assert result.validation_recall == 0.75
    assert result.combinations_evaluated == 972
    assert result.meets_budget is True


# (b) When every point busts the budget, the lowest-rate point is reported as not meeting it.
def test_tune_rules_reports_lowest_rate_when_no_point_meets_budget() -> None:
    """Every account has 100 payers, so every grid point restricts 200 of 200 genuine accounts."""
    rows = [_row(f"genuine_{index:03d}", 100) for index in range(200)]
    rows.append(_row("mule_100", 100, role="mule", depth=1))

    result = tune_rules(rows, load_rules_config(), load_tuning_config(), "validation", 0.5)

    assert result.validation_rate_pct == 100.0
    assert result.validation_recall == 1.0
    assert result.meets_budget is False
    assert result.thresholds == BASELINE_VALUES


# (c) Test and stress splits are refused by both tuners.
def test_tuners_refuse_test_and_stress_splits() -> None:
    """A split that is not declared for tuning raises before any scoring happens."""
    tuning = load_tuning_config()
    rows = _validation_rows()
    scored = [_scored_day("genuine_1", 0.1, "genuine"), _scored_day("mule_1", 0.9, "mule")]

    for split in ("test", "stress"):
        with pytest.raises(ValueError, match="is not allowed for tuning"):
            tune_rules(rows, load_rules_config(), tuning, split, 0.5)
        with pytest.raises(ValueError, match="is not allowed for tuning"):
            tune_anomaly_threshold(scored, tuning, split, 0.5)


# (d) The full grid is deterministic and finishes well inside the time budget.
def test_tune_rules_is_deterministic_over_972_points_under_30_seconds() -> None:
    """Two runs over the same rows return identical results; elapsed time stays under 30 s."""
    rows = _validation_rows()
    config = load_rules_config()
    tuning = load_tuning_config()

    started = time.perf_counter()
    first = tune_rules(rows, config, tuning, "validation", 0.5)
    second = tune_rules(rows, config, tuning, "validation", 0.5)
    elapsed_seconds = time.perf_counter() - started

    assert first == second
    assert first.combinations_evaluated == 972
    assert elapsed_seconds < 30.0


# (e) The anomaly cut is chosen from the genuine score distribution at the matched rate.
def test_tune_anomaly_threshold_picks_0199() -> None:
    """
    200 genuine scores 0.000..0.199 and four mules at 0.5: the cut 0.199 restricts exactly one
    genuine account (0.5%) and catches all four mules; 0.198 would restrict two (1.0%).
    """
    scored = [_scored_day(f"genuine_{index:03d}", 0.001 * index, "genuine") for index in range(200)]
    scored.extend(_scored_day(f"mule_{index}", 0.5, "mule") for index in range(4))

    result = tune_anomaly_threshold(scored, load_tuning_config(), "validation", 0.5)

    assert result.threshold == pytest.approx(0.199)
    assert result.validation_rate_pct == 0.5
    assert result.validation_recall == 1.0
    assert result.meets_budget is True


# (f) A candidate grid that dropped the untuned_start value is rejected at load time.
def test_load_tuning_config_rejects_grid_without_untuned_start(tmp_path: Path) -> None:
    """payer_count candidates [20, 40] omit the baseline value 10 and must raise."""
    tuning_file = tmp_path / "rules_tuning.yaml"
    tuning_file.write_text(
        yaml.safe_dump(
            {
                "label": "policy",
                "reason": "test grid",
                "derived_from": "policy grid around untuned_start",
                "allowed_tuning_splits": ["validation"],
                "thresholds": {
                    "large_credit_amount_paise": {"candidates": [500000, 1000000, 2000000]},
                    "outflow_share": {"candidates": [0.6, 0.8, 0.9]},
                    "time_window_seconds": {"candidates": [1800, 3600, 7200]},
                    "new_account_age_days": {"candidates": [15, 30, 60]},
                    "payer_count": {"candidates": [20, 40]},
                    "near_zero_balance_paise": {"candidates": [5000, 10000, 20000]},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not include the untuned_start value"):
        load_tuning_config(tuning_file)
