"""
WHAT: Unit tests for the pilot's main-budget success criterion.

WHY: The brief allows the model to be called better than the rules only when both scorers met the
validation harm budget and XGBoost strictly beats rules on both test median recall and test median
yield. A tie must fail, and a validation budget miss must make the comparison not judgeable even
when the model looks better on test.
"""

from typing import Any

from evaluation.run_level_18_pilot import _success_criterion


def _metric(median: float) -> dict[str, Any]:
    """Returns a headline-metric summary block carrying a median, as the summariser writes it."""
    return {"world_count": 3, "median": median, "unit": "fraction"}


def _test_across(
    rules_recall: float,
    rules_yield: float,
    xgboost_recall: float,
    xgboost_yield: float,
) -> dict[str, Any]:
    """Returns an across_worlds["test"] block with the headline metrics the criterion reads."""
    return {
        "rules": {
            "recall_at_capacity": _metric(rules_recall),
            "yield_measured": _metric(rules_yield),
            "genuine_restriction_rate": _metric(0.5),
        },
        "xgboost": {
            "recall_at_capacity": _metric(xgboost_recall),
            "yield_measured": _metric(xgboost_yield),
            "genuine_restriction_rate": _metric(0.6),
        },
    }


def _validation(rules_rate: float, xgboost_rate: float, both_meet: bool) -> dict[str, Any]:
    """Returns the main-budget per-scorer validation block from the operating points."""
    return {
        "rules": {
            "validation_rate_pct": rules_rate,
            "validation_recall": 0.4,
            "meets_budget": both_meet,
        },
        "xgboost": {
            "validation_rate_pct": xgboost_rate,
            "validation_recall": 0.6,
            "meets_budget": both_meet,
        },
    }


def test_pass_requires_strictly_higher_recall_and_yield() -> None:
    """Both budgets met and both test medians strictly higher gives status pass."""
    block = _success_criterion(
        _validation(0.5, 0.5, both_meet=True),
        _test_across(0.4, 500.0, 0.7, 900.0),
    )

    assert block["status"] == "pass"
    assert block["rules_test_median_recall_at_capacity"] == 0.4
    assert block["xgboost_test_median_recall_at_capacity"] == 0.7
    assert block["rules_test_median_yield_measured"] == 500.0
    assert block["xgboost_test_median_yield_measured"] == 900.0
    assert block["rules_test_genuine_restriction_rate"] == 0.5
    assert block["xgboost_test_genuine_restriction_rate"] == 0.6


def test_equal_recall_never_passes_even_with_higher_yield() -> None:
    """A tie on median recall fails even when the model's median yield is higher."""
    block = _success_criterion(
        _validation(0.5, 0.5, both_meet=True),
        _test_across(0.6, 500.0, 0.6, 900.0),
    )

    assert block["status"] == "fail"
    assert block["rules_test_median_recall_at_capacity"] == 0.6
    assert block["xgboost_test_median_recall_at_capacity"] == 0.6


def test_equal_yield_never_passes_even_with_higher_recall() -> None:
    """A tie on median yield fails even when the model's median recall is higher."""
    block = _success_criterion(
        _validation(0.5, 0.5, both_meet=True),
        _test_across(0.4, 900.0, 0.7, 900.0),
    )

    assert block["status"] == "fail"


def test_validation_budget_miss_is_not_judgeable_even_when_the_model_wins_on_test() -> None:
    """A budget miss makes the comparison not judgeable and names the missing validation rate."""
    block = _success_criterion(
        _validation(0.9, 0.5, both_meet=False),
        _test_across(0.4, 500.0, 0.7, 900.0),
    )

    assert block["status"] == "not_judgeable"
    assert "0.9" in block["reason"]
    assert block["rules_test_median_recall_at_capacity"] == 0.4
    assert block["xgboost_test_median_recall_at_capacity"] == 0.7
