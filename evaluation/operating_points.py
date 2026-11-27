"""
WHAT: Validation operating points and trade-off curves for the rules, anomaly and XGBoost scorers.

WHY: The model and the rules are compared only at matched genuine-restriction budgets. Reporting
every committed budget from one validation sweep keeps that comparison honest without re-scoring
the validation worlds once per budget, and the curves let the showcase show the whole trade-off
instead of a single point.

TERM: genuine-restriction rate: percentage of genuinely active accounts restricted on at least one
genuine day. TERM: recall: share of mule accounts detected.

All functions are pure except load_harm_budgets, which reads the committed evaluation config.
Missing or malformed data raises a plain ValueError.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from evaluation.level17_scoring import ScoredDay
from evaluation.rules_tuning import (
    RulesGridPoint,
    score_cut_curve,
    select_anomaly_cut,
    select_rules_grid_point,
)

__all__ = [
    "load_harm_budgets",
    "operating_points",
    "rules_curve",
    "score_curve",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_CONFIG = REPO_ROOT / "config" / "evaluation.yaml"


def _require_number(raw: Mapping[str, Any], key: str, path: Path) -> float:
    """
    WHAT: Returns a required numeric field from the fair_comparison block.

    Raises:
        ValueError: If the value is missing, is a bool, or is not a number.
    """
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} requires a numeric '{key}'; got {value!r}")
    return float(value)


def load_harm_budgets(path: Path | str | None = None) -> tuple[float, ...]:
    """
    WHAT: Reads the main and extra genuine-restriction budgets from the evaluation config.

    WHY: Every reported operating point must use committed budgets, so the numbers come from
    config/evaluation.yaml rather than from this module.

    Args:
        path: Optional evaluation YAML path; defaults to config/evaluation.yaml.

    Returns:
        (main budget, *extra budgets) in the order the config declares them.

    Raises:
        ValueError: If the file, fair_comparison block, main budget or extra budget list is missing
            or malformed.
    """
    target = DEFAULT_EVALUATION_CONFIG if path is None else Path(path)
    raw = load_yaml(target)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{target} must hold a mapping at the root")
    fair_comparison = raw.get("fair_comparison")
    if not isinstance(fair_comparison, Mapping):
        raise ValueError(f"{target} requires a 'fair_comparison' mapping")

    main_budget = _require_number(fair_comparison, "main_genuine_restriction_rate_pct", target)
    extras_raw = fair_comparison.get("extra_genuine_restriction_rate_pct")
    if not isinstance(extras_raw, list):
        raise ValueError(f"{target} requires a list 'extra_genuine_restriction_rate_pct'")
    extra_budgets: list[float] = []
    for value in extras_raw:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{target} extra budget must be numeric; got {value!r}")
        extra_budgets.append(float(value))
    return (main_budget, *extra_budgets)


def score_curve(scored: list[ScoredDay]) -> list[dict[str, float]]:
    """
    WHAT: Returns every candidate anomaly cut with its validation rate and recall.

    Args:
        scored: Validation account-days carrying anomaly scores.

    Returns:
        One dict per distinct cut, ascending: {"cut", "rate_pct", "recall"}. The numbers come from
        score_cut_curve, the same sweep the tuner uses.
    """
    curve = score_cut_curve(scored)
    return [
        {
            "cut": point.cut,
            "rate_pct": point.validation_rate_pct,
            "recall": point.validation_recall,
        }
        for point in curve
    ]


def _pareto_front(points: Sequence[tuple[float, float]]) -> list[dict[str, float]]:
    """
    WHAT: Keeps the (rate, recall) pairs that no other pair beats on both axes.

    Args:
        points: (rate_pct, recall) pairs.

    Returns:
        The front in ascending rate order, as dicts {"rate_pct", "recall"}. A pair stays only when
        no other pair has lower-or-equal rate and higher-or-equal recall with at least one strict;
        identical pairs are kept once.

    Complexity:
        O(P log P) time for P points: sort by rate then descending recall and sweep the running
        best recall.
    """
    ordered = sorted(set(points), key=lambda pair: (pair[0], -pair[1]))
    front: list[dict[str, float]] = []
    best_recall = float("-inf")
    for rate_pct, recall in ordered:
        if recall > best_recall:
            front.append({"rate_pct": rate_pct, "recall": recall})
            best_recall = recall
    return front


def rules_curve(grid_points: Sequence[RulesGridPoint]) -> dict[str, Any]:
    """
    WHAT: Returns the validation rate and recall of every evaluated rules grid point plus its
    Pareto front.

    Args:
        grid_points: points collected by tune_rules(..., collect_grid_points=True).

    Returns:
        {"points": [...], "pareto_front": [...]} where each point is {"rate_pct", "recall"} in
        declared grid order and the front uses the no-domination rule from _pareto_front.
    """
    points = [
        {"rate_pct": point.validation_rate_pct, "recall": point.validation_recall} for point in grid_points
    ]
    pairs = [(point["rate_pct"], point["recall"]) for point in points]
    return {"points": points, "pareto_front": _pareto_front(pairs)}


def operating_points(
    *,
    rules_grid_points: Sequence[RulesGridPoint],
    anomaly_scored: list[ScoredDay],
    xgboost_scored: list[ScoredDay],
    run_manifest_id: str,
    evaluation_config_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    WHAT: Reports the tuned validation point of each scorer at every committed harm budget.

    WHY: The comparison must hold harm equal at more than the main budget, and every extra point
    must come from the same validation sweep and the same selection rule as the main point.

    Args:
        rules_grid_points: rules points collected by one tune_rules scoring pass.
        anomaly_scored: validation account-days scored by the detector.
        xgboost_scored: validation account-days scored by the XGBoost model.
        run_manifest_id: result run key written next to every point.
        evaluation_config_path: optional evaluation YAML with the budgets.

    Returns:
        {"simulated": True, "run_manifest_id", "budgets_pct": [...], "points": [...]} where each
        point holds one budget and, per scorer, its cut or thresholds, validation rate, recall and
        meets_budget.

    Raises:
        ValueError: If the config budgets or the scored days are malformed, or a curve is empty.
    """
    budgets = load_harm_budgets(evaluation_config_path)
    anomaly_curve = score_cut_curve(anomaly_scored)
    xgboost_curve = score_cut_curve(xgboost_scored)

    points: list[dict[str, Any]] = []
    for budget_pct in budgets:
        rules_point = select_rules_grid_point(rules_grid_points, budget_pct)
        anomaly_point = select_anomaly_cut(anomaly_curve, budget_pct)
        xgboost_point = select_anomaly_cut(xgboost_curve, budget_pct)
        points.append(
            {
                "budget_pct": budget_pct,
                "scorers": {
                    "rules": {
                        "thresholds": dict(rules_point.thresholds),
                        "validation_rate_pct": rules_point.validation_rate_pct,
                        "validation_recall": rules_point.validation_recall,
                        "meets_budget": rules_point.validation_rate_pct <= budget_pct,
                    },
                    "anomaly": {
                        "threshold": anomaly_point.threshold,
                        "validation_rate_pct": anomaly_point.validation_rate_pct,
                        "validation_recall": anomaly_point.validation_recall,
                        "meets_budget": anomaly_point.meets_budget,
                    },
                    "xgboost": {
                        "threshold": xgboost_point.threshold,
                        "validation_rate_pct": xgboost_point.validation_rate_pct,
                        "validation_recall": xgboost_point.validation_recall,
                        "meets_budget": xgboost_point.meets_budget,
                    },
                },
            }
        )

    return {
        "simulated": True,
        "run_manifest_id": run_manifest_id,
        "budgets_pct": list(budgets),
        "points": points,
    }
