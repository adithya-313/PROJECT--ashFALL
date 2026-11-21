"""
WHAT: Tests for evaluation/operating_points.py and the grid-point collection it relies on.

WHY: Several harm budgets are reported from one validation sweep, so the tests prove the Pareto
front rule on a hand-built example, that budgets really come from the evaluation config, that the
score curve never improves as the cut gets stricter, and that collecting every rules grid point
leaves the tuned point unchanged.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from core.config_validator import load_yaml
from core.event_clock import IST
from decisions.rules_config import load_rules_config
from evaluation.level17_scoring import ScoredDay
from evaluation.operating_points import load_harm_budgets, operating_points, rules_curve, score_curve
from evaluation.rules_tuning import RulesGridPoint, load_tuning_config, tune_rules

REPO_ROOT = Path(__file__).resolve().parents[3]
TUNING_DAY = date(2025, 1, 1)


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
    Builds the same hand-checkable rows as the tuning tests: 200 genuine accounts, three of them
    at payer counts 10, 20 and 40, plus four mules.
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


def _scored_day(account_id: str, score: float, role: str) -> ScoredDay:
    """Builds one scored validation day; rules fields are unused by the score curve."""
    return ScoredDay(
        world_id="world_a",
        bank_id="bank_a",
        account_id=account_id,
        day=TUNING_DAY,
        rules_score=0,
        rules_fired=(),
        anomaly_score=score,
        role=role,
        depth=1 if role == "mule" else None,
    )


def _scored_days() -> list[ScoredDay]:
    """Builds four genuine accounts at scores 0.1, 0.1, 0.5, 0.9 and one mule at 0.8."""
    return [
        _scored_day("g1", 0.9, "genuine"),
        _scored_day("g2", 0.5, "genuine"),
        _scored_day("g3", 0.1, "genuine"),
        _scored_day("g4", 0.1, "genuine"),
        _scored_day("m1", 0.8, "mule"),
    ]


def _grid_point(rate_pct: float, recall: float, index: int) -> RulesGridPoint:
    """Builds one rules grid point with a marker threshold so the chosen point is identifiable."""
    return RulesGridPoint(
        thresholds={"payer_count": index},
        validation_rate_pct=rate_pct,
        validation_recall=recall,
        changed_count=0,
        grid_index=index,
    )


def test_pareto_front_keeps_only_undominated_pairs() -> None:
    """
    WHAT: Proves the front rule on a hand-built grid.

    (2, 0.4) and (3, 0.6) are dominated by (2, 0.6); (0, 0), (1, 0.5), (2, 0.6) and (4, 0.9) are
    not dominated by any other point.
    """
    points = [
        _grid_point(0.0, 0.0, 0),
        _grid_point(1.0, 0.5, 1),
        _grid_point(2.0, 0.4, 2),
        _grid_point(2.0, 0.6, 3),
        _grid_point(3.0, 0.6, 4),
        _grid_point(4.0, 0.9, 5),
    ]

    curve = rules_curve(points)

    assert len(curve["points"]) == 6
    assert curve["pareto_front"] == [
        {"rate_pct": 0.0, "recall": 0.0},
        {"rate_pct": 1.0, "recall": 0.5},
        {"rate_pct": 2.0, "recall": 0.6},
        {"rate_pct": 4.0, "recall": 0.9},
    ]


def test_default_budgets_come_from_the_committed_evaluation_config() -> None:
    """WHAT: Proves load_harm_budgets reads the main and extra budgets from config/evaluation.yaml."""
    raw = load_yaml(REPO_ROOT / "config" / "evaluation.yaml")
    fair_comparison = raw["fair_comparison"]

    budgets = load_harm_budgets()

    assert budgets == (
        float(fair_comparison["main_genuine_restriction_rate_pct"]),
        *[float(value) for value in fair_comparison["extra_genuine_restriction_rate_pct"]],
    )


def test_operating_points_follow_the_budgets_in_the_config(tmp_path: Path) -> None:
    """
    WHAT: Proves the reported budgets change when the config changes.

    The temporary config declares 0, 40 and 60 percent. At 0 no point meets the budget, so the
    lowest-rate rules point is chosen with meets_budget False; at 60 the highest-recall point that
    stays inside the budget is chosen. The anomaly curve at 60 picks the 0.5 cut, which restricts
    50 percent of genuine accounts and catches the mule.
    """
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "fair_comparison": {
                    "main_genuine_restriction_rate_pct": 0,
                    "extra_genuine_restriction_rate_pct": [40, 60],
                }
            }
        ),
        encoding="utf-8",
    )
    points = [
        _grid_point(10.0, 0.2, 0),
        _grid_point(50.0, 0.8, 1),
        _grid_point(90.0, 0.9, 2),
    ]

    result = operating_points(
        rules_grid_points=points,
        anomaly_scored=_scored_days(),
        xgboost_scored=_scored_days(),
        run_manifest_id="run_test",
        evaluation_config_path=config_path,
    )

    assert result["simulated"] is True
    assert result["run_manifest_id"] == "run_test"
    assert result["budgets_pct"] == [0.0, 40.0, 60.0]
    assert [point["budget_pct"] for point in result["points"]] == [0.0, 40.0, 60.0]

    no_budget, forty, sixty = result["points"]
    assert no_budget["scorers"]["rules"]["thresholds"] == {"payer_count": 0}
    assert no_budget["scorers"]["rules"]["meets_budget"] is False
    assert forty["scorers"]["rules"]["thresholds"] == {"payer_count": 0}
    assert forty["scorers"]["rules"]["meets_budget"] is True
    assert sixty["scorers"]["rules"]["thresholds"] == {"payer_count": 1}
    assert sixty["scorers"]["rules"]["validation_recall"] == 0.8
    assert sixty["scorers"]["anomaly"]["threshold"] == 0.5
    assert sixty["scorers"]["anomaly"]["validation_rate_pct"] == 50.0
    assert sixty["scorers"]["anomaly"]["validation_recall"] == 1.0
    assert sixty["scorers"]["anomaly"]["meets_budget"] is True


def test_score_curve_rates_never_decrease_as_the_cut_falls() -> None:
    """
    WHAT: Proves the score curve is monotone.

    Every distinct cut appears once in ascending order, and walking from the strictest cut to the
    loosest one the restriction rate never falls, because lowering a cut can only add accounts.
    """
    curve = score_curve(_scored_days())

    assert [point["cut"] for point in curve] == [0.1, 0.5, 0.9]
    for stricter, looser in zip(curve[1:], curve[:-1], strict=True):
        assert stricter["rate_pct"] <= looser["rate_pct"]
        assert stricter["recall"] <= looser["recall"]


def test_collecting_grid_points_leaves_the_tuned_point_unchanged() -> None:
    """
    WHAT: Proves collect_grid_points only adds data.

    Two tune_rules calls on the same rows return the same thresholds, rate, recall and verdict;
    only the collected points differ (972 grid points versus none).
    """
    config = load_rules_config()
    tuning = load_tuning_config()
    plain = tune_rules(_validation_rows(), config, tuning, "validation", 0.5)
    collected = tune_rules(
        _validation_rows(),
        config,
        tuning,
        "validation",
        0.5,
        collect_grid_points=True,
    )

    assert collected.thresholds == plain.thresholds
    assert collected.validation_rate_pct == plain.validation_rate_pct
    assert collected.validation_recall == plain.validation_recall
    assert collected.meets_budget == plain.meets_budget
    assert len(collected.grid_points) == plain.combinations_evaluated == 972
    assert plain.grid_points == ()
