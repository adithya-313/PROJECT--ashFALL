"""
WHAT: Tests the staged full-run driver's resume rule, its fingerprint refusal, the learning-curve
doubling rule and the PU-method choice rule.

WHY: These four rules decide how much data is trained on, which model variant is compared and which
one reaches test, so each must be proven on hand numbers before the driver runs at full scale.
"""

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import evaluation.run_level_18_full as run_level_18_full
from decisions.mule_account_model import load_mule_model_config
from evaluation.level17_scoring import ScoredDay
from evaluation.run_level_18_full import (
    PU_METHODS,
    _choose_pu_method,
    _learning_curve_choice,
    _maturity_decision,
    _read_matching_stage,
    _run_stage,
    _tune_each_pu_method,
    _write_chosen_config,
)

FINGERPRINT: dict[str, Any] = {
    "code_version": "abc123",
    "mule_model_config_sha256": "00" * 32,
    "training_dataset_ids": ["training-a"],
    "validation_dataset_ids": ["validation-a"],
    "test_dataset_ids": ["test-a"],
    "training_world_counts": [40, 80, 160],
}


def test_resume_skips_finished_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A stage whose fingerprint matches is skipped with its recorded result and not recomputed."""
    calls: list[int] = []

    def compute() -> dict[str, Any]:
        """Returns a fixed result and counts its invocations."""
        calls.append(1)
        return {"value": 7}

    first = _run_stage("learning_curve", tmp_path, FINGERPRINT, compute)
    capsys.readouterr()
    second = _run_stage("learning_curve", tmp_path, FINGERPRINT, compute)
    output = capsys.readouterr().out.strip()

    assert calls == [1]
    assert first == {"value": 7}
    assert second == {"value": 7}
    assert output == f"SKIP stage learning_curve: fingerprint matches {tmp_path / 'learning_curve.json'}"


def test_changed_fingerprint_is_refused(tmp_path: Path) -> None:
    """A stage file written for other inputs is refused, naming the changed key, not reused."""
    _run_stage("tune_each_pu_method", tmp_path, FINGERPRINT, lambda: {"value": 7})
    changed = {**FINGERPRINT, "mule_model_config_sha256": "11" * 32}

    with pytest.raises(ValueError, match="mule_model_config_sha256"):
        _read_matching_stage(tmp_path / "tune_each_pu_method.json", "tune_each_pu_method", changed)

    with pytest.raises(ValueError, match="different inputs"):
        _run_stage("tune_each_pu_method", tmp_path, changed, lambda: {"value": 8})


TOLERANCE = 0.002


def test_maturity_decision_clear_winner_is_chosen_whichever_variant_it_is() -> None:
    """A gap far above the tolerance decides the contest for each variant in turn."""
    for winner in ("drop_28", "all_days", "all_days_trust_weighted"):
        scores = {"drop_28": 0.40, "all_days": 0.40, "all_days_trust_weighted": 0.40}
        scores[winner] = 0.45

        chosen, reason = _maturity_decision(scores, TOLERANCE)

        assert chosen == winner
        assert winner in reason


def test_maturity_decision_tie_within_tolerance_follows_the_preference_order() -> None:
    """All three within 0.002 of the best: the preferred all_days wins although it scored lowest."""
    scores = {"drop_28": 0.5010, "all_days": 0.5000, "all_days_trust_weighted": 0.5015}

    chosen, _ = _maturity_decision(scores, TOLERANCE)

    assert chosen == "all_days"


def test_maturity_decision_prefers_weighted_over_drop_when_only_those_two_tie() -> None:
    """drop_28 leads by 0.0015, inside the tolerance of weighted but far above all_days."""
    scores = {"drop_28": 0.5075, "all_days": 0.5000, "all_days_trust_weighted": 0.5060}

    chosen, _ = _maturity_decision(scores, TOLERANCE)

    assert chosen == "all_days_trust_weighted"


def test_maturity_decision_drop_28_wins_only_when_it_leads_both_others_by_more_than_the_tolerance() -> None:
    """Gaps of 0.004 and 0.005 are more than the tolerance, so the data-dropping variant is chosen."""
    scores = {"drop_28": 0.510, "all_days": 0.505, "all_days_trust_weighted": 0.506}

    chosen, _ = _maturity_decision(scores, TOLERANCE)

    assert chosen == "drop_28"


def test_maturity_decision_refuses_missing_variants_and_a_bad_tolerance() -> None:
    with pytest.raises(ValueError, match="must cover exactly"):
        _maturity_decision({"drop_28": 0.5, "all_days": 0.5}, TOLERANCE)
    with pytest.raises(ValueError, match="must be positive"):
        _maturity_decision({"drop_28": 0.5, "all_days": 0.5, "all_days_trust_weighted": 0.5}, 0.0)


def _maturity_record(chosen_variant: str) -> dict[str, Any]:
    """Returns a stage-1 record whose variants carry the three real window and weighting settings."""
    return {
        "chosen_variant": chosen_variant,
        "variants": [
            {"name": "drop_28", "label_maturity_days": 28, "label_trust_weighting": False},
            {"name": "all_days", "label_maturity_days": 0, "label_trust_weighting": False},
            {"name": "all_days_trust_weighted", "label_maturity_days": 0, "label_trust_weighting": True},
        ],
    }


@pytest.mark.parametrize(
    ("chosen_variant", "expected_days", "expected_weighting"),
    [("drop_28", 28, False), ("all_days", 0, False), ("all_days_trust_weighted", 0, True)],
)
def test_chosen_config_copy_carries_both_values_of_the_chosen_variant(
    tmp_path: Path, chosen_variant: str, expected_days: int, expected_weighting: bool
) -> None:
    source = Path(__file__).resolve().parents[3] / "config" / "mule_model.yaml"
    destination = tmp_path / "chosen_mule_model.yaml"

    _write_chosen_config(source, destination, _maturity_record(chosen_variant))

    chosen = load_mule_model_config(destination)
    assert chosen.label_maturity_days == expected_days
    assert chosen.label_trust_weighting is expected_weighting


def test_chosen_config_refuses_a_variant_missing_from_the_record(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[3] / "config" / "mule_model.yaml"

    with pytest.raises(ValueError, match="no variant named"):
        _write_chosen_config(source, tmp_path / "chosen.yaml", _maturity_record("nonsense"))


def test_learning_curve_flattens_at_80() -> None:
    """A second doubling that gains under 0.01 keeps the count before that doubling."""
    points = [
        {"training_world_count": 40, "validation_truth_average_precision": 0.50},
        {"training_world_count": 80, "validation_truth_average_precision": 0.52},
        {"training_world_count": 160, "validation_truth_average_precision": 0.524},
    ]

    chosen, more_needed, improvements = _learning_curve_choice(points)

    assert chosen == 80
    assert more_needed is False
    assert improvements == [
        {
            "from_training_worlds": 40,
            "to_training_worlds": 80,
            "average_precision_improvement": pytest.approx(0.02),
        },
        {
            "from_training_worlds": 80,
            "to_training_worlds": 160,
            "average_precision_improvement": pytest.approx(0.004),
        },
    ]


def test_learning_curve_still_rising_at_160_flags_more_worlds() -> None:
    """When no doubling gains under 0.01, the largest count is used and the flag is set."""
    points = [
        {"training_world_count": 40, "validation_truth_average_precision": 0.50},
        {"training_world_count": 80, "validation_truth_average_precision": 0.52},
        {"training_world_count": 160, "validation_truth_average_precision": 0.55},
    ]

    chosen, more_needed, improvements = _learning_curve_choice(points)

    assert chosen == 160
    assert more_needed is True
    assert [record["to_training_worlds"] for record in improvements] == [80, 160]


def test_learning_curve_ignores_non_doubling_pair() -> None:
    """A pair that does not double is not part of the rule, so a 2,4,6 count list reads 2->4 only."""
    points = [
        {"training_world_count": 2, "validation_truth_average_precision": 0.30},
        {"training_world_count": 4, "validation_truth_average_precision": 0.305},
        {"training_world_count": 6, "validation_truth_average_precision": 0.40},
    ]

    chosen, more_needed, improvements = _learning_curve_choice(points)

    assert chosen == 2
    assert more_needed is False
    assert improvements == [
        {
            "from_training_worlds": 2,
            "to_training_worlds": 4,
            "average_precision_improvement": pytest.approx(0.005),
        }
    ]


def test_learning_curve_needs_two_points() -> None:
    """One measured count cannot show a doubling, so the rule refuses it."""
    with pytest.raises(ValueError, match="at least two"):
        _learning_curve_choice([{"training_world_count": 40, "validation_truth_average_precision": 0.5}])


def _method_record(
    method: str,
    *,
    recall: float,
    average_precision: float,
    rate_pct: float,
    meets_budget: bool,
) -> dict[str, Any]:
    """Returns one stage-2 method record with only the fields the choice rule reads set meaningfully."""
    return {
        "pu_method": method,
        "validation_recall": recall,
        "validation_truth_average_precision": average_precision,
        "validation_rate_pct": rate_pct,
        "meets_budget": meets_budget,
    }


def test_pu_choice_prefers_recall_then_average_precision() -> None:
    """Among methods meeting the budget, the higher truth recall wins; AP breaks a recall tie."""
    records = [
        _method_record("none", recall=0.5, average_precision=0.5, rate_pct=0.4, meets_budget=True),
        _method_record("elkan_noto", recall=0.7, average_precision=0.6, rate_pct=0.4, meets_budget=True),
        _method_record("bagging", recall=0.7, average_precision=0.65, rate_pct=0.4, meets_budget=True),
    ]

    choice = _choose_pu_method(records)

    assert choice["pu_method"] == "bagging"
    assert choice["meets_budget"] is True
    assert choice["validation_recall"] == 0.7
    assert choice["validation_truth_average_precision"] == 0.65


def test_pu_choice_exact_tie_prefers_none() -> None:
    """An exact tie on recall and truth AP follows the fixed order none > elkan_noto > bagging."""
    records = [
        _method_record("bagging", recall=0.6, average_precision=0.5, rate_pct=0.4, meets_budget=True),
        _method_record("elkan_noto", recall=0.6, average_precision=0.5, rate_pct=0.4, meets_budget=True),
        _method_record("none", recall=0.6, average_precision=0.5, rate_pct=0.4, meets_budget=True),
    ]

    assert _choose_pu_method(records)["pu_method"] == "none"


def test_pu_choice_none_meets_budget_uses_lowest_rate() -> None:
    """With no method inside the budget, the lowest genuine-restriction rate is chosen and flagged."""
    records = [
        _method_record("none", recall=0.9, average_precision=0.9, rate_pct=1.2, meets_budget=False),
        _method_record("elkan_noto", recall=0.4, average_precision=0.4, rate_pct=0.7, meets_budget=False),
        _method_record("bagging", recall=0.4, average_precision=0.4, rate_pct=0.7, meets_budget=False),
    ]

    choice = _choose_pu_method(records)

    assert choice["pu_method"] == "elkan_noto"
    assert choice["meets_budget"] is False
    assert choice["validation_rate_pct"] == 0.7


def test_pu_choice_rejects_empty_records() -> None:
    """A choice without any measured method is refused rather than guessed."""
    with pytest.raises(ValueError, match="at least one method record"):
        _choose_pu_method([])


def test_maturity_decision_gap_of_exactly_the_tolerance_is_a_tie() -> None:
    """Binary-exact values: a lead of exactly the tolerance ties, a lead of more than it wins."""
    exact_gap = {"drop_28": 0.75, "all_days": 0.5, "all_days_trust_weighted": 0.5}
    larger_gap = {"drop_28": 0.8125, "all_days": 0.5, "all_days_trust_weighted": 0.5}

    exact_choice, _ = _maturity_decision(exact_gap, 0.25)
    larger_choice, _ = _maturity_decision(larger_gap, 0.25)

    assert exact_choice == "all_days"
    assert larger_choice == "drop_28"


class _FakeTuningHarness:
    """Replaces everything heavy in _tune_each_pu_method and counts the per-method trainings."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.trained_methods: list[str] = []
        scored_day = ScoredDay(
            bank_id="bank_0",
            account_id="acc_0",
            day=date(2026, 1, 1),
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role="mule",
            depth=None,
        )
        split = SimpleNamespace(
            rows=SimpleNamespace(matrix=[[0.0]], targets=[1], sample_weights=None),
            days=SimpleNamespace(aligned_features=[[0.0]]),
        )
        monkeypatch.setattr(run_level_18_full, "load_compact_split", lambda *args, **kwargs: split)
        monkeypatch.setattr(
            run_level_18_full, "_validation_scored_in_model_row_order", lambda *args: [scored_day]
        )
        monkeypatch.setattr(run_level_18_full, "_compact_row_keys", lambda rows: [])
        monkeypatch.setattr(run_level_18_full, "_train_pu_method", self._train_pu_method)
        monkeypatch.setattr(run_level_18_full, "validation_truth_average_precision", lambda *args: 0.5)
        monkeypatch.setattr(
            run_level_18_full,
            "tune_anomaly_threshold",
            lambda *args: SimpleNamespace(validation_rate_pct=1.0, validation_recall=0.25, meets_budget=True),
        )

    def _train_pu_method(self, method: str, *args: object) -> tuple[None, dict[str, Any], list[float], dict[str, Any]]:
        """Counts the call and returns one fixed probability with an empty search record."""
        self.trained_methods.append(method)
        return None, {"max_depth": 3}, [0.9], {"search": "off"}

    def run(self, stages_dir: Path, fingerprint: dict[str, Any]) -> dict[str, Any]:
        """Calls the real _tune_each_pu_method with placeholder arguments the fakes ignore."""
        return _tune_each_pu_method(
            [], ["world_0"], 1, [], Path("truth"), None, None, 1.0,  # type: ignore[arg-type]
            SimpleNamespace(hyperparameter_search="off"), 7, fingerprint, stages_dir,  # type: ignore[arg-type]
        )


def test_each_pu_method_is_checkpointed_and_a_second_call_trains_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first call trains and checkpoints every method; the second reuses all three records."""
    harness = _FakeTuningHarness(monkeypatch)

    first = harness.run(tmp_path, FINGERPRINT)
    first_output = capsys.readouterr().out
    second = harness.run(tmp_path, FINGERPRINT)
    second_output = capsys.readouterr().out

    assert harness.trained_methods == list(PU_METHODS)
    for method in PU_METHODS:
        assert (tmp_path / f"tune_method_{method}.json").is_file()
        assert f"DONE method {method}" in first_output
        assert f"SKIP method {method}: fingerprint matches" in second_output
    assert [record["pu_method"] for record in first["methods"]] == list(PU_METHODS)
    assert second == first


def test_deleting_one_checkpoint_retrains_only_that_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing checkpoint costs one method's training and nothing else."""
    harness = _FakeTuningHarness(monkeypatch)
    harness.run(tmp_path, FINGERPRINT)
    harness.trained_methods.clear()
    (tmp_path / f"tune_method_{PU_METHODS[1]}.json").unlink()

    harness.run(tmp_path, FINGERPRINT)

    assert harness.trained_methods == [PU_METHODS[1]]


def test_a_checkpoint_with_a_different_fingerprint_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint written for other inputs raises instead of being reused or retrained over."""
    harness = _FakeTuningHarness(monkeypatch)
    harness.run(tmp_path, FINGERPRINT)
    harness.trained_methods.clear()
    changed = {**FINGERPRINT, "mule_model_config_sha256": "11" * 32}

    with pytest.raises(ValueError, match="mule_model_config_sha256"):
        harness.run(tmp_path, changed)

    assert harness.trained_methods == []
