"""
WHAT: Joint threshold tuning for the rules decision table and for the anomaly detector, using
validation worlds only.

WHY: The rules baseline ships with untuned_start thresholds. Before it can be compared with the
model at a matched genuine-restriction rate, every threshold must be chosen from a pre-declared
grid on validation worlds, never on test or stress worlds. The search therefore enumerates the
whole grid, rejects any point that restricts more genuine accounts than the budget allows,
and reports the point with the highest mule recall at that budget.

TERM: validation rate: percentage of genuinely active accounts (genuine or victim) restricted on
at least one genuine day.
TERM: validation recall: share of mule accounts (mule, collector, cash_out) with at least one
restricted mule day.

Every function is pure apart from load_tuning_config, which reads one YAML file. Missing or
malformed data raises a plain ValueError instead of being guessed.
"""

from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from decisions.rules_config import RulesConfig, load_rules_config, with_thresholds
from evaluation.level17_scoring import (
    GENUINE_ROLES,
    ScoredDay,
    anomaly_restricted_days,
    genuine_restricted_days,
    ground_sets,
    one_row_per_account_day,
    rules_restricted_days,
    score_account_days,
)

__all__ = [
    "AnomalyTuningResult",
    "RuleStandalone",
    "RulesTuningResult",
    "TuningConfig",
    "load_tuning_config",
    "tune_anomaly_threshold",
    "tune_rules",
]


_ScoredPoint = tuple[
    tuple[float, float, int, int],
    dict[str, int | float],
    float,
    float,
    list[ScoredDay],
]


@dataclass(frozen=True)
class TuningConfig:
    """
    WHAT: Validated candidate grids and the dataset splits they may be searched on.

    ATTRIBUTES:
        candidates: threshold key -> ordered candidate values, each list containing the
            baseline untuned_start value.
        allowed_tuning_splits: dataset splits the tuners may run on; must contain 'validation'.
        label: Decision governance label of the grid.
        reason: Plain-English justification of the grid.
        derived_from: Source the grid was derived from.
    """

    candidates: dict[str, tuple[int | float, ...]]
    allowed_tuning_splits: tuple[str, ...]
    label: str
    reason: str
    derived_from: str


@dataclass(frozen=True)
class RuleStandalone:
    """
    WHAT: Validation performance of one rule firing on its own.

    ATTRIBUTES:
        reason_code: Reason code of the rule.
        validation_rate_pct: Percentage of genuinely active accounts this rule alone restricts.
        validation_recall: Share of mule accounts this rule alone detects.
    """

    reason_code: str
    validation_rate_pct: float
    validation_recall: float


@dataclass(frozen=True)
class RulesTuningResult:
    """
    WHAT: Outcome of the joint rules-threshold search on validation worlds.

    ATTRIBUTES:
        thresholds: chosen value for every baseline threshold key.
        validation_rate_pct: percentage of genuinely active accounts the chosen point restricts.
        validation_recall: share of mule accounts the chosen point detects.
        meets_budget: True when validation_rate_pct is within the genuine-restriction budget.
        combinations_evaluated: number of grid points scored.
        per_rule_standalone: each rule's rate and recall on its own at the chosen thresholds.
    """

    thresholds: dict[str, int | float]
    validation_rate_pct: float
    validation_recall: float
    meets_budget: bool
    combinations_evaluated: int
    per_rule_standalone: tuple[RuleStandalone, ...]


@dataclass(frozen=True)
class AnomalyTuningResult:
    """
    WHAT: Outcome of the anomaly-threshold search on validation worlds.

    ATTRIBUTES:
        threshold: chosen inclusive anomaly score cut.
        validation_rate_pct: percentage of genuinely active accounts the cut restricts.
        validation_recall: share of mule accounts the cut detects.
        meets_budget: True when validation_rate_pct is within the genuine-restriction budget.
    """

    threshold: float
    validation_rate_pct: float
    validation_recall: float
    meets_budget: bool


def _require_non_empty_string(raw: dict[str, Any], key: str, path: Path) -> str:
    """
    WHAT: Returns a required non-empty string field from the tuning YAML root.

    Args:
        raw: Parsed tuning YAML mapping.
        key: Field name.
        path: File path used in the error message.

    Raises:
        ValueError: If the field is missing, empty, or not a string.
    """
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Tuning config {path} requires a non-empty string '{key}'")
    return value.strip()


def _parse_candidates(
    raw: dict[str, Any], key: str, path: Path, baseline_value: int | float
) -> tuple[int | float, ...]:
    """
    WHAT: Parses one threshold's candidate list and proves it contains the untuned_start value.

    Args:
        raw: Parsed tuning YAML mapping.
        key: Threshold key.
        path: File path used in error messages.
        baseline_value: The untuned_start value from the baseline rules config.

    Returns:
        The candidate values as a tuple, in declared order.

    Raises:
        ValueError: If the block is not a mapping, the list is missing or empty, a candidate is
            not a number, candidates repeat, or the untuned_start value is absent.
    """
    block = raw.get(key)
    if not isinstance(block, dict):
        raise ValueError(f"Tuning threshold '{key}' must be a mapping in {path}")
    candidates = block.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Tuning threshold '{key}' requires a non-empty 'candidates' list in {path}")
    parsed: list[int | float] = []
    for candidate in candidates:
        if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
            raise ValueError(f"Tuning threshold '{key}' candidate {candidate!r} is not a number in {path}")
        parsed.append(candidate)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Tuning threshold '{key}' has duplicate candidates in {path}")
    if baseline_value not in parsed:
        raise ValueError(
            f"Tuning threshold '{key}' candidates in {path} do not include the untuned_start value {baseline_value!r}"
        )
    return tuple(parsed)


def load_tuning_config(path: Path | str | None = None) -> TuningConfig:
    """
    WHAT: Loads and validates config/rules_tuning.yaml against the baseline thresholds.

    WHY: Every threshold's untuned_start value must be reachable by the search, and every declared
    threshold must exist in the baseline table, so a typo cannot silently tune nothing.

    Args:
        path: Optional tuning YAML path; defaults to config/rules_tuning.yaml in the repository.

    Returns:
        A validated TuningConfig.

    Raises:
        FileNotFoundError: If the tuning file does not exist.
        ValueError: On a malformed root, missing fields, empty or non-numeric candidates,
            duplicate candidates, a threshold missing from the baseline, an undeclared baseline
            threshold, or a candidate list without the untuned_start value.
    """
    if path is None:
        root_dir = Path(__file__).resolve().parent.parent
        resolved_path = root_dir / "config" / "rules_tuning.yaml"
    else:
        resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Tuning configuration file not found at: {resolved_path}")

    raw = load_yaml(resolved_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid tuning configuration in {resolved_path}; expected mapping at root.")

    label = _require_non_empty_string(raw, "label", resolved_path)
    derived_from = _require_non_empty_string(raw, "derived_from", resolved_path)
    reason = _require_non_empty_string(raw, "reason", resolved_path)

    raw_splits = raw.get("allowed_tuning_splits")
    if not isinstance(raw_splits, list) or not raw_splits:
        raise ValueError(f"Tuning config {resolved_path} requires a non-empty 'allowed_tuning_splits' list")
    splits: list[str] = []
    for split in raw_splits:
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"Tuning config {resolved_path} has an invalid split {split!r}")
        splits.append(split.strip())

    raw_thresholds = raw.get("thresholds")
    if not isinstance(raw_thresholds, dict) or not raw_thresholds:
        raise ValueError(f"Tuning config {resolved_path} requires a non-empty 'thresholds' mapping")

    baseline = load_rules_config()
    baseline_keys = set(baseline.thresholds)
    declared_keys = set(raw_thresholds)
    if declared_keys != baseline_keys:
        missing = sorted(baseline_keys - declared_keys)
        unknown = sorted(declared_keys - baseline_keys)
        raise ValueError(
            f"Tuning thresholds in {resolved_path} must match the baseline keys exactly; "
            f"missing={missing}, unknown={unknown}"
        )

    candidates: dict[str, tuple[int | float, ...]] = {}
    for key, threshold in baseline.thresholds.items():
        candidates[key] = _parse_candidates(raw_thresholds, key, resolved_path, threshold.value)

    return TuningConfig(
        candidates=candidates,
        allowed_tuning_splits=tuple(splits),
        label=label,
        reason=reason,
        derived_from=derived_from,
    )


def _check_split(tuning: TuningConfig, split: str) -> None:
    """
    WHAT: Refuses any split that is not declared for tuning.

    Args:
        tuning: Loaded tuning config.
        split: Dataset split being tuned on.

    Raises:
        ValueError: If the split is not in allowed_tuning_splits.
    """
    if split not in tuning.allowed_tuning_splits:
        raise ValueError(
            f"Split {split!r} is not allowed for tuning; allowed splits are {list(tuning.allowed_tuning_splits)}"
        )


def _check_budget(budget_pct: float) -> None:
    """
    WHAT: Refuses a genuine-restriction budget outside 0..100 percent.

    Args:
        budget_pct: Maximum allowed percentage of genuinely active accounts that may be restricted.

    Raises:
        ValueError: If budget_pct is outside 0..100 or is not a number.
    """
    if not isinstance(budget_pct, (int, float)) or isinstance(budget_pct, bool) or not 0.0 <= budget_pct <= 100.0:
        raise ValueError(f"Genuine-restriction budget must be a percentage between 0 and 100; got {budget_pct!r}")


def _validation_metrics(
    scored: list[ScoredDay],
    restricted: dict[str, set[date]],
) -> tuple[float, float]:
    """
    WHAT: Computes the validation genuine-restriction rate and mule recall of one restriction set.

    Args:
        scored: Scored account-days defining roles and mule depth.
        restricted: account -> restricted days produced by the point being measured.

    Returns:
        (validation_rate_pct, validation_recall). The rate counts genuinely active accounts
        restricted on at least one genuine day over all genuinely active accounts, the same
        definition the evaluation reports. Recall is 0.0 when there are no mules.

    Raises:
        ValueError: If any restricted day has no scored row, or there are no genuinely active accounts.
    """
    mules, genuine_active, _, is_mule = ground_sets(scored)
    if not genuine_active:
        raise ValueError("Cannot measure a genuine-restriction rate without genuine accounts")
    genuine_restricted = genuine_restricted_days(restricted, scored)
    validation_rate_pct = 100.0 * len(genuine_restricted) / len(genuine_active)
    detected = {
        account_id
        for account_id, days in restricted.items()
        for day in days
        if is_mule(account_id, day)
    }
    validation_recall = len(detected) / len(mules) if mules else 0.0
    return validation_rate_pct, validation_recall


def _per_rule_standalone(scored: list[ScoredDay], config: RulesConfig) -> tuple[RuleStandalone, ...]:
    """
    WHAT: Measures each rule on its own at the chosen thresholds.

    Args:
        scored: Account-days already scored with the chosen thresholds.
        config: Chosen rules config, whose rule order defines the result order.

    Returns:
        One RuleStandalone per rule in table order.
    """
    results: list[RuleStandalone] = []
    for rule in config.rules:
        restricted: dict[str, set[date]] = {}
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


def tune_rules(
    rows: list[dict[str, Any]],
    config: RulesConfig,
    tuning: TuningConfig,
    split: str,
    budget_pct: float,
) -> RulesTuningResult:
    """
    WHAT: Searches the whole threshold grid on validation account-days and returns the best point
    that respects the genuine-restriction budget.

    WHY: Thresholds must be chosen jointly, because rules share features, and the comparison with
    the model is only fair when both restrict the same share of genuinely active accounts.

    Args:
        rows: Snapshot rows from validation worlds; deduplicated to one row per account-day.
        config: Baseline rules config supplying the untuned_start values.
        tuning: Candidate grid and allowed splits.
        split: Dataset split; must be allowed for tuning.
        budget_pct: Maximum percentage of genuinely active accounts that may be restricted.

    Returns:
        RulesTuningResult for the selected point. When no point meets the budget, the lowest-rate
        point is returned with meets_budget False.

    Key logic:
        Every combination of candidate values is evaluated in declared order. Selection prefers,
        among budget-meeting points, highest recall, then lowest rate, then fewest thresholds
        changed from untuned_start, then the earliest combination. Without a budget-meeting point
        the same ordering starts from the lowest rate.

    Raises:
        ValueError: If the split is not allowed, the budget is invalid, or the rows lack genuine
            account-days.
    """
    _check_split(tuning, split)
    _check_budget(budget_pct)

    day_rows = one_row_per_account_day(rows)
    threshold_keys = list(config.thresholds)
    candidate_lists = [tuning.candidates[key] for key in threshold_keys]

    combinations_evaluated = 0
    best_meeting: _ScoredPoint | None = None
    best_any: _ScoredPoint | None = None

    for index, combination in enumerate(product(*candidate_lists)):
        new_values: dict[str, int | float] = dict(zip(threshold_keys, combination, strict=True))
        tuned_config = with_thresholds(config, new_values, tuning.derived_from)
        scored = score_account_days(day_rows, tuned_config, None)
        restricted = rules_restricted_days(scored)
        validation_rate_pct, validation_recall = _validation_metrics(scored, restricted)
        changed = sum(1 for key in threshold_keys if new_values[key] != config.thresholds[key].value)
        combinations_evaluated += 1

        if validation_rate_pct <= budget_pct:
            meeting_key = (-validation_recall, validation_rate_pct, changed, index)
            if best_meeting is None or meeting_key < best_meeting[0]:
                best_meeting = (meeting_key, new_values, validation_rate_pct, validation_recall, scored)
        any_key = (validation_rate_pct, -validation_recall, changed, index)
        if best_any is None or any_key < best_any[0]:
            best_any = (any_key, new_values, validation_rate_pct, validation_recall, scored)

    chosen = best_meeting if best_meeting is not None else best_any
    if chosen is None:
        raise ValueError("Tuning grid is empty; cannot select a threshold point")
    _, chosen_values, validation_rate_pct, validation_recall, chosen_scored = chosen
    chosen_config = with_thresholds(config, chosen_values, tuning.derived_from)

    return RulesTuningResult(
        thresholds=dict(chosen_values),
        validation_rate_pct=validation_rate_pct,
        validation_recall=validation_recall,
        meets_budget=validation_rate_pct <= budget_pct,
        combinations_evaluated=combinations_evaluated,
        per_rule_standalone=_per_rule_standalone(chosen_scored, chosen_config),
    )


def tune_anomaly_threshold(
    scored: list[ScoredDay],
    tuning: TuningConfig,
    split: str,
    budget_pct: float,
) -> AnomalyTuningResult:
    """
    WHAT: Chooses the anomaly score cut that spends the genuine-restriction budget with the
    highest mule recall on validation account-days.

    WHY: The model and the rules must be compared at matched operating points, so the anomaly cut
    is tuned the same way the rules thresholds are.

    Args:
        scored: Validation account-days already carrying anomaly scores.
        tuning: Loaded tuning config supplying the allowed splits.
        split: Dataset split; must be allowed for tuning.
        budget_pct: Maximum percentage of genuinely active accounts that may be restricted.

    Returns:
        AnomalyTuningResult for the selected cut.

    Key logic:
        Candidate cuts are the distinct anomaly scores seen on genuine account-days, so the chosen
        point uses the budget rather than an arbitrary stricter cut. Selection prefers, among
        budget-meeting cuts, highest recall, then lowest rate, then the earliest candidate.
        Without a budget-meeting cut, the lowest-rate cut is returned with meets_budget False.

    Raises:
        ValueError: If the split is not allowed, the budget is invalid, there are no genuine
            account-days, or a row has no anomaly score.
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
