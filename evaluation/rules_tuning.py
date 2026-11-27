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

import numbers
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy

from core.config_validator import load_yaml
from core.event_clock import IST
from decisions.rules_config import Rule, RulesConfig, load_rules_config, load_rules_config_with_labels
from evaluation.level17_scoring import (
    GENUINE_ROLES,
    MULE_ROLES,
    AccountKey,
    ScoredDay,
    account_key,
    feature_values_only,
    genuine_restricted_days,
    ground_sets,
    one_row_per_account_day,
)

__all__ = [
    "AnomalyScoreCut",
    "AnomalyTuningResult",
    "RuleStandalone",
    "RulesDayMatrix",
    "RulesGridPoint",
    "RulesTuningResult",
    "TuningConfig",
    "build_rules_day_matrix",
    "load_labels_tuning_config",
    "load_tuning_config",
    "score_cut_curve",
    "select_anomaly_cut",
    "select_rules_grid_point",
    "tune_anomaly_threshold",
    "tune_rules",
    "tune_rules_matrix",
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
class RulesGridPoint:
    """
    WHAT: Validation numbers of one rules threshold combination from the tuning grid.

    ATTRIBUTES:
        thresholds: the combination's value for every baseline threshold key.
        validation_rate_pct: percentage of genuinely active accounts this point restricts.
        validation_recall: share of mule accounts this point detects.
        changed_count: how many thresholds differ from the config's untuned_start values.
        grid_index: position of the combination in the declared grid order.
    """

    thresholds: dict[str, int | float]
    validation_rate_pct: float
    validation_recall: float
    changed_count: int
    grid_index: int


@dataclass(frozen=True)
class AnomalyScoreCut:
    """
    WHAT: Validation numbers of one inclusive anomaly score cut.

    ATTRIBUTES:
        cut: inclusive anomaly score threshold.
        validation_rate_pct: percentage of genuinely active accounts the cut restricts.
        validation_recall: share of mule accounts the cut detects.
    """

    cut: float
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
        grid_points: every evaluated point, in declared order, when the caller asked to collect
            them; empty otherwise.
    """

    thresholds: dict[str, int | float]
    validation_rate_pct: float
    validation_recall: float
    meets_budget: bool
    combinations_evaluated: int
    per_rule_standalone: tuple[RuleStandalone, ...]
    grid_points: tuple[RulesGridPoint, ...] = ()


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


@dataclass(frozen=True)
class RulesDayMatrix:
    """
    WHAT: Account-day features and truth roles in flat numpy arrays, the tuner's vectorised input.

    WHY: Evaluating a 2,916-point grid must not rebuild a rule decision for every row and point.
    One column per feature the rules read, plus one role string per account-day, is all the grid
    needs.

    Attributes:
        features: Rule-referenced feature name -> (row_count,) float64 array; NaN means the snapshot
            carried None for that feature.
        account_ids: (row_count,) object array of account keys (world_id, bank_id, account_id), per
            account-day. The world is part of the key because account ids repeat across worlds.
        day_ordinals: (row_count,) int64 date.toordinal() of each account-day's IST date.
        roles: (row_count,) object array of truth_role strings.
    """

    features: dict[str, numpy.ndarray]
    account_ids: numpy.ndarray
    day_ordinals: numpy.ndarray
    roles: numpy.ndarray


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


def _load_extension_tuning_thresholds(extension_path: Path) -> dict[str, Any]:
    """
    WHAT: Reads and validates the thresholds block of an extension file's tuning grid.

    Args:
        extension_path: Rules extension YAML holding a tuning block.

    Returns:
        threshold key -> raw candidate block, exactly as declared in the extension file.

    Raises:
        FileNotFoundError: If the extension file does not exist.
        ValueError: If the tuning block or its thresholds mapping is missing or malformed, or a
            candidate list is not a non-empty mapping for that threshold.
    """
    if not extension_path.exists():
        raise FileNotFoundError(f"Rules extension file not found at: {extension_path}")

    raw = load_yaml(extension_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid rules extension in {extension_path}; expected mapping at root.")
    tuning_block = raw.get("tuning")
    if not isinstance(tuning_block, dict):
        raise ValueError(f"Rules extension {extension_path} requires a 'tuning' mapping")
    raw_thresholds = tuning_block.get("thresholds")
    if not isinstance(raw_thresholds, dict) or not raw_thresholds:
        raise ValueError(f"Rules extension {extension_path} requires a non-empty tuning 'thresholds' mapping")
    for key, block in raw_thresholds.items():
        if not isinstance(block, dict):
            raise ValueError(f"Tuning threshold '{key}' must be a mapping in {extension_path}")
    return raw_thresholds


def load_tuning_config(
    path: Path | str | None = None,
    *,
    rules_config: RulesConfig | None = None,
    extension_path: Path | str | None = None,
) -> TuningConfig:
    """
    WHAT: Loads and validates config/rules_tuning.yaml against the baseline thresholds.

    WHY: Every threshold's untuned_start value must be reachable by the search, and every declared
    threshold must exist in the baseline table, so a typo cannot silently tune nothing.

    Args:
        path: Optional tuning YAML path; defaults to config/rules_tuning.yaml in the repository.
        rules_config: Optional rules table whose threshold keys the grid must match; defaults to
            load_rules_config(). Pass the merged rules config to tune the label extension.
        extension_path: Optional rules extension YAML whose tuning block adds candidate lists for
            thresholds not present in the main tuning file. When None the main grid stands alone.

    Returns:
        A validated TuningConfig.

    Raises:
        FileNotFoundError: If a tuning or extension file does not exist.
        ValueError: On a malformed root, missing fields, empty or non-numeric candidates,
            duplicate candidates, a threshold missing from the rules config, an undeclared
            threshold, a candidate list without the untuned_start value, or a threshold declared by
            both the main grid and the extension grid.
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

    if extension_path is not None:
        resolved_extension_path = Path(extension_path)
        extension_thresholds = _load_extension_tuning_thresholds(resolved_extension_path)
        duplicate_keys = sorted(set(raw_thresholds) & set(extension_thresholds))
        if duplicate_keys:
            raise ValueError(
                f"Thresholds declared by both {resolved_path} and {resolved_extension_path}: {duplicate_keys}"
            )
        raw_thresholds = {**raw_thresholds, **extension_thresholds}

    baseline = rules_config if rules_config is not None else load_rules_config()
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


def load_labels_tuning_config(
    path: Path | str | None = None,
    extension_path: Path | str | None = None,
) -> TuningConfig:
    """
    WHAT: Returns the merged tuning grid for the rules table that includes rule_6.

    Args:
        path: Optional main tuning YAML path; defaults to config/rules_tuning.yaml.
        extension_path: Optional rules extension YAML path; defaults to
            config/rules_labels_extension.yaml.

    Returns:
        A TuningConfig whose candidate keys cover the baseline and extension thresholds, with the
        policy metadata taken from the main tuning file.

    Raises:
        FileNotFoundError: If a tuning or extension file does not exist.
        ValueError: On any validation error raised by load_tuning_config or the rules loaders.
    """
    if extension_path is None:
        root_dir = Path(__file__).resolve().parent.parent
        resolved_extension_path = root_dir / "config" / "rules_labels_extension.yaml"
    else:
        resolved_extension_path = Path(extension_path)
    return load_tuning_config(
        path,
        rules_config=load_rules_config_with_labels(extension_path=resolved_extension_path),
        extension_path=resolved_extension_path,
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
    restricted: dict[AccountKey, set[date]],
) -> tuple[float, float]:
    """
    WHAT: Computes the validation genuine-restriction rate and mule recall of one restriction set.

    Args:
        scored: Scored account-days defining roles and mule depth.
        restricted: account key -> restricted days produced by the point being measured.

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
        account
        for account, days in restricted.items()
        for day in days
        if is_mule(account, day)
    }
    validation_recall = len(detected) / len(mules) if mules else 0.0
    return validation_rate_pct, validation_recall


@dataclass(frozen=True)
class _MatrixIndex:
    """
    WHAT: Row-to-pair and row-to-account identity of a RulesDayMatrix, built once per tuner call.

    WHY: Every grid point restricts the same account-days and differs only in which rows are
    restricted, so how a row maps to its account and to its (account, IST day) pair never changes.
    Building that mapping once turns each point's rate and recall into two counting passes.

    Attributes:
        row_count: Number of account-day rows.
        pair_index: (row_count,) index of the (account, IST day) pair each row belongs to.
        pair_count: Number of distinct (account, IST day) pairs.
        pair_roles: (pair_count,) role of the pair's last row, matching ground_sets' last-wins map
            from (account, day) to role.
        pair_genuine: (pair_count,) True when the pair's last-wins role is genuine or victim.
        pair_mule: (pair_count,) True when the pair's last-wins role is a mule role.
        pair_account_index: (pair_count,) account index of each pair.
        account_count: Number of distinct accounts.
        genuine_total: Accounts with a genuine or victim day, the rate's denominator.
        mule_total: Accounts with a mule, collector or cash_out day, the recall's denominator.
    """

    row_count: int
    pair_index: numpy.ndarray
    pair_count: int
    pair_roles: tuple[str, ...]
    pair_genuine: numpy.ndarray
    pair_mule: numpy.ndarray
    pair_account_index: numpy.ndarray
    account_count: int
    genuine_total: int
    mule_total: int


def _rule_feature_names(config: RulesConfig) -> tuple[str, ...]:
    """
    WHAT: Returns every feature the rules read, in rule-table order, without duplicates.

    Args:
        config: Rules table to scan.

    Returns:
        Feature names in first-reference order.
    """
    names: list[str] = []
    seen: set[str] = set()
    for rule in config.rules:
        for condition in rule.conditions:
            if condition.feature not in seen:
                seen.add(condition.feature)
                names.append(condition.feature)
    return tuple(names)


def _matrix_index(matrix: RulesDayMatrix) -> _MatrixIndex:
    """
    WHAT: Builds the row-to-pair and row-to-account index of a rules day matrix.

    WHY: The old tuner recomputed ground_sets and the last-wins role map for every grid point.
    Both are properties of the data, not of the thresholds, so they are computed once here with
    exactly the definitions level17_scoring uses: pair roles are last-wins in row order, the mule
    and genuine account sets use every row's role, and each pair belongs to one account.

    Args:
        matrix: Account-day features and roles.

    Returns:
        The precomputed index.

    Raises:
        ValueError: If a role is not a string or an account id is not a (world, bank, account) key.

    Complexity:
        O(rows) time and O(rows) memory, with one Python pass over the rows.
    """
    row_count = int(matrix.account_ids.shape[0])
    row_roles: list[str] = []
    account_lookup: dict[AccountKey, int] = {}
    account_index_by_row = numpy.empty(row_count, dtype=numpy.int64)
    pair_lookup: dict[tuple[AccountKey, int], int] = {}
    pair_index = numpy.empty(row_count, dtype=numpy.int64)

    for row_index in range(row_count):
        account_value = matrix.account_ids[row_index]
        if not isinstance(account_value, tuple) or len(account_value) != 3:
            raise ValueError("RulesDayMatrix account_ids must be (world_id, bank_id, account_id) keys")
        account_index = account_lookup.get(account_value)
        if account_index is None:
            account_index = len(account_lookup)
            account_lookup[account_value] = account_index
        account_index_by_row[row_index] = account_index

        day_ordinal = int(matrix.day_ordinals[row_index])
        pair_key = (account_value, day_ordinal)
        pair = pair_lookup.get(pair_key)
        if pair is None:
            pair = len(pair_lookup)
            pair_lookup[pair_key] = pair
        pair_index[row_index] = pair

        role_value = matrix.roles[row_index]
        if not isinstance(role_value, str):
            raise ValueError("RulesDayMatrix roles must be strings")
        row_roles.append(role_value)

    account_count = len(account_lookup)
    pair_count = len(pair_lookup)
    pair_roles = [""] * pair_count
    pair_account_index = numpy.empty(pair_count, dtype=numpy.int64)
    for row_index in range(row_count):
        pair = int(pair_index[row_index])
        pair_roles[pair] = row_roles[row_index]
        pair_account_index[pair] = account_index_by_row[row_index]

    genuine_flags = numpy.zeros(account_count, dtype=bool)
    mule_flags = numpy.zeros(account_count, dtype=bool)
    for row_index, role in enumerate(row_roles):
        account_index = int(account_index_by_row[row_index])
        if role in GENUINE_ROLES:
            genuine_flags[account_index] = True
        if role in MULE_ROLES:
            mule_flags[account_index] = True

    pair_genuine = numpy.array([role in GENUINE_ROLES for role in pair_roles], dtype=bool)
    pair_mule = numpy.array([role in MULE_ROLES for role in pair_roles], dtype=bool)

    return _MatrixIndex(
        row_count=row_count,
        pair_index=pair_index,
        pair_count=pair_count,
        pair_roles=tuple(pair_roles),
        pair_genuine=pair_genuine,
        pair_mule=pair_mule,
        pair_account_index=pair_account_index,
        account_count=account_count,
        genuine_total=int(numpy.count_nonzero(genuine_flags)),
        mule_total=int(numpy.count_nonzero(mule_flags)),
    )


@dataclass(frozen=True)
class _RuleFiredGrid:
    """
    WHAT: One rule's fired account-day arrays for every combination of its referenced thresholds.

    WHY: A rule's conditions read only the thresholds named in its own conditions, so its fired
    array for a grid point depends on that small subset. Precomputing one array per subset value
    keeps the full grid a set of OR operations instead of a rule evaluation per point.

    Attributes:
        reason_code: Reason code recorded when the rule fires.
        positions: Positions in the full threshold-key list that this rule depends on.
        combo_lookup: Candidate-index tuple over positions -> index into fired_arrays.
        fired_arrays: Boolean (row_count,) arrays in combo_lookup order; True where the rule fires.
    """

    reason_code: str
    positions: tuple[int, ...]
    combo_lookup: dict[tuple[int, ...], int]
    fired_arrays: tuple[numpy.ndarray, ...]


def _rule_fired_grid(
    rule: Rule,
    threshold_keys: list[str],
    candidate_values: dict[str, tuple[int | float, ...]],
    features: dict[str, numpy.ndarray],
) -> _RuleFiredGrid:
    """
    WHAT: Precomputes one rule's fired array for every combination of its referenced candidates.

    WHY: evaluate_rules fires an 'all' rule only when every condition's comparison passes and an
    'any' rule when at least one present condition passes; a missing feature never passes. With
    NaN standing in for a missing feature, a NaN comparison is False, so AND over the condition
    comparisons reproduces 'all' (missing -> not evaluable -> not fired) and OR reproduces 'any'
    (missing -> skipped, all missing -> not fired) without a separate presence test.

    Args:
        rule: Rule from the tuned baseline config.
        threshold_keys: Every threshold key in grid order.
        candidate_values: Candidate values per threshold key.
        features: Rule-referenced feature columns, NaN for a missing value.

    Returns:
        The rule's precomputed arrays and their lookup.

    Raises:
        ValueError: If a condition operator or match mode is unsupported.

    Complexity:
        O(rule_combinations * rows) time and the same in memory, where rule_combinations is the
        product of the candidate counts of this rule's own thresholds.
    """
    referenced_keys = [
        key for key in threshold_keys if any(condition.threshold_key == key for condition in rule.conditions)
    ]
    positions = tuple(threshold_keys.index(key) for key in referenced_keys)
    value_lists = [candidate_values[key] for key in referenced_keys]
    combos = list(product(*[range(len(values)) for values in value_lists]))

    fired_arrays: list[numpy.ndarray] = []
    for combo in combos:
        passed_conditions: list[numpy.ndarray] = []
        for condition in rule.conditions:
            column = features[condition.feature]
            if condition.threshold_key is None:
                threshold: int | float = condition.threshold_value
            else:
                referenced_position = referenced_keys.index(condition.threshold_key)
                threshold = value_lists[referenced_position][combo[referenced_position]]
            if condition.op == ">=":
                passed = column >= threshold
            elif condition.op == "<=":
                passed = column <= threshold
            elif condition.op == "==":
                passed = column == threshold
            else:
                raise ValueError(f"Unsupported comparison operator '{condition.op}' in rule '{rule.id}'")
            passed_conditions.append(passed)
        if rule.match == "all":
            fired = numpy.logical_and.reduce(passed_conditions)
        elif rule.match == "any":
            fired = numpy.logical_or.reduce(passed_conditions)
        else:
            raise ValueError(f"Unsupported match value '{rule.match}' in rule '{rule.id}'")
        fired_arrays.append(fired)

    return _RuleFiredGrid(
        reason_code=rule.reason_code,
        positions=positions,
        combo_lookup={combo: index for index, combo in enumerate(combos)},
        fired_arrays=tuple(fired_arrays),
    )


def _measure_restricted_rows(restricted_rows: numpy.ndarray, index: _MatrixIndex) -> tuple[float, float]:
    """
    WHAT: Computes the validation rate and recall of one boolean account-day restriction array.

    WHY: Both metrics only ask whether an account was restricted on at least one day of a role, so
    they reduce to counting restricted pairs and then restricted accounts. The arithmetic is the
    same expression _validation_metrics uses, keeping the floats bit-identical.

    Args:
        restricted_rows: (row_count,) True where the point restricts that account-day.
        index: Precomputed row, pair and account identity.

    Returns:
        (validation_rate_pct, validation_recall): restricted genuine accounts over all genuinely
        active accounts, and detected mule accounts over all mule accounts (0.0 without mules).

    Raises:
        ValueError: If there are no genuinely active accounts.

    Complexity:
        O(rows) time with two bincount passes; memory O(rows + pairs + accounts).
    """
    if index.genuine_total == 0:
        raise ValueError("Cannot measure a genuine-restriction rate without genuine accounts")

    pair_counts = numpy.bincount(
        index.pair_index,
        weights=restricted_rows.astype(numpy.float64),
        minlength=index.pair_count,
    )
    restricted_pairs = pair_counts > 0.0

    genuine_account_hits = numpy.bincount(
        index.pair_account_index,
        weights=(restricted_pairs & index.pair_genuine).astype(numpy.float64),
        minlength=index.account_count,
    )
    genuine_restricted = int(numpy.count_nonzero(genuine_account_hits > 0.0))
    validation_rate_pct = 100.0 * genuine_restricted / index.genuine_total

    mule_account_hits = numpy.bincount(
        index.pair_account_index,
        weights=(restricted_pairs & index.pair_mule).astype(numpy.float64),
        minlength=index.account_count,
    )
    detected = int(numpy.count_nonzero(mule_account_hits > 0.0))
    validation_recall = detected / index.mule_total if index.mule_total else 0.0
    return validation_rate_pct, validation_recall


def _combine_and_measure(
    fired_arrays_by_rule: Sequence[Sequence[numpy.ndarray]],
    combo_indices: Sequence[int],
    index: _MatrixIndex,
) -> tuple[float, float]:
    """
    WHAT: ORs every rule's precomputed fired array for one grid point and measures the result.

    WHY: rules_restricted_days restricts an account-day exactly when at least one rule fired, so
    the point's restriction array is the OR of the rules' fired arrays; nothing else in the
    restriction depends on the thresholds.

    Args:
        fired_arrays_by_rule: Per rule, its fired arrays indexed by the rule's combo index.
        combo_indices: Per rule, the combo index this grid point selects.
        index: Precomputed row, pair and account identity.

    Returns:
        The point's (validation_rate_pct, validation_recall).
    """
    restricted_rows = numpy.zeros(index.row_count, dtype=bool)
    for fired_arrays, combo_index in zip(fired_arrays_by_rule, combo_indices, strict=True):
        numpy.logical_or(restricted_rows, fired_arrays[combo_index], out=restricted_rows)
    return _measure_restricted_rows(restricted_rows, index)


def build_rules_day_matrix(rows: list[dict[str, Any]], config: RulesConfig) -> RulesDayMatrix:
    """
    WHAT: Converts validation snapshot rows into the flat feature and role arrays the tuner reads.

    WHY: Scoring 2,916 threshold points must not re-deduplicate rows or rebuild feature mappings
    for every point. The rows are deduplicated exactly once with one_row_per_account_day, and only
    the features the rules actually read are kept.

    Args:
        rows: Validation snapshot rows, possibly with several snapshots per account-day.
        config: Rules table whose conditions name the features to keep.

    Returns:
        RulesDayMatrix whose rows are in one_row_per_account_day order.

    Raises:
        ValueError: If a snapshot is malformed (raised by one_row_per_account_day), a rule feature
            is neither a number nor None, truth_role is missing, truth_depth has the wrong type, or
            a mule day carries no truth_depth. The messages match score_account_days and
            ground_sets, whose validations these rows would otherwise reach later.

    Complexity:
        O(rows * rule features) time and O(rows * rule features) memory; the snapshot dicts are
        not retained.
    """
    day_rows = one_row_per_account_day(rows)
    feature_names = _rule_feature_names(config)
    features = {name: numpy.empty(len(day_rows), dtype=numpy.float64) for name in feature_names}
    account_ids = numpy.empty(len(day_rows), dtype=object)
    day_ordinals = numpy.empty(len(day_rows), dtype=numpy.int64)
    roles = numpy.empty(len(day_rows), dtype=object)

    for row_index, row in enumerate(day_rows):
        feature_values = feature_values_only(row)
        for name, column in features.items():
            value = feature_values.get(name)
            if value is None:
                column[row_index] = numpy.nan
            elif isinstance(value, numbers.Real):
                column[row_index] = float(value)
            else:
                raise ValueError(f"Rule feature '{name}' must be a number or None; got {value!r}")

        account_ids[row_index] = _row_account_key(row)
        day_ordinals[row_index] = row["snapshot_time"].astimezone(IST).date().toordinal()

        role = row.get("truth_role")
        if not isinstance(role, str) or not role:
            raise ValueError("score_account_days requires a non-empty string 'truth_role' on every row")
        roles[row_index] = role

        depth = row.get("truth_depth")
        if depth is not None and (not isinstance(depth, int) or isinstance(depth, bool)):
            raise ValueError(f"truth_depth must be an int or None; got {depth!r}")
        if role == "mule" and depth is None:
            raise ValueError(f"Mule account {account_ids[row_index]} has no truth_depth")

    return RulesDayMatrix(
        features=features,
        account_ids=account_ids,
        day_ordinals=day_ordinals,
        roles=roles,
    )


def _row_account_key(row: dict[str, Any]) -> AccountKey:
    """
    WHAT: Returns (world_id, bank_id, account_id) of one snapshot row.

    WHY: The tuning matrix counts accounts, and account ids repeat across worlds, so each row
    carries its full account identity.

    Raises:
        ValueError: If the row lacks a non-empty string world_id, bank_id or account_id.
    """
    parts: list[str] = []
    for name in ("world_id", "bank_id", "account_id"):
        value = row.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"build_rules_day_matrix requires a non-empty string '{name}' on every row")
        parts.append(value)
    return (parts[0], parts[1], parts[2])


def tune_rules_matrix(
    matrix: RulesDayMatrix,
    config: RulesConfig,
    tuning: TuningConfig,
    split: str,
    budget_pct: float,
    *,
    collect_grid_points: bool = False,
) -> RulesTuningResult:
    """
    WHAT: Searches the whole threshold grid on a prebuilt day matrix.

    WHY: The pilot loads and deduplicates its validation worlds once; this entry point lets every
    later tuning pass reuse those arrays instead of rebuilding them from snapshot dicts.

    Args:
        matrix: Account-day features and roles, as built by build_rules_day_matrix.
        config: Baseline rules config supplying the untuned_start values.
        tuning: Candidate grid and allowed splits.
        split: Dataset split; must be allowed for tuning.
        budget_pct: Maximum percentage of genuinely active accounts that may be restricted.
        collect_grid_points: When true, keep every evaluated point on the result.

    Returns:
        RulesTuningResult for the selected point, chosen with select_rules_grid_point exactly as
        the row-based tuner chooses it.

    Key logic:
        For every combination of candidate values, each rule's fired array for that point is ORed
        into one restriction array, and the rate and recall are counted from that array. The
        chosen point's per-rule numbers come from the same arrays.

    Raises:
        ValueError: If the split is not allowed, the budget is invalid, or the matrix has no
            genuinely active account.

    Complexity:
        O(grid_points * (rules * rows + rows)) time after the one-time per-rule arrays; memory is
        O(rows * sum of per-rule combinations) plus the collected grid points.
    """
    _check_split(tuning, split)
    _check_budget(budget_pct)

    index = _matrix_index(matrix)
    threshold_keys = list(config.thresholds)
    candidate_values = {key: tuning.candidates[key] for key in threshold_keys}
    rule_grids = [_rule_fired_grid(rule, threshold_keys, candidate_values, matrix.features) for rule in config.rules]

    points: list[RulesGridPoint] = []
    chosen_rule_indices: list[tuple[int, ...]] = []
    for grid_index, combination in enumerate(product(*[range(len(candidate_values[key])) for key in threshold_keys])):
        rule_indices = tuple(
            rule_grid.combo_lookup[tuple(combination[position] for position in rule_grid.positions)]
            for rule_grid in rule_grids
        )
        validation_rate_pct, validation_recall = _combine_and_measure(
            [rule_grid.fired_arrays for rule_grid in rule_grids],
            rule_indices,
            index,
        )
        new_values: dict[str, int | float] = {
            key: candidate_values[key][combination[position]] for position, key in enumerate(threshold_keys)
        }
        changed = sum(1 for key in threshold_keys if new_values[key] != config.thresholds[key].value)
        points.append(
            RulesGridPoint(
                thresholds=new_values,
                validation_rate_pct=validation_rate_pct,
                validation_recall=validation_recall,
                changed_count=changed,
                grid_index=grid_index,
            )
        )
        chosen_rule_indices.append(rule_indices)

    chosen = select_rules_grid_point(points, budget_pct)
    standalone: list[RuleStandalone] = []
    for rule, rule_grid, rule_index in zip(
        config.rules, rule_grids, chosen_rule_indices[chosen.grid_index], strict=True
    ):
        rate_pct, recall = _measure_restricted_rows(rule_grid.fired_arrays[rule_index], index)
        standalone.append(
            RuleStandalone(
                reason_code=rule.reason_code,
                validation_rate_pct=rate_pct,
                validation_recall=recall,
            )
        )

    return RulesTuningResult(
        thresholds=dict(chosen.thresholds),
        validation_rate_pct=chosen.validation_rate_pct,
        validation_recall=chosen.validation_recall,
        meets_budget=chosen.validation_rate_pct <= budget_pct,
        combinations_evaluated=len(points),
        per_rule_standalone=tuple(standalone),
        grid_points=tuple(points) if collect_grid_points else (),
    )


def select_rules_grid_point(points: Sequence[RulesGridPoint], budget_pct: float) -> RulesGridPoint:
    """
    WHAT: Picks the rules grid point the tuning rule chooses for one genuine-restriction budget.

    WHY: Operating points at several budgets must use exactly the ordering tune_rules applies, so
    both call this one function instead of repeating the comparison keys.

    Args:
        points: every evaluated grid point, in declared grid order.
        budget_pct: maximum percentage of genuinely active accounts that may be restricted.

    Returns:
        The point with the highest recall among budget-meeting points, then the lowest rate, then
        the fewest changed thresholds, then the earliest grid position. When no point meets the
        budget, the ordering starts from the lowest rate instead.

    Raises:
        ValueError: If the grid is empty.
    """
    _check_budget(budget_pct)

    best_meeting: tuple[tuple[float, float, int, int], RulesGridPoint] | None = None
    best_any: tuple[tuple[float, float, int, int], RulesGridPoint] | None = None
    for point in points:
        if point.validation_rate_pct <= budget_pct:
            meeting_key = (
                -point.validation_recall,
                point.validation_rate_pct,
                point.changed_count,
                point.grid_index,
            )
            if best_meeting is None or meeting_key < best_meeting[0]:
                best_meeting = (meeting_key, point)
        any_key = (point.validation_rate_pct, -point.validation_recall, point.changed_count, point.grid_index)
        if best_any is None or any_key < best_any[0]:
            best_any = (any_key, point)

    chosen = best_meeting if best_meeting is not None else best_any
    if chosen is None:
        raise ValueError("Tuning grid is empty; cannot select a threshold point")
    return chosen[1]


def tune_rules(
    rows: list[dict[str, Any]],
    config: RulesConfig,
    tuning: TuningConfig,
    split: str,
    budget_pct: float,
    *,
    collect_grid_points: bool = False,
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
        collect_grid_points: When true, keep every evaluated point on the result so operating
            points at other budgets can be selected from the same scoring pass.

    Returns:
        RulesTuningResult for the selected point. When no point meets the budget, the lowest-rate
        point is returned with meets_budget False.

    Key logic:
        The rows are reduced to a RulesDayMatrix once, then tune_rules_matrix evaluates every
        combination of candidate values in declared order and delegates the choice to
        select_rules_grid_point, so multi-budget reporting uses the same ordering.

    Raises:
        ValueError: If the split is not allowed, the budget is invalid, or the rows lack genuine
            account-days.
    """
    _check_split(tuning, split)
    _check_budget(budget_pct)
    matrix = build_rules_day_matrix(rows, config)
    return tune_rules_matrix(
        matrix,
        config,
        tuning,
        split,
        budget_pct,
        collect_grid_points=collect_grid_points,
    )


def _account_peak_scores(scored: list[ScoredDay]) -> tuple[dict[AccountKey, float], dict[AccountKey, float]]:
    """
    WHAT: Reduces scored account-days to the highest anomaly score each account reached on its
    genuine days and on its mule days.

    WHY: A cut restricts an account on a day when the day's score clears the cut, so an account is
    restricted on at least one genuine (or mule) day exactly when its best such day clears the cut.
    Collapsing each account-day to its last role and highest score keeps the day-level definition
    intact even if a caller passes two rows for the same account and day. An account is identified
    by its world, bank and account id: account ids repeat across worlds, and keying by the bare id
    merged different worlds' accounts and let one world's mule rows lift another's genuine peaks.

    Args:
        scored: Scored account-days; every row must carry an anomaly score.

    Returns:
        (genuine_peak, mule_peak), each mapping account key to its highest score on days whose last
        role for that account belongs to the matching role set. Accounts with no such day are
        absent.

    Raises:
        ValueError: If any row has no anomaly score, because a numeric cut cannot be applied to it.

    Complexity:
        O(rows) time and O(account-days) memory; each account-day is visited once.
    """
    day_best: dict[tuple[AccountKey, date], tuple[float, str]] = {}
    for scored_day in scored:
        if scored_day.anomaly_score is None:
            raise ValueError("anomaly_restricted_days requires anomaly scores on every row")
        key = (account_key(scored_day), scored_day.day)
        previous = day_best.get(key)
        if previous is None or scored_day.anomaly_score > previous[0]:
            day_best[key] = (scored_day.anomaly_score, scored_day.role)
        else:
            day_best[key] = (previous[0], scored_day.role)

    genuine_peak: dict[AccountKey, float] = {}
    mule_peak: dict[AccountKey, float] = {}
    for (account, _day), (score, role) in day_best.items():
        if role in GENUINE_ROLES:
            if score > genuine_peak.get(account, float("-inf")):
                genuine_peak[account] = score
        elif role in MULE_ROLES:
            if score > mule_peak.get(account, float("-inf")):
                mule_peak[account] = score
    return genuine_peak, mule_peak


def score_cut_curve(scored: list[ScoredDay]) -> tuple[AnomalyScoreCut, ...]:
    """
    WHAT: Measures every candidate anomaly cut once, in ascending cut order.

    WHY: Tuning one budget and reporting several budgets must share one sweep, so this is the only
    place that counts restricted accounts per cut.

    Args:
        scored: Scored account-days; every row must carry an anomaly score.

    Returns:
        One AnomalyScoreCut per distinct score seen on a genuine account-day, ascending by cut.

    Key logic:
        Candidate cuts are the distinct anomaly scores on genuine account-days. Each account's peak
        score on its genuine days (for the rate) and on its mule days (for recall) is computed once;
        bisect then counts how many peaks clear each cut.

    Complexity:
        O(rows + C log C + C log A) time for C cuts and A peaks; memory O(rows).

    Raises:
        ValueError: If there are no genuine account-days or a row has no anomaly score.
    """
    genuine_days = [scored_day for scored_day in scored if scored_day.role in GENUINE_ROLES]
    if not genuine_days:
        raise ValueError("Cannot tune an anomaly threshold without genuine account-days")

    candidate_thresholds: list[float] = []
    for scored_day in genuine_days:
        if scored_day.anomaly_score is None:
            raise ValueError("anomaly scores are required on every genuine account-day for tuning")
        candidate_thresholds.append(scored_day.anomaly_score)
    thresholds = sorted(set(candidate_thresholds))

    genuine_peak, mule_peak = _account_peak_scores(scored)
    sorted_genuine_peaks = sorted(genuine_peak.values())
    sorted_mule_peaks = sorted(mule_peak.values())
    genuine_total = len(sorted_genuine_peaks)
    mule_total = len(sorted_mule_peaks)
    if genuine_total == 0:
        raise ValueError("Cannot measure a genuine-restriction rate without genuine accounts")

    curve: list[AnomalyScoreCut] = []
    for threshold in thresholds:
        genuine_restricted = genuine_total - bisect_left(sorted_genuine_peaks, threshold)
        validation_rate_pct = 100.0 * genuine_restricted / genuine_total
        if mule_total:
            mule_detected = mule_total - bisect_left(sorted_mule_peaks, threshold)
            validation_recall = mule_detected / mule_total
        else:
            validation_recall = 0.0
        curve.append(
            AnomalyScoreCut(
                cut=threshold,
                validation_rate_pct=validation_rate_pct,
                validation_recall=validation_recall,
            )
        )
    return tuple(curve)


def select_anomaly_cut(curve: Sequence[AnomalyScoreCut], budget_pct: float) -> AnomalyTuningResult:
    """
    WHAT: Applies the tuning selection rule to an already measured score-cut curve.

    WHY: Several budgets can be reported from one sweep only if the selection order lives in one
    function that both the tuner and the multi-budget report call.

    Args:
        curve: cuts in ascending order, as returned by score_cut_curve.
        budget_pct: maximum percentage of genuinely active accounts that may be restricted.

    Returns:
        AnomalyTuningResult for the selected cut: highest recall, then lowest rate, then earliest
        candidate among budget-meeting cuts; otherwise lowest rate, then highest recall, then
        earliest candidate with meets_budget False.

    Raises:
        ValueError: If the curve is empty or the budget is invalid.
    """
    _check_budget(budget_pct)

    best_meeting: tuple[tuple[float, float, int], AnomalyScoreCut] | None = None
    best_any: tuple[tuple[float, float, int], AnomalyScoreCut] | None = None
    for index, point in enumerate(curve):
        if point.validation_rate_pct <= budget_pct:
            meeting_key = (-point.validation_recall, point.validation_rate_pct, index)
            if best_meeting is None or meeting_key < best_meeting[0]:
                best_meeting = (meeting_key, point)
        any_key = (point.validation_rate_pct, -point.validation_recall, index)
        if best_any is None or any_key < best_any[0]:
            best_any = (any_key, point)

    chosen = best_meeting if best_meeting is not None else best_any
    if chosen is None:
        raise ValueError("No anomaly score candidates available; cannot select a threshold")
    point = chosen[1]
    return AnomalyTuningResult(
        threshold=point.cut,
        validation_rate_pct=point.validation_rate_pct,
        validation_recall=point.validation_recall,
        meets_budget=point.validation_rate_pct <= budget_pct,
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
        Every cut's metrics come from one pass: the scores are collapsed to each account's peak on
        its genuine days and on its mule days, and a cut restricts an account exactly when that
        peak clears the cut. Counting with bisect over the sorted peaks keeps the sweep exact.

    Complexity:
        O(rows + C log C + C log A) time for C candidate cuts and A peak scores, instead of the old
        O(C * rows); memory is O(rows).

    Raises:
        ValueError: If the split is not allowed, the budget is invalid, there are no genuine
            account-days, or a row has no anomaly score.
    """
    _check_split(tuning, split)
    _check_budget(budget_pct)
    curve = score_cut_curve(scored)
    return select_anomaly_cut(curve, budget_pct)
