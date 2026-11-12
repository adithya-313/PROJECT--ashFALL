# File: decisions/rules_config.py
# Purpose: Configuration loader and schema validation for baseline heuristic detection rules.
# Issue: Level 17a Round 1 rules baseline config and decision table.
"""
WHAT:
    Defines immutable data containers and validation routines for the baseline
    heuristic decision rules table and associated threshold parameters.

WHY:
    Heuristic rule thresholds and table definitions must be strictly validated
    against canonical feature catalogs and configuration schemas at system initialization
    to prevent runtime evaluation failures, typo drift, or untracked policy shifts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from features.feature_log import canonical_feature_names

__all__ = [
    "ALLOWED_OPERATORS",
    "Rule",
    "RuleCondition",
    "RuleThreshold",
    "RulesConfig",
    "all_thresholds_tuned",
    "load_rules_config",
]

ALLOWED_OPERATORS: frozenset[str] = frozenset({">=", "<=", "=="})


@dataclass(frozen=True)
class RuleThreshold:
    """
    PURPOSE:
        Encapsulates a single threshold parameter with origin justification and tuning lifecycle state.

    ATTRIBUTES:
        key: Unique threshold identifier name.
        value: Numeric threshold magnitude (int or float).
        label: Decision governance label ('policy', 'sourced', etc.).
        reason: Plain-English operational justification for the threshold.
        derived_from: Source specification or architecture decision reference.
        status: Tuning lifecycle status ('untuned_start', 'tuned').
    """

    key: str
    value: int | float
    label: str
    reason: str
    derived_from: str
    status: str


@dataclass(frozen=True)
class RuleCondition:
    """
    PURPOSE:
        Represents a single predicate comparing an account feature against a threshold or literal.

    ATTRIBUTES:
        feature: Canonical feature identifier evaluated in this condition.
        op: Comparison operator ('>=', '<=', or '==').
        threshold: Configured threshold key or literal comparison value.
        threshold_key: Resolved string threshold key if referenced, or None for literals.
        threshold_value: Concrete numeric threshold value used during evaluation.
    """

    feature: str
    op: str
    threshold: str | int | float
    threshold_key: str | None
    threshold_value: int | float


@dataclass(frozen=True)
class Rule:
    """
    PURPOSE:
        Encapsulates a heuristic detection rule consisting of an identifier, reason code, and conditions.

    ATTRIBUTES:
        id: Unique rule identifier string (e.g. 'rule_1').
        reason_code: Machine-readable reason code (e.g. 'RB1_FAST_PASS_THROUGH').
        plain_english: Human-readable description of rule logic.
        conditions: Ordered tuple of predicates that must all hold for the rule to fire.
    """

    id: str
    reason_code: str
    plain_english: str
    conditions: tuple[RuleCondition, ...]


@dataclass(frozen=True)
class RulesConfig:
    """
    PURPOSE:
        Validated container holding baseline heuristic rules and threshold specifications.

    ATTRIBUTES:
        thresholds: Mapping of threshold keys to RuleThreshold instances.
        rules: Ordered tuple of validated Rule instances.
    """

    thresholds: dict[str, RuleThreshold]
    rules: tuple[Rule, ...]

    def all_thresholds_tuned(self) -> bool:
        """
        PURPOSE:
            Returns True if all configured thresholds have been tuned beyond untuned_start.

        RETURNS:
            bool: True if every threshold has status == 'tuned', False otherwise.
        """
        return all_thresholds_tuned(self)


def all_thresholds_tuned(config: RulesConfig | None = None) -> bool:
    """
    PURPOSE:
        Determines whether all thresholds in the rules configuration have completed tuning.

    INPUTS:
        config: Optional pre-loaded RulesConfig instance. If None, loads default config.

    KEY LOGIC:
        Iterates over all thresholds and checks if status is strictly 'tuned'.
        Returns False if any threshold remains in 'untuned_start' or any other non-tuned status.

    RETURNS:
        bool: True if all thresholds have status 'tuned', False otherwise.
    """
    if config is None:
        config = load_rules_config()

    if not config.thresholds:
        return False

    for threshold in config.thresholds.values():
        if threshold.status != "tuned":
            return False

    return True


def load_rules_config(path: Path | str | None = None) -> RulesConfig:
    """
    PURPOSE:
        Loads and validates the baseline rules decision table and thresholds from YAML configuration.

    INPUTS:
        path: Optional filesystem path to rules YAML file. Defaults to config/rules_baseline.yaml.

    KEY LOGIC:
        1. Reads YAML file and parses thresholds and rules blocks.
        2. Validates threshold structure: required fields (label, derived_from, status, value).
        3. Validates rule conditions: unknown features, unknown threshold keys, unknown operators.
        4. Validates rule integrity: duplicate rule ids, duplicate reason codes, rules with no conditions.
        5. Constructs immutable frozen dataclass instances.

    SIDE EFFECTS / ERRORS:
        Raises FileNotFoundError: If the target YAML file does not exist.
        Raises ValueError: On any validation error (unknown feature, unknown threshold key,
                           unknown op, duplicate rule id/reason code, empty conditions, missing threshold fields).

    CONSTRAINTS:
        Feature names are strictly verified against canonical_feature_names().
        Operators must be in ALLOWED_OPERATORS ('>=', '<=', '==').
    """
    if path is None:
        root_dir = Path(__file__).resolve().parent.parent
        resolved_path = root_dir / "config" / "rules_baseline.yaml"
    else:
        resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Rules configuration file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw_data: Any = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Invalid rules configuration in {resolved_path}; expected mapping at root.")

    raw_thresholds = raw_data.get("thresholds")
    if not isinstance(raw_thresholds, dict) or len(raw_thresholds) == 0:
        raise ValueError(f"Missing or invalid 'thresholds' block in {resolved_path}")

    parsed_thresholds: dict[str, RuleThreshold] = {}
    for key, item in raw_thresholds.items():
        if not isinstance(item, dict):
            raise ValueError(f"Threshold '{key}' must be a dictionary in {resolved_path}")

        # Check required fields: label, derived_from, status
        label = item.get("label")
        if not label or not isinstance(label, str) or not label.strip():
            raise ValueError(f"Threshold '{key}' missing or empty required field 'label'")

        derived_from = item.get("derived_from")
        if not derived_from or not isinstance(derived_from, str) or not derived_from.strip():
            raise ValueError(f"Threshold '{key}' missing or empty required field 'derived_from'")

        status = item.get("status")
        if not status or not isinstance(status, str) or not status.strip():
            raise ValueError(f"Threshold '{key}' missing or empty required field 'status'")

        if "value" not in item:
            raise ValueError(f"Threshold '{key}' missing required field 'value'")

        val = item["value"]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Threshold '{key}' value must be int or float; got {type(val).__name__}")

        reason = str(item.get("reason", ""))
        parsed_thresholds[key] = RuleThreshold(
            key=key,
            value=val,
            label=label.strip(),
            reason=reason.strip(),
            derived_from=derived_from.strip(),
            status=status.strip(),
        )

    raw_rules = raw_data.get("rules")
    if not isinstance(raw_rules, list) or len(raw_rules) == 0:
        raise ValueError(f"Missing or invalid 'rules' block in {resolved_path}")

    canonical_features = set(canonical_feature_names())
    seen_rule_ids: set[str] = set()
    seen_reason_codes: set[str] = set()
    parsed_rules: list[Rule] = []

    for rule_entry in raw_rules:
        if not isinstance(rule_entry, dict):
            raise ValueError(f"Rule entry must be a dictionary in {resolved_path}; got {rule_entry}")

        rule_id = rule_entry.get("id")
        if not rule_id or not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError(f"Rule entry missing required 'id' in {resolved_path}")
        rule_id = rule_id.strip()

        if rule_id in seen_rule_ids:
            raise ValueError(f"Duplicate rule id '{rule_id}' detected in {resolved_path}")
        seen_rule_ids.add(rule_id)

        reason_code = rule_entry.get("reason_code")
        if not reason_code or not isinstance(reason_code, str) or not reason_code.strip():
            raise ValueError(f"Rule '{rule_id}' missing required 'reason_code'")
        reason_code = reason_code.strip()

        if reason_code in seen_reason_codes:
            raise ValueError(f"Duplicate reason code '{reason_code}' detected in rule '{rule_id}'")
        seen_reason_codes.add(reason_code)

        plain_english = str(rule_entry.get("plain_english", "")).strip()

        raw_conditions = rule_entry.get("conditions")
        if not raw_conditions or not isinstance(raw_conditions, list) or len(raw_conditions) == 0:
            raise ValueError(f"Rule '{rule_id}' has no conditions configured")

        parsed_conditions: list[RuleCondition] = []
        for cond in raw_conditions:
            if not isinstance(cond, dict):
                raise ValueError(f"Condition in rule '{rule_id}' must be a dictionary; got {cond}")

            feature_name = cond.get("feature")
            if not feature_name or not isinstance(feature_name, str) or feature_name not in canonical_features:
                raise ValueError(f"Unknown feature name '{feature_name}' in rule '{rule_id}'")

            op = cond.get("op")
            if op not in ALLOWED_OPERATORS:
                raise ValueError(f"Unknown op '{op}' in rule '{rule_id}'; must be one of {sorted(ALLOWED_OPERATORS)}")

            if "threshold" not in cond:
                raise ValueError(f"Condition in rule '{rule_id}' missing 'threshold' field")

            threshold_ref = cond["threshold"]
            if isinstance(threshold_ref, str):
                if threshold_ref not in parsed_thresholds:
                    raise ValueError(f"Unknown threshold key '{threshold_ref}' in rule '{rule_id}'")
                t_key = threshold_ref
                t_val = parsed_thresholds[threshold_ref].value
            elif isinstance(threshold_ref, (int, float)) and not isinstance(threshold_ref, bool):
                t_key = None
                t_val = threshold_ref
            else:
                raise ValueError(f"Invalid threshold '{threshold_ref}' in rule '{rule_id}'; must be key or number")

            parsed_conditions.append(
                RuleCondition(
                    feature=feature_name,
                    op=op,
                    threshold=threshold_ref,
                    threshold_key=t_key,
                    threshold_value=t_val,
                )
            )

        parsed_rules.append(
            Rule(
                id=rule_id,
                reason_code=reason_code,
                plain_english=plain_english,
                conditions=tuple(parsed_conditions),
            )
        )

    return RulesConfig(
        thresholds=parsed_thresholds,
        rules=tuple(parsed_rules),
    )

