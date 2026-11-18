# File: decisions/rules_engine.py
# Purpose: Pure functional rules decision engine evaluating account features against baseline heuristic policies.
# Issue: Level 17a Round 1 rules decision table engine with reason codes.
"""
WHAT:
    Evaluates account feature maps against configured heuristic rule definitions,
    producing an immutable RulesDecision record detailing fired rules, reason codes,
    un-evaluable rules, and composite risk score.

WHY:
    Heuristic rule evaluation must remain deterministic, side-effect free, and mathematically
    exact. Missing or None feature values must never be silently coerced to zero or assumed
    to satisfy conditions; instead, they explicitly render affected rules un-evaluable.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from decisions.rules_config import RulesConfig, load_rules_config

__all__ = [
    "RulesDecision",
    "evaluate_rules",
]


@dataclass(frozen=True)
class RulesDecision:
    """
    PURPOSE:
        Immutable evaluation outcome of the heuristic rules decision table.

    ATTRIBUTES:
        fired: Ordered tuple of (rule_id, reason_code) pairs for all triggered rules.
        not_evaluable: Ordered tuple of rule_id strings for rules where at least one condition
                       could not be evaluated due to missing or None feature values.
        score: Integer composite risk score equal to the count of fired rules (policy: unweighted).
    """

    fired: tuple[tuple[str, str], ...]
    not_evaluable: tuple[str, ...]
    score: int


def _condition_passes(feature_value: int | float, threshold_value: int | float, op: str, rule_id: str) -> bool:
    """
    PURPOSE:
        Evaluates one condition's comparison, with the boundary value included.

    INPUTS:
        feature_value: Present, non-None feature value from the account.
        threshold_value: Resolved numeric threshold from the condition.
        op: Comparison operator, one of '>=', '<=', '=='.
        rule_id: Rule identifier used in the error message.

    RETURNS:
        bool: True when the comparison holds.

    SIDE EFFECTS / ERRORS:
        Raises ValueError naming the rule if the operator is not supported.
    """
    if op == ">=":
        return feature_value >= threshold_value
    if op == "<=":
        return feature_value <= threshold_value
    if op == "==":
        return feature_value == threshold_value
    raise ValueError(f"Unsupported comparison operator '{op}' in rule '{rule_id}'")


def evaluate_rules(
    features: Mapping[str, float | int | None],
    config: RulesConfig | None = None,
) -> RulesDecision:
    """
    PURPOSE:
        Evaluates a set of account features against the configured heuristic decision table.

    INPUTS:
        features: Mapping of canonical feature names to numeric values or None.
        config: Optional pre-loaded RulesConfig instance. If None, default rules config is loaded.

    KEY LOGIC:
        1. Iterates through configured rules in exact table definition order.
        2. For a rule with match 'all': if any condition's feature is missing or None, the rule is
           marked not_evaluable and immediately skipped. Otherwise the rule fires only when ALL
           conditions pass.
        3. For a rule with match 'any': only conditions whose feature is present and not None take
           part. The rule fires when at least one of those passes, is not_evaluable only when every
           condition's feature is missing or None, and otherwise does not fire.
        4. Conditions use configured comparison operators ('>=', '<=', '=='), including the boundary.
        5. Fired (id, reason_code) pairs are appended in table order.
        6. Computes score as integer count of fired rules (policy: rules have equal unit weights).

    RETURNS:
        RulesDecision: Frozen dataclass with fired pairs, not_evaluable ids, and composite score.

    CONSTRAINTS:
        Pure function: zero I/O, deterministic output given identical inputs.
        None features NEVER count as passing any comparison.
    """
    if config is None:
        config = load_rules_config()

    fired_list: list[tuple[str, str]] = []
    not_evaluable_list: list[str] = []

    for rule in config.rules:
        if rule.match == "all":
            # Every condition's feature must be present and non-None for the rule to be evaluable
            is_evaluable = True
            for cond in rule.conditions:
                val = features.get(cond.feature)
                if val is None:
                    is_evaluable = False
                    break

            if not is_evaluable:
                not_evaluable_list.append(rule.id)
                continue

            all_passed = True
            for cond in rule.conditions:
                feature_val = features[cond.feature]
                assert feature_val is not None  # Guaranteed by is_evaluable check

                if not _condition_passes(feature_val, cond.threshold_value, cond.op, rule.id):
                    all_passed = False
                    break

            if all_passed:
                fired_list.append((rule.id, rule.reason_code))
        elif rule.match == "any":
            # Only conditions with a present, non-None feature take part in an OR rule
            evaluated_any = False
            passed_any = False
            for cond in rule.conditions:
                feature_val = features.get(cond.feature)
                if feature_val is None:
                    continue
                evaluated_any = True
                if _condition_passes(feature_val, cond.threshold_value, cond.op, rule.id):
                    passed_any = True
                    break

            if not evaluated_any:
                not_evaluable_list.append(rule.id)
            elif passed_any:
                fired_list.append((rule.id, rule.reason_code))
        else:
            raise ValueError(f"Unsupported match value '{rule.match}' in rule '{rule.id}'")

    return RulesDecision(
        fired=tuple(fired_list),
        not_evaluable=tuple(not_evaluable_list),
        score=len(fired_list),
    )

