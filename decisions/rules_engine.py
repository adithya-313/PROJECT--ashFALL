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


def evaluate_rules(
    features: Mapping[str, float | int | None],
    config: RulesConfig | None = None,
) -> RulesDecision:
    """
    PURPOSE:
        Evaluates a set of account features against the baseline heuristic decision table.

    INPUTS:
        features: Mapping of canonical feature names to numeric values or None.
        config: Optional pre-loaded RulesConfig instance. If None, default rules config is loaded.

    KEY LOGIC:
        1. Iterates through configured rules in exact table definition order.
        2. Inspects condition inputs for missing keys or None values. If any condition's feature
           is missing or None, the rule is marked not_evaluable and immediately skipped.
        3. For evaluable rules, evaluates all conditions using configured comparison operators
           ('>=', '<=', '=='). Comparisons include the boundary value.
        4. If and only if ALL conditions evaluate to True, the rule fires and its (id, reason_code)
           is appended to the fired tuple in table order.
        5. Computes score as integer count of fired rules (policy: rules have equal unit weights).

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
        # Check if all features required by this rule are present and not None
        is_evaluable = True
        for cond in rule.conditions:
            val = features.get(cond.feature)
            if val is None:
                is_evaluable = False
                break

        if not is_evaluable:
            not_evaluable_list.append(rule.id)
            continue

        # All features are present and non-None; evaluate all conditions
        all_passed = True
        for cond in rule.conditions:
            feature_val = features[cond.feature]
            assert feature_val is not None  # Guaranteed by is_evaluable check

            thresh_val = cond.threshold_value
            op = cond.op

            if op == ">=":
                condition_passed = feature_val >= thresh_val
            elif op == "<=":
                condition_passed = feature_val <= thresh_val
            elif op == "==":
                condition_passed = feature_val == thresh_val
            else:
                raise ValueError(f"Unsupported comparison operator '{op}' in rule '{rule.id}'")

            if not condition_passed:
                all_passed = False
                break

        if all_passed:
            fired_list.append((rule.id, rule.reason_code))

    return RulesDecision(
        fired=tuple(fired_list),
        not_evaluable=tuple(not_evaluable_list),
        score=len(fired_list),
    )

