"""
WHAT:
    Level 19 rules fallback decision module under amendment A-074.
    Evaluates heuristic decision rules on the payee's account-day feature row
    when machine learning scoring is degraded or serving as fallback.

WHY:
    Rules fallback answers the exact same operational question as the ML model
    ("is the payee a mule?"). Visibility is strictly bounded by same-bank view
    and strictly prior account snapshots. Unknown banks or other-bank payees
    cannot be inspected and must safely fallback to 'allow' with recorded reason codes,
    preventing innocent interbank payments from being falsely blocked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from decisions.rules_config import RulesConfig, load_serving_rules_config
from decisions.rules_engine import evaluate_rules

__all__ = [
    "FALLBACK_NO_PAYEE_VIEW",
    "REASON_ANOTHER_BANK",
    "REASON_BANK_UNKNOWN",
    "REASON_NO_ACCOUNT_DAY",
    "REASON_NO_PRIOR_DAY",
    "REASON_OTHER_BANK",
    "RULES_FALLBACK_ACTIVE",
    "payee_rules_decision",
    "rules_action",
]

# Canonical NaN reasons from evaluation.payment_oof_risk
REASON_BANK_UNKNOWN: str = "receiver_bank_unknown"
REASON_OTHER_BANK: str = "receiver_at_other_bank"
REASON_NO_PRIOR_DAY: str = "no_prior_account_day"

# Specification aliases
REASON_ANOTHER_BANK: str = REASON_OTHER_BANK
REASON_NO_ACCOUNT_DAY: str = REASON_NO_PRIOR_DAY

# Decision reason codes
FALLBACK_NO_PAYEE_VIEW: str = "fallback_no_payee_view"
RULES_FALLBACK_ACTIVE: str = "rules_fallback_active"


def rules_action(fired_count: int) -> str:
    """
    PURPOSE:
        Maps the count of triggered heuristic rules to an operational action under A-074.

    INPUTS:
        fired_count: Non-negative integer count of rules triggered.

    RETURNS:
        str: "hold" if >= 2, "warn" if == 1, "allow" if 0.

    ERRORS RAISED:
        ValueError: If fired_count is negative.
    """
    if fired_count < 0:
        raise ValueError(f"fired_count cannot be negative, got {fired_count}")
    if fired_count >= 2:
        return "hold"
    if fired_count == 1:
        return "warn"
    return "allow"


def payee_rules_decision(
    payee_account_days: Sequence[Any],
    decision_time: datetime,
    payee_bank: str | None,
    payment_bank: str,
    rules_config: RulesConfig | Mapping[str, Any] | None = None,
) -> tuple[str, str, str | None]:
    """
    PURPOSE:
        Makes an operational fallback decision on a payment based on the payee's account features
        strictly prior to decision_time under A-074 visibility rules.

    INPUTS:
        payee_account_days: Sequence of account-day records (dicts or objects with snapshot_time).
        decision_time: The payment's decision instant.
        payee_bank: Bank identifier of the payee account, or None if unknown.
        payment_bank: Serving bank identifier where the payment originated.
        rules_config: Optional RulesConfig or config mapping; defaults to serving rules config.

    KEY STEPS:
        1. Check payee bank visibility. If payee_bank is None, return allow with REASON_BANK_UNKNOWN.
           If payee_bank != payment_bank, return allow with REASON_ANOTHER_BANK.
        2. Identify the payee's latest account-day snapshot strictly earlier than decision_time.
           Account-days at or after decision_time are strictly excluded.
        3. If no strictly earlier account-day exists, return allow with REASON_NO_ACCOUNT_DAY.
        4. If a prior account-day is found, evaluate the configured rules against its features.
        5. Map fired rule count to action (>=2 hold, 1 warn, 0 allow) and return with RULES_FALLBACK_ACTIVE.

    RETURNS:
        tuple[str, str, str | None]: (action, reason_code, fallback_cause)

    ERRORS RAISED:
        ValueError: If payment_bank is empty or decision_time is missing.
    """
    if not payment_bank:
        raise ValueError("payment_bank must be a non-empty string")
    if not isinstance(decision_time, datetime):
        raise ValueError(f"decision_time must be a datetime instance, got {type(decision_time)}")

    # 1. Visibility check: unknown bank or interbank payment
    if payee_bank is None:
        return "allow", FALLBACK_NO_PAYEE_VIEW, REASON_BANK_UNKNOWN
    if payee_bank != payment_bank:
        return "allow", FALLBACK_NO_PAYEE_VIEW, REASON_ANOTHER_BANK

    # 2. Find strictly prior account-day
    prior: Any = None
    for day in payee_account_days:
        if isinstance(day, Mapping):
            day_time = day.get("snapshot_time")
        else:
            day_time = getattr(day, "snapshot_time", None)

        if not isinstance(day_time, datetime):
            raise ValueError(f"Account day item missing valid snapshot_time datetime: {day}")

        if day_time < decision_time:
            if prior is None:
                prior = day
            else:
                prior_time = prior.get("snapshot_time") if isinstance(prior, Mapping) else prior.snapshot_time
                if day_time > prior_time:
                    prior = day

    if prior is None:
        return "allow", FALLBACK_NO_PAYEE_VIEW, REASON_NO_ACCOUNT_DAY

    # 3. Extract feature row
    if isinstance(prior, Mapping):
        feature_row = prior
    elif hasattr(prior, "features") and isinstance(prior.features, Mapping):
        feature_row = prior.features
    elif hasattr(prior, "__dict__"):
        feature_row = prior.__dict__
    else:
        raise ValueError(f"Cannot extract feature mapping from account day: {prior}")

    # 4. Resolve rules config
    resolved_config: RulesConfig | None
    if rules_config is None:
        resolved_config = load_serving_rules_config()
    elif isinstance(rules_config, RulesConfig):
        resolved_config = rules_config
    else:
        resolved_config = load_serving_rules_config()

    # 5. Evaluate rules
    decision = evaluate_rules(feature_row, config=resolved_config)
    fired_count = len(decision.fired)
    action = rules_action(fired_count)

    return action, RULES_FALLBACK_ACTIVE, None
