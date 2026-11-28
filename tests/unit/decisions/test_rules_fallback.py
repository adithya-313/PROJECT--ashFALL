"""
WHAT:
    Unit tests for decisions/rules_fallback.py under amendment A-074.
    Validates rules_action mapping, interbank visibility boundaries,
    strict earlier-than temporal lookups, and heuristic multi-rule hold actions.

WHY:
    Level 19 decision policy mandates deterministic fallback behavior.
    If an external bank account cannot be inspected, the payment must not be falsely held.
    Temporal lookups must strictly exclude concurrent snapshots to prevent data leakage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from decisions.rules_fallback import (
    FALLBACK_NO_PAYEE_VIEW,
    REASON_ANOTHER_BANK,
    REASON_NO_ACCOUNT_DAY,
    RULES_FALLBACK_ACTIVE,
    payee_rules_decision,
    rules_action,
)


def _dt(hour: int, minute: int = 0) -> datetime:
    """Helper creating a fixed timezone-aware UTC datetime on 2026-11-28."""
    return datetime(2026, 11, 28, hour, minute, tzinfo=timezone.utc)


def test_rules_action_mapping() -> None:
    """
    PURPOSE:
        Tests rules_action maps rule trigger counts to appropriate policy actions.
    """
    assert rules_action(0) == "allow"
    assert rules_action(1) == "warn"
    assert rules_action(2) == "hold"
    assert rules_action(3) == "hold"

    with pytest.raises(ValueError, match="fired_count cannot be negative"):
        rules_action(-1)


def test_payee_at_another_bank_allows_with_fallback_cause() -> None:
    """
    PURPOSE:
        Proves that a payee at a different bank produces an 'allow' action
        with reason code 'fallback_no_payee_view' and cause REASON_ANOTHER_BANK.
    """
    decision_time = _dt(14)
    # Even if features would trigger rules, another bank cannot be inspected
    account_days = [
        {"snapshot_time": _dt(10), "largest_credit_paise_24h": 999999999, "account_age_days": 1}
    ]

    action, reason, cause = payee_rules_decision(
        payee_account_days=account_days,
        decision_time=decision_time,
        payee_bank="BANK_B",
        payment_bank="BANK_A",
    )

    assert action == "allow"
    assert reason == FALLBACK_NO_PAYEE_VIEW
    assert cause == REASON_ANOTHER_BANK


def test_same_bank_no_prior_account_day_allows() -> None:
    """
    PURPOSE:
        Proves that a same-bank payee with no account-day snapshots strictly before
        decision_time produces an 'allow' action with cause REASON_NO_ACCOUNT_DAY.
    """
    decision_time = _dt(10)
    # Only a later account-day exists
    account_days = [
        {"snapshot_time": _dt(12), "largest_credit_paise_24h": 500000}
    ]

    action, reason, cause = payee_rules_decision(
        payee_account_days=account_days,
        decision_time=decision_time,
        payee_bank="BANK_A",
        payment_bank="BANK_A",
    )

    assert action == "allow"
    assert reason == FALLBACK_NO_PAYEE_VIEW
    assert cause == REASON_NO_ACCOUNT_DAY


def test_account_day_at_decision_time_is_not_used() -> None:
    """
    PURPOSE:
        Proves that an account-day exactly at decision_time is strictly excluded.
    """
    decision_time = _dt(12, 0)
    # Snapshot at exactly 12:00 must NOT be visible
    account_days = [
        {"snapshot_time": _dt(12, 0), "largest_credit_paise_24h": 50000000}
    ]

    action, reason, cause = payee_rules_decision(
        payee_account_days=account_days,
        decision_time=decision_time,
        payee_bank="BANK_A",
        payment_bank="BANK_A",
    )

    assert action == "allow"
    assert reason == FALLBACK_NO_PAYEE_VIEW
    assert cause == REASON_NO_ACCOUNT_DAY


def test_visible_payee_row_firing_two_rules_holds() -> None:
    """
    PURPOSE:
        Proves that when a payee has a prior account day triggering >= 2 rules,
        the fallback decision yields 'hold' with 'rules_fallback_active'.
    """
    decision_time = _dt(15)
    # Features configured to trigger Rule 2 (large credit new account)
    # and Rule 3 (many payers one day)
    features: dict[str, Any] = {
        "snapshot_time": _dt(10),
        # Rule 2: largest_credit_paise_24h >= large_credit_amount_paise and account_age_days <= new_account_age_days
        "largest_credit_paise_24h": 50000000,
        "account_age_days": 5,
        # Rule 3: distinct_payers_24h >= payer_count
        "distinct_payers_24h": 50,
        # Other features populated to avoid None un-evaluable where needed
        "pass_through_ratio_24h": 0.1,
        "median_credit_to_debit_seconds_7d": 100000.0,
        "current_balance_paise": 50000000,
        "new_device_login_24h": 0,
        "device_change_count_30d": 0,
    }

    action, reason, cause = payee_rules_decision(
        payee_account_days=[features],
        decision_time=decision_time,
        payee_bank="BANK_A",
        payment_bank="BANK_A",
    )

    assert action == "hold"
    assert reason == RULES_FALLBACK_ACTIVE
    assert cause is None
