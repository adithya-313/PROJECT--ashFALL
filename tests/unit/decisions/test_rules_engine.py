"""
Unit tests for decisions.rules_engine: boundary checks, un-evaluable handling,
score calculations, constructed mule patterns, and look-alike behaviors.
"""

import pytest

from decisions.rules_config import load_rules_config
from decisions.rules_engine import evaluate_rules


def _all_quiet_features() -> dict[str, float | int | None]:
    """Helper returning a clean account feature dictionary where no rules fire."""
    return {
        "largest_credit_paise_24h": 0,
        "pass_through_ratio_24h": 0.0,
        "median_credit_to_debit_seconds_7d": 86400,
        "account_age_days": 365,
        "distinct_payers_24h": 1,
        "current_balance_paise": 5000000,
        "new_device_r5": 0,
    }


def _build_features(overrides: dict[str, float | int | None]) -> dict[str, float | int | None]:
    """Helper constructing a full feature dictionary from all-quiet defaults plus overrides."""
    feats = _all_quiet_features()
    feats.update(overrides)
    return feats


def test_rule_3_boundary_payer_count() -> None:
    """Verifies rule_3 fires exactly at boundary distinct_payers_24h == 10 and does not fire at 9."""
    config = load_rules_config()

    # 1. One below threshold (9 < 10): rule_3 must NOT fire
    feats_below = _all_quiet_features()
    feats_below["distinct_payers_24h"] = 9
    decision_below = evaluate_rules(feats_below, config)
    assert not any(r[0] == "rule_3" for r in decision_below.fired)

    # 2. Exactly at threshold (10 >= 10): rule_3 MUST fire
    feats_at = _all_quiet_features()
    feats_at["distinct_payers_24h"] = 10
    decision_at = evaluate_rules(feats_at, config)
    fired_ids = [r[0] for r in decision_at.fired]
    assert "rule_3" in fired_ids
    assert ("rule_3", "RB3_MANY_PAYERS_ONE_DAY") in decision_at.fired


def test_rule_1_boundary_outflow_share() -> None:
    """Verifies rule_1 fires exactly at boundary pass_through_ratio_24h == 0.8 and does not fire at 0.79."""
    config = load_rules_config()

    # Base features satisfying largest credit and time window conditions for rule_1
    base_feats = _all_quiet_features()
    base_feats["largest_credit_paise_24h"] = 1000000
    base_feats["median_credit_to_debit_seconds_7d"] = 3600  # Exactly at threshold 3600s
    base_feats["current_balance_paise"] = 1000000  # High balance so rule_4 does not fire

    # 1. One below threshold (0.79 < 0.8): rule_1 must NOT fire
    feats_below = dict(base_feats)
    feats_below["pass_through_ratio_24h"] = 0.79
    decision_below = evaluate_rules(feats_below, config)
    assert not any(r[0] == "rule_1" for r in decision_below.fired)

    # 2. Exactly at threshold (0.8 >= 0.8): rule_1 MUST fire
    feats_at = dict(base_feats)
    feats_at["pass_through_ratio_24h"] = 0.8
    decision_at = evaluate_rules(feats_at, config)
    assert ("rule_1", "RB1_FAST_PASS_THROUGH") in decision_at.fired


def test_none_feature_results_in_not_evaluable_and_not_fired() -> None:
    """Verifies that a None or missing feature renders affected rules un-evaluable and never fires."""
    config = load_rules_config()

    # Features where rule_1 conditions would pass if pass_through_ratio_24h were valid,
    # but pass_through_ratio_24h is None
    feats = _all_quiet_features()
    feats["largest_credit_paise_24h"] = 2000000
    feats["pass_through_ratio_24h"] = None
    feats["median_credit_to_debit_seconds_7d"] = 100

    decision = evaluate_rules(feats, config)

    # Rules reading pass_through_ratio_24h (rule_1, rule_4, rule_5) must be not_evaluable
    assert "rule_1" in decision.not_evaluable
    assert "rule_4" in decision.not_evaluable
    assert "rule_5" in decision.not_evaluable

    # Must NOT be fired
    assert not any(r[0] == "rule_1" for r in decision.fired)
    assert not any(r[0] == "rule_4" for r in decision.fired)
    assert not any(r[0] == "rule_5" for r in decision.fired)


def test_missing_feature_key_results_in_not_evaluable() -> None:
    """Verifies that an omitted key from the features mapping is treated as un-evaluable."""
    config = load_rules_config()
    feats = _all_quiet_features()
    del feats["distinct_payers_24h"]

    decision = evaluate_rules(feats, config)
    assert "rule_3" in decision.not_evaluable
    assert not any(r[0] == "rule_3" for r in decision.fired)


def test_score_equals_len_fired() -> None:
    """Verifies that decision.score strictly equals len(decision.fired) across multiple triggers."""
    config = load_rules_config()

    # Zero rules fired
    d0 = evaluate_rules(_all_quiet_features(), config)
    assert d0.score == 0
    assert d0.score == len(d0.fired)

    # One rule fired (rule_3)
    feats1 = _all_quiet_features()
    feats1["distinct_payers_24h"] = 15
    d1 = evaluate_rules(feats1, config)
    assert d1.score == 1
    assert d1.score == len(d1.fired)


CONSTRUCTED_CASES: list[tuple[str, dict[str, float | int | None], set[str]]] = [
    (
        "mule_fast_pass_through",
        {
            "largest_credit_paise_24h": 1500000,
            "pass_through_ratio_24h": 0.9,
            "median_credit_to_debit_seconds_7d": 600,
            "current_balance_paise": 5000,  # Balance near zero (<= 10000) after outflow
        },
        {"rule_1", "rule_4"},
    ),
    (
        "mule_new_account_large_credit",
        {
            "largest_credit_paise_24h": 1500000,
            "account_age_days": 15,
        },
        {"rule_2"},
    ),
    (
        "mule_fan_in",
        {
            "distinct_payers_24h": 25,
            "largest_credit_paise_24h": 200000,
        },
        {"rule_3"},
    ),
    (
        "mule_new_device_money_moved",
        {
            "new_device_r5": 1,
            "pass_through_ratio_24h": 0.85,
            "largest_credit_paise_24h": 500000,
        },
        {"rule_5"},
    ),
    (
        "tea_stall",
        {
            "distinct_payers_24h": 40,
            "largest_credit_paise_24h": 30000,  # Largest ₹300
            "pass_through_ratio_24h": 0.3,
            "current_balance_paise": 2000000,  # Healthy balance
        },
        {"rule_3"},  # KNOWN FALSE TRIGGER — asserted so it is counted, never hidden
    ),
    (
        "gig_worker",
        {
            "largest_credit_paise_24h": 200000,  # Largest ₹2,000 (< ₹10,000 threshold)
            "pass_through_ratio_24h": 0.95,
            "current_balance_paise": 5000,  # Balance ₹50
            "distinct_payers_24h": 3,
        },
        set(),
    ),
    (
        "os_update_only",
        {
            "new_device_r5": 0,
            "pass_through_ratio_24h": 0.9,
            "largest_credit_paise_24h": 0,
        },
        set(),
    ),
    (
        "genuine_phone_change_no_money",
        {
            "new_device_r5": 1,
            "pass_through_ratio_24h": 0.1,
            "largest_credit_paise_24h": 0,
        },
        set(),
    ),
    (
        "family_pooling",
        {
            "distinct_payers_24h": 4,
            "largest_credit_paise_24h": 800000,  # Largest ₹8,000 (< ₹10,000 threshold)
            "pass_through_ratio_24h": 0.2,
        },
        set(),
    ),
]


@pytest.mark.parametrize("case_name,overrides,expected_fired", CONSTRUCTED_CASES)
def test_constructed_cases_and_lookalikes(
    case_name: str,
    overrides: dict[str, float | int | None],
    expected_fired: set[str],
) -> None:
    """
    Verifies that constructed mule patterns and benign look-alike scenarios
    trigger exactly their expected set of baseline heuristic rules.
    """
    config = load_rules_config()
    features = _build_features(overrides)
    decision = evaluate_rules(features, config)
    fired_ids = {r[0] for r in decision.fired}

    # Print for report reproduction as requested by Level 17a instructions
    print(f"\n{case_name} -> {fired_ids}")

    assert fired_ids == expected_fired
