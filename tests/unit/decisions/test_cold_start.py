"""
Unit tests for decisions.cold_start: label counting, scorer selection, and candidate review ranking.
"""

from datetime import datetime, timedelta, timezone

import pytest

from decisions.anomaly_config import load_cold_start_config
from decisions.cold_start import (
    ReviewCandidate,
    choose_scorer,
    count_positive_labels,
    rank_for_review,
)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
BANK_ID = "bank_01"


def test_choose_scorer_routing_policy() -> None:
    """
    Verifies scorer routing logic against the 200 positive labels threshold:
    - 0 labels -> rules_and_anomaly
    - 200 labels but no model -> rules_and_anomaly
    - 199 labels + model -> rules_and_anomaly
    - 200 labels + model -> model (boundary)
    """
    config = load_cold_start_config()

    # 1. Zero labels
    assert choose_scorer(0, approved_model_available=True, config=config) == "rules_and_anomaly"
    assert choose_scorer(0, approved_model_available=False, config=config) == "rules_and_anomaly"

    # 2. 200 labels but no model available -> rules_and_anomaly
    assert choose_scorer(200, approved_model_available=False, config=config) == "rules_and_anomaly"

    # 3. 199 labels (one below threshold) + model available -> rules_and_anomaly
    assert choose_scorer(199, approved_model_available=True, config=config) == "rules_and_anomaly"

    # 4. 200 labels (exact boundary) + model available -> model
    assert choose_scorer(200, approved_model_available=True, config=config) == "model"

    # 5. > 200 labels + model available -> model
    assert choose_scorer(250, approved_model_available=True, config=config) == "model"


def test_count_positive_labels_filters_verdict_arrival_and_bank() -> None:
    """
    Verifies that count_positive_labels only counts:
    - rows with matching bank_id (ignores other banks)
    - rows with verdict == 'fraud' (ignores non-fraud verdicts)
    - rows with arrival_time <= as_of (ignores late-arriving labels)
    """
    as_of = T0

    labels = [
        # 1. Valid positive fraud label inside cutoff
        {"bank_id": BANK_ID, "verdict": "fraud", "arrival_time": as_of - timedelta(hours=1)},
        # 2. Valid positive fraud label exactly at cutoff
        {"bank_id": BANK_ID, "verdict": "fraud", "arrival_time": as_of},
        # 3. Future arriving label (after as_of) -> must NOT be counted
        {"bank_id": BANK_ID, "verdict": "fraud", "arrival_time": as_of + timedelta(seconds=1)},
        # 4. Non-fraud label (genuine / not_fraud) inside cutoff -> must NOT be counted
        {"bank_id": BANK_ID, "verdict": "not_fraud", "arrival_time": as_of - timedelta(hours=2)},
        {"bank_id": BANK_ID, "verdict": "genuine", "arrival_time": as_of - timedelta(hours=2)},
        # 5. Different bank label inside cutoff -> must NOT be counted
        {"bank_id": "bank_other", "verdict": "fraud", "arrival_time": as_of - timedelta(hours=1)},
    ]

    count = count_positive_labels(labels, bank_id=BANK_ID, as_of=as_of)
    assert count == 2


def test_rank_for_review_ordering_and_tie_breaking() -> None:
    """
    Verifies candidate ranking order:
    1. rules score descending
    2. anomaly score descending
    3. account_id ascending (lexicographical)
    """
    candidates = [
        # Candidate A: rules 2, anomaly 0.5
        ReviewCandidate(account_id="acc_A", rules_score=2, anomaly_score=0.5),
        # Candidate B: rules 3, anomaly 0.1 (higher rules score than A)
        ReviewCandidate(account_id="acc_B", rules_score=3, anomaly_score=0.1),
        # Candidate C: rules 2, anomaly 0.9 (tied with A on rules, higher anomaly)
        ReviewCandidate(account_id="acc_C", rules_score=2, anomaly_score=0.9),
        # Candidate D: rules 2, anomaly 0.9 (tied with C on both rules and anomaly, but id 'acc_D' > 'acc_C')
        ReviewCandidate(account_id="acc_D", rules_score=2, anomaly_score=0.9),
        # Candidate E: rules 0, anomaly 0.99
        ReviewCandidate(account_id="acc_E", rules_score=0, anomaly_score=0.99),
    ]

    # Expected order:
    # 1. acc_B (rules=3)
    # 2. acc_C (rules=2, anomaly=0.9, id='acc_C')
    # 3. acc_D (rules=2, anomaly=0.9, id='acc_D')
    # 4. acc_A (rules=2, anomaly=0.5)
    # 5. acc_E (rules=0, anomaly=0.99)
    ranked = rank_for_review(candidates, capacity=5)
    assert ranked == ["acc_B", "acc_C", "acc_D", "acc_A", "acc_E"]

    # Capacity limits results
    assert rank_for_review(candidates, capacity=2) == ["acc_B", "acc_C"]
    assert rank_for_review(candidates, capacity=3) == ["acc_B", "acc_C", "acc_D"]


def test_rank_for_review_capacity_zero_and_negative() -> None:
    """Verifies that capacity 0 returns empty list and capacity < 0 raises ValueError."""
    candidates = [
        {"account_id": "acc_001", "rules_score": 1, "anomaly_score": 0.4},
    ]

    # Capacity 0 -> empty list
    assert rank_for_review(candidates, capacity=0) == []

    # Capacity negative -> ValueError
    with pytest.raises(ValueError, match="capacity must be non-negative"):
        rank_for_review(candidates, capacity=-1)
    with pytest.raises(ValueError, match="capacity must be non-negative"):
        rank_for_review(candidates, capacity=-100)

