# File: decisions/cold_start.py
# Purpose: Cold-start policy decision, label threshold verification, and candidate review ranking.
# Issue: Level 17b Round 1 anomaly detector and cold-start policy.
"""
WHAT:
    Implements bank-specific cold-start evaluation, counting positive fraud labels up to an as_of cutoff,
    routing between baseline heuristic rules and trained ML models, and ranking candidates for analyst review.

WHY:
    Until a bank accumulates a threshold of verified fraud labels, machine learning models cannot
    be reliably trained or deployed. During cold start, review queues prioritize accounts by composite
    rules score first, breaking ties with continuous unsupervised anomaly scores, and finally account ID.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from decisions.anomaly_config import ColdStartConfig, load_cold_start_config

__all__ = [
    "ReviewCandidate",
    "choose_scorer",
    "count_positive_labels",
    "rank_for_review",
]


@dataclass(frozen=True)
class ReviewCandidate:
    """
    PURPOSE:
        Candidate account evaluated for potential fraud and analyst review prioritization.

    ATTRIBUTES:
        account_id: Unique bank account identifier.
        rules_score: Composite integer score from heuristic decision table.
        anomaly_score: Continuous anomaly score from Isolation Forest (-score_samples).
    """

    account_id: str
    rules_score: int | float
    anomaly_score: float


def _ensure_utc(dt: datetime) -> datetime:
    """Normalizes datetime to UTC timezone."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def count_positive_labels(
    labels: Sequence[Mapping[str, Any]],
    bank_id: str,
    as_of: datetime,
    config: ColdStartConfig | None = None,
) -> int:
    """
    PURPOSE:
        Counts verified positive fraud labels for a specific bank arriving on or before as_of cutoff.

    INPUTS:
        labels: Sequence of label dictionaries (e.g. from labels table or simulation truth sink).
        bank_id: String bank identifier to filter.
        as_of: Datetime timestamp representing point-in-time evaluation cutoff.
        config: Optional pre-loaded ColdStartConfig. If None, default config is loaded.

    KEY LOGIC:
        1. Filters labels strictly by row.bank_id == bank_id.
        2. Filters by row.verdict == config.counted_verdict (defaults to 'fraud').
        3. Filters by row.arrival_time <= as_of to prevent future label leakage.
        4. Increments count for each matching record.

    RETURNS:
        int: Number of positive fraud labels satisfying all three constraints.
    """
    if config is None:
        config = load_cold_start_config()

    cutoff_utc = _ensure_utc(as_of)
    counted_verdict = config.counted_verdict
    count = 0

    for row in labels:
        if row.get("bank_id") != bank_id:
            continue

        if row.get("verdict") != counted_verdict:
            continue

        raw_arrival = row.get("arrival_time")
        if raw_arrival is None:
            continue

        if isinstance(raw_arrival, str):
            arrival_dt = datetime.fromisoformat(raw_arrival)
        elif isinstance(raw_arrival, datetime):
            arrival_dt = raw_arrival
        else:
            continue

        if _ensure_utc(arrival_dt) <= cutoff_utc:
            count += 1

    return count


def choose_scorer(
    positive_label_count: int,
    approved_model_available: bool,
    config: ColdStartConfig | None = None,
) -> str:
    """
    PURPOSE:
        Determines scoring pipeline ('model' vs 'rules_and_anomaly') based on bank cold-start status.

    INPUTS:
        positive_label_count: Count of verified positive fraud labels accumulated by the bank.
        approved_model_available: Boolean flag indicating if an approved supervised model artifact exists.
        config: Optional pre-loaded ColdStartConfig. If None, default config is loaded.

    KEY LOGIC:
        Returns 'model' if and only if:
            positive_label_count >= config.min_positive_labels_per_bank
            AND approved_model_available is True.
        Otherwise returns 'rules_and_anomaly'.

    RETURNS:
        str: 'model' or 'rules_and_anomaly'.
    """
    if config is None:
        config = load_cold_start_config()

    if positive_label_count >= config.min_positive_labels_per_bank and approved_model_available:
        return "model"

    return "rules_and_anomaly"


def _extract_candidate_tuple(candidate: Mapping[str, Any] | Any) -> tuple[str, float, float]:
    """Helper extracting (account_id, rules_score, anomaly_score) from candidate objects or dicts."""
    if isinstance(candidate, Mapping):
        acct_id = str(candidate["account_id"])
        rules_sc = float(candidate.get("rules_score", candidate.get("score", 0.0)))
        anom_sc = float(candidate.get("anomaly_score", candidate.get("anomaly", 0.0)))
    else:
        acct_id = str(getattr(candidate, "account_id"))
        rules_sc = float(getattr(candidate, "rules_score", getattr(candidate, "score", 0.0)))
        anom_sc = float(getattr(candidate, "anomaly_score", getattr(candidate, "anomaly", 0.0)))

    return acct_id, rules_sc, anom_sc


def rank_for_review(
    candidates: Sequence[Mapping[str, Any] | Any],
    capacity: int,
) -> list[str]:
    """
    PURPOSE:
        Ranks candidate accounts for human analyst review by composite rules score and anomaly score.

    INPUTS:
        candidates: Sequence of candidate dictionaries or objects carrying account_id, rules score, anomaly score.
        capacity: Maximum integer number of account IDs to return for review.

    KEY LOGIC:
        1. Validates capacity: negative values raise ValueError; zero returns empty list.
        2. Sorts candidates according to exact deterministic hierarchy:
           - Primary: rules score descending (highest rule hits first).
           - Secondary: anomaly score descending (higher unsupervised outlier score first).
           - Tertiary: account_id ascending (lexicographical tie-breaking).
        3. Returns the first `capacity` account_id strings.

    RETURNS:
        list[str]: Ordered list of account IDs selected for review, bounded by capacity.

    ERRORS:
        Raises ValueError: If capacity < 0.
    """
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative; got {capacity}")

    if capacity == 0 or not candidates:
        return []

    parsed: list[tuple[str, float, float]] = [_extract_candidate_tuple(c) for c in candidates]

    # Sort key: (-rules_score, -anomaly_score, account_id)
    parsed.sort(key=lambda item: (-item[1], -item[2], item[0]))

    return [item[0] for item in parsed[:capacity]]

