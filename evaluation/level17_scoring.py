"""
WHAT: Per-account-day scoring for the Level 17 evaluation: dedupe snapshots to one row per
account-day, score them with rules and an optional anomaly detector, and build the restricted
days and ground-truth sets used by the rules-baseline metrics.

WHY: Models and rules must be compared on the same unit of work that an analyst reviews (an
account on a calendar day), against truth that is only visible to evaluation code. Restricting a
victim is genuine harm: a victim is a real person who was scammed, not a mule.

TERM: IST calendar date: the Asia/Kolkata date of a snapshot_time, computed with core.event_clock.IST.

Every function is pure apart from reading nothing at all: rows and scores are passed in, no file
or database access happens here. Missing or malformed row fields raise a plain ValueError instead
of being guessed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from math import ceil
from typing import Any

from core.event_clock import IST
from decisions.anomaly_detector import TrainedAnomalyDetector, score_rows
from decisions.cold_start import rank_for_review
from decisions.rules_config import RulesConfig
from decisions.rules_engine import evaluate_rules
from features.account_snapshots import account_feature_names

__all__ = [
    "GENUINE_ROLES",
    "MULE_ROLES",
    "ScoredDay",
    "anomaly_restricted_days",
    "daily_escalations",
    "feature_values_only",
    "genuine_restricted_days",
    "ground_sets",
    "one_row_per_account_day",
    "rules_restricted_days",
    "score_account_days",
]

MULE_ROLES: frozenset[str] = frozenset({"mule", "collector", "cash_out"})
GENUINE_ROLES: frozenset[str] = frozenset({"genuine", "victim"})

_SNAPSHOT_KINDS: frozenset[str] = frozenset({"daily", "full"})
_ESCALATION_MODES: frozenset[str] = frozenset({"rules", "anomaly", "rules_and_anomaly"})


@dataclass(frozen=True)
class ScoredDay:
    """
    WHAT: One account's scored state on one IST calendar day.

    ATTRIBUTES:
        bank_id: Bank that holds the account.
        account_id: Account identifier.
        day: IST calendar date of the kept snapshot.
        rules_score: Number of rules that fired.
        rules_fired: Reason codes of the fired rules, in rule-table order.
        anomaly_score: Detector score, or None when no detector was supplied.
        role: Hidden truth role on this day.
        depth: Mule depth on this day, or None for a genuine role.
    """

    bank_id: str
    account_id: str
    day: date
    rules_score: int
    rules_fired: tuple[str, ...]
    anomaly_score: float | None
    role: str
    depth: int | None


def _require_string(row: dict[str, Any], key: str, caller: str) -> str:
    """
    WHAT: Returns a required non-empty string field from a snapshot row.

    Args:
        row: Snapshot row.
        key: Field name to read.
        caller: Function name used in the error message.

    Raises:
        ValueError: If the field is missing, empty, or not a string.
    """
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{caller} requires a non-empty string '{key}' on every row")
    return value


def _snapshot_datetime(row: dict[str, Any]) -> datetime:
    """
    WHAT: Returns the timezone-aware snapshot_time of a row.

    Args:
        row: Snapshot row.

    Raises:
        ValueError: If snapshot_time is missing, not a datetime, or timezone-naive.
    """
    value = row.get("snapshot_time")
    if not isinstance(value, datetime):
        raise ValueError("Snapshot rows require a datetime 'snapshot_time'")
    if value.tzinfo is None:
        raise ValueError("snapshot_time must be timezone-aware so its IST date is unambiguous")
    return value


def _snapshot_kind(row: dict[str, Any]) -> str:
    """
    WHAT: Returns the snapshot_kind of a row, which is either 'daily' or 'full'.

    Args:
        row: Snapshot row.

    Raises:
        ValueError: If snapshot_kind is missing or is not 'daily' or 'full'.
    """
    value = row.get("snapshot_kind")
    if value not in _SNAPSHOT_KINDS:
        raise ValueError(f"snapshot_kind must be 'daily' or 'full'; got {value!r}")
    return str(value)


def _ist_day(row: dict[str, Any]) -> date:
    """
    WHAT: Returns the IST calendar date of a row's snapshot_time.

    Args:
        row: Snapshot row.

    Raises:
        ValueError: If snapshot_time is missing, malformed, or timezone-naive.
    """
    return _snapshot_datetime(row).astimezone(IST).date()


def _truth_depth(row: dict[str, Any]) -> int | None:
    """
    WHAT: Returns truth_depth from a row, which is an int for mules and None for genuine roles.

    Args:
        row: Snapshot row.

    Raises:
        ValueError: If truth_depth is present but is neither an int nor None.
    """
    value = row.get("truth_depth")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"truth_depth must be an int or None; got {value!r}")
    return value


def one_row_per_account_day(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    WHAT: Keeps one snapshot row per (bank, account, IST day): the latest, with 'daily'
    preferred when two rows share the same latest time.

    WHY: An account can be snapshotted more than once a day (for example a full snapshot at
    09:00 and the daily one at 21:00). Scoring both would count the same account-day twice.

    Args:
        rows: Snapshot rows, each carrying bank_id, account_id, snapshot_time and snapshot_kind.

    Returns:
        One row per key, in first-seen key order. No input row is modified.

    Algorithm:
        Walk the rows once and keep a per-key winner: a later snapshot_time replaces the
        current winner; on an exact time tie a 'daily' row wins over a 'full' row.

    Raises:
        ValueError: If a row lacks a required string field, a timezone-aware snapshot_time,
            or a valid snapshot_kind.
    """
    selected: dict[tuple[str, str, date], dict[str, Any]] = {}
    for row in rows:
        bank_id = _require_string(row, "bank_id", "one_row_per_account_day")
        account_id = _require_string(row, "account_id", "one_row_per_account_day")
        day = _ist_day(row)
        key = (bank_id, account_id, day)
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue
        moment = _snapshot_datetime(row)
        current_moment = _snapshot_datetime(current)
        if moment > current_moment:
            selected[key] = row
        elif moment == current_moment:
            if _snapshot_kind(row) == "daily" and _snapshot_kind(current) != "daily":
                selected[key] = row
    return list(selected.values())


def feature_values_only(row: dict[str, Any]) -> dict[str, Any]:
    """
    WHAT: Returns only the canonical account feature values from a snapshot row.

    WHY: Scorers must never see hidden truth. Passing a filtered mapping keeps truth_role,
    truth_depth, truth_owner and every other non-feature column out of the scoring code.

    Args:
        row: Snapshot row with feature columns plus identifiers and truth columns.

    Returns:
        Mapping of canonical account feature name to its value, containing no key that starts
        with 'truth_' and no identifier columns such as bank_id or account_id.

    Constraints:
        Feature names come from features.account_snapshots.account_feature_names(); a row that
        omits a feature simply omits it from the result, which evaluate_rules treats as
        non-evaluable for any rule that needs it.
    """
    allowed_names = set(account_feature_names())
    feature_values: dict[str, Any] = {}
    for name, value in row.items():
        if name.startswith("truth_"):
            continue
        if name in allowed_names:
            feature_values[name] = value
    return feature_values


def score_account_days(
    day_rows: list[dict[str, Any]],
    rules_config: RulesConfig,
    detector: TrainedAnomalyDetector | None,
) -> list[ScoredDay]:
    """
    WHAT: Scores one row-per-account-day with the rules table and, optionally, an anomaly detector.

    WHY: Produces one immutable ScoredDay per account-day so later functions can restrict days,
    compute recall, and rank review queues without touching raw rows again.

    Args:
        day_rows: Rows already reduced by one_row_per_account_day.
        rules_config: Rules table and thresholds to evaluate.
        detector: Optional trained anomaly detector. When None every anomaly_score is None and
            score_rows is never called.

    Returns:
        A ScoredDay per input row, in input order.

    Key logic:
        The detector is called once with the whole batch of feature-only mappings, as
        anomaly_detector.score_rows expects; its scores are then zipped back onto the rows.

    Raises:
        ValueError: If a row lacks a required identifier, truth_role, snapshot_time or has an
            invalid truth_depth.
    """
    feature_dicts = [feature_values_only(row) for row in day_rows]
    anomaly_scores: list[float | None] = []
    if detector is None:
        for _ in day_rows:
            anomaly_scores.append(None)
    else:
        for anomaly_score in score_rows(detector, feature_dicts):
            anomaly_scores.append(anomaly_score)

    scored_days: list[ScoredDay] = []
    for row, features, anomaly_score in zip(day_rows, feature_dicts, anomaly_scores, strict=True):
        decision = evaluate_rules(features, rules_config)
        scored_days.append(
            ScoredDay(
                bank_id=_require_string(row, "bank_id", "score_account_days"),
                account_id=_require_string(row, "account_id", "score_account_days"),
                day=_ist_day(row),
                rules_score=decision.score,
                rules_fired=tuple(reason_code for _, reason_code in decision.fired),
                anomaly_score=anomaly_score,
                role=_require_string(row, "truth_role", "score_account_days"),
                depth=_truth_depth(row),
            )
        )
    return scored_days


def rules_restricted_days(scored: list[ScoredDay]) -> dict[str, set[date]]:
    """
    WHAT: Maps each account to the days on which at least one rule fired.

    WHY: A rules hit is the baseline's restriction decision, and the genuine-harm metric needs
    the exact days an innocent account was restricted.

    Args:
        scored: Scored account-days.

    Returns:
        account_id -> set of IST dates with rules_score >= 1. Accounts with no hit are absent.
    """
    restricted: dict[str, set[date]] = {}
    for scored_day in scored:
        if scored_day.rules_score >= 1:
            restricted.setdefault(scored_day.account_id, set()).add(scored_day.day)
    return restricted


def anomaly_restricted_days(scored: list[ScoredDay], threshold: float) -> dict[str, set[date]]:
    """
    WHAT: Maps each account to the days whose anomaly score is at least the threshold.

    WHY: Anomaly-based restriction is compared with the rules baseline at the same genuine
    restriction rate, so its restricted days must be derived the same way.

    Args:
        scored: Scored account-days.
        threshold: Inclusive anomaly score cut.

    Returns:
        account_id -> set of IST dates whose anomaly_score >= threshold.

    Raises:
        ValueError: If any row has no anomaly score, because a numeric cut cannot be applied
            to rows that were scored without a detector.
    """
    restricted: dict[str, set[date]] = {}
    for scored_day in scored:
        if scored_day.anomaly_score is None:
            raise ValueError("anomaly_restricted_days requires anomaly scores on every row")
        if scored_day.anomaly_score >= threshold:
            restricted.setdefault(scored_day.account_id, set()).add(scored_day.day)
    return restricted


def genuine_restricted_days(
    restricted: dict[str, set[date]],
    scored: list[ScoredDay],
) -> dict[str, set[date]]:
    """
    WHAT: Keeps only the restricted days on which the account's truth role was genuine or victim.

    WHY: A victim is a real person who was scammed, so restricting them is genuine harm, exactly
    like restricting a genuine payer.

    Args:
        restricted: account_id -> restricted dates from a rules or anomaly restriction.
        scored: Scored account-days that define each account's role per day.

    Returns:
        account_id -> restricted dates whose role is in GENUINE_ROLES. Accounts left with no
        such day are omitted.

    Raises:
        ValueError: If a restricted account-day has no scored row, because its role cannot be known.
    """
    roles_by_account_day: dict[tuple[str, date], str] = {
        (scored_day.account_id, scored_day.day): scored_day.role for scored_day in scored
    }
    genuine_restricted: dict[str, set[date]] = {}
    for account_id, days in restricted.items():
        kept_days: set[date] = set()
        for day in days:
            key = (account_id, day)
            if key not in roles_by_account_day:
                raise ValueError(f"Restricted day {day} for account {account_id} has no scored row")
            if roles_by_account_day[key] in GENUINE_ROLES:
                kept_days.add(day)
        if kept_days:
            genuine_restricted[account_id] = kept_days
    return genuine_restricted


def ground_sets(
    scored: list[ScoredDay],
) -> tuple[set[str], set[str], dict[str, int], Callable[[str, date], bool]]:
    """
    WHAT: Builds the truth ground sets used for recall and harm: mules, genuinely active accounts,
    each mule's deepest depth, and a per-day mule predicate.

    WHY: Metrics need stable definitions of who is a mule and who is genuine, including accounts
    whose role changes over time.

    Args:
        scored: Scored account-days carrying truth roles and depths.

    Returns:
        A 4-tuple (mules, genuine_active, depth_of, is_mule) where:
        - mules: accounts whose role is in MULE_ROLES on at least one day.
        - genuine_active: accounts whose role is in GENUINE_ROLES on at least one day.
        - depth_of: account -> deepest truth_depth seen on a mule day.
        - is_mule: predicate returning True only when that account has a mule role on that day.

    Raises:
        ValueError: If a mule day has no truth_depth, because its depth would be unknown.
    """
    mules: set[str] = set()
    genuine_active: set[str] = set()
    depth_of: dict[str, int] = {}
    roles_by_account_day: dict[tuple[str, date], str] = {}
    for scored_day in scored:
        roles_by_account_day[(scored_day.account_id, scored_day.day)] = scored_day.role
        if scored_day.role in MULE_ROLES:
            mules.add(scored_day.account_id)
            if scored_day.depth is None:
                raise ValueError(f"Mule account {scored_day.account_id} has no truth_depth")
            known_depth = depth_of.get(scored_day.account_id)
            if known_depth is None or scored_day.depth > known_depth:
                depth_of[scored_day.account_id] = scored_day.depth
        if scored_day.role in GENUINE_ROLES:
            genuine_active.add(scored_day.account_id)

    def is_mule(account_id: str, day: date) -> bool:
        """True when the account had a mule role on the given IST day."""
        return roles_by_account_day.get((account_id, day)) in MULE_ROLES

    return mules, genuine_active, depth_of, is_mule


def _is_candidate(scored_day: ScoredDay, mode: str, anomaly_threshold: float | None) -> bool:
    """
    WHAT: Decides whether one scored day enters the review queue for the given mode.

    Args:
        scored_day: Scored account-day.
        mode: 'rules', 'anomaly', or 'rules_and_anomaly'.
        anomaly_threshold: Inclusive anomaly cut for the anomaly modes.

    Returns:
        True when the day qualifies under the mode. A None anomaly score never qualifies for
        the anomaly part, because it cannot be compared with a numeric cut.
    """
    rules_hit = scored_day.rules_score >= 1
    anomaly_hit = (
        anomaly_threshold is not None
        and scored_day.anomaly_score is not None
        and scored_day.anomaly_score >= anomaly_threshold
    )
    if mode == "rules":
        return rules_hit
    if mode == "anomaly":
        return anomaly_hit
    return rules_hit or anomaly_hit


def daily_escalations(
    scored: list[ScoredDay],
    mode: str,
    anomaly_threshold: float | None,
    capacity_pct: float,
) -> dict[date, list[str]]:
    """
    WHAT: Builds the per-day escalation list an analyst would receive, respecting per-bank capacity.

    WHY: Recall must be measured at the review capacity that is actually available, not against
    unbounded escalation. Each bank can only escalate a percentage of its active accounts that day.

    Args:
        scored: One scored row per account-day.
        mode: 'rules', 'anomaly', or 'rules_and_anomaly'.
        anomaly_threshold: Inclusive anomaly cut; required for the two anomaly modes.
        capacity_pct: Percentage of a bank's active accounts that can be escalated that day.

    Returns:
        IST date -> ordered account ids, merged across banks. Every day present in scored gets
        an entry, possibly an empty list, so an honest zero-escalation day is still recorded.

    Key logic:
        Per bank and day, capacity = max(1, ceil(active_accounts * capacity_pct / 100)). Candidates
        are ranked with decisions.cold_start.rank_for_review; a missing anomaly score ranks as 0.0
        because ranking must stay deterministic when no detector was used.

    Raises:
        ValueError: If mode is unknown, if an anomaly mode is given no threshold, or if mode
            'anomaly' is requested for rows that have no anomaly scores.
    """
    if mode not in _ESCALATION_MODES:
        raise ValueError(f"Unknown escalation mode {mode!r}; expected one of {sorted(_ESCALATION_MODES)}")
    if mode in ("anomaly", "rules_and_anomaly") and anomaly_threshold is None:
        raise ValueError(f"Escalation mode {mode!r} requires an anomaly_threshold")
    if mode == "anomaly":
        for scored_day in scored:
            if scored_day.anomaly_score is None:
                raise ValueError("Escalation mode 'anomaly' requires anomaly scores on every row")

    days = sorted({scored_day.day for scored_day in scored})
    escalations: dict[date, list[str]] = {day: [] for day in days}
    banks = sorted({scored_day.bank_id for scored_day in scored})
    for bank_id in banks:
        for day in days:
            bank_day_rows = [
                scored_day
                for scored_day in scored
                if scored_day.bank_id == bank_id and scored_day.day == day
            ]
            if not bank_day_rows:
                continue
            capacity = max(1, ceil(len(bank_day_rows) * capacity_pct / 100.0))
            candidates = [
                {
                    "account_id": scored_day.account_id,
                    "rules_score": float(scored_day.rules_score),
                    "anomaly_score": 0.0 if scored_day.anomaly_score is None else scored_day.anomaly_score,
                }
                for scored_day in bank_day_rows
                if _is_candidate(scored_day, mode, anomaly_threshold)
            ]
            escalations[day].extend(rank_for_review(candidates, capacity))
    return escalations
