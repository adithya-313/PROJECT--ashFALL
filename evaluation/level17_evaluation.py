"""
WHAT: Per-world Level 17 evaluation: the untuned guard, time and ground-set helpers, and one
function that scores a single world with the merged scoring and metric layers.

WHY: The rules baseline and the anomaly detector must be reported on the same unit of work (an
account-day), against hidden truth that only evaluation code touches, and only after thresholds
were tuned on validation. These functions are pure: callers pass loaded tables, rows and truth
callables, and nothing here reads files or a database.
"""

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from math import ceil
from typing import Any

from decisions.rules_config import RulesConfig
from evaluation.level17_scoring import (
    MULE_ROLES,
    AccountKey,
    ScoredDay,
    account_key,
    anomaly_restricted_days,
    daily_escalations,
    genuine_restricted_days,
    ground_sets,
    rules_restricted_days,
)
from evaluation.rules_metrics import (
    analyst_load,
    genuine_harm,
    hidden_mule_recall_by_depth,
    recall_at_capacity,
    time_to_detection,
    yield_rate,
)

__all__ = [
    "evaluate_world",
    "first_flag_times",
    "first_illicit_credit",
    "require_tuned",
    "world_start_by_world",
]

_SCORERS: frozenset[str] = frozenset({"rules", "anomaly", "rules_and_anomaly"})


def require_tuned(rules_config: RulesConfig) -> None:
    """
    WHAT: Refuses to evaluate with thresholds that were not tuned on validation.

    WHY: Reporting untuned_start values as results would present a guess as a measurement, so
    every evaluation path must pass this guard before it touches a test world.

    Args:
        rules_config: Rules table whose threshold statuses are checked.

    Raises:
        ValueError: With the message "thresholds not tuned on validation" when any threshold is
            still untuned_start.
    """
    if not rules_config.all_thresholds_tuned():
        raise ValueError("thresholds not tuned on validation")


def world_start_by_world(snapshot_rows: Sequence[Mapping[str, Any]]) -> dict[str, datetime]:
    """
    WHAT: Returns each world's earliest snapshot_time, keyed by world_id.

    WHY: The anomaly detector's early-window filter needs a world start, and the earliest snapshot
    is the only clock information available without reading truth.

    Args:
        snapshot_rows: Snapshot rows carrying world_id and a timezone-aware snapshot_time.

    Returns:
        world_id -> earliest snapshot_time.

    Raises:
        ValueError: If a row lacks a non-empty world_id or a timezone-aware datetime snapshot_time.
    """
    starts: dict[str, datetime] = {}
    for row in snapshot_rows:
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or not world_id:
            raise ValueError("Snapshot rows require a non-empty string 'world_id'")
        moment = row.get("snapshot_time")
        if not isinstance(moment, datetime):
            raise ValueError("Snapshot rows require a datetime 'snapshot_time'")
        if moment.tzinfo is None:
            raise ValueError("snapshot_time must be timezone-aware")
        current = starts.get(world_id)
        if current is None or moment < current:
            starts[world_id] = moment
    return starts


def first_illicit_credit(
    payment_rows: Sequence[Mapping[str, Any]],
    story_of: Callable[[str], Mapping[str, Any]],
) -> dict[str, datetime]:
    """
    WHAT: Returns each mule account's earliest fraud-origin payment decision_time.

    WHY: Detection speed is measured from the first money a mule receives, so the clock starts at
    the earliest fraudulent credit to that account. Genuine stories and payments received by
    accounts that were not mules at that moment must not start the clock.

    Args:
        payment_rows: Joined payment rows carrying payment_id, receiver_account_id, decision_time
            and truth_role_receiver (the receiver's role at that moment).
        story_of: Callable returning the payment story for a payment_id.

    Returns:
        receiver account id -> earliest fraud-origin decision_time.

    Raises:
        ValueError: If a candidate row lacks a required field or decision_time is timezone-naive.
        KeyError: If story_of has no story for a payment id (raised by the truth store).
    """
    first_credit: dict[str, datetime] = {}
    for row in payment_rows:
        if row.get("truth_role_receiver") not in MULE_ROLES:
            continue
        payment_id = row.get("payment_id")
        if not isinstance(payment_id, str) or not payment_id:
            raise ValueError("Payment rows require a non-empty string 'payment_id'")
        receiver = row.get("receiver_account_id")
        if not isinstance(receiver, str) or not receiver:
            raise ValueError("Payment rows require a non-empty string 'receiver_account_id'")
        moment = row.get("decision_time")
        if not isinstance(moment, datetime):
            raise ValueError("Payment rows require a datetime 'decision_time'")
        if moment.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        story = story_of(payment_id)
        if story.get("origin") != "fraud":
            continue
        current = first_credit.get(receiver)
        if current is None or moment < current:
            first_credit[receiver] = moment
    return first_credit


def first_flag_times(
    daily_escalations: Mapping[date, Sequence[AccountKey]],
    scored: Sequence[ScoredDay],
    snapshot_time_by_account_day: Mapping[tuple[AccountKey, date], datetime],
) -> dict[AccountKey, datetime]:
    """
    WHAT: Returns each escalated account's first escalation time, taken from that day's snapshot.

    WHY: Detection speed is measured from the first illicit credit to the moment the account was
    actually seen, so the flag time is the snapshot_time of the escalated account-day, not the
    start of its calendar day.

    Args:
        daily_escalations: IST date -> account keys escalated that day.
        scored: Scored account-days; every escalated account-day must appear in it.
        snapshot_time_by_account_day: (account key, IST date) -> timezone-aware snapshot_time of
            the snapshot row that produced that scored account-day.

    Returns:
        account key -> snapshot_time of its first escalated account-day. Two worlds that reuse an
        account id get two entries, each with its own world's first flag.

    Raises:
        ValueError: If an escalated account-day has no scored row, no snapshot time, or a
            timezone-naive snapshot time.
    """
    scored_keys = {(account_key(scored_day), scored_day.day) for scored_day in scored}
    first_flag: dict[AccountKey, datetime] = {}
    for day in sorted(daily_escalations):
        for account in daily_escalations[day]:
            key = (account, day)
            if key not in scored_keys:
                raise ValueError(f"Escalated day {day} for account {account} has no scored row")
            moment = snapshot_time_by_account_day.get(key)
            if moment is None:
                raise ValueError(f"Escalated day {day} for account {account} has no snapshot time")
            if moment.tzinfo is None:
                raise ValueError(f"Snapshot time for {account} on {day} must be timezone-aware")
            if account not in first_flag:
                first_flag[account] = moment
    return first_flag


def _merge_restrictions(
    left: Mapping[AccountKey, set[date]],
    right: Mapping[AccountKey, set[date]],
) -> dict[AccountKey, set[date]]:
    """
    WHAT: Unions two account-to-days restriction maps.

    Args:
        left: First restriction map.
        right: Second restriction map.

    Returns:
        account -> union of restricted days from both maps.
    """
    merged: dict[AccountKey, set[date]] = {account: set(days) for account, days in left.items()}
    for account, days in right.items():
        merged.setdefault(account, set()).update(days)
    return merged


def evaluate_world(
    scored: list[ScoredDay],
    scorer: str,
    anomaly_threshold: float | None,
    capacity_pct: float,
    inflation_factor: float,
    persona_of: Callable[[str], str],
    weights: dict[str, int],
    default_weight: int,
    snapshot_time_by_account_day: Mapping[tuple[AccountKey, date], datetime],
    first_credit: Mapping[str, datetime],
    reported: set[str],
    repeat_policy: str = "every_day",
) -> dict[str, Any]:
    """
    WHAT: Scores one world's account-days with one scorer and returns the six headline metrics.

    WHY: The three scorers must be judged by identical metrics on identical rows, so every number
    comes from the already-merged rules_metrics and level17_scoring functions.

    Args:
        scored: One scored row per account-day in this world.
        scorer: "rules" (any rule fired), "anomaly" (score >= threshold) or
            "rules_and_anomaly" (either).
        anomaly_threshold: Inclusive anomaly cut; required for the anomaly scorers.
        capacity_pct: Percentage of a bank's active accounts the review team can check per day.
        inflation_factor: Positive divisor applied to true positives by the reweighted yield.
        persona_of: Callable returning an account's persona label.
        weights: Persona label to harm weight.
        default_weight: Harm weight for personas not in weights.
        snapshot_time_by_account_day: (account key, IST date) -> snapshot_time of that account-day,
            used as the detection flag time.
        first_credit: Mule account -> first illicit credit time.
        reported: Accounts already reported to the national registry.
        repeat_policy: Escalation repeats policy forwarded to daily_escalations; "every_day" is the
            Level 17 behaviour, "once_per_world" is the A-043 Level 18 policy.

    Returns:
        A dict with keys recall_at_capacity, yield_rate, genuine_harm,
        hidden_mule_recall_by_depth, time_to_detection and analyst_load.

    Raises:
        ValueError: On an unknown scorer, a missing anomaly threshold, or any invalid input that
            the called metric functions refuse (for example no mules or no genuine accounts).
    """
    if scorer not in _SCORERS:
        raise ValueError(f"Unknown scorer {scorer!r}; expected one of {sorted(_SCORERS)}")
    if scorer in ("anomaly", "rules_and_anomaly") and anomaly_threshold is None:
        raise ValueError(f"Scorer {scorer!r} requires an anomaly_threshold")

    rules_restricted: dict[AccountKey, set[date]] = {}
    if scorer in ("rules", "rules_and_anomaly"):
        rules_restricted = rules_restricted_days(scored)
    anomaly_restricted: dict[AccountKey, set[date]] = {}
    if scorer in ("anomaly", "rules_and_anomaly"):
        anomaly_restricted = anomaly_restricted_days(scored, anomaly_threshold)

    restricted = _merge_restrictions(rules_restricted, anomaly_restricted)
    escalations = daily_escalations(scored, scorer, anomaly_threshold, capacity_pct, repeat_policy=repeat_policy)
    escalation_pairs = [
        (account, day)
        for day, accounts in escalations.items()
        for account in accounts
    ]

    mules, genuine_active, depth_of, is_mule = ground_sets(scored)
    genuine_restricted = genuine_restricted_days(restricted, scored)
    flagged = set(restricted)

    daily_capacity = max(1, ceil(len({account_key(scored_day) for scored_day in scored}) * capacity_pct / 100.0))
    first_flag = first_flag_times(escalations, scored, snapshot_time_by_account_day)

    # persona_of, first_credit and reported come from this one world's truth, which knows accounts by
    # their id alone, so they are read through the key's account id. That is safe because one call
    # holds one world; the accounts themselves were counted by the full key above.
    first_flag_by_account_id = {account[2]: moment for account, moment in first_flag.items()}
    reported_accounts = {account for account in depth_of if account[2] in reported}

    return {
        "recall_at_capacity": recall_at_capacity(escalations, mules),
        "yield_rate": yield_rate(escalation_pairs, is_mule, inflation_factor),
        "genuine_harm": genuine_harm(
            genuine_restricted,
            genuine_active,
            lambda account: persona_of(account[2]),
            weights,
            default_weight,
        ),
        "hidden_mule_recall_by_depth": hidden_mule_recall_by_depth(flagged, depth_of, reported_accounts),
        "time_to_detection": time_to_detection(dict(first_credit), first_flag_by_account_id),
        "analyst_load": analyst_load(escalations, daily_capacity),
    }
