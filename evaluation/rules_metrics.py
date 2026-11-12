"""
WHAT: Pure scoring functions that measure the rules-based detection baseline against hidden
truth passed in by the caller.

WHY: The rules baseline must be quantified in the same terms later models will be judged by:
mule recall at the available review capacity, yield corrected for escalation inflation, harm to
genuinely active people, hidden-mule recall by depth, detection speed, and analyst workload.

TERM: IST calendar date: the calendar day (in India Standard Time) that an event belongs to,
represented as a datetime.date. Callers convert timestamps to IST before calling these functions.

Every function here is pure: it performs no file or truth access and does not mutate its inputs.
Invalid, missing, or undefined inputs raise a plain ValueError instead of returning a silent
fallback, so a broken caller is never reported as a number.
"""

from collections.abc import Callable
from datetime import date, datetime

import numpy as np


def recall_at_capacity(daily_escalations: dict[date, list[str]], mule_accounts: set[str]) -> float:
    """
    WHAT: Share of known mule accounts that were escalated for review on at least one day.

    WHY: Review capacity is limited per day, so recall must be measured against what the rules
    engine actually managed to escalate, not against what it could have escalated.

    Args:
        daily_escalations: Mapping from IST calendar date to the account ids escalated that day.
            An account repeated within the same day is counted once.
        mule_accounts: Account ids that truth marks as mules.

    Returns:
        Fraction of mule accounts escalated on at least one day, in [0.0, 1.0].

    Errors:
        Raises ValueError when mule_accounts is empty, because recall against no mules is
        undefined and must not be reported as a number.
    """
    if not mule_accounts:
        raise ValueError("recall_at_capacity requires at least one mule account")
    escalated_accounts: set[str] = set()
    for accounts in daily_escalations.values():
        escalated_accounts.update(accounts)
    recalled_mules = escalated_accounts & mule_accounts
    return len(recalled_mules) / len(mule_accounts)


def yield_rate(
    escalations: list[tuple[str, date]],
    is_mule: Callable[[str, date], bool],
    inflation_factor: float,
) -> dict[str, float | None]:
    """
    WHAT: Precision of daily escalations, measured directly and after deflating mule hits.

    WHY: Rules escalate far more genuine-looking accounts than truth admits as mules, so the raw
    measured yield overstates usefulness. Dividing the true positives by the configured inflation
    factor estimates yield if each reported mule really stood for that many escalations.

    Args:
        escalations: List of (account_id, IST calendar date) pairs, one per escalation.
        is_mule: Callable deciding whether an account was a mule on the given date.
        inflation_factor: Positive divisor applied to true positives in the reweighted yield.

    Returns:
        ``{"measured": TP / (TP + FP), "reweighted": (TP / f) / (TP / f + FP)}`` where TP counts
        escalations for mules and FP counts escalations for non-mules. Both values are None when
        no escalations were given, because a yield over an empty set is undefined.

    Errors:
        Raises ValueError when inflation_factor is not strictly positive.
    """
    if inflation_factor <= 0:
        raise ValueError("yield_rate requires inflation_factor > 0")
    if not escalations:
        return {"measured": None, "reweighted": None}
    true_positives = 0
    false_positives = 0
    for account_id, escalation_date in escalations:
        if is_mule(account_id, escalation_date):
            true_positives += 1
        else:
            false_positives += 1
    measured = true_positives / (true_positives + false_positives)
    weighted_true_positives = true_positives / inflation_factor
    reweighted = weighted_true_positives / (weighted_true_positives + false_positives)
    return {"measured": measured, "reweighted": reweighted}


def genuine_harm(
    restricted_days: dict[str, set[date]],
    genuine_active: set[str],
    persona_of: Callable[[str], str],
    weights: dict[str, int],
    default_weight: int,
) -> dict[str, float | int | dict[str, str]]:
    """
    WHAT: Counts and weights restrictions placed on genuinely active people, plus placeholders
    for harm that later levels will supply.

    WHY: A baseline that scores well on mule recall can still be unacceptable if it freezes the
    accounts of genuine payers. Weighting each restricted account by persona (for example
    retirees weigh more than small shopkeepers) reflects the uneven real-world cost of a freeze.

    Args:
        restricted_days: Mapping from account id to the set of IST calendar dates during which
            that account was restricted. Accounts with an empty day set are not counted.
        genuine_active: Accounts that truth marks as genuinely active. Only these can be harmed;
            restricted accounts outside this set are ignored entirely.
        persona_of: Callable returning the persona label of an account. It is called only for
            restricted accounts that are in genuine_active.
        weights: Persona label to harm weight. A persona missing here uses default_weight.
        default_weight: Harm weight for restricted accounts whose persona is not in weights.

    Returns:
        A dict with:
        - ``restricted_accounts``: number of genuinely active accounts restricted on any day.
        - ``restriction_days``: total restricted account-days across those accounts.
        - ``weighted_restriction_days``: restricted account-days multiplied by the account's
          persona weight, summed over accounts.
        - ``genuine_restriction_rate``: restricted_accounts divided by the number of genuinely
          active accounts.
        - ``unknown_persona_count``: restricted active accounts whose persona is not in weights.
        - ``downstream_shops``, ``lien_release_wait_days``, ``warnings_to_genuine_payers``: each
          ``{"status": "not run", "reason": "needs liens/warnings from later levels"}``.

    Errors:
        Raises ValueError when genuine_active is empty, because a rate over no people is
        undefined and must not be reported as a number.
    """
    if not genuine_active:
        raise ValueError("genuine_harm requires at least one genuinely active account")
    restricted_accounts = 0
    restriction_days = 0
    weighted_restriction_days = 0
    unknown_persona_count = 0
    for account_id, days in restricted_days.items():
        if account_id not in genuine_active or not days:
            continue
        persona = persona_of(account_id)
        if persona in weights:
            weight = weights[persona]
        else:
            weight = default_weight
            unknown_persona_count += 1
        restricted_accounts += 1
        restriction_days += len(days)
        weighted_restriction_days += len(days) * weight
    return {
        "restricted_accounts": restricted_accounts,
        "restriction_days": restriction_days,
        "weighted_restriction_days": weighted_restriction_days,
        "genuine_restriction_rate": restricted_accounts / len(genuine_active),
        "unknown_persona_count": unknown_persona_count,
        "downstream_shops": {"status": "not run", "reason": "needs liens/warnings from later levels"},
        "lien_release_wait_days": {"status": "not run", "reason": "needs liens/warnings from later levels"},
        "warnings_to_genuine_payers": {"status": "not run", "reason": "needs liens/warnings from later levels"},
    }


def hidden_mule_recall_by_depth(
    flagged: set[str],
    depth_of: dict[str, int],
    reported: set[str],
) -> dict[int, float]:
    """
    WHAT: For mules never reported to the national registry, the share flagged for review at
    each mule depth.

    WHY: Deeper mules are the ones the rules engine is most likely to miss. Measuring recall
    only among hidden (never reported) mules shows whether flagging finds them at all instead
    of leaning on the registry reports it already has.

    Args:
        flagged: Account ids flagged for review by the baseline.
        depth_of: Mapping from mule account id to its mule depth.
        reported: Account ids already reported to the national registry.

    Returns:
        Mapping from depth to recall among hidden mules at that depth. A depth appears only when
        at least one mule at that depth is unreported. Recall is flagged hidden mules divided by
        hidden mules at the depth.

    Side effects:
        None. Flagged accounts that are not mules in depth_of are ignored rather than raising,
        because the flag list may legitimately contain genuine accounts.
    """
    hidden_mules_by_depth: dict[int, list[str]] = {}
    for account_id, depth in depth_of.items():
        if account_id not in reported:
            hidden_mules_by_depth.setdefault(depth, []).append(account_id)
    recall_by_depth: dict[int, float] = {}
    for depth, hidden_mules in hidden_mules_by_depth.items():
        flagged_hidden_mules = 0
        for account_id in hidden_mules:
            if account_id in flagged:
                flagged_hidden_mules += 1
        recall_by_depth[depth] = flagged_hidden_mules / len(hidden_mules)
    return recall_by_depth


def time_to_detection(
    first_illicit_credit: dict[str, datetime],
    first_flag: dict[str, datetime],
) -> dict[str, float | int]:
    """
    WHAT: Median, p90, and detected/undetected counts for the hours between a mule's first
    illicit credit and its first detection flag.

    WHY: Recall alone hides how late a mule is caught; every hour a mule stays active is money
    lost. Reporting the median and the p90 tail shows both the typical and the worst delays.

    Args:
        first_illicit_credit: Mapping from mule account id to the datetime of its first illicit
            credit. These accounts define the evaluated population.
        first_flag: Mapping from account id to the datetime of its first detection flag.

    Returns:
        A dict with:
        - ``median``: median hours from credit to flag over detected mules.
        - ``p90``: 90th percentile hours (numpy linear interpolation) over detected mules.
        - ``detected_count``: mules present in first_flag.
        - ``undetected_count``: mules absent from first_flag.
        A flag that precedes the first illicit credit counts as 0 hours: the account was flagged
        before it started acting as a mule, so no waiting time is attributed to detection.

    Errors:
        Raises ValueError when first_illicit_credit is empty or when no mule has a flag, because
        percentiles over no detected mules are undefined.
    """
    if not first_illicit_credit:
        raise ValueError("time_to_detection requires at least one illicit credit")
    detection_hours: list[float] = []
    undetected_count = 0
    for account_id, credit_time in first_illicit_credit.items():
        if account_id not in first_flag:
            undetected_count += 1
            continue
        hours = (first_flag[account_id] - credit_time).total_seconds() / 3600.0
        detection_hours.append(max(hours, 0.0))
    if not detection_hours:
        raise ValueError("time_to_detection requires at least one detected mule")
    return {
        "median": float(np.percentile(detection_hours, 50, method="linear")),
        "p90": float(np.percentile(detection_hours, 90, method="linear")),
        "detected_count": len(detection_hours),
        "undetected_count": undetected_count,
    }


def analyst_load(daily_escalations: dict[date, list[str]], capacity: int) -> dict[str, float | int]:
    """
    WHAT: Mean and peak daily escalation counts and the mean share of review capacity used.

    WHY: A baseline is only usable if analysts can review what it escalates. Comparing average
    load with the configured daily capacity shows whether the rules fit the review team.

    Args:
        daily_escalations: Mapping from IST calendar date to the account ids escalated that day.
        capacity: Daily number of escalations the review team can handle.

    Returns:
        ``{"mean": total escalations / number of days, "max": largest daily count,
        "mean_share_of_capacity": mean / capacity}``.

    Errors:
        Raises ValueError when capacity is not strictly positive or when daily_escalations is
        empty, because neither share of capacity nor a mean over no days is defined.
    """
    if capacity <= 0:
        raise ValueError("analyst_load requires capacity > 0")
    if not daily_escalations:
        raise ValueError("analyst_load requires at least one day of escalations")
    daily_counts = [len(accounts) for accounts in daily_escalations.values()]
    mean = sum(daily_counts) / len(daily_counts)
    return {
        "mean": mean,
        "max": max(daily_counts),
        "mean_share_of_capacity": mean / capacity,
    }


def summarise_across_worlds(values: list[float], min_worlds: int) -> dict[str, float | int | str]:
    """
    WHAT: Median and spread of a metric across simulated worlds.

    WHY: A single world can flatter or condemn the baseline. Reporting the median and the
    10th-to-90th-percentile band shows how stable the result is, but only once enough worlds
    exist for those percentiles to mean anything.

    Args:
        values: Metric value from each world.
        min_worlds: Minimum number of worlds required before p10 and p90 are reported.

    Returns:
        A dict with ``world_count`` and ``median`` (numpy linear interpolation) always.
        When world_count >= min_worlds it also carries ``p10`` and ``p90`` (numpy linear
        interpolation); otherwise it carries ``"spread": "not enough worlds"``.

    Errors:
        Raises ValueError when values is empty, because a median over no worlds is undefined.
    """
    if not values:
        raise ValueError("summarise_across_worlds requires at least one world value")
    summary: dict[str, float | int | str] = {
        "world_count": len(values),
        "median": float(np.percentile(values, 50, method="linear")),
    }
    if len(values) >= min_worlds:
        summary["p10"] = float(np.percentile(values, 10, method="linear"))
        summary["p90"] = float(np.percentile(values, 90, method="linear"))
    else:
        summary["spread"] = "not enough worlds"
    return summary
