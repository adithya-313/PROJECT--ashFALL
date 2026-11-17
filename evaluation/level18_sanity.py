"""
WHAT: Level 18a4 sanity checks on a dataset split: how many rows actually carry each model feature,
and how well a perfect oracle would score through the pilot's own escalation and metric path.

WHY: A model cannot learn a feature that is missing from most rows, and no ranking can beat the
review capacity. Measuring both before the model is trusted separates "the model is weak" from
"the inputs or the capacity set the ceiling".

TERM: account-day: one account's latest snapshot on one IST calendar day.
TERM: oracle: a score that is 1.0 for every account-day whose hidden truth role is a mule role and
0.0 otherwise. It is a diagnostic ceiling, never a model.

The command line writes one JSON report. Missing or malformed inputs raise a plain ValueError or
FileNotFoundError; nothing is guessed.
"""

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import orjson

from decisions.mule_account_model import account_model_feature_names, load_mule_model_config
from decisions.rules_config import load_rules_config
from evaluation.dataset_truth_join import join_truth
from evaluation.level17_scoring import (
    MULE_ROLES,
    ScoredDay,
    daily_escalations,
    ground_sets,
    one_row_per_account_day,
    score_account_days,
)
from evaluation.rules_metrics import recall_at_capacity, yield_rate
from evaluation.run_level_17 import load_evaluation_settings

__all__ = ["build_sanity_report", "feature_fill_report", "main", "oracle_report"]

REPO_ROOT = Path(__file__).resolve().parent.parent

# policy: a feature missing from more than half the rows is listed as mostly missing, as information.
MOSTLY_MISSING_SHARE: float = 0.5

ORACLE_MULE_SCORE: float = 1.0
ORACLE_GENUINE_SCORE: float = 0.0
ORACLE_THRESHOLD: float = 0.5


def feature_fill_report(
    snapshot_rows: Sequence[Mapping[str, Any]],
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    WHAT: Reports, for every account-model feature, how many rows carry a value and how concentrated
    its values are, then flags the features that cannot separate anything.

    WHY: A feature missing from every row, or holding the same value in almost every row, cannot
    tell two accounts apart; both must be visible before the model treats them as informative. The
    concentration cut is policy read from config/mule_model.yaml, never typed into this module.

    Args:
        snapshot_rows: Snapshot rows carrying the account-model feature columns.
        config_path: Optional mule model config path; defaults to config/mule_model.yaml.

    Returns:
        A dict with the row and feature counts, the configured flag share, the flagged features,
        the mostly-missing features, and one stats block per feature. A feature is flagged when it
        is missing from every row or when its most common value covers at least the flag share, a
        missing value counting as one value. A feature is listed as mostly missing, without being
        flagged for that reason alone, when it is missing from more than half the rows.

    Raises:
        ValueError: If no rows are given or a feature value cannot be counted as one value.
    """
    rows = list(snapshot_rows)
    if not rows:
        raise ValueError("feature_fill_report requires at least one snapshot row")
    flag_share = load_mule_model_config(config_path).feature_fill_flag_share
    feature_names = account_model_feature_names()
    stats_by_feature: dict[str, dict[str, Any]] = {}
    flagged: list[dict[str, Any]] = []
    mostly_missing: list[dict[str, Any]] = []
    for name in feature_names:
        try:
            counts: Counter[Any] = Counter(row.get(name) for row in rows)
        except TypeError as error:
            raise ValueError(f"Feature {name!r} holds a value that is not hashable, so it cannot be counted") from error
        most_common_value, most_common_count = counts.most_common(1)[0]
        row_count = len(rows)
        missing_count = counts.get(None, 0)
        stats = {
            "rows": row_count,
            "fill_share": (row_count - missing_count) / row_count,
            "missing_share": missing_count / row_count,
            "most_common_value": most_common_value,
            "most_common_share": most_common_count / row_count,
            "distinct_count": len(counts),
        }
        stats_by_feature[name] = stats
        if stats["missing_share"] == 1.0 or stats["most_common_share"] >= flag_share:
            flagged.append({"feature": name, **stats})
        if stats["missing_share"] > MOSTLY_MISSING_SHARE:
            mostly_missing.append(
                {"feature": name, "missing_share": stats["missing_share"], "fill_share": stats["fill_share"]}
            )
    return {
        "row_count": len(rows),
        "feature_count": len(feature_names),
        "feature_fill_flag_share": flag_share,
        "mostly_missing_share": MOSTLY_MISSING_SHARE,
        "flagged_features": flagged,
        "mostly_missing_features": mostly_missing,
        "feature_stats": stats_by_feature,
    }


def _truth_mule_accounts(payment_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """
    WHAT: Returns the accounts that at least one joined payment row marks as a mule-role receiver.

    WHY: The recall ground set only holds mules that have a snapshot day with a mule role. Payment
    truth can also name a mule whose snapshots never fall on such a day, and the miss breakdown must
    say so instead of dropping that account.

    Args:
        payment_rows: Joined payment rows of one world carrying receiver_account_id and
            truth_role_receiver.

    Returns:
        Account ids with truth_role_receiver in MULE_ROLES.

    Raises:
        ValueError: If a row lacks a non-empty receiver_account_id or a string truth_role_receiver.
    """
    accounts: set[str] = set()
    for row in payment_rows:
        account_id = row.get("receiver_account_id")
        role = row.get("truth_role_receiver")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("oracle_report requires a non-empty string 'receiver_account_id' on every payment row")
        if not isinstance(role, str):
            raise ValueError("oracle_report requires a string 'truth_role_receiver' on every payment row")
        if role in MULE_ROLES:
            accounts.add(account_id)
    return accounts


def _oracle_misses(
    scored: Sequence[ScoredDay],
    escalations: Mapping[date, Sequence[str]],
    mules: set[str],
    truth_mules: set[str],
) -> dict[str, Any]:
    """
    WHAT: Explains which mules the oracle never escalated and why the capacity slots went elsewhere.

    WHY: Recall below 1.0 under a perfect ranking can only come from missing snapshot days or from
    the review capacity being spent on accounts already escalated, so the report separates the two
    instead of leaving recall unexplained. It reads the escalations the pilot already produced; no
    metric or ranking rule changes here.

    Args:
        scored: Scored account-days of one world.
        escalations: IST date -> escalated account ids, as returned by daily_escalations.
        mules: The recall ground set, as returned by ground_sets.
        truth_mules: Mule accounts named by joined payment truth.

    Returns:
        A dict with the ground-set size, the found count, the two missed groups and their accounts,
        the capacity slots already taken by mules escalated on an earlier day when a missed mule was
        due, the totals of both counts, and the total and repeated escalations. A missed mule is in
        the mule-day group when the ground set holds it and in the no-mule-day group otherwise. For
        each missed mule and each of its mule days, the blocked slots are that day's escalations of
        an account escalated on a strictly earlier day, and the slots are all of that day's
        escalations.
    """
    escalated_accounts = {account_id for account_ids in escalations.values() for account_id in account_ids}
    found = mules & escalated_accounts
    missed = (mules | truth_mules) - escalated_accounts
    missed_with_mule_days = missed & mules
    missed_without_mule_day = missed - mules

    first_escalation_day: dict[str, date] = {}
    repeat_escalations = 0
    total_escalations = 0
    for day in sorted(escalations):
        for account_id in escalations[day]:
            total_escalations += 1
            earlier_day = first_escalation_day.get(account_id)
            if earlier_day is not None and earlier_day < day:
                repeat_escalations += 1
            elif earlier_day is None:
                first_escalation_day[account_id] = day

    mule_days_by_account: dict[str, set[date]] = {}
    for scored_day in scored:
        if scored_day.role in MULE_ROLES:
            mule_days_by_account.setdefault(scored_day.account_id, set()).add(scored_day.day)

    blocked_slots = 0
    missed_mule_day_slots = 0
    for account_id in sorted(missed_with_mule_days):
        for day in sorted(mule_days_by_account[account_id]):
            day_escalations = escalations.get(day, [])
            missed_mule_day_slots += len(day_escalations)
            blocked_slots += sum(
                1
                for escalated_id in day_escalations
                if escalated_id in first_escalation_day and first_escalation_day[escalated_id] < day
            )

    return {
        "ground_set_mules": len(mules),
        "truth_mule_accounts": len(mules | truth_mules),
        "found_mules": len(found),
        "missed_without_mule_day": sorted(missed_without_mule_day),
        "missed_with_mule_days": sorted(missed_with_mule_days),
        "blocked_slots_on_missed_mule_days": blocked_slots,
        "missed_mule_day_slots": missed_mule_day_slots,
        "blocked_slot_share": (blocked_slots / missed_mule_day_slots) if missed_mule_day_slots else 0.0,
        "total_escalations": total_escalations,
        "repeat_escalations": repeat_escalations,
        "repeat_escalation_share": (repeat_escalations / total_escalations) if total_escalations else 0.0,
    }


def oracle_report(
    snapshot_rows: Sequence[Mapping[str, Any]],
    payment_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    WHAT: Scores one world's account-days with the perfect oracle and measures what the pilot's
    escalation and metric functions make of it.

    WHY: The same daily_escalations, recall_at_capacity, yield_rate and ground_sets calls the pilot
    uses on test worlds are applied to the oracle, so its numbers are a ceiling under the same
    capacity policy rather than a parallel metric implementation.

    Args:
        snapshot_rows: Joined snapshot rows of one world, carrying bank_id, account_id,
            snapshot_time, snapshot_kind, truth_role and truth_depth.
        payment_rows: Joined payment rows of the same world, carrying receiver_account_id and
            truth_role_receiver, used only to explain mules the oracle misses.

    Returns:
        A dict with the account-day and mule-account counts, the number of escalations, recall at
        capacity, the measured and reweighted yield, and the miss breakdown.

    Raises:
        ValueError: If the rows are malformed or hold no mule account.
    """
    day_rows = one_row_per_account_day(list(snapshot_rows))
    base_scored = score_account_days(day_rows, load_rules_config(), None)
    scored = [
        replace(
            day,
            anomaly_score=ORACLE_MULE_SCORE if day.role in MULE_ROLES else ORACLE_GENUINE_SCORE,
        )
        for day in base_scored
    ]
    settings = load_evaluation_settings()
    escalations = daily_escalations(
        scored,
        "anomaly",
        ORACLE_THRESHOLD,
        settings.capacity_pct_active_accounts_per_day,
    )
    escalation_pairs = [(account_id, day) for day, account_ids in escalations.items() for account_id in account_ids]
    mules, _, _, is_mule = ground_sets(scored)
    truth_mules = _truth_mule_accounts(payment_rows)
    # An inflation factor of 1.0 leaves the measured yield unchanged; the CLI has no world cards,
    # so the oracle reports the measured yield only.
    oracle_yield = yield_rate(escalation_pairs, is_mule, 1.0)
    return {
        "account_days": len(scored),
        "mule_accounts": len(mules),
        "escalations": len(escalation_pairs),
        "recall_at_capacity": recall_at_capacity(escalations, mules),
        "yield_measured": oracle_yield["measured"],
        "yield_reweighted": oracle_yield["reweighted"],
        "misses": _oracle_misses(scored, escalations, mules, truth_mules),
    }


def _manifest_for(dataset_dir: Path) -> Mapping[str, Any]:
    """
    WHAT: Reads a dataset manifest.

    Raises:
        FileNotFoundError: If the manifest is missing.
        ValueError: If it is not a mapping or has no 'worlds' sequence.
    """
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {path}")
    manifest = orjson.loads(path.read_bytes())
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Dataset manifest {path} must hold a mapping")
    if not isinstance(manifest.get("worlds"), list):
        raise ValueError(f"Dataset manifest {path} requires a 'worlds' sequence")
    return manifest


def build_sanity_report(validation_dirs: Sequence[Path], truth_root: Path) -> dict[str, Any]:
    """
    WHAT: Builds the feature-fill and oracle sanity report for the given validation datasets.

    WHY: One entry point keeps the feature check and the oracle check on the same joined rows, so
    both describe exactly the split a reader passed in.

    Args:
        validation_dirs: Validation dataset folders.
        truth_root: Root folder holding one truth folder per run_id.

    Returns:
        A dict with the checked dataset names, the feature-fill report over every row, and one
        oracle block per world.

    Raises:
        ValueError: If no dataset is given or a world id is absent from its snapshots.
        FileNotFoundError: If the truth or a dataset manifest is missing.
    """
    if not validation_dirs:
        raise ValueError("build_sanity_report requires at least one validation dataset folder")
    all_rows: list[dict[str, Any]] = []
    oracle_blocks: list[dict[str, Any]] = []
    for dataset_dir in validation_dirs:
        manifest = _manifest_for(dataset_dir)
        payments, snapshots = join_truth(dataset_dir, truth_root)
        rows = snapshots.to_pylist()
        payment_rows = payments.to_pylist()
        all_rows.extend(rows)
        for world in manifest["worlds"]:
            world_id = str(world["world_id"])
            world_rows = [row for row in rows if row.get("world_id") == world_id]
            if not world_rows:
                raise ValueError(f"World {world_id!r} has no snapshot rows in {dataset_dir}")
            oracle_blocks.append(
                {
                    "dataset": dataset_dir.name,
                    "world_id": world_id,
                    "oracle": oracle_report(
                        world_rows,
                        [row for row in payment_rows if row.get("world_id") == world_id],
                    ),
                }
            )
    return {
        "validation_datasets": [dataset_dir.name for dataset_dir in validation_dirs],
        "feature_fill": feature_fill_report(all_rows),
        "oracle_by_world": oracle_blocks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command-line entry point; writes the sanity report as JSON.

    Args:
        argv: Optional argument list; defaults to sys.argv.

    Returns:
        Process exit code 0.
    """
    parser = argparse.ArgumentParser(description="Report feature fill and oracle ceilings on a validation split")
    parser.add_argument(
        "--validation", type=Path, action="append", required=True, help="validation dataset folder; repeatable"
    )
    parser.add_argument(
        "--truth-root", type=Path, required=True, help="root folder holding one truth folder per run_id"
    )
    parser.add_argument("--out", type=Path, required=True, help="destination JSON report")
    args = parser.parse_args(argv)

    report = build_sanity_report(tuple(args.validation), args.truth_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(orjson.dumps(report, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
