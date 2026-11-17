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
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import orjson

from decisions.mule_account_model import account_model_feature_names
from decisions.rules_config import load_rules_config
from evaluation.dataset_truth_join import join_truth
from evaluation.level17_scoring import (
    MULE_ROLES,
    daily_escalations,
    ground_sets,
    one_row_per_account_day,
    score_account_days,
)
from evaluation.rules_metrics import recall_at_capacity, yield_rate
from evaluation.run_level_17 import load_evaluation_settings

__all__ = ["FEATURE_FILL_FLAG_SHARE", "build_sanity_report", "feature_fill_report", "main", "oracle_report"]

REPO_ROOT = Path(__file__).resolve().parent.parent

# policy: a feature carried by fewer than 99.9% of the rows is flagged as sparse in the report.
FEATURE_FILL_FLAG_SHARE: float = 0.999

ORACLE_MULE_SCORE: float = 1.0
ORACLE_GENUINE_SCORE: float = 0.0
ORACLE_THRESHOLD: float = 0.5


def feature_fill_report(snapshot_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    WHAT: Reports, for every account-model feature, the share of snapshot rows that carry a value.

    WHY: A feature that is None on most rows is either not produced by the replay or not needed;
    either way it must be visible before the model treats it as missing.

    Args:
        snapshot_rows: Snapshot rows carrying the account-model feature columns.

    Returns:
        A dict with the row and feature counts, the flag threshold, every flagged feature with its
        fill share, and the fill share of every feature.

    Raises:
        ValueError: If no rows are given.
    """
    rows = list(snapshot_rows)
    if not rows:
        raise ValueError("feature_fill_report requires at least one snapshot row")
    feature_names = account_model_feature_names()
    fill_share: dict[str, float] = {}
    for name in feature_names:
        filled = sum(1 for row in rows if row.get(name) is not None)
        fill_share[name] = filled / len(rows)
    flagged = [
        {"feature": name, "fill_share": share} for name, share in fill_share.items() if share < FEATURE_FILL_FLAG_SHARE
    ]
    return {
        "row_count": len(rows),
        "feature_count": len(feature_names),
        "feature_fill_flag_share": FEATURE_FILL_FLAG_SHARE,
        "flagged_features": flagged,
        "fill_share_by_feature": fill_share,
    }


def oracle_report(snapshot_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    WHAT: Scores one world's account-days with the perfect oracle and measures what the pilot's
    escalation and metric functions make of it.

    WHY: The same daily_escalations, recall_at_capacity, yield_rate and ground_sets calls the pilot
    uses on test worlds are applied to the oracle, so its numbers are a ceiling under the same
    capacity policy rather than a parallel metric implementation.

    Args:
        snapshot_rows: Joined snapshot rows of one world, carrying bank_id, account_id,
            snapshot_time, snapshot_kind, truth_role and truth_depth.

    Returns:
        A dict with the account-day and mule-account counts, the number of escalations, recall at
        capacity, and the measured and reweighted yield.

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
        _, snapshots = join_truth(dataset_dir, truth_root)
        rows = snapshots.to_pylist()
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
                    "oracle": oracle_report(world_rows),
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
