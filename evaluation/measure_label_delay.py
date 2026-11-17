"""
WHAT: Measures how long after a mule's first illicit credit the bank's own labels arrive, pooled over
the training worlds, and counts the mules whose credit never got a label.

WHY: A training row near the end of a world sits in a window where labels have not had time to
arrive, so its positives are undercounted. Measuring the delay with the same first_illicit_credit
clock the evaluation uses shows how wide that immature window is before it is configured.

TERM: label delay: first label arrival minus first illicit credit, in days.
TERM: censored mule: an account with a first illicit credit but no bank-visible label in its world.

Missing or malformed inputs raise a plain ValueError; nothing is guessed.
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from evaluation.dataset_truth_join import join_truth
from evaluation.level17_evaluation import first_illicit_credit
from evaluation.rules_metrics import percentile_linear
from evaluation.truth import load_truth
from replay.dataset_builder import load_dataset

__all__ = ["DELAY_PERCENTILES", "build_label_delay_report", "main", "measure_label_delay"]

REPO_ROOT = Path(__file__).resolve().parent.parent

DELAY_PERCENTILES: tuple[float, ...] = (0.5, 0.9, 0.95)

SECONDS_PER_DAY: float = 86400.0


def _require_string(row: Mapping[str, Any], key: str) -> str:
    """
    WHAT: Returns a required non-empty string field from a label row.

    Raises:
        ValueError: If the field is missing, empty, or not a string.
    """
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"measure_label_delay requires a non-empty string '{key}' on every label row")
    return value


def _require_datetime(row: Mapping[str, Any], key: str) -> datetime:
    """
    WHAT: Returns a required timezone-aware datetime field from a label row.

    Raises:
        ValueError: If the field is missing, not a datetime, or timezone-naive.
    """
    value = row.get(key)
    if not isinstance(value, datetime):
        raise ValueError(f"measure_label_delay requires a datetime '{key}' on every label row")
    if value.tzinfo is None:
        raise ValueError(f"measure_label_delay requires a timezone-aware '{key}'")
    return value


def _percentiles(sorted_days: list[float]) -> dict[str, float]:
    """
    WHAT: Returns the configured percentiles of an already sorted delay list.

    Args:
        sorted_days: Delays in days, ascending. Empty returns an empty mapping.

    Returns:
        Percentile name (for example "p90") -> delay in days.
    """
    if not sorted_days:
        return {}
    return {f"p{round(fraction * 100)}": percentile_linear(sorted_days, fraction) for fraction in DELAY_PERCENTILES}


def measure_label_delay(
    payment_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    story_of: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """
    WHAT: Measures the delay from each labelled mule's first illicit credit to its first bank label.

    WHY: first_illicit_credit defines the detection clock for the evaluation, so using it here keeps
    the measured delay on that same clock; a mule with no label is reported as censored instead of
    being silently left out.

    Args:
        payment_rows: Joined payment rows of one world, as first_illicit_credit expects.
        label_rows: Bank-visible label rows of the same world, carrying account_id and arrival_time.
        story_of: Callable returning the payment story for a payment_id.

    Returns:
        A dict with the number of accounts holding a first credit, the labelled count, the sorted
        delay list in days, the percentiles, the min and max, and the sorted censored accounts.

    Raises:
        ValueError: If a label row lacks account_id or a timezone-aware arrival_time.
    """
    first_credit = first_illicit_credit(payment_rows, story_of)
    first_label: dict[str, datetime] = {}
    for row in label_rows:
        account_id = _require_string(row, "account_id")
        arrival = _require_datetime(row, "arrival_time")
        current = first_label.get(account_id)
        if current is None or arrival < current:
            first_label[account_id] = arrival

    delays: list[float] = []
    censored: list[str] = []
    for account_id, credit_moment in first_credit.items():
        labelled_at = first_label.get(account_id)
        if labelled_at is None:
            censored.append(account_id)
            continue
        delays.append((labelled_at - credit_moment).total_seconds() / SECONDS_PER_DAY)
    sorted_delays = sorted(delays)
    return {
        "accounts_with_first_credit": len(first_credit),
        "labelled_accounts": len(sorted_delays),
        "censored_count": len(censored),
        "censored_accounts": sorted(censored),
        "delays_days": sorted_delays,
        "delay_percentiles_days": _percentiles(sorted_delays),
        "min_delay_days": sorted_delays[0] if sorted_delays else None,
        "max_delay_days": sorted_delays[-1] if sorted_delays else None,
    }


def build_label_delay_report(training_dirs: Sequence[Path], truth_root: Path) -> dict[str, Any]:
    """
    WHAT: Builds the per-world and pooled label-delay report for the given training datasets.

    WHY: The maturity window is a per-world property, but the configured value needs one pooled
    distribution over all measured worlds, so both views are reported together.

    Args:
        training_dirs: Training dataset folders.
        truth_root: Root folder holding one truth folder per run_id.

    Returns:
        A dict with the checked dataset names, the pooled statistics and one block per world.

    Raises:
        ValueError: If no dataset is given, or a manifest has no 'worlds' sequence.
        FileNotFoundError: If the truth or a dataset file is missing.
    """
    if not training_dirs:
        raise ValueError("build_label_delay_report requires at least one training dataset folder")
    by_world: list[dict[str, Any]] = []
    pooled_delays: list[float] = []
    for dataset_dir in training_dirs:
        dataset = load_dataset(dataset_dir)
        worlds = dataset.manifest.get("worlds")
        if not isinstance(worlds, list):
            raise ValueError(f"Dataset manifest {dataset_dir / 'manifest.json'} requires a 'worlds' sequence")
        payments, _ = join_truth(dataset_dir, truth_root)
        payment_rows = payments.to_pylist()
        label_rows = dataset.labels.to_pylist()
        for world in worlds:
            world_id = str(world["world_id"])
            run_id = str(world["run_id"])
            truth_store = load_truth(truth_root, run_id)
            result = measure_label_delay(
                [row for row in payment_rows if row["world_id"] == world_id],
                [row for row in label_rows if row["world_id"] == world_id],
                truth_store.story_of,
            )
            by_world.append({"dataset": dataset_dir.name, "world_id": world_id, **result})
            pooled_delays.extend(result["delays_days"])
    pooled_sorted = sorted(pooled_delays)
    return {
        "training_datasets": [dataset_dir.name for dataset_dir in training_dirs],
        "pooled": {
            "accounts_with_first_credit": sum(block["accounts_with_first_credit"] for block in by_world),
            "labelled_accounts": len(pooled_sorted),
            "censored_count": sum(block["censored_count"] for block in by_world),
            "delay_percentiles_days": _percentiles(pooled_sorted),
            "min_delay_days": pooled_sorted[0] if pooled_sorted else None,
            "max_delay_days": pooled_sorted[-1] if pooled_sorted else None,
        },
        "by_world": by_world,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command-line entry point; writes the label-delay report as JSON.

    Args:
        argv: Optional argument list; defaults to sys.argv.

    Returns:
        Process exit code 0.
    """
    parser = argparse.ArgumentParser(description="Measure label delay on training datasets")
    parser.add_argument(
        "--training", type=Path, action="append", required=True, help="training dataset folder; repeatable"
    )
    parser.add_argument(
        "--truth-root", type=Path, required=True, help="root folder holding one truth folder per run_id"
    )
    parser.add_argument("--out", type=Path, required=True, help="destination JSON report")
    args = parser.parse_args(argv)

    report = build_label_delay_report(tuple(args.training), args.truth_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(orjson.dumps(report, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
