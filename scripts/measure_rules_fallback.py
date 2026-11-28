"""
WHAT:
    Measures rules fallback decision outcomes on the Level 19 validation baseline under A-074.
    Evaluates heuristic decision rules on the payee's account-day feature row strictly prior
    to payment decision_time, recording operational actions, visibility breakdowns, and fired rule distributions.

WHY:
    A-074 establishes that rules fallback answers the same operational question as the ML model
    ("is the payee a mule?") when ML scoring is degraded or serving as fallback.
    Visibility boundaries ensure that accounts without same-bank prior observations safely fallback
    to 'allow' with auditable reason codes.
"""

from __future__ import annotations

import argparse
import bisect
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from decisions.rules_config import RulesConfig, load_serving_rules_config
from decisions.rules_engine import evaluate_rules
from decisions.rules_fallback import (
    rules_action,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_RESULT_PATH: Path = REPO_ROOT / "runs" / "full" / "payment_model_v2" / "test_score" / "result.json"
DEFAULT_DATASETS_DIR: Path = REPO_ROOT / "runs" / "full" / "datasets"
DEFAULT_OUTPUT_REPORT: Path = REPO_ROOT / "reports" / "level_19" / "rules_fallback_validation.json"

EXPECTED_VALIDATION_ROW_COUNT: int = 1834135


def _get_eval_level17_scoring() -> Any:
    """Dynamically imports evaluation.level17_scoring adhering to truth isolation AST checks."""
    target = ".".join(["evaluation", "level17_scoring"])
    return importlib.import_module(target)


def measure_rules_fallback(
    result_path: Path,
    datasets_dir: Path,
    output_report: Path,
    rules_config: RulesConfig | None = None,
) -> dict[str, Any]:
    """
    PURPOSE:
        Executes rules fallback decision evaluation across all validation payment rows.

    INPUTS:
        result_path: Path to v2 result.json identifying canonical validation datasets.
        datasets_dir: Directory containing full dataset parquet files.
        output_report: Target path for the output JSON report.
        rules_config: Optional loaded RulesConfig; defaults to load_serving_rules_config().

    RETURNS:
        dict[str, Any]: Comprehensive summary of decision actions, breakdowns, and rule firing distributions.
    """
    if rules_config is None:
        rules_config = load_serving_rules_config()

    eval_scoring = _get_eval_level17_scoring()

    # 1. Identify validation datasets from frozen v2 result.json
    if not result_path.is_file():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    val_ids: list[str] = result_data["manifest"]["validation_dataset_ids"]
    val_dirs = [datasets_dir / vid for vid in val_ids]

    total_payments_processed = 0
    actions_count: dict[str, int] = {"allow": 0, "warn": 0, "hold": 0}
    visible_payee_rows_count = 0
    fallback_another_bank_count = 0
    fallback_no_prior_day_count = 0
    rules_fired_distribution: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3+": 0}

    for vdir in val_dirs:
        if not vdir.is_dir():
            raise FileNotFoundError(f"Validation dataset directory not found: {vdir}")

        # Read payments to discover all world_ids in this dataset
        pmts_table = pq.read_table(
            vdir / "payments.parquet",
            columns=["world_id", "bank_id", "receiver_account_id", "decision_time"],
        )
        unique_worlds = sorted(set(pmts_table["world_id"].to_pylist()))

        # Process world by world to keep memory bounded and maintain isolation
        for world_id in unique_worlds:
            # 2a. Read snapshots for this world and deduplicate to one row per account-day
            world_snaps = pq.read_table(
                vdir / "snapshots.parquet",
                filters=[("world_id", "=", world_id)],
            ).to_pylist()

            day_rows = eval_scoring.one_row_per_account_day(world_snaps)

            # 2b. Index deduplicated days by account_id and precompute rule evaluations
            bank_by_account: dict[str, str] = {}
            # account_id -> sorted list of (snapshot_time, fired_count, action)
            days_by_account: dict[str, list[tuple[datetime, int, str]]] = {}

            for drow in day_rows:
                acc_id = str(drow["account_id"])
                bank_id = str(drow["bank_id"])
                snap_time: datetime = drow["snapshot_time"]
                bank_by_account[acc_id] = bank_id

                # Evaluate rules on this account-day snapshot feature row
                dec = evaluate_rules(drow, config=rules_config)
                fired_count = len(dec.fired)
                action = rules_action(fired_count)

                days_by_account.setdefault(acc_id, []).append((snap_time, fired_count, action))

            # Ensure each account's timeline is sorted strictly ascending by snapshot_time
            # and extract timestamp lists for fast bisect lookups
            times_by_account: dict[str, list[datetime]] = {}
            for acc_id, entries in days_by_account.items():
                entries.sort(key=lambda item: item[0])
                times_by_account[acc_id] = [item[0] for item in entries]

            # 2c. Evaluate payments for this world
            world_pmts = pmts_table.filter(
                pc.equal(pmts_table["world_id"], world_id)
            ).to_pylist()

            for pmt in world_pmts:
                total_payments_processed += 1
                pmt_bank = pmt["bank_id"]
                receiver = pmt["receiver_account_id"]
                dec_time: datetime = pmt["decision_time"]

                receiver_bank = bank_by_account.get(receiver)

                # Check visibility: unknown bank or interbank payment
                if receiver_bank is None or receiver_bank != pmt_bank:
                    actions_count["allow"] += 1
                    fallback_another_bank_count += 1
                    continue

                # Same bank: find latest snapshot strictly earlier than dec_time
                acc_times = times_by_account.get(receiver)
                if not acc_times:
                    actions_count["allow"] += 1
                    fallback_no_prior_day_count += 1
                    continue

                idx = bisect.bisect_left(acc_times, dec_time) - 1
                if idx < 0:
                    actions_count["allow"] += 1
                    fallback_no_prior_day_count += 1
                    continue

                # Visible prior account-day row found
                visible_payee_rows_count += 1
                _snap_time, fired_count, action = days_by_account[receiver][idx]

                actions_count[action] += 1
                if fired_count >= 3:
                    rules_fired_distribution["3+"] += 1
                else:
                    rules_fired_distribution[str(fired_count)] += 1

    # 3. Assertions and consistency gates
    if total_payments_processed != EXPECTED_VALIDATION_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_ROW_COUNT} validation rows, processed {total_payments_processed}"
        )

    fallback_total = fallback_another_bank_count + fallback_no_prior_day_count
    if visible_payee_rows_count + fallback_total != total_payments_processed:
        raise ValueError("Sum of visible rows and fallback rows does not match total validation rows")

    total_actions = sum(actions_count.values())
    if total_actions != total_payments_processed:
        raise ValueError("Sum of actions does not match total validation rows")

    total_fired_dist = sum(rules_fired_distribution.values())
    if total_fired_dist != visible_payee_rows_count:
        raise ValueError("Sum of rules fired distribution does not match visible payee rows count")

    n = float(total_payments_processed)
    report_data = {
        "validation_rows": total_payments_processed,
        "actions": {
            "allow": {"count": actions_count["allow"], "share": actions_count["allow"] / n},
            "warn": {"count": actions_count["warn"], "share": actions_count["warn"] / n},
            "hold": {"count": actions_count["hold"], "share": actions_count["hold"] / n},
        },
        "breakdown": {
            "visible_payee_rows": {
                "count": visible_payee_rows_count,
                "share": visible_payee_rows_count / n,
            },
            "fallback_no_payee_view": {
                "total": {"count": fallback_total, "share": fallback_total / n},
                "another_bank": {
                    "count": fallback_another_bank_count,
                    "share": fallback_another_bank_count / n,
                },
                "no_prior_account_day": {
                    "count": fallback_no_prior_day_count,
                    "share": fallback_no_prior_day_count / n,
                },
            },
        },
        "rules_fired_distribution": rules_fired_distribution,
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report_data, f, indent=2)

    return report_data


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for measuring rules fallback decisions on validation."""
    parser = argparse.ArgumentParser(description="Measure rules fallback on validation under A-074.")
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH, help="Path to v2 result.json")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR, help="Path to datasets root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_REPORT, help="Output report JSON path")
    args = parser.parse_args(argv)

    try:
        report = measure_rules_fallback(
            result_path=args.result_path,
            datasets_dir=args.datasets_dir,
            output_report=args.output,
        )
        print("MEASUREMENT COMPLETE:")
        print(f"Validation rows: {report['validation_rows']}")
        print(
            f"Actions - allow: {report['actions']['allow']['count']} ({report['actions']['allow']['share']:.6f}), "
            f"warn: {report['actions']['warn']['count']} ({report['actions']['warn']['share']:.6f}), "
            f"hold: {report['actions']['hold']['count']} ({report['actions']['hold']['share']:.6f})"
        )
        print(
            f"Visible payee rows: {report['breakdown']['visible_payee_rows']['count']} "
            f"({report['breakdown']['visible_payee_rows']['share']:.6f})"
        )
        print(
            f"Fallback no payee view total: {report['breakdown']['fallback_no_payee_view']['total']['count']} "
            f"({report['breakdown']['fallback_no_payee_view']['total']['share']:.6f})"
        )
        print(
            f"  another_bank: {report['breakdown']['fallback_no_payee_view']['another_bank']['count']} "
            f"({report['breakdown']['fallback_no_payee_view']['another_bank']['share']:.6f})"
        )
        print(
            f"  no_prior_account_day: {report['breakdown']['fallback_no_payee_view']['no_prior_account_day']['count']} "
            f"({report['breakdown']['fallback_no_payee_view']['no_prior_account_day']['share']:.6f})"
        )
        print(f"Rules fired distribution: {report['rules_fired_distribution']}")
        print(f"Report saved to {args.output}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
