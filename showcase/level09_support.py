# File: showcase/level09_support.py
# Purpose: Helpers for the Level 9 showcase and its integration test: read the saved audit results, and check that each
#          result row's MLflow run exists in mlflow.db with the same metric value. The database is only ever opened
#          READ-ONLY, so nothing in it can change.
import sqlite3
from pathlib import Path
from typing import Any

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "reports" / "model" / "v1_audit_results.json"
CHART_PATH = REPO_ROOT / "reports" / "model" / "v1_audit_chart.png"
REPORT_PATH = REPO_ROOT / "reports" / "model" / "v1_audit.md"
MLFLOW_DB_PATH = REPO_ROOT / "mlflow.db"
METRIC_TOLERANCE = 1e-9  # a stored metric and a saved score must agree to this many decimals

# The metric each kind of row is stored under in MLflow: audit runs use test_pr_auc; the original Level 5 run
# stored the same score as overall_test_pr_auc.
AUDIT_METRIC = "test_pr_auc"
ORIGINAL_METRIC = "overall_test_pr_auc"


def load_results() -> dict[str, Any]:
    """Read reports/model/v1_audit_results.json."""
    return orjson.loads(RESULTS_PATH.read_bytes())


def open_mlflow_read_only() -> sqlite3.Connection:
    """Open mlflow.db read-only (mode=ro), so even a bug here cannot change it."""
    return sqlite3.connect(f"{MLFLOW_DB_PATH.resolve().as_uri()}?mode=ro", uri=True)


def stored_metric(connection: sqlite3.Connection, run_id: str, metric_name: str) -> float | None:
    """The latest value of a metric of one run in MLflow, or None if the run or metric does not exist."""
    row = connection.execute(
        "SELECT value FROM latest_metrics WHERE run_uuid = ? AND key = ?", (run_id, metric_name)
    ).fetchone()
    return None if row is None else float(row[0])


def metric_name_for(row: dict[str, Any]) -> str:
    """The MLflow metric name a result row should be stored under."""
    return ORIGINAL_METRIC if row["group"] == "original" else AUDIT_METRIC


def check_mlflow_runs(results: dict[str, Any]) -> tuple[int, list[str]]:
    """
    For every result row: does its MLflow run exist, and does the metric equal the saved score?

    Returns (number of rows verified, list of problems in plain English).
    """
    verified, problems = 0, []
    connection = open_mlflow_read_only()
    try:
        for row in results["rows"]:
            value = stored_metric(connection, row["mlflow_run_id"], metric_name_for(row))
            if value is None:
                problems.append(f"{row['name']}: run {row['mlflow_run_id'][:8]} or its metric is not in mlflow.db")
            elif abs(value - row["pr_auc"]) > METRIC_TOLERANCE:
                problems.append(f"{row['name']}: MLflow has {value:.9f}, the results file has {row['pr_auc']:.9f}")
            else:
                verified += 1
    finally:
        connection.close()
    return verified, problems
