"""
File: tests/integration/test_audit_v1_runs.py
Purpose: Proves every row of reports/model/v1_audit_results.json points to an MLflow run that exists in mlflow.db
         with the same metric value (the database is opened read-only). Skipped, with the reason, when the database or
         the results file is not on this machine (a fresh clone does not have mlflow.db).
"""

import pytest

from showcase.level09_support import MLFLOW_DB_PATH, RESULTS_PATH, check_mlflow_runs, load_results


def test_every_result_row_has_an_mlflow_run_with_the_same_metric() -> None:
    """All rows are verified against mlflow.db, and none is reported as a problem."""
    if not MLFLOW_DB_PATH.exists():
        pytest.skip("mlflow.db is not on this machine (it is git-ignored; restore it with scripts/restore_mlflow.py)")
    if not RESULTS_PATH.exists():
        pytest.skip("reports/model/v1_audit_results.json is missing (run scripts/audit_v1.py)")
    results = load_results()
    verified, problems = check_mlflow_runs(results)
    assert not problems, "; ".join(problems)
    assert verified == len(results["rows"])
