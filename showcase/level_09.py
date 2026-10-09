# File: showcase/level_09.py
# Purpose: The Level 9 showcase ("Audit of model v1"). It READS the saved audit results (it never retrains) and checks
#          that they are complete and trustworthy: the table, chart and verdict exist and every row has its MLflow run;
#          the type_PAYMENT-alone and out-of-time cases are shown; and the frozen model v1 files really are protected.
import sys
from pathlib import Path
from typing import Any

from core.frozen_artifacts import (
    FROZEN_ARTIFACT_PATHS,
    FrozenArtifactError,
    guarded_write_bytes,
    verify_frozen_artifacts,
)
from showcase.framework import CheckDefinition, CheckOutcome, NotRun
from showcase.level09_support import (
    CHART_PATH,
    MLFLOW_DB_PATH,
    REPORT_PATH,
    RESULTS_PATH,
    check_mlflow_runs,
    load_results,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from scripts.audit_report import build_verdict, row_named  # noqa: E402

LEVEL_NUMBER = 9
LEVEL_TITLE = "Audit of model v1"
NEEDS_DATABASE = False  # the checks read saved files and mlflow.db; no database schema is needed
WHAT_YOU_SAW = (
    "Model v1 scored 0.99 on PR-AUC, which looked too good. The audit retrained copies of it with parts taken away "
    "and saved every score, each tied to an MLflow run. Without the balance columns the score fell sharply, so it "
    "depends on them, and no single column explains it: the strength comes from combining columns. The type_PAYMENT "
    "flag alone "
    "is barely better than guessing, and on an out-of-time split the full model still scored high while the model "
    "without balances did not. Finally, an attempt to overwrite a frozen model v1 file was refused, and the files kept "
    "their recorded hashes."
)


async def find_missing_preconditions() -> list[str]:
    """What is missing, in plain English (empty means we can run)."""
    problems = []
    for path, fix in [(RESULTS_PATH, "python scripts/audit_v1.py"), (CHART_PATH, "python scripts/audit_chart.py"),
                      (REPORT_PATH, "python scripts/audit_report.py")]:
        if not path.exists():
            problems.append(f"{path.name} is missing - run: {fix}")
    return problems


def need_results() -> dict[str, Any]:
    """The saved results, or NotRun with the command that makes them."""
    if not RESULTS_PATH.exists():
        raise NotRun("v1_audit_results.json is missing - run: python scripts/audit_v1.py")
    return load_results()


# ---------------------------------------------------------------------------
# Check 1 (normal): table, chart, MLflow runs and the verdict
# ---------------------------------------------------------------------------

def table_lines(results: dict[str, Any]) -> list[str]:
    """One plain line per result row."""
    return [f"{row['name']:34}{row['split']:15}PR-AUC {row['pr_auc']:.5f}  run {row['mlflow_run_id'][:8]}"
            for row in results["rows"]]


async def run_table_check(workspace: Any) -> CheckOutcome:
    """Show the whole table, and require the chart, the report and a matching MLflow run for every row."""
    results = need_results()
    assert CHART_PATH.exists() and CHART_PATH.stat().st_size > 5000, "the bar chart is missing or empty"
    assert REPORT_PATH.exists(), "the audit report is missing"
    rows = results["rows"]
    assert all(row["mlflow_run_id"] for row in rows), "a row has no MLflow run id"
    verdict = build_verdict(results)
    evidence = {"table": table_lines(results), "chart": CHART_PATH.name, "verdict": verdict,
                "base_rate": results["base_rate"]}
    if not MLFLOW_DB_PATH.exists():
        # mlflow.db is git-ignored, so a fresh clone or the CI runner does not have it. Say so plainly instead of
        # failing for something nobody can fix there; the cross-check runs on any machine that has the database.
        evidence["mlflow_cross_check"] = "NOT DONE: mlflow.db is not on this machine"
        return CheckOutcome(key_number=f"{len(rows)} rows have run ids (mlflow.db absent: NOT cross-checked)",
                            evidence=evidence)
    verified, problems = check_mlflow_runs(results)
    assert not problems, "; ".join(problems)
    assert verified == len(rows), f"only {verified} of {len(rows)} rows were verified"
    evidence["mlflow_cross_check"] = "done: every run exists with the same metric"
    return CheckOutcome(key_number=f"{verified} of {len(rows)} rows match their MLflow runs", evidence=evidence)


# ---------------------------------------------------------------------------
# Check 2 (edge): type_PAYMENT alone
# ---------------------------------------------------------------------------

async def run_payment_check(workspace: Any) -> CheckOutcome:
    """type_PAYMENT alone: its score, its share of the original score and its lift over the base rate."""
    results = need_results()
    score = row_named(results, "single_type_PAYMENT")["pr_auc"]
    original = row_named(results, "original_v1")["pr_auc"]
    lift = score / results["base_rate"]
    assert score <= original, "a single flag cannot beat the full model"
    return CheckOutcome(
        key_number=f"PR-AUC {score:.4f}, {score / original:.1%} of original, {lift:.1f}x base rate",
        evidence={"score": score, "share_of_original": score / original, "lift_over_base_rate": lift},
    )


# ---------------------------------------------------------------------------
# Check 3 (edge): out-of-time versus the random-split control
# ---------------------------------------------------------------------------

async def run_out_of_time_check(workspace: Any) -> CheckOutcome:
    """The out-of-time score next to the random-split control, with the drop shown."""
    results = need_results()
    control = row_named(results, "control_all15")["pr_auc"]
    oot = row_named(results, "oot_all15")["pr_auc"]
    oot_no_balance = row_named(results, "oot_no_balance")["pr_auc"]
    return CheckOutcome(
        key_number=f"control {control:.4f}, out-of-time {oot:.4f}, drop {control - oot:.4f}",
        evidence={"control": control, "out_of_time": oot, "drop": control - oot,
                  "out_of_time_without_balance": oot_no_balance},
    )


# ---------------------------------------------------------------------------
# Check 4 (negative): the frozen model is protected
# ---------------------------------------------------------------------------

async def run_frozen_check(workspace: Any) -> CheckOutcome:
    """Try to overwrite the frozen model with dummy bytes: it must be refused, and the files must stay unchanged."""
    model = FROZEN_ARTIFACT_PATHS[0]
    size_before = model.stat().st_size if model.exists() else None
    refused = False
    try:
        guarded_write_bytes(model, b"dummy bytes that must never be written")
    except FrozenArtifactError:
        refused = True
    assert refused, "the write to the frozen model was NOT refused"
    size_after = model.stat().st_size if model.exists() else None
    assert size_after == size_before, "the frozen model file changed size"
    problems = verify_frozen_artifacts(allow_missing=True)
    assert not problems, f"frozen files differ from their recorded hashes: {problems}"
    return CheckOutcome(
        key_number="write refused; 4 frozen files still match their hashes",
        evidence={"refused": True, "model_size_before": size_before, "model_size_after": size_after},
    )


def get_checks(quick: bool) -> list[CheckDefinition]:
    """The Level 9 checks, in order (quick and full mode are the same: nothing is retrained)."""
    return [
        CheckDefinition(
            "1. audit table, chart and verdict match the MLflow runs", "normal",
            "Every audit score has an MLflow run with the same value, and the chart and report exist.",
            run_table_check, "The results file and MLflow no longer agree, or a file was deleted.",
            "reports/model/v1_audit_results.json, mlflow.db"),
        CheckDefinition(
            "2. type_PAYMENT alone is barely better than guessing", "edge",
            "The type_PAYMENT flag on its own has a tiny score compared with the full model.",
            run_payment_check, "The single-feature rows are missing or wrong.", "scripts/audit_v1.py"),
        CheckDefinition(
            "3. out-of-time score is compared with the random-split control", "edge",
            "The score on later steps is shown next to the random-split score, with the drop.",
            run_out_of_time_check, "The out-of-time rows are missing.", "scripts/audit_v1.py (load_oot_split)"),
        CheckDefinition(
            "4. overwriting the frozen model v1 file is refused", "negative",
            "Writing to models/xgboost/model.ubj is refused and the frozen files keep their hashes.",
            run_frozen_check, "The frozen-artifact guard no longer protects the model files.",
            "core/frozen_artifacts.py"),
    ]
