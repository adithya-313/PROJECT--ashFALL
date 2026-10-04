# File: scripts/fraud_model_card.py
# Purpose: Sub-phase 5.9 - builds reports/model/model_card.md for the winning XGBoost from the reports and saved
#          artifacts that already exist (evaluation.md, calibration_summary.md, best_params.json, the saved model),
#          regenerates the fairness table with the evaluation script's own functions, and logs both to MLflow.
#
# No figure in the card is typed by hand: tables are copied from the reports, and numbers quoted in the prose are
# read from those reports (or measured from the saved model) when the card is built.
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import mlflow
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from fraud_baseline import read_dvc_data_hash  # noqa: E402

from core.mlflow_setup import setup_mlflow  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "model"
EVALUATION_PATH = REPORT_DIR / "evaluation.md"
BEST_PARAMS_PATH = REPORT_DIR / "best_params.json"
FAIRNESS_PATH = REPORT_DIR / "fairness_table.md"
CARD_PATH = REPORT_DIR / "model_card.md"
CALIBRATION_SUMMARY_PATH = REPO_ROOT / "reports" / "calibration" / "calibration_summary.md"
REPRODUCIBILITY_PATH = REPO_ROOT / "REPRODUCIBILITY.md"
ADVERSARIAL_CONFIG_PATH = REPO_ROOT / "data" / "adversarial" / "adversarial_config.yaml"

EXPERIMENT_NAME = "level5_fraud_evaluation"
EVALUATION_RUN_NAME = "evaluation_xgboost"
CARD_RUN_NAME = "model_card_xgboost"
TOP_FEATURES_IN_CARD = 5


def load_evaluation_module():
    """Load scripts/fraud_evaluation.py by file path, to reuse its fairness functions (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("fraud_evaluation", SCRIPTS_DIR / "fraud_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Reading the existing reports -------------------------------------------------------------------------
def table_after_heading(text: str, heading_start: str) -> list[str]:
    """The markdown table lines that follow the first heading starting with `heading_start`."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading_start))
    table = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break
    return table


def parse_table(table_lines: list[str]) -> list[dict]:
    """Turn markdown table lines into a list of {column name: cell text} dictionaries."""
    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    headers = cells(table_lines[0])
    return [dict(zip(headers, cells(line))) for line in table_lines[2:]]


def row_where(rows: list[dict], column: str, starts_with: str) -> dict:
    """The first table row whose `column` text starts with `starts_with`."""
    return next(row for row in rows if row[column].startswith(starts_with))


def read_reports() -> dict:
    """Read every table and figure the card reuses from the existing reports."""
    evaluation = EVALUATION_PATH.read_text()
    calibration = CALIBRATION_SUMMARY_PATH.read_text()
    models_table = table_after_heading(evaluation, "## Test-set results")
    segments_table = table_after_heading(evaluation, "## Per-segment metrics")
    confusion_table = table_after_heading(evaluation, "## Confusion matrix")
    return {
        "models_table": models_table, "models": parse_table(models_table),
        "segments_table": segments_table, "segments": parse_table(segments_table),
        "confusion_table": confusion_table,
        "calibration_test_bins": parse_table(table_after_heading(calibration, "## Reliability on the test split")),
        "cut_line": next(line for line in evaluation.splitlines() if line.startswith("Amount tiers use")),
        "best_params": json.loads(BEST_PARAMS_PATH.read_text())["xgboost"],
    }


# --- Reproducibility facts --------------------------------------------------------------------------------
def git_output(*arguments: str) -> str:
    """Run a read-only git command in the repo and return its text output."""
    return subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def find_evaluation_run_id() -> str:
    """The MLflow run ID of the 5.4 evaluation run."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = [r for r in client.search_runs([experiment.experiment_id], max_results=100)
            if r.data.tags.get("mlflow.runName") == EVALUATION_RUN_NAME]
    assert len(runs) == 1, f"Expected exactly one {EVALUATION_RUN_NAME} run, found {len(runs)}"
    return runs[0].info.run_id


def read_data_story() -> dict:
    """Facts about the data from the reports and config (original fraud count, injected counts)."""
    reproducibility = REPRODUCIBILITY_PATH.read_text()
    match = re.search(r"([\d,]+) rows = ([\d,]+) original PaySim \+ ([\d,]+) injected", reproducibility)
    config = yaml.safe_load(ADVERSARIAL_CONFIG_PATH.read_text())["adversarial_injection"]
    return {
        "total_fraud": match.group(1) if match else None, "original_fraud": match.group(2) if match else None,
        "injected_fraud": match.group(3) if match else None, "injected_counts": config["injected_fraud_counts"],
    }


def runner_up_gap(models: list[dict]) -> float:
    """The largest PR-AUC gap between the winner (first row, the table is sorted) and any runner-up."""
    winner_score = float(models[0]["PR-AUC"])
    return max(winner_score - float(row["PR-AUC"]) for row in models[1:])


def top_features(model, feature_names: list[str]) -> list[tuple[str, float]]:
    """The saved model's most important features (XGBoost's own importance, normalised to sum to 1)."""
    pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda pair: pair[1], reverse=True)
    return [(name, float(value)) for name, value in pairs[:TOP_FEATURES_IN_CARD]]


# --- Writing the card -------------------------------------------------------------------------------------
def indent_headings(markdown: str, drop_title: bool = True) -> list[str]:
    """Nest a standalone report under a card section: remove its top title and push other headings down a level."""
    lines = markdown.splitlines()
    if drop_title and lines and lines[0].startswith("# "):
        lines = lines[1:]
    return [("#" + line) if line.startswith("## ") else line for line in lines]


def calibration_sparsity_sentence(bins: list[dict]) -> str:
    """Sentences about how thin the middle calibration bins are, built from the test-split reliability table."""
    def rows_in(bin_row: dict) -> int:
        return int(bin_row["Rows"].replace(",", ""))

    def deviation_of(bin_row: dict) -> float:
        return 0.0 if bin_row["Deviation"] == "-" else float(bin_row["Deviation"])

    edge_bins = ("0.0-0.1", "0.9-1.0")
    middle = [b for b in bins if b["Bin"] not in edge_bins]
    lowest = next(b for b in bins if b["Bin"] == "0.0-0.1")
    smallest = min(middle, key=rows_in)
    worst = max(bins, key=deviation_of)
    fewest, most = min(map(rows_in, middle)), max(map(rows_in, middle))
    text = (f"On the test split, the eight middle probability bins (0.1 to 0.9) hold only {fewest:,} "
            f"to {most:,} rows each, against {rows_in(lowest):,} in the lowest bin. The thinnest "
            f"middle bin is {smallest['Bin']} ({rows_in(smallest)} rows). ")
    if worst["Bin"] == smallest["Bin"]:
        text += f"It is also the worst-calibrated bin on test (deviation {worst['Deviation']}). "
    else:
        text += f"The worst-calibrated bin on test is {worst['Bin']} (deviation {worst['Deviation']}). "
    return text + "Calibration in the middle of the range is therefore not well established."


def build_card(facts: dict) -> str:
    """Assemble the full model card text."""
    reports = facts["reports"]
    data = facts["data"]
    native = row_where(reports["segments"], "Segment", "PaySim-native")
    takeover = row_where(reports["segments"], "Segment", "Pattern: account_takeover")
    mule = row_where(reports["segments"], "Segment", "Pattern: mule_account")
    overall = row_where(reports["segments"], "Segment", "Overall")
    runner_ups = [m for m in reports["models"] if m["Model"] != "xgboost"]
    params = "; ".join(f"`{key}` = {value}" for key, value in reports["best_params"].items())
    importance = ", ".join(f"`{name}` ({share:.0%})" for name, share in facts["top_features"])
    evaluation_run = facts["evaluation_run_id"]
    calibration_text = calibration_sparsity_sentence(reports["calibration_test_bins"])

    lines = [
        "# Model card: fraud scoring model (XGBoost)",
        "",
        "## 1. What the model does and what it was trained on",
        "",
        "The model scores a single payment transaction with a probability that it is fraud (`isFraud`). It is an "
        "XGBoost "
        "classifier that takes **15 numeric features** in a fixed order: `amount`, the four account balances "
        "(`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`), five engineered signals "
        "(`account_avg_amount`, `many_small_inbound_count`, `large_outbound_amount`, `new_device_flag`, "
        "`geo_velocity_spike`) and a one-hot encoding of the transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, "
        "TRANSFER). Identifiers, labels and `step` are never inputs.",
        "",
        f"**Data.** The corrected PaySim dataset (DVC hash `{facts['dvc_hash']}`), split 60/20/20 into train / "
        f"calibration / test with a fixed seed (42), stratified on the fraud label, so each split has about 0.5% "
        f"fraud. "
        f"The model trained on the train split ({facts['train_rows']:,} rows). Fraud in the dataset is "
        f"{data['total_fraud'] or 'about 0.5% of rows'}: {data['original_fraud'] or 'the original'} PaySim fraud rows "
        f"plus {data['injected_fraud'] or 'the injected'} synthetic rows "
        f"({data['injected_counts']['account_takeover']:,} account-takeover and "
        f"{data['injected_counts']['mule_account']:,} mule-account), made by `scripts/inject_adversarial.py`.",
        "",
        f"**Model.** XGBoost, chosen as the winner because it had the highest test PR-AUC of three tuned models. "
        f"Hyperparameters (tuned with Optuna on a 25% train sample): {params}. It was trained with a heavy "
        "fraud weighting (`scale_pos_weight`, about 199), so its raw output is **not a probability**. A separate "
        "isotonic calibrator turns it into one, and the serving client always applies it (`serving/fraud_client.py`). "
        f"Native feature importance is led by {importance}.",
        "",
        "## 2. Performance (test split)",
        "",
        f"Overall: PR-AUC {overall['PR-AUC']}, ROC-AUC {overall['ROC-AUC']} on {overall['Rows']} rows with "
        f"{overall['Fraud']} fraud. The per-segment table below is copied from `reports/model/evaluation.md`.",
        "",
        *reports["segments_table"],
        "",
        reports["cut_line"],
        "",
        "Confusion matrix at the provisional raw threshold of 0.5 (not a recommended operating point):",
        "",
        *reports["confusion_table"],
        "",
        "## 3. Fairness: false-positive rate by segment",
        "",
        "Produced by `scripts/fraud_evaluation.py --fairness` (same script as the performance section) and copied from "
        "`reports/model/fairness_table.md`. The real axes are transaction type and amount tier. Read the merchant-tier "
        "note below the tables before using those rows.",
        "",
        *facts["fairness_lines"],
        "",
        "## 4. Comparison with the runner-up models",
        "",
        "Overall test results for all three tuned models (copied from `evaluation.md`):",
        "",
        *reports["models_table"],
        "",
        f"The runners-up are {', '.join(r['Model'] for r in runner_ups)}, within {facts['runner_up_gap']:.4f} PR-AUC "
        "of the winner at most. Per-segment metrics and the fairness table were computed for the winner only. "
        "Producing them for the runners-up would mean retraining those models (the random forest alone takes several "
        "minutes), so that "
        "comparison was not done.",
        "",
        "## 5. Known limitations",
        "",
        "- **The data is synthetic.** PaySim is a simulator, not real transactions. Scores on it say how well the "
        "model learned this simulator, not how it would do on real fraud.",
        f"- **The signal is strongly balance- and type-driven.** The leading features are {importance}. That reflects "
        "how "
        "the simulator makes fraud (accounts being emptied) and is optimistic compared with real-world fraud, which is "
        "rarely this clean. No no-balance version was measured on the final dataset, so this is stated "
        "qualitatively, without a number.",
        "- **The adversarial patterns are injected.** The account-takeover and mule-account rows are synthetic, built "
        "from documented real-world behaviour (`reports/data/fraud_pattern_research.md`) but still artificial. They "
        "cannot carry network, session or time-window signals that the tabular format does not hold.",
        "- **Not validated on real fraud data.** No real-world evaluation has been done.",
        f"- **Native versus injected fraud.** PaySim's own fraud scores PR-AUC {native['PR-AUC']} "
        f"({native['Fraud']} test fraud rows), against {takeover['PR-AUC']} (account takeover) and {mule['PR-AUC']} "
        f"(mule account) for the injected patterns. The overall {overall['PR-AUC']} is dominated by the injected "
        "rows, which make up about three quarters of the test fraud, so the native figure is the more "
        "conservative one.",
        "- **Calibration is thin in the middle of the range.** " + calibration_text,
        "- **Merchant risk tier is not a valid fairness axis.** It is confounded with transaction type (see the "
        "note in section 3), so the model has not been shown to be fair across merchant risk levels.",
        "- **No operating point is chosen.** The thresholds shown are provisional. The final threshold is a business "
        "decision about missed fraud versus false alarms.",
        "- **Not connected to live requests.** The model needs an already-prepared row of 15 features. Building those "
        "features from a raw payment request (account history, balances) is not done yet, and the API's `/score` "
        "endpoint is still a stub that does not call the model.",
        "",
        "## 6. Reproducibility",
        "",
        f"- Code: git commit `{facts['git_sha']}` (this card is added in the commit that follows it).",
        f"- Data: DVC content hash `{facts['dvc_hash']}`.",
        f"- MLflow: evaluation run `{evaluation_run}` in experiment `{EXPERIMENT_NAME}` (store `mlflow.db`). This card "
        f"and the fairness table are also logged there as artifacts of the run named `{CARD_RUN_NAME}`.",
        "- Seed: 42. Reload the model from `models/xgboost/model.ubj` (not committed; DVC-tracked later), calibrator "
        "from `models/calibration/isotonic_calibrator.pkl`, ONNX export in `models/xgboost/model.onnx`.",
        "",
    ]
    return "\n".join(lines)


# --- MLflow ----------------------------------------------------------------------------------------------
def metric_name(group: str, segment: str, threshold_key: str) -> str:
    """A safe MLflow metric name for one fairness cell."""
    raw = f"fpr_{group}_{segment}_{threshold_key}".lower()
    return "_".join(part for part in re.sub(r"[^a-z0-9.]+", "_", raw).split("_") if part)[:120]


def log_to_mlflow(facts: dict, fairness_rows: list[dict], thresholds: list[dict]) -> str:
    """Log the card and fairness table as artifacts of a NEW run (the finished 5.4 run is left untouched)."""
    with mlflow.start_run(run_name=CARD_RUN_NAME) as run:
        mlflow.log_params({
            "dvc_data_hash": facts["dvc_hash"], "code_commit_base": facts["git_sha"], "seed": 42,
            "evaluation_run_id": facts["evaluation_run_id"], "model_path": "models/xgboost/model.ubj",
            "calibrator_path": "models/calibration/isotonic_calibrator.pkl",
            "fairness_thresholds": "; ".join(t["label"] for t in thresholds),
        })
        for row in fairness_rows:
            for threshold in thresholds:
                cell = row["thresholds"][threshold["key"]]
                if cell["fpr"] is not None:
                    mlflow.log_metric(metric_name(row["group"], row["segment"], threshold["key"]), cell["fpr"])
        mlflow.log_artifact(str(CARD_PATH))
        mlflow.log_artifact(str(FAIRNESS_PATH))
        return run.info.run_id


def main() -> int:
    """Regenerate the fairness table with the evaluation script, build the card, and log both."""
    evaluation = load_evaluation_module()
    setup_mlflow(EXPERIMENT_NAME)

    # The fairness table comes from the evaluation script's own functions (reload the saved winner; no retraining).
    X_test, y_test, segments = evaluation.load_test_with_segments()
    model = evaluation.load_saved_winner()
    scores = evaluation.score_raw_and_calibrated(model, X_test)
    rows, cuts, confound = evaluation.compute_fairness_rows(y_test, scores, segments)
    evaluation.write_fairness_table(rows, cuts, confound)

    # Check that the fairness numbers agree with the confusion matrix already published in evaluation.md.
    reports = read_reports()
    matrix = parse_table(reports["confusion_table"])
    published_fp = int(row_where(matrix, "", "Actual non-fraud")["Predicted fraud"].replace(",", ""))
    first_key = evaluation.FAIRNESS_THRESHOLDS[0]["key"]
    overall = next(r for r in rows if r["group"] == "All rows")
    assert overall["thresholds"][first_key]["fp"] == published_fp, "Fairness table disagrees with evaluation.md"

    evaluation_run_id = find_evaluation_run_id()
    facts = {
        "reports": reports, "data": read_data_story(), "dvc_hash": read_dvc_data_hash(),
        "git_sha": git_output("rev-parse", "HEAD"), "evaluation_run_id": evaluation_run_id,
        "runner_up_gap": runner_up_gap(reports["models"]),
        "top_features": top_features(model, list(X_test.columns)),
        "fairness_lines": indent_headings(FAIRNESS_PATH.read_text()),
        "train_rows": int(mlflow.MlflowClient().get_run(evaluation_run_id).data.params["train_rows"]),
    }
    CARD_PATH.write_text(build_card(facts))
    print(f"Wrote {CARD_PATH.relative_to(REPO_ROOT).as_posix()} and {FAIRNESS_PATH.relative_to(REPO_ROOT).as_posix()}")

    run_id = log_to_mlflow(facts, rows, evaluation.FAIRNESS_THRESHOLDS)
    print(f"MLflow experiment '{EXPERIMENT_NAME}', run '{CARD_RUN_NAME}', run ID {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
