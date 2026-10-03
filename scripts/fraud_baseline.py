# File: scripts/fraud_baseline.py
# Purpose: Trains five baseline fraud models on isFraud (default-ish settings, no tuning), compares them on the
#          test split, and logs every model as its own MLflow run.
import sys
import time
from pathlib import Path

import catboost
import lightgbm
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import xgboost
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Draw plots straight to files; no window is needed.
matplotlib.use("Agg")

# Allow running as "python scripts/fraud_baseline.py" from anywhere: put the repo root on the import path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.mlflow_setup import setup_mlflow  # noqa: E402

TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test.parquet"
DVC_POINTER_PATH = REPO_ROOT / "data" / "enriched" / "paysim_enriched.parquet.dvc"
REPORT_DIR = REPO_ROOT / "reports" / "model"

EXPERIMENT_NAME = "level5_fraud_baseline"
RANDOM_SEED = 42

# --- Feature contract -------------------------------------------------------------------------------------
# Features are picked BY NAME, never "all columns", so a stray column (for example a persisted index column in
# the parquet file) can never leak into a model.
#
# Deliberately NOT used as model input:
#   step, merchant_risk_tier  - out of scope for model input per the plan.
#   isFlaggedFraud, is_adversarial, adversarial_pattern_type - labels / metadata that would leak the answer.
#   nameOrig, nameDest, geo_coordinates, device_fingerprint  - identifiers (near-unique per row).
TARGET_COLUMN = "isFraud"
AMOUNT_AND_BALANCE_COLUMNS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
ENGINEERED_BOOL_COLUMNS = ["new_device_flag", "geo_velocity_spike"]
ENGINEERED_NUMBER_COLUMNS = ["account_avg_amount", "many_small_inbound_count", "large_outbound_amount"]

# Fixed, hardcoded order so the one-hot columns are identical for any data, now or in the future.
TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# A model at or below this ROC-AUC is suspiciously close to a coin flip (0.5), so the sanity check fails.
MIN_SANE_ROC_AUC = 0.55


def read_dvc_data_hash() -> str:
    """Read the md5 of the enriched dataset from its DVC pointer file, so runs record exactly which data was used."""
    with open(DVC_POINTER_PATH) as pointer_file:
        pointer = yaml.safe_load(pointer_file)
    return pointer["outs"][0]["md5"]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model input table (15 named columns) and the target from a raw split."""
    # Take only the named numeric columns. Booleans are turned into 0/1 because models need numbers.
    features = df[AMOUNT_AND_BALANCE_COLUMNS + ENGINEERED_NUMBER_COLUMNS].copy()
    for column in ENGINEERED_BOOL_COLUMNS:
        features[column] = df[column].astype(int)

    # One-hot encode `type` by hand against the fixed category list: one 0/1 column per category, always in the
    # same order. A row with an unexpected type would simply get zeros in every type column.
    for category in TYPE_CATEGORIES:
        features["type_" + category] = (df["type"] == category).astype(int)

    return features, df[TARGET_COLUMN].astype(int)


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load one split from disk and return its features and target."""
    df = pd.read_parquet(path)
    return prepare_features(df)


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Return negatives / positives from the real training labels (about 199 for a 0.5% fraud rate)."""
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    return negatives / positives


def build_xgboost(scale_pos_weight: float, use_gpu: bool) -> xgboost.XGBClassifier:
    """XGBoost on the GPU (device='cuda') or CPU; scale_pos_weight makes each fraud row count extra."""
    return xgboost.XGBClassifier(
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        random_state=RANDOM_SEED,
        eval_metric="aucpr",
    )


def build_catboost(use_gpu: bool) -> catboost.CatBoostClassifier:
    """CatBoost on the GPU or CPU; auto_class_weights='Balanced' handles the rarity of fraud."""
    settings = {
        "iterations": 300,
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_SEED,
        "verbose": 0,
        "allow_writing_files": False,  # do not drop a catboost_info/ folder into the working directory
    }
    if use_gpu:
        return catboost.CatBoostClassifier(task_type="GPU", devices="0", **settings)
    return catboost.CatBoostClassifier(task_type="CPU", **settings)


def get_model_specs(scale_pos_weight: float) -> list[dict]:
    """Describe the five models: a name, a builder, the settings to log, and whether it first tries the GPU."""
    return [
        {
            "name": "logreg",
            "build": lambda gpu: LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {"max_iter": 1000, "class_weight": "balanced", "feature_scaling": "StandardScaler"},
            "gpu": False,
        },
        {
            "name": "random_forest",
            "build": lambda gpu: RandomForestClassifier(
                n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
            ),
            "params": {"n_estimators": 100, "class_weight": "balanced", "device": "cpu"},
            "gpu": False,
        },
        {
            "name": "xgboost",
            "build": lambda gpu: build_xgboost(scale_pos_weight, gpu),
            "params": {"n_estimators": 300, "tree_method": "hist", "scale_pos_weight": round(scale_pos_weight, 4)},
            "gpu": True,
        },
        {
            "name": "lightgbm",
            "build": lambda gpu: lightgbm.LGBMClassifier(
                n_estimators=300, scale_pos_weight=scale_pos_weight, device_type="cpu",
                random_state=RANDOM_SEED, verbose=-1,
            ),
            "params": {"n_estimators": 300, "scale_pos_weight": round(scale_pos_weight, 4), "device": "cpu"},
            "gpu": False,
        },
        {
            "name": "catboost",
            "build": lambda gpu: build_catboost(gpu),
            "params": {"iterations": 300, "auto_class_weights": "Balanced"},
            "gpu": True,
        },
    ]


def fit_model(spec: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit one model (scaling first for logistic regression); fall back to CPU if a GPU model errors."""
    name = spec["name"]
    use_gpu = spec["gpu"]
    fallback_note = ""

    # Logistic regression needs scaled inputs (balances reach tens of millions). Tree models split on thresholds,
    # so scaling does not matter to them and they train on the raw values.
    scaler = None
    X_fit = X_train
    if name == "logreg":
        scaler = StandardScaler().fit(X_train)
        X_fit = scaler.transform(X_train)

    start = time.perf_counter()
    model = spec["build"](use_gpu)
    try:
        model.fit(X_fit, y_train)
    except Exception as error:  # noqa: BLE001 - we want to report any GPU failure exactly, then retry on CPU
        if not use_gpu:
            raise
        # Do not hide a GPU failure: print the exact error, say so loudly, and retry this one model on the CPU.
        fallback_note = f"{type(error).__name__}: {error}"
        print(f"  GPU ERROR for {name}: {fallback_note}")
        print(f"  FALLING BACK TO CPU for {name}")
        start = time.perf_counter()
        model = spec["build"](False)
        model.fit(X_fit, y_train)
    fit_seconds = time.perf_counter() - start

    return {"model": model, "scaler": scaler, "fit_seconds": fit_seconds, "fallback_note": fallback_note}


def evaluate_model(fitted: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score a fitted model on the test split: ROC-AUC, PR-AUC and the points of both curves."""
    X_in = fitted["scaler"].transform(X_test) if fitted["scaler"] is not None else X_test
    fraud_probability = fitted["model"].predict_proba(X_in)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, fraud_probability)
    precision, recall, _ = precision_recall_curve(y_test, fraud_probability)
    return {
        "roc_auc": roc_auc_score(y_test, fraud_probability),
        # average_precision_score is the standard single-number summary of the precision-recall curve (PR-AUC).
        "pr_auc": average_precision_score(y_test, fraud_probability),
        "roc_points": (fpr, tpr),
        "pr_points": (recall, precision),
    }


def log_run_metrics(spec: dict, fitted: dict, scores: dict, shared_params: dict) -> None:
    """Write one model's parameters and metrics to the currently open MLflow run."""
    mlflow.log_params(shared_params)
    mlflow.log_param("model", spec["name"])
    mlflow.log_params(spec["params"])
    mlflow.log_param("gpu_fallback_to_cpu", "yes" if fitted["fallback_note"] else "no")
    if fitted["fallback_note"]:
        mlflow.log_param("gpu_fallback_error", fitted["fallback_note"][:250])

    mlflow.log_metric("test_roc_auc", scores["roc_auc"])
    mlflow.log_metric("test_pr_auc", scores["pr_auc"])
    mlflow.log_metric("train_seconds", fitted["fit_seconds"])


def make_plots(results: list[dict]) -> list[Path]:
    """Save the ROC overlay, PR overlay and training-time bar chart; return the three file paths."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    roc_path = REPORT_DIR / "fraud_roc_curves.png"
    pr_path = REPORT_DIR / "fraud_pr_curves.png"
    time_path = REPORT_DIR / "fraud_train_time.png"

    # ROC curves: closer to the top-left corner is better; the dashed diagonal is a random guess.
    plt.figure(figsize=(7, 6))
    for result in results:
        fpr, tpr = result["roc_points"]
        plt.plot(fpr, tpr, label=f"{result['name']} (AUC={result['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Fraud baselines: ROC curves (test split)")
    plt.legend()
    plt.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close()

    # PR curves matter more than ROC here because fraud is rare: they ignore the huge pile of true negatives.
    plt.figure(figsize=(7, 6))
    for result in results:
        recall, precision = result["pr_points"]
        plt.plot(recall, precision, label=f"{result['name']} (PR-AUC={result['pr_auc']:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fraud baselines: precision-recall curves (test split)")
    plt.legend()
    plt.savefig(pr_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Training time per model, to show the cost side of the comparison.
    plt.figure(figsize=(7, 5))
    plt.bar([r["name"] for r in results], [r["fit_seconds"] for r in results])
    plt.xlabel("Model")
    plt.ylabel("Fit time (seconds)")
    plt.title("Fraud baselines: training time")
    plt.savefig(time_path, dpi=120, bbox_inches="tight")
    plt.close()

    return [roc_path, pr_path, time_path]


def write_report(results: list[dict], data_hash: str) -> Path:
    """Write the results table, sorted by PR-AUC (best first), to reports/model/fraud_baseline_results.md."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "fraud_baseline_results.md"

    lines = ["| Model | PR-AUC | ROC-AUC | Train seconds |", "|---|---|---|---|"]
    for result in sorted(results, key=lambda r: r["pr_auc"], reverse=True):
        lines.append(
            f"| {result['name']} | {result['pr_auc']:.4f} | {result['roc_auc']:.4f} | {result['fit_seconds']:.1f} |"
        )
    note = (
        "These results supersede the pre-correction baseline results (they were produced on the corrected "
        f"dataset, DVC data hash `{data_hash}`). Baseline settings, no tuning; winner selection comes later."
    )
    report_path.write_text("\n".join(lines) + "\n\n" + note + "\n")
    return report_path


def log_artifacts_to_runs(run_ids: list[str], files: list[Path]) -> None:
    """Attach the plots and the results table to EACH model's run, so every run is self-contained."""
    for run_id in run_ids:
        with mlflow.start_run(run_id=run_id):
            for file_path in files:
                mlflow.log_artifact(str(file_path))


def check_results(results: list[dict], report_path: Path) -> bool:
    """Acceptance checks: every model trained, every ROC-AUC clears the sanity floor, and the results file exists."""
    all_passed = True
    print("\nAcceptance checks:")
    print(f"  [{'PASS' if len(results) == 5 else 'FAIL'}] all 5 models trained and scored ({len(results)} of 5)")
    all_passed = all_passed and len(results) == 5

    for result in results:
        passed = result["roc_auc"] > MIN_SANE_ROC_AUC
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {result['name']} ROC-AUC {result['roc_auc']:.4f} > {MIN_SANE_ROC_AUC}")

    written = report_path.exists() and report_path.stat().st_size > 0
    all_passed = all_passed and written
    print(f"  [{'PASS' if written else 'FAIL'}] results written to {report_path.name}")
    return all_passed


def main() -> int:
    """Load data, train and score the five models (one MLflow run each), then save plots, report and checks."""
    tracking_uri = setup_mlflow(EXPERIMENT_NAME)
    print(f"MLflow experiment '{EXPERIMENT_NAME}' at {tracking_uri}")

    data_hash = read_dvc_data_hash()
    print(f"DVC data hash: {data_hash}")

    print("Loading train and test splits (the calibration split is not touched)...")
    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)
    assert list(X_train.columns) == list(X_test.columns), "Train and test feature columns differ"
    print(f"Features ({X_train.shape[1]}): {list(X_train.columns)}")
    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight (negatives / positives) = {scale_pos_weight:.4f}")

    # Settings every run shares, logged as MLflow params so each run records what it was trained on.
    shared_params = {
        "seed": RANDOM_SEED,
        "dvc_data_hash": data_hash,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": X_train.shape[1],
        "target": TARGET_COLUMN,
    }

    results = []
    run_ids = []
    for spec in get_model_specs(scale_pos_weight):
        print(f"Training {spec['name']}...")
        with mlflow.start_run(run_name=spec["name"]) as run:
            fitted = fit_model(spec, X_train, y_train)
            scores = evaluate_model(fitted, X_test, y_test)
            log_run_metrics(spec, fitted, scores, shared_params)
            run_ids.append(run.info.run_id)

        results.append({"name": spec["name"], "fit_seconds": fitted["fit_seconds"], **scores})
        print(
            f"  fit {fitted['fit_seconds']:.1f}s  ROC-AUC {scores['roc_auc']:.4f}  PR-AUC {scores['pr_auc']:.4f}"
        )

    print("Saving plots and report...")
    plot_paths = make_plots(results)
    report_path = write_report(results, data_hash)
    log_artifacts_to_runs(run_ids, plot_paths + [report_path])

    print()
    print(report_path.read_text())
    passed = check_results(results, report_path)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
