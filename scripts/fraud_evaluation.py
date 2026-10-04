# File: scripts/fraud_evaluation.py
# Purpose: Sub-phase 5.4 - retrain the three tuned models on the full train split, score them once on the test
#          split, pick the winner by PR-AUC, and produce per-segment metrics, a confusion matrix, feature
#          importance and a SHAP explainer (saved to the frozen path models/xgboost/shap_model.pkl).
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Draw plots straight to files; no window is needed.
matplotlib.use("Agg")

# Make the repo root and this scripts/ folder importable, so the file works when run directly or loaded by tests.
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from fraud_baseline import (  # noqa: E402
    AMOUNT_AND_BALANCE_COLUMNS,
    ENGINEERED_BOOL_COLUMNS,
    ENGINEERED_NUMBER_COLUMNS,
    TARGET_COLUMN,
    compute_scale_pos_weight,
    prepare_features,
    read_dvc_data_hash,
)
from fraud_tuning import build_model, load_train_features  # noqa: E402

from core.mlflow_setup import setup_mlflow  # noqa: E402

# XGBoost is asked to predict with CPU data while the model sits on the GPU; it handles this itself and warns.
warnings.filterwarnings("ignore", message=".*mismatched devices.*")

TEST_PATH = REPO_ROOT / "data" / "processed" / "test.parquet"
REPORT_DIR = REPO_ROOT / "reports" / "model"
BEST_PARAMS_PATH = REPORT_DIR / "best_params.json"
EVALUATION_PATH = REPORT_DIR / "evaluation.md"
IMPORTANCE_PLOT_PATH = REPORT_DIR / "fraud_feature_importance_top20.png"
SHAP_PLOT_PATH = REPORT_DIR / "fraud_shap_summary.png"

# The frozen path. Its name says "xgboost", but it holds the explainer of whichever model actually wins.
SHAP_PICKLE_PATH = REPO_ROOT / "models" / "xgboost" / "shap_model.pkl"
SHAP_NOTE_PATH = REPO_ROOT / "models" / "xgboost" / "shap_model_NOTE.md"

TUNING_EXPERIMENT = "level5_fraud_tuning"
EVALUATION_EXPERIMENT = "level5_fraud_evaluation"
MODEL_NAMES = ["xgboost", "catboost", "random_forest"]
RANDOM_SEED = 42
THRESHOLD = 0.5  # provisional operating point; real threshold tuning is not part of this step
SHAP_SAMPLE_SIZE = 50_000
TOP_FEATURES = 20

# Columns used only to slice the test set into segments (several are also model inputs).
SEGMENT_COLUMNS = ["type", "is_adversarial", "adversarial_pattern_type", "amount", "merchant_risk_tier"]
ALL_TYPES = ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]


# --- Step 1: tuned hyperparameters -> best_params.json ----------------------------------------------------
def convert_param(text: str):
    """Turn a hyperparameter that MLflow stored as text back into a real Python value."""
    # MLflow only stores text. "None" must become the real None (for example an unlimited tree depth), and the
    # words "sqrt" / "log2" are genuine string options for random forest max_features, so they stay strings.
    if text == "None":
        return None
    if text in ("sqrt", "log2"):
        return text
    # Whole numbers become ints. Anything else (for example max_features "0.5") must become a FLOAT, because a
    # random forest treats a float max_features as a fraction of the features, which is what was tuned.
    try:
        return int(text)
    except ValueError:
        return float(text)


def read_best_params_from_mlflow() -> dict:
    """Read each model's best_* hyperparameters from its tuning parent run in MLflow, with real types."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(TUNING_EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id], max_results=500)

    best_params = {}
    for name in MODEL_NAMES:
        parent_runs = [r for r in runs if r.data.tags.get("mlflow.runName") == f"tune_{name}"]
        assert len(parent_runs) == 1, f"Expected exactly one tune_{name} run, found {len(parent_runs)}"
        params = parent_runs[0].data.params
        # best_trial_number is bookkeeping, not a hyperparameter.
        best_params[name] = {
            key[len("best_"):]: convert_param(value)
            for key, value in params.items()
            if key.startswith("best_") and key != "best_trial_number"
        }
    return best_params


def save_best_params(best_params: dict) -> None:
    """Write the tuned hyperparameters to reports/model/best_params.json."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as params_file:
        json.dump(best_params, params_file, indent=2)


def load_best_params() -> dict:
    """Read the hyperparameters back from the file (the rest of this script uses the file, not live MLflow)."""
    with open(BEST_PARAMS_PATH) as params_file:
        return json.load(params_file)


# --- Step 2: retrain on the full train split ----------------------------------------------------------------
def build_cpu_fallback(name: str, params: dict, scale_pos_weight: float):
    """Build the CPU version of a GPU model, used only if the GPU fit fails."""
    model = build_model(name, params, scale_pos_weight)
    if name == "xgboost":
        model.set_params(device="cpu")
        return model
    # CatBoost: the same fixed settings as the GPU builder, but on the CPU.
    import catboost

    return catboost.CatBoostClassifier(
        task_type="CPU", auto_class_weights="Balanced", random_seed=RANDOM_SEED, verbose=0,
        allow_writing_files=False, **params,
    )


def fit_full_train(name: str, params: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit one tuned model on the full train split; if the GPU errors, report it exactly and refit on the CPU."""
    scale_pos_weight = compute_scale_pos_weight(y_train)
    model = build_model(name, params, scale_pos_weight)
    fallback_error = ""
    start = time.perf_counter()
    try:
        model.fit(X_train, y_train)
    except Exception as error:  # noqa: BLE001 - any GPU failure is reported exactly, then retried once on the CPU
        if name == "random_forest":
            raise
        fallback_error = f"{type(error).__name__}: {error}"
        print(f"  GPU ERROR for {name}: {fallback_error}")
        print(f"  FALLING BACK TO CPU for {name}")
        start = time.perf_counter()
        model = build_cpu_fallback(name, params, scale_pos_weight)
        model.fit(X_train, y_train)
    return {"model": model, "fit_seconds": time.perf_counter() - start, "fallback_error": fallback_error}


# --- Data ---------------------------------------------------------------------------------------------------
def load_test_with_segments() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Read the test split (needed columns only) and return features, target and the segment columns."""
    needed = list(
        dict.fromkeys(
            AMOUNT_AND_BALANCE_COLUMNS + ENGINEERED_BOOL_COLUMNS + ENGINEERED_NUMBER_COLUMNS
            + SEGMENT_COLUMNS + [TARGET_COLUMN]
        )
    )
    raw = pd.read_parquet(TEST_PATH, columns=needed).reset_index(drop=True)
    X, y = prepare_features(raw)
    return X, y, raw[SEGMENT_COLUMNS]


# --- Step 3: scoring and winner ----------------------------------------------------------------------------
def score_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score a fitted model on the test split: fraud probabilities, PR-AUC and ROC-AUC."""
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "probability": probability,
        "pr_auc": average_precision_score(y_test, probability),
        "roc_auc": roc_auc_score(y_test, probability),
    }


def select_winner(results: dict) -> str:
    """Return the model with the highest test PR-AUC; an exact tie is broken by the higher ROC-AUC."""
    return max(results, key=lambda name: (results[name]["pr_auc"], results[name]["roc_auc"]))


# --- Step 4: per-segment metrics ----------------------------------------------------------------------------
def segment_scores(y: pd.Series, probability: np.ndarray, mask: pd.Series, label: str) -> dict:
    """PR-AUC and ROC-AUC on the rows picked by `mask`; undefined (None) when fraud or non-fraud is missing."""
    y_part = y[mask]
    frauds = int(y_part.sum())
    row = {"segment": label, "rows": int(len(y_part)), "fraud": frauds, "pr_auc": None, "roc_auc": None}
    if frauds == 0:
        row["note"] = "no fraud present - metric undefined"
    elif frauds == len(y_part):
        row["note"] = "no non-fraud rows - metric undefined"
    else:
        row["pr_auc"] = average_precision_score(y_part, probability[mask.to_numpy()])
        row["roc_auc"] = roc_auc_score(y_part, probability[mask.to_numpy()])
        row["note"] = ""
    return row


def build_segment_table(y: pd.Series, probability: np.ndarray, segments: pd.DataFrame) -> tuple[list[dict], dict]:
    """Score the winner overall and on every segment; return the rows and the amount cut points used."""
    rows = [segment_scores(y, probability, pd.Series(True, index=y.index), "Overall")]

    # By transaction type. Types with no fraud come back as "undefined" instead of an invented number.
    for transaction_type in ALL_TYPES:
        mask = segments["type"] == transaction_type
        rows.append(segment_scores(y, probability, mask, f"Type: {transaction_type}"))

    # By amount tier, with the cut points recomputed from the test split's own amounts: bottom quartile,
    # middle two quartiles, top quartile.
    q1, q3 = segments["amount"].quantile([0.25, 0.75])
    tiers = {
        "Amount: bottom quartile": segments["amount"] <= q1,
        "Amount: middle two quartiles": (segments["amount"] > q1) & (segments["amount"] < q3),
        "Amount: top quartile": segments["amount"] >= q3,
    }
    for label, mask in tiers.items():
        rows.append(segment_scores(y, probability, mask, label))

    # Each injected pattern is scored against ALL non-fraud rows (its own rows are all fraud).
    for pattern in ("account_takeover", "mule_account"):
        mask = (segments["adversarial_pattern_type"] == pattern) | (y == 0)
        rows.append(segment_scores(y, probability, mask, f"Pattern: {pattern} vs non-fraud"))

    # PaySim-native fraud: rows that are not adversarial (native fraud plus all non-fraud rows).
    native_mask = ~segments["is_adversarial"].astype(bool)
    rows.append(segment_scores(y, probability, native_mask, "PaySim-native fraud (is_adversarial=False)"))
    return rows, {"q1": float(q1), "q3": float(q3)}


# --- Step 5: confusion matrix and importance ----------------------------------------------------------------
def confusion_counts(y: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    """Confusion matrix counts at the given threshold."""
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def save_importance_plot(model, feature_names: list[str], winner: str) -> list[tuple[str, float]]:
    """Save a bar chart of the model's native feature importance (top 20; the contract only has 15 features)."""
    importance = sorted(zip(feature_names, model.feature_importances_), key=lambda pair: pair[1], reverse=True)
    top = importance[:TOP_FEATURES]
    plt.figure(figsize=(8, 6))
    plt.barh([name for name, _ in reversed(top)], [float(value) for _, value in reversed(top)])
    plt.xlabel("Native feature importance")
    plt.title(f"{winner}: top feature importances")
    plt.savefig(IMPORTANCE_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    return [(name, float(value)) for name, value in top]


# --- Step 6: SHAP -------------------------------------------------------------------------------------------
def make_shap_sample(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Draw a stratified 50,000-row sample of the test set (fraud ratio preserved), seed 42."""
    sample, _ = train_test_split(X_test, train_size=SHAP_SAMPLE_SIZE, stratify=y_test, random_state=RANDOM_SEED)
    return sample


def positive_class_shap_values(raw_values) -> np.ndarray:
    """Return a (rows x features) array of SHAP values for the fraud class, whatever shape the explainer gave."""
    # Depending on the model, SHAP returns one array, a list with one array per class, or a 3-D array with a
    # class axis. In the last two cases we keep the fraud class (index 1).
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1])
    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def save_shap_outputs(model, shap_sample: pd.DataFrame, winner: str, feature_names: list[str]) -> shap.TreeExplainer:
    """Build the TreeExplainer, draw the SHAP summary plot, and pickle the explainer to the frozen path."""
    explainer = shap.TreeExplainer(model)
    shap_values = positive_class_shap_values(explainer.shap_values(shap_sample))

    plt.figure()
    shap.summary_plot(shap_values, shap_sample, show=False)
    plt.savefig(SHAP_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()

    SHAP_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, SHAP_PICKLE_PATH)
    SHAP_NOTE_PATH.write_text(
        "# shap_model.pkl\n\n"
        f"- Contents: a fitted `shap.TreeExplainer` (saved with `joblib.dump`) for the winning model: **{winner}**.\n"
        f"- Features it expects, in this exact order ({len(feature_names)}): {', '.join(feature_names)}.\n"
        "- Load with `joblib.load('models/xgboost/shap_model.pkl')`, then call `explainer.shap_values(X)` with a\n"
        "  DataFrame holding those columns in that order.\n"
        "- The path says `xgboost` because the path is frozen; the explainer is for whichever model won evaluation.\n"
        "- The `.pkl` file itself is gitignored (models/**/*.pkl); this note is tracked.\n"
    )
    return explainer


# --- Step 7: report -----------------------------------------------------------------------------------------
def format_metric(value) -> str:
    """Show a metric to 4 decimals, or 'undefined' when it could not be computed."""
    return "undefined" if value is None else f"{value:.4f}"


def write_evaluation_report(context: dict) -> None:
    """Write reports/model/evaluation.md from the collected results."""
    results = context["results"]
    lines = [
        "# Fraud model evaluation (sub-phase 5.4)",
        "",
        f"Data: corrected PaySim dataset, DVC data hash `{context['data_hash']}`. Models trained on the full train "
        f"split ({context['train_rows']:,} rows) with tuned hyperparameters, scored once on the test split "
        f"({context['test_rows']:,} rows, {context['test_fraud']:,} fraud).",
        "",
        "## Test-set results (sorted by PR-AUC)",
        "",
        "| Model | PR-AUC | ROC-AUC | Train seconds |",
        "|---|---|---|---|",
    ]
    for name in sorted(results, key=lambda n: results[n]["pr_auc"], reverse=True):
        r = results[name]
        lines.append(f"| {name} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['fit_seconds']:.1f} |")
    lines += [
        "",
        f"**Winner: {context['winner']}** - selection rule: highest test PR-AUC (an exact tie would be broken by "
        "ROC-AUC).",
        "",
        f"## Per-segment metrics for {context['winner']}",
        "",
        "| Segment | Rows | Fraud | PR-AUC | ROC-AUC | Note |",
        "|---|---|---|---|---|---|",
    ]
    for row in context["segments"]:
        lines.append(
            f"| {row['segment']} | {row['rows']:,} | {row['fraud']:,} | {format_metric(row['pr_auc'])} | "
            f"{format_metric(row['roc_auc'])} | {row['note']} |"
        )
    cuts = context["cuts"]
    cm = context["confusion"]
    lines += [
        "",
        f"Amount tiers use the test split's own quartile cut points: {cuts['q1']:,.2f} and {cuts['q3']:,.2f}.",
        "",
        f"## Confusion matrix at threshold {THRESHOLD} (provisional operating point, not tuned)",
        "",
        "| | Predicted non-fraud | Predicted fraud |",
        "|---|---|---|",
        f"| Actual non-fraud | {cm['tn']:,} | {cm['fp']:,} |",
        f"| Actual fraud | {cm['fn']:,} | {cm['tp']:,} |",
        "",
        "## Plots",
        "",
        f"- Feature importance: `reports/model/{IMPORTANCE_PLOT_PATH.name}`",
        f"- SHAP summary ({SHAP_SAMPLE_SIZE:,}-row stratified test sample): `reports/model/{SHAP_PLOT_PATH.name}`",
        "",
        "Note: PaySim-native fraud is a small segment of the test set (about a quarter of the fraud rows), so the "
        "overall score is dominated by the injected account_takeover and mule_account patterns; read the native "
        "segment line to see how the model does on native fraud alone.",
    ]
    EVALUATION_PATH.write_text("\n".join(lines) + "\n")


# --- Step 8: MLflow ------------------------------------------------------------------------------------------
def segment_metric_name(label: str) -> str:
    """Turn a segment label into a safe MLflow metric name."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in label.lower())
    return "seg_" + "_".join(part for part in safe.split("_") if part)[:60]


def log_to_mlflow(context: dict) -> str:
    """Log params, metrics and artifacts for the evaluation as one MLflow run; return its run ID."""
    winner = context["winner"]
    with mlflow.start_run(run_name=f"evaluation_{winner}") as run:
        mlflow.log_params(
            {
                "winner": winner, "seed": RANDOM_SEED, "dvc_data_hash": context["data_hash"],
                "train_rows": context["train_rows"], "test_rows": context["test_rows"],
                "threshold": THRESHOLD, "shap_sample_size": SHAP_SAMPLE_SIZE,
                "selection_rule": "highest test PR-AUC",
            }
        )
        mlflow.log_params({f"winner_{key}": str(value) for key, value in context["best_params"][winner].items()})
        for name, result in context["results"].items():
            mlflow.log_metric(f"{name}_test_pr_auc", result["pr_auc"])
            mlflow.log_metric(f"{name}_test_roc_auc", result["roc_auc"])
        mlflow.log_metric("overall_test_pr_auc", context["results"][winner]["pr_auc"])
        mlflow.log_metric("overall_test_roc_auc", context["results"][winner]["roc_auc"])
        for row in context["segments"]:
            if row["pr_auc"] is not None:
                mlflow.log_metric(segment_metric_name(row["segment"]) + "_pr_auc", row["pr_auc"])
        for key, value in context["confusion"].items():
            mlflow.log_metric(f"confusion_{key}", value)
        for artifact in (EVALUATION_PATH, BEST_PARAMS_PATH, IMPORTANCE_PLOT_PATH, SHAP_PLOT_PATH, SHAP_PICKLE_PATH):
            mlflow.log_artifact(str(artifact))
        return run.info.run_id


# --- Acceptance checks --------------------------------------------------------------------------------------
def check_results(context: dict) -> bool:
    """Acceptance: winner follows the rule, every segment has a real number or 'undefined', SHAP pickle reloads."""
    results = context["results"]
    print("\nAcceptance checks:")

    best_by_hand = max(results, key=lambda name: results[name]["pr_auc"])
    rule_ok = context["winner"] == best_by_hand
    scores = ", ".join(f"{name} {results[name]['pr_auc']:.4f}" for name in results)
    print(f"  [{'PASS' if rule_ok else 'FAIL'}] winner {context['winner']} has the highest test PR-AUC ({scores})")

    expected_segments = 1 + len(ALL_TYPES) + 3 + 2 + 1
    segments_ok = len(context["segments"]) == expected_segments and all(
        (row["pr_auc"] is not None) or ("undefined" in row["note"]) for row in context["segments"]
    )
    print(f"  [{'PASS' if segments_ok else 'FAIL'}] all {expected_segments} segments have a real number or 'undefined'")

    shap_ok = SHAP_PICKLE_PATH.exists()
    if shap_ok:
        reloaded = joblib.load(SHAP_PICKLE_PATH)
        shap_ok = isinstance(reloaded, shap.TreeExplainer)
    shap_location = SHAP_PICKLE_PATH.relative_to(REPO_ROOT).as_posix()
    print(f"  [{'PASS' if shap_ok else 'FAIL'}] SHAP explainer saved at {shap_location} and loads back")
    return rule_ok and segments_ok and shap_ok


def main() -> int:
    """Run the full 5.4 evaluation."""
    tracking_uri = setup_mlflow(TUNING_EXPERIMENT)
    print(f"MLflow store: {tracking_uri}")

    # Step 1: tuned hyperparameters from MLflow into a file, then read them back from the file.
    save_best_params(read_best_params_from_mlflow())
    best_params = load_best_params()
    print(f"Saved and reloaded {BEST_PARAMS_PATH.relative_to(REPO_ROOT)}")

    data_hash = read_dvc_data_hash()
    print("Loading the full train split...")
    X_train, y_train = load_train_features()
    train_rows = len(X_train)
    print(f"Train rows: {train_rows:,}  fraud ratio {y_train.mean() * 100:.4f}%")

    # Step 2 and 3: retrain on the full train split, then score once on the test split.
    fitted = {}
    for name in MODEL_NAMES:
        print(f"Training {name} on the full train split...")
        fitted[name] = fit_full_train(name, best_params[name], X_train, y_train)
        print(f"  fit {fitted[name]['fit_seconds']:.1f}s")
    del X_train, y_train  # the test split is loaded only after training is finished

    X_test, y_test, segments = load_test_with_segments()
    feature_names = list(X_test.columns)
    print(f"Test rows: {len(X_test):,}  fraud {int(y_test.sum()):,}")

    results = {}
    for name in MODEL_NAMES:
        scores = score_model(fitted[name]["model"], X_test, y_test)
        results[name] = {**scores, "fit_seconds": fitted[name]["fit_seconds"]}
        print(f"  {name:<14} test PR-AUC {scores['pr_auc']:.4f}  ROC-AUC {scores['roc_auc']:.4f}")
    winner = select_winner(results)
    print(f"WINNER (highest test PR-AUC): {winner}")

    # Step 4 and 5: per-segment metrics, confusion matrix, feature importance (winner only).
    winner_model = fitted[winner]["model"]
    segment_rows, cuts = build_segment_table(y_test, results[winner]["probability"], segments)
    confusion = confusion_counts(y_test, results[winner]["probability"], THRESHOLD)
    top_features = save_importance_plot(winner_model, feature_names, winner)

    # Step 6: SHAP on a stratified subsample of the test set, explainer pickled to the frozen path.
    save_shap_outputs(winner_model, make_shap_sample(X_test, y_test), winner, feature_names)

    context = {
        "results": results, "winner": winner, "segments": segment_rows, "cuts": cuts, "confusion": confusion,
        "best_params": best_params, "data_hash": data_hash, "train_rows": train_rows,
        "test_rows": len(X_test), "test_fraud": int(y_test.sum()),
    }

    write_evaluation_report(context)
    setup_mlflow(EVALUATION_EXPERIMENT)
    run_id = log_to_mlflow(context)

    print("\nPer-segment metrics:")
    for row in segment_rows:
        print(f"  {row['segment']:<46} rows {row['rows']:>9,} fraud {row['fraud']:>6,} "
              f"PR-AUC {format_metric(row['pr_auc'])} ROC-AUC {format_metric(row['roc_auc'])} {row['note']}")
    print(f"Confusion matrix at {THRESHOLD}: {confusion}")
    print(f"Top features: {[name for name, _ in top_features[:5]]} ...")
    print(f"MLflow experiment '{EVALUATION_EXPERIMENT}', run ID {run_id}")
    return 0 if check_results(context) else 1


if __name__ == "__main__":
    sys.exit(main())
