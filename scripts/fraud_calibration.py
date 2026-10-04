# File: scripts/fraud_calibration.py
# Purpose: Sub-phase 5.5 - retrain and SAVE the winning XGBoost, calibrate its scores with isotonic regression on
#          the calibration split, check reliability (10 bins within 10 percentage points), and analyse thresholds.
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xgboost
from sklearn.isotonic import IsotonicRegression

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
    prepare_features,
    read_dvc_data_hash,
)
from fraud_evaluation import fit_full_train, load_best_params  # noqa: E402
from fraud_tuning import load_train_features  # noqa: E402

from core.mlflow_setup import setup_mlflow  # noqa: E402

CALIBRATION_PATH = REPO_ROOT / "data" / "processed" / "calibration.parquet"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test.parquet"
MODEL_PATH = REPO_ROOT / "models" / "xgboost" / "model.ubj"
CALIBRATOR_PATH = REPO_ROOT / "models" / "calibration" / "isotonic_calibrator.pkl"
REPORT_DIR = REPO_ROOT / "reports" / "calibration"
RELIABILITY_PLOT_PATH = REPORT_DIR / "reliability_diagram.png"
THRESHOLD_PLOT_PATH = REPORT_DIR / "threshold_analysis.png"
SUMMARY_PATH = REPORT_DIR / "calibration_summary.md"

EXPERIMENT_NAME = "level5_fraud_calibration"
RANDOM_SEED = 42
N_BINS = 10
BIN_TOLERANCE = 0.10  # every bin must be within 10 percentage points of the diagonal
HIGH_RECALL_TARGET = 0.995


# --- Step 1: the single saved model ------------------------------------------------------------------------
def train_and_save_model() -> xgboost.XGBClassifier:
    """Retrain the tuned XGBoost on the full train split and save it to disk."""
    best_params = load_best_params()["xgboost"]
    X_train, y_train = load_train_features()
    print(f"Retraining XGBoost on {len(X_train):,} train rows...")
    fitted = fit_full_train("xgboost", best_params, X_train, y_train)
    print(f"  fit {fitted['fit_seconds']:.1f}s" + (f"  (GPU->CPU fallback: {fitted['fallback_error']})"
                                                   if fitted["fallback_error"] else "  (GPU, no fallback)"))

    # This saved file is the ONE source of truth: calibration, ONNX export and threshold analysis all load it, so
    # every downstream artifact describes exactly the same model. (.ubj is XGBoost's compact binary format.)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fitted["model"].save_model(str(MODEL_PATH))
    return fitted["model"]


def load_saved_model() -> xgboost.XGBClassifier:
    """Load the saved XGBoost back from disk."""
    model = xgboost.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read one split (needed columns only) and build the 15-feature matrix and target."""
    needed = AMOUNT_AND_BALANCE_COLUMNS + ENGINEERED_BOOL_COLUMNS + ENGINEERED_NUMBER_COLUMNS + ["type", TARGET_COLUMN]
    raw = pd.read_parquet(path, columns=needed)
    return prepare_features(raw)


# --- Step 2: calibration -----------------------------------------------------------------------------------
def fit_calibrator(raw_scores: np.ndarray, y: pd.Series) -> IsotonicRegression:
    """Fit isotonic regression mapping raw model scores to calibrated fraud probabilities."""
    # y_min / y_max keep outputs inside [0, 1]; out_of_bounds="clip" gives sensible answers for scores outside
    # the range seen during fitting (isotonic regression is a step function, so it cannot extrapolate).
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_scores, y.to_numpy())
    return calibrator


# --- Step 3: reliability ------------------------------------------------------------------------------------
def reliability_table(y: pd.Series, probability: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Group rows into equal-width probability bins; per bin report count, mean prediction and actual fraud rate."""
    y_values = y.to_numpy()
    # A probability of exactly 1.0 belongs in the last bin, so bin numbers are capped at n_bins - 1.
    bin_number = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        in_bin = bin_number == bin_index
        count = int(in_bin.sum())
        row = {"bin": f"{bin_index / n_bins:.1f}-{(bin_index + 1) / n_bins:.1f}", "count": count,
               "mean_predicted": None, "actual": None, "deviation": None}
        if count > 0:
            row["mean_predicted"] = float(probability[in_bin].mean())
            row["actual"] = float(y_values[in_bin].mean())
            row["deviation"] = abs(row["mean_predicted"] - row["actual"])
        rows.append(row)
    return rows


def check_bins(rows: list[dict], tolerance: float = BIN_TOLERANCE) -> bool:
    """True when every non-empty bin is within `tolerance` of the diagonal; empty bins cannot be checked."""
    return all(row["deviation"] <= tolerance for row in rows if row["deviation"] is not None)


def check_calibrated_probabilities(calibrator: IsotonicRegression, raw_scores: np.ndarray) -> dict:
    """Confirm calibrated outputs stay inside [0, 1] and never decrease as the raw score rises."""
    order = np.argsort(raw_scores)
    calibrated = calibrator.predict(raw_scores[order])
    return {
        "in_range": bool(calibrated.min() >= 0.0 and calibrated.max() <= 1.0),
        "monotonic": bool(np.all(np.diff(calibrated) >= 0.0)),
        "min": float(calibrated.min()),
        "max": float(calibrated.max()),
    }


# --- Step 4: thresholds -------------------------------------------------------------------------------------
def threshold_grid() -> np.ndarray:
    """Thresholds to try: a few very low ones (calibrated fraud scores can be tiny), then 0.01 to 0.99."""
    return np.concatenate([[0.001, 0.002, 0.005], np.round(np.arange(0.01, 1.0, 0.01), 2)])


def metrics_at(y: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1 and the raw counts when everything at or above `threshold` is flagged as fraud."""
    actual = y.to_numpy() == 1
    flagged = probability >= threshold
    tp = int((flagged & actual).sum())
    fp = int((flagged & ~actual).sum())
    fn = int((~flagged & actual).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def sweep_thresholds(y: pd.Series, probability: np.ndarray) -> list[dict]:
    """Compute the metrics at every threshold in the grid."""
    return [metrics_at(y, probability, threshold) for threshold in threshold_grid()]


def choose_f1_optimal(rows: list[dict]) -> dict:
    """The threshold with the best F1 (if several tie, the first/lowest one is kept)."""
    return max(rows, key=lambda row: row["f1"])


def choose_high_recall(rows: list[dict], min_recall: float = HIGH_RECALL_TARGET) -> dict | None:
    """The HIGHEST threshold that still keeps recall at or above `min_recall`; None if no threshold reaches it."""
    qualifying = [row for row in rows if row["recall"] >= min_recall]
    return max(qualifying, key=lambda row: row["threshold"]) if qualifying else None


# --- Plots --------------------------------------------------------------------------------------------------
def save_reliability_plot(calibration_rows: list[dict], test_rows: list[dict]) -> None:
    """Reliability diagram: mean predicted probability vs actual fraud rate per bin, against the diagonal."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for rows, label, marker in ((calibration_rows, "calibration split (used to fit)", "o"),
                                (test_rows, "test split (unseen)", "s")):
        points = [(r["mean_predicted"], r["actual"]) for r in rows if r["count"] > 0]
        plt.plot([p[0] for p in points], [p[1] for p in points], marker=marker, label=label)
    plt.xlabel("Mean predicted fraud probability (calibrated)")
    plt.ylabel("Actual fraud fraction")
    plt.title("Reliability diagram (10 equal-width bins)")
    plt.legend()
    plt.savefig(RELIABILITY_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()


def save_threshold_plot(rows: list[dict], f1_point: dict, recall_point: dict | None) -> None:
    """Precision, recall and F1 against the threshold, with the two candidate operating points marked."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = [r["threshold"] for r in rows]
    plt.figure(figsize=(8, 5))
    for key in ("precision", "recall", "f1"):
        plt.plot(thresholds, [r[key] for r in rows], label=key)
    plt.axvline(f1_point["threshold"], color="green", linestyle=":", label=f"F1-optimal ({f1_point['threshold']})")
    if recall_point is not None:
        plt.axvline(recall_point["threshold"], color="red", linestyle=":",
                    label=f"high-recall ({recall_point['threshold']})")
    plt.xlabel("Threshold on calibrated probability")
    plt.ylabel("Score (calibration split)")
    plt.title("Threshold analysis")
    plt.legend()
    plt.savefig(THRESHOLD_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()


# --- Report and MLflow --------------------------------------------------------------------------------------
def fmt(value, digits: int = 4) -> str:
    """Format a number, or a dash when it does not exist (an empty bin)."""
    return "-" if value is None else f"{value:.{digits}f}"


def bin_table_lines(rows: list[dict]) -> list[str]:
    """Markdown table lines for a reliability table."""
    lines = [
        "| Bin | Rows | Mean predicted | Actual fraud rate | Deviation | Within 10pp |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        verdict = "empty" if row["deviation"] is None else ("PASS" if row["deviation"] <= BIN_TOLERANCE else "FAIL")
        lines.append(f"| {row['bin']} | {row['count']:,} | {fmt(row['mean_predicted'])} | {fmt(row['actual'])} | "
                     f"{fmt(row['deviation'])} | {verdict} |")
    return lines


def point_line(label: str, point: dict | None, test_point: dict | None) -> str:
    """One markdown table row describing a candidate operating point on calibration and test."""
    if point is None:
        return f"| {label} | not reachable | - | - | - | - |"
    return (f"| {label} | {point['threshold']} | "
            f"{point['precision']:.4f} / {point['recall']:.4f} / {point['f1']:.4f} | "
            f"{point['fp']:,} / {point['fn']:,} | {test_point['precision']:.4f} / {test_point['recall']:.4f} / "
            f"{test_point['f1']:.4f} | {test_point['fp']:,} / {test_point['fn']:,} |")


def write_summary(context: dict) -> None:
    """Write reports/calibration/calibration_summary.md."""
    checks = context["checks"]
    lines = [
        "# Calibration summary (sub-phase 5.5)",
        "",
        f"Model: tuned XGBoost retrained on the full train split and saved at `models/xgboost/model.ubj`. "
        f"Isotonic calibrator fitted on the calibration split only ({context['calibration_rows']:,} rows). "
        f"DVC data hash `{context['data_hash']}`.",
        "",
        "## Reliability on the calibration split (10 equal-width bins; the acceptance check)",
        "",
        *bin_table_lines(context["calibration_bins"]),
        "",
        f"All non-empty bins within 10pp: **{'PASS' if context['calibration_bins_ok'] else 'FAIL'}**. This split "
        "is the one the calibrator was fitted on, so it is an in-sample check and is expected to look good.",
        "",
        "## Reliability on the test split (unseen; confirmation only)",
        "",
        *bin_table_lines(context["test_bins"]),
        "",
        f"All non-empty bins within 10pp on test: **{'PASS' if context['test_bins_ok'] else 'FAIL'}**. Bins with few "
        "rows swing easily, so read each deviation together with its row count.",
        "",
        "## Probability checks",
        "",
        f"- Calibrated probabilities inside [0, 1]: **{checks['in_range']}** (min {checks['min']:.6f}, "
        f"max {checks['max']:.6f})",
        f"- Monotonic in the raw score: **{checks['monotonic']}**",
        "",
        "## Threshold analysis (thresholds chosen on the calibration split; test is confirmation only)",
        "",
        "| Operating point | Threshold | Calibration precision / recall / F1 | Calibration FP / FN | "
        "Test precision / recall / F1 | Test FP / FN |",
        "|---|---|---|---|---|---|",
        point_line("(a) F1-optimal", context["f1_point"], context["f1_test"]),
        point_line(f"(b) High-recall (>= {HIGH_RECALL_TARGET})", context["recall_point"], context["recall_test"]),
        "",
        "Neither point is recommended here. The final operating point is a business decision (missed fraud versus "
        "false alarms) and is not finalised in Level 5.",
        "",
        f"Plots: `reports/calibration/{RELIABILITY_PLOT_PATH.name}`, `reports/calibration/{THRESHOLD_PLOT_PATH.name}`.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def log_to_mlflow(context: dict) -> str:
    """Log the calibration run to MLflow and return its run ID."""
    with mlflow.start_run(run_name="calibration_xgboost") as run:
        mlflow.log_params({"seed": RANDOM_SEED, "dvc_data_hash": context["data_hash"],
                           "calibration_rows": context["calibration_rows"], "n_bins": N_BINS,
                           "bin_tolerance": BIN_TOLERANCE, "model_path": "models/xgboost/model.ubj"})
        deviations = [r["deviation"] for r in context["calibration_bins"] if r["deviation"] is not None]
        test_deviations = [r["deviation"] for r in context["test_bins"] if r["deviation"] is not None]
        mlflow.log_metric("max_bin_deviation_calibration", max(deviations))
        mlflow.log_metric("max_bin_deviation_test", max(test_deviations))
        for label, point, test_point in (("f1_optimal", context["f1_point"], context["f1_test"]),
                                         ("high_recall", context["recall_point"], context["recall_test"])):
            if point is None:
                continue
            mlflow.log_metric(f"{label}_threshold", point["threshold"])
            for key in ("precision", "recall", "f1"):
                mlflow.log_metric(f"{label}_test_{key}", test_point[key])
            mlflow.log_metric(f"{label}_test_fp", test_point["fp"])
            mlflow.log_metric(f"{label}_test_fn", test_point["fn"])
        for artifact in (RELIABILITY_PLOT_PATH, THRESHOLD_PLOT_PATH, SUMMARY_PATH, CALIBRATOR_PATH, MODEL_PATH):
            mlflow.log_artifact(str(artifact))
        return run.info.run_id


def main() -> int:
    """Retrain and save the model, calibrate it, check reliability, analyse thresholds, and report."""
    setup_mlflow(EXPERIMENT_NAME)
    data_hash = read_dvc_data_hash()

    X_cal, y_cal = load_features(CALIBRATION_PATH)
    X_test, y_test = load_features(TEST_PATH)

    in_memory_model = train_and_save_model()
    model = load_saved_model()  # from here on everything uses the SAVED model, proving it is the single source
    print(f"Saved and reloaded {MODEL_PATH.relative_to(REPO_ROOT).as_posix()}")

    # Sanity check that saving and loading lost nothing: both copies must give the same scores on a sample.
    sample = X_cal.iloc[:20_000]
    gap = float(np.abs(in_memory_model.predict_proba(sample)[:, 1] - model.predict_proba(sample)[:, 1]).max())
    print(f"Max score difference, in-memory vs reloaded model (20,000 rows): {gap:.2e}")
    assert gap <= 1e-6, f"Reloaded model differs from the trained model by {gap}"
    del in_memory_model

    raw_cal = model.predict_proba(X_cal)[:, 1]
    raw_test = model.predict_proba(X_test)[:, 1]
    print(f"Calibration rows {len(y_cal):,} (fraud {int(y_cal.sum()):,}); test rows {len(y_test):,}")

    # Step 2: fit the calibrator on the calibration split ONLY, then apply it to both splits.
    calibrator = fit_calibrator(raw_cal, y_cal)
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    prob_cal = calibrator.predict(raw_cal)
    prob_test = calibrator.predict(raw_test)

    # Step 3: reliability tables and checks.
    calibration_bins = reliability_table(y_cal, prob_cal)
    test_bins = reliability_table(y_test, prob_test)
    checks = check_calibrated_probabilities(calibrator, raw_cal)
    save_reliability_plot(calibration_bins, test_bins)

    # Step 4: thresholds are CHOSEN on the calibration split and only CONFIRMED on the test split.
    sweep = sweep_thresholds(y_cal, prob_cal)
    f1_point = choose_f1_optimal(sweep)
    recall_point = choose_high_recall(sweep)
    f1_test = metrics_at(y_test, prob_test, f1_point["threshold"])
    recall_test = metrics_at(y_test, prob_test, recall_point["threshold"]) if recall_point else None
    save_threshold_plot(sweep, f1_point, recall_point)

    context = {
        "data_hash": data_hash, "calibration_rows": len(y_cal), "calibration_bins": calibration_bins,
        "test_bins": test_bins, "calibration_bins_ok": check_bins(calibration_bins),
        "test_bins_ok": check_bins(test_bins), "checks": checks, "f1_point": f1_point,
        "recall_point": recall_point, "f1_test": f1_test, "recall_test": recall_test,
    }
    write_summary(context)
    run_id = log_to_mlflow(context)

    print("\nReliability (calibration split):")
    print("\n".join(bin_table_lines(calibration_bins)))
    print("\nReliability (test split):")
    print("\n".join(bin_table_lines(test_bins)))
    print(f"\nCalibrated probabilities in [0,1]: {checks['in_range']} | monotonic: {checks['monotonic']}")
    print(f"F1-optimal: {f1_point}\n  on test: {f1_test}")
    print(f"High-recall: {recall_point}\n  on test: {recall_test}")
    print(f"MLflow experiment '{EXPERIMENT_NAME}', run ID {run_id}")

    passed = context["calibration_bins_ok"] and checks["in_range"] and checks["monotonic"]
    print(f"\nAcceptance: bins within 10pp {'PASS' if context['calibration_bins_ok'] else 'FAIL'}; "
          f"[0,1] {'PASS' if checks['in_range'] else 'FAIL'}; monotonic {'PASS' if checks['monotonic'] else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    start = time.perf_counter()
    exit_code = main()
    print(f"Total {time.perf_counter() - start:.0f}s")
    sys.exit(exit_code)
