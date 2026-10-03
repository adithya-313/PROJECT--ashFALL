# File: scripts/fraud_baseline.py
# Purpose: Trains five baseline fraud models on isFraud (default-ish settings, no tuning) and compares them.
import time
from pathlib import Path

import catboost
import lightgbm
import matplotlib

matplotlib.use("Agg")  # draw plots to files only, so no window is needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test.parquet"
REPORT_DIR = REPO_ROOT / "reports" / "model"

RANDOM_SEED = 42

# The five numeric columns used directly as features.
NUMERIC_FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

# Fixed list of transaction types. Using a fixed list guarantees train and test get identical one-hot columns in
# identical order, even if one split happened to miss a value.
TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Any model scoring at or below this ROC-AUC is suspicious (0.5 is a coin flip) and triggers a warning.
MIN_SANE_AUC = 0.55


def load_and_prepare(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load a split, drop merchant_collusion rows, and return the feature matrix, target and column names."""
    df = pd.read_parquet(path)

    # merchant_collusion rows belong to a different data stream (and none of them are fraud), so they are removed.
    df = df[df["adversarial_pattern_type"] != "merchant_collusion"]

    # One-hot encode `type`. Making it a Categorical with a fixed category list means get_dummies always creates
    # all five columns in the same order, so the model sees the same layout at train and test time.
    type_column = pd.Categorical(df["type"], categories=TYPE_CATEGORIES)
    type_dummies = pd.get_dummies(type_column, prefix="type").astype(int)
    type_dummies.index = df.index

    # Final feature matrix: the 5 numeric columns plus the 5 one-hot columns.
    features = pd.concat([df[NUMERIC_FEATURES], type_dummies], axis=1)
    return features, df["isFraud"], list(features.columns)


def get_models(scale_pos_weight: float) -> dict:
    """Return the five untrained models, configured for the class imbalance and the right device."""
    return {
        # Logistic regression and random forest are CPU only. class_weight="balanced" is their way of handling
        # the rarity of fraud.
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        ),
        # XGBoost runs on the GPU. scale_pos_weight (non-fraud / fraud) makes each fraud row count that much more.
        "xgboost": xgboost.XGBClassifier(
            tree_method="hist",
            device="cuda",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            n_estimators=300,
            eval_metric="aucpr",
        ),
        # LightGBM runs on the CPU because the installed wheel was built without GPU support.
        "lightgbm": lightgbm.LGBMClassifier(
            device_type="cpu", scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, n_estimators=300
        ),
        "catboost": catboost.CatBoostClassifier(
            task_type="GPU",
            devices="0",
            random_state=RANDOM_SEED,
            iterations=300,
            verbose=0,
            scale_pos_weight=scale_pos_weight,
        ),
    }


def train_one(name: str, model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit one model (scaling features first for logistic regression) and return it, its scaler and fit seconds."""
    scaler = None
    X_fit = X_train

    # Logistic regression needs scaled inputs: balances reach tens of millions, which stops it converging.
    # Tree models split on thresholds, so scaling does not matter to them and they train on the raw values.
    if name == "logreg":
        scaler = StandardScaler().fit(X_train)
        X_fit = scaler.transform(X_train)

    start = time.perf_counter()
    model.fit(X_fit, y_train)
    return model, scaler, time.perf_counter() - start


def predict_fraud_probability(model, scaler, X: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability that each row is fraud (applying the saved scaler if there is one)."""
    X_in = scaler.transform(X) if scaler is not None else X
    return model.predict_proba(X_in)[:, 1]


def evaluate_one(name: str, model, scaler, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, tuple, tuple]:
    """Score one model on the test split: ROC-AUC, PR-AUC and the points of both curves."""
    probabilities = predict_fraud_probability(model, scaler, X_test)
    auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    return {"name": name, "auc": auc, "pr_auc": pr_auc}, (fpr, tpr), (recall, precision)


def make_plots(results: list[dict], roc_data: dict, pr_data: dict) -> None:
    """Save the ROC overlay, PR overlay and training-time bar chart to reports/model/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    by_name = {r["name"]: r for r in results}

    # ROC curves: closer to the top-left corner is better; the dashed diagonal is a random guess.
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={by_name[name]['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Fraud baselines: ROC curves (test split)")
    plt.legend()
    plt.savefig(REPORT_DIR / "fraud_roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close()

    # PR curves: more useful than ROC when fraud is this rare, because they ignore the huge pile of true negatives.
    plt.figure(figsize=(7, 6))
    for name, (recall, precision) in pr_data.items():
        plt.plot(recall, precision, label=f"{name} (PR-AUC={by_name[name]['pr_auc']:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fraud baselines: precision-recall curves (test split)")
    plt.legend()
    plt.savefig(REPORT_DIR / "fraud_pr_curves.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Training time per model, to show the cost side of the comparison.
    plt.figure(figsize=(7, 5))
    names = [r["name"] for r in results]
    plt.bar(names, [r["fit_seconds"] for r in results])
    plt.xlabel("Model")
    plt.ylabel("Fit time (seconds)")
    plt.title("Fraud baselines: training time")
    plt.savefig(REPORT_DIR / "fraud_train_time.png", dpi=120, bbox_inches="tight")
    plt.close()


def write_report(results: list[dict]) -> None:
    """Write the comparison table, sorted by PR-AUC (best first), to reports/model/fraud_baseline_results.md."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["| Model | ROC-AUC | PR-AUC | Train time (s) |", "|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r["pr_auc"], reverse=True):
        lines.append(f"| {r['name']} | {r['auc']:.4f} | {r['pr_auc']:.4f} | {r['fit_seconds']:.1f} |")
    note = "Baseline, default-ish params, no tuning yet. Winner selection happens after hyperparameter tuning."
    (REPORT_DIR / "fraud_baseline_results.md").write_text("\n".join(lines) + "\n\n" + note + "\n")


def main() -> None:
    """Load data, train and evaluate all five models, then save plots and the report."""
    print("Loading train and test splits (calibration is not touched)...")
    X_train, y_train, feature_columns = load_and_prepare(TRAIN_PATH)
    X_test, y_test, test_columns = load_and_prepare(TEST_PATH)
    assert feature_columns == test_columns, "Train and test feature columns differ"
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")

    # neg / pos ratio on the train split only.
    positives = int(y_train.sum())
    scale_pos_weight = (len(y_train) - positives) / positives
    print(f"scale_pos_weight = {scale_pos_weight:.4f}")

    results, roc_data, pr_data = [], {}, {}
    for name, model in get_models(scale_pos_weight).items():
        print(f"Training {name}...")
        model, scaler, fit_seconds = train_one(name, model, X_train, y_train)
        print(f"  fit time {fit_seconds:.1f}s; evaluating...")
        metrics, roc_points, pr_points = evaluate_one(name, model, scaler, X_test, y_test)
        metrics["fit_seconds"] = fit_seconds
        results.append(metrics)
        roc_data[name], pr_data[name] = roc_points, pr_points

        # Sanity check: a model this close to a coin flip means something is wrong. Warn, but keep going.
        if metrics["auc"] <= MIN_SANE_AUC:
            print(f"  WARNING: {name} ROC-AUC {metrics['auc']:.4f} is <= {MIN_SANE_AUC}")

    print("Saving plots and report...")
    make_plots(results, roc_data, pr_data)
    write_report(results)

    print()
    print((REPORT_DIR / "fraud_baseline_results.md").read_text())


if __name__ == "__main__":
    main()
