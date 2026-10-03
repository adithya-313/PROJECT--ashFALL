import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# =============================================================================
# Section 1 — Setup
#
# Load the enriched parquet and define the two AUC tests to run.
#
# AUC BAND: 0.55 to 0.70, tested against isFraud. A single feature that scores
# above 0.70 on fraud vs non-fraud means it alone is almost sufficient to
# detect the pattern — which defeats the purpose of multi-signal fraud
# patterns. Noise injection exists to prevent this. If AUC > 0.70, noise
# probability must be increased before proceeding. A score below 0.55 means the
# signal is basically noise and the injected pattern carries no usable signal.
#
# MERCHANT COLLUSION is explicitly excluded from logistic regression testing.
# It has only 11 rows in the PaySim corpus. A 70/30 split yields ~3 fraud rows
# in the test set — too few for a statistically meaningful AUC estimate. This
# is documented below in Section 3 and is intentional, not a gap.
# =============================================================================

PARQUET_PATH = "data/enriched/paysim_enriched.parquet"
AUC_LOWER = 0.55   # below this, the signal is basically noise — too weak
AUC_UPPER = 0.70   # above this, the signal alone predicts fraud too well (overpowering)
NON_ADV_SAMPLE_SIZE = 50_000
RESULTS_PATH = Path("reports/data/level3_auc_results.json")

# Each entry describes one AUC test.
# cast_to_int: logistic regression requires numeric input. Boolean columns
# (dtype bool) must be cast to int (True→1, False→0) before fitting.
TESTS = [
    {
        "pattern": "account_takeover",
        "feature": "geo_velocity_spike",
        "cast_to_int": True,
    },
    {
        "pattern": "mule_account",
        "feature": "many_small_inbound_count",
        "cast_to_int": False,
    },
]


# =============================================================================
# Section 2 — AUC test function
#
# Runs a single-feature logistic regression for one fraud pattern and returns
# the AUC score. Prints a clear pass/fail verdict against the threshold.
# =============================================================================

def judge_auc(auc: float) -> tuple[str, str]:
    """Return ("PASS" or "FAIL", reason) for an AUC score against the 0.55–0.70 band."""
    if auc < AUC_LOWER:
        return "FAIL", "signal too weak (noise)"
    if auc > AUC_UPPER:
        return "FAIL", "signal too strong (overpowering) — consider raising noise_probability"
    return "PASS", "within band"


def write_results(results: dict, features: dict) -> None:
    """Write the gate results to reports/data/level3_auc_results.json."""
    tests = []
    for pattern, auc in results.items():
        status, reason = judge_auc(auc)
        tests.append(
            {"pattern": pattern, "feature": features[pattern], "auc": round(float(auc), 4),
             "status": status, "reason": reason}
        )
    report = {"tests": tests, "band": {"lower": AUC_LOWER, "upper": AUC_UPPER}, "target": "isFraud"}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)


def run_auc_test(df: pd.DataFrame, pattern: str, feature: str) -> float:
    """
    Test whether a single feature can separate one injected pattern's rows (all fraud)
    from a sample of non-adversarial rows, using isFraud as the label. Returns the AUC score.

    A passing score (0.55 to 0.70) means the feature carries some signal but is
    NOT sufficient on its own to detect fraud — noise injection has made the
    pattern hard to learn from a single signal.
    """
    fraud_rows = df[df["adversarial_pattern_type"] == pattern].copy()
    non_adv_rows = df[df["is_adversarial"] == False].sample(
        n=min(NON_ADV_SAMPLE_SIZE, len(df)), random_state=42
    ).copy()

    combined = pd.concat([fraud_rows, non_adv_rows], ignore_index=True)
    # The label is isFraud: we test whether a single signal predicts FRAUD (the corrected target), not pattern
    # membership (the old circular test, which was true by construction).
    combined["label"] = combined["isFraud"].astype(int)

    X = combined[[feature]].copy()
    if X[feature].dtype == bool:
        X[feature] = X[feature].astype(int)

    y = combined["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    verdict, _ = judge_auc(auc)
    print(f"  Pattern:  {pattern}")
    print(f"  Feature:  {feature}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Band:     {AUC_LOWER} <= AUC <= {AUC_UPPER}")
    print(f"  Result:   {verdict}")
    print()

    return auc


# =============================================================================
# Section 3 — Main
#
# Run the AUC test for Account Takeover and Mule Account.
# Document the Merchant Collusion deferral explicitly.
# Print a final summary and exit non-zero if any testable pattern fails.
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("Single-feature AUC quality gate")
    print(f"Target: isFraud   Band: {AUC_LOWER} <= AUC <= {AUC_UPPER}")
    print("=" * 60)
    print()

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} rows from {PARQUET_PATH}")
    print()

    results = {}

    for test in TESTS:
        print(f"--- Running: {test['pattern']} ---")
        auc = run_auc_test(df, test["pattern"], test["feature"])
        results[test["pattern"]] = auc

    # Merchant Collusion — explicitly deferred, not silently skipped.
    print("--- Merchant Collusion: DEFERRED ---")
    print("  Pattern:  merchant_collusion")
    print("  Reason:   Only 11 rows exist in the PaySim corpus for this pattern.")
    print("            A 70/30 stratified split yields ~3 fraud rows in the test")
    print("            set — far below the ~20-row minimum for a statistically")
    print("            meaningful AUC estimate. Any number produced would be noise.")
    print("  Decision: This pattern is architecturally deferred to the e-commerce")
    print("            store pipeline, where merchant-customer collusion occurs")
    print("            naturally and at sufficient volume to be testable. This is")
    print("            documented and intentional, not a data quality gap.")
    print()

    # Final summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    any_failed = False
    for pattern, auc in results.items():
        status, reason = judge_auc(auc)
        print(f"  {pattern:<30} AUC={auc:.4f}  {status}  ({reason})")
        if status == "FAIL":
            any_failed = True
    print(f"  {'merchant_collusion':<30} DEFERRED (11 rows, untestable)")
    print()

    # Results are written before any non-zero exit, so a failing run still leaves its numbers on disk.
    write_results(results, {t["pattern"]: t["feature"] for t in TESTS})
    print(f"Results written to {RESULTS_PATH}")
    print()

    if any_failed:
        failed = [p for p, a in results.items() if judge_auc(a)[0] == "FAIL"]
        print(f"GATE FAILED: {failed}")
        print(f"One or more patterns have AUC outside the {AUC_LOWER}-{AUC_UPPER} band.")
        print("If too strong, increase noise_probability in data/adversarial/adversarial_config.yaml")
        print("and re-run scripts/inject_adversarial.py before proceeding.")
        sys.exit(1)
    else:
        print("ALL AUC CHECKS PASSED")


if __name__ == "__main__":
    main()
