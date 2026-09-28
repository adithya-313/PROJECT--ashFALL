import sys
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
# AUC THRESHOLD: 0.70. A single feature that scores above this on fraud vs
# non-fraud means it alone is almost sufficient to detect the pattern — which
# defeats the purpose of multi-signal fraud patterns. Noise injection exists
# to prevent this. If AUC > 0.70, noise probability must be increased before
# proceeding.
#
# MERCHANT COLLUSION is explicitly excluded from logistic regression testing.
# It has only 11 rows in the PaySim corpus. A 70/30 split yields ~3 fraud rows
# in the test set — too few for a statistically meaningful AUC estimate. This
# is documented below in Section 3 and is intentional, not a gap.
# =============================================================================

PARQUET_PATH = "data/enriched/paysim_enriched.parquet"
AUC_THRESHOLD = 0.70
NON_ADV_SAMPLE_SIZE = 50_000

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

def run_auc_test(df: pd.DataFrame, pattern: str, feature: str) -> float:
    """
    Test whether a single feature can trivially separate one fraud pattern from
    non-adversarial rows. Returns the AUC score.

    A passing score (≤ 0.70) means the feature alone is NOT sufficient to
    detect the pattern — noise injection has made the pattern hard to learn
    from a single signal.
    """
    fraud_rows = df[df["adversarial_pattern_type"] == pattern].copy()
    non_adv_rows = df[df["is_adversarial"] == False].sample(
        n=min(NON_ADV_SAMPLE_SIZE, len(df)), random_state=42
    ).copy()

    combined = pd.concat([fraud_rows, non_adv_rows], ignore_index=True)
    combined["label"] = (combined["adversarial_pattern_type"] == pattern).astype(int)

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

    verdict = "PASS" if auc <= AUC_THRESHOLD else "FAIL"
    print(f"  Pattern:  {pattern}")
    print(f"  Feature:  {feature}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Threshold: <= {AUC_THRESHOLD}")
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
    print(f"Threshold: AUC <= {AUC_THRESHOLD}")
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
        status = "PASS" if auc <= AUC_THRESHOLD else "FAIL"
        print(f"  {pattern:<30} AUC={auc:.4f}  {status}")
        if auc > AUC_THRESHOLD:
            any_failed = True
    print(f"  {'merchant_collusion':<30} DEFERRED (11 rows, untestable)")
    print()

    if any_failed:
        failed = [p for p, a in results.items() if a > AUC_THRESHOLD]
        print(f"GATE FAILED: {failed}")
        print("One or more patterns have AUC > 0.70.")
        print("Increase noise_probability in data/adversarial/adversarial_config.yaml")
        print("and re-run scripts/inject_adversarial.py before proceeding.")
        sys.exit(1)
    else:
        print("ALL AUC CHECKS PASSED")


if __name__ == "__main__":
    main()
