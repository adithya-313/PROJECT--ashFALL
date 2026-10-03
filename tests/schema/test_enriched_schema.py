"""
tests/schema/test_enriched_schema.py

What this file does:
  Runs two complementary checks against the Pandera schema in
  data/schemas/enriched_transactions.py using fully synthetic in-memory
  dataframes — no parquet file on disk, no Docker, no network needed.

Why both checks are necessary:
  The first test (valid data) confirms the schema does not produce false
  positives — it must not reject data that genuinely meets the contract.
  The second test (invalid data) confirms the schema does not produce false
  negatives — it must reject data that violates the contract. A schema that
  only passes valid data but silently accepts bad data is broken in the worst
  possible way: it gives a green light to corrupt training data. The second
  test catches exactly that failure mode.

This file is intentionally standalone (no pytest dependency) so it can be
run directly as `python tests/schema/test_enriched_schema.py` in CI without
the full test suite installed.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data.schemas.enriched_transactions import validate_enriched_parquet


def _make_valid_df() -> pd.DataFrame:
    """Return a 5-row synthetic dataframe that satisfies every schema constraint."""
    return pd.DataFrame({
        # --- Original PaySim columns ---
        "step":              pd.array([1, 2, 3, 4, 5], dtype="int64"),
        "type":              ["PAYMENT", "TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"],
        "amount":            pd.array([100.0, 250.5, 800.0, 1.0, 42.0], dtype="float64"),
        "nameOrig":          ["C1001", "C1002", "C1003", "C1004", "C1005"],
        "oldbalanceOrg":     pd.array([500.0, 1000.0, 900.0, 50.0, 200.0], dtype="float64"),
        "newbalanceOrig":    pd.array([400.0, 749.5, 100.0, 49.0, 158.0], dtype="float64"),
        "nameDest":          ["M2001", "C3001", "C3002", "M2002", "M2003"],
        "oldbalanceDest":    pd.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype="float64"),
        "newbalanceDest":    pd.array([100.0, 250.5, 800.0, 1.0, 42.0], dtype="float64"),
        # Rows 1 and 4 are adversarial (see is_adversarial below), so they must be fraud (adversarial implies fraud).
        "isFraud":           pd.array([0, 1, 1, 0, 1], dtype="int64"),
        "isFlaggedFraud":    pd.array([0, 0, 0, 0, 0], dtype="int64"),
        # --- Enriched columns ---
        "geo_coordinates":   ["12.9716,77.5946", "28.7041,77.1025",
                              "40.7128,-74.0060", "19.0760,72.8777", "51.5074,-0.1278"],
        "device_fingerprint": [
            "dev_id=ab12cd34|os=iOS 17|browser=Safari",
            "dev_id=ef56gh78|os=Android 13|browser=Chrome",
            "dev_id=ij90kl12|os=Windows 11|browser=Edge",
            "dev_id=mn34op56|os=macOS 14|browser=Safari",
            "dev_id=qr78st90|os=Linux|browser=Firefox",
        ],
        "merchant_risk_tier": ["high", "medium", "low", "medium", "high"],
        # --- Signal columns ---
        "new_device_flag":          pd.array([True, False, True, False, False], dtype="bool"),
        "geo_velocity_spike":       pd.array([False, False, True, False, False], dtype="bool"),
        "account_avg_amount":       pd.array([80.0, 200.0, 400.0, 0.5, 30.0], dtype="float64"),
        "many_small_inbound_count": pd.array([2, 15, 4, 0, 7], dtype="int64"),
        "large_outbound_amount":    pd.array([50.0, 100.0, 300.0, 0.0, 20.0], dtype="float64"),
        # --- Label columns ---
        # is_adversarial must have exactly 2 unique values (True and False)
        "is_adversarial":           pd.array([False, True, False, False, True], dtype="bool"),
        # numpy object array bypasses pandas 3.x StringDtype inference entirely.
        # The schema contract requires dtype('object') — numpy guarantees it.
        "adversarial_pattern_type": np.array(
            [None, "account_takeover", None, None, "mule_account"], dtype=object
        ),
    })


def test_valid_data_passes() -> None:
    df = _make_valid_df()
    try:
        validate_enriched_parquet(df)
        print("VALID DATA: schema passed")
    except Exception as exc:
        print(f"VALID DATA: schema FAILED on valid data — this is a bug\n{exc}")
        sys.exit(1)


def test_invalid_data_fails() -> None:
    df = _make_valid_df()
    # Introduce exactly one violation: a third cardinality on is_adversarial.
    # The schema requires exactly 2 unique values (True and False).
    df["is_adversarial"] = df["is_adversarial"].astype(object)
    df.loc[df.index[0], "is_adversarial"] = "maybe"

    try:
        validate_enriched_parquet(df)
        # If we reach here, the schema did NOT catch the violation — that is a bug.
        print("INVALID DATA: schema FAILED to catch violation — this is a bug")
        sys.exit(1)
    except (ValueError, Exception):
        # Any exception from validate_enriched_parquet means the schema caught it.
        print("INVALID DATA: schema correctly rejected bad data")


if __name__ == "__main__":
    test_valid_data_passes()
    test_invalid_data_fails()
    print("ALL SCHEMA GATE CHECKS PASSED")
