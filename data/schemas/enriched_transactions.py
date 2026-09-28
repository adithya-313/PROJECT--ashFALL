"""
File: data/schemas/enriched_transactions.py
Purpose: Pandera schema that acts as a quality gate for the enriched PaySim parquet.

This schema is the contract between the ingestion pipeline and every downstream
consumer (feature engineering, model training, serving). If any assertion here
fails, the pipeline halts immediately with a clear error — it never silently
passes bad data downstream.

Why each assertion exists:
  - Column dtypes: a dtype mismatch would silently corrupt downstream aggregations
    (e.g. a string "True" where a bool is expected causes silent failures in numpy).
  - nameOrig / step null rate < 1%: these are the primary keys of the dataset.
    A null account ID or step makes a row unroutable in the feature store.
  - amount >= 0: negative amounts are impossible in PaySim's transaction model.
    We use >= 0 (not > 0) because 16 rows in the dataset genuinely have amount = 0.0
    — these are valid zero-value transactions (e.g. DEBIT rows with no charge amount)
    and should not be rejected.
  - is_adversarial cardinality == 2: only True and False are valid. A third value
    (e.g. None, 'maybe', NaN) would silently corrupt the binary fraud label used
    in model training.
  - adversarial_pattern_type allowed values: any value outside the known set means
    a new pattern was injected without updating the schema — a breaking change that
    must be caught immediately.
  - merchant_risk_tier allowed values: the three tiers are a categorical feature.
    An unexpected tier value would be treated as a new category by one-hot encoders,
    silently producing a feature vector of the wrong length.
"""

import sys
import pandas as pd
import pandera as pa


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

ENRICHED_SCHEMA = pa.DataFrameSchema(
    columns={
        # --- Original PaySim columns ---
        "step": pa.Column(
            int,
            checks=[
                # null rate below 1% — step is the primary time reference
                pa.Check(lambda s: s.isnull().sum() / len(s) < 0.01,
                         element_wise=False,
                         error="step null rate >= 1%"),
            ],
            nullable=False,
        ),
        "type": pa.Column(str, nullable=False),
        "amount": pa.Column(
            float,
            checks=[
                # 16 rows in the dataset have amount = 0.0 (valid zero-value
                # transactions such as DEBIT rows with no charge). We use >= 0
                # rather than > 0 to allow these through.
                pa.Check(lambda s: (s >= 0).all(),
                         element_wise=False,
                         error="amount contains negative values"),
            ],
            nullable=False,
        ),
        "nameOrig": pa.Column(
            str,
            checks=[
                # null rate below 1% — nameOrig is the account ID (primary key)
                pa.Check(lambda s: s.isnull().sum() / len(s) < 0.01,
                         element_wise=False,
                         error="nameOrig (account_id) null rate >= 1%"),
            ],
            nullable=False,
        ),
        "oldbalanceOrg":    pa.Column(float, nullable=False),
        "newbalanceOrig":   pa.Column(float, nullable=False),
        "nameDest":         pa.Column(str,   nullable=False),
        "oldbalanceDest":   pa.Column(float, nullable=False),
        "newbalanceDest":   pa.Column(float, nullable=False),
        "isFraud":          pa.Column(int,   nullable=False),
        "isFlaggedFraud":   pa.Column(int,   nullable=False),

        # --- Enriched columns added by ingest_paysim.py ---
        "geo_coordinates":   pa.Column(str,   nullable=False),
        "device_fingerprint": pa.Column(str,  nullable=False),
        "merchant_risk_tier": pa.Column(
            str,
            checks=[
                pa.Check(lambda s: s.isin(["high", "medium", "low"]).all(),
                         element_wise=False,
                         error="merchant_risk_tier contains value outside {high, medium, low}"),
            ],
            nullable=False,
        ),

        # --- Signal columns added by inject_adversarial.py ---
        "new_device_flag":         pa.Column(bool,  nullable=False),
        "geo_velocity_spike":      pa.Column(bool,  nullable=False),
        "account_avg_amount":      pa.Column(float, nullable=False),
        "many_small_inbound_count": pa.Column(int,  nullable=False),
        "large_outbound_amount":   pa.Column(float, nullable=False),

        # --- Adversarial label columns ---
        "is_adversarial": pa.Column(
            bool,
            checks=[
                # Exactly two cardinalities: True and False.
                # A third value (None, NaN, 'maybe') would corrupt the binary label.
                pa.Check(lambda s: s.nunique() == 2,
                         element_wise=False,
                         error="is_adversarial must have exactly 2 unique values (True and False)"),
            ],
            nullable=False,
        ),
        "adversarial_pattern_type": pa.Column(
            object,
            checks=[
                # Only values from the known set (or null) are allowed.
                # A new pattern name not in this set means inject_adversarial.py
                # was changed without updating this schema — a breaking change.
                pa.Check(
                    lambda s: s.dropna().isin(
                        ["account_takeover", "mule_account", "merchant_collusion"]
                    ).all(),
                    element_wise=False,
                    error="adversarial_pattern_type contains value outside "
                          "{account_takeover, mule_account, merchant_collusion, null}",
                ),
            ],
            nullable=True,
        ),
    },
    strict=False,   # extra columns are allowed (future-proofing)
    coerce=False,   # never silently cast types — fail loudly on dtype mismatch
)


# ---------------------------------------------------------------------------
# Validation function
# ---------------------------------------------------------------------------

def validate_enriched_parquet(df: pd.DataFrame) -> None:
    """
    Validate a DataFrame against ENRICHED_SCHEMA.

    Raises:
        ValueError: with a human-readable description of which assertion failed
                    and why. The original Pandera exception is included in the
                    message so the full context is never lost.
    """
    try:
        ENRICHED_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        # Build a readable summary of every failure.
        # Pandera's SchemaErrors has a .failure_cases attribute — a DataFrame
        # listing which column, which check, and which value caused the failure.
        failure_summary = exc.failure_cases[
            ["schema_context", "column", "check", "failure_case"]
        ].to_string(index=False)
        raise ValueError(
            f"Enriched parquet schema validation FAILED.\n"
            f"Assertion failures:\n{failure_summary}\n\n"
            f"Original Pandera error:\n{exc}"
        ) from exc
    except pa.errors.SchemaError as exc:
        raise ValueError(
            f"Enriched parquet schema validation FAILED: {exc.args[0]}\n"
            f"Original Pandera error:\n{exc}"
        ) from exc
