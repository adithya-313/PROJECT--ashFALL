"""
File: data/schemas/ecommerce_transactions.py
Purpose: Pandera schema for the live e-commerce scoring stream.

This schema validates the five API-boundary fields that the e-commerce store
sends to POST /score. It is intentionally narrower than the enriched PaySim
schema — no fraud labels, no signal columns, no adversarial metadata. These
are raw, unlabeled transactions from a live stream.

How this differs from data/schemas/enriched_transactions.py:

  enriched_transactions.py — offline batch gate.
    A failure halts the entire pipeline immediately. The PaySim corpus is a
    one-time artifact produced by a controlled script. Bad data here means a
    bug in the pipeline that must be fixed before training proceeds. Raising an
    exception is correct.

  ecommerce_transactions.py — live stream gate.
    A failure drops the offending rows and logs a warning. The scoring stream
    cannot be paused — the e-commerce store sends transactions continuously and
    a single malformed row must never shut down the service. Bad rows are
    discarded and the stream continues. This matches standard stream-processing
    contracts (e.g. Kafka consumers, Flink jobs) where partial data is
    preferable to a full outage.

Logging: uses Python's built-in logging module, not structlog. This module is
imported directly by scripts/ecommerce_store.py which runs outside the FastAPI
app and does not have the structlog context configured by core/logging.py.
"""

import logging
import pandas as pd
import pandera as pa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

ECOMMERCE_SCHEMA = pa.DataFrameSchema(
    columns={
        "amount": pa.Column(
            float,
            checks=[
                # E-commerce transactions must be positive — zero or negative
                # amounts indicate a data error (refunds use a separate flow).
                # element_wise=True so pandera records the failing row index,
                # which lets validate_ecommerce_transaction drop only bad rows.
                pa.Check(lambda x: x > 0,
                         element_wise=True,
                         error="amount must be greater than zero"),
            ],
            nullable=False,
        ),
        "account_id":        pa.Column(str, nullable=False),
        "merchant_id":       pa.Column(str, nullable=False),
        "device_fingerprint": pa.Column(str, nullable=False),
        "geo_coordinates":   pa.Column(str, nullable=False),
    },
    strict=False,   # extra columns are silently ignored
    coerce=False,   # never cast types — a wrong dtype is always a caller bug
)


# ---------------------------------------------------------------------------
# Validation function
# ---------------------------------------------------------------------------

def validate_ecommerce_transaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a DataFrame against ECOMMERCE_SCHEMA.

    On success: returns the dataframe unchanged.
    On failure: logs a warning with the failure details, drops the invalid
                rows, and returns the cleaned dataframe.

    Never raises an exception — the live scoring stream cannot be halted by
    a single bad row.
    """
    try:
        ECOMMERCE_SCHEMA.validate(df, lazy=True)
        return df
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        bad_indices = set(
            failure_cases["index"].dropna().astype(int).tolist()
        )
        logger.warning(
            "E-commerce schema validation dropped %d row(s). Failures: %s",
            len(bad_indices),
            failure_cases[["column", "check", "failure_case"]].to_string(index=False),
        )
        return df.drop(index=list(bad_indices)).reset_index(drop=True)
    except pa.errors.SchemaError as exc:
        logger.warning("E-commerce schema validation error: %s", exc)
        return df.iloc[0:0].reset_index(drop=True)
