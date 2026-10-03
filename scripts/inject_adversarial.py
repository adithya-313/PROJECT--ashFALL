# =============================================================================
# Section 1 — Imports and constants
#
# This script reads the enriched parquet produced by ingest_paysim.py, adds
# five synthesized signal columns and two adversarial label columns, then
# overwrites the same file. It is a separate script from ingest_paysim.py so
# each pipeline stage can be run independently and DVC can track them as
# separate steps.
#
# IMPORTANT: always run ingest_paysim.py first, then this script. This script
# flips isFraud from 0 to 1 on the rows it injects, so running it twice on the
# same file would treat the first run's injected fraud as "original" fraud and
# inject again on top of it.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.schemas.enriched_transactions import validate_enriched_parquet

ENRICHED_PATH = Path("data/enriched/paysim_enriched.parquet")
CONFIG_PATH = Path("data/adversarial/adversarial_config.yaml")

# The patterns this script injects. merchant_collusion is intentionally absent:
# it moved to the e-commerce stream.
PATTERNS = ["account_takeover", "mule_account"]


# =============================================================================
# Section 2 — Load config
#
# We read the seed, the noise probability and the per-pattern injected fraud
# counts from the YAML config so the injection is reproducible and configurable
# without touching the code.
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load the adversarial injection config from YAML and return the full dict."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "adversarial_injection" not in config:
        raise KeyError(
            f"Expected key 'adversarial_injection' not found in {config_path}. "
            "Check that the file contains 'adversarial_injection: seed, noise_probability, injected_fraud_counts'."
        )
    settings = config["adversarial_injection"]
    for key in ("seed", "noise_probability", "injected_fraud_counts"):
        if key not in settings:
            raise KeyError(f"Expected key '{key}' not found under 'adversarial_injection' in {config_path}.")
    for pattern in PATTERNS:
        if pattern not in settings["injected_fraud_counts"]:
            raise KeyError(f"injected_fraud_counts is missing an entry for '{pattern}' in {config_path}.")
    return config


# =============================================================================
# Section 3 — Background signal synthesis functions
#
# PaySim has ~6.35M unique accounts across 6.36M rows — almost every account
# appears exactly once. This means we cannot compute per-account history signals
# (like "account average amount") from real transaction history. Instead we
# synthesize them using the fixed seed so every run produces identical values.
#
# These functions give EVERY row a "normal" background value. The fraud
# patterns in Section 4 then overwrite these values on the rows they inject.
#
# All functions are vectorized — no df.apply() or Python loops over rows.
# =============================================================================

def add_new_device_flag(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add a 'new_device_flag' boolean column.

    True means this transaction used a device different from the account's
    known device (the one assigned deterministically during enrichment).
    15% background rate — realistic for device upgrades, app reinstalls, etc.
    """
    # Raised from 0.05 to 0.15 — a 5% background rate made new_device_flag
    # trivially distinctive in fraud rows, causing AUC > 0.70 on the quality gate.
    df["new_device_flag"] = rng.random(len(df)) < 0.15
    return df


def add_geo_velocity_spike(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add a 'geo_velocity_spike' boolean column.

    True means the transaction location is impossibly far from the account's
    home (e.g., a transaction in Tokyo 20 minutes after one in London).
    20% background rate (raised from an original 3% to keep the signal from being
    trivially distinctive on its own).
    """
    # Parse the lat/lon string into two temporary float columns so we have
    # numeric coordinates if needed by downstream code. We don't actually use
    # them to compute distance here because each account has only one transaction
    # (no previous location to compare against), so the spike is synthesized.
    coords = df["geo_coordinates"].str.split(",", expand=True).astype(float)
    df["_lat"] = coords[0]
    df["_lon"] = coords[1]

    # Raised from 0.03 to 0.20 — a 3% background rate made geo_velocity_spike
    # trivially distinctive in fraud rows, causing AUC > 0.70 on the quality gate.
    df["geo_velocity_spike"] = rng.random(len(df)) < 0.20

    # Drop the temporary coordinate columns — they were only needed here.
    df = df.drop(columns=["_lat", "_lon"])
    return df


def add_transaction_signals(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add three numeric signal columns in one pass (the background, "normal" level).

    account_avg_amount:
        Synthesized "typical transaction size" for the account. Set to a
        fraction of the current transaction amount so the real amount looks
        1.4x–3.3x larger than the average.

    many_small_inbound_count:
        Synthesized count of small inbound transfers received by this account.
        Background rows are drawn from 0–19.

    large_outbound_amount:
        Synthesized size of a large outbound transfer from this account.
        Set to 0–50% of the current transaction amount so most rows have a
        small or zero outbound relative to their inbound.
    """
    # account_avg_amount: amount * U(0.3, 0.7) — background level.
    df["account_avg_amount"] = df["amount"] * rng.uniform(0.3, 0.7, len(df))

    # many_small_inbound_count: integers 0–19 for background rows.
    df["many_small_inbound_count"] = rng.integers(0, 20, len(df))

    # large_outbound_amount: 0–50% of amount (most are small relative to inbound)
    df["large_outbound_amount"] = df["amount"] * rng.uniform(0.0, 0.5, len(df))

    return df


# =============================================================================
# Section 4 — Controlled adversarial pattern injection
#
# Earlier versions of this script only LABELLED rows that happened to match a
# pattern and never touched isFraud, so the patterns were unrelated to fraud.
# This version injects fraud deliberately: for each pattern it picks a fixed
# number of currently non-fraud rows, makes them fraud (isFraud = 1), labels
# them, and stamps the pattern's signals onto them. Original PaySim fraud rows
# are never changed.
#
# All random draws come from ONE seeded RNG, in this fixed order:
#   1. account_takeover: row selection, signal draws, noise draws
#   2. mule_account:     row selection, signal draws, noise draws
# Reordering or adding draws changes every downstream result.
# =============================================================================

def init_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Start every row as not adversarial; isFraud is deliberately left untouched."""
    df["is_adversarial"] = False
    df["adversarial_pattern_type"] = None
    return df


def select_injection_rows(df: pd.DataFrame, rng: np.random.Generator, count: int, pattern: str) -> pd.Index:
    """Randomly pick `count` rows that are not fraud and not already used by another pattern."""
    # The pool excludes original fraud (so we never relabel real fraud) and rows already taken by an earlier
    # pattern (so a row can never belong to two patterns).
    pool = df.index[((df["isFraud"] == 0) & (~df["is_adversarial"])).to_numpy()]
    if count > len(pool):
        raise ValueError(f"Asked for {count} injected rows for {pattern} but only {len(pool)} are available.")
    return pd.Index(rng.choice(pool.to_numpy(), size=count, replace=False))


def label_as_fraud(df: pd.DataFrame, rows: pd.Index, pattern: str) -> pd.DataFrame:
    """Mark the chosen rows as fraud and as belonging to this pattern."""
    df.loc[rows, "isFraud"] = 1
    df.loc[rows, "is_adversarial"] = True
    df.loc[rows, "adversarial_pattern_type"] = pattern
    return df


def stamp_account_takeover_signals(df: pd.DataFrame, rows: pd.Index, rng: np.random.Generator) -> pd.DataFrame:
    """Give the chosen rows the account-takeover signature: new device, location spike, unusually large amount."""
    df.loc[rows, "new_device_flag"] = True
    df.loc[rows, "geo_velocity_spike"] = True

    # "Unusually large amount" means amount is well above the account's usual amount. We leave amount alone,
    # because it is real PaySim data, and instead set account_avg_amount to amount * U(0.1, 0.3). That makes
    # amount 3.3x–10x the average, so the old condition (amount > 3 * account_avg_amount) is always met.
    amounts = df.loc[rows, "amount"].to_numpy()
    df.loc[rows, "account_avg_amount"] = amounts * rng.uniform(0.1, 0.3, len(rows))
    return df


def stamp_mule_signals(df: pd.DataFrame, rows: pd.Index, rng: np.random.Generator) -> pd.DataFrame:
    """Give the chosen rows the mule signature: many small inbound transfers, then a large outbound."""
    # Mule accounts receive many small transfers, so the inbound count is set above the old threshold of 10
    # (drawn from 11–19).
    df.loc[rows, "many_small_inbound_count"] = rng.integers(11, 20, len(rows))

    # The large outbound is set to 1.5x–3x the account's average amount, so it is clearly above it (the old
    # condition was large_outbound_amount > account_avg_amount). account_avg_amount keeps its background value.
    averages = df.loc[rows, "account_avg_amount"].to_numpy()
    df.loc[rows, "large_outbound_amount"] = averages * rng.uniform(1.5, 3.0, len(rows))
    return df


def apply_noise_account_takeover(df: pd.DataFrame, rows: pd.Index, rng: np.random.Generator, noise_prob: float) -> int:
    """Hide the location spike on about noise_prob of the injected rows; they stay labelled as fraud."""
    # Noise mimics real fraud that does not show the obvious signal: the fraud and its label stay, but the
    # primary signal is removed so no single column gives the pattern away.
    noisy_rows = rows[rng.random(len(rows)) < noise_prob]
    df.loc[noisy_rows, "geo_velocity_spike"] = False
    return len(noisy_rows)


def apply_noise_mule(df: pd.DataFrame, rows: pd.Index, rng: np.random.Generator, noise_prob: float) -> int:
    """Push the inbound count below the threshold on about noise_prob of the injected rows; they stay fraud."""
    # Same idea as above: the count is overwritten with a normal-looking 0–9, but isFraud and the label stay.
    noisy_rows = rows[rng.random(len(rows)) < noise_prob]
    df.loc[noisy_rows, "many_small_inbound_count"] = rng.integers(0, 10, len(noisy_rows))
    return len(noisy_rows)


def inject_patterns(df: pd.DataFrame, seed: int, noise_prob: float, counts: dict) -> pd.DataFrame:
    """Run the controlled injection for every pattern and print how many rows each one received."""
    # One RNG for ALL draws in this function — order is fixed for reproducibility.
    rng = np.random.default_rng(seed)
    df = init_labels(df)
    fraud_before = int(df["isFraud"].sum())

    # account_takeover first, then mule_account (order matters: see the Section 4 comment).
    stampers = {"account_takeover": stamp_account_takeover_signals, "mule_account": stamp_mule_signals}
    noisers = {"account_takeover": apply_noise_account_takeover, "mule_account": apply_noise_mule}
    for pattern in PATTERNS:
        rows = select_injection_rows(df, rng, int(counts[pattern]), pattern)
        df = label_as_fraud(df, rows, pattern)
        df = stampers[pattern](df, rows, rng)
        noisy = noisers[pattern](df, rows, rng, noise_prob)
        print(
            f"  {pattern}: {len(rows)} rows injected as fraud, "
            f"{noisy} had their primary signal hidden (~{noise_prob * 100:.0f}% target)"
        )

    injected_total = sum(int(counts[p]) for p in PATTERNS)
    assert int(df["isFraud"].sum()) == fraud_before + injected_total, "Fraud count did not grow by the injected total"
    return df


def check_adversarial_implies_fraud(df: pd.DataFrame) -> None:
    """Fail loudly if any adversarial row is not marked as fraud (the rule the whole dataset relies on)."""
    broken = int((df["is_adversarial"] & (df["isFraud"] != 1)).sum())
    assert broken == 0, f"{broken} rows have is_adversarial=True but isFraud != 1"


# =============================================================================
# Section 5 — Main function
#
# Orchestrates the pipeline: load config → load parquet → add background signals →
# inject patterns → check the invariant → save. Each step is logged so it's easy
# to track progress.
# =============================================================================

def main() -> None:
    config = load_config(CONFIG_PATH)["adversarial_injection"]
    seed = config["seed"]
    noise_prob = config["noise_probability"]
    counts = config["injected_fraud_counts"]

    # One global RNG for background signal synthesis. Pattern injection creates its own RNG from the same
    # seed inside inject_patterns.
    rng = np.random.default_rng(seed)

    print("Loading enriched parquet...")
    df = pd.read_parquet(ENRICHED_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    print("Adding background signal columns...")
    df = add_new_device_flag(df, rng)
    df = add_geo_velocity_spike(df, rng)
    df = add_transaction_signals(df, rng)

    print("Injecting adversarial patterns...")
    df = inject_patterns(df, seed, noise_prob, counts)

    # The key invariant: every adversarial row must be fraud. Checked before saving so a bad result never
    # overwrites the file on disk.
    check_adversarial_implies_fraud(df)

    print("Saving enriched parquet with adversarial columns...")
    df.to_parquet(ENRICHED_PATH, compression="snappy", index=False)

    print(f"Done. Total rows: {len(df):,}")
    print(f"isFraud=1 rows: {int(df['isFraud'].sum()):,} ({df['isFraud'].mean() * 100:.4f}%)")
    print(f"is_adversarial value counts:\n{df['is_adversarial'].value_counts()}")
    print(f"adversarial_pattern_type value counts:\n{df['adversarial_pattern_type'].value_counts(dropna=False)}")

    # Validate the final 21-column parquet from disk — catches serialization
    # differences and confirms the full schema contract before downstream use.
    print("Validating final parquet against schema...")
    try:
        saved_df = pd.read_parquet(ENRICHED_PATH)
        validate_enriched_parquet(saved_df)
        print("Schema validation passed.")
    except ValueError as exc:
        print(f"Schema validation FAILED:\n{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
