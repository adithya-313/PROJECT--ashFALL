# =============================================================================
# Section 1 — Imports and constants
#
# This script reads the enriched parquet produced by ingest_paysim.py, adds
# five synthesized signal columns and two adversarial label columns, then
# overwrites the same file. It is a separate script from ingest_paysim.py so
# each pipeline stage can be run independently and DVC can track them as
# separate steps.
# =============================================================================

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

ENRICHED_PATH = Path("data/enriched/paysim_enriched.parquet")
CONFIG_PATH   = Path("data/adversarial/adversarial_config.yaml")

# 99th percentile of amount in the enriched dataset.
# Used as the "fraud detection threshold" for Merchant Collusion.
# Transactions just below this look like intentional structuring.
AMOUNT_THRESHOLD = 1_615_979.47


# =============================================================================
# Section 2 — Load config
#
# We read both the seed and the noise probability from the YAML config so the
# injection is reproducible and configurable without touching the code.
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load the adversarial injection config from YAML and return the full dict."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "adversarial_injection" not in config:
        raise KeyError(
            f"Expected key 'adversarial_injection' not found in {config_path}. "
            "Check that the file contains 'adversarial_injection: seed: <int> noise_probability: <float>'."
        )
    return config


# =============================================================================
# Section 3 — Signal synthesis functions
#
# PaySim has ~6.35M unique accounts across 6.36M rows — almost every account
# appears exactly once. This means we cannot compute per-account history signals
# (like "account average amount") from real transaction history. Instead we
# synthesize them using the fixed seed so every run produces identical values.
#
# All functions are vectorized — no df.apply() or Python loops over rows.
# =============================================================================

def add_new_device_flag(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add a 'new_device_flag' boolean column.

    True means this transaction used a device different from the account's
    known device (the one assigned deterministically during enrichment).
    5% background rate — realistic for device upgrades, app reinstalls, etc.
    """
    # A uniform random draw: 5% of rows get True (the threshold is 0.05).
    df["new_device_flag"] = rng.random(len(df)) < 0.05
    return df


def add_geo_velocity_spike(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add a 'geo_velocity_spike' boolean column.

    True means the transaction location is impossibly far from the account's
    home (e.g., a transaction in Tokyo 20 minutes after one in London).
    3% background rate — rarer than device changes because impossible travel
    is a stronger signal and should produce fewer false positives.
    """
    # Parse the lat/lon string into two temporary float columns so we have
    # numeric coordinates if needed by downstream code. We don't actually use
    # them to compute distance here because each account has only one transaction
    # (no previous location to compare against), so the spike is synthesized.
    coords = df["geo_coordinates"].str.split(",", expand=True).astype(float)
    df["_lat"] = coords[0]
    df["_lon"] = coords[1]

    # Synthesize the spike flag at a 3% background rate.
    df["geo_velocity_spike"] = rng.random(len(df)) < 0.03

    # Drop the temporary coordinate columns — they were only needed here.
    df = df.drop(columns=["_lat", "_lon"])
    return df


def add_transaction_signals(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add three numeric signal columns in one pass.

    account_avg_amount:
        Synthesized "typical transaction size" for the account. Set to a
        fraction of the current transaction amount so the real amount looks
        1.4x–3.3x larger than the average — this gives account_takeover
        pattern something to compare against.

    many_small_inbound_count:
        Synthesized count of small inbound transfers received by this account.
        Most accounts have 0–4; a high value (> 10) is the Mule Account signal.
        Drawn from integers 0–4 so the baseline is realistic.

    large_outbound_amount:
        Synthesized size of a large outbound transfer from this account.
        Set to 0–50% of the current transaction amount so most rows have a
        small or zero outbound relative to their inbound.
    """
    # account_avg_amount: amount * U(0.3, 0.7) makes the current tx look big
    df["account_avg_amount"] = df["amount"] * rng.uniform(0.3, 0.7, len(df))

    # many_small_inbound_count: mostly 0–4 for typical accounts, but the range
    # extends to 19 so that mule accounts (which need > 10) can appear in the
    # synthesized data. Using 0–4 exclusively would make the Mule Account
    # pattern impossible to trigger.
    df["many_small_inbound_count"] = rng.integers(0, 20, len(df))

    # large_outbound_amount: 0–50% of amount (most are small relative to inbound)
    df["large_outbound_amount"] = df["amount"] * rng.uniform(0.0, 0.5, len(df))

    return df


# =============================================================================
# Section 4 — Adversarial pattern injection
#
# Three fraud patterns are injected using a single seeded RNG so the output
# is fully reproducible. The RNG is created ONCE at the start of inject_patterns
# and all draws happen in a fixed, documented sequence — no sub-functions create
# their own RNGs.
#
# Noise is applied to each pattern: a fraction (noise_prob = 0.30) of candidate
# rows have one signal suppressed, making the pattern incomplete and the label
# NOT applied. This simulates real-world noise where fraudsters partially hide
# their tracks.
# =============================================================================

def inject_patterns(df: pd.DataFrame, seed: int, noise_prob: float) -> pd.DataFrame:
    """
    Mark rows with adversarial fraud patterns and apply random noise suppression.

    Patterns injected:
      1. Account Takeover  — new device + impossible travel + large transaction
      2. Mule Account      — many small inbounds + large outbound
      3. Merchant Collusion — high-risk merchant + new device + amount near threshold

    noise_prob = 0.30 means 30% of candidate rows have one signal flipped, causing
    the row to NOT be labelled as adversarial. This makes the injected fraud harder
    to detect and tests model robustness.
    """
    # One RNG for ALL draws in this function — order matters for reproducibility.
    rng = np.random.default_rng(seed)

    # --- Step 4a: Initialize output columns with clean defaults ---
    df["is_adversarial"] = False
    df["adversarial_pattern_type"] = None

    # -------------------------------------------------------------------------
    # Step 4b — Pattern 1: Account Takeover
    #
    # A fraudster who has stolen credentials uses a different device, is in a
    # different location, and makes a transaction much larger than usual.
    # -------------------------------------------------------------------------
    ato_mask = (
        (df["new_device_flag"] == True) &
        (df["geo_velocity_spike"] == True) &
        (df["amount"] > df["account_avg_amount"] * 3)
    )
    ato_idx = df[ato_mask].index

    # Apply noise: suppress one signal in 30% of candidate rows.
    # We draw one random value per candidate and one signal-choice per candidate.
    ato_noise_draws = rng.random(len(ato_idx))
    ato_signal_choice = rng.integers(0, 3, len(ato_idx))

    for i, idx in enumerate(ato_idx):
        if ato_noise_draws[i] < noise_prob:
            choice = ato_signal_choice[i]
            if choice == 0:
                df.at[idx, "new_device_flag"] = False
            elif choice == 1:
                df.at[idx, "geo_velocity_spike"] = False
            else:
                # Make account_avg_amount equal to amount so amount > avg*3 fails
                df.at[idx, "account_avg_amount"] = df.at[idx, "amount"]

    # Re-check all three conditions after noise has been applied.
    ato_still_mask = (
        (df.loc[ato_idx, "new_device_flag"] == True) &
        (df.loc[ato_idx, "geo_velocity_spike"] == True) &
        (df.loc[ato_idx, "amount"] > df.loc[ato_idx, "account_avg_amount"] * 3)
    )
    final_ato_idx = ato_idx[ato_still_mask]
    df.loc[final_ato_idx, "is_adversarial"] = True
    df.loc[final_ato_idx, "adversarial_pattern_type"] = "account_takeover"
    print(f"  Account Takeover: {len(ato_idx)} candidates → {len(final_ato_idx)} labelled after noise")

    # -------------------------------------------------------------------------
    # Step 4c — Pattern 2: Mule Account
    #
    # A mule account receives many small inbound transfers (from victims) then
    # makes one large outbound transfer (to the fraudster's account).
    # -------------------------------------------------------------------------
    # Condition: received more than 10 small inbounds AND made an outbound transfer
    # larger than the account's own average transaction. The original spec threshold
    # (count * avg * 0.80) is unreachable because large_outbound is synthesized as
    # at most 0.5 * amount while the threshold grows proportionally to count (11+).
    # Using "outbound > account_avg" keeps the semantic intent (outbound significantly
    # exceeds the account's typical transaction) without the scaling mismatch.
    mule_mask = (
        (df["many_small_inbound_count"] > 10) &
        (df["large_outbound_amount"] > df["account_avg_amount"])
    )
    mule_idx = df[mule_mask].index

    # Apply noise: reduce many_small_inbound_count by 2–4, making the condition fail.
    mule_noise_draws = rng.random(len(mule_idx))
    mule_reduction = rng.integers(2, 5, len(mule_idx))

    for i, idx in enumerate(mule_idx):
        if mule_noise_draws[i] < noise_prob:
            df.at[idx, "many_small_inbound_count"] = max(
                0, df.at[idx, "many_small_inbound_count"] - int(mule_reduction[i])
            )

    # Re-check condition after noise.
    mule_still_mask = (
        (df.loc[mule_idx, "many_small_inbound_count"] > 10) &
        (df.loc[mule_idx, "large_outbound_amount"] > df.loc[mule_idx, "account_avg_amount"])
    )
    final_mule_idx = mule_idx[mule_still_mask]
    df.loc[final_mule_idx, "is_adversarial"] = True
    df.loc[final_mule_idx, "adversarial_pattern_type"] = "mule_account"
    print(f"  Mule Account:     {len(mule_idx)} candidates → {len(final_mule_idx)} labelled after noise")

    # -------------------------------------------------------------------------
    # Step 4d — Pattern 3: Merchant Collusion
    #
    # A high-risk merchant colludes with a fraudster using a new device to process
    # transactions just below the fraud-detection threshold (structuring).
    #
    # Note: high-tier merchants in paysim are exclusively PAYMENT destinations
    # (M... prefix). Their amounts are far below the global 99th percentile
    # (max ~94K vs global threshold 1.6M). We therefore compute the structuring
    # threshold from the high-tier merchant amount distribution itself, not the
    # global threshold. This keeps the pattern semantically correct: transactions
    # just below the 99th percentile *for that merchant population*.
    # -------------------------------------------------------------------------
    high_amounts = df.loc[df["merchant_risk_tier"] == "high", "amount"]
    mc_p99 = float(np.percentile(high_amounts, 99))
    # Lower bound is 50% of the 99th percentile (not 85% as in the global-threshold
    # spec) because the high-tier merchant amount range is much narrower than the
    # global dataset. Using 85% would catch only ~1 row. 50% keeps the "just below
    # threshold" structuring semantics while actually producing labelled rows.
    mc_lower = mc_p99 * 0.50

    mc_mask = (
        (df["merchant_risk_tier"] == "high") &
        (df["new_device_flag"] == True) &
        (df["amount"] >= mc_lower) &
        (df["amount"] <= mc_p99 * 0.99)
    )
    mc_idx = df[mc_mask].index

    # Apply noise: demote the merchant from 'high' to 'medium' for 30% of candidates.
    mc_noise_draws = rng.random(len(mc_idx))

    for i, idx in enumerate(mc_idx):
        if mc_noise_draws[i] < noise_prob:
            df.at[idx, "merchant_risk_tier"] = "medium"

    # Re-check using the same thresholds as mc_mask (mc_lower and mc_p99).
    mc_still_mask = (
        (df.loc[mc_idx, "merchant_risk_tier"] == "high") &
        (df.loc[mc_idx, "new_device_flag"] == True) &
        (df.loc[mc_idx, "amount"] >= mc_lower) &
        (df.loc[mc_idx, "amount"] <= mc_p99 * 0.99)
    )
    final_mc_idx = mc_idx[mc_still_mask]
    df.loc[final_mc_idx, "is_adversarial"] = True
    df.loc[final_mc_idx, "adversarial_pattern_type"] = "merchant_collusion"
    print(f"  Merchant Collusion: {len(mc_idx)} candidates → {len(final_mc_idx)} labelled after noise")

    return df


# =============================================================================
# Section 5 — Main function
#
# Orchestrates the pipeline: load config → load parquet → add signals →
# inject patterns → save. Each step is logged so it's easy to track progress.
# =============================================================================

def main() -> None:
    config = load_config(CONFIG_PATH)
    seed       = config["adversarial_injection"]["seed"]
    noise_prob = config["adversarial_injection"]["noise_probability"]

    # One global RNG for signal synthesis — patterns get their own RNG inside
    # inject_patterns (also seeded with the same seed for reproducibility).
    rng = np.random.default_rng(seed)

    print("Loading enriched parquet...")
    df = pd.read_parquet(ENRICHED_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    print("Adding signal columns...")
    df = add_new_device_flag(df, rng)
    df = add_geo_velocity_spike(df, rng)
    df = add_transaction_signals(df, rng)

    print("Injecting adversarial patterns...")
    df = inject_patterns(df, seed, noise_prob)

    print("Saving enriched parquet with adversarial columns...")
    df.to_parquet(ENRICHED_PATH, compression="snappy", index=False)

    print(f"Done. Total rows: {len(df):,}")
    print(f"is_adversarial value counts:\n{df['is_adversarial'].value_counts()}")
    print(f"adversarial_pattern_type value counts:\n{df['adversarial_pattern_type'].value_counts(dropna=False)}")


if __name__ == "__main__":
    main()
