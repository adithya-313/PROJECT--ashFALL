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
    Mark rows with adversarial fraud patterns, then corrupt one signal on noise_prob
    fraction of already-labelled rows. The fraud label is kept after corruption.

    Patterns injected:
      1. Account Takeover  — new device + impossible travel + large transaction
      2. Mule Account      — many small inbounds + large outbound
      3. Merchant Collusion — high-risk merchant + new device + amount near threshold

    Noise order: find candidates → label ALL → corrupt signal on 30% → keep label.
    This means ~30% of fraud rows will have one signal missing, making the pattern
    harder to detect by a single-feature model and testing model robustness.
    """
    # One RNG for ALL draws in this function — order is fixed for reproducibility.
    # All random draws for all three patterns must come from this one RNG in
    # the documented sequence below. Adding draws or reordering them changes
    # all downstream results.
    rng = np.random.default_rng(seed)

    # --- Step 4a: Initialize output columns with clean defaults ---
    df["is_adversarial"] = False
    df["adversarial_pattern_type"] = None

    # -------------------------------------------------------------------------
    # Step 4b — Pattern 1: Account Takeover
    #
    # A fraudster who has stolen credentials uses a different device, is in a
    # different location, and makes a transaction much larger than usual.
    #
    # Noise: after labelling, ~30% of fraud rows get one signal corrupted.
    # The fraud label is NOT removed — this simulates real-world imperfect fraud
    # where one signal is masked but the fraud still happened. A junior dev note:
    # we label first, corrupt second. If we did it the other way, every labelled
    # fraud row would have all signals intact and a single-feature model would
    # get perfect AUC — that defeats the purpose of noise.
    # -------------------------------------------------------------------------
    ato_mask = (
        (df["new_device_flag"] == True) &
        (df["geo_velocity_spike"] == True) &
        (df["amount"] > df["account_avg_amount"] * 3)
    )
    ato_idx = df[ato_mask].index

    # Step 4b-i: Label ALL candidates as fraud immediately.
    df.loc[ato_idx, "is_adversarial"] = True
    df.loc[ato_idx, "adversarial_pattern_type"] = "account_takeover"

    # Step 4b-ii: Apply noise to already-labelled rows. Draw order is fixed:
    # noise_draws first, then signal_choice — do not swap.
    ato_noise_draws = rng.random(len(ato_idx))
    ato_signal_choice = rng.integers(0, 3, len(ato_idx))
    ato_noisy_count = 0

    for i, idx in enumerate(ato_idx):
        if ato_noise_draws[i] < noise_prob:
            ato_noisy_count += 1
            choice = ato_signal_choice[i]
            if choice == 0:
                df.at[idx, "new_device_flag"] = False
            elif choice == 1:
                df.at[idx, "geo_velocity_spike"] = False
            else:
                # Set account_avg_amount to amount/2 so amount > avg*3 no longer holds
                # (amount > (amount/2)*3 = amount*1.5 → True, so use amount itself)
                df.at[idx, "account_avg_amount"] = df.at[idx, "amount"]

    print(f"  Account Takeover: {len(ato_idx)} candidates labelled, "
          f"{ato_noisy_count} received signal corruption (~{noise_prob*100:.0f}% target), "
          f"{len(ato_idx)} fraud rows remain")

    # -------------------------------------------------------------------------
    # Step 4c — Pattern 2: Mule Account
    #
    # A mule account receives many small inbound transfers (from victims) then
    # makes one large outbound transfer (to the fraudster's account).
    #
    # Noise: reduce many_small_inbound_count by 2–4 on ~30% of fraud rows.
    # The fraud label stays. This creates labelled mule rows with count <= 10,
    # which a single-feature model on many_small_inbound_count cannot perfectly
    # classify — exactly what we want for a non-trivial AUC gate.
    # -------------------------------------------------------------------------
    mule_mask = (
        (df["many_small_inbound_count"] > 10) &
        (df["large_outbound_amount"] > df["account_avg_amount"])
    )
    mule_idx = df[mule_mask].index

    # Step 4c-i: Label ALL candidates as fraud immediately.
    df.loc[mule_idx, "is_adversarial"] = True
    df.loc[mule_idx, "adversarial_pattern_type"] = "mule_account"

    # Step 4c-ii: Apply noise. Draw order: noise_draws first, reduction second.
    mule_noise_draws = rng.random(len(mule_idx))
    mule_reduction = rng.integers(2, 5, len(mule_idx))
    mule_noisy_count = 0

    for i, idx in enumerate(mule_idx):
        if mule_noise_draws[i] < noise_prob:
            mule_noisy_count += 1
            df.at[idx, "many_small_inbound_count"] = max(
                0, df.at[idx, "many_small_inbound_count"] - int(mule_reduction[i])
            )

    print(f"  Mule Account: {len(mule_idx)} candidates labelled, "
          f"{mule_noisy_count} received signal corruption (~{noise_prob*100:.0f}% target), "
          f"{len(mule_idx)} fraud rows remain")

    # -------------------------------------------------------------------------
    # Step 4d — Pattern 3: Merchant Collusion
    #
    # A high-risk merchant colludes with a fraudster using a new device to process
    # transactions just below the fraud-detection threshold (structuring).
    #
    # Note: high-tier merchants in paysim are exclusively PAYMENT destinations
    # (M... prefix). Their amounts are far below the global 99th percentile
    # (max ~94K vs global threshold 1.6M). We compute the structuring threshold
    # from the high-tier merchant amount distribution itself.
    #
    # Noise: demote merchant_risk_tier to 'medium' on ~30% of fraud rows.
    # The fraud label stays. This creates labelled MC rows where the merchant
    # tier signal is missing — making the pattern harder to learn from tier alone.
    # -------------------------------------------------------------------------
    high_amounts = df.loc[df["merchant_risk_tier"] == "high", "amount"]
    mc_p99 = float(np.percentile(high_amounts, 99))
    # Lower bound 50% of p99 — 85% would catch only ~1 row given how narrow
    # the high-tier merchant amount range is relative to the global dataset.
    mc_lower = mc_p99 * 0.50

    mc_mask = (
        (df["merchant_risk_tier"] == "high") &
        (df["new_device_flag"] == True) &
        (df["amount"] >= mc_lower) &
        (df["amount"] <= mc_p99 * 0.99)
    )
    mc_idx = df[mc_mask].index

    # Step 4d-i: Label ALL candidates as fraud immediately.
    df.loc[mc_idx, "is_adversarial"] = True
    df.loc[mc_idx, "adversarial_pattern_type"] = "merchant_collusion"

    # Step 4d-ii: Apply noise — demote merchant tier for ~30% of fraud rows.
    mc_noise_draws = rng.random(len(mc_idx))
    mc_noisy_count = 0

    for i, idx in enumerate(mc_idx):
        if mc_noise_draws[i] < noise_prob:
            mc_noisy_count += 1
            df.at[idx, "merchant_risk_tier"] = "medium"

    print(f"  Merchant Collusion: {len(mc_idx)} candidates labelled, "
          f"{mc_noisy_count} received signal corruption (~{noise_prob*100:.0f}% target), "
          f"{len(mc_idx)} fraud rows remain")

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
