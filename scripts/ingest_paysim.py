# =============================================================================
# Section 1 — Imports and constants
#
# We only need the standard library plus pandas/numpy/yaml.
# All paths are defined once here so the rest of the file never has bare strings.
# =============================================================================

import hashlib
import logging
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from data.schemas.enriched_transactions import validate_enriched_parquet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Input and output paths
RAW_PATH = Path("data/raw/paysim.csv")
ENRICHED_PATH = Path("data/enriched/paysim_enriched.parquet")
CONFIG_PATH = Path("data/adversarial/adversarial_config.yaml")

# A list of 20 real cities with their latitude and longitude.
# Each account gets one of these as its "home city" — chosen
# deterministically from the account ID so the same account
# always maps to the same city.
CITIES = [
    ("Mumbai",        19.0760,   72.8777),
    ("Delhi",         28.7041,   77.1025),
    ("Bangalore",     12.9716,   77.5946),
    ("Chennai",       13.0827,   80.2707),
    ("Kolkata",       22.5726,   88.3639),
    ("London",        51.5074,   -0.1278),
    ("New York",      40.7128,  -74.0060),
    ("Paris",         48.8566,    2.3522),
    ("Tokyo",         35.6762,  139.6503),
    ("Sydney",       -33.8688,  151.2093),
    ("Dubai",         25.2048,   55.2708),
    ("Singapore",      1.3521,  103.8198),
    ("Lagos",          6.5244,    3.3792),
    ("São Paulo",    -23.5505,  -46.6333),
    ("Mexico City",   19.4326,  -99.1332),
    ("Toronto",       43.6532,  -79.3832),
    ("Berlin",        52.5200,   13.4050),
    ("Johannesburg", -26.2041,   28.0473),
    ("Cairo",         30.0444,   31.2357),
    ("Bangkok",       13.7563,  100.5018),
]

# Device profiles. Each account gets one device permanently assigned,
# same logic as cities — deterministic from account ID.
DEVICES = [
    ("iOS 17",      "Safari"),
    ("Android 13",  "Chrome"),
    ("Windows 11",  "Edge"),
    ("macOS 14",    "Safari"),
    ("Android 12",  "Firefox"),
    ("iOS 16",      "Chrome"),
    ("Windows 10",  "Chrome"),
    ("Linux",       "Firefox"),
]


# =============================================================================
# Section 2 — Load the seed
#
# We read the random seed from a YAML config file rather than hardcoding it.
# This means we can reproduce any run exactly by using the same config, and
# changing the seed is a deliberate, documented action (it updates the DVC hash).
# =============================================================================

def load_seed(config_path: Path) -> int:
    """Read the random seed from the adversarial config YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Raise a clear error if the expected key structure is missing, rather than
    # letting Python raise a confusing KeyError or NoneType crash later.
    try:
        return int(config["adversarial_injection"]["seed"])
    except (KeyError, TypeError) as e:
        raise KeyError(
            f"Could not find 'adversarial_injection.seed' in {config_path}. "
            f"Make sure the file contains 'adversarial_injection: seed: <int>'. Original error: {e}"
        )


# =============================================================================
# Section 3 — Three enrichment functions
#
# Each function takes the dataframe, adds exactly one new column, and returns
# the dataframe. They are called in sequence in main().
#
# All three use vectorized pandas/numpy operations — no df.apply() or Python
# loops over rows. On 6.3M rows, a Python loop would take minutes; vectorized
# operations take seconds.
# =============================================================================

def add_geo_coordinates(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add a 'geo_coordinates' column to df.

    Each account (nameOrig) gets a fixed home city chosen by hashing its ID.
    We then add a small random jitter (±0.05 degrees ≈ ±5 km) so individual
    transactions from the same account don't all land on the exact same point.
    """
    # Step 1: For each unique account ID, pick a city index by hashing the ID.
    # md5 is fast and gives a uniform hash — we don't need cryptographic strength here.
    unique_accounts = df["nameOrig"].unique()

    city_lats = np.array([c[1] for c in CITIES])
    city_lons = np.array([c[2] for c in CITIES])

    # Build lookup dicts: account_id -> base latitude and base longitude
    base_lat_map = {}
    base_lon_map = {}
    for account_id in unique_accounts:
        city_idx = int(hashlib.md5(account_id.encode()).hexdigest(), 16) % len(CITIES)
        base_lat_map[account_id] = city_lats[city_idx]
        base_lon_map[account_id] = city_lons[city_idx]

    # Step 2: Map the dicts onto every row in the dataframe (vectorized lookup).
    base_lat = df["nameOrig"].map(base_lat_map)
    base_lon = df["nameOrig"].map(base_lon_map)

    # Step 3: Add GPS jitter — a few km of realistic drift per transaction.
    lat_jitter = rng.uniform(-0.05, 0.05, len(df))
    lon_jitter = rng.uniform(-0.05, 0.05, len(df))

    final_lat = (base_lat + lat_jitter).round(4)
    final_lon = (base_lon + lon_jitter).round(4)

    # Step 4: Combine into a single string column "lat,lon".
    df["geo_coordinates"] = final_lat.astype(str) + "," + final_lon.astype(str)
    return df


def add_device_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'device_fingerprint' column to df.

    Each account gets a permanently assigned device profile, chosen
    deterministically by hashing its account ID. The fingerprint string
    encodes a short device ID, OS, and browser so downstream models can
    treat device type as a feature.
    """
    unique_accounts = df["nameOrig"].unique()

    # Build a lookup dict: account_id -> fingerprint string
    fingerprint_map = {}
    for account_id in unique_accounts:
        # Use only the first 8 hex characters of the md5 for the device index,
        # and the full first 8 chars as the short device ID in the string.
        md5_hex = hashlib.md5(account_id.encode()).hexdigest()
        device_idx = int(md5_hex[:8], 16) % len(DEVICES)
        os_name, browser = DEVICES[device_idx]
        short_id = md5_hex[:8]
        fingerprint_map[account_id] = f"dev_id={short_id}|os={os_name}|browser={browser}"

    # Map the dict onto every row (vectorized lookup).
    df["device_fingerprint"] = df["nameOrig"].map(fingerprint_map)
    return df


def add_merchant_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'merchant_risk_tier' column to df.

    Merchants with a high transaction volume OR high average transaction amount
    are 'high' risk (they process lots of money and are more attractive targets
    for fraud). Merchants with very low volume are 'low' risk. Everyone else
    is 'medium'.

    Thresholds: top/bottom 20% by transaction count or mean amount.
    """
    # Step 1: Compute per-merchant stats — count and mean amount.
    # In paysim, nameDest holds two kinds of destinations:
    #   - M... IDs: true merchants (PAYMENT transactions)
    #   - C... IDs: other accounts (TRANSFER / CASH_OUT)
    # Account-to-account transfers appear only once or twice, which makes
    # the count distribution degenerate if we include them in the percentile
    # computation. We compute tiers only on M... merchants, then assign
    # "medium" to all C... destinations (account transfers aren't merchants).
    merchant_df = df[df["nameDest"].str.startswith("M")]
    merchant_stats = merchant_df.groupby("nameDest").agg(
        tx_count=("amount", "count"),
        mean_amount=("amount", "mean"),
    )

    # Step 2: Compute the 20th and 80th percentile thresholds on real merchants.
    count_high_thresh = merchant_stats["tx_count"].quantile(0.80)
    amount_high_thresh = merchant_stats["mean_amount"].quantile(0.80)
    count_low_thresh = merchant_stats["tx_count"].quantile(0.20)

    # Step 3: Assign tiers using vectorized boolean conditions.
    # "high" wins over "low" if both conditions somehow apply.
    is_high = (
        (merchant_stats["tx_count"] >= count_high_thresh) |
        (merchant_stats["mean_amount"] >= amount_high_thresh)
    )
    is_low = merchant_stats["tx_count"] <= count_low_thresh

    # Apply tiers in priority order: high first, then low overrides it.
    # A merchant with very few transactions (bottom 20% by count) is low-risk
    # by volume even if their per-transaction amounts happen to be large —
    # we don't have enough data to call them "high". Low wins over high.
    merchant_stats["merchant_risk_tier"] = "medium"
    merchant_stats.loc[is_high, "merchant_risk_tier"] = "high"
    merchant_stats.loc[is_low, "merchant_risk_tier"] = "low"   # low overrides high

    # Step 4: Merge the tier back onto the main dataframe by merchant ID.
    # Rows whose nameDest is a C... account (not in merchant_stats) get NaN,
    # which we fill with "medium" — account transfers have no merchant risk tier.
    df = df.merge(
        merchant_stats[["merchant_risk_tier"]],
        left_on="nameDest",
        right_index=True,
        how="left",
    )
    df["merchant_risk_tier"] = df["merchant_risk_tier"].fillna("medium")
    return df


# =============================================================================
# Section 4 — Main function
#
# Orchestrates the pipeline: load → enrich → save.
# Each step is logged so it's easy to see where time is spent.
# =============================================================================

def main() -> None:
    # Step 1: Load the reproducibility seed from the config file.
    seed = load_seed(CONFIG_PATH)
    log.info(f"Using seed: {seed}")

    # Step 2: Create a seeded random number generator.
    # numpy's default_rng is preferred over the older np.random module —
    # it's faster, more statistically sound, and fully reproducible with a seed.
    rng = np.random.default_rng(seed)

    # Step 3: Load the raw CSV.
    log.info("Loading paysim.csv...")
    df = pd.read_csv(RAW_PATH)
    log.info(f"Loaded {len(df):,} rows.")

    # Step 4: Add the three new columns in order.
    log.info("Adding geo_coordinates...")
    df = add_geo_coordinates(df, rng)

    log.info("Adding device_fingerprint...")
    df = add_device_fingerprint(df)

    log.info("Adding merchant_risk_tier...")
    df = add_merchant_risk_tier(df)

    # Step 5: Save the enriched dataframe as a snappy-compressed parquet file.
    # Parquet is column-oriented and far smaller and faster to read than CSV
    # for downstream ML training. Snappy gives a good compression/speed tradeoff.
    log.info("Saving enriched parquet...")
    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ENRICHED_PATH, compression="snappy", index=False)

    new_columns = ["geo_coordinates", "device_fingerprint", "merchant_risk_tier"]
    log.info(f"Done. Rows: {len(df):,}, Columns: {new_columns}")

    # Validate the parquet that was actually written to disk — not the in-memory df.
    # We reload from disk to catch any serialization differences between the
    # in-memory DataFrame and the bytes written (e.g. dtype promotion by parquet).
    log.info("Validating enriched parquet against schema...")
    try:
        saved_df = pd.read_parquet(ENRICHED_PATH)
        validate_enriched_parquet(saved_df)
        log.info("Schema validation passed.")
    except ValueError as exc:
        log.error(f"Schema validation FAILED:\n{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
