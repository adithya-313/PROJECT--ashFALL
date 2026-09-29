# =============================================================================
# Section 1 — Imports and configuration
#
# The e-commerce store simulates continuous traffic to the fraud scoring API.
# It generates synthetic transactions, validates them against the e-commerce
# Pandera schema, and posts each valid transaction to POST /score.
#
# Run this script in the background while the FastAPI app is running:
#   python scripts/ecommerce_store.py
# It loops indefinitely until interrupted with Ctrl+C.
# =============================================================================

import logging
import random
import sys
import time

import httpx
import numpy as np
import pandas as pd
from faker import Faker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.schemas.ecommerce_transactions import validate_ecommerce_transaction

# Conservative defaults — 5 transactions every 2 seconds gives the scoring
# endpoint time to process each batch without overwhelming it during development.
# Increase TRANSACTIONS_PER_BATCH or reduce BATCH_INTERVAL_SECONDS for load testing.
SCORE_URL = "http://localhost:8000/score"
TRANSACTIONS_PER_BATCH = 5
BATCH_INTERVAL_SECONDS = 2

# City pool — same cities as ingest_paysim.py so geo_coordinates are realistic
# and consistent across the two pipelines. Jitter is applied at generation time.
CITIES = [
    (19.0760,   72.8777),   # Mumbai
    (28.7041,   77.1025),   # Delhi
    (12.9716,   77.5946),   # Bangalore
    (13.0827,   80.2707),   # Chennai
    (51.5074,   -0.1278),   # London
    (40.7128,  -74.0060),   # New York
    (48.8566,    2.3522),   # Paris
    (35.6762,  139.6503),   # Tokyo
    (-33.8688, 151.2093),   # Sydney
    (25.2048,   55.2708),   # Dubai
]

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
# Section 2 — Transaction generator
#
# Each call produces one synthetic transaction dict with the five API-boundary
# fields. The lognormal distribution for amount is standard for financial
# transactions — it produces mostly small-to-medium values with a long tail of
# large orders, matching real e-commerce spending patterns.
# =============================================================================

def generate_transaction(fake: Faker, rng: np.random.Generator, merchant_pool: list) -> dict:
    """
    Generate one synthetic e-commerce transaction.

    Returns a dict with the five API-boundary fields:
      amount, account_id, merchant_id, device_fingerprint, geo_coordinates.

    The ecom_ prefix on account_id makes these instantly distinguishable from
    PaySim accounts (C... prefix) in logs and dashboards.
    """
    # Amount from lognormal — mostly $20–200, occasional large orders
    amount = round(float(rng.lognormal(mean=4.0, sigma=1.5)), 2)

    # account_id — ecom_ prefix marks these as e-commerce stream transactions
    account_id = f"ecom_{fake.uuid4()[:8]}"

    # merchant_id — drawn from a fixed pool created at startup (repeat merchants)
    merchant_id = rng.choice(merchant_pool)

    # device_fingerprint — same format as PaySim enrichment for consistency
    os_name, browser = random.choice(DEVICES)
    device_fingerprint = f"dev_id={fake.uuid4()[:8]}|os={os_name}|browser={browser}"

    # geo_coordinates — random city with ±0.05 degree jitter
    base_lat, base_lon = random.choice(CITIES)
    lat = base_lat + rng.uniform(-0.05, 0.05)
    lon = base_lon + rng.uniform(-0.05, 0.05)
    geo_coordinates = f"{lat:.4f},{lon:.4f}"

    return {
        "amount": amount,
        "account_id": account_id,
        "merchant_id": merchant_id,
        "device_fingerprint": device_fingerprint,
        "geo_coordinates": geo_coordinates,
    }


# =============================================================================
# Section 3 — Send and log
#
# Each transaction is posted individually. The scoring endpoint is stateless —
# there is no batch endpoint — so we loop over the validated batch.
# =============================================================================

def send_transaction(client: httpx.Client, transaction: dict) -> None:
    """
    POST one transaction to the scoring endpoint and log the result.

    On success (HTTP 200): logs fraud_score and account_id at INFO.
    On HTTP error: logs the status code and response body at WARNING.
    On network error: logs the exception message at WARNING.
    Never raises.
    """
    try:
        response = client.post(SCORE_URL, json=transaction)
        if response.status_code == 200:
            body = response.json()
            logging.info(
                "scored  account_id=%s  fraud_score=%.4f  model=%s",
                body.get("account_id"),
                body.get("fraud_score"),
                body.get("model_version"),
            )
        else:
            logging.warning(
                "score endpoint returned %d for account_id=%s: %s",
                response.status_code,
                transaction["account_id"],
                response.text[:200],
            )
    except httpx.RequestError as exc:
        logging.warning(
            "network error sending account_id=%s: %s",
            transaction["account_id"],
            exc,
        )


# =============================================================================
# Section 4 — Main loop
#
# This function simulates continuous e-commerce traffic. It is designed to run
# in the background while other development work happens — the server logs will
# show a steady stream of scoring requests, making it easy to verify the API is
# healthy and the model is responding to realistic input.
# =============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    fake = Faker()
    Faker.seed(42)
    rng = np.random.default_rng(42)

    # Merchant pool — generated once at startup so the same 20 merchants appear
    # repeatedly across batches, matching real e-commerce patterns.
    merchant_pool = np.array([f"merch_{fake.uuid4()[:8]}" for _ in range(20)])

    client = httpx.Client(timeout=5.0)
    batch_num = 0

    logging.info("E-commerce store started — sending to %s", SCORE_URL)
    logging.info(
        "Batch size: %d  Interval: %ds  Ctrl+C to stop",
        TRANSACTIONS_PER_BATCH,
        BATCH_INTERVAL_SECONDS,
    )

    try:
        while True:
            batch_num += 1
            logging.info("--- batch %d ---", batch_num)

            raw_batch = [
                generate_transaction(fake, rng, merchant_pool)
                for _ in range(TRANSACTIONS_PER_BATCH)
            ]

            df = pd.DataFrame(raw_batch)
            original_len = len(df)
            df = validate_ecommerce_transaction(df)
            dropped = original_len - len(df)

            if dropped:
                logging.warning("schema validation dropped %d row(s) this batch", dropped)
            else:
                logging.info("schema validation passed — all %d rows valid", len(df))

            for _, row in df.iterrows():
                send_transaction(client, row.to_dict())

            time.sleep(BATCH_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logging.info("E-commerce store stopped.")
        client.close()


if __name__ == "__main__":
    main()
