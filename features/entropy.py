# File: features/entropy.py
# Purpose: Merchant-entropy and device-ratio session features computed from an account's recent history.
import logging
import math

logger = logging.getLogger(__name__)

DEVICE_ID_PREFIX = "dev_id="


def compute_merchant_entropy(merchant_ids: list[str]) -> float:
    """Return the Shannon entropy (base 2) of the merchant ID distribution, or 0.0 for an empty list."""
    # NOTE: merchant_id is used as a proxy for merchant category because the schema has no category field.
    if not merchant_ids:
        return 0.0

    # Step 1: count how many times each unique merchant_id appears.
    counts: dict[str, int] = {}
    for merchant_id in merchant_ids:
        counts[merchant_id] = counts.get(merchant_id, 0) + 1

    # Step 2: turn each count into a probability (count / total).
    total = len(merchant_ids)

    # Step 3: entropy = -sum(p * log2(p)); every p here is > 0 because it comes from an observed count.
    # abs() avoids returning -0.0 when a single merchant has p = 1.
    return abs(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def parse_device_id(device_fingerprint: str) -> str:
    """Extract the dev_id value from a 'dev_id=...|os=...|browser=...' fingerprint string."""
    # The fingerprint is split into segments on "|"; the one starting with "dev_id=" holds the identifier.
    for segment in device_fingerprint.split("|"):
        if segment.startswith(DEVICE_ID_PREFIX):
            return segment[len(DEVICE_ID_PREFIX):]
    raise ValueError(f"No 'dev_id=' segment found in device_fingerprint: {device_fingerprint!r}")


def compute_device_ratio(device_fingerprints: list[str]) -> float:
    """Return unique device IDs divided by transaction count, or 0.0 for an empty list."""
    if not device_fingerprints:
        return 0.0

    device_ids = []
    for fingerprint in device_fingerprints:
        # A malformed fingerprint should not break the whole feature; skip it and log it.
        try:
            device_ids.append(parse_device_id(fingerprint))
        except ValueError as exc:
            logger.warning("Skipping unparseable device_fingerprint: %s", exc)

    # Spec: divide by the total number of transactions passed in (skipped entries still count in the denominator).
    return len(set(device_ids)) / len(device_fingerprints)
