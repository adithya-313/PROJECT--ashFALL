"""
nationalsim/keyed_hash.py

Deterministic keyed HMAC-SHA256 hashing for national shared systems identifiers (DPIP).
Encapsulates message construction under Ground Rule 32:
    Message = "<run_id>|<identifier_type>|<namespaced raw value>"
Ensures that raw account and device identifiers are never exposed beyond participant boundaries.
"""

import hashlib
import hmac
import re

ALLOWED_IDENTIFIER_TYPES: frozenset[str] = frozenset({"account", "device"})
HEX_64_PATTERN: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")


def validate_identifier_hash(hash_str: str) -> None:
    """
    Validate that an identifier hash string consists of exactly 64 lowercase hex characters.

    Args:
        hash_str: The hash string to validate.

    Raises:
        ValueError: If hash_str is not 64 lowercase hexadecimal characters.
    """
    if not isinstance(hash_str, str) or not HEX_64_PATTERN.match(hash_str):
        raise ValueError(
            f"Identifier hash must be 64 lowercase hex characters, got {hash_str!r}"
        )


def compute_identifier_hash(
    key: str | bytes,
    run_id: str,
    identifier_type: str,
    raw_value: str,
    bank_index: int | str | None = None,
) -> str:
    """
    Compute keyed HMAC-SHA256 digest for an identifier according to DPIP specification.

    Constructs the message format:
        "<run_id>|<identifier_type>|<namespaced raw value>"
    where for account identifiers the raw value is namespaced by bank index ("<bank_index>:<account_id>")
    because account numbers can repeat across banks, and for devices the raw value is the device fingerprint.

    Args:
        key: Secret HMAC key (string or bytes) held strictly by DPIP participants.
        run_id: Simulation or environment run identifier namespace.
        identifier_type: Must be either "account" or "device".
        raw_value: Raw account identifier or device fingerprint.
        bank_index: Optional bank index for account namespacing if not already embedded in raw_value.

    Returns:
        64-character lowercase hexadecimal HMAC-SHA256 digest.

    Raises:
        ValueError: If key is empty, identifier_type is unrecognized, or account namespacing is missing.
    """
    # 1. Validate secret key
    if not key:
        raise ValueError("HMAC key cannot be empty")
    key_bytes: bytes = key.encode("utf-8") if isinstance(key, str) else key
    if len(key_bytes) == 0:
        raise ValueError("HMAC key cannot be empty")

    # 2. Validate identifier type
    if identifier_type not in ALLOWED_IDENTIFIER_TYPES:
        raise ValueError(
            f"Unknown identifier_type {identifier_type!r}; allowed types are {sorted(ALLOWED_IDENTIFIER_TYPES)}"
        )

    # 3. Format namespaced raw value based on identifier type
    if identifier_type == "account":
        if bank_index is not None:
            namespaced_value = f"{bank_index}:{raw_value}"
        elif ":" in raw_value:
            namespaced_value = raw_value
        else:
            raise ValueError(
                "account identifier requires a bank_index or '<bank_index>:<account_id>' format"
            )
    else:
        # Device fingerprints are naturally cross-bank and not namespaced by bank index
        namespaced_value = raw_value

    # 4. Form canonical message and compute HMAC-SHA256 digest
    message = f"{run_id}|{identifier_type}|{namespaced_value}".encode("utf-8")
    digest = hmac.new(key_bytes, message, hashlib.sha256).hexdigest().lower()
    return digest
