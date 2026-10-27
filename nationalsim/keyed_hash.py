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

ALLOWED_IDENTIFIER_TYPES: frozenset[str] = frozenset({"account", "device", "phone"})
HEX_64_PATTERN: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")


def validate_identifier_hash(hash_str: str) -> None:
    """Validate that an identifier hash string consists of exactly 64 lowercase hex characters.

    What it is for:
        Enforces cryptographic hygiene on incoming and stored fraud hashes under Ground Rule 32.
        Rejects raw values or malformed digests before they touch business logic.

    Inputs:
        hash_str: The hash string to validate (e.g. 'a' * 64).

    What it returns:
        None.

    How it works:
        1. Check that hash_str is a string.
        2. Match against regular expression for 64 lowercase hex characters.
        3. Raise ValueError if validation fails.

    Errors and side effects:
        Raises ValueError if hash_str is not exactly 64 lowercase hex characters.

    Limits:
        Validates string format only; does not verify pre-image existence.
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
    """Compute keyed HMAC-SHA256 digest for an identifier according to national specifications.

    What it is for:
        Hashes sensitive bank accounts, device hardware fingerprints, and telephone numbers
        on the reporting bank's side before submission to national registries (DPIP / I4C).

    Inputs:
        key: Secret HMAC key (string or bytes) held strictly by authorized participants.
        run_id: Simulation or environment run namespace string (e.g. 'run_001').
        identifier_type: One of 'account', 'device', or 'phone'.
        raw_value: Raw account identifier, device fingerprint, or synthetic phone string.
        bank_index: Optional bank index for account namespacing if not embedded in raw_value.

    What it returns:
        64-character lowercase hexadecimal HMAC-SHA256 digest string.

    How it works:
        1. Validates that the HMAC secret key is non-empty.
        2. Validates that identifier_type is in ALLOWED_IDENTIFIER_TYPES ('account', 'device', 'phone').
        3. Formats namespaced raw value: accounts prefix with bank index ('<bank>:<account>'),
           while devices and phone numbers use global canonical values.
        4. Forms message '<run_id>|<identifier_type>|<namespaced_value>' and computes HMAC-SHA256.

    Errors and side effects:
        Raises ValueError if key is empty, identifier_type is unknown, or account namespacing is missing.

    Limits:
        HMAC-SHA256 computation is deterministic for identical inputs and keys.
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
