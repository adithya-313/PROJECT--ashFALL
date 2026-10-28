# File: api/v1/signing.py
# Purpose: HMAC request signing, canonical request construction, and wall-clock freshness verification.
import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from api.v1.bank_scope import UnknownBankError, load_known_banks
from core.config import config
from core.config_validator import load_yaml
from core.event_clock import Clock

SIGNATURE_SCHEME = "ASHFALL-HMAC-SHA256-v1"
DEFAULT_CONFIG_PATH = Path("config/bank_api.yaml")


class SigningNotConfiguredError(ValueError):
    """Raised when HMAC signing secret has not been configured in application settings for a bank."""


@dataclass(frozen=True)
class BankApiPolicy:
    """Policy bounds and timeouts for commercial bank APIs."""

    signature_max_age_seconds: int
    nonce_retention_seconds: int
    idempotency_lifetime_hours: int
    max_request_body_bytes: int
    fuzz_max_examples_per_endpoint: int = 50


def load_bank_api_policy(config_path: str | Path = DEFAULT_CONFIG_PATH) -> BankApiPolicy:
    """
    WHAT: Loads and validates bank API policy parameters from config/bank_api.yaml.

    WHY: Centralizes policy bounds enforcement (signature freshness, nonce window, body limits)
    and validates consistency rules (e.g. nonce retention >= 2x signature max age) at load time.

    Returns:
        Validated BankApiPolicy instance.

    Raises:
        ValueError: If a required setting is missing, non-positive, or inconsistent.
    """
    data = load_yaml(config_path)
    bank_api_block = data.get("bank_api", {})

    def _extract_int(key: str) -> int:
        entry = bank_api_block.get(key)
        if entry is None:
            entry = data.get(key)
        if entry is None:
            raise ValueError(f"Missing bank API policy setting: {key}")

        if isinstance(entry, dict) and "value" in entry:
            val = entry["value"]
        else:
            val = entry

        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"Policy setting {key} must be an integer, got {val!r}")
        if val <= 0:
            raise ValueError(f"Policy setting {key} must be positive, got {val}")
        return val

    sig_max_age = _extract_int("signature_max_age_seconds")
    nonce_retention = _extract_int("nonce_retention_seconds")
    idem_hours = _extract_int("idempotency_lifetime_hours")
    max_body = _extract_int("max_request_body_bytes")
    fuzz_max = _extract_int("fuzz_max_examples_per_endpoint")

    if nonce_retention < 2 * sig_max_age:
        raise ValueError(
            f"Inconsistent policy: nonce_retention_seconds ({nonce_retention}) "
            f"must be >= 2 * signature_max_age_seconds ({sig_max_age})"
        )

    return BankApiPolicy(
        signature_max_age_seconds=sig_max_age,
        nonce_retention_seconds=nonce_retention,
        idempotency_lifetime_hours=idem_hours,
        max_request_body_bytes=max_body,
        fuzz_max_examples_per_endpoint=fuzz_max,
    )


def build_canonical_request(
    method: str,
    path: str,
    query: str,
    timestamp: int | str,
    nonce: str,
    body_bytes: bytes,
) -> str:
    """
    Constructs the canonical request string for HMAC-SHA256 signature calculation.

    Lines are joined with newline ('\\n') in exact sequence:
      1. Signature scheme (SIGNATURE_SCHEME)
      2. HTTP method in uppercase (method.upper())
      3. Request path (path)
      4. Query string (raw query string, empty string if none)
      5. Timestamp in seconds (string representation of integer Unix seconds)
      6. Nonce (unique per request)
      7. Lowercase hex SHA-256 digest of body_bytes

    Security rationale:
      - Including method, path, and query binds the signature to the specific endpoint and action,
        preventing an attacker from replaying a valid signature against a different API route.
      - Including the SHA-256 hash of body_bytes ensures cryptographic integrity of the payload,
        preventing tampering or in-flight modification of request parameters.
      - Including the timestamp and nonce guarantees temporal freshness and prevents replay attacks.
    """
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    ts_str = str(int(timestamp))
    parts = [
        SIGNATURE_SCHEME,
        method.upper(),
        path,
        query if query else "",
        ts_str,
        nonce,
        body_hash,
    ]
    return "\n".join(parts)


def sign(secret_bytes: bytes, canonical: str) -> str:
    """Computes lowercase hex HMAC-SHA256 of the canonical request using secret bytes."""
    return hmac.new(secret_bytes, canonical.encode("utf-8"), hashlib.sha256).hexdigest().lower()


def signatures_match(expected_hex: str, presented_hex: str) -> bool:
    """Compares two hexadecimal signature strings in constant time to prevent timing side-channel attacks."""
    return hmac.compare_digest(expected_hex.lower(), presented_hex.lower())


def is_fresh(timestamp_seconds: int | float, now: datetime, max_age_seconds: int) -> bool:
    """
    Validates that the transport request timestamp is within the allowable window around wall-clock now.

    Freshness condition:
      abs(now.timestamp() - timestamp_seconds) <= max_age_seconds

    Security and Architecture Note:
      This check evaluates strictly the TRANSPORT time from the HTTP header against the host's wall clock.
      Per Ground Rule 30 and Finding 14-4, the payload's domain event_time is NEVER evaluated here.
      A payment event generated days ago during network partition or batch reconciliation may be submitted
      today with a fresh HTTP transport signature, and must not be rejected by transport authentication.
    """
    now_ts = now.timestamp()
    return abs(now_ts - float(timestamp_seconds)) <= float(max_age_seconds)


def get_signing_secret(bank_id: str) -> bytes:
    """
    Retrieves the HMAC secret bytes configured for the specified bank.

    Args:
        bank_id: Identifier of the bank (e.g. 'bank_0', 'bank_1', 'bank_2').

    Returns:
        Secret bytes.

    Raises:
        UnknownBankError: If bank_id is not in known banks from config/bank_api.yaml.
        SigningNotConfiguredError: If the signing secret is not configured or is empty.
    """
    known_banks = load_known_banks()
    if bank_id not in known_banks:
        raise UnknownBankError(f"Unknown bank: {bank_id!r}")

    secret_field_name = f"bank_signing_secret_{bank_id}"
    secret_obj = getattr(config, secret_field_name, None)
    if secret_obj is None:
        raise SigningNotConfiguredError(f"Signing secret not configured for bank {bank_id!r}")

    val = secret_obj.get_secret_value() if hasattr(secret_obj, "get_secret_value") else str(secret_obj)
    if not val:
        raise SigningNotConfiguredError(f"Signing secret is empty for bank {bank_id!r}")

    return val.encode("utf-8")


def sign_outgoing(
    bank_id: str,
    method: str,
    path: str,
    body_bytes: bytes,
    clock: Clock,
    nonce: str,
) -> dict[str, str]:
    """
    Generates authentication headers for an outgoing HTTP request.

    Used by mock banks, test suites, and downstream webhook dispatches.
    Uses the exact same canonical request format and HMAC-SHA256 scheme in both directions.

    Note:
        The `path` parameter must be the exact percent-encoded path that will be sent
        on the wire (e.g. '/v1/accounts/acc%201/risk', not a decoded form containing literal spaces).
    """
    secret_bytes = get_signing_secret(bank_id)
    now_dt = clock.now_ist()
    ts = int(now_dt.timestamp())
    canonical = build_canonical_request(
        method=method,
        path=path,
        query="",
        timestamp=ts,
        nonce=nonce,
        body_bytes=body_bytes,
    )
    sig = sign(secret_bytes, canonical)
    return {
        "X-Ashfall-Key": bank_id,
        "X-Ashfall-Timestamp": str(ts),
        "X-Ashfall-Nonce": nonce,
        "X-Ashfall-Signature": sig,
    }
