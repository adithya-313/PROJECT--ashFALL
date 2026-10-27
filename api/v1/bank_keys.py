"""
File: api/v1/bank_keys.py
Purpose: Issues, verifies, and revokes commercial bank API keys with cryptographic hash storage.
"""

import hashlib
import hmac
import secrets
from pathlib import Path

import asyncpg

from core.config_validator import load_yaml

DEFAULT_CONFIG_PATH: str = "config/bank_api.yaml"
DEFAULT_SECRET_BYTES: int = 32


def _get_secret_bytes(config_path: str | Path = DEFAULT_CONFIG_PATH) -> int:
    """Reads configured secret byte length from bank API config."""
    try:
        data = load_yaml(config_path)
        val = data.get("bank_api", {}).get("api_key_secret_bytes", DEFAULT_SECRET_BYTES)
        return int(val)
    except Exception:
        return DEFAULT_SECRET_BYTES


async def issue_api_key(
    conn: asyncpg.Connection,
    bank_id: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[str, str]:
    """
    WHAT: Generates a high-entropy API key for a commercial bank and persists its hash.

    WHY: Authenticates commercial banks on UPI endpoints. The plaintext key is returned exactly once
    to the caller upon issuance and is never stored in the database or written to logs.

    KEY ROTATION PROCEDURE:
    To rotate an active bank API key without service downtime:
      1. Issue a new API key for the bank using issue_api_key. Both the old and new keys will be active
         simultaneously because verification resolves any non-revoked key matching key_id.
      2. Deploy the newly issued key to the commercial bank's client configuration.
      3. Verify the bank successfully sends requests using the new key.
      4. Revoke the old key using revoke_api_key(conn, old_key_id).

    WHY SHA-256 (NOT BCRYPT):
    Human passwords have low entropy, requiring slow, memory-hard algorithms (bcrypt/Argon2) with salt
    to defend against dictionary and rainbow-table attacks. Machine-generated API keys generated via
    cryptographically secure random bytes (secrets.token_urlsafe(32)) possess 256 bits of entropy.
    Brute-force and precomputation attacks against a 256-bit random space are computationally infeasible.
    A single-round SHA-256 digest allows instant, CPU-efficient verification at high transaction volumes
    without inducing latency bottlenecks in real-time banking auth paths.

    EXECUTION PRIVILEGES:
    This function must be executed on an application owner/superuser connection. The restricted
    'ashfall_bank_api' role deliberately has NO grants on the 'bank_api_keys' table.

    Args:
        conn: Open asyncpg database connection (owner/superuser role).
        bank_id: Commercial bank identifier (e.g. 'bank_0').
        config_path: Optional override path for bank_api.yaml.

    Returns:
        Tuple of (key_id, plaintext_key), formatted as ('<hex16>', 'ak_<hex16>.<secret>').
    """
    secret_bytes = _get_secret_bytes(config_path)
    key_id = secrets.token_hex(8)
    secret = secrets.token_urlsafe(secret_bytes)
    plaintext_key = f"ak_{key_id}.{secret}"

    key_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()

    await conn.execute(
        """
        INSERT INTO bank_api_keys (bank_id, key_id, key_hash)
        VALUES ($1, $2, $3)
        """,
        bank_id,
        key_id,
        key_hash,
    )
    return key_id, plaintext_key


async def resolve_bank_for_key(
    conn: asyncpg.Connection,
    presented_key: str,
) -> str | None:
    """
    WHAT: Resolves a presented API key string to its authenticated bank identifier.

    WHY: Validates incoming request credentials against active stored key hashes using constant-time
    comparison to prevent timing attacks.

    EXECUTION PRIVILEGES:
    Runs on an application owner/superuser connection. The restricted 'ashfall_bank_api' role has
    no access to the 'bank_api_keys' table.

    Args:
        conn: Open asyncpg database connection (owner/superuser role).
        presented_key: Plaintext key string presented in the request header.

    Returns:
        The bank_id string if authentication succeeds and key is not revoked; otherwise None.
    """
    if not isinstance(presented_key, str) or not presented_key.startswith("ak_"):
        return None
    if "." not in presented_key:
        return None

    prefix_and_id, secret = presented_key.split(".", 1)
    key_id = prefix_and_id[3:]  # Strip 'ak_' prefix
    if not key_id or not secret:
        return None

    row = await conn.fetchrow(
        """
        SELECT bank_id, key_hash
        FROM bank_api_keys
        WHERE key_id = $1 AND revoked_at IS NULL
        """,
        key_id,
    )
    if row is None:
        return None

    computed_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(computed_hash, row["key_hash"]):
        return None

    return str(row["bank_id"])


async def revoke_api_key(
    conn: asyncpg.Connection,
    key_id: str,
) -> bool:
    """
    WHAT: Deactivates an active API key by timestamping its revoked_at column.

    WHY: Instantly cuts off access for compromised or rotated credentials.

    Args:
        conn: Open asyncpg database connection (owner/superuser role).
        key_id: Key identifier string to revoke.

    Returns:
        True if an active key was revoked; False if the key did not exist or was already revoked.
    """
    result = await conn.execute(
        """
        UPDATE bank_api_keys
        SET revoked_at = now()
        WHERE key_id = $1 AND revoked_at IS NULL
        """,
        key_id,
    )
    # asyncpg execute returns status tag string, e.g. "UPDATE 1"
    return result == "UPDATE 1"
