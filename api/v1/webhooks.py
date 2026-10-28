# File: api/v1/webhooks.py
# Purpose: Builds and verifies signed webhooks that ashFALL sends to a bank (alerts and decisions pushed to it).
#
# A webhook is signed the same way a bank signs its requests to us (same canonical request, same HMAC), in the
# other direction. Actually DELIVERING webhooks (the HTTP calls, retries, timeouts) is Level 14c, not here.
from collections.abc import Mapping
from typing import Any

import orjson

from api.v1.request_auth import HEADER_NONCE, HEADER_SIGNATURE, HEADER_TIMESTAMP
from api.v1.signing import (
    build_canonical_request,
    get_signing_secret,
    is_fresh,
    sign,
    sign_outgoing,
    signatures_match,
)
from core.event_clock import Clock

WEBHOOK_PATH = "/webhooks/ashfall"
WEBHOOK_METHOD = "POST"


def build_signed_webhook(
    bank_id: str,
    event_type: str,
    payload: dict[str, Any],
    clock: Clock,
    nonce: str,
) -> tuple[bytes, dict[str, str]]:
    """
    WHAT: Builds the body and the signed headers of one webhook for one bank.

    WHY: The bank must be able to prove a webhook really came from ashFALL and was not changed or replayed late.
    The signature covers the method, the path, a timestamp, a nonce and a hash of the body, and the secret is the
    one shared with this bank only.

    Args:
        bank_id: The receiving bank; its signing secret signs the webhook.
        event_type: Short name of what happened, such as "decision.ready".
        payload: The event details; must be JSON-serialisable.
        clock: Transport clock giving the timestamp (the real wall clock in production).
        nonce: A value unique to this webhook (16 to 64 letters, digits, "-" or "_").

    Returns:
        (body_bytes, headers): the exact bytes to send, and the headers to send them with.

    Raises:
        UnknownBankError: If bank_id is not a configured bank.
        SigningNotConfiguredError: If the bank has no signing secret configured.
    """
    body_bytes = orjson.dumps({"event_type": event_type, "payload": payload})
    headers = sign_outgoing(bank_id, WEBHOOK_METHOD, WEBHOOK_PATH, body_bytes, clock, nonce)
    headers["Content-Type"] = "application/json"
    return body_bytes, headers


def verify_signed_webhook(
    bank_id: str,
    headers: Mapping[str, str],
    body_bytes: bytes,
    clock: Clock,
    max_age_seconds: int,
) -> bool:
    """
    WHAT: Checks that a webhook was signed with the given bank's secret, was not changed, and is not too old.

    WHY: This is what the receiving side runs. A webhook signed for another bank, with a tampered body, or with an
    old timestamp must all fail, and the signatures are compared in constant time.

    Args:
        bank_id: The bank whose secret the signature must match.
        headers: The received headers (names are matched without regard to case).
        body_bytes: The received body, byte for byte.
        clock: Transport clock giving the current time.
        max_age_seconds: Largest allowed difference between the timestamp and now, in either direction.

    Returns:
        True only if all three checks pass; False for any missing or malformed header too.

    Raises:
        UnknownBankError: If bank_id is not a configured bank.
        SigningNotConfiguredError: If the bank has no signing secret configured.
    """
    lowered = {name.lower(): value for name, value in headers.items()}
    timestamp_text = lowered.get(HEADER_TIMESTAMP.lower())
    nonce = lowered.get(HEADER_NONCE.lower())
    presented_signature = lowered.get(HEADER_SIGNATURE.lower())
    if timestamp_text is None or nonce is None or presented_signature is None:
        return False
    try:
        timestamp = int(timestamp_text)
    except ValueError:
        return False

    canonical = build_canonical_request(WEBHOOK_METHOD, WEBHOOK_PATH, "", timestamp, nonce, body_bytes)
    expected_signature = sign(get_signing_secret(bank_id), canonical)
    if not signatures_match(expected_signature, presented_signature):
        return False
    return is_fresh(timestamp, clock.now_ist(), max_age_seconds)
