# File: tests/unit/test_bank_webhooks.py
# Purpose: Proves signed webhooks verify when untouched and fail when the body is changed, the secret is wrong, the
#          timestamp is stale, or the webhook was signed for a different bank.
from datetime import datetime, timedelta

import pytest
from pydantic import SecretStr

from api.v1.webhooks import build_signed_webhook, verify_signed_webhook
from core.config import config
from core.event_clock import IST, ManualClock

NOW = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
MAX_AGE_SECONDS = 300
NONCE = "webhook_nonce_0123456789"
PAYLOAD = {"payment_id": "pmt_wh_001", "decision": "unscored"}


@pytest.fixture(autouse=True)
def bank_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gives bank_0 and bank_1 different signing secrets for every test in this file."""
    monkeypatch.setattr(config, "bank_signing_secret_bank_0", SecretStr("webhook_secret_bank_0_32_chars_min!"))
    monkeypatch.setattr(config, "bank_signing_secret_bank_1", SecretStr("webhook_secret_bank_1_32_chars_min!"))


def test_round_trip_verifies() -> None:
    """A webhook built for bank_0 verifies for bank_0 at the same moment and again 299 seconds later."""
    clock = ManualClock(NOW)
    body, headers = build_signed_webhook("bank_0", "decision.ready", PAYLOAD, clock, NONCE)

    assert verify_signed_webhook("bank_0", headers, body, clock, MAX_AGE_SECONDS) is True
    clock.advance(timedelta(seconds=299))
    assert verify_signed_webhook("bank_0", headers, body, clock, MAX_AGE_SECONDS) is True


def test_tampered_body_fails() -> None:
    """Changing one byte of the body after signing makes verification fail."""
    clock = ManualClock(NOW)
    body, headers = build_signed_webhook("bank_0", "decision.ready", PAYLOAD, clock, NONCE)
    tampered = body.replace(b"unscored", b"approved")

    assert tampered != body
    assert verify_signed_webhook("bank_0", headers, tampered, clock, MAX_AGE_SECONDS) is False


def test_wrong_secret_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A webhook signed with one secret does not verify once the verifier holds a different secret for the bank."""
    clock = ManualClock(NOW)
    body, headers = build_signed_webhook("bank_0", "decision.ready", PAYLOAD, clock, NONCE)
    monkeypatch.setattr(config, "bank_signing_secret_bank_0", SecretStr("a_completely_different_secret_value_!"))

    assert verify_signed_webhook("bank_0", headers, body, clock, MAX_AGE_SECONDS) is False


def test_stale_timestamp_fails() -> None:
    """A webhook signed 301 seconds ago fails when the limit is 300 seconds, even though signature and body match."""
    signing_clock = ManualClock(NOW - timedelta(seconds=301))
    body, headers = build_signed_webhook("bank_0", "decision.ready", PAYLOAD, signing_clock, NONCE)

    assert verify_signed_webhook("bank_0", headers, body, ManualClock(NOW), MAX_AGE_SECONDS) is False


def test_webhook_for_bank_0_does_not_verify_as_bank_1() -> None:
    """A webhook signed for bank_0 fails when checked against bank_1's secret."""
    clock = ManualClock(NOW)
    body, headers = build_signed_webhook("bank_0", "decision.ready", PAYLOAD, clock, NONCE)

    assert verify_signed_webhook("bank_0", headers, body, clock, MAX_AGE_SECONDS) is True
    assert verify_signed_webhook("bank_1", headers, body, clock, MAX_AGE_SECONDS) is False


def test_missing_or_malformed_headers_fail() -> None:
    """A webhook without a signature header, or with a non-numeric timestamp, fails instead of raising."""
    clock = ManualClock(NOW)
    body, headers = build_signed_webhook("bank_0", "decision.ready", PAYLOAD, clock, NONCE)
    without_signature = {name: value for name, value in headers.items() if name != "X-Ashfall-Signature"}
    bad_timestamp = {**headers, "X-Ashfall-Timestamp": "not-a-number"}

    assert verify_signed_webhook("bank_0", without_signature, body, clock, MAX_AGE_SECONDS) is False
    assert verify_signed_webhook("bank_0", bad_timestamp, body, clock, MAX_AGE_SECONDS) is False
