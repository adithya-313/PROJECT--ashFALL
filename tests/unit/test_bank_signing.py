# File: tests/unit/test_bank_signing.py
# Purpose: Proves HMAC-SHA256 canonical signing, parameter tampering sensitivity,
#          wall-clock freshness bounds, and secret configuration handling.
from datetime import datetime, timezone

from pydantic import SecretStr
import pytest

from api.v1.bank_scope import UnknownBankError
from api.v1.signing import (
    SigningNotConfiguredError,
    build_canonical_request,
    get_signing_secret,
    is_fresh,
    sign,
    sign_outgoing,
    signatures_match,
)
from core.config import config
from core.event_clock import IST, ManualClock

SECRET = b"test_secret_32_bytes_long_random_value!"


@pytest.mark.parametrize(
    ("tamper_field", "tampered_args"),
    [
        ("method", ("GET", "/api/v1/decide", "", 1700000000, "nonce1234567890ab", b'{"data": 1}')),
        ("path", ("POST", "/api/v1/other", "", 1700000000, "nonce1234567890ab", b'{"data": 1}')),
        ("query", ("POST", "/api/v1/decide", "foo=bar", 1700000000, "nonce1234567890ab", b'{"data": 1}')),
        ("timestamp", ("POST", "/api/v1/decide", "", 1700000001, "nonce1234567890ab", b'{"data": 1}')),
        ("nonce", ("POST", "/api/v1/decide", "", 1700000000, "nonce_different_01", b'{"data": 1}')),
        ("body", ("POST", "/api/v1/decide", "", 1700000000, "nonce1234567890ab", b'{"data": 2}')),
    ],
)
def test_tampering_any_parameter_changes_signature(tamper_field: str, tampered_args: tuple) -> None:
    """Tampering with any one canonical element (method, path, query, ts, nonce, body) alters the signature."""
    base_args = ("POST", "/api/v1/decide", "", 1700000000, "nonce1234567890ab", b'{"data": 1}')
    canonical_base = build_canonical_request(*base_args)
    sig_base = sign(SECRET, canonical_base)

    canonical_tampered = build_canonical_request(*tampered_args)
    sig_tampered = sign(SECRET, canonical_tampered)

    assert canonical_base != canonical_tampered, f"Canonical string did not change when tampering {tamper_field}"
    assert sig_base != sig_tampered, f"Signature did not change when tampering {tamper_field}"
    assert not signatures_match(sig_base, sig_tampered)


def test_fresh_signature_with_old_event_time_in_body_is_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A request whose JSON body contains an event_time from years ago is fresh if the HTTP header
    timestamp is current. Transport freshness evaluates only the HTTP timestamp against the wall clock.
    """
    now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
    clock = ManualClock(now)
    ts = int(now.timestamp())
    body = b'{"event_time": "2020-01-01T00:00:00+05:30", "payment_id": "pmt_old_001"}'

    monkeypatch.setattr(config, "bank_signing_secret_bank_0", SecretStr("secret_key_bank_0_test_value_32_bytes!"))

    # is_fresh evaluates the transport timestamp against wall clock now
    assert is_fresh(ts, now, max_age_seconds=300) is True

    headers = sign_outgoing("bank_0", "POST", "/api/v1/decide", body, clock, "nonce_fresh_12345678")
    assert headers["X-Ashfall-Timestamp"] == str(ts)

    # Re-verifying the signature using the same transport timestamp succeeds
    canonical = build_canonical_request("POST", "/api/v1/decide", "", ts, "nonce_fresh_12345678", body)
    expected_sig = sign(b"secret_key_bank_0_test_value_32_bytes!", canonical)
    assert signatures_match(expected_sig, headers["X-Ashfall-Signature"])


def test_stale_timestamp_is_refused() -> None:
    """A timestamp older than max_age_seconds (now - 301s) is refused, while the boundary (now - 300s) is accepted."""
    now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    ts_boundary = int(now.timestamp()) - 300
    ts_stale = int(now.timestamp()) - 301

    assert is_fresh(ts_boundary, now, max_age_seconds=300) is True
    assert is_fresh(ts_stale, now, max_age_seconds=300) is False


def test_future_timestamp_beyond_window_is_refused() -> None:
    """A future timestamp beyond the window (now + 301s) is refused, while the boundary (now + 300s) is accepted."""
    now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    ts_boundary = int(now.timestamp()) + 300
    ts_future = int(now.timestamp()) + 301

    assert is_fresh(ts_boundary, now, max_age_seconds=300) is True
    assert is_fresh(ts_future, now, max_age_seconds=300) is False


def test_missing_secret_raises_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """If no secret is configured for a known bank in AppConfig, get_signing_secret raises SigningNotConfiguredError."""
    monkeypatch.setattr(config, "bank_signing_secret_bank_1", None)
    with pytest.raises(SigningNotConfiguredError) as exc_info:
        get_signing_secret("bank_1")
    assert "bank_1" in str(exc_info.value)


def test_unknown_bank_raises() -> None:
    """An unknown bank identifier raises UnknownBankError."""
    with pytest.raises(UnknownBankError) as exc_info:
        get_signing_secret("bank_unknown_999")
    assert "bank_unknown_999" in str(exc_info.value)
