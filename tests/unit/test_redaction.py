"""
tests/unit/test_redaction.py

Unit test suite for privacy redaction in logs and trace attributes (core/redaction.py).
Verifies that simulated account IDs, device fingerprints, and phone numbers are scrubbed,
while ordinary log and telemetry text remains completely intact.
"""

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core.logging import configure_logging, get_logger
from core.redaction import (
    redact_raw_identifiers,
    set_redacted_attribute,
)


def test_each_raw_identifier_pattern_is_redacted() -> None:
    """Proves that account IDs, device fingerprints, and phone numbers are converted to redaction tokens."""
    # 1. Account IDs
    acc_text = "Transaction from acc_victim_001 to acc_mule_99 approved"
    redacted_acc = redact_raw_identifiers(acc_text)
    assert "acc_victim_001" not in redacted_acc
    assert "acc_mule_99" not in redacted_acc
    assert redacted_acc == "Transaction from [redacted:account_id] to [redacted:account_id] approved"

    # 2. Device Fingerprints
    fp_text = "Device registered: fp_android_v3_705a5661 for user"
    redacted_fp = redact_raw_identifiers(fp_text)
    assert "fp_android_v3_705a5661" not in redacted_fp
    assert redacted_fp == "Device registered: [redacted:fingerprint] for user"

    # 3. Synthetic Phone Numbers
    phone_text = "OTP dispatch to sim-0000000042 verified"
    redacted_phone = redact_raw_identifiers(phone_text)
    assert "sim-0000000042" not in redacted_phone
    assert redacted_phone == "OTP dispatch to [redacted:phone_number] verified"


def test_ordinary_text_is_untouched() -> None:
    """Proves that standard logging statements and attributes with no raw identifiers remain unaltered."""
    normal_text = "Order payment 50000 paise confirmed for merchant item SKU-9912 at minute 1440."
    assert redact_raw_identifiers(normal_text) == normal_text


def test_structlog_log_line_containing_account_id_is_emitted_redacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Proves that structlog outputs redact account IDs from emitted JSON log records."""
    configure_logging("INFO")
    logger = get_logger("test_redaction_logger")

    test_account = "acc_compromised_99"
    logger.info("hold directive placed", target_account=test_account, reason="suspected mule acc_mule_77")

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert lines, "Expected log output on stdout"

    found_record: dict[str, Any] | None = None
    for line in lines:
        try:
            data = json.loads(line)
            if data.get("event") == "hold directive placed":
                found_record = data
                break
        except json.JSONDecodeError:
            continue

    assert found_record is not None, f"Could not find log record in: {lines}"
    raw_json_str = json.dumps(found_record)

    # Prove raw account IDs never leak into the emitted JSON log line
    assert "acc_compromised_99" not in raw_json_str
    assert "acc_mule_77" not in raw_json_str
    assert found_record["target_account"] == "[redacted:account_id]"
    assert "[redacted:account_id]" in found_record["reason"]


def test_span_attribute_containing_fingerprint_is_exported_redacted() -> None:
    """Proves OpenTelemetry span attributes containing device fingerprints export in sanitized format."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test_redaction_tracer")

    fp_raw = "fp_android_v3_705a5661"
    with tracer.start_as_current_span("client_request") as span:
        set_redacted_attribute(span, "device.fingerprint", fp_raw)
        set_redacted_attribute(span, "unrelated.meta", "standard_text")

    finished_spans = exporter.get_finished_spans()
    assert len(finished_spans) == 1
    span_attrs = finished_spans[0].attributes

    assert span_attrs is not None
    assert "device.fingerprint" in span_attrs
    # Verify raw token is absent and redacted token is present
    assert span_attrs["device.fingerprint"] == "[redacted:fingerprint]"
    assert span_attrs["device.fingerprint"] != fp_raw
    assert span_attrs["unrelated.meta"] == "standard_text"
