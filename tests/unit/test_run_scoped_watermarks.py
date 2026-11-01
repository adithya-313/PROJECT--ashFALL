# File: tests/unit/test_run_scoped_watermarks.py
# Purpose: Unit tests for run-scoped ingestion watermark keys and visibility of refused backward moves.
"""
WHAT: Unit tests covering run-scoped watermark storage, Redis key layout, Lua refusal handling,
Prometheus rejected-moves counter increments, and run_id format validation.

WHY: Verifies Architect decisions: world-clock simulation runs must write watermarks under
run-scoped keys to avoid colliding with real-clock watermarks, backward moves must log warnings
and increment metrics rather than failing silently, and aggregate bank watermarks must read only
the corresponding run's partitions and pollers.
"""

import logging
from datetime import datetime, timezone

import fakeredis
import pytest
from prometheus_client import REGISTRY

from features.account_keys import (
    validate_run_id,
    watermark_key_for_poller,
)
from features.ingest_watermark import (
    get_bank_watermark,
    get_payment_watermark,
    get_poller_watermark,
    record_payment_watermark,
    record_poller_watermark,
)

RUN_ID_1 = "66db9aa439b092ee"
RUN_ID_2 = "77ec0bb540c1a3ff"


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


def test_a_run_scoped_poller_does_not_collide_with_real_clock(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """
    Test a: Redis holds legacy 'acctwm:bank_0:complaints' at Nov 2026.
    record_poller_watermark with run_id=R and Sept 2026 returns True;
    get_poller_watermark(run_id=R) returns Sept; legacy key still holds Nov.
    """
    t_nov = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
    t_sept = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Legacy key holds Nov 2026
    stored_legacy = record_poller_watermark(redis_client, "bank_0", "complaints", t_nov, run_id=None)
    assert stored_legacy is True
    assert redis_client.get("acctwm:bank_0:complaints") == str(int(t_nov.timestamp())).encode("utf-8")

    # Run-scoped record with Sept 2026 time returns True
    res = record_poller_watermark(redis_client, "bank_0", "complaints", t_sept, run_id=RUN_ID_1)
    assert res is True

    # Run-scoped key returns Sept 2026
    wm_run = get_poller_watermark(redis_client, "bank_0", "complaints", run_id=RUN_ID_1)
    assert wm_run == t_sept

    # Legacy key remains unchanged at Nov 2026
    wm_legacy = get_poller_watermark(redis_client, "bank_0", "complaints", run_id=None)
    assert wm_legacy == t_nov
    assert redis_client.get("acctwm:bank_0:complaints") == str(int(t_nov.timestamp())).encode("utf-8")


def test_b_runs_r1_and_r2_isolation(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """
    Test b: Runs R1 and R2 on one Redis: each reads only its own poller AND payment watermarks.
    """
    t_r1_poller = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
    t_r2_poller = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t_r1_pay = datetime(2026, 5, 2, 10, 0, 0, tzinfo=timezone.utc)
    t_r2_pay = datetime(2026, 6, 2, 10, 0, 0, tzinfo=timezone.utc)

    # Record R1 poller and payments
    record_poller_watermark(redis_client, "bank_0", "events", t_r1_poller, run_id=RUN_ID_1)
    record_payment_watermark(redis_client, 0, t_r1_pay, run_id=RUN_ID_1)

    # Record R2 poller and payments
    record_poller_watermark(redis_client, "bank_0", "events", t_r2_poller, run_id=RUN_ID_2)
    record_payment_watermark(redis_client, 0, t_r2_pay, run_id=RUN_ID_2)

    # Validate R1 reads its own
    assert get_poller_watermark(redis_client, "bank_0", "events", run_id=RUN_ID_1) == t_r1_poller
    assert get_payment_watermark(redis_client, 0, run_id=RUN_ID_1) == t_r1_pay

    # Validate R2 reads its own
    assert get_poller_watermark(redis_client, "bank_0", "events", run_id=RUN_ID_2) == t_r2_poller
    assert get_payment_watermark(redis_client, 0, run_id=RUN_ID_2) == t_r2_pay


def test_c_run_id_none_writes_legacy_literal_keys(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """
    Test c: run_id None writes the literal keys 'acctwm:bank_0:complaints' and 'acctwm:payments:0'.
    """
    t_now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    record_poller_watermark(redis_client, "bank_0", "complaints", t_now, run_id=None)
    record_payment_watermark(redis_client, 0, t_now, run_id=None)

    assert redis_client.exists("acctwm:bank_0:complaints") == 1
    assert redis_client.exists("acctwm:payments:0") == 1
    assert redis_client.sismember("acctwm:payments:partitions", "0")


def test_d_backward_move_refused_logged_and_counted(
    redis_client: fakeredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Test d: Backward move returns False, stored value unchanged, caplog has a WARNING
    containing the key and both values, and counter is incremented by exactly 1.
    """
    t_high = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    t_low = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)

    # Sample Prometheus counter before
    before_sample = (
        REGISTRY.get_sample_value(
            "ashfall_watermark_rejected_moves_total",
            {"source": "complaints"},
        )
        or 0.0
    )

    # Initial write at higher time
    stored = record_poller_watermark(redis_client, "bank_0", "complaints", t_high, run_id=RUN_ID_1)
    assert stored is True

    # Attempt backward move with logging capture at WARNING
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        refused = record_poller_watermark(redis_client, "bank_0", "complaints", t_low, run_id=RUN_ID_1)

    assert refused is False

    # Stored value unchanged
    assert get_poller_watermark(redis_client, "bank_0", "complaints", run_id=RUN_ID_1) == t_high

    # Counter increased by exactly 1
    after_sample = (
        REGISTRY.get_sample_value(
            "ashfall_watermark_rejected_moves_total",
            {"source": "complaints"},
        )
        or 0.0
    )
    assert after_sample == before_sample + 1.0

    # Warning logged with key, stored value, and refused value
    expected_key = watermark_key_for_poller("bank_0", "complaints", run_id=RUN_ID_1)
    stored_str = str(int(t_high.timestamp()))
    refused_str = str(int(t_low.timestamp()))

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        expected_key in msg and stored_str in msg and refused_str in msg
        for msg in warning_msgs
    ), f"Expected key and values in logs: {warning_msgs}"


def test_e_get_bank_watermark_run_scoped(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """
    Test e: get_bank_watermark(..., run_id=R) with all 5 run-scoped sources present returns their MIN;
    with one missing returns None.
    """
    # Initially none present
    assert get_bank_watermark(redis_client, "bank_0", run_id=RUN_ID_1) is None

    # Record 4 poller sources
    record_poller_watermark(redis_client, "bank_0", "events", 200, run_id=RUN_ID_1)
    record_poller_watermark(redis_client, "bank_0", "withdrawals", 300, run_id=RUN_ID_1)
    record_poller_watermark(redis_client, "bank_0", "feedback", 400, run_id=RUN_ID_1)
    record_poller_watermark(redis_client, "bank_0", "complaints", 500, run_id=RUN_ID_1)

    # Missing payments partition watermark -> returns None
    assert get_bank_watermark(redis_client, "bank_0", run_id=RUN_ID_1) is None

    # Record payment partition watermark with lowest ts (100)
    record_payment_watermark(redis_client, 0, 100, run_id=RUN_ID_1)

    # All 5 sources present -> returns MIN (100)
    wm = get_bank_watermark(redis_client, "bank_0", run_id=RUN_ID_1)
    assert wm == datetime.fromtimestamp(100, tz=timezone.utc)

    # Missing one poller on a different run -> returns None for other run
    assert get_bank_watermark(redis_client, "bank_0", run_id=RUN_ID_2) is None


def test_f_validate_run_id_rejects_invalid_inputs() -> None:
    """
    Test f: validate_run_id rejects wrong length, uppercase, non-hex.
    """
    # Valid 16-hex lowercase string succeeds
    validate_run_id("66db9aa439b092ee")
    validate_run_id("0123456789abcdef")

    # Wrong length
    with pytest.raises(ValueError, match="invalid length"):
        validate_run_id("66db9aa439b092e")  # 15 chars
    with pytest.raises(ValueError, match="invalid length"):
        validate_run_id("66db9aa439b092eee")  # 17 chars
    with pytest.raises(ValueError, match="invalid length"):
        validate_run_id("")

    # Uppercase
    with pytest.raises(ValueError, match="only lowercase hexadecimal"):
        validate_run_id("66DB9AA439B092EE")
    with pytest.raises(ValueError, match="only lowercase hexadecimal"):
        validate_run_id("66db9aa439b092Ee")

    # Non-hex characters
    with pytest.raises(ValueError, match="only lowercase hexadecimal"):
        validate_run_id("66db9aa439b092zz")
    with pytest.raises(ValueError, match="only lowercase hexadecimal"):
        validate_run_id("66db9aa439b092-e")

    # Non-string
    with pytest.raises(ValueError, match="must be a string"):
        validate_run_id(12345)  # type: ignore[arg-type]
