# File: tests/unit/test_run_scoped_decision_wiring.py
# Purpose: Unit tests for world-mode decisions wiring (run_scoped_watermarks and per-client feature logging).
from datetime import datetime, timezone

import fakeredis
import pytest

from api.v1.models import DecisionRequest
from features.ingest_watermark import record_payment_watermark, record_poller_watermark
from features.registry_prefetch import prepare_registry_questions

pytestmark = pytest.mark.anyio

BANK_ID = "bank_0"
RUN_ID = "66db9aa439b092ee"
PAYMENT_ID = f"{RUN_ID}-0001"


def make_request(payment_id: str = PAYMENT_ID, event_time: datetime | None = None) -> DecisionRequest:
    """Helper to build a valid DecisionRequest."""
    return DecisionRequest(
        payment_id=payment_id,
        sender_account_id="acc_sender_1",
        receiver_bank_id="bank_1",
        amount_paise=100000,
        event_time=event_time or datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc),
        channel="upi_app",
        payee_type="person",
    )


def test_prepare_registry_questions_run_scoped_reads_run_keys_only() -> None:
    """
    prepare_registry_questions with run_scoped_watermarks=True reads run-scoped keys,
    returning None when only shared keys exist, and returning MIN of run keys when present.
    Default False reads shared keys.
    """
    client = fakeredis.FakeRedis()
    req = make_request()

    # 1. Seed only shared keys (run_id=None)
    t_shared_min = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    ts_shared = int(t_shared_min.timestamp())
    record_payment_watermark(client, 0, ts_shared, run_id=None)
    for src in ("events", "complaints", "feedback", "withdrawals"):
        record_poller_watermark(client, BANK_ID, src, ts_shared + 100, run_id=None)

    # With run_scoped_watermarks=True: run keys do not exist yet -> returns None
    prepared_run = prepare_registry_questions(
        client,
        BANK_ID,
        req,
        dpip_key=None,
        run_scoped_watermarks=True,
    )
    assert prepared_run is None

    # With default run_scoped_watermarks=False: shared keys exist -> returns shared watermark
    prepared_shared = prepare_registry_questions(
        client,
        BANK_ID,
        req,
        dpip_key=None,
        run_scoped_watermarks=False,
    )
    assert prepared_shared is not None
    cutoff_shared, _ = prepared_shared
    assert cutoff_shared == t_shared_min

    # 2. Seed all 5 run-scoped keys (run_id=RUN_ID)
    t_run_min = datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc)
    ts_run_min = int(t_run_min.timestamp())
    record_payment_watermark(client, 0, ts_run_min, run_id=RUN_ID)
    record_poller_watermark(client, BANK_ID, "events", ts_run_min + 50, run_id=RUN_ID)
    record_poller_watermark(client, BANK_ID, "withdrawals", ts_run_min + 150, run_id=RUN_ID)
    record_poller_watermark(client, BANK_ID, "feedback", ts_run_min + 200, run_id=RUN_ID)
    record_poller_watermark(client, BANK_ID, "complaints", ts_run_min + 300, run_id=RUN_ID)

    # Now with run_scoped_watermarks=True: returns run MIN
    prepared_run_seeded = prepare_registry_questions(
        client,
        BANK_ID,
        req,
        dpip_key=None,
        run_scoped_watermarks=True,
    )
    assert prepared_run_seeded is not None
    cutoff_run, questions = prepared_run_seeded
    assert cutoff_run == t_run_min
    assert isinstance(questions, list)

