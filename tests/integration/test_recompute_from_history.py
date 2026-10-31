# File: tests/integration/test_recompute_from_history.py
# Purpose: Integration tests for historical feature replay and point-in-time recomputation verification.
# Issue: Level 15c Round 2a (recompute a logged feature row from history).
"""
WHAT: Integration tests proving historical feature reproducibility from relational database state
into an isolated scratch Redis instance against feature_log records produced by live ingestion.

WHY: Verifies Section 22 determinism and point-in-time correctness:
1. Exact Match: Recomputing a decision feature row from historical transactions, account events,
   withdrawals, feedback, and complaints reproduces the logged feature row bit-for-bit with strict
   type and value equality.
2. Late Event Excluded: A transaction occurring before decision time but arriving after the ingestion
   watermark (arrival_cutoff) is strictly excluded during replay, leaving recomputed features identical.
3. Detects Drift: Any mutation to pre-cutoff historical transactions causes recomputation mismatches,
   proving drift detection on amount-dependent features.
"""

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import fakeredis
import orjson
import pytest
import redis
from pydantic import SecretStr

from api.services.outbox_relay import relay_one_batch
from api.v1.bank_keys import issue_api_key
from api.v1.bank_service import InProcessBankClient
from api.v1.upi_outbox import UpiPaymentSide, record_upi_side
from core.config import config
from core.event_clock import ManualClock
from features.account_keys import namespace_pattern
from features.account_labels import apply_complaints_since, apply_feedback_since
from features.account_stream import (
    apply_account_events_since,
    apply_withdrawals_since,
    run_payment_updater,
)
from features.feature_log import canonical_feature_names, read_feature_log_row
from features.feature_replay import recompute_logged_row
from features.ingest_watermark import get_bank_watermark, record_poller_watermark
from features.m1_features import compute_account_features
from features.registry_screening import FakeRegistryScreener, ScreenOutcome
from showcase.kafka_tools import create_topic, delete_topic, make_producer, make_topic_name
from tests.integration.conftest import get_test_database_url

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


def host_side_redis_url() -> str:
    """Resolve Redis connection string for test runner on host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


@dataclass
class BaselineSetup:
    """Container for baseline history fixtures and identifiers."""

    bank_id: str
    run_id: str
    payer_acc: str
    decision_payment_id: str
    side1_payment_id: str
    decision_time: datetime
    topic: str
    producer: Any
    live_redis: redis.Redis
    db_pool: asyncpg.Pool


@asynccontextmanager
async def setup_baseline_history(
    connection: asyncpg.Connection,
    schema_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[BaselineSetup]:
    """Sets up relational event history, streaming Kafka pipeline, and logged decision."""
    monkeypatch.setattr(config, "decision_feature_logging_enabled", True)
    bank_id = "bank_0"
    counterparty_bank_id = "bank_1"
    secret_text = f"test_secret_for_{bank_id}_32_chars_min!"
    monkeypatch.setattr(config, f"bank_signing_secret_{bank_id}", SecretStr(secret_text))
    await issue_api_key(connection, bank_id)

    run_id = uuid.uuid4().hex[:16]
    payer_acc = "acc_payer_001"
    payee_vpa = "acc_payee_002@bank1"

    t0 = datetime(2026, 10, 25, 10, 0, 0, tzinfo=timezone.utc)
    t_day1 = t0
    t_day2 = t0 + timedelta(days=1)
    t_day3 = t0 + timedelta(days=2)
    t_dec = t_day3 + timedelta(hours=3)
    t_day4 = t0 + timedelta(days=3)

    # Pre-create partition for decision date
    await connection.execute("SELECT create_feature_log_partition($1::date)", t_dec.date())

    db_pool = await asyncpg.create_pool(
        dsn=get_test_database_url(),
        min_size=1,
        max_size=3,
        server_settings={"search_path": schema_name},
    )
    assert db_pool is not None

    topic = make_topic_name(prefix="recomp")
    await create_topic(topic)
    producer = await make_producer()
    live_redis = redis.Redis.from_url(host_side_redis_url())

    try:
        # 1. Day 1: Debit payment side (1,000 INR)
        side1_pid = f"{run_id}-p-day1"
        s1 = UpiPaymentSide(
            payment_id=side1_pid,
            bank_id=bank_id,
            side="debit",
            account_id=payer_acc,
            counterparty_bank_id=counterparty_bank_id,
            sender_bank_id=bank_id,
            receiver_bank_id=counterparty_bank_id,
            amount_paise=100000,
            event_time=t_day1,
            status="success",
            counterparty_vpa=payee_vpa,
            payee_type="person",
            channel="upi_app",
            sender_device_fingerprint="fp_test_dev",
            sender_os_generation=2,
            balance_after_paise=900000,
        )
        stored1 = await record_upi_side(connection, s1)
        created_day1 = t_day1 + timedelta(seconds=1)
        await connection.execute(
            "UPDATE transactions SET created_at = $1 WHERE transaction_id = $2",
            created_day1,
            uuid.UUID(stored1.transaction_id),
        )
        await connection.execute(
            "UPDATE outbox SET payload = jsonb_set(payload, '{recorded_at}', $1::jsonb) WHERE transaction_id = $2",
            f'"{created_day1.isoformat()}"',
            uuid.UUID(stored1.transaction_id),
        )

        # Day 1: Account lifecycle event: account_opened
        await connection.execute(
            """
            INSERT INTO bank_account_events (bank_id, event_id, account_id, kind, event_time, details, created_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            """,
            bank_id,
            f"{run_id}-evt-open",
            payer_acc,
            "account_opened",
            t_day1,
            '{"home_region": "KA"}',
            t_day1 + timedelta(seconds=2),
        )

        # 2. Day 2: Debit payment side (2,000 INR)
        side2_pid = f"{run_id}-p-day2"
        s2 = UpiPaymentSide(
            payment_id=side2_pid,
            bank_id=bank_id,
            side="debit",
            account_id=payer_acc,
            counterparty_bank_id=counterparty_bank_id,
            sender_bank_id=bank_id,
            receiver_bank_id=counterparty_bank_id,
            amount_paise=200000,
            event_time=t_day2,
            status="success",
            counterparty_vpa=payee_vpa,
            payee_type="person",
            channel="upi_app",
            sender_device_fingerprint="fp_test_dev",
            sender_os_generation=2,
            balance_after_paise=700000,
        )
        stored2 = await record_upi_side(connection, s2)
        created_day2 = t_day2 + timedelta(seconds=1)
        await connection.execute(
            "UPDATE transactions SET created_at = $1 WHERE transaction_id = $2",
            created_day2,
            uuid.UUID(stored2.transaction_id),
        )
        await connection.execute(
            "UPDATE outbox SET payload = jsonb_set(payload, '{recorded_at}', $1::jsonb) WHERE transaction_id = $2",
            f'"{created_day2.isoformat()}"',
            uuid.UUID(stored2.transaction_id),
        )

        # Day 2: ATM cash withdrawal (500 INR)
        await connection.execute(
            """
            INSERT INTO bank_withdrawals (
                bank_id, withdrawal_id, account_id, channel, amount_paise,
                status, failure_domain, event_time, balance_after_paise, recorded_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            bank_id,
            f"{run_id}-wth-01",
            payer_acc,
            "atm",
            50000,
            "success",
            None,
            t_day2 + timedelta(hours=1),
            650000,
            t_day2 + timedelta(hours=1),
        )

        # Day 2: Feedback confirmed_fraud on Day 2 payment
        await connection.execute(
            """
            INSERT INTO bank_feedback (
                bank_id, feedback_id, payment_id, outcome, event_time, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            bank_id,
            f"{run_id}-fb-01",
            side2_pid,
            "confirmed_fraud",
            t_day2 + timedelta(hours=2),
            t_day2 + timedelta(hours=2),
        )

        # 3. Day 3: Credit payment side (3,000 INR)
        side3_pid = f"{run_id}-p-day3"
        s3 = UpiPaymentSide(
            payment_id=side3_pid,
            bank_id=bank_id,
            side="credit",
            account_id=payer_acc,
            counterparty_bank_id=counterparty_bank_id,
            sender_bank_id=counterparty_bank_id,
            receiver_bank_id=bank_id,
            amount_paise=300000,
            event_time=t_day3,
            status="success",
            counterparty_vpa="other@bank1",
            payee_type="person",
            channel="upi_app",
            balance_after_paise=950000,
        )
        stored3 = await record_upi_side(connection, s3)
        created_day3 = t_day3 + timedelta(seconds=1)
        await connection.execute(
            "UPDATE transactions SET created_at = $1 WHERE transaction_id = $2",
            created_day3,
            uuid.UUID(stored3.transaction_id),
        )
        await connection.execute(
            "UPDATE outbox SET payload = jsonb_set(payload, '{recorded_at}', $1::jsonb) WHERE transaction_id = $2",
            f'"{created_day3.isoformat()}"',
            uuid.UUID(stored3.transaction_id),
        )

        # Day 3: Complaint against payer_acc
        await connection.execute(
            """
            INSERT INTO bank_complaints (
                bank_id, complaint_id, complainant_account, reported_account, reported_bank_id,
                payment_ids, claimed_type, received_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            bank_id,
            f"{run_id}-cmp-01",
            "acc_complainant_10",
            payer_acc,
            bank_id,
            [side3_pid],
            "fraud",
            t_day3 + timedelta(hours=1),
            t_day3 + timedelta(hours=1),
        )

        # 4. Day 4: Subsequent side to advance Kafka watermark past Day 3 + commit_grace
        side4_pid = f"{run_id}-p-day4"
        s4 = UpiPaymentSide(
            payment_id=side4_pid,
            bank_id=bank_id,
            side="debit",
            account_id="acc_unrelated_99",
            counterparty_bank_id=counterparty_bank_id,
            sender_bank_id=bank_id,
            receiver_bank_id=counterparty_bank_id,
            amount_paise=10000,
            event_time=t_day4,
            status="success",
            counterparty_vpa="other@bank1",
            payee_type="person",
            channel="upi_app",
        )
        stored4 = await record_upi_side(connection, s4)
        created_day4 = t_day4
        await connection.execute(
            "UPDATE transactions SET created_at = $1 WHERE transaction_id = $2",
            created_day4,
            uuid.UUID(stored4.transaction_id),
        )
        await connection.execute(
            "UPDATE outbox SET payload = jsonb_set(payload, '{recorded_at}', $1::jsonb) WHERE transaction_id = $2",
            f'"{created_day4.isoformat()}"',
            uuid.UUID(stored4.transaction_id),
        )

        # Relay all outbox rows to Kafka
        while True:
            batch_res = await relay_one_batch(connection, producer, batch_size=10, topic=topic)
            if batch_res.waiting == 0:
                break

        # Consume payments from Kafka into live Redis
        processed_payments = await run_payment_updater(
            live_redis,
            group_id=f"grp-{run_id}",
            idle_timeout_seconds=2.0,
            topic=topic,
        )
        assert processed_payments >= 4

        # Run Postgres pollers to populate Redis labels and watermarks
        await apply_account_events_since(connection, live_redis, bank_id=bank_id)
        await apply_withdrawals_since(connection, live_redis, bank_id=bank_id)
        await apply_feedback_since(connection, live_redis, bank_id=bank_id)
        await apply_complaints_since(connection, live_redis, bank_id=bank_id)

        # Setup Registry Screener and InProcessBankClient
        screener = FakeRegistryScreener(
            table={
                (payer_acc, "account"): ScreenOutcome(status="no_match"),
                (payee_vpa, "account"): ScreenOutcome(status="no_match"),
            }
        )
        clock = ManualClock(t_dec)
        in_process = InProcessBankClient(
            pool=db_pool,
            bank_id=bank_id,
            clock=clock,
            redis_client=live_redis,
            decision_clock=clock,
            screener=screener,
        )

        # Post decision request
        dec_pid = f"{run_id}-dec-001"
        payload = {
            "payment_id": dec_pid,
            "sender_account_id": payer_acc,
            "receiver_bank_id": counterparty_bank_id,
            "receiver_vpa": payee_vpa,
            "payee_type": "person",
            "amount_paise": 150000,
            "event_time": t_dec.isoformat(),
        }
        resp = await in_process.post("/v1/decisions", orjson.dumps(payload), idempotency_key=f"idem-{dec_pid}")
        assert resp.status == 200

        yield BaselineSetup(
            bank_id=bank_id,
            run_id=run_id,
            payer_acc=payer_acc,
            decision_payment_id=dec_pid,
            side1_payment_id=side1_pid,
            decision_time=t_dec,
            topic=topic,
            producer=producer,
            live_redis=live_redis,
            db_pool=db_pool,
        )
    finally:
        await producer.stop()
        await delete_topic(topic)
        # Prune test Redis keys
        for pat in (namespace_pattern(run_id, bank_id), f"*{run_id}*"):
            for k in live_redis.scan_iter(match=pat):
                live_redis.delete(k)
        live_redis.close()
        await db_pool.close()


async def test_recompute_exact_match(isolated_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test 1 (EXACT MATCH): Proves historical replay produces a feature vector matching the logged row
    bit-for-bit with exact value and type equality across all canonical features.
    """
    connection, schema_name = isolated_schema
    async with setup_baseline_history(connection, schema_name, monkeypatch) as base:
        log_row = await read_feature_log_row(connection, base.bank_id, base.decision_payment_id)
        assert log_row is not None
        assert log_row.status == "ok"

        scratch_redis = fakeredis.FakeRedis()
        result = await recompute_logged_row(
            pg_connection=connection,
            scratch_redis=scratch_redis,
            bank_id=base.bank_id,
            payment_id=base.decision_payment_id,
        )

        assert result.mismatches == [], f"Expected zero mismatches, got: {result.mismatches}"
        assert len(result.logged) > 0
        assert result.logged == result.recomputed
        for l_val, r_val in zip(result.logged, result.recomputed):
            assert type(l_val) is type(r_val), f"Type mismatch: {type(l_val)} vs {type(r_val)}"
            assert l_val == r_val

        not_none_count = sum(1 for v in result.logged if v is not None)
        print(f"T1 non-None canonical features count: {not_none_count}")
        assert not_none_count >= 10, f"Expected >= 10 non-None features, got {not_none_count}"


async def test_recompute_late_arriving_event_excluded(
    isolated_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test 2 (LATE EVENT): Proves that a payment side with event_time < decision_time but
    arrival_time > arrival_cutoff is strictly excluded during replay, leaving recomputation identical.
    """
    connection, schema_name = isolated_schema
    async with setup_baseline_history(connection, schema_name, monkeypatch) as base:
        # (a) Read feature_log row BEFORE the late event
        row_before = await read_feature_log_row(connection, base.bank_id, base.decision_payment_id)
        assert row_before is not None
        assert row_before.arrival_cutoff is not None

        # Late-arriving payment side: event_time < decision_time,
        # but arrival is strictly after the decision arrival_cutoff
        t_late_event = base.decision_time - timedelta(minutes=10)
        t_late_arrival = row_before.arrival_cutoff + timedelta(hours=1)
        late_pid = f"{base.run_id}-p-late"

        late_side = UpiPaymentSide(
            payment_id=late_pid,
            bank_id=base.bank_id,
            side="debit",
            account_id=base.payer_acc,
            counterparty_bank_id="bank_1",
            sender_bank_id=base.bank_id,
            receiver_bank_id="bank_1",
            amount_paise=999999,
            event_time=t_late_event,
            status="success",
            counterparty_vpa="other@bank1",
            payee_type="person",
            channel="upi_app",
        )
        stored_late = await record_upi_side(connection, late_side)
        await connection.execute(
            "UPDATE transactions SET created_at = $1 WHERE transaction_id = $2",
            t_late_arrival,
            uuid.UUID(stored_late.transaction_id),
        )
        await connection.execute(
            "UPDATE outbox SET payload = jsonb_set(payload, '{recorded_at}', $1::jsonb) WHERE transaction_id = $2",
            f'"{t_late_arrival.isoformat()}"',
            uuid.UUID(stored_late.transaction_id),
        )

        # Subsequent advancer payment side to advance watermark past late arrival
        t_adv = t_late_arrival + timedelta(hours=1)
        adv_pid = f"{base.run_id}-p-adv"
        adv_side = UpiPaymentSide(
            payment_id=adv_pid,
            bank_id=base.bank_id,
            side="debit",
            account_id="acc_unrelated_98",
            counterparty_bank_id="bank_1",
            sender_bank_id=base.bank_id,
            receiver_bank_id="bank_1",
            amount_paise=10000,
            event_time=t_adv,
            status="success",
            counterparty_vpa="other@bank1",
            payee_type="person",
            channel="upi_app",
        )
        stored_adv = await record_upi_side(connection, adv_side)
        await connection.execute(
            "UPDATE transactions SET created_at = $1 WHERE transaction_id = $2",
            t_adv,
            uuid.UUID(stored_adv.transaction_id),
        )
        await connection.execute(
            "UPDATE outbox SET payload = jsonb_set(payload, '{recorded_at}', $1::jsonb) WHERE transaction_id = $2",
            f'"{t_adv.isoformat()}"',
            uuid.UUID(stored_adv.transaction_id),
        )

        # Relay to Kafka and consume into live Redis
        while True:
            batch_res = await relay_one_batch(connection, base.producer, batch_size=10, topic=base.topic)
            if batch_res.waiting == 0:
                break
        await run_payment_updater(
            base.live_redis,
            group_id=f"grp-{base.run_id}-late",
            idle_timeout_seconds=2.0,
            topic=base.topic,
        )
        for poller_src in ("events", "withdrawals", "feedback", "complaints"):
            record_poller_watermark(base.live_redis, base.bank_id, poller_src, t_adv)

        # (a) Read feature_log row AFTER the late event and assert identical
        row_after = await read_feature_log_row(connection, base.bank_id, base.decision_payment_id)
        assert row_after is not None
        assert row_before.feature_values == row_after.feature_values
        assert row_before.arrival_cutoff == row_after.arrival_cutoff
        assert row_before.decided_at == row_after.decided_at
        assert row_before == row_after

        # (b) Prove late event reached live Redis
        new_wm = get_bank_watermark(base.live_redis, base.bank_id)
        assert new_wm is not None
        live_feats = compute_account_features(
            client=base.live_redis,
            run_id=base.run_id,
            bank_id=base.bank_id,
            account_id=base.payer_acc,
            decision_time=base.decision_time,
            arrival_cutoff=new_wm,
        )
        logged_map = dict(zip(canonical_feature_names(), row_before.feature_values))
        differing = [k for k, v in live_feats.items() if k in logged_map and logged_map[k] != v]
        assert len(differing) > 0, "Expected live features with new watermark to differ after late event"

        # Recompute into scratch Redis: late event must be excluded by arrival cutoff
        scratch_redis = fakeredis.FakeRedis()
        result = await recompute_logged_row(
            pg_connection=connection,
            scratch_redis=scratch_redis,
            bank_id=base.bank_id,
            payment_id=base.decision_payment_id,
        )
        assert result.mismatches == [], f"Late event should not affect recompute, got: {result.mismatches}"


async def test_recompute_detects_drift(isolated_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test 3 (DETECTS DRIFT): Proves that mutating pre-cutoff historical transactions in PostgreSQL
    causes recompute_logged_row to flag mismatches on amount-dependent feature values.
    """
    connection, schema_name = isolated_schema
    async with setup_baseline_history(connection, schema_name, monkeypatch) as base:
        # Directly corrupt amount_paise of pre-cutoff Day 1 transaction in DB
        await connection.execute(
            """
            UPDATE transactions
            SET amount = amount * 10,
                amount_paise = amount_paise * 10
            WHERE bank_id = $1 AND payment_id = $2
            """,
            base.bank_id,
            base.side1_payment_id,
        )

        scratch_redis = fakeredis.FakeRedis()
        result = await recompute_logged_row(
            pg_connection=connection,
            scratch_redis=scratch_redis,
            bank_id=base.bank_id,
            payment_id=base.decision_payment_id,
        )

        assert len(result.mismatches) > 0, "Drift was not detected by historical recomputation!"
        mismatched_feature_names = {m[0] for m in result.mismatches}
        # Amount-dependent features must be among the mismatches
        amount_dependent = {
            "amount_vs_payer_median_ratio",
            "amount_vs_payer_max_ratio",
            "velocity_sum_paise_7d",
            "velocity_sum_paise_30d",
        }
        assert mismatched_feature_names & amount_dependent, (
            f"Expected drift in amount-dependent features, got mismatches: {mismatched_feature_names}"
        )
