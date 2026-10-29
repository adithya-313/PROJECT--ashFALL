# File: tests/integration/test_account_summaries_end_to_end.py
# Purpose: End-to-end integration test proving long-history summaries, cash-out state, and M1 features.
# Issue: Level 15b long-history summaries and cash-out state (plan Task 1, G1, G2, R5, R7).
"""
WHAT: Full-stack integration test connecting real PostgreSQL (outbox, bank_withdrawals, bank_account_events),
Redpanda (Kafka upi.payments topic), and Redis (sliding window, long-history summaries, lifetime state).

WHY: Proves end-to-end correctness across relational event streaming, transactional cash withdrawals,
payment processing, closed-day summary aggregation, and point-in-time lifetime metrics.
Kafka gives no per-account ordering; state is correct in any order because all writes are repeat-safe.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import redis

from api.services.outbox_relay import relay_one_batch
from api.v1.upi_outbox import UpiPaymentSide, record_upi_side
from core.config import config
from features.account_keys import load_account_features_settings, namespace_pattern
from features.account_reader import read_account_snapshot
from features.account_stream import (
    apply_account_events_since,
    apply_withdrawals_since,
    run_payment_updater,
)
from features.account_summaries import (
    history_days_available,
    read_counterparty_days,
    read_day_totals,
    read_lifetime,
    to_day_number,
)
from showcase.kafka_tools import create_topic, delete_topic, make_producer, make_topic_name

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


def host_side_redis_url() -> str:
    """
    Resolve the Redis connection string for tests running on the host machine.

    The .env Redis address uses the Docker service name ('redis'), which only resolves
    inside Docker containers. On the host runner, this swaps 'redis' for 'localhost'.
    """
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


async def test_account_summaries_and_cash_out_end_to_end(isolated_schema: tuple[Any, str]) -> None:
    """
    WHAT: End-to-end integration scenario testing withdrawals, payments, account events, and summaries.

    SCENARIO:
      1. Inserts historical 'account_opened' event and 'device_binding_change' event into bank_account_events.
      2. Inserts successful ATM withdrawal (G2) and failed ATM withdrawal (R7) into bank_withdrawals.
      3. Records debit and credit payment sides via transactional outbox, relays to Kafka, updates Redis.
      4. Polls bank_account_events and bank_withdrawals into Redis state via poller adapters.
      5. Verifies raw window snapshot contains all payments, withdrawals, and events with exact balances.
      6. Verifies closed-day summary aggregates:
         - Unclosed when evaluated at event day;
         - Closed when evaluated after day end + allowed lateness;
         - Failed withdrawal increments failed_attempt_count but is excluded from withdrawal_paise;
         - Counterparty active days tracked per direction (G1);
         - Lifetime summary extracts account age and device fingerprints (R5).
    """
    connection, _ = isolated_schema
    redis_client = redis.Redis.from_url(host_side_redis_url())
    settings = load_account_features_settings()

    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_sum_01"
    counterparty_bank_id = "bank_1"
    cp_payee_account = "acc_payee_02"
    cp_payer_account = "acc_payer_03"

    now = datetime.now(timezone.utc)
    t_open = now - timedelta(days=90)

    # 1. Insert account events into relational storage (bank_account_events)
    # Account opened 90 days ago
    await connection.execute(
        """
        INSERT INTO bank_account_events (bank_id, event_id, account_id, kind, event_time, details, created_at)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        """,
        bank_id,
        f"{run_id}-open-01",
        account_id,
        "account_opened",
        t_open,
        '{"home_region": "KA"}',
        now,
    )
    # Device change observed today
    await connection.execute(
        """
        INSERT INTO bank_account_events (bank_id, event_id, account_id, kind, event_time, details, created_at)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        """,
        bank_id,
        f"{run_id}-dev-01",
        account_id,
        "device_binding_change",
        now,
        '{"fingerprint_before": "fp_sum_alpha", "fingerprint_after": "fp_sum_beta"}',
        now,
    )

    # 2. Insert cash withdrawals into relational storage (bank_withdrawals, G2)
    # Successful ATM withdrawal: 5,000 INR
    await connection.execute(
        """
        INSERT INTO bank_withdrawals (
            bank_id, withdrawal_id, account_id, channel, amount_paise,
            status, failure_domain, event_time, balance_after_paise, recorded_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
        bank_id,
        f"{run_id}-w-0001",
        account_id,
        "atm",
        500000,
        "success",
        None,
        now,
        4500000,
        now,
    )
    # Failed ATM withdrawal: 1,000 INR (R7: counts in failed_attempt_count only, never in amount)
    await connection.execute(
        """
        INSERT INTO bank_withdrawals (
            bank_id, withdrawal_id, account_id, channel, amount_paise,
            status, failure_domain, event_time, balance_after_paise, recorded_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
        bank_id,
        f"{run_id}-w-0002",
        account_id,
        "atm",
        100000,
        "failed",
        "customer",
        now,
        4500000,
        now,
    )

    # 3. Record payment sides via outbox, relay to Kafka topic, and consume into Redis
    topic = make_topic_name(prefix="acctf-sum")
    await create_topic(topic)
    producer = await make_producer()

    try:
        # Outbound payment side (debit): 1,500 INR
        debit_side = UpiPaymentSide(
            payment_id=f"{run_id}-p-0001",
            bank_id=bank_id,
            side="debit",
            account_id=account_id,
            counterparty_bank_id=counterparty_bank_id,
            sender_bank_id=bank_id,
            receiver_bank_id=counterparty_bank_id,
            amount_paise=150000,
            event_time=now,
            status="success",
            counterparty_vpa=cp_payee_account,
            payee_type="person",
            channel="upi_app",
            sender_device_fingerprint="fp_sum_alpha",
            sender_os_generation=2,
            balance_after_paise=4350000,
        )
        # Inbound payment side (credit): 2,000 INR
        credit_side = UpiPaymentSide(
            payment_id=f"{run_id}-p-0002",
            bank_id=bank_id,
            side="credit",
            account_id=account_id,
            counterparty_bank_id=counterparty_bank_id,
            sender_bank_id=counterparty_bank_id,
            receiver_bank_id=bank_id,
            amount_paise=200000,
            event_time=now,
            status="success",
            counterparty_vpa=cp_payer_account,
            payee_type="person",
            channel="upi_app",
            balance_after_paise=4550000,
        )

        await record_upi_side(connection, debit_side)
        await record_upi_side(connection, credit_side)

        # Relay outbox to Kafka
        while True:
            res = await relay_one_batch(connection, producer, batch_size=10, topic=topic)
            if res.waiting == 0:
                break

        # Consume from Kafka into Redis
        processed_payments = await run_payment_updater(
            redis_client,
            group_id=f"grp-{run_id}",
            idle_timeout_seconds=2.0,
            topic=topic,
            settings=settings,
        )
        assert processed_payments == 2

        # 4. Ingest relational events and withdrawals into Redis via pollers
        ev_id = await apply_account_events_since(connection, redis_client, settings=settings, bank_id=bank_id)
        assert ev_id > 0

        w_id = await apply_withdrawals_since(connection, redis_client, settings=settings, bank_id=bank_id)
        assert w_id > 0

        # 5. Verify Raw Window AccountSnapshot
        snap_query_time = now + timedelta(minutes=5)
        snapshot = read_account_snapshot(
            redis_client, run_id, bank_id, account_id, snap_query_time, snap_query_time, settings
        )
        assert snapshot.has_history is True
        assert len(snapshot.side_records) == 2
        assert len(snapshot.withdrawals) == 2
        # account_opened from 90 days ago was pruned from the 7-day raw window by retention policy,
        # but is preserved in lifetime opened_at. Only the recent device event remains in the raw window.
        assert len(snapshot.account_events) == 1

        # Prove balance_after_paise populated
        balances = {rec.balance_after_paise for rec in snapshot.side_records}
        assert 4350000 in balances
        assert 4550000 in balances

        w_statuses = {w.status for w in snapshot.withdrawals}
        assert "success" in w_statuses
        assert "failed" in w_statuses

        # 6. Verify Point-in-Time Closed-Day Summaries
        # (a) Evaluated at decision_time = now: day is NOT closed yet -> empty summary
        totals_unclosed = read_day_totals(
            redis_client, run_id, bank_id, account_id, now, now, settings
        )
        assert totals_unclosed == ()

        # (b) Evaluated at decision_time = now + 2 days (past day end + allowed lateness): day is closed
        future_decision = now + timedelta(days=2)
        totals_closed = read_day_totals(
            redis_client, run_id, bank_id, account_id, future_decision, future_decision, settings
        )
        assert len(totals_closed) == 1
        day_tot = totals_closed[0]
        assert day_tot.day == to_day_number(now)
        assert day_tot.credit_count == 1
        assert day_tot.credit_paise == 200000
        assert day_tot.debit_count == 1
        assert day_tot.debit_paise == 150000
        assert day_tot.withdrawal_count == 1
        assert day_tot.withdrawal_paise == 500000
        assert day_tot.failed_attempt_count == 1

        # 7. Verify Counterparty Active Days (G1)
        cp_outbound = f"{counterparty_bank_id}:{cp_payee_account}"
        cp_debit_map = read_counterparty_days(
            redis_client, run_id, bank_id, account_id, "debit",
            future_decision, future_decision, settings,
        )
        assert cp_outbound in cp_debit_map
        assert len(cp_debit_map[cp_outbound]) == 1

        cp_inbound = f"{counterparty_bank_id}:{cp_payer_account}"
        cp_credit_map = read_counterparty_days(
            redis_client, run_id, bank_id, account_id, "credit",
            future_decision, future_decision, settings,
        )
        assert cp_inbound in cp_credit_map
        assert len(cp_credit_map[cp_inbound]) == 1

        # 8. Verify Lifetime Metrics and History Days Available
        lifetime = read_lifetime(
            redis_client, run_id, bank_id, account_id, future_decision, future_decision, settings
        )
        assert lifetime.opened_at == t_open
        assert lifetime.last_active_day == to_day_number(now)
        assert set(lifetime.fingerprints) >= {"fp_sum_alpha", "fp_sum_beta"}

        available_days = history_days_available(
            lifetime.opened_at,
            future_decision,
            settings.summary_horizon_days,
        )
        assert available_days == settings.summary_horizon_days

    finally:
        await producer.stop()
        await delete_topic(topic)
        # Purge Redis keys under run namespace
        for pat in (namespace_pattern(run_id, bank_id), namespace_pattern(run_id, counterparty_bank_id)):
            for k in redis_client.scan_iter(match=pat):
                redis_client.delete(k)
        redis_client.close()
