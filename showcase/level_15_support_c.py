# File: showcase/level_15_support_c.py
# Purpose: Level 15c verification check implementations and shared seeded history support fixtures.
# Issue: Level 15c latency benchmark and showcase part c.
"""
WHAT: Implementation of Level 15 part c showcase checks and shared relational/Redis history builder:
  15c-1 [edge]: Late-arriving event leaves previously logged decision features byte-identical,
        while a fresh computation at the advanced watermark reflects the change.
  15c-2 [consistency]: Historical replay into isolated scratch Redis reproduces the logged feature
        row bit-for-bit (52 of 52 equal, >= 10 non-None).
  15c-3 [normal]: In-process decision latency and feature extraction latency benchmark against the
        pre-registered evaluation budget.
  15c-4 [negative]: Engine feature computation failure gracefully falls back to status='error' with
        null arrival cutoff while preserving the bank's unscored decision response.
  15c-5 [consistency]: Confirms zero unexpected error rows across all normal and edge decision runs.

WHY: Verifies end-to-end production integrity, Section 22 causal immutability, zero-tolerance replay
determinism, and latency budgets for real-time inference under operational Kafka and PostgreSQL ingress.
"""

from __future__ import annotations

import platform
import random
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import asyncpg
import fakeredis
import redis
from aiokafka import AIOKafkaProducer
from pydantic import SecretStr

import api.v1.bank_service as bank_service
from api.services.outbox_relay import relay_one_batch
from api.v1.bank_keys import issue_api_key
from api.v1.bank_service import record_decision
from api.v1.models import DecisionRequest
from api.v1.upi_outbox import UpiPaymentSide, record_upi_side
from benchmarks.decision_latency import (
    BENCHMARK_PAYER_DEVICE,
    load_latency_budget,
    measure_against_mock_registry,
    measure_decision_latency,
)
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
from showcase.framework import FIXED_SEED, CheckOutcome
from showcase.isolation import Workspace
from showcase.kafka_tools import create_topic, delete_topic, make_producer, make_topic_name
from showcase.level_15 import host_side_redis_url

# The seeded history reaches decision time with the same data in every run (fixed seed -> fixed run ids).
BANK_ID = "bank_0"
COUNTERPARTY_BANK_ID = "bank_1"
CANONICAL_FEATURE_COUNT = 52
# Shape of the seeded payment graph around the decided account (see _seed_network_history).
SAME_BANK_COUNTERPARTIES = 10
EXTRA_LABELLED_ACCOUNTS = 4


def seeded_run_id(offset: int) -> str:
    """A 16-hex run id that is the same on every run: FIXED_SEED plus an offset picks the random stream."""
    return f"{random.Random(FIXED_SEED + offset).getrandbits(64):016x}"


@dataclass
class PartCFixture:
    """Seeded Postgres, Kafka and Redis state shared by the part c checks and the latency benchmark."""

    bank_id: str
    run_id: str
    payer_acc: str
    payee_vpa: str
    decision_time: datetime
    topic: str
    producer: AIOKafkaProducer
    live_redis: redis.Redis
    connection: asyncpg.Connection
    screener: FakeRegistryScreener
    # Set by 15c-1 once its decision is logged; later checks read it.
    decision_payment_id: str | None = None
    # Set by 15c-4: the separate run id its forced-failure decision uses.
    error_run_id: str | None = None

    async def cleanup(self) -> None:
        """Stop the producer, drop the Kafka topic and delete every Redis key this fixture's runs created."""
        try:
            await self.producer.stop()
        finally:
            try:
                await delete_topic(self.topic)
            finally:
                self._delete_redis_keys()

    def _delete_redis_keys(self) -> None:
        """Delete the account-state keys of the main run and (if 15c-4 ran) of the error run, then close."""
        try:
            run_ids = [self.run_id] if self.error_run_id is None else [self.run_id, self.error_run_id]
            for run_id in run_ids:
                for pattern in (namespace_pattern(run_id, self.bank_id), f"*{run_id}*"):
                    for key in self.live_redis.scan_iter(match=pattern):
                        self.live_redis.delete(key)
        finally:
            self.live_redis.close()


async def _store_side(connection: asyncpg.Connection, side: UpiPaymentSide, arrived_at: datetime) -> None:
    """Store one payment side and stamp when it 'arrived'."""
    stored = await record_upi_side(connection, side)
    await _stamp_arrival(connection, stored.transaction_id, arrived_at)


async def _seed_same_bank_payment(
    connection: asyncpg.Connection,
    *,
    bank_id: str,
    payment_id: str,
    sender_acc: str,
    receiver_acc: str,
    amount_paise: int,
    event_time: datetime,
) -> None:
    """Store both sides of a payment between two accounts of this bank (a same-bank edge in the graph)."""
    arrived_at = event_time + timedelta(seconds=1)
    for side, account_id, counterparty in (("debit", sender_acc, receiver_acc), ("credit", receiver_acc, sender_acc)):
        await _store_side(
            connection,
            UpiPaymentSide(
                payment_id=payment_id,
                bank_id=bank_id,
                side=side,
                account_id=account_id,
                counterparty_bank_id=bank_id,
                sender_bank_id=bank_id,
                receiver_bank_id=bank_id,
                amount_paise=amount_paise,
                event_time=event_time,
                status="success",
                counterparty_vpa=counterparty,
                payee_type="person",
                channel="upi_app",
            ),
            arrived_at,
        )


async def _insert_confirmed_fraud_feedback(
    connection: asyncpg.Connection, *, bank_id: str, feedback_id: str, payment_id: str, at: datetime
) -> None:
    """Record a confirmed_fraud outcome on a payment; the poller then labels that payment's credit-side account."""
    await connection.execute(
        """
        INSERT INTO bank_feedback (bank_id, feedback_id, payment_id, outcome, event_time, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        bank_id,
        feedback_id,
        payment_id,
        "confirmed_fraud",
        at,
        at,
    )


async def _seed_network_history(
    connection: asyncpg.Connection,
    *,
    bank_id: str,
    run_id: str,
    payer_acc: str,
    payee_vpa: str,
    start: datetime,
) -> None:
    """
    WHAT: Gives the decided account a payment neighbourhood at its own bank: SAME_BANK_COUNTERPARTIES
    same-bank counterparties, one confirmed_fraud-labelled account two hops away (through the first
    counterparty), and EXTRA_LABELLED_ACCOUNTS more labelled accounts with 7-day history that share the
    payee counterparty with the decided account.

    WHY: Without neighbours and labels the network feature group does no graph work, so the latency
    benchmark would time an empty case. The graph is fixed (no randomness), so every run seeds the same history.
    """
    counterparties = [f"acc_nb_{run_id[:8]}_{i:02d}" for i in range(SAME_BANK_COUNTERPARTIES)]
    flagged_acc = f"acc_flag_{run_id[:8]}"
    labelled_accs = [f"acc_lab_{run_id[:8]}_{i:02d}" for i in range(EXTRA_LABELLED_ACCOUNTS)]
    step = 0

    def next_time() -> datetime:
        nonlocal step
        step += 1
        return start + timedelta(minutes=10 * step)

    for i, counterparty in enumerate(counterparties):
        await _seed_same_bank_payment(
            connection,
            bank_id=bank_id,
            payment_id=f"{run_id}-p-nb{i:02d}",
            sender_acc=payer_acc,
            receiver_acc=counterparty,
            amount_paise=50000 + 1000 * i,
            event_time=next_time(),
        )

    # The flagged account is paid by the first counterparty, so it is two hops from the decided account.
    flagged_payment = f"{run_id}-p-flag"
    await _seed_same_bank_payment(
        connection,
        bank_id=bank_id,
        payment_id=flagged_payment,
        sender_acc=counterparties[0],
        receiver_acc=flagged_acc,
        amount_paise=75000,
        event_time=next_time(),
    )
    labelled_payments = [flagged_payment]

    # These accounts were paid from another bank by the payee the decided account also pays.
    for i, labelled_acc in enumerate(labelled_accs):
        payment_id = f"{run_id}-p-lab{i:02d}"
        event_time = next_time()
        await _store_side(
            connection,
            UpiPaymentSide(
                payment_id=payment_id,
                bank_id=bank_id,
                side="credit",
                account_id=labelled_acc,
                counterparty_bank_id=COUNTERPARTY_BANK_ID,
                sender_bank_id=COUNTERPARTY_BANK_ID,
                receiver_bank_id=bank_id,
                amount_paise=60000 + 1000 * i,
                event_time=event_time,
                status="success",
                counterparty_vpa=payee_vpa,
                payee_type="person",
                channel="upi_app",
            ),
            event_time + timedelta(seconds=1),
        )
        labelled_payments.append(payment_id)

    feedback_at = next_time()
    for i, payment_id in enumerate(labelled_payments):
        await _insert_confirmed_fraud_feedback(
            connection, bank_id=bank_id, feedback_id=f"{run_id}-fb-net{i:02d}", payment_id=payment_id, at=feedback_at
        )


async def setup_part_c_history(connection: asyncpg.Connection) -> PartCFixture:
    """
    WHAT: Seeds one payer account with four days of history (payments, an account event, an ATM withdrawal,
    a feedback label, a complaint), relays it through Kafka into live Redis and runs the four pollers.

    WHY: One shared builder gives the part c checks and the latency benchmark the same realistic state
    that tests/integration/test_recompute_from_history.py uses, without importing test code.
    """
    config.decision_feature_logging_enabled = True
    bank_id = BANK_ID
    counterparty_bank_id = COUNTERPARTY_BANK_ID
    secret_text = f"test_secret_for_{bank_id}_32_chars_min!"
    setattr(config, f"bank_signing_secret_{bank_id}", SecretStr(secret_text))
    await issue_api_key(connection, bank_id)

    run_id = seeded_run_id(0)
    payer_acc = f"acc_c_{run_id[:8]}"
    payee_vpa = f"acc_payee_{run_id[:8]}@bank1"

    t0 = datetime(2026, 10, 25, 10, 0, 0, tzinfo=timezone.utc)
    t_day1 = t0
    t_day2 = t0 + timedelta(days=1)
    t_day3 = t0 + timedelta(days=2)
    t_dec = t_day3 + timedelta(hours=3)
    t_day4 = t0 + timedelta(days=3)

    # Pre-create daily feature log partition for decision date
    await connection.execute("SELECT create_feature_log_partition($1::date)", t_dec.date())

    topic = make_topic_name(prefix="sc15c")
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
            sender_device_fingerprint=BENCHMARK_PAYER_DEVICE,
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
            sender_device_fingerprint=BENCHMARK_PAYER_DEVICE,
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

        # 4. Day 4: Advancer payment side to push Kafka watermark past Day 3 + commit_grace
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

        await _seed_network_history(
            connection, bank_id=bank_id, run_id=run_id, payer_acc=payer_acc, payee_vpa=payee_vpa, start=t_day2
        )

        # Relay outbox to Kafka
        while True:
            batch_res = await relay_one_batch(connection, producer, batch_size=10, topic=topic)
            if batch_res.waiting == 0:
                break

        # Ingest into live Redis
        processed_payments = await run_payment_updater(
            live_redis,
            group_id=f"grp-{run_id}",
            idle_timeout_seconds=2.0,
            topic=topic,
        )
        if processed_payments < 4:
            raise RuntimeError(f"expected the updater to consume at least 4 payment sides, got {processed_payments}")

        # Run relational pollers
        await apply_account_events_since(connection, live_redis, bank_id=bank_id)
        await apply_withdrawals_since(connection, live_redis, bank_id=bank_id)
        await apply_feedback_since(connection, live_redis, bank_id=bank_id)
        await apply_complaints_since(connection, live_redis, bank_id=bank_id)

        screener = FakeRegistryScreener(
            table={
                (payer_acc, "account"): ScreenOutcome(status="no_match"),
                (payee_vpa, "account"): ScreenOutcome(status="no_match"),
            }
        )

        return PartCFixture(
            bank_id=bank_id,
            run_id=run_id,
            payer_acc=payer_acc,
            payee_vpa=payee_vpa,
            decision_time=t_dec,
            topic=topic,
            producer=producer,
            live_redis=live_redis,
            connection=connection,
            screener=screener,
        )
    except BaseException:
        # Setup failed halfway: release what was already opened, then let the original error surface.
        await producer.stop()
        await delete_topic(topic)
        live_redis.close()
        raise


# The checks run in order and share one seeded history, so it is built once and held here until 15c-5 cleans up.
_fixture: PartCFixture | None = None


async def get_part_c_fixture(workspace: Workspace) -> PartCFixture:
    """Return the shared seeded history, building it on first use."""
    global _fixture
    if _fixture is None:
        _fixture = await setup_part_c_history(workspace.connection)
    return _fixture


async def _stamp_arrival(connection: asyncpg.Connection, transaction_id: str, arrived_at: datetime) -> None:
    """Set when a stored payment side 'arrived' (transactions.created_at and the outbox recorded_at)."""
    await connection.execute(
        "UPDATE transactions SET created_at = $1 WHERE transaction_id = $2", arrived_at, uuid.UUID(transaction_id)
    )
    await connection.execute(
        "UPDATE outbox SET payload = jsonb_set(payload, '{recorded_at}', $1::jsonb) WHERE transaction_id = $2",
        f'"{arrived_at.isoformat()}"',
        uuid.UUID(transaction_id),
    )


async def _relay_all(fix: PartCFixture) -> None:
    """Publish every waiting outbox row to this fixture's Kafka topic."""
    while True:
        batch = await relay_one_batch(fix.connection, fix.producer, batch_size=10, topic=fix.topic)
        if batch.waiting == 0:
            return


async def check_15c_1_late_event_leaves_logged_unchanged(workspace: Workspace) -> CheckOutcome:
    """
    Check 15c-1 (edge): an event that arrives after a decision leaves its logged features unchanged.

    Logs a decision, records a late side for the same account and runs the updater. The feature_log row must be
    identical afterwards (values, arrival_cutoff, decided_at), while a fresh computation at the new watermark must
    differ, which proves the late event really reached Redis. Key number: count of changed values.
    """
    fix = await get_part_c_fixture(workspace)
    decision_pid = f"{fix.run_id}-dec-001"

    request = DecisionRequest(
        payment_id=decision_pid,
        sender_account_id=fix.payer_acc,
        receiver_bank_id=COUNTERPARTY_BANK_ID,
        receiver_vpa=fix.payee_vpa,
        payee_type="person",
        amount_paise=150000,
        event_time=fix.decision_time,
    )
    status_code, body = await record_decision(
        connection=fix.connection,
        bank_id=fix.bank_id,
        request=request,
        redis_client=fix.live_redis,
        decision_clock=ManualClock(fix.decision_time),
        screener=fix.screener,
    )
    assert status_code == 200
    assert body["decision"] == "unscored"

    row_before = await read_feature_log_row(fix.connection, fix.bank_id, decision_pid)
    assert row_before is not None
    assert row_before.status == "ok"
    assert row_before.arrival_cutoff is not None
    fix.decision_payment_id = decision_pid

    # A side that happened BEFORE the decision but was recorded an hour AFTER the decision's arrival cutoff.
    late_event_time = fix.decision_time - timedelta(minutes=15)
    late_arrival = row_before.arrival_cutoff + timedelta(hours=1)
    late_side = UpiPaymentSide(
        payment_id=f"{fix.run_id}-p-late",
        bank_id=fix.bank_id,
        side="debit",
        account_id=fix.payer_acc,
        counterparty_bank_id=COUNTERPARTY_BANK_ID,
        sender_bank_id=fix.bank_id,
        receiver_bank_id=COUNTERPARTY_BANK_ID,
        amount_paise=999999,
        event_time=late_event_time,
        status="success",
        counterparty_vpa="other@bank1",
        payee_type="person",
        channel="upi_app",
    )
    stored_late = await record_upi_side(fix.connection, late_side)
    await _stamp_arrival(fix.connection, stored_late.transaction_id, late_arrival)

    # An unrelated later side pushes the Kafka watermark past the late arrival, so the late side is ingested.
    advance_time = late_arrival + timedelta(hours=1)
    advance_side = UpiPaymentSide(
        payment_id=f"{fix.run_id}-p-adv-late",
        bank_id=fix.bank_id,
        side="debit",
        account_id="acc_unrelated_97",
        counterparty_bank_id=COUNTERPARTY_BANK_ID,
        sender_bank_id=fix.bank_id,
        receiver_bank_id=COUNTERPARTY_BANK_ID,
        amount_paise=10000,
        event_time=advance_time,
        status="success",
        counterparty_vpa="other@bank1",
        payee_type="person",
        channel="upi_app",
    )
    stored_advance = await record_upi_side(fix.connection, advance_side)
    await _stamp_arrival(fix.connection, stored_advance.transaction_id, advance_time)

    await _relay_all(fix)
    await run_payment_updater(
        fix.live_redis, group_id=f"grp-{fix.run_id}-late", idle_timeout_seconds=2.0, topic=fix.topic
    )
    for poller_source in ("events", "withdrawals", "feedback", "complaints"):
        record_poller_watermark(fix.live_redis, fix.bank_id, poller_source, advance_time)

    # (a) The logged row did not move.
    row_after = await read_feature_log_row(fix.connection, fix.bank_id, decision_pid)
    assert row_after is not None
    assert row_after.feature_values == row_before.feature_values
    assert row_after.arrival_cutoff == row_before.arrival_cutoff
    assert row_after.decided_at == row_before.decided_at
    assert row_after == row_before

    # (b) A fresh computation at the new watermark does differ: the late event really arrived.
    new_watermark = get_bank_watermark(fix.live_redis, fix.bank_id)
    assert new_watermark is not None
    fresh = compute_account_features(
        client=fix.live_redis,
        run_id=fix.run_id,
        bank_id=fix.bank_id,
        account_id=fix.payer_acc,
        decision_time=fix.decision_time,
        arrival_cutoff=new_watermark,
    )
    logged_by_name = dict(zip(canonical_feature_names(), row_before.feature_values))
    changed = [name for name, value in fresh.items() if name in logged_by_name and logged_by_name[name] != value]
    assert changed, "a fresh computation at the new watermark must differ after the late event arrived"

    return CheckOutcome(
        key_number=f"{len(changed)} changed values in the fresh computation, 0 in the logged row",
        evidence={
            "payment_id": decision_pid,
            "changed_feature_count": len(changed),
            "changed_features": changed,
            "logged_arrival_cutoff": row_before.arrival_cutoff.isoformat(),
            "new_watermark": new_watermark.isoformat(),
        },
    )


async def check_15c_2_recompute_exact_match(workspace: Workspace) -> CheckOutcome:
    """
    Check 15c-2 (consistency): a logged vector recomputed from history matches exactly.

    Replays the logged decision from Postgres history into a separate fakeredis. Key number: "52 of 52 equal, N
    non-None".
    """
    fix = await get_part_c_fixture(workspace)
    if fix.decision_payment_id is None:
        raise RuntimeError("15c-2 needs the decision that 15c-1 logs, and 15c-1 did not log one")

    log_row = await read_feature_log_row(fix.connection, fix.bank_id, fix.decision_payment_id)
    assert log_row is not None
    assert log_row.status == "ok"

    result = await recompute_logged_row(
        pg_connection=fix.connection,
        scratch_redis=fakeredis.FakeRedis(),
        bank_id=fix.bank_id,
        payment_id=fix.decision_payment_id,
    )
    assert result.mismatches == [], f"recompute differs from the logged row: {result.mismatches}"
    assert len(result.logged) == CANONICAL_FEATURE_COUNT, f"expected {CANONICAL_FEATURE_COUNT} features"
    assert result.logged == result.recomputed

    non_none = sum(1 for value in result.logged if value is not None)
    assert non_none >= 10, f"expected at least 10 non-None features, got {non_none}"

    return CheckOutcome(
        key_number=f"{len(result.logged)} of {CANONICAL_FEATURE_COUNT} equal, {non_none} non-None",
        evidence={
            "payment_id": fix.decision_payment_id,
            "mismatches": len(result.mismatches),
            "non_none": non_none,
        },
    )


def make_check_15c_3(quick: bool) -> Callable[[Workspace], Awaitable[CheckOutcome]]:
    """Build check 15c-3 for the chosen mode. Quick mode records the measurement; full mode judges the budget."""

    async def check_15c_3_decision_latency_against_budget(workspace: Workspace) -> CheckOutcome:
        """
        Check 15c-3 (normal): decision latency against the budget.

        Runs the benchmark's measurement in-process. Full mode asserts both p99 budgets read from
        config/evaluation.yaml. Quick mode (CI runners are not the budget machine) passes once measured and recorded.
        """
        fix = await get_part_c_fixture(workspace)
        budget = load_latency_budget()
        warmup, timed = (2, 5) if quick else (5, 300)

        if quick:
            # Quick mode (CI): the instant in-memory registry, measured and recorded, not judged.
            result = await measure_decision_latency(
                connection=fix.connection,
                redis_client=fix.live_redis,
                bank_id=fix.bank_id,
                run_id=fix.run_id,
                payer_acc=fix.payer_acc,
                payee_vpa=fix.payee_vpa,
                decision_time=fix.decision_time,
                screener=fix.screener,
                warmup_count=warmup,
                timed_count=timed,
            )
            delay_off_result = None
            key_number = (
                f"measured on {platform.platform()}, budget judged only on {budget['hardware']}: "
                f"decision p99={result.record_decision_p99_ms:.1f}ms, feature p99={result.compute_row_p99_ms:.1f}ms "
                f"(registry: in-memory fake, quick mode)"
            )
        else:
            # Full mode: the real mock registry over real HTTP. The run with the added registry delay is judged
            # against the budget; the run without the delay is measured and printed beside it.
            measured = {}
            for delay_on in (False, True):
                measured[delay_on] = await measure_against_mock_registry(
                    connection=fix.connection,
                    redis_client=fix.live_redis,
                    bank_id=fix.bank_id,
                    run_id=fix.run_id,
                    payer_acc=fix.payer_acc,
                    payee_vpa=fix.payee_vpa,
                    decision_time=fix.decision_time,
                    delay_on=delay_on,
                    warmup_count=warmup,
                    timed_count=timed,
                )
            delay_off_result, result = measured[False], measured[True]
            off_summary = (
                f"delay off: decision p99={delay_off_result.record_decision_p99_ms:.1f}ms, "
                f"feature p99={delay_off_result.compute_row_p99_ms:.1f}ms, "
                f"timeouts={delay_off_result.registry_timeouts}"
            )
            on_summary = (
                f"decision p99={result.record_decision_p99_ms:.1f}ms, feature p99={result.compute_row_p99_ms:.1f}ms, "
                f"timeouts={result.registry_timeouts}"
            )
            assert result.record_decision_p99_ms <= budget["max_ms"], (
                f"record_decision p99 {result.record_decision_p99_ms:.1f} ms is over the {budget['max_ms']} ms budget "
                f"(registry http, delay on; {off_summary})"
            )
            assert result.compute_row_p99_ms <= budget["feature_computation_max_ms"], (
                f"compute_decision_row p99 {result.compute_row_p99_ms:.1f} ms is over the "
                f"{budget['feature_computation_max_ms']} ms budget (registry http, delay on; {off_summary})"
            )
            key_number = f"registry http, delay on: {on_summary} within the budget; {off_summary}"

        return CheckOutcome(
            key_number=key_number,
            evidence={
                "mode": "quick" if quick else "full",
                "platform": result.platform_name,
                "warmup_count": result.warmup_count,
                "timed_count": result.timed_count,
                "budget": result.budget,
                "scope_note": result.scope_note,
                "registry": {
                    "mode": result.registry_mode,
                    "delay": result.registry_delay,
                    "questions_per_decision": result.registry_questions,
                    "timeouts_in_timed_calls": result.registry_timeouts,
                },
                "record_decision_ms": {
                    "p50": result.record_decision_p50_ms,
                    "p95": result.record_decision_p95_ms,
                    "p99": result.record_decision_p99_ms,
                    "max": result.record_decision_max_ms,
                    "histogram": result.record_decision_histogram,
                },
                "compute_decision_row_ms": {
                    "p50": result.compute_row_p50_ms,
                    "p95": result.compute_row_p95_ms,
                    "p99": result.compute_row_p99_ms,
                    "max": result.compute_row_max_ms,
                    "histogram": result.compute_row_histogram,
                },
                "delay_off_run": None
                if delay_off_result is None
                else {
                    "registry_timeouts": delay_off_result.registry_timeouts,
                    "record_decision_p50_p95_p99_ms": [
                        delay_off_result.record_decision_p50_ms,
                        delay_off_result.record_decision_p95_ms,
                        delay_off_result.record_decision_p99_ms,
                    ],
                    "compute_decision_row_p50_p95_p99_ms": [
                        delay_off_result.compute_row_p50_ms,
                        delay_off_result.compute_row_p95_ms,
                        delay_off_result.compute_row_p99_ms,
                    ],
                },
            },
        )

    return check_15c_3_decision_latency_against_budget


async def check_15c_4_feature_failure_unscored(workspace: Workspace) -> CheckOutcome:
    """
    Check 15c-4 (negative): a feature failure never changes the bank's answer.

    Forces compute_decision_row to raise (patched only for this call), posts a decision and expects the normal
    "unscored" answer plus a feature_log row with status 'error', an error_type and a NULL arrival_cutoff.
    """
    fix = await get_part_c_fixture(workspace)
    fix.error_run_id = seeded_run_id(1)
    error_pid = f"{fix.error_run_id}-dec-err"

    request = DecisionRequest(
        payment_id=error_pid,
        sender_account_id=fix.payer_acc,
        receiver_bank_id=COUNTERPARTY_BANK_ID,
        receiver_vpa=fix.payee_vpa,
        payee_type="person",
        amount_paise=10000,
        event_time=fix.decision_time,
    )
    with patch.object(
        bank_service, "compute_decision_row", side_effect=RuntimeError("forced showcase feature failure")
    ):
        status_code, body = await record_decision(
            connection=fix.connection,
            bank_id=fix.bank_id,
            request=request,
            redis_client=fix.live_redis,
            decision_clock=ManualClock(fix.decision_time),
            screener=fix.screener,
        )

    assert status_code == 200
    assert body["decision"] == "unscored"

    row = await read_feature_log_row(fix.connection, fix.bank_id, error_pid)
    assert row is not None
    assert row.status == "error"
    assert row.error_type == "RuntimeError"
    assert row.arrival_cutoff is None

    return CheckOutcome(
        key_number="answer still 'unscored'; logged row status=error, error_type=RuntimeError, arrival_cutoff NULL",
        evidence={
            "payment_id": error_pid,
            "response_decision": body["decision"],
            "row_status": row.status,
            "error_type": row.error_type,
            "error_message": row.error_message,
        },
    )


async def check_15c_5_no_unexpected_error_rows(workspace: Workspace) -> CheckOutcome:
    """
    Check 15c-5 (consistency): no unexpected error rows.

    After 15c-1..15c-3 there must be no feature_log row with status 'error'. 15c-4 deliberately makes one, under
    its own run id, so it is excluded by run id (not by time). Also releases the shared history (Kafka, Redis).
    """
    global _fixture
    fix = await get_part_c_fixture(workspace)
    try:
        if fix.error_run_id is None:
            raise RuntimeError("15c-5 needs the separate run id that 15c-4 uses, and 15c-4 did not set one")

        error_rows = await fix.connection.fetchval(
            "SELECT count(*) FROM feature_log WHERE status = 'error' AND run_id <> $1", fix.error_run_id
        )
        ok_rows = await fix.connection.fetchval(
            "SELECT count(*) FROM feature_log WHERE status = 'ok' AND run_id <> $1", fix.error_run_id
        )
        assert error_rows == 0, f"found {error_rows} unexpected feature_log error rows"
        assert ok_rows >= 1, "no logged decisions were found to check"
    finally:
        _fixture = None
        await fix.cleanup()

    return CheckOutcome(
        key_number=f"0 error rows among {ok_rows} logged decisions (15c-4 excluded by run id)",
        evidence={"error_rows": error_rows, "ok_rows": ok_rows, "excluded_run_id": fix.error_run_id},
    )
