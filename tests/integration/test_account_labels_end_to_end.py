# File: tests/integration/test_account_labels_end_to_end.py
# Purpose: End-to-end integration test proving confirmed fraud labels, complaints, and M1 network features.
# Issue: Level 15b M1 network features (plan Task 1, G1, R8, architect decision 9b).
"""
WHAT: Full-stack integration test connecting PostgreSQL (transactions, bank_feedback, bank_complaints)
and Redis (labelled_accounts, complained_accounts, payment graph edges).

WHY: Proves end-to-end correctness across relational feedback/complaint streaming,
point-in-time lifetime label visibility, tenant isolation, and graph traversal BFS.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import redis

from core.config import config
from features.account_keys import (
    load_account_features_settings,
    namespace_pattern,
)
from features.account_labels import (
    apply_complaints_since,
    apply_feedback_since,
    complained_accounts_visible,
    labelled_accounts_visible,
)
from features.account_reader import PaymentSideRecord
from features.account_updater import apply_payment_side
from features.m1_features import compute_account_features

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


def host_side_redis_url() -> str:
    """
    Resolve the Redis connection string for tests running on the host machine.

    The .env Redis address uses the Docker service name ('redis'), which only resolves
    inside Docker containers. On the host runner, this swaps 'redis' for 'localhost'.
    """
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


async def test_account_labels_and_network_features_end_to_end(isolated_schema: tuple[Any, str]) -> None:
    """
    WHAT: End-to-end scenario proving feedback polling, complaint ingestion, and network BFS hops.

    SCENARIO:
      1. Inserts cross-bank transaction into PostgreSQL transactions:
         - Debit leg on victim account at bank_1.
         - Credit leg on mule account at bank_0.
      2. Inserts 'confirmed_fraud' feedback for bank_0 matching the payment ID.
      3. Inserts complaint for bank_0 against a reported account.
      4. Ingests feedback and complaints into Redis via apply_feedback_since and apply_complaints_since.
      5. Asserts labelled_accounts_visible contains only the credit-side mule (victim is NOT labelled).
      6. Asserts complained_accounts_visible contains the reported account.
      7. Establishes same-bank payment edge in Redis between peer account and mule account at bank_0.
      8. Calls compute_account_features for peer account and asserts hops_to_confirmed_label == 1.
    """
    connection, _ = isolated_schema
    redis_client = redis.Redis.from_url(host_side_redis_url())
    settings = load_account_features_settings()

    run_id = uuid.uuid4().hex[:16]
    bank_0 = "bank_0"
    bank_1 = "bank_1"
    acc_mule = "acc_mule_01"
    acc_victim = "acc_victim_02"
    acc_complained = "acc_comp_03"
    acc_peer = "acc_peer_04"

    now = datetime.now(timezone.utc)
    t_payment = now - timedelta(hours=2)
    t_feedback = now - timedelta(hours=1)
    t_complaint = now - timedelta(minutes=45)
    payment_id = f"{run_id}-p-fraud-01"

    try:
        # 1. Insert cross-bank payment's two sides into PostgreSQL transactions
        # Debit side on victim at bank_1
        await connection.execute(
            """
            INSERT INTO transactions (
                source, account_id, amount, payment_id, bank_id, side,
                counterparty_bank_id, counterparty_vpa, amount_paise, event_time, status
            ) VALUES ('world', $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            acc_victim,
            1000.0,
            payment_id,
            bank_1,
            "debit",
            bank_0,
            acc_mule,
            100000,
            t_payment,
            "success",
        )
        # Credit side on mule at bank_0
        await connection.execute(
            """
            INSERT INTO transactions (
                source, account_id, amount, payment_id, bank_id, side,
                counterparty_bank_id, counterparty_vpa, amount_paise, event_time, status
            ) VALUES ('world', $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            acc_mule,
            1000.0,
            payment_id,
            bank_0,
            "credit",
            bank_1,
            acc_victim,
            100000,
            t_payment,
            "success",
        )

        # 2. Insert confirmed_fraud feedback into bank_feedback for bank_0
        fb_id = f"{run_id}-fb-0001"
        await connection.execute(
            """
            INSERT INTO bank_feedback (
                bank_id, feedback_id, payment_id, outcome, event_time, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            bank_0,
            fb_id,
            payment_id,
            "confirmed_fraud",
            t_feedback,
            t_feedback,
        )

        # 3. Insert complaint into bank_complaints for bank_0
        comp_id = f"{run_id}-comp-0001"
        await connection.execute(
            """
            INSERT INTO bank_complaints (
                bank_id, complaint_id, complainant_account, reported_account, reported_bank_id,
                payment_ids, claimed_type, received_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            bank_0,
            comp_id,
            "acc_complainant_99",
            acc_complained,
            bank_0,
            [payment_id],
            "fraud",
            t_complaint,
            t_complaint,
        )

        # 4. Ingest relational feedback and complaints into Redis via pollers
        processed_fb = await apply_feedback_since(
            connection, redis_client, settings=settings, bank_id=bank_0
        )
        assert processed_fb > 0

        processed_comp = await apply_complaints_since(
            connection, redis_client, settings=settings, bank_id=bank_0
        )
        assert processed_comp > 0

        # 5. Assert labelled_accounts_visible contains acc_mule only
        decision_time = now
        labelled = labelled_accounts_visible(
            redis_client, run_id, bank_0, decision_time, decision_time
        )
        assert labelled == {acc_mule}
        assert acc_victim not in labelled

        # 6. Assert complained_accounts_visible contains acc_complained
        complained = complained_accounts_visible(
            redis_client, run_id, bank_0, decision_time, decision_time
        )
        assert complained == {acc_complained}

        # 7. Setup a same-bank transaction in Redis between acc_peer and acc_mule (via apply_payment_side)
        t_peer = now - timedelta(minutes=30)
        p_peer_deb = PaymentSideRecord(
            run_id=run_id,
            bank_id=bank_0,
            account_id=acc_peer,
            payment_id=f"{run_id}-p-peer-deb",
            side="debit",
            counterparty_bank_id=bank_0,
            counterparty_vpa=acc_mule,
            amount_paise=50000,
            status="success",
            failure_domain=None,
            payee_type="person",
            channel="upi_app",
            sender_device_fingerprint=None,
            sender_os_generation=None,
            event_time=t_peer,
            arrival_time=t_peer,
            balance_after_paise=950000,
        )
        p_mule_crd = PaymentSideRecord(
            run_id=run_id,
            bank_id=bank_0,
            account_id=acc_mule,
            payment_id=f"{run_id}-p-peer-crd",
            side="credit",
            counterparty_bank_id=bank_0,
            counterparty_vpa=acc_peer,
            amount_paise=50000,
            status="success",
            failure_domain=None,
            payee_type="person",
            channel="upi_app",
            sender_device_fingerprint=None,
            sender_os_generation=None,
            event_time=t_peer,
            arrival_time=t_peer,
            balance_after_paise=1050000,
        )
        apply_payment_side(redis_client, p_peer_deb, settings)
        apply_payment_side(redis_client, p_mule_crd, settings)

        # 8. Compute network features for acc_peer and assert hops_to_confirmed_label == 1
        feats = compute_account_features(
            client=redis_client,
            run_id=run_id,
            bank_id=bank_0,
            account_id=acc_peer,
            decision_time=decision_time,
            arrival_cutoff=decision_time,
            groups=("network",),
            settings=settings,
        )
        assert feats["hops_to_confirmed_label"] == 1
        assert feats["hops_to_complained_account"] is None

    finally:
        # Clean up Redis keys under run namespace
        for pat in (namespace_pattern(run_id, bank_0), namespace_pattern(run_id, bank_1)):
            for k in redis_client.scan_iter(match=pat):
                redis_client.delete(k)
        redis_client.close()

