# File: tests/integration/test_network_label_cutoff.py
# Purpose: A label that arrives after a decision's arrival cutoff must not change that decision's network features.
# Issue: Level 15 decision latency (network features stay at decision time).
"""
WHAT: Logs a decision for a payer whose nearest flagged account is two hops away. Then a confirmed_fraud
feedback arrives, after the decision's arrival cutoff, for a same-bank neighbour of the payer (one hop away),
and the feedback poller applies it. The logged feature_log row must be unchanged and must recompute exactly
from history, while a fresh computation at the new watermark sees the closer flagged account.

WHY: The network features are computed at decision time from batched Redis reads. This proves the batching and
the shared per-decision reads never let information that arrived later leak into an already logged decision.
"""

from datetime import timedelta

import fakeredis
import pytest

from api.v1.bank_service import record_decision
from api.v1.models import DecisionRequest
from api.v1.upi_outbox import UpiPaymentSide, record_upi_side
from core.config import config
from core.event_clock import ManualClock
from features.account_labels import apply_feedback_since, labelled_accounts_visible
from features.account_stream import run_payment_updater
from features.feature_log import canonical_feature_names, read_feature_log_row
from features.feature_replay import recompute_logged_row
from features.ingest_watermark import get_bank_watermark, record_poller_watermark
from features.m1_features import compute_account_features
from showcase.isolation import cleanup_workspace, create_workspace
from showcase.level_15_support_c import COUNTERPARTY_BANK_ID, _relay_all, _stamp_arrival, setup_part_c_history

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

NEIGHBOUR_NUMBER = 3


async def test_label_arriving_after_the_cutoff_leaves_the_decision_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logged row unchanged and exactly recomputable; a fresh computation at the new watermark sees the label."""
    # The shared history builder switches these on directly; monkeypatch restores them after the test.
    monkeypatch.setattr(config, "decision_feature_logging_enabled", True)
    monkeypatch.setattr(config, "bank_signing_secret_bank_0", config.bank_signing_secret_bank_0, raising=False)

    workspace = await create_workspace()
    try:
        fix = await setup_part_c_history(workspace.connection)
        try:
            decision_pid = f"{fix.run_id}-dec-label"
            # Decide a little after the bank watermark, so the arrival cutoff (the watermark), not the decision
            # time, is what limits which labels the decision may see.
            watermark = get_bank_watermark(fix.live_redis, fix.bank_id)
            assert watermark is not None
            decision_time = watermark + timedelta(hours=3)
            status_code, body = await record_decision(
                connection=fix.connection,
                bank_id=fix.bank_id,
                request=DecisionRequest(
                    payment_id=decision_pid,
                    sender_account_id=fix.payer_acc,
                    receiver_bank_id=COUNTERPARTY_BANK_ID,
                    receiver_vpa=fix.payee_vpa,
                    payee_type="person",
                    amount_paise=150000,
                    event_time=decision_time,
                ),
                redis_client=fix.live_redis,
                decision_clock=ManualClock(decision_time),
                screener=fix.screener,
            )
            assert status_code == 200
            assert body["decision"] == "unscored"

            row_before = await read_feature_log_row(fix.connection, fix.bank_id, decision_pid)
            assert row_before is not None
            assert row_before.status == "ok"
            assert row_before.arrival_cutoff is not None
            assert row_before.arrival_cutoff < decision_time
            logged_hops = dict(zip(canonical_feature_names(), row_before.feature_values))["hops_to_confirmed_label"]
            assert logged_hops == 2, "the seeded payer should be two hops from the nearest flagged account"

            # A confirmed_fraud verdict on the payment that paid the payer's first same-bank neighbour. It happened
            # before the decision, but was recorded an hour AFTER the logged decision's arrival cutoff.
            neighbour = f"acc_nb_{fix.run_id[:8]}_{NEIGHBOUR_NUMBER:02d}"
            label_available_at = row_before.arrival_cutoff + timedelta(hours=1)
            await fix.connection.execute(
                """
                INSERT INTO bank_feedback (bank_id, feedback_id, payment_id, outcome, event_time, created_at)
                VALUES ($1, $2, $3, 'confirmed_fraud', $4, $5)
                """,
                fix.bank_id,
                f"{fix.run_id}-fb-late-label",
                f"{fix.run_id}-p-nb{NEIGHBOUR_NUMBER:02d}",
                row_before.arrival_cutoff + timedelta(minutes=30),
                label_available_at,
            )
            await apply_feedback_since(fix.connection, fix.live_redis, bank_id=fix.bank_id)

            # An unrelated later payment pushes the Kafka watermark past the label, as in check 15c-1.
            advance_time = label_available_at + timedelta(hours=1)
            advance_side = UpiPaymentSide(
                payment_id=f"{fix.run_id}-p-adv-label",
                bank_id=fix.bank_id,
                side="debit",
                account_id="acc_unrelated_96",
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
                fix.live_redis, group_id=f"grp-{fix.run_id}-label", idle_timeout_seconds=2.0, topic=fix.topic
            )
            for poller_source in ("events", "withdrawals", "feedback", "complaints"):
                record_poller_watermark(fix.live_redis, fix.bank_id, poller_source, advance_time)

            new_watermark = get_bank_watermark(fix.live_redis, fix.bank_id)
            assert new_watermark is not None
            assert new_watermark >= label_available_at
            assert neighbour in labelled_accounts_visible(
                fix.live_redis, fix.run_id, fix.bank_id, decision_time, new_watermark
            ), "the late label must really be visible at the new watermark"
            assert neighbour not in labelled_accounts_visible(
                fix.live_redis, fix.run_id, fix.bank_id, decision_time, row_before.arrival_cutoff
            ), "the late label must not be visible at the logged decision's arrival cutoff"

            # (a) The logged row did not move.
            row_after = await read_feature_log_row(fix.connection, fix.bank_id, decision_pid)
            assert row_after == row_before

            # (b) The logged row recomputes exactly from history: the late label is not in the replay either.
            result = await recompute_logged_row(
                pg_connection=fix.connection,
                scratch_redis=fakeredis.FakeRedis(),
                bank_id=fix.bank_id,
                payment_id=decision_pid,
            )
            assert result.mismatches == []

            # (c) A fresh computation at the new watermark DOES see the flagged neighbour, one hop away.
            fresh = compute_account_features(
                client=fix.live_redis,
                run_id=fix.run_id,
                bank_id=fix.bank_id,
                account_id=fix.payer_acc,
                decision_time=decision_time,
                arrival_cutoff=new_watermark,
                groups=("network",),
            )
            assert fresh["hops_to_confirmed_label"] == 1
            assert fresh["hops_to_confirmed_label"] != logged_hops
        finally:
            await fix.cleanup()
    finally:
        await cleanup_workspace(workspace)
