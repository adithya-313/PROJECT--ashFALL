# File: tests/integration/test_registry_prefetch_http.py
# Purpose: The decision's registry screening against the real mock registry over real HTTP.
# Issue: Level 15 decision latency (registry round trips inside the feature budget).
"""
WHAT: Starts the real nationalsim app under uvicorn on a localhost port and points the real
NationalsimBankClient at it (real TCP, no ASGITransport, no instant fake). Checks that:
  - parallel: with a fixed added delay D on every call, three questions finish in under 2*D, not 3*D;
  - timeout: with a delay above the policy timeout the intelligence and payee features are None, every logged
    registry answer is 'unknown' with timed_out true, the timeout counter rose by the number of questions and the
    bank's answer is still "unscored";
  - completeness: PrefetchedScreener never meets an un-prefetched question across a run of decisions;
  - recompute: a row logged with timeouts recomputes with mismatches == [].

WHY: These are the architect's rulings for registry calls inside a decision: wait for the slowest call, never turn
a slow registry into "no match", and keep every decision exactly replayable.
"""

import asyncio
import time
from typing import Any

import fakeredis
import pytest

from api.v1.bank_service import record_decision
from api.v1.models import DecisionRequest
from benchmarks.decision_latency import (
    BENCHMARK_DPIP_KEY,
    benchmark_registry_entries,
    registry_timeouts_so_far,
    seed_registry_over_http,
    start_mock_registry,
)
from core.config import config
from core.event_clock import ManualClock
from features import decision_features
from features.decision_features import compute_decision_row
from features.feature_log import canonical_feature_names, read_feature_log_row
from features.feature_replay import recompute_logged_row
from features.ingest_watermark import get_bank_watermark
from features.registry_prefetch import PrefetchedScreener, run_registry_prefetch
from features.registry_screening import NationalsimRegistryScreener, ScreenOutcome
from nationalsim.app import ScreenDelay
from nationalsim.bank_client import NationalsimBankClient
from showcase.isolation import cleanup_workspace, create_workspace
from showcase.level_15_support_c import COUNTERPARTY_BANK_ID, setup_part_c_history

pytestmark = pytest.mark.needs_stack

RUN_ID = "66db9aa439b092ee"
DELAY_MS = 150.0
QUESTIONS = [("account", "a" * 64), ("device", "b" * 64), ("account", "c" * 64)]
REGISTRY_FEATURES = ("registry_match_count", "registry_match_age_days", "payee_registry_match")


def test_questions_are_asked_in_parallel_over_real_http() -> None:
    """Three calls that each take D ms finish in under 2*D ms; asked one by one they would take 3*D."""
    registry = start_mock_registry(RUN_ID, BENCHMARK_DPIP_KEY, [], ScreenDelay(DELAY_MS, DELAY_MS, seed=1))
    try:
        client = NationalsimBankClient(registry.base_url, bank_id="bank_0", run_id=RUN_ID)
        started = time.perf_counter()
        answers = run_registry_prefetch(client, QUESTIONS, timeout_seconds=5.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        registry.stop()

    assert set(answers) == set(QUESTIONS)
    assert all(not outcome.timed_out and outcome.status == "no_match" for outcome in answers.values())
    assert elapsed_ms >= DELAY_MS, "the registry delay must really have been applied"
    assert elapsed_ms < 2 * DELAY_MS, f"3 parallel calls of {DELAY_MS} ms took {elapsed_ms:.0f} ms (the sum is 3x)"


def _request(fix: Any, payment_id: str) -> DecisionRequest:
    return DecisionRequest(
        payment_id=payment_id,
        sender_account_id=fix.payer_acc,
        receiver_bank_id=COUNTERPARTY_BANK_ID,
        receiver_vpa=fix.payee_vpa,
        payee_type="person",
        amount_paise=150000,
        event_time=fix.decision_time,
    )


@pytest.mark.anyio
async def test_timeouts_give_unknown_are_counted_and_recompute_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry slower than the timeout: None features, unknown logged answers, counter up, row replays exactly."""
    monkeypatch.setattr(config, "decision_feature_logging_enabled", True)
    monkeypatch.setattr(config, "bank_signing_secret_bank_0", config.bank_signing_secret_bank_0, raising=False)

    workspace = await create_workspace()
    try:
        fix = await setup_part_c_history(workspace.connection)
        registry = start_mock_registry(
            fix.run_id,
            BENCHMARK_DPIP_KEY,
            benchmark_registry_entries(fix.run_id, fix.payee_vpa),
            ScreenDelay(DELAY_MS, DELAY_MS, seed=1),
        )
        try:
            client = NationalsimBankClient(registry.base_url, bank_id=fix.bank_id, run_id=fix.run_id)
            before = registry_timeouts_so_far()
            payment_id = f"{fix.run_id}-dec-timeout"
            status_code, body = await record_decision(
                connection=fix.connection,
                bank_id=fix.bank_id,
                request=_request(fix, payment_id),
                redis_client=fix.live_redis,
                decision_clock=ManualClock(fix.decision_time),
                screener=NationalsimRegistryScreener(client),
                dpip_key=BENCHMARK_DPIP_KEY,
            )
            assert status_code == 200
            assert body["decision"] == "unscored"

            row = await read_feature_log_row(fix.connection, fix.bank_id, payment_id)
            assert row is not None
            assert row.status == "ok"
            values = dict(zip(canonical_feature_names(), row.feature_values))
            for name in REGISTRY_FEATURES:
                assert values[name] is None, name

            # The features logged the questions they asked: the payer account (unknown, so the intelligence group
            # stopped before the device) and the payee. All 3 prefetched questions timed out and were counted.
            assert len(row.registry_answers) == 2
            for answer in row.registry_answers:
                assert answer["outcome"]["status"] == "unknown"
                assert answer["outcome"]["timed_out"] is True
            assert registry_timeouts_so_far() - before == 3

            result = await recompute_logged_row(
                pg_connection=fix.connection,
                scratch_redis=fakeredis.FakeRedis(),
                bank_id=fix.bank_id,
                payment_id=payment_id,
                dpip_key=BENCHMARK_DPIP_KEY,
            )
            assert result.mismatches == []
        finally:
            registry.stop()
            await fix.cleanup()
    finally:
        await cleanup_workspace(workspace)


@pytest.mark.anyio
async def test_prefetch_covers_every_question_the_features_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Across a run of decisions the features asked only prefetched questions (no LookupError), 3 per decision."""
    monkeypatch.setattr(config, "decision_feature_logging_enabled", True)
    monkeypatch.setattr(config, "bank_signing_secret_bank_0", config.bank_signing_secret_bank_0, raising=False)
    asked: list[tuple[str, str]] = []
    real_screen = PrefetchedScreener.screen

    def counting_screen(
        self: PrefetchedScreener, identifier_hash: str, identifier_type: str, as_of: Any = None
    ) -> ScreenOutcome:
        outcome = real_screen(self, identifier_hash, identifier_type, as_of)
        asked.append((identifier_type, identifier_hash))
        return outcome

    monkeypatch.setattr(PrefetchedScreener, "screen", counting_screen)
    # This test is about coverage, not timeouts, so give every call time to answer.
    monkeypatch.setattr(decision_features, "load_registry_timeout_seconds", lambda: 5.0)

    workspace = await create_workspace()
    try:
        fix = await setup_part_c_history(workspace.connection)
        registry = start_mock_registry(
            fix.run_id, BENCHMARK_DPIP_KEY, benchmark_registry_entries(fix.run_id, fix.payee_vpa), None
        )
        try:
            client = NationalsimBankClient(registry.base_url, bank_id=fix.bank_id, run_id=fix.run_id)
            screener = NationalsimRegistryScreener(client)
            clock = ManualClock(fix.decision_time)
            decisions = 10
            for i in range(decisions):
                row = await asyncio.to_thread(
                    compute_decision_row,
                    redis_client=fix.live_redis,
                    bank_id=fix.bank_id,
                    request=_request(fix, f"{fix.run_id}-dec-cover-{i:02d}"),
                    clock=clock,
                    screener=screener,
                    dpip_key=BENCHMARK_DPIP_KEY,
                )
                assert row.status == "ok"
                assert len(row.registry_answers) == 3
            assert get_bank_watermark(fix.live_redis, fix.bank_id) is not None
        finally:
            registry.stop()
            await fix.cleanup()
    finally:
        await cleanup_workspace(workspace)

    assert len(asked) == 3 * decisions, f"expected 3 answered questions per decision, got {len(asked)}"


def test_entries_seeded_over_http_are_screened_as_matches() -> None:
    """The container mode seeds a running registry through POST /ingest; those entries must then match."""
    registry = start_mock_registry(RUN_ID, BENCHMARK_DPIP_KEY, [], None)
    try:
        entries = benchmark_registry_entries(RUN_ID, "acc_payee_x@bank1")
        seed_registry_over_http(registry.base_url, RUN_ID, entries)
        registry.make_entries_visible()
        client = NationalsimBankClient(registry.base_url, bank_id="bank_0", run_id=RUN_ID)
        unreported = ("account", "e" * 64)
        answers = run_registry_prefetch(client, [*entries, unreported], timeout_seconds=5.0)
    finally:
        registry.stop()
    assert [answers[entry].status for entry in entries] == ["match", "match"]
    assert answers[unreported].status == "no_match"
