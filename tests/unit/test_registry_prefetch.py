# File: tests/unit/test_registry_prefetch.py
# Purpose: Unit tests for the parallel registry prefetch, its timeout policy and its screener.
# Issue: Level 15 decision latency (registry round trips inside the feature budget).
"""
WHAT: Checks, with a stub async client, that every question is asked together, that a slow call becomes
'unknown' with timed_out set and is counted (never 'no_match'), that the client's 'unavailable' becomes
'unknown', that other errors are not swallowed, that PrefetchedScreener refuses an un-prefetched question,
that the questions are listed in order without duplicates, and that the timeout comes from configuration.

WHY: A registry that is slow or down must degrade a feature to None, never to a false "no match", and the
decision must wait for the slowest call, not the sum of all of them.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fakeredis
import pytest
from prometheus_client import REGISTRY

from features.account_keys import load_account_features_settings
from features.account_records import PaymentSideRecord
from features.account_updater import apply_payment_side
from features.registry_prefetch import (
    PrefetchedScreener,
    list_registry_questions,
    load_registry_timeout_seconds,
    prefetch_registry_answers,
)
from features.registry_screening import ScreenOutcome
from nationalsim.bank_client import ScreenResult

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
DPIP_KEY = "test_dpip_key_32_bytes_long_exact"
NOW = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
FIRST_VISIBLE = datetime(2026, 10, 20, 9, 0, 0, tzinfo=timezone.utc)


def _timeouts(identifier_type: str) -> float:
    value = REGISTRY.get_sample_value("ashfall_registry_timeouts_total", {"identifier_type": identifier_type})
    return 0.0 if value is None else value


class _StubClient:
    """An async screen() that takes `delay` seconds and returns a fixed ScreenResult (or raises)."""

    def __init__(self, delay: float, result: ScreenResult | None = None, error: Exception | None = None) -> None:
        self.delay = delay
        self.result = result
        self.error = error
        self.asked: list[tuple[str, str, datetime | None]] = []

    async def screen(
        self, identifier_hash: str, identifier_type: str = "account", as_of: datetime | None = None
    ) -> ScreenResult:
        self.asked.append((identifier_hash, identifier_type, as_of))
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _result(status: str) -> ScreenResult:
    if status == "unavailable":
        return ScreenResult(
            status="unavailable",
            match=None,
            risk_score=None,
            distinct_accounts_linked=None,
            is_unavailable=True,
            first_visible_at=None,
        )
    matched = status == "match"
    return ScreenResult(
        status=status,
        match=matched,
        risk_score=None,
        distinct_accounts_linked=None,
        is_unavailable=False,
        first_visible_at=FIRST_VISIBLE if matched else None,
    )


QUESTIONS = [("account", "a" * 64), ("device", "b" * 64), ("account", "c" * 64)]


def test_answers_are_mapped_and_asked_live() -> None:
    """match / no_match / unavailable map to match / no_match / unknown, and as_of is never sent."""
    for status, expected, visible in (
        ("match", "match", FIRST_VISIBLE),
        ("no_match", "no_match", None),
        ("unavailable", "unknown", None),
    ):
        client = _StubClient(0.0, _result(status))
        answers = asyncio.run(prefetch_registry_answers(client, QUESTIONS, timeout_seconds=1.0))
        assert set(answers) == set(QUESTIONS)
        for outcome in answers.values():
            assert outcome == ScreenOutcome(status=expected, first_visible_at=visible, timed_out=False)
        assert all(as_of is None for _, _, as_of in client.asked)


def test_a_slow_call_is_unknown_timed_out_and_counted() -> None:
    """A call slower than the timeout is 'unknown' with timed_out (never 'no_match'), one count per question."""
    before = {"account": _timeouts("account"), "device": _timeouts("device")}
    client = _StubClient(0.5, _result("no_match"))
    answers = asyncio.run(prefetch_registry_answers(client, QUESTIONS, timeout_seconds=0.02))
    for outcome in answers.values():
        assert outcome == ScreenOutcome(status="unknown", first_visible_at=None, timed_out=True)
    assert _timeouts("account") - before["account"] == 2
    assert _timeouts("device") - before["device"] == 1


def test_a_fast_call_is_not_counted_as_a_timeout() -> None:
    """Within the timeout nothing is flagged and the counter does not move."""
    before = _timeouts("account")
    client = _StubClient(0.001, _result("match"))
    answers = asyncio.run(prefetch_registry_answers(client, QUESTIONS, timeout_seconds=1.0))
    assert not any(outcome.timed_out for outcome in answers.values())
    assert _timeouts("account") == before


def test_questions_are_asked_together_not_one_by_one() -> None:
    """Three calls of 0.15 s finish in about 0.15 s, not the 0.45 s of the sum."""
    client = _StubClient(0.15, _result("no_match"))
    started = time.perf_counter()
    asyncio.run(prefetch_registry_answers(client, QUESTIONS, timeout_seconds=1.0))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.30


def test_other_errors_are_not_swallowed() -> None:
    """Only a timeout is turned into 'unknown'; any other exception reaches the caller."""
    client = _StubClient(0.0, error=RuntimeError("registry client bug"))
    with pytest.raises(RuntimeError, match="registry client bug"):
        asyncio.run(prefetch_registry_answers(client, QUESTIONS, timeout_seconds=1.0))


def test_prefetched_screener_answers_and_refuses_unknown_questions() -> None:
    """It returns the prefetched outcome whatever as_of is passed, and raises LookupError for another question."""
    outcome = ScreenOutcome(status="match", first_visible_at=FIRST_VISIBLE)
    screener = PrefetchedScreener({("account", "a" * 64): outcome})
    assert screener.screen("a" * 64, "account", as_of=NOW) is outcome
    assert screener.screen("a" * 64, "account") is outcome
    with pytest.raises(LookupError, match="not prefetched"):
        screener.screen("a" * 64, "device")
    with pytest.raises(LookupError, match="not prefetched"):
        screener.screen("z" * 64, "account")


def test_timeout_comes_from_configuration() -> None:
    """The repository config says 20 ms; a missing or non-positive value is an error, not a default."""
    assert load_registry_timeout_seconds() == pytest.approx(0.020)


def test_timeout_config_errors_are_loud(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    missing.write_text("account_features: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing decision_registry.timeout_ms.value"):
        load_registry_timeout_seconds(missing)
    zero = tmp_path / "zero.yaml"
    zero.write_text("decision_registry:\n  timeout_ms:\n    value: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a positive number"):
        load_registry_timeout_seconds(zero)


def _debit(account: str, number: int, fingerprint: str | None) -> PaymentSideRecord:
    return PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payment_id=f"{RUN_ID}-p-{number:03d}",
        account_id=account,
        side="debit",
        counterparty_vpa="acc_payee_1",
        counterparty_bank_id="bank_1",
        amount_paise=10000,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=fingerprint,
        sender_os_generation=2 if fingerprint else None,
        event_time=NOW - timedelta(hours=number),
        arrival_time=NOW - timedelta(hours=number) + timedelta(seconds=1),
        balance_after_paise=900000,
    )


def test_questions_are_listed_in_order_without_duplicates() -> None:
    """Payer account, each visible device (sorted), then the payee; a repeated hash is asked once."""
    settings = load_account_features_settings()
    client = fakeredis.FakeRedis()
    apply_payment_side(client, _debit("acc_payer", 1, "fp_b"), settings)
    apply_payment_side(client, _debit("acc_payer", 2, "fp_a"), settings)

    questions = list_registry_questions(
        client, RUN_ID, BANK_0, "acc_payer", "bank_1", "acc_payee_1", NOW, NOW, DPIP_KEY, settings
    )
    assert [kind for kind, _ in questions] == ["account", "device", "device", "account"]
    assert len({digest for _, digest in questions}) == 4

    repeated = list_registry_questions(
        client, RUN_ID, BANK_0, "acc_payer", BANK_0, "acc_payer", NOW, NOW, DPIP_KEY, settings
    )
    assert [kind for kind, _ in repeated] == ["account", "device", "device"]

    no_payee = list_registry_questions(
        client, RUN_ID, BANK_0, "acc_payer", "bank_1", None, NOW, NOW, DPIP_KEY, settings
    )
    assert [kind for kind, _ in no_payee] == ["account", "device", "device"]


def test_no_questions_without_a_key() -> None:
    """Without a DPIP key the features are unknown and nothing is asked."""
    settings = load_account_features_settings()
    client = fakeredis.FakeRedis()
    # config.dpip_participant_key may be unset in the test environment; an explicit empty key behaves the same.
    assert list_registry_questions(client, RUN_ID, BANK_0, "acc_payer", "bank_1", "p", NOW, NOW, "", settings) == []
