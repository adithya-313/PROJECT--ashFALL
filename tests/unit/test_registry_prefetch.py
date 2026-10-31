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
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import nationalsim.app as nationalsim_app
from core.event_clock import IST, ManualClock
from features import registry_prefetch
from features.account_keys import load_account_features_settings
from features.account_records import PaymentSideRecord
from features.account_updater import apply_payment_side
from features.registry_prefetch import (
    PrefetchedScreener,
    compute_registry_timeout_ms,
    list_registry_questions,
    load_registry_timeout_seconds,
    prefetch_registry_answers,
)
from features.registry_screening import ScreenOutcome
from nationalsim.app import ScreenDelay, create_app, load_screen_delay_settings
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
    """The repository config says 33.19 ms (from the A2 formula); a missing or bad value is an error, not a default."""
    assert load_registry_timeout_seconds() == pytest.approx(0.03319)


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


# --- the mock registry's optional added /screen delay (off by default, seeded when on) ---

SCREEN_PAYLOAD = {
    "requesting_bank_id": "bank_0",
    "identifier_hash": "a" * 64,
    "identifier_type": "account",
    "run_id": "run_delay",
}
DELAY_APP_SETTINGS = {"listing_delay_minutes": 15, "banks": {"bank_0": {"dpip_participation": True}}}


def _screen_app(screen_delay: ScreenDelay | None) -> TestClient:
    app = create_app(
        clock=ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST)),
        key_provider=lambda: "test_key_secret",
        settings=DELAY_APP_SETTINGS,
        screen_delay=screen_delay,
    )
    return TestClient(app)


def test_screen_adds_no_delay_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a ScreenDelay the /screen route never sleeps and answers as before."""
    sleeps: list[float] = []
    monkeypatch.setattr(nationalsim_app.time, "sleep", sleeps.append)
    response = _screen_app(None).post("/screen", json=SCREEN_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["match"] is False
    assert sleeps == []


def test_screen_delay_is_drawn_from_a_seeded_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a ScreenDelay each request sleeps for the next drawn value, and the same seed repeats the sequence."""
    sleeps: list[float] = []
    monkeypatch.setattr(nationalsim_app.time, "sleep", sleeps.append)
    client = _screen_app(ScreenDelay(10, 30, seed=7))
    for _ in range(5):
        assert client.post("/screen", json=SCREEN_PAYLOAD).status_code == 200

    expected = ScreenDelay(10, 30, seed=7)
    assert sleeps == [expected.next_seconds() for _ in range(5)]
    assert all(0.010 <= seconds <= 0.030 for seconds in sleeps)
    other = ScreenDelay(10, 30, seed=8)
    assert [other.next_seconds() for _ in range(5)] != sleeps


def test_screen_delay_rejects_an_invalid_range() -> None:
    with pytest.raises(ValueError, match="Invalid screen delay range"):
        ScreenDelay(30, 10, seed=1)
    with pytest.raises(ValueError, match="Invalid screen delay range"):
        ScreenDelay(-1, 10, seed=1)


def test_screen_delay_settings_come_from_configuration(tmp_path: Path) -> None:
    """The repository config gives the range and seed; a missing or bad block is an error, not a default."""
    settings = load_screen_delay_settings()
    assert (settings.min_ms, settings.max_ms) == (2.0, 15.0)
    assert settings.build().next_seconds() == ScreenDelay(2, 15, settings.seed).next_seconds()
    slow = load_screen_delay_settings(profile="registry-slow")
    assert (slow.min_ms, slow.max_ms) == (15.0, 60.0)
    assert slow.seed != settings.seed
    with pytest.raises(ValueError, match="Unknown screen delay profile"):
        load_screen_delay_settings(profile="fast")

    missing = tmp_path / "missing.yaml"
    missing.write_text("nationalsim:\n  listing_delay_minutes: 15\n", encoding="utf-8")
    with pytest.raises(ValueError, match="screen_added_delay_ms"):
        load_screen_delay_settings(missing)
    backwards = tmp_path / "backwards.yaml"
    backwards.write_text(
        "nationalsim:\n  screen_added_delay_ms:\n    min: 30\n    max: 10\n    seed: 1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="max is below min"):
        load_screen_delay_settings(backwards)


# --- honest timeout: never cut a call before its real timeout, on any clock ---

COARSE_TICK = 0.015625  # the Windows monotonic clock tick


class _CoarseClockLoop(asyncio.SelectorEventLoop):
    """
    An event loop that behaves like the one on Windows: its clock only moves in 15.6 ms steps AND, as asyncio does
    with the platform's clock resolution, it treats any timer due within one tick as ready. Together timers fire
    up to two ticks early. (Setting the resolution makes this hold on Linux too, where the real one is 1 ns.)
    """

    def __init__(self) -> None:
        super().__init__()
        self._clock_resolution = COARSE_TICK

    def time(self) -> float:
        return math.floor(super().time() / COARSE_TICK) * COARSE_TICK


class _RealDurationClient:
    """screen() that really takes `seconds` (a worker thread sleeps), whatever the event loop's clock says."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    async def screen(
        self, identifier_hash: str, identifier_type: str = "account", as_of: datetime | None = None
    ) -> ScreenResult:
        await asyncio.get_running_loop().run_in_executor(None, time.sleep, self.seconds)
        return _result("no_match")


def _timed_out_count_on_coarse_clock(repetitions: int, timeout_seconds: float, call_seconds: float) -> int:
    timed_out = 0
    with asyncio.Runner(loop_factory=_CoarseClockLoop) as runner:
        for _ in range(repetitions):
            answers = runner.run(
                prefetch_registry_answers(_RealDurationClient(call_seconds), [QUESTIONS[0]], timeout_seconds)
            )
            timed_out += sum(1 for outcome in answers.values() if outcome.timed_out)
    return timed_out


def test_a_call_is_never_cut_before_its_real_timeout_on_a_coarse_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a clock that ticks every 15.6 ms, an 8 ms call with a 20 ms timeout is never marked timed out."""
    monkeypatch.setattr(registry_prefetch, "clock_resolution_seconds", lambda: COARSE_TICK)
    assert _timed_out_count_on_coarse_clock(200, 0.020, 0.008) == 0


def test_the_wait_is_the_timeout_plus_two_clock_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deterministic form of the guard: wait_for is given timeout + 2 ticks, so an early timer cannot cut a call."""
    given: list[float] = []
    real_wait_for = asyncio.wait_for

    async def recording_wait_for(awaitable: Any, timeout: float) -> Any:
        given.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(registry_prefetch.asyncio, "wait_for", recording_wait_for)
    monkeypatch.setattr(registry_prefetch, "clock_resolution_seconds", lambda: COARSE_TICK)
    asyncio.run(prefetch_registry_answers(_RealDurationClient(0.001), [QUESTIONS[0]], 0.020))
    assert given == [pytest.approx(0.020 + 2 * COARSE_TICK)]


def test_an_answer_that_arrives_after_the_timeout_is_unknown_and_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inside the slack the answer is returned, but 30 ms is over the 20 ms timeout, so it is 'unknown'."""
    monkeypatch.setattr(registry_prefetch, "clock_resolution_seconds", lambda: 0.050)
    before = _timeouts("account")
    answers = asyncio.run(prefetch_registry_answers(_RealDurationClient(0.030), [QUESTIONS[0]], 0.020))
    assert answers[QUESTIONS[0]] == ScreenOutcome(status="unknown", first_visible_at=None, timed_out=True)
    assert _timeouts("account") - before == 1


def test_a_real_5_ms_call_is_never_timed_out_with_the_production_timeout() -> None:
    """Native clock, production timeout from config: 200 calls that really take 5 ms are all answered."""
    timeout_seconds = load_registry_timeout_seconds()
    timed_out = 0
    for _ in range(200):
        answers = asyncio.run(prefetch_registry_answers(_RealDurationClient(0.005), [QUESTIONS[0]], timeout_seconds))
        timed_out += sum(1 for outcome in answers.values() if outcome.timed_out)
    assert timed_out == 0


# --- the timeout formula ---


def test_registry_timeout_formula_from_measured_inputs() -> None:
    """100 - 50 - 8.4 (overhead p99) - 10 (margin) - 0.000001 (clock) = 31.599999 ms."""
    value = compute_registry_timeout_ms(100.0, 50.0, 8.4, 10.0, 0.000001, minimum_ms=5.0)
    assert value == pytest.approx(31.599999)


def test_registry_timeout_below_the_minimum_is_an_error_not_a_value() -> None:
    """The stop rule: a result under 5 ms raises; exactly 5 ms is allowed."""
    assert compute_registry_timeout_ms(100.0, 50.0, 25.0, 10.0, 10.0, minimum_ms=5.0) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="below the 5.0 ms minimum"):
        compute_registry_timeout_ms(100.0, 50.0, 25.0, 10.0, 15.6, minimum_ms=5.0)
