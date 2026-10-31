# File: features/registry_prefetch.py
# Purpose: Ask the national registry every question of one decision at once, with a per-call timeout.
# Issue: Level 15 decision latency (registry round trips inside the feature budget).
"""
WHAT: list_registry_questions() lists the registry questions one decision will ask (the payer account, each
visible lifetime device fingerprint of the payer, and the payee account) using the same hash function and
visibility rules as the features. prefetch_registry_answers() asks them all together and waits for the
slowest, not the sum. PrefetchedScreener hands the answers to the unchanged feature code.

WHY: Screening one question at a time made the registry cost the sum of its calls. A call that takes longer than
the policy timeout is answered "unknown" (never "no match"), flagged timed_out, and counted in
ashfall_registry_timeouts_total, so a slow registry degrades a feature to None instead of stalling the decision.
There is deliberately no answer cache here: caching registry answers belongs to Level 19.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
from prometheus_client import Counter

from api.v1.models import DecisionRequest
from core.config import config
from core.config_validator import load_yaml
from features.account_keys import AccountFeaturesSettings, extract_run_id, get_default_settings
from features.account_summaries import read_lifetime
from features.ingest_watermark import get_bank_watermark
from features.m1_intelligence import get_bank_index
from features.registry_screening import ScreenOutcome, outcome_from_screen_result
from nationalsim.bank_client import (
    NationalsimBankClient,
    ScreenResult,
    create_shared_http_client,
    load_bank_client_config,
    load_registry_pool_settings,
)
from nationalsim.keyed_hash import compute_identifier_hash

__all__ = [
    "RegistryGateway",
    "create_registry_gateway",
    "prepare_registry_questions",
    "REGISTRY_TIMEOUTS_TOTAL",
    "clock_resolution_seconds",
    "compute_registry_timeout_ms",
    "PrefetchedScreener",
    "RegistryQuestion",
    "list_registry_questions",
    "load_registry_timeout_seconds",
    "prefetch_registry_answers",
    "run_registry_prefetch",
]

# (identifier_type, identifier_hash): the key the features and the replay screener use.
RegistryQuestion = tuple[str, str]

REGISTRY_TIMEOUTS_TOTAL = Counter(
    "ashfall_registry_timeouts_total",
    "Registry screening calls that exceeded the per-call decision timeout and were answered unknown",
    ["identifier_type"],
)


class AsyncScreenClient(Protocol):
    """The part of NationalsimBankClient the prefetch uses."""

    async def screen(
        self, identifier_hash: str, identifier_type: str = "account", as_of: datetime | None = None
    ) -> ScreenResult: ...


def load_registry_timeout_seconds(config_path: str | Path = "config/account_features.yaml") -> float:
    """Read decision_registry.timeout_ms from configuration and return it in seconds."""
    raw = load_yaml(config_path)
    try:
        timeout_ms = raw["decision_registry"]["timeout_ms"]["value"]
    except (KeyError, TypeError) as err:
        raise ValueError(f"Missing decision_registry.timeout_ms.value in {config_path}") from err
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, (int, float)) or timeout_ms <= 0:
        raise ValueError(f"decision_registry.timeout_ms.value in {config_path} must be a positive number")
    return float(timeout_ms) / 1000.0


def list_registry_questions(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    payee_bank_id: str | None,
    payee_vpa: str | None,
    decision_time: datetime,
    arrival_cutoff: datetime,
    dpip_key: str | None = None,
    settings: AccountFeaturesSettings | None = None,
) -> list[RegistryQuestion]:
    """
    WHAT: The ordered, de-duplicated registry questions one decision asks: payer account, then each visible
    lifetime device fingerprint (sorted), then the payee account when there is one. Empty without a DPIP key.

    WHY: Built with the same key resolution, hash function and visibility rule (read_lifetime) as
    compute_intelligence_features and compute_payee_vpa_payment_features, so PrefetchedScreener never meets
    a question it was not given.
    """
    cfg = settings if settings is not None else get_default_settings()
    resolved_key = dpip_key
    if resolved_key is None and config.dpip_participant_key is not None:
        resolved_key = config.dpip_participant_key.get_secret_value()
    if not resolved_key:
        return []

    questions: list[RegistryQuestion] = [
        (
            "account",
            compute_identifier_hash(
                key=resolved_key,
                run_id=run_id,
                identifier_type="account",
                raw_value=account_id,
                bank_index=get_bank_index(bank_id),
            ),
        )
    ]
    lifetime = read_lifetime(client, run_id, bank_id, account_id, decision_time, arrival_cutoff, cfg)
    for fingerprint in sorted(lifetime.fingerprints):
        questions.append(
            (
                "device",
                compute_identifier_hash(
                    key=resolved_key,
                    run_id=run_id,
                    identifier_type="device",
                    raw_value=fingerprint,
                    bank_index=None,
                ),
            )
        )
    if payee_vpa and payee_bank_id is not None:
        questions.append(
            (
                "account",
                compute_identifier_hash(
                    key=resolved_key,
                    run_id=run_id,
                    identifier_type="account",
                    raw_value=payee_vpa,
                    bank_index=get_bank_index(payee_bank_id),
                ),
            )
        )
    return list(dict.fromkeys(questions))


def clock_resolution_seconds() -> float:
    """Resolution of the monotonic clock asyncio uses on this machine (about 0.0156 s on Windows, 1e-9 s on Linux)."""
    return time.get_clock_info("monotonic").resolution


async def _answer_one(client: AsyncScreenClient, question: RegistryQuestion, timeout_seconds: float) -> ScreenOutcome:
    """
    Ask one question; a call that takes longer than the timeout is 'unknown' with timed_out set and counted.

    On a coarse clock asyncio can fire a timer up to TWO clock ticks early: the clock itself only moves in ticks, and
    the loop also treats any timer due within one tick as ready (`end_time = self.time() + self._clock_resolution` in
    BaseEventLoop._run_once). So the wait is allowed `timeout + two ticks`, and the REAL time taken is then measured
    with perf_counter: a call is never cut off before its real timeout, and an answer that arrives after the timeout
    (but inside the slack) is still treated as timed out.
    """
    identifier_type, identifier_hash = question
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            client.screen(identifier_hash, identifier_type, as_of=None),
            timeout=timeout_seconds + 2 * clock_resolution_seconds(),
        )
    except asyncio.TimeoutError:
        result = None
    elapsed = time.perf_counter() - start
    if result is None or elapsed > timeout_seconds:
        REGISTRY_TIMEOUTS_TOTAL.labels(identifier_type=identifier_type).inc()
        return ScreenOutcome(status="unknown", first_visible_at=None, timed_out=True)
    return outcome_from_screen_result(result)


def compute_registry_timeout_ms(
    max_ms: float,
    feature_computation_max_ms: float,
    other_overhead_p99_ms: float,
    margin_ms: float,
    clock_resolution_ms: float,
    minimum_ms: float,
) -> float:
    """
    WHAT: The registry timeout from the latency budget (architect ruling A2):
    max_ms - feature_computation_max_ms - other_overhead_p99_ms - margin_ms - clock_resolution_ms.

    WHY: Whatever the budget does not give to our own feature code, the request handling and a safety margin is
    what the registry may use. A result below `minimum_ms` is an error (ValueError), never a quietly used value.
    """
    timeout_ms = max_ms - feature_computation_max_ms - other_overhead_p99_ms - margin_ms - clock_resolution_ms
    if timeout_ms < minimum_ms:
        raise ValueError(
            f"The computed registry timeout {timeout_ms:.3f} ms is below the {minimum_ms:.1f} ms minimum "
            f"(max {max_ms}, features {feature_computation_max_ms}, overhead p99 {other_overhead_p99_ms}, "
            f"margin {margin_ms}, clock resolution {clock_resolution_ms}); stop and report, do not set it."
        )
    return timeout_ms


async def prefetch_registry_answers(
    client: AsyncScreenClient,
    questions: list[RegistryQuestion],
    timeout_seconds: float,
) -> dict[RegistryQuestion, ScreenOutcome]:
    """
    WHAT: Ask every question at once and return the answers keyed by (identifier_type, identifier_hash).

    WHY: asyncio.gather makes the decision wait for the slowest call. The client's own 'unavailable' result
    becomes 'unknown'; any other exception is not caught.
    """
    outcomes = await asyncio.gather(*(_answer_one(client, question, timeout_seconds) for question in questions))
    return dict(zip(questions, outcomes))


def prepare_registry_questions(
    redis_client: Any,
    bank_id: str,
    request: DecisionRequest,
    dpip_key: str | None,
    settings: AccountFeaturesSettings | None = None,
) -> tuple[datetime, list[RegistryQuestion]] | None:
    """
    WHAT: Reads the bank watermark (the decision's arrival cutoff) and lists the registry questions the decision
    will ask at that cutoff. Returns None when the bank has no watermark (the decision then logs no_watermark).

    WHY: Blocking Redis work, so the async decision path runs it in a worker thread, then asks the registry on the
    server's own event loop, then hands both the cutoff and the answers to the feature computation.
    """
    arrival_cutoff = get_bank_watermark(redis_client, bank_id)
    if arrival_cutoff is None:
        return None
    questions = list_registry_questions(
        redis_client,
        extract_run_id(request.payment_id),
        bank_id,
        request.sender_account_id,
        request.receiver_bank_id,
        request.receiver_vpa,
        request.event_time,
        arrival_cutoff,
        dpip_key,
        settings,
    )
    return arrival_cutoff, questions


class RegistryGateway:
    """
    WHAT: What the API needs to screen against the registry from its own event loop: the shared connection pool,
    the registry address and the per-call timeout, all read once at startup.

    WHY: One pooled AsyncClient is shared by every decision (no new client or TCP connection per call). Each
    decision still gets its own cheap NationalsimBankClient because the bank and the run differ per request.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        client_timeout_seconds: float,
        retry_backoff_seconds: float,
        registry_timeout_seconds: float,
    ) -> None:
        self.http_client = http_client
        self.base_url = base_url
        self.client_timeout_seconds = client_timeout_seconds
        self.retry_backoff_seconds = retry_backoff_seconds
        self.registry_timeout_seconds = registry_timeout_seconds

    def client_for(self, bank_id: str, run_id: str) -> NationalsimBankClient:
        """A bank client for this decision that sends its requests through the shared pool."""
        return NationalsimBankClient(
            self.base_url,
            bank_id=bank_id,
            run_id=run_id,
            timeout_seconds=self.client_timeout_seconds,
            retry_backoff_seconds=self.retry_backoff_seconds,
            http_client=self.http_client,
        )

    async def aclose(self) -> None:
        """Close the shared pool (call once, at shutdown)."""
        await self.http_client.aclose()


def create_registry_gateway() -> RegistryGateway:
    """Build the gateway from configuration: pool size, address and timeouts (nationalsim.yaml), decision timeout."""
    base_url, max_connections, max_keepalive = load_registry_pool_settings()
    client_timeout, retry_backoff = load_bank_client_config()
    return RegistryGateway(
        http_client=create_shared_http_client(max_connections, max_keepalive, client_timeout),
        base_url=base_url,
        client_timeout_seconds=client_timeout,
        retry_backoff_seconds=retry_backoff,
        registry_timeout_seconds=load_registry_timeout_seconds(),
    )


def run_registry_prefetch(
    client: AsyncScreenClient,
    questions: list[RegistryQuestion],
    timeout_seconds: float,
) -> dict[RegistryQuestion, ScreenOutcome]:
    """
    WHAT: Run prefetch_registry_answers in its own event loop.

    WHY: The decision computes its features synchronously in a worker thread (asyncio.to_thread), where no event
    loop is running, so one asyncio.run per decision is the way to drive the async client from there. It must
    not be called from a thread that already runs an event loop.
    """
    return asyncio.run(prefetch_registry_answers(client, questions, timeout_seconds))


class PrefetchedScreener:
    """
    WHAT: A RegistryScreener that answers from the prefetched answers.

    WHY: The feature code is unchanged and asks its questions as before. A question that was not prefetched
    raises LookupError, which proves list_registry_questions is complete. As with LiveRegistryScreener, any
    as_of passed by the features is ignored: live decisions screen the currently active registry.
    """

    def __init__(self, answers: dict[RegistryQuestion, ScreenOutcome]) -> None:
        self._answers: dict[RegistryQuestion, ScreenOutcome] = dict(answers)

    def screen(
        self,
        identifier_hash: str,
        identifier_type: str,
        as_of: datetime | None = None,
    ) -> ScreenOutcome:
        """Return the prefetched outcome, or raise LookupError for a question that was not prefetched."""
        key = (identifier_type, identifier_hash)
        if key not in self._answers:
            raise LookupError(
                f"Registry question identifier_type={identifier_type!r}, identifier_hash={identifier_hash!r} "
                f"was not prefetched for this decision"
            )
        return self._answers[key]
