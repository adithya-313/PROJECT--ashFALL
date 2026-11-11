"""
WHAT: The registry screener a replay bank uses at decision time: it asks the replay's in-process RegistryCore, as of the
decision time, and answers "unknown" when the drawn screen delay is longer than the registry timeout.

WHY: In replay the registry is a RegistryCore in this process, not an HTTP service, so a bank screens it directly. The
delay is not slept: a replay runs without waiting. It is drawn from the world seed (own stream) and compared with the
decision's registry timeout, exactly as the live prefetch path treats a registry that answers too late.

TERM: screen delay: the time one registry answer takes, drawn per question in a configured range, at least 1 ms.
TERM: unknown: the answer when the registry did not answer within the timeout; never "no match".
TERM: as_of: the decision time the registry is asked about; entries not yet visible then do not count.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.event_clock import Clock
from features.registry_screening import ScreenOutcome
from nationalsim.registry import RegistryCore

__all__ = ["CoreRegistryScreener", "ScreenStats", "MIN_SCREEN_DELAY_MS"]

MIN_SCREEN_DELAY_MS = 1.0


@dataclass
class ScreenStats:
    """
    WHAT: Counts of registry questions asked in a replay and how many were answered "unknown" by the delay rule.
    WHY: The report needs both numbers; one object is shared by every bank's screener.
    Args:
        screens: Questions asked.
        unknown: Questions answered unknown because the drawn delay exceeded the timeout.
    Example:
        ScreenStats(screens=10, unknown=2)
    """

    screens: int = 0
    unknown: int = 0


class CoreRegistryScreener:
    """
    WHAT: RegistryScreener over a RegistryCore for one DPIP-participant bank.
    WHY: Lets the unchanged decision features read registry answers in replay.
    Args:
        core: The replay's in-process registry.
        bank_id: The asking bank (must be a DPIP participant).
        run_id: The run id.
        clock: The replay world clock (used when no as_of is given).
        rng: The screen-delay stream (numpy Generator), shared by all banks.
        delay_ms: (low, high) of the delay range.
        timeout_ms: The registry timeout; a longer drawn delay gives "unknown".
        stats: Shared counters.
    Example:
        CoreRegistryScreener(core, "bank_0", run_id, clock, rng, (2.0, 15.0), 33.19, ScreenStats())
    """

    def __init__(
        self,
        core: RegistryCore,
        bank_id: str,
        run_id: str,
        clock: Clock,
        rng: Any,
        delay_ms: tuple[float, float],
        timeout_ms: float,
        stats: ScreenStats,
    ) -> None:
        self._core = core
        self._bank_id = bank_id
        self._run_id = run_id
        self._clock = clock
        self._rng = rng
        self._delay_ms = delay_ms
        self._timeout_ms = timeout_ms
        self._stats = stats

    def screen(self, identifier_hash: str, identifier_type: str, as_of: datetime | None = None) -> ScreenOutcome:
        """
        WHAT: Draws this question's delay (one uniform draw, at least 1 ms); over the timeout the answer is unknown,
        otherwise the registry is asked as of as_of (the world clock now when none is given).
        WHY: One draw per question keeps the stream aligned however the answers turn out.
        Args:
            identifier_hash: Registry identifier hash.
            identifier_type: 'account', 'device' or 'phone'.
            as_of: Decision time to ask about.
        Returns:
            The screening outcome.
        Example:
            screener.screen(account_hash, "account", as_of=decision_time)
        """
        low, high = self._delay_ms
        delay = max(MIN_SCREEN_DELAY_MS, float(self._rng.uniform(low, high)))
        self._stats.screens += 1
        if delay > self._timeout_ms:
            self._stats.unknown += 1
            return ScreenOutcome(status="unknown", first_visible_at=None)
        when = as_of if as_of is not None else self._clock.now_ist()
        result = self._core.screen(self._bank_id, identifier_hash, identifier_type, self._run_id, as_of=when)
        if result.match:
            return ScreenOutcome(status="match", first_visible_at=result.first_visible_at)
        return ScreenOutcome(status="no_match", first_visible_at=None)
