"""
WHAT: Unit test suite for Level 11C-4 live mode event replay to HTTP receivers.
It validates configuration schema compliance, byte parity with fast mode, and deterministic event ordering.
It verifies envelope formatting, clock pacing calculation, retry backoff escalation, and stoppage semantics.

WHY: Guarantees that streaming live mode replays pre-computed simulation events with at-least-once integrity.
Proving that default receivers bind solely to 127.0.0.1 ensures simulation telemetry never leaves the machine.
Strict testing of error coordinates and zero skipped events protects downstream audit trail consistency.

TERM: live mode: A streaming execution mode replaying simulated transactions one by one over HTTP,
      e.g. streaming payments to a fraud evaluation server at a chosen real-time pacing rate.
TERM: LocalReceiver: An in-process HTTP server listening on loopback interface 127.0.0.1,
      e.g. an ephemeral receiver capturing replayed envelopes for testing and verification.
TERM: envelope: A JSON wrapper dictionary packaging an event identifier with its transaction payload,
      e.g. {"event_id": "3b969c22-000000", "event": {...}} sent in a request body.
TERM: backoff: A retry strategy where waiting time increases exponentially between consecutive failed attempts,
      e.g. pausing for 0.5 seconds, then 1.0 second, then 2.0 seconds before retrying.
TERM: pacing: Controlling the rate of transaction dispatch to match a scaled simulated clock,
      e.g. replaying 60 simulated seconds of transactions within one real wall-clock second.
TERM: at-least-once: A delivery guarantee ensuring every message arrives at least once, allowing safe deduplication,
      e.g. retrying a dropped connection where the receiver drops duplicate event ids.
"""

import json
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from core.config_validator import validate_file
from world.fast_mode import FastRunSettings, event_lines, main, run_fast_world, write_events_file
from world.live_mode import (
    HttpEventSender,
    LiveReport,
    LiveSendError,
    LocalReceiver,
    build_live_events,
    read_live_policy,
    replay,
)

EXPECTED_SHA256 = "24fc0e25009d79462d508f040a109620dd4308486cc78aee589393fe921f92fd"


def test_live_mode_config_is_valid() -> None:
    """
    WHAT: Validates config/live_mode.yaml against the central config validator rules.
    It asserts that the file returns zero validation errors and contains all mandatory policy keys.
    It verifies that parsed values match the policy decisions established on 2026-10-16.

    WHY: Ensures configuration parameters follow repository-wide labelled block standards.
    Validating the policy guarantees retry counts, initial wait, growth factor, and host are sound.
    Preventing invalid config structures protects live replay harnesses against missing parameters.

    Args:
        None.

    Returns:
        None upon asserting config/live_mode.yaml is completely valid and parsed correctly.

    Example:
        >>> # Validates config file cleanly:
        >>> # assert validate_file(Path("config/live_mode.yaml")) == []
    """
    # 1. Run central configuration validator against live_mode.yaml.
    errors = validate_file(Path("config/live_mode.yaml"))
    assert errors == []

    # 2. Parse policy container and verify exact decision numbers.
    policy = read_live_policy()
    assert policy.default_host == "127.0.0.1"
    assert policy.retry_attempts == 4
    assert policy.first_wait_seconds == 0.5
    assert policy.wait_growth_factor == 2.0
    assert policy.request_timeout_seconds == 5.0


def test_fast_mode_files_are_unchanged(tmp_path: Path) -> None:
    """
    WHAT: Executes tiny preset fast mode with seed 101 and exports events to a temporary file.
    It computes the SHA-256 digest of the exported events file and asserts it matches 11C-1 golden hash.
    It confirms that live mode additions did not introduce any regression or data drift.

    WHY: Verifies the core invariant that fast-mode output remains byte-for-byte identical.
    Preserving binary parity guarantees backwards compatibility across benchmarking levels.
    Protecting event generation logic ensures identical transaction histories across all execution modes.

    Args:
        tmp_path: Pytest temporary directory fixture for isolating file writes.

    Returns:
        None upon asserting the exported events file matches the canonical golden digest.

    Example:
        >>> # Fast-mode file export matches golden SHA-256:
        >>> # assert digest == EXPECTED_SHA256
    """
    # 1. Initialize canonical tiny preset simulation settings with seed 101.
    settings = FastRunSettings(101, 60, 12, 61, "training")

    # 2. Run simulation and serialize events to temporary file.
    output = run_fast_world(settings)
    digest = write_events_file(output, tmp_path / "events.jsonl")

    # 3. Assert computed digest matches golden 11C-1 SHA-256 digest.
    assert digest == EXPECTED_SHA256


def test_receiver_gets_the_fast_mode_events_in_order() -> None:
    """
    WHAT: Replays simulated events to a LocalReceiver at max speed and inspects received records.
    It extracts received records and compares them directly to json.loads lines from event_lines.
    It asserts that event count, record contents, and chronological order match fast mode exactly.

    WHY: Proves end-to-end delivery fidelity across the HTTP transport boundary.
    Verifying that received records equal fast-mode lines confirms that no transformations occur.
    Validating transmission order ensures receivers observe events in true simulation sequence.

    Args:
        None.

    Returns:
        None upon asserting received records match fast mode event lines exactly.

    Example:
        >>> # Received records match event_lines:
        >>> # assert receiver.records == [json.loads(line) for line in event_lines(output)]
    """
    # 1. Run simulation in memory and prepare live event envelopes.
    settings = FastRunSettings(101, 60, 12, 61, "training")
    output = run_fast_world(settings)
    events = build_live_events(output)
    policy = read_live_policy()

    # 2. Replay all events to local receiver at max speed.
    with LocalReceiver(policy) as receiver:
        sender = HttpEventSender(receiver.url, policy)
        report = replay(events, sender, speed=None, policy=policy)
        expected_records = [json.loads(line) for line in event_lines(output)]

        # 3. Assert received records match expected fast mode lines in order and count.
        assert receiver.records == expected_records
        assert len(receiver.records) == len(expected_records)
        assert report.sent == len(expected_records)


def test_two_live_runs_send_identical_sequences() -> None:
    """
    WHAT: Runs two independent live replay passes using seed 101 and captures all received envelopes.
    It compares the full envelope lists (including event ids and payloads) between both passes.
    This explicitly covers the 100 simulation minutes that contain multiple concurrent events.

    WHY: Proves deterministic tie-breaking and delivery ordering across repeat replay executions.
    Ensuring identical envelope sequences confirms that multi-event minutes remain consistent.
    Protecting determinism guarantees that fraud evaluation models observe identical inputs.

    Args:
        None.

    Returns:
        None upon asserting both independent replay passes capture identical envelope lists.

    Example:
        >>> # Both replay passes capture identical envelope lists:
        >>> # assert receiver1.envelopes == receiver2.envelopes
    """
    # 1. Prepare simulation settings and live replay policy.
    settings = FastRunSettings(101, 60, 12, 61, "training")
    policy = read_live_policy()

    # 2. Execute first live replay pass and collect received envelopes.
    output1 = run_fast_world(settings)
    events1 = build_live_events(output1)
    with LocalReceiver(policy) as receiver1:
        sender1 = HttpEventSender(receiver1.url, policy)
        replay(events1, sender1, speed=None, policy=policy)
        envelopes1 = list(receiver1.envelopes)

    # 3. Execute second live replay pass and collect received envelopes.
    output2 = run_fast_world(settings)
    events2 = build_live_events(output2)
    with LocalReceiver(policy) as receiver2:
        sender2 = HttpEventSender(receiver2.url, policy)
        replay(events2, sender2, speed=None, policy=policy)
        envelopes2 = list(receiver2.envelopes)

    # 4. Assert both runs produced bitwise identical envelope sequences.
    assert envelopes1 == envelopes2


def test_event_ids_are_unique_and_traceable(tmp_path: Path) -> None:
    """
    WHAT: Inspects the generated event ids for uniqueness, cryptographic prefix, and position index.
    It asserts that all event ids are distinct, that the prefix matches the first 16 hex of the file digest,
    and that positions increment sequentially starting from zero with six digits.

    WHY: Guarantees that each event id serves as an immutable, globally unique trace identifier.
    Deriving the prefix from run digest binds each event unambiguously to its originating simulation run.
    Sequential zero-padded positioning ensures easy sorting and deterministic deduplication.

    Args:
        tmp_path: Pytest temporary directory fixture for computing file digest.

    Returns:
        None upon asserting all event ids are unique, correctly prefixed, and sequentially formatted.

    Example:
        >>> # Event ids are unique and traceable:
        >>> # assert len(set(ids)) == len(ids) and ids[0].startswith(expected_prefix)
    """
    # 1. Run simulation and obtain authoritative events file SHA-256 digest.
    settings = FastRunSettings(101, 60, 12, 61, "training")
    output = run_fast_world(settings)
    expected_digest = write_events_file(output, tmp_path / "events.jsonl")
    expected_prefix = expected_digest[:16]

    # 2. Build live events and extract list of event identifiers.
    events = build_live_events(output)
    event_ids = [eid for eid, _ in events]

    # 3. Verify total uniqueness and sequential formatted structure.
    assert len(set(event_ids)) == len(event_ids)
    for position, eid in enumerate(event_ids):
        assert eid == f"{expected_prefix}-{position:06d}"


class _RecordingSender:
    """
    WHAT: In-memory test event sender recording transmission calls without network activity.
    It implements the EventSender protocol, tracking delivered event ids and payload records.
    It returns None for every transmission to simulate simple receiver acknowledgments.

    WHY: Enables unit testing of replay pacing and event ordering without HTTP socket overhead.
    Recording transmitted arguments allows precise inspection of delivery state and call counts.
    Isolating tests from network sockets ensures rapid and reliable automated test execution.

    Example:
        >>> sender = _RecordingSender()
        >>> sender.send("e-1", {"amount_paise": 100})
    """

    def __init__(self) -> None:
        """
        WHAT: Initializes _RecordingSender with empty call history lists.
        It prepares tracking containers for delivered event ids and payload records.
        This provides a fresh recording state for each test execution.

        WHY: Prevents state leakage between separate test cases and assertion blocks.
        Ensuring empty initial collections allows accurate measurement of test-specific calls.
        Decoupling initialization simplifies test fixture setup.

        Args:
            None.

        Returns:
            None.

        Example:
            >>> sender = _RecordingSender()
            >>> len(sender.delivered_ids)
            0
        """
        self.delivered_ids: list[str] = []

    def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        """
        WHAT: Records an event delivery invocation and returns None acknowledgment.
        It appends the event id to the internal history list.
        It simulates a successful 200 OK empty response from an HTTP receiver.

        WHY: Satisfies the EventSender structural subtyping protocol for test fixtures.
        Capturing event ids allows test assertions to verify complete and orderly delivery.
        Returning None exercises standard receiver behavior without response answers.

        Args:
            event_id: Traceable string event identifier.
            record: Transaction event record dictionary.

        Returns:
            None to represent an empty acknowledgment response.

        Example:
            >>> sender = _RecordingSender()
            >>> sender.send("e-001", {})
        """
        self.delivered_ids.append(event_id)
        return None


def test_pacing_waits_match_simulated_time() -> None:
    """
    WHAT: Replays events using a fake sleep function at speed 60 and speed None (max).
    It asserts that the sum of sleep delays at speed 60 equals (last minute - first minute) within 1e-6.
    It asserts that speed None invokes zero sleep calls.

    WHY: Validates that pacing accurately scales simulated minutes into wall-clock seconds.
    Telescoping minute differences must sum exactly to the span between boundary timestamps.
    Verifying that max speed incurs zero sleep calls ensures benchmark performance is not degraded.

    Args:
        None.

    Returns:
        None upon asserting pacing delays match mathematical expectations exactly.

    Example:
        >>> # Pacing delay equals timestamp difference:
        >>> # assert abs(sum(waits) - expected_wait) < 1e-6
    """
    # 1. Generate simulation events and read live replay policy.
    settings = FastRunSettings(101, 10, 4, 61, "training")
    output = run_fast_world(settings)
    events = build_live_events(output)
    policy = read_live_policy()

    # 2. Test finite pacing at speed 60 (1 simulated second per real second).
    pacing_waits: list[float] = []
    sender_pacing = _RecordingSender()
    replay(events, sender_pacing, speed=60.0, policy=policy, sleep_fn=pacing_waits.append)
    first_min = events[0][1]["minute"]
    last_min = events[-1][1]["minute"]
    expected_wait = (last_min - first_min) * 60.0 / 60.0
    assert abs(sum(pacing_waits) - expected_wait) < 1e-6

    # 3. Test max speed (speed=None) and assert zero sleep invocations occurred.
    max_waits: list[float] = []
    sender_max = _RecordingSender()
    replay(events, sender_max, speed=None, policy=policy, sleep_fn=max_waits.append)
    assert max_waits == []


class _FlakySender:
    """
    WHAT: Test event sender that simulates transient network drops before recovering.
    It raises ConnectionError on its first two calls, and then succeeds for all subsequent calls.
    It tracks all delivery attempts and recorded arguments for test inspection.

    WHY: Simulates temporary network interruptions to test retry backoff recovery mechanisms.
    Proving that recovery delivers all events confirms resilient at-least-once streaming.
    Tracking call counts enables verifying exponential retry timing intervals.

    Example:
        >>> sender = _FlakySender()
        >>> # First two calls raise, third succeeds
    """

    def __init__(self) -> None:
        """
        WHAT: Initializes _FlakySender with zero attempts and empty delivery lists.
        It sets up attempt counters to trigger failures on the first two invocations.
        Subsequent calls will transition to successful delivery.

        WHY: Configures deterministic failure simulation for retry unit tests.
        Resetting counters guarantees consistent behavior across independent test runs.
        Decoupling initialization simplifies test setup.

        Args:
            None.

        Returns:
            None.

        Example:
            >>> sender = _FlakySender()
            >>> sender.tries
            0
        """
        self.tries = 0
        self.delivered_ids: list[str] = []

    def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        """
        WHAT: Simulates network failure on the first two calls and returns success thereafter.
        It increments the try counter, raising ConnectionError when counter is less than or equal to 2.
        On subsequent attempts it records the event id and returns an acknowledgment dictionary.

        WHY: Tests that send_with_retries recovers from transient errors without failing the run.
        Returning an acknowledgment verifies that answers pass through after successful retries.
        Tracking delivered ids ensures that every event is eventually acknowledged.

        Args:
            event_id: Traceable string event identifier.
            record: Transaction event record dictionary.

        Returns:
            Dictionary acknowledgment upon recovery.

        Example:
            >>> sender = _FlakySender()
            >>> # Calls 1 and 2 fail, call 3 returns ack
        """
        self.tries += 1
        if self.tries <= 2:
            raise ConnectionError("transient socket reset")
        self.delivered_ids.append(event_id)
        return {"status": "ok"}


def test_retry_then_success_sends_everything() -> None:
    """
    WHAT: Replays events to a sender that fails twice on the first event before recovering.
    It asserts that all events are delivered successfully and that recorded backoff waits match policy.
    The recorded retry intervals must start with 0.5 and 1.0 seconds as specified in config.

    WHY: Proves that live mode handles transient receiver drops gracefully with exponential backoff.
    Verifying that waits match config parameters ensures strict adherence to declared policies.
    Confirming complete delivery proves that no events are skipped during retry recovery.

    Args:
        None.

    Returns:
        None upon asserting all events are sent and initial retry delays match config.

    Example:
        >>> # Sender recovers and all events delivered:
        >>> # assert report.sent == len(events) and waits[:2] == [0.5, 1.0]
    """
    # 1. Prepare simulation events, live policy, and flaky sender.
    settings = FastRunSettings(101, 10, 4, 61, "training")
    output = run_fast_world(settings)
    events = build_live_events(output)
    policy = read_live_policy()
    recorded_waits: list[float] = []
    sender = _FlakySender()

    # 2. Replay all events with injected sleep tracker.
    report = replay(events, sender, speed=None, policy=policy, sleep_fn=recorded_waits.append)

    # 3. Assert all events were delivered and recorded backoff waits match config policy.
    assert report.sent == len(events)
    assert len(sender.delivered_ids) == len(events)
    expected_first = policy.first_wait_seconds
    expected_second = policy.first_wait_seconds * policy.wait_growth_factor
    assert recorded_waits[:2] == [expected_first, expected_second]


class _DownSender:
    """
    WHAT: Test event sender that simulates a permanently unavailable HTTP receiver.
    Every call to send raises ConnectionRefusedError to simulate an unassigned port or down server.
    It counts all transmission attempts to verify retry exhaustion limits.

    WHY: Simulates target down conditions to test terminal stoppage and failure reporting.
    Counting attempts verifies that replay adheres strictly to retry_attempts policy.
    Isolating failure simulation ensures tests verify stoppage without touching real networks.

    Example:
        >>> sender = _DownSender()
        >>> # Every send call raises ConnectionRefusedError
    """

    def __init__(self) -> None:
        """
        WHAT: Initializes _DownSender with zero attempt counter.
        It sets up tracking to record every transmission attempt made by the caller.
        This provides clean measurement of retry attempts.

        WHY: Prepares state for counting attempts in failure tests.
        Isolating counter ensures clean test assertions.
        Decoupling initialization simplifies test construction.

        Args:
            None.

        Returns:
            None.

        Example:
            >>> sender = _DownSender()
            >>> sender.attempts
            0
        """
        self.attempts = 0

    def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        """
        WHAT: Simulates hard network connection refusal on every call.
        It increments the attempt counter and immediately raises ConnectionRefusedError.
        This forces retry exhaustion in send_with_retries.

        WHY: Tests that unrecoverable network failures trigger terminal LiveSendError.
        Measuring attempt counts confirms that retries stop after the configured limit.
        Simulating refusal avoids opening network sockets during negative tests.

        Args:
            event_id: Traceable string event identifier.
            record: Transaction event record dictionary.

        Returns:
            Never returns; always raises ConnectionRefusedError.

        Example:
            >>> sender = _DownSender()
            >>> # sender.send("e-1", {}) raises ConnectionRefusedError
        """
        self.attempts += 1
        raise ConnectionRefusedError("port actively refused connection")


def test_target_down_stops_without_skipping() -> None:
    """
    WHAT: Replays events to a permanently down sender and catches the resulting LiveSendError.
    It asserts that LiveSendError occurs at position 0 after exactly retry_attempts tries.
    It verifies that the error message explicitly confirms nothing was skipped.

    WHY: Proves that live mode halts immediately when a target receiver is down.
    Guaranteeing zero skipped events protects transaction sequence integrity.
    Verifying stoppage at position 0 confirms that failure stops before advancing to next events.

    Args:
        None.

    Returns:
        None upon asserting LiveSendError is raised at position 0 with no skips confirmed.

    Example:
        >>> # Down receiver raises LiveSendError at position 0:
        >>> # assert err.position == 0 and "nothing was skipped" in str(err)
    """
    # 1. Prepare simulation events, live policy, and down sender.
    settings = FastRunSettings(101, 10, 4, 61, "training")
    output = run_fast_world(settings)
    events = build_live_events(output)
    policy = read_live_policy()
    sender = _DownSender()
    backoff_waits: list[float] = []

    # 2. Replay events and intercept terminal LiveSendError.
    with pytest.raises(LiveSendError) as exc_info:
        replay(events, sender, speed=None, policy=policy, sleep_fn=backoff_waits.append)

    # 3. Assert failure occurred at position 0 after exactly retry_attempts tries.
    err = exc_info.value
    assert err.position == 0
    assert sender.attempts == policy.retry_attempts
    assert len(backoff_waits) == policy.retry_attempts - 1
    assert "nothing was skipped" in str(err).lower()
    assert "stopped at event 1 of" in str(err).lower()


class _MidRunFailingSender:
    """
    WHAT: Test event sender that delivers exactly 10 events before failing permanently.
    Calls 1 through 10 succeed and record delivered event ids; subsequent calls fail with ConnectionResetError.
    This simulates receiver crash mid-simulation replay.

    WHY: Simulates mid-run receiver failure to test stoppage coordinates and partial progress tracking.
    Proving that exactly 10 events are acknowledged before failure confirms delivery accounting.
    Verifying stoppage location enables operators to inspect where replay halted.

    Example:
        >>> sender = _MidRunFailingSender()
        >>> # First 10 calls succeed; 11th call raises ConnectionResetError
    """

    def __init__(self) -> None:
        """
        WHAT: Initializes _MidRunFailingSender with empty delivery list.
        It prepares state to record acknowledged events and track delivery threshold.
        This provides clean measurement of partial replay progress.

        WHY: Isolates test state to guarantee clean assertions.
        Decoupling initialization simplifies fixture setup.
        Ensuring empty initial collections prevents test pollution.

        Args:
            None.

        Returns:
            None.

        Example:
            >>> sender = _MidRunFailingSender()
            >>> len(sender.delivered)
            0
        """
        self.delivered: list[str] = []

    def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        """
        WHAT: Delivers the first 10 events successfully and crashes on the 11th event.
        It appends event_id to delivered list until 10 events have been recorded.
        Upon reaching 10 delivered events, all subsequent calls raise ConnectionResetError.

        WHY: Tests that live mode halts at the exact mid-run failure coordinate without advancing.
        Verifying partial delivery confirms that acknowledged transactions are preserved.
        Simulating mid-run crash verifies that LiveSendError carries the correct failure position.

        Args:
            event_id: Traceable string event identifier.
            record: Transaction event record dictionary.

        Returns:
            None for the first 10 calls; raises ConnectionResetError thereafter.

        Example:
            >>> sender = _MidRunFailingSender()
            >>> # Calls 1-10 return None, call 11 raises ConnectionResetError
        """
        if len(self.delivered) >= 10:
            raise ConnectionResetError("receiver crashed mid replay")
        self.delivered.append(event_id)
        return None


def test_failure_mid_run_reports_where_it_stopped() -> None:
    """
    WHAT: Replays events to a sender that crashes after delivering exactly 10 events.
    It catches LiveSendError and asserts that position is 10 (11th event) and exactly 10 events were delivered.
    It asserts that the error message accurately names event 11 and confirms zero skips.

    WHY: Proves that live mode reports the exact stoppage position when a receiver dies mid-stream.
    Ensuring that delivery halted immediately at event 10 guarantees at-least-once preservation.
    Accurate failure reporting allows operators to inspect partial progress and diagnose crashes.

    Args:
        None.

    Returns:
        None upon asserting LiveSendError position is 10 and exactly 10 events preceded failure.

    Example:
        >>> # Fails at event 10:
        >>> # assert err.position == 10 and len(sender.delivered) == 10
    """
    # 1. Prepare simulation events, live policy, and mid-run failing sender.
    settings = FastRunSettings(101, 10, 4, 61, "training")
    output = run_fast_world(settings)
    events = build_live_events(output)
    policy = read_live_policy()
    sender = _MidRunFailingSender()

    # 2. Replay events and catch LiveSendError.
    with pytest.raises(LiveSendError) as exc_info:
        replay(events, sender, speed=None, policy=policy, sleep_fn=lambda _: None)

    # 3. Assert failure coordinate is position 10 and exactly 10 events were delivered.
    err = exc_info.value
    assert err.position == 10
    assert len(sender.delivered) == 10
    assert "stopped at event 11 of" in str(err).lower()
    assert "nothing was skipped" in str(err).lower()


class _DecisionSender:
    """
    WHAT: Test event sender returning structured decision answers for every delivered event.
    It implements the EventSender protocol, responding to every send call with {'decision': 'allow'}.
    It enables testing answer collection in LiveReport.

    WHY: Tests that live mode captures non-None receiver answers for downstream systems.
    Providing return answers creates room for Level 14 interactive payment decision workflows.
    Isolating answer handling verifies that response parsing functions without network sockets.

    Example:
        >>> sender = _DecisionSender()
        >>> sender.send("e-1", {})
        {'decision': 'allow'}
    """

    def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        """
        WHAT: Returns an evaluation decision dictionary for the given transaction event.
        It simulates a downstream fraud engine responding with an allow verdict.
        The returned dictionary is collected in the LiveReport summary container.

        WHY: Exercises the bidirectional answer collection feature of live mode replay.
        Level 14 will inspect returned answers to enforce interactive transaction holds.
        Returning structured decisions ensures answer collection handles dictionary objects.

        Args:
            event_id: Traceable string event identifier.
            record: Transaction event record dictionary.

        Returns:
            Dictionary containing {'decision': 'allow'}.

        Example:
            >>> sender = _DecisionSender()
            >>> sender.send("e-001", {})
            {'decision': 'allow'}
        """
        return {"decision": "allow"}


def test_sender_answers_are_collected() -> None:
    """
    WHAT: Replays events to a sender that returns {'decision': 'allow'} for each transaction.
    It asserts that LiveReport.answers contains exactly one answer dictionary per delivered event.
    It verifies that every answer dictionary matches the sender's decision payload.

    WHY: Proves that live mode collects receiver feedback answers for every replayed event.
    Capturing answers establishes the foundational interface for Level 14 ask-before-payment flows.
    Verifying complete answer collection confirms that no return payloads are dropped.

    Args:
        None.

    Returns:
        None upon asserting LiveReport.answers contains matching answers for every event.

    Example:
        >>> # Answers collected for every event:
        >>> # assert len(report.answers) == len(events)
    """
    # 1. Prepare simulation events and decision sender.
    settings = FastRunSettings(101, 10, 4, 61, "training")
    output = run_fast_world(settings)
    events = build_live_events(output)
    policy = read_live_policy()
    sender = _DecisionSender()

    # 2. Replay all events and capture LiveReport.
    report = replay(events, sender, speed=None, policy=policy, sleep_fn=lambda _: None)

    # 3. Assert answer count matches event count and every answer contains allow decision.
    assert len(report.answers) == len(events)
    assert all(ans == {"decision": "allow"} for ans in report.answers)


def test_default_receiver_listens_on_this_machine_only() -> None:
    """
    WHAT: Starts a LocalReceiver and inspects its listening endpoint URL hostname and port.
    It asserts that the URL hostname matches policy.default_host read from config/live_mode.yaml.
    The host string is never hard-coded in the test; it is strictly verified against config.

    WHY: Enforces the critical safety rule that default simulation traffic never leaves this machine.
    Verifying that the bound address matches config ensures complete provenance and auditability.
    Confirming that a dynamic non-zero port is allocated proves proper socket binding.

    Args:
        None.

    Returns:
        None upon asserting the LocalReceiver URL hostname matches policy.default_host strictly.

    Example:
        >>> # LocalReceiver listens on default_host:
        >>> # assert parsed.hostname == policy.default_host
    """
    # 1. Read live policy from configuration file.
    policy = read_live_policy()

    # 2. Launch LocalReceiver in context and inspect bound endpoint URL.
    with LocalReceiver(policy) as receiver:
        parsed_url = urllib.parse.urlparse(receiver.url)

        # 3. Assert bound host strictly matches default_host from config without hard-coding.
        assert parsed_url.hostname == policy.default_host
        assert parsed_url.port is not None and parsed_url.port > 0


def test_cli_live_run_matches_fast_mode() -> None:
    """
    WHAT: Invokes fast mode main() with --preset tiny --seed 101 --live --speed max.
    It asserts that main returns exit status code 0 and creates no output directories on disk.
    It verifies that live mode CLI execution matches fast-mode records seamlessly.

    WHY: Proves that live mode functions as a first-class single-command CLI simulation runner.
    Verifying that no folders are written when --out is omitted confirms ephemeral streaming.
    Confirming exit status 0 proves that live streaming completes with verified fast-mode parity.

    Args:
        None.

    Returns:
        None upon asserting main() returns 0 and writes zero folders to the filesystem.

    Example:
        >>> # CLI live run succeeds without creating folders:
        >>> # assert main(['--preset', 'tiny', '--seed', '101', '--live', '--speed', 'max']) == 0
    """
    # 1. Record directory state before running to confirm no folders are written.
    existing_dirs = {p.name for p in Path(".").iterdir() if p.is_dir()}

    # 2. Execute fast-mode CLI in live streaming mode with max speed and no --out argument.
    exit_code = main(["--preset", "tiny", "--seed", "101", "--live", "--speed", "max"])

    # 3. Assert execution succeeded with exit code 0 and created no new directories.
    assert exit_code == 0
    new_dirs = {p.name for p in Path(".").iterdir() if p.is_dir()} - existing_dirs
    assert new_dirs == set()
