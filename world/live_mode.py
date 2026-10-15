"""
WHAT: Live-mode streaming orchestrator replaying pre-computed simulated events to an HTTP receiver.
It packages each chronological payment record into a signed envelope with a unique traceable event identifier.
It paces event delivery according to simulated timestamps, applying exponential backoff retries on failure.

WHY: Connects offline batch world simulations to live real-time fraud detection systems.
Replaying pre-generated deterministic events ensures streaming tests evaluate the exact same transactions as batch runs.
Providing retries without skips guarantees that down receivers recover without missing any audit records.

TERM: live mode: A streaming execution mode replaying simulated transactions one by one over HTTP,
      e.g. streaming payments to a fraud evaluation server at a chosen real-time pacing rate.
TERM: receiver: An HTTP service receiving replayed transaction event envelopes,
      e.g. a local evaluation server listening on 127.0.0.1 for incoming transactions.
TERM: envelope: A JSON wrapper dictionary packaging an event identifier with its transaction payload,
      e.g. {"event_id": "3b969c22-000000", "event": {...}} sent in a request body.
TERM: event id: A unique string identifier tying an event to a simulation run hash and file position,
      e.g. "3b969c224f612de6-000000" linking an envelope to run digest and row index.
TERM: backoff: A retry strategy where waiting time increases exponentially between consecutive failed attempts,
      e.g. pausing for 0.5 seconds, then 1.0 second, then 2.0 seconds before retrying.
TERM: at-least-once: A delivery guarantee ensuring every message arrives at least once, allowing safe deduplication,
      e.g. retrying a dropped connection where the receiver drops duplicate event ids.
TERM: pacing: Controlling the rate of transaction dispatch to match a scaled simulated clock,
      e.g. replaying 60 simulated seconds of transactions within one real wall-clock second.
"""

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from world.fast_mode import FastRunOutput, event_lines


class EventSender(Protocol):
    """
    WHAT: Formal structural interface for delivering transaction event envelopes to an external target.
    Any conforming sender must implement a synchronous send method accepting an event id and payload.
    It returns an optional dictionary response received from the downstream evaluation engine.

    WHY: Decouples the simulation replay loop from concrete network transport implementations.
    Isolating the transport allows unit tests to inject fast in-memory fake senders without networking.
    Level 14 will use the returned answer to enable interactive ask-before-each-payment transaction flows.

    Example:
        >>> class EchoSender:
        ...     def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        ...         return {"received": event_id}
        >>> sender: EventSender = EchoSender()
        >>> sender.send("abc-000000", {"amount_paise": 100})
        {'received': 'abc-000000'}
    """

    def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        """
        WHAT: Transmits a single event envelope containing an event id and record to a receiver.
        Implementations deliver the payload synchronously and return any reply dictionary provided by the receiver.
        If the receiver responds with an empty body or no content, it returns None.

        WHY: Establishes a standard contract for streaming transaction delivery across network boundaries.
        Level 14 will use the returned answer to enable interactive ask-before-each-payment transaction flows.
        Standardizing the signature permits pluggable fake senders for fast and deterministic unit testing.

        Args:
            event_id: Traceable string identifier formatted as hash prefix and zero-padded position index.
            record: Dictionary holding the 14 standardized transaction fields from events-v1 format.

        Returns:
            Parsed dictionary from receiver response body, or None if the response was empty.

        Example:
            >>> class DummySender:
            ...     def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
            ...         return {"ack": event_id}
            >>> sender = DummySender()
            >>> sender.send("hash-000001", {"minute": 5})
            {'ack': 'hash-000001'}
        """
        ...


@dataclass(frozen=True)
class LivePolicy:
    """
    WHAT: Immutable container holding configuration numbers and policies for live mode streaming.
    It encapsulates default host address, retry attempt limit, initial wait interval, growth factor, and HTTP timeout.
    All attributes are read directly from config/live_mode.yaml without hidden default fallbacks.

    WHY: Centralizes retry timing and network connection parameters into a validated typed structure.
    Strictly reading configuration values prevents hidden bugs and ensures audits match file declarations.
    Freezing the dataclass guarantees that policies cannot be modified during active simulation replay.

    Args:
        default_host: IPv4 string address designating local host interface (127.0.0.1).
        retry_attempts: Total maximum transmission attempts per event before halting (4).
        first_wait_seconds: Initial backoff delay duration in seconds following the first failed attempt (0.5).
        wait_growth_factor: Multiplier applied to the backoff interval on each subsequent retry (2).
        request_timeout_seconds: Maximum duration in seconds to await an HTTP response per request (5).

    Example:
        >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
        >>> (policy.default_host, policy.retry_attempts, policy.first_wait_seconds)
        ('127.0.0.1', 4, 0.5)
    """

    default_host: str
    retry_attempts: int
    first_wait_seconds: float
    wait_growth_factor: float
    request_timeout_seconds: float


def read_live_policy(path: Path = Path("config/live_mode.yaml")) -> LivePolicy:
    """
    WHAT: Reads live mode configuration from YAML and parses it into an immutable LivePolicy dataclass.
    It strictly validates and extracts default host, retry attempts, initial wait, growth factor, and timeout.
    Missing keys or malformed structures raise KeyError or TypeError immediately without falling back to defaults.

    WHY: Enforces strict provenance and prevents hidden default parameters from masking configuration errors.
    Standardizes loading logic so CLI runners, replay functions, and test fixtures use identical settings.
    Isolating file I/O permits passing custom config paths during isolated integration tests.

    Args:
        path: Filesystem Path pointing to the live mode YAML configuration file.

    Returns:
        LivePolicy instance holding typed configuration attributes parsed from the file.

    Example:
        >>> from pathlib import Path
        >>> from world.live_mode import read_live_policy
        >>> policy = read_live_policy(Path("config/live_mode.yaml"))
        >>> (policy.default_host, policy.retry_attempts)
        ('127.0.0.1', 4)
    """
    # 1. Read YAML text from disk using explicit UTF-8 encoding.
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))

    # 2. Extract the live_mode policy section dictionary strictly.
    cfg = raw_data["live_mode"]

    # 3. Construct and return the immutable LivePolicy container with typed attributes.
    return LivePolicy(
        default_host=str(cfg["default_host"]),
        retry_attempts=int(cfg["retry_attempts"]),
        first_wait_seconds=float(cfg["first_wait_seconds"]),
        wait_growth_factor=float(cfg["wait_growth_factor"]),
        request_timeout_seconds=float(cfg["request_timeout_seconds"]),
    )


def build_live_events(
    output: FastRunOutput,
) -> list[tuple[str, dict[str, Any]]]:
    """
    WHAT: Prepares ordered event tuples pairing traceable event identifiers with transaction records.
    It serializes events with event_lines, computes the 16-character SHA-256 prefix of all lines combined,
    and formats each event id as prefix-position with a six-digit zero-padded index.

    WHY: Binds streaming event payloads to the exact cryptographic digest of the simulation run.
    Formatting event ids with the run hash and position index ensures idempotency and auditability.
    Preserving the fast-mode line order guarantees deterministic delivery to receiving systems.

    Args:
        output: FastRunOutput container holding the simulated payment events, telemetry, and clock.

    Returns:
        List of (event_id, record) tuples in the deterministic chronological order of the simulation run.

    Example:
        >>> from world.fast_mode import FastRunSettings, run_fast_world
        >>> from world.live_mode import build_live_events
        >>> settings = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
        >>> output = run_fast_world(settings)
        >>> events = build_live_events(output)
        >>> len(events)
        499
        >>> events[0][0]
        '3b969c224f612de6-000000'
        >>> events[0][1]["minute"]
        211
    """
    # 1. Obtain serialized JSON Lines for all simulation events.
    lines = event_lines(output)

    # 2. Compute the 16-character SHA-256 hexadecimal prefix over the entire joined event lines payload.
    full_digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    prefix = full_digest[:16]

    # 3. Assemble (event_id, record) tuples in file order with zero-padded 6-digit positions.
    live_events: list[tuple[str, dict[str, Any]]] = []
    for position, line in enumerate(lines):
        event_id = f"{prefix}-{position:06d}"
        record = json.loads(line)
        live_events.append((event_id, record))

    # Return chronological list of prepared live mode event pairs.
    return live_events


@dataclass(frozen=True)
class HttpEventSender:
    """
    WHAT: HTTP event sender delivering transaction envelopes to a remote HTTP endpoint via POST requests.
    It packages the event identifier and transaction dictionary into an envelope JSON payload.
    It transmits the payload synchronously with urllib.request using timeout settings from LivePolicy.

    WHY: Implements the EventSender protocol for remote HTTP network endpoints without external dependencies.
    Packaging transaction data in envelopes allows receivers to identify duplicate transmissions.
    Parsing JSON responses enables downstream systems to return evaluation feedback to the replay loop.

    Args:
        url: Destination HTTP or HTTPS endpoint URL string.
        policy: LivePolicy instance holding network timeout and retry configuration.

    Example:
        >>> from world.live_mode import HttpEventSender, LivePolicy
        >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
        >>> sender = HttpEventSender("http://127.0.0.1:8000/events", policy)
        >>> sender.url
        'http://127.0.0.1:8000/events'
    """

    url: str
    policy: LivePolicy

    def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        """
        WHAT: Transmits a single transaction event envelope to the configured HTTP receiver URL.
        It builds the request envelope {"event_id": event_id, "event": record} and POSTs it as UTF-8 JSON.
        It returns the parsed dictionary if the response body contains JSON, or None if the response is empty.

        WHY: Provides standard HTTP transport implementation of the EventSender protocol.
        Raising exceptions on HTTP status codes >= 400 or network errors triggers configured retry logic.
        Returning parsed responses allows future levels to receive interactive decision feedback.

        Args:
            event_id: Traceable string identifier formatted as hash prefix and zero-padded position index.
            record: Dictionary holding the 14 standardized transaction fields from events-v1 format.

        Returns:
            Parsed dictionary from receiver response body, or None if the response was empty.

        Example:
            >>> from world.live_mode import HttpEventSender, LivePolicy, LocalReceiver
            >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
            >>> with LocalReceiver(policy) as receiver:
            ...     sender = HttpEventSender(receiver.url, policy)
            ...     ans = sender.send("hash-000000", {"amount_paise": 100})
            >>> ans is None
            True
        """
        # 1. Package the event id and transaction record into a JSON envelope byte string.
        envelope = {"event_id": event_id, "event": record}
        payload = json.dumps(envelope).encode("utf-8")

        # 2. Build the HTTP POST request with application/json content header.
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # 3. Transmit the request with configured timeout and inspect the response.
        with urllib.request.urlopen(req, timeout=self.policy.request_timeout_seconds) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP request failed with status {response.status}")
            raw_body = response.read()

        # 4. Return parsed JSON dictionary, or None if the body was empty.
        text = raw_body.decode("utf-8").strip()
        if not text:
            return None
        return json.loads(text)


class _ReceiverHandler(BaseHTTPRequestHandler):
    """
    WHAT: Internal HTTP request handler for the in-process LocalReceiver server.
    It receives POST requests, parses incoming event envelope JSON payloads, and appends them to the server list.
    It responds with HTTP 200 OK and an empty body with Content-Length set to zero.

    WHY: Serves as the concrete HTTP protocol implementation for local live mode replay.
    Thread-safe collection ensures arriving envelopes are recorded accurately without race conditions.
    Suppresses standard console logging to prevent test output clutter.

    Example:
        >>> # Handled automatically by http.server machinery
        >>> pass
    """

    def do_POST(self) -> None:
        """
        WHAT: Handles incoming HTTP POST requests containing transaction event envelopes.
        It parses the JSON body, appends the envelope to the server's thread-safe collection, and responds with 200 OK.
        The response contains an empty body with Content-Length set to zero.

        WHY: Provides an in-memory sink for live mode event replay without external services.
        Thread synchronization ensures that concurrent requests do not corrupt the envelope list order.
        Returning an empty 200 OK body signals successful receipt to HttpEventSender.

        Args:
            None (request attributes are read from self).

        Returns:
            None.

        Example:
            >>> # Handled automatically during HTTP POST requests
            >>> pass
        """
        # Read the raw byte payload according to the Content-Length header.
        raw_length = self.headers["Content-Length"]
        content_length = int(raw_length) if raw_length else 0
        body = self.rfile.read(content_length)

        # Parse JSON envelope and record it thread-safely in arrival order.
        envelope = json.loads(body.decode("utf-8"))
        receiver: "LocalReceiver" = getattr(self.server, "receiver")
        with receiver.lock:
            receiver.envelopes.append(envelope)

        # Send standard 200 OK response with empty body.
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """
        WHAT: Suppresses default HTTP server request logging to standard error.
        It intercepts logging calls from BaseHTTPRequestHandler and discards them silently.
        This prevents noisy log spam during high-volume event replay in test suites and CLI runs.

        WHY: Keeps terminal output clean and focused on simulation progress and verification output.
        Preventing stdout/stderr clutter ensures CLI output matches golden formatting requirements.
        Silencing standard server logs avoids interfering with automated test result matchers.

        Args:
            format: Format string containing logging template directives.
            *args: Variable positional arguments interpolated into the format string.

        Returns:
            None.

        Example:
            >>> # Called automatically by http.server; produces no output
            >>> pass
        """
        return


class LocalReceiver:
    """
    WHAT: In-process HTTP receiver running a ThreadingHTTPServer on the local loopback interface.
    It captures incoming event envelopes in an in-memory list protected by a threading lock.
    It provides lifecycle methods start, stop, and context management protocol support.

    WHY: Enables offline end-to-end replay testing without requiring external Docker services or network access.
    Binding strictly to 127.0.0.1 guarantees that simulation traffic never leaves the host machine.
    Collecting received envelopes in arrival order enables exact verification against fast-mode records.

    Args:
        policy: LivePolicy instance supplying the default host interface to bind.

    Example:
        >>> from world.live_mode import LivePolicy, LocalReceiver
        >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
        >>> with LocalReceiver(policy) as receiver:
        ...     receiver.url.startswith("http://127.0.0.1:")
        True
    """

    def __init__(self, policy: LivePolicy) -> None:
        """
        WHAT: Initializes LocalReceiver with policy settings and empty envelope storage.
        It creates a threading lock for safe concurrent collection and prepares server references.
        The server is not started until start() or __enter__() is invoked.

        WHY: Prepares synchronization primitives and state containers before starting background listener threads.
        Encapsulating policy parameters ensures socket binding adheres strictly to local address rules.
        Decoupling initialization from socket creation allows flexible setup in tests and CLI callers.

        Args:
            policy: LivePolicy instance supplying default_host configuration.

        Returns:
            None.

        Example:
            >>> from world.live_mode import LivePolicy, LocalReceiver
            >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
            >>> receiver = LocalReceiver(policy)
            >>> len(receiver.envelopes)
            0
        """
        self.policy = policy
        self.envelopes: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.url = ""
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def records(self) -> list[dict[str, Any]]:
        """
        WHAT: Extracts the inner transaction event dictionaries from all received envelopes in arrival order.
        It unwraps each envelope and returns the raw 14-field event records.
        The extracted records correspond directly to the lines produced by fast mode.

        WHY: Simplifies comparison between received HTTP payloads and fast-mode golden records.
        Decoupling envelope metadata from transaction payload aids downstream test assertions.
        Preserving arrival order guarantees deterministic validation of streaming delivery.

        Args:
            None (reads from self.envelopes).

        Returns:
            List of 14-field transaction event dictionaries in the order they were received.

        Example:
            >>> from world.live_mode import LivePolicy, LocalReceiver
            >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
            >>> receiver = LocalReceiver(policy)
            >>> receiver.records
            []
        """
        with self.lock:
            return [envelope["event"] for envelope in self.envelopes]

    def start(self) -> str:
        """
        WHAT: Binds a ThreadingHTTPServer to the policy default host and dynamic port 0 on a background thread.
        It starts serving incoming requests asynchronously and sets the listening URL property.
        It returns the fully qualified HTTP URL string of the running receiver.

        WHY: Dynamically allocating port 0 avoids port collisions in parallel test runners or multi-process runs.
        Running in a background daemon thread allows the main execution thread to drive event replay.
        Returning the URL provides callers with the exact endpoint address needed by HttpEventSender.

        Args:
            None.

        Returns:
            Fully qualified HTTP URL string where the receiver is listening.

        Example:
            >>> from world.live_mode import LivePolicy, LocalReceiver
            >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
            >>> receiver = LocalReceiver(policy)
            >>> url = receiver.start()
            >>> url.startswith("http://127.0.0.1:")
            True
            >>> receiver.stop()
        """
        # 1. Bind ThreadingHTTPServer to the configured local host interface and dynamic port 0.
        server = ThreadingHTTPServer((self.policy.default_host, 0), _ReceiverHandler)
        server.receiver = self  # type: ignore[attr-defined]
        self._server = server
        self.url = f"http://{self.policy.default_host}:{server.server_port}"

        # 2. Launch server event loop in a background daemon thread.
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._thread = thread

        # 3. Return the listening endpoint URL string.
        return self.url

    def stop(self) -> None:
        """
        WHAT: Shuts down the background ThreadingHTTPServer and closes listening sockets.
        It signals the server loop to stop, waits for the background thread to finish, and resets server state.
        Calling stop when already stopped is safely ignored without error.

        WHY: Cleans up background threads and releases operating system socket descriptors promptly.
        Guarantees that test cases do not leave orphan sockets open or interfere with subsequent tests.
        Providing idempotent shutdown logic prevents exceptions during abnormal termination cleanup.

        Args:
            None.

        Returns:
            None.

        Example:
            >>> from world.live_mode import LivePolicy, LocalReceiver
            >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
            >>> receiver = LocalReceiver(policy)
            >>> _ = receiver.start()
            >>> receiver.stop()
            >>> receiver._server is None
            True
        """
        # 1. Shut down active HTTP server instance and close network sockets.
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            if self._thread is not None:
                self._thread.join()
            self._server = None
            self._thread = None

    def __enter__(self) -> "LocalReceiver":
        """
        WHAT: Enters runtime context by starting the background HTTP receiver server.
        It starts the listener and returns the receiver instance ready for transaction replay.
        It enables use of LocalReceiver in Python with statements for automatic cleanup.

        WHY: Provides standard context management protocol for safe resource lifecycle management.
        Guarantees that listener threads and sockets are always cleaned up even if exceptions occur.
        Improves ergonomics of integration tests and CLI live mode harnesses.

        Args:
            None.

        Returns:
            Self instance with running HTTP server and active URL property.

        Example:
            >>> from world.live_mode import LivePolicy, LocalReceiver
            >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
            >>> with LocalReceiver(policy) as receiver:
            ...     receiver.url.startswith("http://127.0.0.1:")
            True
        """
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        WHAT: Exits runtime context by stopping the HTTP receiver and closing socket resources.
        It delegates to stop() to terminate the listener thread and close active server sockets.
        It does not suppress any exceptions that occurred inside the context block.

        WHY: Ensures deterministic socket release and thread termination upon block exit.
        Adheres to standard Python context manager contract by passing exceptions through.
        Prevents resource leaks during test execution failures or unexpected runtime errors.

        Args:
            exc_type: Exception class if an exception was raised, else None.
            exc_val: Exception instance if an exception was raised, else None.
            exc_tb: Traceback object if an exception was raised, else None.

        Returns:
            None.

        Example:
            >>> from world.live_mode import LivePolicy, LocalReceiver
            >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
            >>> receiver = LocalReceiver(policy)
            >>> receiver.__enter__()  # doctest: +ELLIPSIS
            <world.live_mode.LocalReceiver object at ...>
            >>> receiver.__exit__(None, None, None)
        """
        self.stop()


@dataclass(frozen=True)
class LiveReport:
    """
    WHAT: Summary container recording the final delivery statistics of a live mode replay session.
    It tracks the total number of events scheduled, count of successfully delivered events, and collected answers.
    Answers consist of non-None dictionary replies returned by the receiver for delivered events.

    WHY: Provides an immutable audit trail verifying complete and un-skipped transaction replay.
    Preserving non-None answers from the receiver enables downstream processing or confirmation in Level 14.
    Returning a structured report allows programmatic verification across unit tests and CLI runs.

    Args:
        total: Total number of transaction events scheduled for replay in the world simulation.
        sent: Number of transaction events successfully transmitted to and acknowledged by the receiver.
        answers: Chronological list of non-None dictionary answers returned by the receiver.

    Example:
        >>> report = LiveReport(total=10, sent=10, answers=[{"status": "ok"}])
        >>> (report.total, report.sent, len(report.answers))
        (10, 10, 1)
    """

    total: int
    sent: int
    answers: list[dict[str, Any]]


class LiveSendError(Exception):
    """
    WHAT: Exception raised when repeated transmission attempts to a receiver fail without recovery.
    It records the zero-indexed event position, total event count, event identifier, and root failure cause.
    The formatted message explicitly declares the stopped position and guarantees that no events were skipped.

    WHY: Halts simulation replay immediately upon persistent communication failure to protect delivery order.
    Providing exact stoppage coordinates allows operators to resume or diagnose receiver availability.
    Explicitly stating that nothing was skipped confirms that at-least-once delivery semantics remain intact.

    Args:
        position: Zero-indexed position of the event that could not be delivered.
        total: Total count of events scheduled in the current replay session.
        event_id: Cryptographic and positional identifier of the unacknowledged event.
        last_error: String description of the final underlying transmission exception.

    Example:
        >>> err = LiveSendError(0, 100, "abc-000000", "Connection refused")
        >>> (err.position, err.total, err.event_id)
        (0, 100, 'abc-000000')
        >>> "nothing was skipped" in str(err)
        True
    """

    def __init__(
        self,
        position: int,
        total: int,
        event_id: str,
        last_error: str,
    ) -> None:
        """
        WHAT: Initializes LiveSendError with failure coordinates and builds an explanatory message.
        It calculates the human-readable 1-indexed event number and formats the message string.
        The resulting string states the exact stoppage location and confirms no events were skipped.

        WHY: Ensures consistent error reporting across all network and transport failure scenarios.
        Recording structured attributes permits programmatic inspection by CLI handlers and test assertions.
        Standardizing the text message aids operational observability during automated replay runs.

        Args:
            position: Zero-indexed position of the event that could not be delivered.
            total: Total count of events scheduled in the current replay session.
            event_id: Cryptographic and positional identifier of the unacknowledged event.
            last_error: String description of the final underlying transmission exception.

        Returns:
            None.

        Example:
            >>> err = LiveSendError(2, 50, "hash-000002", "Timeout")
            >>> str(err).startswith("Stopped at event 3 of 50")
            True
        """
        message = (
            f"Stopped at event {position + 1} of {total} ({event_id}): "
            f"{last_error}; nothing was skipped."
        )
        super().__init__(message)
        self.position = position
        self.total = total
        self.event_id = event_id
        self.last_error = last_error


def send_with_retries(
    sender: EventSender,
    event_id: str,
    record: dict[str, Any],
    policy: LivePolicy,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any] | None:
    """
    WHAT: Transmits an event envelope with exponential backoff retries upon network or HTTP errors.
    It attempts delivery up to retry_attempts times, pausing between failed attempts according to backoff policy.
    If all attempts are exhausted, it raises the final underlying exception.

    WHY: Implements resilient at-least-once delivery for transient network hiccups and receiver restarts.
    Backing off exponentially prevents overwhelming a recovering receiver with aggressive retry storms.
    Raising the error up when retries fail ensures that the replay loop halts without skipping any event.

    Args:
        sender: EventSender implementation handling the concrete transport.
        event_id: Traceable string identifier formatted as hash prefix and zero-padded position index.
        record: Dictionary holding the 14 standardized transaction fields from events-v1 format.
        policy: LivePolicy instance supplying retry_attempts, first_wait_seconds, and wait_growth_factor.
        sleep_fn: Callable used for pausing execution (defaults to time.sleep; injected in tests).

    Returns:
        Optional dictionary response returned by the sender upon successful delivery.

    Example:
        >>> class FlakySender:
        ...     def __init__(self):
        ...         self.tries = 0
        ...     def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        ...         self.tries += 1
        ...         if self.tries < 2:
        ...             raise ConnectionError("down")
        ...         return {"ack": event_id}
        >>> from world.live_mode import LivePolicy, send_with_retries
        >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
        >>> waits = []
        >>> ans = send_with_retries(FlakySender(), "e-1", {"minute": 1}, policy, sleep_fn=waits.append)
        >>> (ans, waits)
        ({'ack': 'e-1'}, [0.5])
    """
    # 1. Initialize tracker for the most recent transmission exception.
    last_error: Exception | None = None

    # 2. Attempt delivery across up to retry_attempts attempts.
    for attempt_index in range(policy.retry_attempts):
        try:
            return sender.send(event_id, record)
        except Exception as exc:
            last_error = exc
            # If attempts remain, compute exponential backoff delay and pause.
            if attempt_index < policy.retry_attempts - 1:
                wait_seconds = policy.first_wait_seconds * (
                    policy.wait_growth_factor ** attempt_index
                )
                sleep_fn(wait_seconds)

    # 3. Raise the final transmission exception if all retry attempts failed.
    if last_error is not None:
        raise last_error
    return None


def replay(
    events: list[tuple[str, dict[str, Any]]],
    sender: EventSender,
    speed: float | None,
    policy: LivePolicy,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> LiveReport:
    """
    WHAT: Sequentially replays an ordered list of simulated events to an EventSender with paced timing and retries.
    It pauses between events according to simulated timestamp differences scaled by the speed factor.
    On transmission failure it halts immediately and raises LiveSendError without skipping any event.

    WHY: Drives the core live mode streaming simulation loop while strictly preserving file order.
    Pacing reproduction matches real-world transaction arrival rates for realistic streaming evaluation.
    Halting on persistent delivery failure protects at-least-once delivery and stream integrity.

    Args:
        events: List of (event_id, record) tuples in deterministic file order.
        sender: EventSender implementation delivering envelopes to the target receiver.
        speed: Simulation speed factor (simulated seconds per real second), or None for max speed.
        policy: LivePolicy specifying retry counts, backoff timings, and timeouts.
        sleep_fn: Callable used for pacing delays and retry backoff pauses.

    Returns:
        LiveReport recording total event count, successfully delivered count, and collected answers.

    Example:
        >>> class EchoSender:
        ...     def send(self, event_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        ...         return {"id": event_id}
        >>> from world.live_mode import LivePolicy, replay
        >>> policy = LivePolicy("127.0.0.1", 4, 0.5, 2.0, 5.0)
        >>> events = [("e-000000", {"minute": 10}), ("e-000001", {"minute": 12})]
        >>> report = replay(events, EchoSender(), speed=None, policy=policy, sleep_fn=lambda s: None)
        >>> (report.total, report.sent, len(report.answers))
        (2, 2, 2)
    """
    # 1. Initialize report counters and answer collection list.
    total = len(events)
    sent = 0
    answers: list[dict[str, Any]] = []
    previous_minute = events[0][1]["minute"] if events else 0

    # 2. Replay each event sequentially in file order.
    for position, (event_id, record) in enumerate(events):
        # Apply clock pacing delay if a finite speed factor is specified and not the first event.
        if speed is not None and position > 0:
            delay_seconds = (record["minute"] - previous_minute) * 60.0 / speed
            if delay_seconds > 0:
                sleep_fn(delay_seconds)
        previous_minute = record["minute"]

        # Attempt transmission with retries; raise LiveSendError on exhausted retries.
        try:
            answer = send_with_retries(sender, event_id, record, policy, sleep_fn=sleep_fn)
        except Exception as exc:
            raise LiveSendError(position, total, event_id, str(exc)) from exc

        # Record successful transmission and collect any non-None response answer.
        sent += 1
        if answer is not None:
            answers.append(answer)

    # 3. Return completed LiveReport summary container.
    return LiveReport(total=total, sent=sent, answers=answers)
