# File: showcase/level07_helpers.py
# Purpose: Small building blocks for the Level 7 showcase checks: fake requests, a second database connection,
#          clearing tables, draining the relay, a throwaway topic that is always deleted afterwards, and the
#          broker-outage machinery shared by the two broker checks.
import asyncio
import random
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import asyncpg

from api.routers.scoring import TransactionRequest
from api.services.outbox import StoredTransaction, record_transaction_with_outbox
from api.services.outbox_relay import relay_one_batch
from showcase.docker_control import start_broker, stop_broker, wait_for_broker
from showcase.framework import FIXED_SEED
from showcase.isolation import Workspace
from showcase.kafka_tools import create_topic, delete_topic, make_producer, make_topic_name, read_events

MODEL_VERSION = "showcase-model-1"
FRAUD_SCORE = 0.25
RELAY_BATCH_SIZE = 50
# Safety stop for draining: a healthy drain finishes long before this many rounds.
MAX_DRAIN_ROUNDS = 1000

# How long the broker stays stopped in each kill, in seconds.
DOWN_SECONDS = 3.0
# Pause between two writes, so the writer keeps going while the broker is down.
WRITE_PAUSE_SECONDS = 0.02
# How many more writes are allowed while the broker is stopped.
WRITES_DURING_OUTAGE = 10
# After a failure the relay waits this long before it makes a new producer (the real relay backs off in a similar way).
RELAY_RETRY_SECONDS = 0.5
RELAY_POLL_SECONDS = 0.2
# Time allowed for each kill cycle, including waiting for the relay to catch up.
SECONDS_ALLOWED_PER_KILL = 120
# After the run, reading the topic stops once nothing new arrives for this long.
READ_IDLE_SECONDS = 3.0

BandList = list[tuple[float, float, str]]


def make_request(index: int, seed_rng: random.Random, key: str | None) -> TransactionRequest:
    """Build a fake transaction. The random generator is seeded by the caller, so the same seed gives the same data."""
    amount = round(seed_rng.uniform(1.0, 500.0), 2)
    return TransactionRequest(
        amount=amount,
        account_id=f"showcase-account-{index % 10}",
        merchant_id=f"showcase-merchant-{index % 7}",
        device_fingerprint=f"showcase-device-{index}",
        geo_coordinates="40.7128,-74.0060",
        idempotency_key=key,
    )


async def open_second_connection(workspace: Workspace) -> asyncpg.Connection:
    """
    Open another connection into the SAME throwaway schema.

    The relay and the writer need separate connections, because one connection can only do one thing at a time.
    """
    return await asyncpg.connect(dsn=workspace.dsn, timeout=5, server_settings={"search_path": workspace.schema_name})


async def clear_workspace_tables(conn: asyncpg.Connection) -> None:
    """Empty every table in the workspace in one statement, so each check starts from nothing."""
    await conn.execute("TRUNCATE transactions, outbox, processed_events, demo_event_counts")


async def write_transactions(
    conn: asyncpg.Connection, requests: list[TransactionRequest],
) -> list[StoredTransaction]:
    """Write each request with the real atomic writer, one after the other."""
    stored: list[StoredTransaction] = []
    for request in requests:
        stored.append(await record_transaction_with_outbox(conn, request, FRAUD_SCORE, MODEL_VERSION))
    return stored


async def drain_relay(
    conn: asyncpg.Connection, producer: Any, topic: str, batch_size: int = RELAY_BATCH_SIZE,
) -> int:
    """
    Run the real relay batch by batch until nothing is waiting, or until a batch reports an error.

    Returns how many events were published in total.
    """
    total_published = 0
    for _ in range(MAX_DRAIN_ROUNDS):
        result = await relay_one_batch(conn, producer, batch_size, topic)
        total_published += result.published
        if result.error is not None or result.waiting == 0:
            break
    return total_published


async def drain_with_new_producer(conn: asyncpg.Connection, topic: str) -> int:
    """Create a producer, drain the relay with it, and always stop the producer."""
    producer = await make_producer()
    try:
        return await drain_relay(conn, producer, topic)
    finally:
        await producer.stop()


async def fetch_outbox_event_ids(conn: asyncpg.Connection) -> list[uuid.UUID]:
    """All outbox event ids in the relay's own order (created_at, then id)."""
    rows = await conn.fetch("SELECT event_id FROM outbox ORDER BY created_at, id")
    return [row["event_id"] for row in rows]


async def check_order_rules_agree(conn: asyncpg.Connection) -> bool:
    """True when ordering by (created_at, id) gives the same sequence as ordering by id alone."""
    by_time = [row["id"] for row in await conn.fetch("SELECT id FROM outbox ORDER BY created_at, id")]
    by_id = [row["id"] for row in await conn.fetch("SELECT id FROM outbox ORDER BY id")]
    return by_time == by_id


@asynccontextmanager
async def throwaway_topic() -> AsyncIterator[str]:
    """Create a fresh topic, hand out its name, and ALWAYS delete it afterwards (even if the check failed)."""
    topic = make_topic_name()
    await create_topic(topic)
    try:
        yield topic
    finally:
        await delete_topic(topic)


# ---------------------------------------------------------------------------
# Broker outage machinery (used by "Broker stopped mid-run" and "Repeated broker kills lose nothing")
# ---------------------------------------------------------------------------

@dataclass
class OutageState:
    """What the writer, the relay and the kill loop share while an outage run is going."""

    started: float = field(default_factory=time.monotonic)
    written: int = 0
    write_limit: int = 0  # the writer may not write beyond this many transactions yet
    delivered: int = 0
    stop_relay: bool = False
    timeline: list[tuple[float, int]] = field(default_factory=list)

    def elapsed(self) -> float:
        """Seconds since the run started."""
        return time.monotonic() - self.started


@dataclass
class KillTimes:
    """Timing facts collected by the kill loop (they differ on every run, so they go into evidence only)."""

    down_seconds: list[float] = field(default_factory=list)
    ready_seconds: list[float] = field(default_factory=list)
    bands: BandList = field(default_factory=list)


@dataclass
class OutageResult:
    """What an outage run found."""

    committed: int
    outbox_rows: int
    unprocessed: int
    raw_count: int
    kafka_ids: set[str]
    outbox_ids: set[str]
    kills: KillTimes
    timeline: list[tuple[float, int]]


def choose_kill_points(total_writes: int, kill_count: int) -> list[int]:
    """Spread the kills over the writes: with 100 writes, 1 kill happens at 33 and 5 kills at 14, 29, 43, 57, 71."""
    return [round(total_writes * (number + 1) / (kill_count + 2)) for number in range(kill_count)]


def compress_timeline(points: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Keep only the points where the delivered count changed (plus the first and last), so the chart stays small."""
    if len(points) <= 2:
        return list(points)
    kept = [points[0]]
    for point in points[1:-1]:
        if point[1] != kept[-1][1]:
            kept.append(point)
    kept.append(points[-1])
    return kept


def build_outage_chart(timeline: list[tuple[float, int]], bands: BandList) -> dict[str, Any]:
    """The chart dict the report writer draws: events delivered over time, with a band for each outage."""
    return {
        "type": "timeline",
        "points": compress_timeline(timeline),
        "shaded": bands,
        "title": "Events delivered while the broker was stopped",
        "x_label": "seconds",
        "y_label": "events delivered",
    }


async def write_with_gate(conn: asyncpg.Connection, state: OutageState, total_writes: int, run_id: str) -> None:
    """Write transactions one by one, but never beyond state.write_limit (the kill loop moves the limit forward)."""
    rng = random.Random(FIXED_SEED)
    while state.written < total_writes:
        if state.written >= state.write_limit:
            await asyncio.sleep(0.05)
            continue
        request = make_request(state.written, rng, f"showcase-l07-{run_id}-{state.written}")
        await record_transaction_with_outbox(conn, request, FRAUD_SCORE, MODEL_VERSION)
        state.written += 1
        await asyncio.sleep(WRITE_PAUSE_SECONDS)


async def close_producer_quietly(producer: Any) -> None:
    """Stop a producer and ignore any error (it may already be broken because the broker is down)."""
    if producer is None:
        return
    try:
        await asyncio.wait_for(producer.stop(), 5)
    except Exception:  # noqa: BLE001 - nothing useful can be done about a broken producer
        pass


async def relay_until_stopped(workspace: Workspace, topic: str, state: OutageState) -> None:
    """
    Our own small relay loop: relay a batch, remember how many are delivered, and after any failure throw the
    producer away and make a new one. This is the same recovery idea as run_relay in the real relay, but it uses the
    showcase's own schema connection and topic.
    """
    conn = await open_second_connection(workspace)
    producer: Any = None
    try:
        while not state.stop_relay:
            try:
                producer = producer or await make_producer()
                result = await relay_one_batch(conn, producer, RELAY_BATCH_SIZE, topic)
                state.delivered += result.published
                state.timeline.append((state.elapsed(), state.delivered))
                if result.error is not None:
                    raise result.error
            except Exception:  # noqa: BLE001 - broker down: drop the producer, wait, and try again
                await close_producer_quietly(producer)
                producer = None
                await asyncio.sleep(RELAY_RETRY_SECONDS)
                continue
            await asyncio.sleep(RELAY_POLL_SECONDS)
    finally:
        await close_producer_quietly(producer)
        await conn.close()


async def wait_until_written(state: OutageState, count: int, deadline: float) -> None:
    """Wait until the writer has written at least `count` transactions."""
    while state.written < count:
        if time.monotonic() > deadline:
            raise TimeoutError(f"the writer did not reach {count} writes in time")
        await asyncio.sleep(0.05)


async def kill_broker_once(state: OutageState) -> tuple[float, float]:
    """
    Stop the broker, keep the writer going for a few seconds, start the broker and wait until it answers.

    Returns (seconds the broker was down, seconds it needed to answer again). The broker is ALWAYS started again in
    the finally block, even if something fails or the user presses Ctrl-C, so a showcase can never leave the shared
    broker stopped.
    """
    stop_time = time.monotonic()
    broker_stopped = False
    try:
        await asyncio.to_thread(stop_broker)
        broker_stopped = True
        state.write_limit += WRITES_DURING_OUTAGE
        await asyncio.sleep(DOWN_SECONDS)
    finally:
        if broker_stopped:
            print("Starting the broker again. If the broker is ever left stopped, run: docker compose start redpanda")
            start_broker()
    down_seconds = time.monotonic() - stop_time
    ready_seconds = await wait_for_broker()
    return down_seconds, ready_seconds


async def run_kill_loop(state: OutageState, total_writes: int, kill_count: int) -> KillTimes:
    """Run every kill cycle in turn, moving the writer's limit forward between them."""
    deadline = time.monotonic() + SECONDS_ALLOWED_PER_KILL * kill_count
    times = KillTimes()
    points = choose_kill_points(total_writes, kill_count)
    for index, point in enumerate(points):
        # Let the writer reach this kill point, then kill the broker.
        state.write_limit = point
        await wait_until_written(state, point, deadline)
        band_start = state.elapsed()
        down_seconds, ready_seconds = await kill_broker_once(state)
        times.down_seconds.append(down_seconds)
        times.ready_seconds.append(ready_seconds)
        times.bands.append((band_start, state.elapsed(), "broker down"))
        # Open the writer up to the next kill point (or to the end after the last kill).
        state.write_limit = points[index + 1] if index + 1 < len(points) else total_writes
    state.write_limit = total_writes
    return times


async def wait_for_relay_to_catch_up(workspace: Workspace, state: OutageState, total_writes: int) -> None:
    """After the last write, wait until every outbox row has been delivered (or the time is up)."""
    deadline = time.monotonic() + SECONDS_ALLOWED_PER_KILL
    while time.monotonic() < deadline:
        waiting = await workspace.connection.fetchval("SELECT count(*) FROM outbox WHERE processed_at IS NULL")
        if state.written >= total_writes and waiting == 0:
            return
        await asyncio.sleep(0.2)


async def collect_outage_result(
    workspace: Workspace, topic: str, state: OutageState, total_writes: int, kills: KillTimes,
) -> OutageResult:
    """Read the topic and the database after the run and sum everything up."""
    conn = workspace.connection
    # Read up to twice as many messages as were written: at-least-once delivery may produce duplicates, and the raw
    # count is reported next to the distinct count instead of being hidden.
    events = await read_events(topic, total_writes * 2, READ_IDLE_SECONDS)
    outbox_ids = {str(event_id) for event_id in await fetch_outbox_event_ids(conn)}
    return OutageResult(
        committed=await conn.fetchval("SELECT count(*) FROM transactions"),
        outbox_rows=await conn.fetchval("SELECT count(*) FROM outbox"),
        unprocessed=await conn.fetchval("SELECT count(*) FROM outbox WHERE processed_at IS NULL"),
        raw_count=len(events),
        kafka_ids={key for _, key, _ in events},
        outbox_ids=outbox_ids,
        kills=kills,
        timeline=state.timeline,
    )


async def run_outage_scenario(workspace: Workspace, topic: str, total_writes: int, kill_count: int) -> OutageResult:
    """
    The shared machinery of the two broker checks: a writer and a relay run together while the broker is killed
    kill_count times, then everything is compared.
    """
    await clear_workspace_tables(workspace.connection)
    state = OutageState()
    run_id = uuid.uuid4().hex[:6]
    writer = asyncio.create_task(write_with_gate(workspace.connection, state, total_writes, run_id))
    relay = asyncio.create_task(relay_until_stopped(workspace, topic, state))
    try:
        kills = await run_kill_loop(state, total_writes, kill_count)
        await writer
        await wait_for_relay_to_catch_up(workspace, state, total_writes)
    finally:
        # Stop both background tasks whatever happened above.
        state.stop_relay = True
        writer.cancel()
        await asyncio.gather(writer, relay, return_exceptions=True)
    return await collect_outage_result(workspace, topic, state, total_writes, kills)
