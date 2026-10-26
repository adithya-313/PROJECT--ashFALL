"""
core/raw_identifier_scan.py

Multi-target auditor for personal and operational raw identifiers across national systems.
Scans five discrete data planes for a simulation run:
  1. The nationalsim store export (in-memory or exported JSON)
  2. ashFALL Postgres (national_outbox table and 13c records)
  3. Kafka payloads on registry.events and hold.requests
  4. Local JSON log output and Grafana Loki
  5. Grafana Tempo trace span attributes

Guarantees that unhashed accounts, hardware fingerprints, and telephone numbers never leak
into shared national infrastructure, centralized logging, or distributed telemetry.
"""

import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence

import asyncpg
import httpx
import orjson

from core.config import config
from core.logging import get_logger
from core.redaction import (
    ACCOUNT_PATTERN,
    FINGERPRINT_PATTERN,
    PHONE_PATTERN,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class PlaceScanResult:
    """Outcome of scanning a single data plane for raw simulated identifiers.

    What it is for:
        Records audit telemetry for one destination (e.g. Postgres, Kafka, or Tempo).
        Distinguishes between verified clean targets ('CHECKED') and targets that could
        not be reached or audited ('NOT CHECKED').

    Inputs:
        place: String identifier of the target data plane.
        status: Either 'CHECKED' or 'NOT CHECKED'.
        items_checked: Count of records, messages, or spans evaluated.
        raw_hits: Count of raw identifier leakages detected.
        reason: Optional explanation string if status is 'NOT CHECKED' or hits were detected.
        details: List of individual hit descriptors or audit notes.
    """

    place: str
    status: str
    items_checked: int
    raw_hits: int
    reason: str | None = None
    details: list[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        """A target is clean strictly when it was successfully checked with >0 items and 0 raw hits."""
        return self.status == "CHECKED" and self.items_checked > 0 and self.raw_hits == 0


@dataclass(frozen=True)
class RawScanReport:
    """Consolidated audit report across all five candidate data planes.

    What it is for:
        Aggregates results from all targets into a single decision container.
        Enforces that any 'NOT CHECKED' target prevents an overall clean verdict.
    """

    run_id: str
    places: dict[str, PlaceScanResult]
    total_items_checked: int
    total_raw_hits: int
    all_clean: bool


def find_raw_identifiers_in_text(
    text: str,
    known_raw_identifiers: set[str],
) -> list[str]:
    """Inspects arbitrary text for exact known raw identifiers and regex patterns.

    What it is for:
        Core pattern detector searching for leaked plaintext identifiers in payloads, logs, or traces.

    Inputs:
        text: String content to inspect.
        known_raw_identifiers: Set of known plaintext strings (accounts, fingerprints, phones).

    What it returns:
        List of detected raw identifier strings.
    """
    if not isinstance(text, str) or not text:
        return []

    hits: list[str] = []

    # 1. Exact matching against known run identifiers
    for raw_id in known_raw_identifiers:
        if raw_id in text:
            hits.append(raw_id)

    # 2. Pattern matching for synthetic accounts, device fingerprints, and phones
    for acc in ACCOUNT_PATTERN.findall(text):
        if acc not in hits:
            hits.append(acc)

    for fp in FINGERPRINT_PATTERN.findall(text):
        if fp not in hits:
            hits.append(fp)

    for phone in PHONE_PATTERN.findall(text):
        if phone not in hits:
            hits.append(phone)

    return hits


def _scan_json_object(
    obj: Any,
    known_raw_identifiers: set[str],
) -> list[str]:
    """Helper to traverse arbitrary JSON data structures and find raw identifiers."""
    hits: list[str] = []
    if isinstance(obj, str):
        hits.extend(find_raw_identifiers_in_text(obj, known_raw_identifiers))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                hits.extend(find_raw_identifiers_in_text(k, known_raw_identifiers))
            hits.extend(_scan_json_object(v, known_raw_identifiers))
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            hits.extend(_scan_json_object(item, known_raw_identifiers))
    return hits


def scan_nationalsim_store(
    store_export: dict[str, Any] | None,
    known_raw_identifiers: set[str],
) -> PlaceScanResult:
    """Audits the exported registry store for raw identifiers.

    What it is for:
        Verifies that nationalsim internal data structures store only pre-hashed tokens.
    """
    if store_export is None:
        return PlaceScanResult(
            place="nationalsim_store",
            status="NOT CHECKED",
            items_checked=0,
            raw_hits=0,
            reason="Store export not supplied",
        )

    entries = store_export.get("entries", [])
    feedback = store_export.get("feedback", [])
    resolutions = store_export.get("resolutions", [])
    total_items = len(entries) + len(feedback) + len(resolutions)
    if total_items == 0:
        return PlaceScanResult(
            place="nationalsim_store",
            status="EMPTY",
            items_checked=0,
            raw_hits=0,
            reason="Store export is reachable but contains 0 items",
        )

    hits: list[str] = []
    for item in entries + feedback + resolutions:
        item_hits = _scan_json_object(item, known_raw_identifiers)
        hits.extend(item_hits)

    return PlaceScanResult(
        place="nationalsim_store",
        status="CHECKED",
        items_checked=total_items,
        raw_hits=len(hits),
        details=hits[:20],
    )


async def scan_ashfall_postgres(
    run_id: str,
    known_raw_identifiers: set[str],
    database_url: str | None = None,
) -> PlaceScanResult:
    """Audits ashFALL Postgres national_outbox payloads for raw identifiers."""
    import os

    db_url = (
        database_url
        or os.environ.get("TEST_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or config.database_url
    )
    if "@postgres:" in db_url:
        db_url = db_url.replace("@postgres:", "@localhost:")
    try:
        conn = await asyncpg.connect(db_url, timeout=3.0)
    except (asyncpg.PostgresError, OSError) as err:
        return PlaceScanResult(
            place="ashfall_postgres",
            status="NOT CHECKED",
            items_checked=0,
            raw_hits=0,
            reason=f"Postgres unreachable: {type(err).__name__}: {err}",
        )

    try:
        rows = await conn.fetch(
            "SELECT id, payload FROM national_outbox WHERE run_id = $1",
            run_id,
        )
        if len(rows) == 0:
            return PlaceScanResult(
                place="ashfall_postgres",
                status="EMPTY",
                items_checked=0,
                raw_hits=0,
                reason=f"national_outbox is reachable but contains 0 rows for run_id '{run_id}'",
            )

        hits: list[str] = []
        for row in rows:
            payload_str = (
                row["payload"]
                if isinstance(row["payload"], str)
                else orjson.dumps(row["payload"]).decode("utf-8")
            )
            row_hits = find_raw_identifiers_in_text(payload_str, known_raw_identifiers)
            hits.extend(row_hits)

        return PlaceScanResult(
            place="ashfall_postgres",
            status="CHECKED",
            items_checked=len(rows),
            raw_hits=len(hits),
            details=hits[:20],
        )
    except asyncpg.PostgresError as err:
        return PlaceScanResult(
            place="ashfall_postgres",
            status="NOT CHECKED",
            items_checked=0,
            raw_hits=0,
            reason=f"Postgres query failed: {err}",
        )
    finally:
        await conn.close()


async def scan_kafka_payloads(
    run_id: str,
    known_raw_identifiers: set[str],
    bootstrap_servers: str | None = None,
    topics: Sequence[str] = ("registry.events", "hold.requests"),
) -> PlaceScanResult:
    """Audits Kafka topics for raw identifiers on published national event payloads."""
    from aiokafka import AIOKafkaConsumer

    servers = bootstrap_servers or config.kafka_bootstrap_servers
    try:
        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=servers,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            request_timeout_ms=3000,
        )
        await consumer.start()
    except Exception as err:
        return PlaceScanResult(
            place="kafka_payloads",
            status="NOT CHECKED",
            items_checked=0,
            raw_hits=0,
            reason=f"Kafka unreachable: {type(err).__name__}: {err}",
        )

    try:
        hits: list[str] = []
        items_checked = 0
        for _ in range(10):
            batches = await consumer.getmany(timeout_ms=1000, max_records=1000)
            if not batches or sum(len(recs) for recs in batches.values()) == 0:
                break
            for records in batches.values():
                for record in records:
                    val_text = record.value.decode("utf-8", errors="replace") if record.value else ""
                    key_text = record.key.decode("utf-8", errors="replace") if record.key else ""

                    # Filter strictly for messages matching the given simulation run_id
                    msg_run_id = None
                    try:
                        payload_dict = orjson.loads(val_text)
                        if isinstance(payload_dict, dict):
                            msg_run_id = payload_dict.get("run_id")
                    except Exception:
                        pass

                    if not msg_run_id and key_text:
                        if ":" in key_text:
                            msg_run_id = key_text.split(":", 1)[0]
                        elif key_text == run_id:
                            msg_run_id = run_id

                    if msg_run_id != run_id:
                        continue

                    items_checked += 1
                    hits.extend(find_raw_identifiers_in_text(val_text, known_raw_identifiers))
                    hits.extend(find_raw_identifiers_in_text(key_text, known_raw_identifiers))

        if items_checked == 0:
            return PlaceScanResult(
                place="kafka_payloads",
                status="EMPTY",
                items_checked=0,
                raw_hits=0,
                reason=f"Kafka is reachable but contains 0 messages on topics {topics} for run_id '{run_id}'",
            )

        return PlaceScanResult(
            place="kafka_payloads",
            status="CHECKED",
            items_checked=items_checked,
            raw_hits=len(hits),
            details=hits[:20],
        )
    finally:
        await consumer.stop()


async def scan_logs_and_loki(
    run_id: str,
    known_raw_identifiers: set[str],
    local_log_lines: list[str] | None = None,
    loki_url: str | None = "http://localhost:3100",
    place_name: str = "logs_and_loki",
) -> PlaceScanResult:
    """Audits local JSON logs and queries Loki HTTP query API for raw identifiers."""
    items_checked = 0
    hits: list[str] = []
    loki_reachable = False
    loki_error: str | None = None

    # 1. Audit local JSON log lines if provided
    if local_log_lines is not None:
        for line in local_log_lines:
            items_checked += 1
            hits.extend(find_raw_identifiers_in_text(line, known_raw_identifiers))

    # 2. Query Loki HTTP query API if configured
    if loki_url is not None:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(
                    f"{loki_url.rstrip('/')}/loki/api/v1/query_range",
                    params={"query": f'{{environment=~".+"}} |= "{run_id}"', "limit": 100},
                )
                if resp.status_code == 200:
                    loki_reachable = True
                    data = resp.json()
                    streams = data.get("data", {}).get("result", [])
                    for stream in streams:
                        for val in stream.get("values", []):
                            items_checked += 1
                            line_text = val[1] if len(val) > 1 else ""
                            hits.extend(find_raw_identifiers_in_text(line_text, known_raw_identifiers))
                else:
                    loki_error = f"Loki returned HTTP {resp.status_code}"
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, OSError) as err:
            loki_error = f"Loki unreachable: {type(err).__name__}"

    # If local logs were not supplied and Loki was not reached, report NOT CHECKED
    if local_log_lines is None and not loki_reachable:
        return PlaceScanResult(
            place=place_name,
            status="NOT CHECKED",
            items_checked=0,
            raw_hits=0,
            reason=loki_error or "Logs and Loki unreachable",
        )

    if items_checked == 0:
        return PlaceScanResult(
            place=place_name,
            status="EMPTY",
            items_checked=0,
            raw_hits=0,
            reason=f"Logging source is reachable but contains 0 log records for run_id '{run_id}'",
        )

    return PlaceScanResult(
        place=place_name,
        status="CHECKED",
        items_checked=items_checked,
        raw_hits=len(hits),
        reason=None if (loki_reachable or loki_url is None) else f"Checked local logs only ({loki_error})",
        details=hits[:20],
    )


async def scan_tempo_spans(
    run_id: str,
    known_raw_identifiers: set[str],
    tempo_url: str | None = "http://localhost:3200",
    local_spans: list[dict[str, Any]] | None = None,
    place_name: str = "tempo_spans",
) -> PlaceScanResult:
    """Audits Tempo HTTP search API and local span attributes for raw identifiers."""
    items_checked = 0
    hits: list[str] = []
    tempo_reachable = False
    tempo_error: str | None = None

    # 1. Audit local span attributes if provided
    if local_spans is not None:
        for span in local_spans:
            items_checked += 1
            hits.extend(_scan_json_object(span, known_raw_identifiers))

    # 2. Query Tempo HTTP search API if configured
    if tempo_url is not None:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(
                    f"{tempo_url.rstrip('/')}/api/search",
                    params={"tags": f"run_id={run_id}", "limit": 50},
                )
                if resp.status_code == 200:
                    tempo_reachable = True
                    data = resp.json()
                    traces = data.get("traces", [])
                    items_checked += len(traces)
                    for trace in traces:
                        hits.extend(_scan_json_object(trace, known_raw_identifiers))
                else:
                    tempo_error = f"Tempo search returned HTTP {resp.status_code}"
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, OSError) as err:
            tempo_error = f"Tempo unreachable at {tempo_url}: {type(err).__name__}"

    # If local spans were not supplied and Tempo was not reached, report NOT CHECKED
    if local_spans is None and not tempo_reachable:
        return PlaceScanResult(
            place=place_name,
            status="NOT CHECKED",
            items_checked=0,
            raw_hits=0,
            reason=tempo_error or "Tempo unreachable",
        )

    if items_checked == 0:
        return PlaceScanResult(
            place=place_name,
            status="EMPTY",
            items_checked=0,
            raw_hits=0,
            reason=f"Tracing source is reachable but contains 0 spans for run_id '{run_id}'",
        )

    return PlaceScanResult(
        place=place_name,
        status="CHECKED",
        items_checked=items_checked,
        raw_hits=len(hits),
        reason=None if (tempo_reachable or tempo_url is None) else f"Checked local spans only ({tempo_error})",
        details=hits[:20],
    )


async def scan_all_five_places(
    run_id: str,
    known_raw_identifiers: set[str],
    store_export: dict[str, Any] | None = None,
    local_log_lines: list[str] | None = None,
    local_spans: list[dict[str, Any]] | None = None,
    database_url: str | None = None,
    kafka_servers: str | None = None,
    loki_url: str | None = "http://localhost:3100",
    tempo_url: str | None = "http://localhost:3200",
    mode: str = "full",
) -> RawScanReport:
    """Runs concurrent audit across all five candidate data planes for a run."""
    # Place 1: Store export
    res_store = scan_nationalsim_store(store_export, known_raw_identifiers)

    # Place 2: Postgres
    res_pg = await scan_ashfall_postgres(run_id, known_raw_identifiers, database_url)

    # Place 3: Kafka
    res_kafka = await scan_kafka_payloads(run_id, known_raw_identifiers, kafka_servers)

    if mode == "quick":
        # Quick mode (CI: Postgres and Redpanda only):
        # Places 4 and 5 are audited from local run logs and spans without querying Loki or Tempo.
        res_logs = await scan_logs_and_loki(
            run_id=run_id,
            known_raw_identifiers=known_raw_identifiers,
            local_log_lines=local_log_lines,
            loki_url=None,
            place_name="local_logs",
        )
        res_tempo = await scan_tempo_spans(
            run_id=run_id,
            known_raw_identifiers=known_raw_identifiers,
            tempo_url=None,
            local_spans=local_spans,
            place_name="local_spans",
        )
        places = {
            "nationalsim_store": res_store,
            "ashfall_postgres": res_pg,
            "kafka_payloads": res_kafka,
            "local_logs": res_logs,
            "local_spans": res_tempo,
        }
    else:
        # Full mode: queries live Loki and live Tempo by run_id
        res_logs = await scan_logs_and_loki(
            run_id=run_id,
            known_raw_identifiers=known_raw_identifiers,
            local_log_lines=local_log_lines,
            loki_url=loki_url,
            place_name="logs_and_loki",
        )
        res_tempo = await scan_tempo_spans(
            run_id=run_id,
            known_raw_identifiers=known_raw_identifiers,
            tempo_url=tempo_url,
            local_spans=local_spans,
            place_name="tempo_spans",
        )
        places = {
            "nationalsim_store": res_store,
            "ashfall_postgres": res_pg,
            "kafka_payloads": res_kafka,
            "logs_and_loki": res_logs,
            "tempo_spans": res_tempo,
        }

    total_items = sum(p.items_checked for p in places.values())
    total_hits = sum(p.raw_hits for p in places.values())
    # All clean strictly requires every place was CHECKED (>0 items) and had 0 hits
    all_clean = all(p.is_clean() for p in places.values())

    return RawScanReport(
        run_id=run_id,
        places=places,
        total_items_checked=total_items,
        total_raw_hits=total_hits,
        all_clean=all_clean,
    )


def scan_all_five_places_sync(
    run_id: str,
    known_raw_identifiers: set[str],
    store_export: dict[str, Any] | None = None,
    local_log_lines: list[str] | None = None,
    local_spans: list[dict[str, Any]] | None = None,
    database_url: str | None = None,
    kafka_servers: str | None = None,
    loki_url: str | None = "http://localhost:3100",
    tempo_url: str | None = "http://localhost:3200",
    mode: str = "full",
) -> RawScanReport:
    """Synchronous entry point for audit invocation."""
    return asyncio.run(
        scan_all_five_places(
            run_id=run_id,
            known_raw_identifiers=known_raw_identifiers,
            store_export=store_export,
            local_log_lines=local_log_lines,
            local_spans=local_spans,
            database_url=database_url,
            kafka_servers=kafka_servers,
            loki_url=loki_url,
            tempo_url=tempo_url,
            mode=mode,
        )
    )


def main() -> None:
    """CLI entry point (python -m core.raw_identifier_scan)."""
    parser = argparse.ArgumentParser(description="Audit five data planes for raw identifier leaks.")
    parser.add_argument("--run-id", required=True, help="Simulation run identifier string.")
    parser.add_argument(
        "--known-ids",
        nargs="*",
        default=[],
        help="List of known raw identifier strings to search for.",
    )
    args = parser.parse_args()

    known_set = set(args.known_ids)
    report = scan_all_five_places_sync(run_id=args.run_id, known_raw_identifiers=known_set)

    print(f"\nRAW IDENTIFIER AUDIT REPORT FOR RUN: {report.run_id}")
    print("=" * 70)
    for name, p in report.places.items():
        if p.status == "CHECKED":
            print(f"  [{p.status}] {name:20s}: {p.items_checked:5d} items checked, {p.raw_hits:3d} raw hits")
        else:
            print(f"  [{p.status}] {name:20s}: reason: {p.reason}")
    print("=" * 70)
    print(f"Total items checked: {report.total_items_checked}")
    print(f"Total raw hits:     {report.total_raw_hits}")
    print(f"Overall Clean:       {'YES' if report.all_clean else 'NO'}\n")

    if not report.all_clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
