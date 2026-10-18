"""
WHAT: Builds "events-v2" lines for an interactive run: each line is Level 11's 14-field event record, made by Level
11's own event_to_record, plus two ids: payment_id (this event) and planned_payment_id (the payment it is a piece of).
No file is written here; the files that go with fraud truth arrive in Level 12b.

WHY: A bank feed must be able to name every event and group the pieces of one split payment, so later levels can say
"this payment was held" or "these three pieces were one transfer". The line deliberately carries no field saying
whether an event came from fraud: rules 15 and 29 keep the truth out of anything a model or bank can read.

TERM: events-v1: Level 11's event line format, 14 fields such as minute, accounts, amount, status and reason.
TERM: events-v2: events-v1 plus payment_id and planned_payment_id, 16 fields. It is the name of this world event line
      format only and has nothing to do with the outbox message schema version "2" used by the Level 7 services.
TERM: payment id: the name of one event, e.g. "<run id>-000042" for the 43rd event.
TERM: planned payment id: the name shared by all pieces of one payment, e.g. "<run id>-p-000017".
TERM: JSON line: one record written as compact JSON text with sorted keys, ending in a newline.
"""

import hashlib
import json
from pathlib import Path

from world.fast_mode import event_to_record
from world.interactive_records import InteractiveWorldOutput

# The format name recorded with these lines when they are written to files in a later part.
EVENTS_V2_FORMAT: str = "events-v2"


def interactive_event_lines(output: InteractiveWorldOutput) -> list[str]:
    """
    WHAT: Returns one events-v2 JSON line per event, in run order. Each line is Level 11's record for the event (from
    event_to_record, exactly as event_lines builds it) with payment_id and planned_payment_id added, written with
    sorted keys and no spaces, ending in a newline.

    WHY: Reusing Level 11's own record builder means the 14 shared fields can never drift from events-v1. With an empty
    fraud plan, removing the two ids from a line gives back Level 11's line exactly.

    Args:
        output: An InteractiveWorldOutput.

    Returns:
        The lines, e.g. 2,915 lines for the seed-101 tiny world.

    Example:
        The first line of the seed-101 world starts '{"amount_paise":19192,"failure_domain":null,...' and has 16 keys.
    """
    # The events and the lists that line up with them.
    result = output.result
    run_result = result.run_result
    lines: list[str] = []
    for position, event in enumerate(run_result.events):
        # Level 11's record for this event, then the two ids.
        record = event_to_record(
            event, run_result.event_purposes[position], run_result.event_sender_devices[position], output.plan.clock
        )
        record["payment_id"] = result.payment_ids[position]
        record["planned_payment_id"] = result.planned_payment_ids[position]
        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return lines
 
 
def write_events_v2_file(output: InteractiveWorldOutput, path: Path) -> str:
    """
    WHAT: Writes all chronological payment events from an interactive simulation run to an events-v2 JSON Lines file.
    Each line is serialized with 16 keys, sorted keys and compact separators into a newline-terminated line.
    After completing writing, it reads back the file bytes and calculates its SHA-256 cryptographic digest.

    WHY: Emits transaction events in a standard JSON Lines representation containing both payment_id and
    planned_payment_id for downstream tracking without leaking ground-truth fraud simulation labels.
    Enforcing explicit UTF-8 encoding and LF newlines ensures bit-for-bit identical outputs across operating systems.
    Returning the SHA-256 digest allows immediate verification of file content and inclusion in the manifest.

    Args:
        output: InteractiveWorldOutput container holding simulated payment events, telemetry, clock, and payment IDs.
        path: Destination filesystem Path where the events-v2 file will be written.

    Returns:
        Hexadecimal string representing the SHA-256 cryptographic digest of the written events file.

    Example:
        # Given an interactive run output:
        # digest = write_events_v2_file(output, Path("runs/test/events.jsonl"))
        # assert len(digest) == 64
    """
    # 1. Build deterministic JSON lines for all simulated events with 16 fields.
    lines = interactive_event_lines(output)

    # 2. Open destination file with explicit UTF-8 encoding and standard Unix line endings.
    with open(path, "w", encoding="utf-8", newline="\n") as events_file:
        for line in lines:
            events_file.write(line)

    # 3. Read back raw file bytes to calculate authoritative SHA-256 cryptographic digest.
    return hashlib.sha256(path.read_bytes()).hexdigest()
