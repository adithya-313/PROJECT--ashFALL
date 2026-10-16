# Level 11C-4: Live Mode Decisions

## 1. What Live Mode Is (Plan Level 11 Task 16)
Live mode streams simulated transactions event by event from an already decided simulation run to an HTTP receiver in deterministic file order. It connects offline batch world simulations to real-time stream consumers (such as fraud detection systems) at a chosen pacing rate. When no target URL is provided, live mode automatically starts a built-in receiver on local host address 127.0.0.1, enabling self-contained offline testing without Docker or external network dependencies. Fast-mode batch file generation remains completely unchanged and byte-identical.

## 2. The Decisions (Plan Author and Owner, 2026-10-16)
The following architectural decisions were established by the plan author and owner on 2026-10-16:
- **Target Receiver and Default Host:** Target can be any HTTP endpoint URL provided by the operator via `--target`. When omitted, it defaults strictly to an internal ephemeral receiver bound to `127.0.0.1`. Nothing leaves the machine on its own unless an operator explicitly designates an external target.
- **Envelope Wrapping:** Each HTTP POST request carries a single event wrapped in a signed JSON envelope:
  `{"event_id": "<first 16 hex of events_sha256>-<position, 6 digits>", "event": <events-v1 record>}`.
  The event payload contains the exact 14 standardized keys produced by fast mode.
- **Pacing and Speed:** Replay rate is controlled via `--speed N` (N simulated seconds per real second) or `--speed max` (no waiting delays). Specifying `--speed` is mandatory whenever `--live` is used, preventing accidental unbounded stalls or runaway network flooding.
- **Zero Skips and Exponential Backoff Retries:** On communication or HTTP failures, live mode attempts delivery up to 4 times with exponential backoff delays (starting at 0.5 seconds, scaling by a growth factor of 2: 0.5s, 1.0s, 2.0s) and a 5-second request timeout. If retries are exhausted, live mode halts immediately, raises `LiveSendError`, and reports the stopped position and event id. An event is never skipped.
- **Pluggable EventSender Interface:** Replay transmission sits behind a small `EventSender` protocol (`send(event_id, record) -> dict | None`). This allows unit testing with zero-overhead in-memory fakes and leaves room for bidirectional response collection.

## 3. Why Events Get an Envelope Instead of a New Field
The existing `events.jsonl` format defines exactly 14 keys and must remain byte-identical across all runs. Adding an event id key directly to the transaction record would break file hash parity and golden regression benchmarks (such as SHA-256 `24fc0e25009d79462d508f040a109620dd4308486cc78aee589393fe921f92fd` for seed 101). Wrapping records in an external envelope `{"event_id": ..., "event": ...}` keeps the core transaction record byte-identical while binding each transmission to the run manifest hash and its zero-padded file index. Receivers can safely drop duplicate event ids to guarantee at-least-once delivery.

## 4. Why v1 /score Is Not Used
The v1 `/score` endpoint belongs to frozen historical levels and is scheduled for removal in Level 20. Live mode connects to independent downstream receivers and must not couple to legacy scoring APIs or obsolete schemas.

## 5. Known Limit
Live mode is a one-way replay of a world already decided, which is fine for shadow mode; from Level 19, when banks act on decisions, the world must react to each answer, and the EventSender interface (send an event, get an optional answer) is where Level 14 adds an ask-before-each-payment mode without a rewrite.
