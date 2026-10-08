# Level 7: the transactional outbox

This page explains, in plain English, how Level 7 keeps payments and their Kafka events in step, what the relay does
when something goes wrong, and how to see it working.

## The problem: two writes that can disagree

When `/score` receives a transaction it has to do two things:

1. save the transaction in Postgres, and
2. tell the rest of the system about it by sending an event to Kafka.

If the code simply did both one after the other, a crash between them would leave the two out of step: a saved payment
nobody hears about, or an event for a payment that was never saved. Postgres and Kafka cannot be part of one shared
transaction, so we cannot ask either of them to undo the other.

## The outbox pattern

We write both things to **Postgres only**, in one database transaction:

- a row in `transactions` (the payment), and
- a row in `outbox` (the event that must be sent, with its JSON payload).

Because both rows are written in the same transaction, they are saved together or not at all. The code is
`record_transaction_with_outbox` in `api/services/outbox.py`.

A repeated request that carries the same `idempotency_key` gets the same `event_id` (a deterministic UUID), so the
second attempt is recognised as a replay and returns the original transaction instead of storing a second copy.

## The relay

The relay (`api/services/outbox_relay.py`, its own service in docker compose) is the second half. Over and over it:

1. reads the oldest outbox rows that have not been published, in order;
2. sends each one to Kafka and waits for the broker to confirm;
3. only then marks the row as processed.

If Kafka is down, nothing is marked, the events simply wait in the table, and the relay retries with growing pauses
(1 s, 2 s, 4 s ... up to `RELAY_MAX_BACKOFF_SECONDS`). The API never needs Kafka to be up.

## At-least-once delivery and the idempotent consumer

If the relay crashes after Kafka confirmed a message but before the row was marked, the message is sent again. So
delivery is **at least once**: a message may arrive twice, but it is never lost.

The consumer side copes with that. `handle_once` in `api/services/idempotent_consumer.py` writes
`(consumer_name, event_id)` into `processed_events` in the same database transaction as the real work. The second
delivery finds the note already there and does nothing, so handling an event twice changes state only once.

## Dead-letter handling: poison messages versus outages

Two very different things can go wrong when the relay sends a message:

- **An outage:** the broker is down, slow, or changing leader. The message itself is fine. The relay stops the round,
  waits (with the growing pause), and tries again later. **Outages never count as a failed attempt and never park a
  message.**
- **A poison message:** the broker rejects *this particular message*, for example because it is larger than the 1 MB
  limit. Retrying forever would block every message behind it.

Only these Kafka errors count as poison: `MessageSizeTooLargeError`, `RecordTooLargeError`, `InvalidTopicError` and
`CorruptRecordException`. Everything else is treated as "try again later".

What happens to a poison message:

1. Each rejection is counted in `outbox.attempt_count` (with the error in `last_error`).
2. While the count is below the limit the relay keeps the order: it retries on the normal 1-second poll interval and
   does not grow the outage pause.
3. At the limit the message is **parked**: a copy goes into the table `outbox_dead_letter` and
   `outbox.dead_lettered_at` is set. The relay then carries on with the messages behind it.

The limit is the setting `RELAY_MAX_ATTEMPTS` (default 5, minimum 1; see `.env.example` for the reasoning). The outbox
row is never deleted, because a replayed `idempotency_key` needs it to find the original transaction.

Parked messages live in `outbox_dead_letter` (outbox id, event id, transaction id, payload, error type and message,
attempt count, timestamps). The relay serves a Prometheus counter, `outbox_dead_lettered_total`, on port 9108 inside
the compose network; the Prometheus job is called `outbox-relay`.

## See it working

Run the Level 7 showcase. It uses its own throwaway database schema and Kafka topic and prints a checklist:

```
python -m showcase --level 7          # full run (about 65 seconds)
python -m showcase --level 7 --quick  # faster run
make show LEVEL=7                     # the same thing, where make is installed
make show-all QUICK=1                 # every level, quick mode
```

It needs Postgres and Redpanda running (`docker compose up -d postgres redpanda`). One of its checks stops the shared
Redpanda broker for a few seconds, then starts it again. A report is written to `reports/showcase/level_07/`.

## PaySim data

PaySim is deliberately **not** loaded into the `transactions` table. This is by design: PaySim is used only for the
model v1 audit in Level 9. The `source` column still allows the value `'paysim'` for forward compatibility, but nothing
in Level 7 writes it, and `scripts/ingest_paysim.py` only produces parquet files.

## Known limits

- Delivery is at-least-once only. Exactly-once is a later level.
- A parked message needs a human decision: there is no automatic replay tool yet.
