-- File: migrations/0003_create_processed_events.sql
-- Purpose: Level 7 - the bookkeeping that lets a Kafka consumer handle every event exactly once, even when Kafka
--          delivers the same event twice.
--
-- The story in plain English:
--   Kafka (with our outbox relay) promises "at least once": an event may arrive more than once, for example after a
--   crash or a retry. A consumer must not count, charge or learn from the same event twice. So each consumer writes
--   down "I have handled event X" in processed_events, in the SAME database transaction as its real work. The
--   PRIMARY KEY (consumer_name, event_id) means the second attempt to write that note is refused, which is how the
--   duplicate is recognised. consumer_name keeps the notes of different consumers apart: the same event must be
--   handled once PER consumer, not once overall.

CREATE TABLE IF NOT EXISTS processed_events (
  consumer_name TEXT NOT NULL,
  event_id      UUID NOT NULL,
  processed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (consumer_name, event_id)
);

-- Only for the example consumer (api/services/demo_counter_consumer.py): a counter we can read afterwards to prove
-- that a duplicated message was counted once. Real consumers will have their own tables.
CREATE TABLE IF NOT EXISTS demo_event_counts (
  consumer_name  TEXT PRIMARY KEY,
  events_counted BIGINT NOT NULL DEFAULT 0
);
