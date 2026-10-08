-- File: migrations/0004_add_outbox_dead_letter.sql
-- Purpose: Level 7 - let the outbox relay give up on a "poison" message (one that Kafka rejects for its own sake,
--          for example because it is too large) WITHOUT blocking the messages behind it.
--
-- How it works, in plain English:
--   * Four new columns on outbox remember how often a message was tried and why it failed. Rows that were already
--     in the table simply start with attempt_count 0, so this migration changes no existing behaviour.
--   * When a message has failed too many times it is "parked": a copy goes into the new table outbox_dead_letter
--     and outbox.dead_lettered_at is set, so the relay stops picking it up.
--   * The outbox row is NEVER deleted. A repeated request with the same idempotency_key looks up outbox.event_id to
--     find the original transaction, so that row must stay.
--
-- This migration is additive only: it does not touch the transactions table or the payload shape.

ALTER TABLE outbox
  ADD COLUMN IF NOT EXISTS attempt_count    integer NOT NULL DEFAULT 0,  -- how many times Kafka rejected this message
  ADD COLUMN IF NOT EXISTS last_error       text,                        -- the most recent rejection, shortened
  ADD COLUMN IF NOT EXISTS first_failed_at  timestamptz,                 -- when the first rejection happened
  ADD COLUMN IF NOT EXISTS dead_lettered_at timestamptz;                 -- set when the message was parked

-- The parked messages. One row per parked outbox row (the UNIQUE rule below stops a row from being parked twice).
CREATE TABLE IF NOT EXISTS outbox_dead_letter (
  id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  outbox_id        BIGINT NOT NULL REFERENCES outbox(id),
  event_id         UUID NOT NULL,
  transaction_id   UUID NOT NULL,
  payload          JSONB NOT NULL,
  error_type       TEXT NOT NULL,
  error_message    TEXT NOT NULL,
  attempt_count    INTEGER NOT NULL,
  first_failed_at  TIMESTAMPTZ NOT NULL,
  dead_lettered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_outbox_dead_letter_outbox_id UNIQUE (outbox_id)
);
