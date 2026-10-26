-- File: migrations/0005_create_national_outbox.sql
-- Purpose: Level 13c - create the national_outbox table for national fraud registry and administrative hold events.
--
-- The story in plain English:
--   * The existing Level 7 "outbox" table (migration 0001) has a foreign key referencing transactions(transaction_id)
--     and schema_version locked at "1". Registry events (mule reports, status changes) and 1930 administrative hold
--     requests have no transaction, so they cannot use that table.
--   * "national_outbox" stores national-level events awaiting publication to Kafka (Redpanda).
--   * event_id is the unique idempotency key: duplicate submissions with the same event_id are safely ignored.
--   * bank_key represents the message key (<run_id>:<bank_id>), establishing the bank and run namespace.
--   * topic is strictly constrained to either 'registry.events' or 'hold.requests'.
--   * Payloads contain only privacy-preserving hashes, numeric amounts in integer paise, and system identifiers.
--     No raw personal identifiers (phone numbers, PANs, names) are ever stored.
--
-- Note on gen_random_uuid(): built into PostgreSQL 13 and newer (Postgres 15 in Compose).

CREATE TABLE IF NOT EXISTS national_outbox (
  id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  event_id       UUID UNIQUE NOT NULL,
  run_id         TEXT NOT NULL,
  topic          TEXT NOT NULL CHECK (topic IN ('registry.events', 'hold.requests')),
  bank_key       TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  payload        JSONB NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  processed_at   TIMESTAMPTZ
);

-- The relay queries unprocessed events oldest-first.
-- A partial index on unprocessed rows remains tiny as processed events accumulate.
CREATE INDEX IF NOT EXISTS idx_national_outbox_unprocessed
  ON national_outbox (created_at) WHERE processed_at IS NULL;
