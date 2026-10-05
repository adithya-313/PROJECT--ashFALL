-- File: migrations/0001_create_transactions_and_outbox.sql
-- Purpose: Level 7 - create the two core tables for the transactional outbox pattern.
--
-- The story in plain English:
--   * "transactions" is where every payment we receive is stored, whether it came from the e-commerce store or from
--     the PaySim dataset. Fields that only PaySim has (balances, type) are left empty (NULL) for e-commerce rows.
--   * "outbox" holds one "event to publish" row per transaction. The service will write the transaction row AND its
--     outbox row inside ONE database transaction, so either both exist or neither does. A background relay will
--     later read the rows that have not been processed yet and send them to Kafka. This way we never save a
--     transaction and then lose its event (or send an event for a transaction that was never saved).
--
-- Note on gen_random_uuid(): it is built into PostgreSQL 13 and newer, so no extension (pgcrypto) is needed on the
-- Postgres 15 we run. This was checked against the running container before writing this file.

-- ---------------------------------------------------------------------------
-- transactions: one row per payment.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
  id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  -- A public, unguessable ID for the transaction. Other tables and Kafka events refer to this, not to "id".
  transaction_id     UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
  source             TEXT NOT NULL,            -- 'ecommerce' or 'paysim'
  account_id         TEXT NOT NULL,
  merchant_id        TEXT,
  amount             NUMERIC(18,2) NOT NULL,   -- exact money, never float
  device_fingerprint TEXT,
  geo_coordinates    TEXT,
  txn_type           TEXT,                     -- PaySim 'type'; NULL for ecommerce
  old_balance_orig   NUMERIC(18,2),            -- PaySim only, NULL for ecommerce
  new_balance_orig   NUMERIC(18,2),            -- PaySim only
  old_balance_dest   NUMERIC(18,2),            -- PaySim only
  new_balance_dest   NUMERIC(18,2),            -- PaySim only
  fraud_score        DOUBLE PRECISION,         -- from /score, nullable
  model_version      TEXT,
  details            JSONB NOT NULL DEFAULT '{}'::jsonb,  -- source-specific extras
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- Guard rails: the database itself refuses bad values, even if some code forgets to check.
  CONSTRAINT chk_source CHECK (source IN ('ecommerce','paysim')),
  CONSTRAINT chk_amount_nonneg CHECK (amount >= 0)
);

-- ---------------------------------------------------------------------------
-- outbox: events waiting to be published to Kafka.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS outbox (
  id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  -- Every event belongs to a saved transaction. The database enforces that link.
  transaction_id UUID NOT NULL REFERENCES transactions(transaction_id),
  event_type     TEXT NOT NULL,
  payload        JSONB NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  processed_at   TIMESTAMPTZ                   -- stays NULL until the relay has published the event
);

-- ---------------------------------------------------------------------------
-- Indexes: make the two most common lookups fast.
-- ---------------------------------------------------------------------------
-- The relay constantly asks "which events are not processed yet, oldest first?". This index only contains the
-- unprocessed rows, so it stays tiny no matter how many events have already been published.
CREATE INDEX IF NOT EXISTS idx_outbox_unprocessed
  ON outbox (created_at) WHERE processed_at IS NULL;

-- Feature code asks "what has this account done recently?", so look up by account and time.
CREATE INDEX IF NOT EXISTS idx_transactions_account_time
  ON transactions (account_id, created_at);
