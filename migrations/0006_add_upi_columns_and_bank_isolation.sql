-- File: migrations/0006_add_upi_columns_and_bank_isolation.sql
-- Purpose: Level 14a - add UPI transaction columns, bank API key registry, and bank row-level security.
--
-- The story in plain English:
--   * Level 14 introduces bank-facing UPI APIs where commercial banks (bank_0, bank_1, bank_2) submit
--     debit and credit legs of UPI payment events.
--   * The core "transactions" table is extended with 11 nullable columns capturing UPI event metadata:
--     payment_id, bank_id, side (debit/credit), counterparty VPA/bank, integer paise amount, event time with
--     explicit timezone offset, payment status, failure attribution domain, payee type, and channel.
--   * The source CHECK constraint is broadened from ('ecommerce', 'paysim') to also accept 'world'.
--     For 'world' records, key UPI fields are strictly required NOT NULL, and failure domains are enforced
--     exclusively when status is 'failed'. Payee_type and channel value sets are defined downstream by the
--     Level 14c projection and remain unconstrained here.
--   * Strict idempotency and replay safety are guaranteed by a composite unique constraint on (bank_id,
--     payment_id, side), paired with an index on (bank_id, account_id, event_time) for temporal history lookups.
--   * Multi-tenant security:
--     - "bank_api_keys" stores hashed API credentials for authenticating bank client requests.
--     - A dedicated cluster role "ashfall_bank_api" is provisioned with least privileges: it can only
--       SELECT and INSERT into transactions, and INSERT into outbox. It is deliberately denied read access to
--       the outbox table, preventing any bank from inspecting queued messages belonging to other institutions.
--     - Row-level security (RLS) is enabled on transactions for "ashfall_bank_api", enforcing that each bank
--       can only query and insert rows matching its session configuration 'app.bank_id'. If the setting is unset
--       or empty, RLS fails closed (no rows visible).
--     - RLS is NOT forced (NO FORCE ROW LEVEL SECURITY), ensuring the owner/superuser Level 7 path continues
--       to operate unchanged without regression.
--

-- ---------------------------------------------------------------------------
-- 1. Broaden source check constraint
-- ---------------------------------------------------------------------------
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_source;
ALTER TABLE transactions ADD CONSTRAINT chk_source CHECK (source IN ('ecommerce', 'paysim', 'world'));

-- ---------------------------------------------------------------------------
-- 2. Add nullable UPI columns in exact specified sequence
-- ---------------------------------------------------------------------------
ALTER TABLE transactions
  ADD COLUMN IF NOT EXISTS payment_id TEXT,
  ADD COLUMN IF NOT EXISTS bank_id TEXT,
  ADD COLUMN IF NOT EXISTS side TEXT,
  ADD COLUMN IF NOT EXISTS counterparty_vpa TEXT,
  ADD COLUMN IF NOT EXISTS counterparty_bank_id TEXT,
  ADD COLUMN IF NOT EXISTS amount_paise BIGINT,
  ADD COLUMN IF NOT EXISTS event_time TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS status TEXT,
  ADD COLUMN IF NOT EXISTS failure_domain TEXT,
  ADD COLUMN IF NOT EXISTS payee_type TEXT,
  ADD COLUMN IF NOT EXISTS channel TEXT;

-- ---------------------------------------------------------------------------
-- 3. Check constraints for data integrity
-- ---------------------------------------------------------------------------
-- Side must be either debit or credit when provided
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_side;
ALTER TABLE transactions ADD CONSTRAINT chk_side
  CHECK (side IS NULL OR side IN ('debit', 'credit'));

-- Status must be either success or failed when provided
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_status;
ALTER TABLE transactions ADD CONSTRAINT chk_status
  CHECK (status IS NULL OR status IN ('success', 'failed'));

-- Failure domain must be one of the five realistic domains defined in world/failures.py
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_failure_domain;
ALTER TABLE transactions ADD CONSTRAINT chk_failure_domain
  CHECK (failure_domain IS NULL OR failure_domain IN (
    'npci_central', 'sender_bank', 'receiver_bank', 'app_or_network', 'customer'
  ));

-- For 'world' payments, core UPI operational fields must all be present
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_world_required_fields;
ALTER TABLE transactions ADD CONSTRAINT chk_world_required_fields
  CHECK (source <> 'world' OR (
    payment_id IS NOT NULL AND
    bank_id IS NOT NULL AND
    side IS NOT NULL AND
    amount_paise IS NOT NULL AND
    event_time IS NOT NULL AND
    status IS NOT NULL
  ));

-- For 'world' payments, failure_domain is required if and only if status is 'failed'
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_world_failure_domain;
ALTER TABLE transactions ADD CONSTRAINT chk_world_failure_domain
  CHECK (source <> 'world' OR ((status = 'failed') = (failure_domain IS NOT NULL)));

-- Amount in paise must match numeric amount * 100 exactly, and amount_paise must be positive
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_amount_paise_match;
ALTER TABLE transactions ADD CONSTRAINT chk_amount_paise_match
  CHECK (amount_paise IS NULL OR (amount_paise > 0 AND amount * 100 = amount_paise));

-- NOTE on payee_type and channel: value sets are defined downstream by the Level 14c projection,
-- so no static CHECK constraint is placed on them here.

-- ---------------------------------------------------------------------------
-- 4. Unique idempotency constraint and query performance index
-- ---------------------------------------------------------------------------
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS uq_transactions_bank_payment_side;
ALTER TABLE transactions ADD CONSTRAINT uq_transactions_bank_payment_side
  UNIQUE (bank_id, payment_id, side);

CREATE INDEX IF NOT EXISTS idx_transactions_bank_account_event_time
  ON transactions (bank_id, account_id, event_time);

-- ---------------------------------------------------------------------------
-- 5. Bank API key authentication table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bank_api_keys (
  id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  bank_id    TEXT NOT NULL,
  key_id     TEXT NOT NULL UNIQUE,
  key_hash   TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_bank_api_keys_active
  ON bank_api_keys (bank_id) WHERE revoked_at IS NULL;

-- ---------------------------------------------------------------------------
-- 6. Role creation and least-privilege schema grants
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ashfall_bank_api') THEN
    CREATE ROLE ashfall_bank_api NOLOGIN;
  END IF;
END
$$;

DO $$
BEGIN
  -- Grant least privileges scoped to current schema.
  -- The bank API role can read and write transactions, and stage events in outbox.
  -- Crucially, it must NEVER be granted SELECT on outbox, bank_api_keys, national_outbox,
  -- or processed_events. Allowing SELECT on outbox would allow a bank to read queued outbox
  -- events from other banks, violating tenant isolation.
  EXECUTE format('GRANT USAGE ON SCHEMA %I TO ashfall_bank_api', current_schema());
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.transactions TO ashfall_bank_api', current_schema());
  EXECUTE format('GRANT INSERT ON TABLE %I.outbox TO ashfall_bank_api', current_schema());
END
$$;

-- ---------------------------------------------------------------------------
-- 7. Row Level Security on transactions
-- ---------------------------------------------------------------------------
-- Enable RLS on transactions without FORCE, ensuring owner/superuser Level 7 operations
-- bypass RLS and continue to run without regression.
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS bank_isolation ON transactions;
CREATE POLICY bank_isolation ON transactions
  FOR ALL
  TO ashfall_bank_api
  USING (bank_id = current_setting('app.bank_id', true))
  WITH CHECK (bank_id = current_setting('app.bank_id', true));

-- Comment: current_setting('app.bank_id', true) with missing_ok=true returns NULL or empty string
-- when the setting is not initialized, matching no rows and failing closed.
