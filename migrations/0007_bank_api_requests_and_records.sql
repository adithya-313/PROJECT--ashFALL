-- File: migrations/0007_bank_api_requests_and_records.sql
-- Purpose: Level 14b - bank API request nonces, idempotency keys, and tenant records with row-level security.
--
-- The story in plain English:
--   * Level 14b builds the security and storage foundation for commercial bank API endpoints.
--   * To defend against replay attacks on the transport layer, "bank_request_nonces" tracks random nonces
--     presented by banks within the signature freshness window. Because authentication occurs on the database
--     owner connection before any bank-scoped tenant transaction is assumed, the tenant role "ashfall_bank_api"
--     is granted NO privileges on bank_request_nonces (least privilege; a tenant cannot read other banks' nonces).
--   * To support idempotent request handling with conflict detection, "bank_idempotency_keys" stores the SHA-256
--     digest of each request body and cached response. The "ashfall_bank_api" role is granted SELECT, INSERT,
--     and UPDATE privileges on bank_idempotency_keys to check and record response states within bank-scoped transactions.
--   * Four specialized tenant record tables store commercial banking domain entities:
--     - "bank_decision_requests": shadow/live evaluation requests from banks for transaction decisioning.
--     - "bank_account_events": life-cycle events (device bindings, SIM swaps, app reinstalls, account openings/activations).
--     - "bank_complaints": fraud complaints referencing at least one payment ID.
--     - "bank_feedback": post-transaction ground truth feedback (confirmed fraud, not fraud, customer dispute).
--   * Strict data integrity is enforced via CHECK constraints on decisions ('unscored'), modes ('shadow', 'live'),
--     event kinds, feedback outcomes, feedback note lengths (<= 1000 chars), and non-empty complaint payment arrays.
--   * Row-Level Security (RLS) is enabled on tables (b) through (f), with policies ensuring that any query
--     or insert by "ashfall_bank_api" strictly filters by current_setting('app.bank_id', true).
--

-- ---------------------------------------------------------------------------
-- 1. Table: bank_request_nonces (Replay defense)
-- ---------------------------------------------------------------------------
-- Nonce tracking table evaluated during transport authentication on the owner connection.
-- The tenant role ashfall_bank_api receives NO GRANTS on this table.
CREATE TABLE IF NOT EXISTS bank_request_nonces (
  bank_id TEXT NOT NULL,
  nonce TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (bank_id, nonce)
);

CREATE INDEX IF NOT EXISTS idx_bank_request_nonces_created_at
  ON bank_request_nonces (created_at);

-- ---------------------------------------------------------------------------
-- 2. Table: bank_idempotency_keys (Idempotency and response caching)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bank_idempotency_keys (
  bank_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  response_status INTEGER NULL,
  response_body JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (bank_id, idempotency_key)
);

-- ---------------------------------------------------------------------------
-- 3. Table: bank_decision_requests
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bank_decision_requests (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  bank_id TEXT NOT NULL,
  payment_id TEXT NOT NULL,
  request JSONB NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('unscored')),
  mode TEXT NOT NULL CHECK (mode IN ('shadow', 'live')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_bank_decision_requests_bank_payment UNIQUE (bank_id, payment_id)
);

-- ---------------------------------------------------------------------------
-- 4. Table: bank_account_events
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bank_account_events (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  bank_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN (
    'device_binding_change',
    'sim_swap',
    'app_reinstall',
    'os_update',
    'account_opened',
    'account_reactivated'
  )),
  event_time TIMESTAMPTZ NOT NULL,
  details JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_bank_account_events_bank_event UNIQUE (bank_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_bank_account_events_lookup
  ON bank_account_events (bank_id, account_id, event_time);

-- ---------------------------------------------------------------------------
-- 5. Table: bank_complaints
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bank_complaints (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  bank_id TEXT NOT NULL,
  complaint_id TEXT NOT NULL,
  complainant_account TEXT NOT NULL,
  reported_account TEXT NOT NULL,
  reported_bank_id TEXT NOT NULL,
  payment_ids TEXT[] NOT NULL CHECK (cardinality(payment_ids) >= 1),
  claimed_type TEXT NOT NULL,
  received_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_bank_complaints_bank_complaint UNIQUE (bank_id, complaint_id)
);

-- ---------------------------------------------------------------------------
-- 6. Table: bank_feedback
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bank_feedback (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  bank_id TEXT NOT NULL,
  feedback_id TEXT NOT NULL,
  payment_id TEXT NOT NULL,
  outcome TEXT NOT NULL CHECK (outcome IN ('confirmed_fraud', 'not_fraud', 'customer_dispute')),
  note TEXT NULL CHECK (note IS NULL OR length(note) <= 1000),
  event_time TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_bank_feedback_bank_feedback UNIQUE (bank_id, feedback_id)
);

-- ---------------------------------------------------------------------------
-- 7. Role Privileges for ashfall_bank_api
-- ---------------------------------------------------------------------------
-- Note: bank_request_nonces receives NO grant to ashfall_bank_api. Request authentication,
-- nonce verification, and retention cleanup run exclusively on the owner connection before
-- a bank-scoped transaction is established.
DO $$
BEGIN
  EXECUTE format('GRANT SELECT, INSERT, UPDATE ON TABLE %I.bank_idempotency_keys TO ashfall_bank_api;', current_schema());
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.bank_decision_requests TO ashfall_bank_api;', current_schema());
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.bank_account_events TO ashfall_bank_api;', current_schema());
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.bank_complaints TO ashfall_bank_api;', current_schema());
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.bank_feedback TO ashfall_bank_api;', current_schema());
END
$$;

-- ---------------------------------------------------------------------------
-- 8. Row Level Security Policies
-- ---------------------------------------------------------------------------
ALTER TABLE bank_idempotency_keys ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_isolation_bank_idempotency_keys ON bank_idempotency_keys;
CREATE POLICY bank_isolation_bank_idempotency_keys ON bank_idempotency_keys
  FOR ALL TO ashfall_bank_api
  USING (bank_id = current_setting('app.bank_id', true))
  WITH CHECK (bank_id = current_setting('app.bank_id', true));

ALTER TABLE bank_decision_requests ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_isolation_bank_decision_requests ON bank_decision_requests;
CREATE POLICY bank_isolation_bank_decision_requests ON bank_decision_requests
  FOR ALL TO ashfall_bank_api
  USING (bank_id = current_setting('app.bank_id', true))
  WITH CHECK (bank_id = current_setting('app.bank_id', true));

ALTER TABLE bank_account_events ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_isolation_bank_account_events ON bank_account_events;
CREATE POLICY bank_isolation_bank_account_events ON bank_account_events
  FOR ALL TO ashfall_bank_api
  USING (bank_id = current_setting('app.bank_id', true))
  WITH CHECK (bank_id = current_setting('app.bank_id', true));

ALTER TABLE bank_complaints ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_isolation_bank_complaints ON bank_complaints;
CREATE POLICY bank_isolation_bank_complaints ON bank_complaints
  FOR ALL TO ashfall_bank_api
  USING (bank_id = current_setting('app.bank_id', true))
  WITH CHECK (bank_id = current_setting('app.bank_id', true));

ALTER TABLE bank_feedback ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_isolation_bank_feedback ON bank_feedback;
CREATE POLICY bank_isolation_bank_feedback ON bank_feedback
  FOR ALL TO ashfall_bank_api
  USING (bank_id = current_setting('app.bank_id', true))
  WITH CHECK (bank_id = current_setting('app.bank_id', true));
