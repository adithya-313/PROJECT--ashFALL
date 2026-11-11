-- File: migrations/0015_create_labels.sql
-- Purpose: Level 16a Round 1 - labels table for ground-truth and external fraud labels.
--
-- Why two timestamps exist (Ground Rule 30):
-- arrival_time is the simulation world time (IST) when the bank learned about the label.
-- Label maturity, historical replay, and training cutoffs use ONLY arrival_time to prevent
-- lookahead bias.
-- recorded_at is the database server's real wall clock (clock_timestamp()) assigned at
-- insert time. It is strictly for transport, audit, and operational tracking, and is never
-- evaluated as simulation domain time.
--
-- Why each complaint is its own row (G3):
-- Under Ground Rule G3 (mule lifecycle and victim fan-in), an account may be the subject
-- of multiple complaints over time from distinct victims. Storing each complaint as its
-- own row (keyed by bank_id, source, label_id, account_id) preserves the full arrival history,
-- individual payment references, and evidence provenance rather than collapsing multiple
-- complaints into a single mutable account status.

-- ---------------------------------------------------------------------------
-- 1. Table: labels
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS labels (
  run_id TEXT NOT NULL,
  bank_id TEXT NOT NULL,
  label_id TEXT NOT NULL,
  source TEXT NOT NULL CHECK (source IN ('complaint', 'registry', 'analyst', 'bulk_confirmation')),
  account_id TEXT NOT NULL,
  payment_ids TEXT[] NOT NULL DEFAULT '{}',
  verdict TEXT NOT NULL CHECK (verdict IN ('fraud', 'not_fraud')),
  arrival_time TIMESTAMPTZ NOT NULL,
  region INTEGER NULL,
  batch_id TEXT NULL,
  claimed_type TEXT NULL,
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (bank_id, source, label_id, account_id),
  CHECK (left(label_id, length(run_id) + 1) = run_id || '-'),
  CHECK (region IS NULL OR region >= 0)
);

-- ---------------------------------------------------------------------------
-- 2. Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_labels_bank_run_arrival
  ON labels (bank_id, run_id, arrival_time);

-- ---------------------------------------------------------------------------
-- 3. Role Privileges for ashfall_bank_api
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.labels TO ashfall_bank_api;', current_schema());
END;
$$;

-- ---------------------------------------------------------------------------
-- 4. Row Level Security Policies
-- ---------------------------------------------------------------------------
ALTER TABLE labels ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_labels_isolation ON labels;
CREATE POLICY bank_labels_isolation ON labels
  FOR ALL TO ashfall_bank_api
  USING (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  )
  WITH CHECK (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  );

