-- File: migrations/0020_create_label_status_changes.sql
-- Purpose: Level 16d-1 - record that a label was revoked or contested, without ever editing the labels row.
--
-- A registry listing can be wrong (an innocent holder of a recycled phone number). When the registry revokes a
-- listing, or a customer disputes it, the bank records a status change here. The labels table is append-only, so the
-- history is kept: every change is one row with the world time the bank learned it (arrival_time). Readers drop a label
-- revoked at or before their cutoff; a change that arrives after the cutoff does not exist yet for that cutoff.
--
--   * "revoked":   the label no longer counts from arrival_time on.
--   * "contested": the label is disputed; it still counts and is reported as contested.

CREATE TABLE IF NOT EXISTS label_status_changes (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id TEXT NOT NULL,
  bank_id TEXT NOT NULL,
  source TEXT NOT NULL,
  label_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  change TEXT NOT NULL CHECK (change IN ('revoked', 'contested')),
  arrival_time TIMESTAMPTZ NOT NULL,
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  FOREIGN KEY (bank_id, source, label_id, account_id) REFERENCES labels (bank_id, source, label_id, account_id)
);

CREATE INDEX IF NOT EXISTS idx_label_status_changes_label
  ON label_status_changes (bank_id, source, label_id, account_id, arrival_time);

DO $$
BEGIN
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.label_status_changes TO ashfall_bank_api;', current_schema());
END;
$$;

ALTER TABLE label_status_changes ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_label_status_changes_isolation ON label_status_changes;
CREATE POLICY bank_label_status_changes_isolation ON label_status_changes
  FOR ALL TO ashfall_bank_api
  USING (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  )
  WITH CHECK (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  );
