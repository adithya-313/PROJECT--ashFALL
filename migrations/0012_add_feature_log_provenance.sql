-- File: migrations/0012_add_feature_log_provenance.sql
-- Purpose: Level 15c - Add provenance and error columns, enforce constraints on feature_log.

ALTER TABLE feature_log
  ADD COLUMN IF NOT EXISTS run_id TEXT NOT NULL,
  ADD COLUMN IF NOT EXISTS account_id TEXT NOT NULL,
  ADD COLUMN IF NOT EXISTS arrival_cutoff TIMESTAMPTZ NULL,
  ADD COLUMN IF NOT EXISTS feature_set_version TEXT NOT NULL,
  ADD COLUMN IF NOT EXISTS feature_names_sha256 TEXT NOT NULL,
  ADD COLUMN IF NOT EXISTS error_type TEXT NULL,
  ADD COLUMN IF NOT EXISTS error_message TEXT NULL;

ALTER TABLE feature_log
  ALTER COLUMN missing_request_fields SET DEFAULT '{}',
  ALTER COLUMN missing_request_fields SET NOT NULL,
  ALTER COLUMN registry_answers SET DEFAULT '[]'::jsonb,
  ALTER COLUMN registry_answers SET NOT NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'chk_feature_log_arrival_cutoff'
  ) THEN
    ALTER TABLE feature_log
      ADD CONSTRAINT chk_feature_log_arrival_cutoff
      CHECK (arrival_cutoff IS NOT NULL OR status = 'no_watermark');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'chk_feature_log_error_fields'
  ) THEN
    ALTER TABLE feature_log
      ADD CONSTRAINT chk_feature_log_error_fields
      CHECK (
        (status = 'error' AND error_type IS NOT NULL AND error_message IS NOT NULL)
        OR
        (status != 'error' AND error_type IS NULL AND error_message IS NULL)
      );
  END IF;
END;
$$;
