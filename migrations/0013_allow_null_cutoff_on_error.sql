-- File: migrations/0013_allow_null_cutoff_on_error.sql
-- Purpose: Level 15c - Allow arrival_cutoff to be NULL when status is 'error'.

ALTER TABLE feature_log
  DROP CONSTRAINT IF EXISTS chk_feature_log_arrival_cutoff;

ALTER TABLE feature_log
  ADD CONSTRAINT chk_feature_log_arrival_cutoff
  CHECK (arrival_cutoff IS NOT NULL OR status IN ('no_watermark', 'error'));

