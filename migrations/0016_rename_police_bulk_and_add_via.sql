-- File: migrations/0016_rename_police_bulk_and_add_via.sql
-- Purpose: Level 16a Round 2 - rename bulk_confirmation source to police_bulk and add via column to labels.
--
-- Why the rename (Architect Decision D):
-- Level 34 batch recognition categorizes regulatory and law enforcement confirmations specifically
-- as 'police_bulk' (e.g. state police / cyber cell hold request batches), distinguishing them from
-- generic or future bulk feeds. Existing records with source 'bulk_confirmation' are safely updated.
--
-- Migration integrity:
-- Migration 0015 is an immutable historical migration file and is NEVER edited. All schema evolution,
-- constraint alterations, and data backfills are performed strictly forward through migration 0016.

-- ---------------------------------------------------------------------------
-- 1. Drop old source check constraint on labels dynamically (without guessing names)
-- ---------------------------------------------------------------------------
DO $$
DECLARE
  r RECORD;
BEGIN
  FOR r IN (
    SELECT c.conname
    FROM pg_constraint c
    JOIN pg_class t ON c.conrelid = t.oid
    JOIN pg_namespace n ON t.relnamespace = n.oid
    WHERE n.nspname = current_schema()
      AND t.relname = 'labels'
      AND c.contype = 'c'
      AND pg_get_constraintdef(c.oid) LIKE '%source%'
  ) LOOP
    EXECUTE format('ALTER TABLE %I.labels DROP CONSTRAINT %I;', current_schema(), r.conname);
  END LOOP;
END;
$$;

-- ---------------------------------------------------------------------------
-- 2. Update existing rows from bulk_confirmation to police_bulk
-- ---------------------------------------------------------------------------
UPDATE labels
SET source = 'police_bulk'
WHERE source = 'bulk_confirmation';

-- ---------------------------------------------------------------------------
-- 3. Add new check constraint on source
-- ---------------------------------------------------------------------------
ALTER TABLE labels
  ADD CONSTRAINT chk_labels_source
  CHECK (source IN ('complaint', 'registry', 'analyst', 'police_bulk'));

-- ---------------------------------------------------------------------------
-- 4. Add via column and non-blank check constraint
-- ---------------------------------------------------------------------------
ALTER TABLE labels
  ADD COLUMN IF NOT EXISTS via TEXT NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint c
    JOIN pg_class t ON c.conrelid = t.oid
    JOIN pg_namespace n ON t.relnamespace = n.oid
    WHERE n.nspname = current_schema()
      AND t.relname = 'labels'
      AND c.conname = 'chk_labels_via'
  ) THEN
    ALTER TABLE labels
      ADD CONSTRAINT chk_labels_via
      CHECK (via IS NULL OR length(btrim(via)) > 0);
  END IF;
END;
$$;

