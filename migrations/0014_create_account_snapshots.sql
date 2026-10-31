-- File: migrations/0014_create_account_snapshots.sql
-- Purpose: Level 15c Round 2b - day-partitioned account snapshots table with RLS and helper function.

-- ---------------------------------------------------------------------------
-- 1. Table: account_snapshots
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS account_snapshots (
  run_id TEXT NOT NULL,
  bank_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  snapshot_time TIMESTAMPTZ NOT NULL,
  snapshot_kind TEXT NOT NULL CHECK (snapshot_kind IN ('daily', 'full')),
  arrival_cutoff TIMESTAMPTZ NOT NULL,
  feature_set_version TEXT NOT NULL,
  feature_names_sha256 TEXT NOT NULL,
  feature_values JSONB NOT NULL,
  logged_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (bank_id, run_id, account_id, snapshot_time, snapshot_kind)
) PARTITION BY RANGE (snapshot_time);

-- ---------------------------------------------------------------------------
-- 2. Default Partition
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS account_snapshots_default
  PARTITION OF account_snapshots DEFAULT;

-- ---------------------------------------------------------------------------
-- 3. Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_account_snapshots_lookup
  ON account_snapshots (bank_id, run_id, account_id, snapshot_time DESC);

-- ---------------------------------------------------------------------------
-- 4. Daily Partition Helper Function
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION create_account_snapshots_partition(for_day DATE)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
  partition_name TEXT;
  start_date TEXT;
  end_date TEXT;
  stmt TEXT;
BEGIN
  partition_name := 'account_snapshots_' || to_char(for_day, 'YYYY_MM_DD');
  start_date := to_char(for_day, 'YYYY-MM-DD');
  end_date := to_char(for_day + INTERVAL '1 day', 'YYYY-MM-DD');
  stmt := format(
    'CREATE TABLE IF NOT EXISTS %I PARTITION OF account_snapshots FOR VALUES FROM (%L) TO (%L);',
    partition_name,
    start_date,
    end_date
  );
  EXECUTE stmt;

  BEGIN
    EXECUTE format('GRANT SELECT, INSERT ON TABLE %I TO ashfall_bank_api;', partition_name);
  EXCEPTION WHEN undefined_object THEN
    NULL;
  END;

  RETURN partition_name;
END;
$$;

-- Pre-create partitions for the migration day and 7 days forward
DO $$
DECLARE
  d DATE;
BEGIN
  FOR d IN SELECT generate_series(CURRENT_DATE, CURRENT_DATE + INTERVAL '7 days', INTERVAL '1 day')::date LOOP
    PERFORM create_account_snapshots_partition(d);
  END LOOP;
END;
$$;

-- ---------------------------------------------------------------------------
-- 5. Role Privileges for ashfall_bank_api
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.account_snapshots TO ashfall_bank_api;', current_schema());
END;
$$;

-- ---------------------------------------------------------------------------
-- 6. Row Level Security Policies
-- ---------------------------------------------------------------------------
ALTER TABLE account_snapshots ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_account_snapshots_isolation ON account_snapshots;
CREATE POLICY bank_account_snapshots_isolation ON account_snapshots
  FOR ALL TO ashfall_bank_api
  USING (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  )
  WITH CHECK (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  );

