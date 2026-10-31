-- File: migrations/0010_create_feature_log.sql
-- Purpose: Level 15c - create bank_feature_log table partitioned by decision_time range,
--          with daily partition helper, indexes, permissions, and row-level security.

-- ---------------------------------------------------------------------------
-- 1. Table: bank_feature_log
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bank_feature_log (
  id BIGINT GENERATED ALWAYS AS IDENTITY,
  bank_id VARCHAR(32) NOT NULL,
  payment_id VARCHAR(64) NOT NULL,
  decision_id VARCHAR(64) NOT NULL,
  decision_time TIMESTAMPTZ NOT NULL,
  arrival_watermark TIMESTAMPTZ NULL,
  status VARCHAR(32) NOT NULL CHECK (status IN ('ok', 'incomplete_request', 'no_watermark', 'error')),
  feature_values JSONB NOT NULL,
  feature_values_ordered DOUBLE PRECISION[] NULL,
  computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT pk_bank_feature_log PRIMARY KEY (id, decision_time),
  CONSTRAINT uq_bank_feature_log UNIQUE (bank_id, decision_id, decision_time)
) PARTITION BY RANGE (decision_time);

-- ---------------------------------------------------------------------------
-- 2. Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_bank_feature_log_lookup
  ON bank_feature_log (bank_id, payment_id, decision_time DESC);

CREATE INDEX IF NOT EXISTS idx_bank_feature_log_timeline
  ON bank_feature_log (bank_id, decision_time DESC);

-- ---------------------------------------------------------------------------
-- 3. Daily Partition Helper Function
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION create_feature_log_partition(for_day DATE)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
  partition_name TEXT;
  start_date TEXT;
  end_date TEXT;
  stmt TEXT;
BEGIN
  partition_name := 'feature_log_' || to_char(for_day, 'YYYYMMDD');
  start_date := to_char(for_day, 'YYYY-MM-DD');
  end_date := to_char(for_day + INTERVAL '1 day', 'YYYY-MM-DD');
  stmt := format(
    'CREATE TABLE IF NOT EXISTS %I PARTITION OF bank_feature_log FOR VALUES FROM (%L) TO (%L);',
    partition_name,
    start_date,
    end_date
  );
  EXECUTE stmt;

  -- Grant table permissions to ashfall_bank_api on the child partition
  BEGIN
    EXECUTE format('GRANT SELECT, INSERT ON TABLE %I TO ashfall_bank_api;', partition_name);
  EXCEPTION WHEN undefined_object THEN
    NULL;
  END;

  RETURN partition_name;
END;
$$;

-- Pre-create partitions for the current day + 7 days forward
DO $$
DECLARE
  d DATE;
BEGIN
  FOR d IN SELECT generate_series(CURRENT_DATE, CURRENT_DATE + INTERVAL '7 days', INTERVAL '1 day')::date LOOP
    PERFORM create_feature_log_partition(d);
  END LOOP;
END;
$$;

-- ---------------------------------------------------------------------------
-- 4. Role Privileges for ashfall_bank_api
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.bank_feature_log TO ashfall_bank_api;', current_schema());
  BEGIN
    EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE %I.bank_feature_log_id_seq TO ashfall_bank_api;', current_schema());
  EXCEPTION WHEN undefined_object THEN
    NULL;
  END;
END;
$$;

-- ---------------------------------------------------------------------------
-- 5. Row Level Security Policies
-- ---------------------------------------------------------------------------
ALTER TABLE bank_feature_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_feature_log_isolation_policy ON bank_feature_log;
CREATE POLICY bank_feature_log_isolation_policy ON bank_feature_log
  FOR ALL TO ashfall_bank_api
  USING (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  )
  WITH CHECK (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  );
