-- File: migrations/0011_reshape_feature_log.sql
-- Purpose: Level 15c - reshape feature log table to planned contract (feature_log).

DROP TABLE IF EXISTS bank_feature_log CASCADE;

CREATE TABLE IF NOT EXISTS feature_log (
  bank_id VARCHAR(64) NOT NULL,
  payment_id VARCHAR(128) NOT NULL,
  decision_time TIMESTAMPTZ NOT NULL,
  decided_at TIMESTAMPTZ NOT NULL,
  watermark_lag_ms BIGINT,
  status VARCHAR(32) NOT NULL CHECK (status IN ('ok', 'incomplete_request', 'no_watermark', 'error')),
  feature_values JSONB NOT NULL,
  missing_request_fields TEXT[],
  registry_answers JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CONSTRAINT pk_feature_log PRIMARY KEY (bank_id, payment_id, decision_time)
) PARTITION BY RANGE (decision_time);

CREATE INDEX IF NOT EXISTS idx_feature_log_lookup
  ON feature_log (bank_id, payment_id, decision_time DESC);

CREATE INDEX IF NOT EXISTS idx_feature_log_timeline
  ON feature_log (bank_id, decision_time DESC);

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
    'CREATE TABLE IF NOT EXISTS %I PARTITION OF feature_log FOR VALUES FROM (%L) TO (%L);',
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

DO $$
DECLARE
  d DATE;
BEGIN
  FOR d IN SELECT generate_series(CURRENT_DATE, CURRENT_DATE + INTERVAL '7 days', INTERVAL '1 day')::date LOOP
    PERFORM create_feature_log_partition(d);
  END LOOP;
END;
$$;

DO $$
BEGIN
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.feature_log TO ashfall_bank_api;', current_schema());
END;
$$;

ALTER TABLE feature_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS feature_log_isolation_policy ON feature_log;
CREATE POLICY feature_log_isolation_policy ON feature_log
  FOR ALL TO ashfall_bank_api
  USING (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  )
  WITH CHECK (
    bank_id = current_setting('app.current_bank_id', true)
    OR bank_id = current_setting('app.bank_id', true)
  );
