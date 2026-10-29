-- File: migrations/0008_add_sender_device_columns.sql
-- Purpose: Level 14c - add optional sender device fingerprint and OS generation to transactions.
--
-- The story in plain English:
--   * Level 14c integrates telemetry from client devices with commercial bank transaction feeds.
--   * When a commercial bank records the debit side of an originated payment, client hardware telemetry
--     (device fingerprint string and integer operating system generation rating) may be known.
--   * The core "transactions" table is extended with two nullable columns:
--     - "sender_device_fingerprint" (TEXT): synthetic client hardware token for the sending device.
--     - "sender_os_generation" (INTEGER): integer generation rating of the client operating system.
--   * Data integrity rules:
--     - sender_os_generation must be non-negative (>= 0) whenever present.
--     - Telemetry belongs strictly to the originating client device: both columns must remain NULL
--       on credit sides (receiving bank books) or unclassified transaction legs.
--   * Security and tenant isolation:
--     - The table-level grants and Row Level Security policy established in migration 0006 for
--       the "ashfall_bank_api" role automatically extend to these new columns without requiring
--       privilege or policy modifications.
--

-- ---------------------------------------------------------------------------
-- 1. Add nullable sender device columns in transactions
-- ---------------------------------------------------------------------------
ALTER TABLE transactions
  ADD COLUMN IF NOT EXISTS sender_device_fingerprint TEXT,
  ADD COLUMN IF NOT EXISTS sender_os_generation INTEGER;

-- ---------------------------------------------------------------------------
-- 2. Data integrity CHECK constraints
-- ---------------------------------------------------------------------------
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_sender_os_generation;
ALTER TABLE transactions ADD CONSTRAINT chk_sender_os_generation
  CHECK (sender_os_generation IS NULL OR sender_os_generation >= 0);

ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_device_columns_debit_only;
ALTER TABLE transactions ADD CONSTRAINT chk_device_columns_debit_only
  CHECK (side = 'debit' OR (sender_device_fingerprint IS NULL AND sender_os_generation IS NULL));

