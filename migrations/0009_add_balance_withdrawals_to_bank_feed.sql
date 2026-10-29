-- File: migrations/0009_add_balance_withdrawals_to_bank_feed.sql
-- Purpose: Level 15b - add balance after transaction to transactions and create bank_withdrawals table.
--
-- The story in plain English:
--   * Level 15b enhances the bank feed so commercial banks see the balance after each transaction
--     on their own accounts, as well as cash withdrawals (ATM and branch).
--   * The core "transactions" table is extended with:
--     - "balance_after_paise" (BIGINT NULL): the account's running balance in paise after the transaction.
--       Nullable so old rows stay valid. Must be non-negative (>= 0) when provided.
--   * A new bank-scoped table "bank_withdrawals" records cash withdrawals:
--     - id BIGSERIAL PRIMARY KEY
--     - bank_id VARCHAR(32) NOT NULL
--     - withdrawal_id VARCHAR(64) NOT NULL
--     - account_id VARCHAR(64) NOT NULL
--     - channel VARCHAR(16) NOT NULL CHECK (channel IN ('atm', 'branch'))
--     - amount_paise BIGINT NOT NULL CHECK (amount_paise > 0)
--     - status VARCHAR(16) NOT NULL CHECK (status IN ('success', 'failed'))
--     - failure_domain VARCHAR(32) NULL
--     - event_time TIMESTAMPTZ NOT NULL
--     - balance_after_paise BIGINT NULL CHECK (balance_after_paise IS NULL OR balance_after_paise >= 0)
--     - recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
--     - CONSTRAINT bank_withdrawals_idempotent UNIQUE (bank_id, withdrawal_id)
--     - index on (bank_id, account_id, event_time DESC)
--   * Row Level Security (RLS) is enabled for ashfall_bank_api matching app.bank_id, and SELECT, INSERT
--     privileges are granted to ashfall_bank_api.
--

-- ---------------------------------------------------------------------------
-- 1. Add balance_after_paise to transactions
-- ---------------------------------------------------------------------------
ALTER TABLE transactions
  ADD COLUMN IF NOT EXISTS balance_after_paise BIGINT NULL;

ALTER TABLE transactions DROP CONSTRAINT IF EXISTS chk_balance_after_paise;
ALTER TABLE transactions ADD CONSTRAINT chk_balance_after_paise
  CHECK (balance_after_paise IS NULL OR balance_after_paise >= 0);

-- ---------------------------------------------------------------------------
-- 2. Table: bank_withdrawals
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bank_withdrawals (
  id BIGSERIAL PRIMARY KEY,
  bank_id VARCHAR(32) NOT NULL,
  withdrawal_id VARCHAR(64) NOT NULL,
  account_id VARCHAR(64) NOT NULL,
  channel VARCHAR(16) NOT NULL CHECK (channel IN ('atm', 'branch')),
  amount_paise BIGINT NOT NULL CHECK (amount_paise > 0),
  status VARCHAR(16) NOT NULL CHECK (status IN ('success', 'failed')),
  failure_domain VARCHAR(32) NULL CHECK (failure_domain IS NULL OR failure_domain IN (
    'npci_central', 'sender_bank', 'receiver_bank', 'app_or_network', 'customer'
  )),
  event_time TIMESTAMPTZ NOT NULL,
  balance_after_paise BIGINT NULL CHECK (balance_after_paise IS NULL OR balance_after_paise >= 0),
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT chk_bank_withdrawals_failure_domain CHECK (
    (status = 'success' AND failure_domain IS NULL) OR
    (status = 'failed' AND failure_domain IS NOT NULL)
  ),
  CONSTRAINT bank_withdrawals_idempotent UNIQUE (bank_id, withdrawal_id)
);

CREATE INDEX IF NOT EXISTS idx_bank_withdrawals_lookup
  ON bank_withdrawals (bank_id, account_id, event_time DESC);

-- ---------------------------------------------------------------------------
-- 3. Role Privileges for ashfall_bank_api
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  EXECUTE format('GRANT SELECT, INSERT ON TABLE %I.bank_withdrawals TO ashfall_bank_api;', current_schema());
  EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE %I.bank_withdrawals_id_seq TO ashfall_bank_api;', current_schema());
END
$$;

-- ---------------------------------------------------------------------------
-- 4. Row Level Security Policies
-- ---------------------------------------------------------------------------
ALTER TABLE bank_withdrawals ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bank_withdrawals_isolation_policy ON bank_withdrawals;
CREATE POLICY bank_withdrawals_isolation_policy ON bank_withdrawals
  FOR ALL TO ashfall_bank_api
  USING (bank_id = current_setting('app.bank_id', true))
  WITH CHECK (bank_id = current_setting('app.bank_id', true));
