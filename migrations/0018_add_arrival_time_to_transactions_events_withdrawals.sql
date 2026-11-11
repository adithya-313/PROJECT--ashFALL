-- File: migrations/0018_add_arrival_time_to_transactions_events_withdrawals.sql
-- Purpose: Level 16a Round 3a - add arrival_time column to transactions, bank_account_events, bank_withdrawals.
--
-- Arrival time vs DB audit clock:
-- arrival_time represents the timestamp from the injected ingestion clock (world/simulation time)
-- at the moment the bank ingested the transaction, account event, or withdrawal.
-- created_at (or recorded_at for bank_withdrawals) remains the database server's real wall clock (DEFAULT now())
-- strictly for transport audit.
-- NULL in arrival_time indicates "written without an injected clock" (real-clock legacy mode),
-- allowing readers to gracefully fall back per Ground Rule 13.

ALTER TABLE transactions
  ADD COLUMN IF NOT EXISTS arrival_time TIMESTAMPTZ NULL;

ALTER TABLE bank_account_events
  ADD COLUMN IF NOT EXISTS arrival_time TIMESTAMPTZ NULL;

ALTER TABLE bank_withdrawals
  ADD COLUMN IF NOT EXISTS arrival_time TIMESTAMPTZ NULL;

