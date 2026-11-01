-- File: migrations/0017_add_arrival_time_to_complaints_and_feedback.sql
-- Purpose: Level 16a Round 2 - add arrival_time column to bank_complaints and bank_feedback.
--
-- Arrival time vs DB audit clock:
-- arrival_time represents the timestamp from the injected ingestion clock (world/simulation time)
-- at the moment the bank ingested the complaint or feedback.
-- created_at remains the database server's real wall clock (DEFAULT now()) strictly for transport audit.
-- NULL in arrival_time indicates "written without an injected clock" (real-clock legacy mode),
-- allowing readers to gracefully fall back to created_at per Ground Rule 13.

ALTER TABLE bank_complaints
  ADD COLUMN IF NOT EXISTS arrival_time TIMESTAMPTZ NULL;

ALTER TABLE bank_feedback
  ADD COLUMN IF NOT EXISTS arrival_time TIMESTAMPTZ NULL;

