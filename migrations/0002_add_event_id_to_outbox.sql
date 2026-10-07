-- File: migrations/0002_add_event_id_to_outbox.sql
-- Purpose: Level 7 - give every outbox event its own permanent identity (event_id) and make duplicates impossible.
--
-- The story in plain English:
--   Every event we publish needs an ID of its own, separate from the transaction's ID. Consumers use it to spot an
--   event they have already handled. When a client retries a request with the same idempotency key, the service
--   computes the SAME event_id again, and the UNIQUE rule below makes the database refuse the second copy. That is
--   how a replay is detected without any extra locking in the application.
--
-- Why four steps instead of one: the outbox table may already contain rows. A column that is NOT NULL cannot be added
-- while old rows have no value for it, so we add it as nullable, fill the old rows, and only then tighten the rules.
-- No rows are deleted.

-- Step 1: add the column. Nullable for now, so existing rows are still valid.
ALTER TABLE outbox ADD COLUMN IF NOT EXISTS event_id UUID;

-- Step 2: backfill. Old rows get a fresh random ID (they never had a client key to derive one from).
UPDATE outbox SET event_id = gen_random_uuid() WHERE event_id IS NULL;

-- Step 3: now that every row has a value, forbid empty values from here on.
ALTER TABLE outbox ALTER COLUMN event_id SET NOT NULL;

-- Step 4: forbid two events with the same ID. This is what turns a retry into "already done".
-- (ADD CONSTRAINT has no IF NOT EXISTS in PostgreSQL; the migration runner applies this file only once.)
ALTER TABLE outbox ADD CONSTRAINT uq_outbox_event_id UNIQUE (event_id);
