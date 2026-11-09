-- File: migrations/0019_add_phone_to_bank_account_events.sql
-- Purpose: Level 16d-0 - store the KYC phone a bank holds for its own account holder on account_opened events.
--
-- "phone" is the synthetic phone string from the account holder's KYC record. It is NULL when the feed does not
-- carry a phone (live default, switch kyc_phone_on_account_opened off) and for accounts that have no phone.
-- Banks match this value, hashed, against the national registry; it is never read from hidden truth.

ALTER TABLE bank_account_events
  ADD COLUMN IF NOT EXISTS phone TEXT NULL;
