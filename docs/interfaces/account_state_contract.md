# Account State Contract

**Level 15 Architecture & Interface Specification**  
**Document Version:** 1.0 (Level 15a)  
**Status:** Canonical Interface Definition

---

## 1. Schema-2 UPI Outbox Payloads

Commercial bank transactions are captured atomically into PostgreSQL `outbox` records conforming to schema version `"2"`.
Defined in `api/v1/upi_outbox.py` on `main`:

### 1.1 Data Payload (`UpiEventData`, `api/v1/upi_outbox.py` lines 78–93)
```python
class UpiEventData(BaseModel):
    """The business payload inside a schema-2 UPI outbox event."""

    bank_id: str
    account_id: str
    counterparty_vpa: str | None = None
    counterparty_bank_id: str
    sender_bank_id: str
    receiver_bank_id: str
    amount_paise: int
    status: str
    failure_domain: str | None = None
    payee_type: str | None = None
    channel: str | None = None
    sender_device_fingerprint: str | None = None
    sender_os_generation: int | None = None
```

### 1.2 Event Envelope (`UpiEventPayload`, `api/v1/upi_outbox.py` lines 96–108)
```python
class UpiEventPayload(BaseModel):
    """The outer envelope of a schema-2 UPI outbox event."""

    event_type: str = EVENT_TYPE_UPI_PAYMENT_RECORDED
    schema_version: str = UPI_SCHEMA_VERSION
    event_id: str
    transaction_id: str
    payment_id: str
    side: str
    event_time: str
    recorded_at: str
    source: str = "world"
    data: UpiEventData
```

### 1.3 Arrival Timestamp Contract (`recorded_at`)
As defined in `api/v1/upi_outbox.py` line 267 and line 277:
```python
created_at_iso = row["created_at"].isoformat()
...
recorded_at=created_at_iso,
```
The `recorded_at` field is populated directly from PostgreSQL's server transaction timestamp (`transactions.created_at`). It represents the physical persistence arrival time in ashFALL storage, strictly separating business occurrence time (`event_time`) from storage availability time (`arrival_time`).

bank_withdrawals uses recorded_at (Postgres DEFAULT now()) as its arrival time, the same clock as bank_account_events.created_at and transactions.created_at.

---

## 2. Bank Account Events Table (`bank_account_events`)

Lifecycle events (device changes, SIM swaps, account openings) reside in relational storage.
Defined in `migrations/0007_bank_api_requests_and_records.sql` lines 70–88 on `main`:

```sql
CREATE TABLE IF NOT EXISTS bank_account_events (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  bank_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN (
    'device_binding_change',
    'sim_swap',
    'app_reinstall',
    'os_update',
    'account_opened',
    'account_reactivated'
  )),
  event_time TIMESTAMPTZ NOT NULL,
  details JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_bank_account_events_bank_event UNIQUE (bank_id, event_id)
);
```

### 2.1 Allowed Details Keys
Allowed details fields per event kind:
- `device_binding_change`: `platform`, `fingerprint_before`, `fingerprint_after`
- `sim_swap`: `phone_number_hash`
- `app_reinstall`: `platform`
- `os_update`: `platform`, `os_generation`
- `account_opened`: empty or initial metadata
- `account_reactivated`: empty or reactivation channel

---

## 3. Partitioning and Ordering Guarantees

### 3.1 Kafka Partitioning
Kafka ordering per account is not guaranteed and not relied on; no partition-key switch exists; proven by the shuffle test.

Because all state mutations on Redis are strictly idempotent (set union, hash overwrite, LT/GT watermark progression, and deterministic Lua cutoff trimming), outbox messages may be delivered in arbitrary partition order, interleaved, or replayed without state corruption.

### 3.2 Relational Poller Overlap (Architect Decision 5)
The Postgres poller adapter `apply_account_events_since` queries:
```sql
SELECT id, bank_id, event_id, account_id, kind, event_time, details, created_at
FROM bank_account_events
WHERE bank_id = $1 AND id > max(0, after_id - account_event_overlap_rows)
ORDER BY id
LIMIT $3
```
- `account_event_overlap_rows = 200`: The poller always re-reads the trailing 200 rows of committed events.
- **Race Condition Addressed:** In PostgreSQL, transactions obtain sequence `id` values upon insertion, but may commit out of sequence (transaction with `id=1` commits after transaction with `id=2`). Re-reading the 200-row overlap guarantees that late-committing lower-ID transactions are never permanently skipped.
- **Idempotency Protection:** Re-reading already-applied events in the overlap window is a complete no-op on Redis.

---

## 4. Point-in-Time Snapshot Contract & Future-Leak Rule

The snapshot reader `read_account_snapshot` provides causally consistent views for fraud detection scoring.

### 4.1 Section 22 Future-Leak Invariant
A payment side or account event is visible in the snapshot if and only if:
1. `min_allowed_time <= record.event_time <= decision_time` where `min_allowed_time = decision_time - retention_seconds`.
2. `record.arrival_time <= arrival_cutoff` where `arrival_cutoff` is the watermark cutoff of the scoring evaluator.

Any transaction that occurred prior to `decision_time` whose storage timestamp (`recorded_at` / `arrival_time`) is later than `arrival_cutoff` is strictly excluded. This guarantees simulation evaluations never leak future information.

### 4.2 Watermark Contract
The snapshot populates three deterministic watermarks over all visible records:
- `first_event_time`: Earliest `event_time` observed among visible records (`None` if no history).
- `last_event_time`: Latest `event_time` observed among visible records (`None` if no history).
- `arrival_watermark`: Highest `arrival_time` observed among visible records (`None` if no history).

---

## 5. Redis Data Structures, Key Formats, and TTL Lifecycle

Every account feature structure is namespaced under a tenant-isolated prefix:
`acctf:{run_id}:{bank_id}:{account_id}:<structure>`

| Structure | Key Suffix | Redis Type | Contents |
|---|---|---|---|
| Side Event Times | `side_times` | ZSET | Member: `payment_id:side`, Score: Unix seconds of `event_time` |
| Side Payloads | `side_records` | HASH | Field: `payment_id:side`, Value: Canonical serialized JSON record |
| Lifecycle Times | `event_times` | ZSET | Member: `event_id`, Score: Unix seconds of `event_time` |
| Lifecycle Payloads | `event_records` | HASH | Field: `event_id`, Value: Canonical serialized JSON record |
| Withdrawal Times | `withdrawal_times` | ZSET | Member: `withdrawal_id`, Score: Unix seconds of `event_time` |
| Withdrawal Payloads | `withdrawal_records` | HASH | Field: `withdrawal_id`, Value: Canonical serialized JSON record |
| Watermarks & Marks | `marks` | ZSET | Members: `first_event` (LT), `last_event` (GT), `arrival_watermark` (GT) |

### 5.1 TTL Expiration Rule
Every key write refreshes the Redis expiration TTL to:
$$\text{TTL} = \text{retention\_seconds} + \text{allowed\_lateness\_seconds} + \text{key\_expiry\_margin\_seconds}$$
With default policy settings:
$$\text{TTL} = 604800\text{ s (7d)} + 86400\text{ s (1d)} + 86400\text{ s (1d)} = 777600\text{ s (9d)}$$
This guarantees that active sliding windows never expire mid-computation, while inactive/quiet accounts are reclaimed automatically.

### 5.2 Atomic Lua Trim (Architect Decision 3)
Sliding-window pruning is performed via a registered atomic Lua script:
```lua
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if #expired > 0 then
    redis.call('HDEL', KEYS[2], unpack(expired))
    redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
end
return #expired
```
- **Cutoff Boundary:** `cutoff = marks.last_event - (retention_seconds + allowed_lateness_seconds)`.
- **Atomicity:** Expired members are queried from the ZSET, purged from the corresponding HASH, and removed from the ZSET in a single atomic transaction on the Redis engine. No orphan hash entries can ever be created.

---

## 6. Replay and Shuffle Guarantees

1. **Replay Idempotency:** Submitting the exact same transaction or lifecycle event $N$ times leaves key counts, sorted set cardinality, and serialized payloads completely unchanged.
2. **Order Independence (Shuffle Invariance):** Applying any set of valid records across an account in chronological order, reverse order, or pseudo-randomly shuffled order yields identical byte dumps in Redis.

---

## 7. Long-Lived Summaries and Cash-Out State (Level 15b)

While sliding windows retain 7 days of raw records per account, certain M1 fraud detection features (routine life, device novelty, dormancy, account age, and cash-out behavior) require memory spanning months or the account's entire lifetime. Long-lived summaries maintain compact, point-in-time consistent aggregates alongside the raw window under strict architectural constraints.

### 7.1 Closed-Day Rule (Point-in-Time Correctness)
Day numbers are defined as the IST calendar date integer (`event_time.astimezone(IST).date().toordinal()`).
To prevent future data leakage while ensuring deterministic replay:
1. **WRITE:** An incoming event updates daily summaries only if its physical arrival timestamp satisfies:
   $$\text{arrival\_time} \le \text{day\_end\_ist} + \text{allowed\_lateness\_seconds}$$
   Any event arriving later than $\text{day\_end\_ist} + \text{allowed\_lateness}$ is tagged with outcome `"too_late_for_summary"`, logged, and excluded from summary aggregates. It is still stored in the 7-day raw window.
2. **READ:** Summary readers return daily aggregates only for closed days where:
   $$\text{day\_end\_ist} + \text{allowed\_lateness\_seconds} \le \text{decision\_time} \quad \text{and} \quad \le \text{arrival\_cutoff}$$
   Recent, unclosed days are evaluated exclusively from the 7-day raw window snapshot. This rule guarantees zero lookahead leakage in model evaluation and feature computation.

### 7.2 Cash-Out State and Failed Attempts (G2, R7)
Cash withdrawals from ATM or branch channels are stored in `bank_withdrawals` and mirrored in Redis:
- Successful withdrawals increment `withdrawal_count` and add to `withdrawal_paise`.
- Failed withdrawals (or failed payment sides) increment `failed_attempt_count` only; their amounts are strictly excluded from `withdrawal_paise`, `credit_paise`, or `debit_paise` (Issue R7).

### 7.3 Long-Lived Key Layout & TTL
Summary keys use an extended retention TTL ($62\text{ days horizon} + 1\text{ day lateness} + 1\text{ day margin} = 64\text{ days} = 5,529,600\text{ seconds}$), while lifetime keys receive no expiry (they live as long as the run's namespace):

| Structure | Key Suffix | Redis Type | Contents | TTL Policy |
|---|---|---|---|---|
| Days Index | `days_index` | ZSET | Member: `day`, Score: `day` (IST date ordinal) | Summary TTL (64d) |
| Day Totals | `day_totals_{day}` | HASH | Aggregated counts and paise amounts for that IST day | Summary TTL (64d) |
| Day Seen Set | `day_seen_{day}` | HASH | Set of processed `record_member` keys to guarantee update idempotency | Summary TTL (64d) |
| Counterparty Days | `cp_days_{direction}_{token}` | ZSET | Member: `day`, Score: `day` for distinct counterparty active days | Summary TTL (64d) |
| Counterparty Index | `cp_index_{direction}` | ZSET | Member: `counterparty`, Score: latest `day` (ZADD GT) | Summary TTL (64d) |
| Active Days | `active_days` | ZSET | Member: `day`, Score: `day` for closed money activity days | No TTL (Lifetime) |
| Account Opening | `opened_at` | ZSET | Member: `opened_at`, Score: Unix seconds of `event_time` (ZADD LT) | No TTL (Lifetime) |
| Device Fingerprints | `fingerprints` | ZSET | Member: `fingerprint`, Score: $\max(\text{event\_time}, \text{arrival\_time})$ (ZADD LT) | No TTL (Lifetime) |

### 7.4 Generated Key List Verbatim
Generated via the key builders in `features/account_keys.py`:
```text
Raw Window & Marks:
  side_times:           acctf:66db9aa439b092ee:bank_0:acc_101:side_times
  side_records:         acctf:66db9aa439b092ee:bank_0:acc_101:side_records
  event_times:          acctf:66db9aa439b092ee:bank_0:acc_101:event_times
  event_records:        acctf:66db9aa439b092ee:bank_0:acc_101:event_records
  withdrawal_times:     acctf:66db9aa439b092ee:bank_0:acc_101:withdrawal_times
  withdrawal_records:   acctf:66db9aa439b092ee:bank_0:acc_101:withdrawal_records
  marks:                acctf:66db9aa439b092ee:bank_0:acc_101:marks

Long-lived Summaries:
  days_index:           acctf:66db9aa439b092ee:bank_0:acc_101:days_index
  day_totals:           acctf:66db9aa439b092ee:bank_0:acc_101:day_totals_20742
  day_seen:             acctf:66db9aa439b092ee:bank_0:acc_101:day_seen_20742
  counterparty_days:    acctf:66db9aa439b092ee:bank_0:acc_101:cp_days_inbound_91ba9265940ec8cc
  counterparty_index:   acctf:66db9aa439b092ee:bank_0:acc_101:cp_index_inbound

Lifetime Keys (No TTL):
  active_days:          acctf:66db9aa439b092ee:bank_0:acc_101:active_days
  opened_at:            acctf:66db9aa439b092ee:bank_0:acc_101:opened_at
  fingerprints:         acctf:66db9aa439b092ee:bank_0:acc_101:fingerprints
```

### 7.5 Lifetime Values (No TTL Expiry)
Lifetime keys live as long as the run's namespace; a run's keys are removed by namespace cleanup, never by a short TTL.
- `active_days`: Tracks distinct closed calendar days where money activity occurred (payment sides and cash withdrawals). Account lifecycle events (device changes, SIM swaps, account opening) are not money activity and never write to `active_days`.
- `last_active_day`: Derived dynamically on read as the largest day in `active_days` whose closed threshold ($\text{day\_end\_ist} + \text{allowed\_lateness}$) is $\le \min(\text{decision\_time}, \text{arrival\_cutoff})$; `None` if no closed active day exists.
- `history_days_available`: Measures whole calendar days from the START of the earliest closed active day to `decision_time`, capped at `summary_horizon_days` (62 days); returns 0 if no closed active day exists. It deliberately does not measure account opening age, ensuring an old account early in a run still evaluates as "too new to know" (warm-up).
- `opened_at`: Recorded in dedicated lifetime ZSET `opened_at` with member `"opened_at"` and score `event_time` (ZADD LT) from the `account_opened` event.
- `fingerprints`: Recorded in dedicated lifetime ZSET `fingerprints` with score $\max(\text{event\_time}, \text{arrival\_time})$ using `ZADD LT`. When querying, only fingerprints with score $\le \min(\text{decision\_time}, \text{arrival\_cutoff})$ are visible. If the count exceeds `fingerprint_warning_count` (10), a warning is logged, but no entries are dropped.

### 7.6 Warm-up Period
The system enforces `routine_life_warm_up_days: 62` (two monthly cycles plus weekend/holiday slack). Before an account or world reaches 62 days of elapsed history, routine-life features yield "not enough history" / empty defaults.

### 7.7 Opaque Counterparty VPA (Architect Decision 6B)
For each payment side, the simulator populates `counterparty_vpa` with the counterparty's raw account ID. This identifier is treated as an opaque string formatted as `counterparty_bank_id:counterparty_vpa`. Features may only perform exact equality matching against known accounts in the local bank and must never parse run, bank, or account components from it.

---

## 8. Registry Screening (DPIP / I4C Intelligence)

### 8.1 Replay Screening with `as_of`
When evaluating transaction decisions during historical simulation replay (Level 16), feature extractors query the national fraud registry (DPIP) using the optional testbed replay parameter `as_of`. The registry computes the screening verdict purely from historical timestamps without mutating in-memory entries:
- An entry is active at $T$ if $\text{visible\_at} \le T$, not revoked at or before $T$, and not resolved at or before $T$.
- An entry is contested at $T$ if $\text{contested\_at} \le T < \text{resolved\_at}$.
- `as_of` must be timezone-aware and $\le$ the registry clock; requests with a future `as_of` are rejected with HTTP 400 (`ValueError: as_of is later than the registry clock`) to prevent future information leakage.

### 8.2 Unknown vs No Match (Rule 3 Honest Degradation)
When national registry connectivity fails or a non-participating bank receives HTTP 403 Forbidden, the screening outcome is marked strictly as `'unknown'`. Downstream intelligence features (`registry_match_count`, `registry_match_age_days`, `payee_registry_match`) evaluate to `None` (`unknown`), never `False` or `0`. An outage or unconfigured key is never mistaken for a clean record.

### 8.3 Bank-Side Hashing Key
Each bank hashes accounts and device fingerprints using HMAC-SHA256 with the shared DPIP key (`dpip_participant_key`, env `DPIP_PARTICIPANT_KEY`) and `nationalsim.keyed_hash.compute_identifier_hash`. Account hashes namespace with the integer bank index loaded from `config/bank_feed.yaml` (`bank_ids_by_index`), while device fingerprints use the global raw fingerprint. A bank without the configured key cannot screen, causing registry features to degrade honestly to `unknown`.

### 8.4 Known Labelling Blind Spot
In a mule-to-mule hop the sending side is also a mule and the credit-side rule does not label it; this is a safe miss, never a wrong label.

---

## 9. Decision-Time Feature Logging & Ingestion Watermarks (Level 15c)

### 9.1 Decision Feature Log Contract (`bank_feature_log`)
When `config.decision_feature_logging_enabled` is True, commercial banks shadow-log point-in-time canonical M1 features during `/v1/decisions` execution. The feature log row is recorded in `bank_feature_log` in the same PostgreSQL transaction as the `bank_decision_requests` record. If the feature log insert fails, the transaction rolls back atomically and HTTP 500 is returned.

The table is range-partitioned daily on `decision_time` with tenant row-level security (`ashfall_bank_api` role).

### 9.2 Watermark Guard Against Section 22 Violations
To satisfy Section 22 evaluation correctness, features for a payment decision at $T = \text{decision\_time}$ are computed only if the bank's ingestion watermark $W \ge T$. The bank watermark aggregates 5 independent ingestion sources:
$$W = \min(W_{\text{payments}}, W_{\text{account\_events}}, W_{\text{withdrawals}}, W_{\text{feedback}}, W_{\text{complaints}})$$
where $W_{\text{payments}} = \min_{p \in \text{assigned\_partitions}}(W_{\text{payments}, p})$.

If $W < T$ or $W$ is missing, evaluation degrades safely to `status = 'no_watermark'` with all canonical features set to `None`. The system never guesses or falls back to wall-clock time.

### 9.3 Live Registry Screener (`LiveRegistryScreener`)
In live production scoring, screen requests must evaluate strictly against current entries. The `LiveRegistryScreener` adapter wraps any `RegistryScreener` and strips any caller-supplied `as_of` parameter (always forwarding `as_of=None`), guaranteeing simulation replay timestamps never leak into live traffic.

### 9.4 Open for Level 16: Decision Clock vs Event Time Clock Gap
In Level 15c, live operational decisions record `decision_time` from the system clock (`SystemClock` / host wall-clock). In Level 16 historical replay, the decision clock must align with simulated world time rather than host time, resolving the clock gap between asynchronous event production and decision evaluation. A partition that has never received a message is not in the set, so it does not hold the watermark back; Level 16 replay must confirm that every partition it produces to has received at least one message before decisions start.
