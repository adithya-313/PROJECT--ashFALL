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

`redis` is now a pinned runtime dependency of the API (`redis==8.1.0` in `requirements.txt`); it was previously installed only through `fakeredis`, so the API image did not have it.

---

## 10. Historical Feature Replay & Point-in-Time Recomputation (Level 15c Round 2a)

### 10.1 Scratch Redis Isolation
Historical feature recomputation reconstructs sliding-window account state inside a dedicated, isolated scratch Redis instance (such as `fakeredis.FakeRedis()` or an isolated test Redis database). Scratch Redis operations never mutate, trim, or read live operational keys, namespaces, or watermarks.

### 10.2 Merged Chronological Ingestion Sequence
Historical replay reads recorded relational rows from all 5 feeds up to the logged decision `arrival_cutoff`:
1. `transactions` (arrival: `created_at`)
2. `bank_account_events` (arrival: `created_at`)
3. `bank_withdrawals` (arrival: `recorded_at`)
4. `bank_feedback` (arrival: `created_at`)
5. `bank_complaints` (arrival: `created_at`)

Rows are sorted in merged chronological arrival sequence `(arrival_time, table_order, id_str)` and applied via domain updaters (`apply_payment_side`, `apply_account_event`, `apply_withdrawal`, `apply_feedback_row`, `apply_complaint_row`) without writing watermarks to Redis. Any late-arriving row (`arrival_time > arrival_cutoff`) is strictly excluded.

Identifier parsing uses `extract_run_id` to strictly validate run identifiers: any row with an unparseable identifier raises a plain `ValueError` naming the database table and the offending row ID. Rows from valid other runs are skipped via explicit equality comparison against `run_id`.

### 10.3 Logged Registry Screener (`LoggedAnswerScreener`)
Point-in-time replay eliminates external network dependency and guarantees deterministic screening by using `LoggedAnswerScreener`. The screener answers queries exclusively using the `registry_answers` audit payload captured in `feature_log` during the original decision. Any query not present in the logged answers raises a `LookupError`, ensuring no silent fallback or mock drift.

### 10.3a Live Registry Prefetch and Timeout
In a live decision with the real registry client, `compute_decision_row` lists every registry question the decision will ask (the payer account, each visible lifetime device fingerprint of the payer, and the payee account, in that order and without duplicates, built with the same hash function and visibility rules as the features) and asks them all at once with a per-call timeout (`decision_registry.timeout_ms` in `config/account_features.yaml`). The decision waits for the slowest call, not the sum. The feature code is unchanged and reads the answers through `PrefetchedScreener`, which raises `LookupError` for a question that was not prefetched.

A call that exceeds the timeout is answered `'unknown'` (never `'no_match'`), so the dependent features are `None`. Each such timeout increments the Prometheus counter `ashfall_registry_timeouts_total{identifier_type}`. Every entry of `registry_answers` carries `outcome.timed_out` (true only for that case). Replay reads the logged `'unknown'` through `LoggedAnswerScreener`, so a logged timeout recomputes exactly. Registry answers are not cached; that belongs to Level 19.

In the API the questions are asked on the server's own event loop through ONE shared connection pool (an `httpx.AsyncClient` created at startup when a DPIP participant key is set, with the TLS context built once, explicit `max_connections` / `max_keepalive_connections` from `config/nationalsim.yaml`, and closed with `aclose()` at shutdown). `record_decision` lists the questions in a worker thread, awaits the answers on the event loop, then computes the features in a worker thread from them; there is no per-decision event loop. `NationalsimBankClient` without a shared client behaves as before.

`decision_registry.timeout_ms` is derived, not chosen: `max_ms - feature_computation_max_ms - other_overhead_p99 - timeout_margin_ms - clock_resolution_ms` (a result below `timeout_minimum_ms` is an error). A call is "unknown" only if it really took longer than the timeout, measured with `perf_counter`; because asyncio can fire timers up to two clock ticks early on coarse clocks (about 31 ms on Windows), the wait is allowed `timeout + two ticks` and the measured time decides. The registry wait counts against the 100 ms whole-decision limit only; the 50 ms feature limit covers our own feature code (`latency_scope` in `config/evaluation.yaml`).

### 10.4 Zero-Tolerance Parity Verification
`recompute_logged_row` compares the logged feature vector against the recomputed feature vector. Feature parity requires exact value equality (`==`) and identical runtime types (`type(l) is type(r)`). Any numerical drift, missing field, or subtle type mismatch is captured in `RecomputeResult.mismatches`.

---

## 11. Account Feature Snapshots (Level 15c Round 2b)

### 11.1 Purpose and Dual-Cadence Snapshot Strategy
Machine learning models for live mule detection require dense, point-in-time account feature vectors for both active transacting accounts and quiet or dormant customer accounts. Taking periodic snapshots avoids online recomputation latency during training set generation:
1. **Daily Active Snapshots (`snapshot_kind = 'daily'`)**:
   Captures all accounts belonging to the bank that had any event arrive within the simulated day `[day_start, day_end)` across any of the 5 operational feeds:
   - `transactions` (debit and credit payment legs posted by this bank)
   - `bank_account_events` (lifecycle events)
   - `bank_withdrawals` (cash withdrawals)
   - `bank_feedback` (`confirmed_fraud` feedback linking to the credit-side mule account)
   - `bank_complaints` (complaints naming a reported account at this bank)
2. **Weekly Full Snapshots (`snapshot_kind = 'full'`)**:
   Evaluated when `day_number % full_interval_days == 0` (configured as 7 days in `config/account_features.yaml`). Selects all accounts opened by or before the arrival cutoff, ensuring quiet, dormant, and non-transacting customer accounts appear in training datasets.

### 11.2 Storage Schema & Ingestion Watermark Cutoff (`account_snapshots`)
Snapshot feature vectors are stored in PostgreSQL table `account_snapshots` (migration 0014), range-partitioned daily by `snapshot_time`, with `account_snapshots_default` ensuring uninterrupted persistence, and protected by Row Level Security for role `ashfall_bank_api`:
- **Primary Key**: `(bank_id, run_id, account_id, snapshot_time, snapshot_kind)`
- **Arrival Cutoff Contract**: Every snapshot row enforces `arrival_cutoff = get_bank_watermark(redis, bank_id)`. If no watermark is available for the bank, snapshot writing raises `NoWatermarkError` and aborts; it never fabricates an arrival cutoff or falls back to `snapshot_time`. Any event arriving after `arrival_cutoff` is strictly excluded from computed feature state.
- **Feature Schema**: Features are serialized as a JSONB array matching `account_feature_names()` order (the canonical 52 M1 features minus the 8 `payment_level` features = 44 account features) with `feature_names_sha256` digest verification.
- **Settings Contract**: `write_account_snapshots` takes `settings: AccountFeaturesSettings | None = None`; if `None`, repository defaults are loaded. Any invalid type is rejected statically by type checking rather than falling back at runtime.
- **Intelligence Features in Level 15c**: In Level 15c, snapshots compute account features with `screener=None`. Consequently, intelligence features (`registry_match_count`, `registry_match_age_days`) evaluate to `None` (`null` in JSONB). This is intentional and preserved for full offline screener integration in Level 16.


## 12. Geo-velocity in M1

M1 uses no geo-velocity. The bank feed carries a region for each account, not a location, so there is nothing to measure a travel speed from. The geo-velocity kernel (`features/geo_velocity.py` and its GPU path `features/gpu_geo_velocity.py`) stays a Level 4.5 artifact, kept for future location data. No M1, account-state or decision module imports it.

Checked with `git grep -n "geo_velocity\|gpu_geo_velocity\|haversine" -- features/` at Level 15 closing; the whole output is:

```
features/__init__.py:9:#   geo_velocity.py - HOW FAST would the user have had to travel between two transactions? (km/h)
features/__init__.py:16:#   from features.geo_velocity import haversine_distance_km
features/geo_velocity.py:1:# File: features/geo_velocity.py
features/geo_velocity.py:63:def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
features/geo_velocity.py:97:    # Step 3: apply the haversine formula. "a" is the squared half-chord length between the two points on the
features/geo_velocity.py:118:def compute_geo_velocity(current_geo: str, last_geo: str | None, elapsed_seconds: float) -> float | None:
features/geo_velocity.py:163:    distance_km = haversine_distance_km(last_lat, last_lon, current_lat, current_lon)
features/geo_velocity.py:169:def haversine_distance_km_array(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
features/geo_velocity.py:171:    Haversine distance in kilometres for many point pairs at once (the same five steps as haversine_distance_km).
features/geo_velocity.py:173:    Why a second copy of the formula: haversine_distance_km works on one pair and returns a Python float. This one
features/geo_velocity.py:176:    the same numbers. If you change one, change the other (tests/unit/test_geo_velocity.py compares them).
features/geo_velocity.py:183:    # The haversine term "a", kept inside [0, 1] so the square roots below never see a tiny negative number.
features/geo_velocity.py:197:def compute_geo_velocity_cpu_vectorized(
features/geo_velocity.py:209:    # It follows the same order as compute_geo_velocity (current location, then last location), so a bad value
features/geo_velocity.py:228:    # exactly as in compute_geo_velocity), then speed = distance / hours, all rows at once.
features/geo_velocity.py:229:    distances_km = haversine_distance_km_array(
features/geo_velocity.py:235:def compute_geo_velocity_batch(
features/geo_velocity.py:242:    compute_geo_velocity does for a single transaction.
features/geo_velocity.py:262:        from features.gpu_geo_velocity import compute_geo_velocity_gpu
features/geo_velocity.py:267:        gpu_values = compute_geo_velocity_gpu(
features/geo_velocity.py:279:        speeds = compute_geo_velocity_cpu_vectorized(current_geos, last_geos, elapsed_seconds_list, valid_indices)
features/gpu_geo_velocity.py:1:# File: features/gpu_geo_velocity.py
features/gpu_geo_velocity.py:3:#          This is the GPU path; the CPU reference path lives in features/geo_velocity.py.
features/gpu_geo_velocity.py:15:# Mean radius of the Earth in kilometres; identical to EARTH_RADIUS_KM in features/geo_velocity.py so the CPU and
features/gpu_geo_velocity.py:28:def haversine_triton_kernel(
features/gpu_geo_velocity.py:64:    # Step 6: apply the haversine formula exactly as the CPU version does. "a" is the squared half-chord length
features/gpu_geo_velocity.py:85:    # batch dispatcher in features/geo_velocity.py filters such rows out before they reach the GPU.)
features/gpu_geo_velocity.py:92:def compute_geo_velocity_gpu(
features/gpu_geo_velocity.py:115:    haversine_triton_kernel[grid](
features/gpu_geo_velocity.py:124:# It is imported conditionally by features/geo_velocity.py based on batch size.
```

The only files that match are `features/__init__.py` (a package docstring), `features/geo_velocity.py` and `features/gpu_geo_velocity.py` themselves.


## 13. One Ingestion Clock for Complaint and Feedback Arrival

### 13.1 Principle
Event time is when an event occurred in the world (`received_at`, `event_time`).
Arrival time is when ashFALL learned it (`arrival_time`).
Under Ground Rule 30 and Architect Decisions A, B, and D, point-in-time domain visibility, label cutoffs, and watermarks compare arrival time with decision time on ONE ingestion clock. The database's server clock (`now()`, `created_at`) remains an audit stamp for physical row insertion, but never determines domain visibility, cutoffs, watermarks, or merge ordering when an ingestion clock is provided.

### 13.2 Real-Clock Rule (`clock: Clock | None = None`)
In accordance with Ground Rule 13 (backward compatibility), functions accepting a clock parameter default to `clock = None`. When `clock is None`, the behaviour is byte-for-byte identical to legacy real-clock operations:
- Writers stamp `arrival_time = NULL` in relational tables (`bank_complaints`, `bank_feedback`).
- Pollers derive watermarks by querying `SELECT now()` directly from PostgreSQL.

### 13.3 Legacy NULL Fallback
When reading historical rows inserted prior to migration 0017, or rows written in real-clock mode without an injected clock, `arrival_time` is `NULL`. The helper `arrival_of(row)` safely falls back to `row["created_at"]` only when `row.get("arrival_time")` is `None`. This is the single, isolated location in the codebase where this fallback lives.

### 13.4 Migration Roadmap
- **Completed in Level 16a Round 2**:
  - `bank_complaints` and `bank_feedback` schema migration 0017 (`arrival_time` column).
  - In-process client and HTTP router clock injection (`InProcessBankClient`, `post_feedback`).
  - Label pollers and visibility handlers (`apply_complaints_since`, `apply_feedback_since`, `apply_complaint_row`, `apply_feedback_row`).
- **Completed in Level 16a Round 2b**:
  - Run-scoped watermark keys and forward-only refused move observability.
  - Decision feature extraction fail-closed switch (`run_scoped_watermarks` in `compute_decision_row`).
- **Completed in Level 16a Round 3a**:
  - `transactions`, `bank_account_events`, and `bank_withdrawals` schema migration 0018 (`arrival_time` column).
  - Ingestion clock stamping on all writers (`record_transaction`, `record_upi_side`, `record_account_event`, `record_withdrawal`).
  - Schema-2 outbox payload optional `arrival_time` key with strict omission rule when `None`.
  - Four Guards (GUARD 1 clock validation, GUARD 2 fail-loud on corrupt arrival/run-prefix, GUARD 3 golden byte equality, GUARD 4 arrival before event warning).
  - Shared fallback helper `arrival_from_row(row, fallback_column)` across records and labels.
  - Payment consumer watermark routing to run-scoped key `acctwm:{run_id}:payments:{partition}` for world records.
- **Completed in Level 16a Round 3b**:
  - Events and withdrawals pollers writing run-scoped watermarks (`acctwm:{run_id}:{bank_id}:events`, `acctwm:{run_id}:{bank_id}:withdrawals`) using `Clock` without querying database `now()`.
  - Account snapshots selection windowing by `COALESCE(arrival_time, ...)` and `write_account_snapshots` supporting run-scoped watermark cutoffs (`run_scoped_watermarks: bool = False`).
  - Removal of the `no_watermark` intermediate state with `test_world_mode_decision_uses_real_run_watermarks` verifying all five sources through real writers.
- **Follow-up in Level 16b**:
  - `feature_replay.py` moves to the ingestion clock in 16b.

### 13.5 Run-Scoped Ingestion Watermarks

#### Key Layout Table
| Source / Entity | Real-Clock Key (`run_id=None`) | World-Clock Key (`run_id` given) |
|---|---|---|
| Bank Poller | `acctwm:{bank_id}:{source}` | `acctwm:{run_id}:{bank_id}:{source}` |
| Payment Partition | `acctwm:payments:{partition}` | `acctwm:{run_id}:payments:{partition}` |
| Payment Partitions Set | `acctwm:payments:partitions` | `acctwm:{run_id}:payments:partitions` |

All keys are constructed strictly via `features/account_keys.py`:
- `watermark_key_for_poller(bank_id, source, run_id=None)`
- `watermark_key_for_payment_partition(partition, run_id=None)`
- `watermark_key_for_payment_partitions_set(run_id=None)`

#### Forward-Only Refusal & Observability
Watermark updates execute via `SET_WATERMARK_LUA`. If a caller attempts to move a watermark backwards:
1. The update is refused and returns `0` (stored value remains untouched). Equal values return `1` (treated as stored-unchanged).
2. A warning is logged at `logging.WARNING` containing the Redis key, currently stored timestamp, and refused candidate timestamp.
3. Prometheus Counter `ashfall_watermark_rejected_moves_total` is incremented with label `source` (`"payments"` for Kafka partitions; `"events"`, `"withdrawals"`, `"feedback"`, or `"complaints"` for database pollers).

#### Decision D1: Fail-Closed Decision Features
`compute_decision_row` accepts `run_scoped_watermarks: bool = False`:
- When `False` (legacy default): queries `get_bank_watermark(redis_client, bank_id)`.
- When `True` (world clock): queries `get_bank_watermark(redis_client, bank_id, run_id=run_id)`.
- If aggregate watermark is `None`, the function fails closed immediately, returning `status="no_watermark"`, all canonical features `None`, and performing zero account-state reads.

### 13.6 World Arrival for Payments, Events, and Withdrawals (Level 16a-3a)

#### Schema-2 Envelope Field and Omission Rule
The schema-2 outbox payload (`UpiEventPayload`) includes an optional top-level `arrival_time: str | None = None`:
- When written with an injected clock, `arrival_time` contains the timezone-aware ISO string formatted timestamp.
- When written in legacy / real-clock mode (`clock=None`), `arrival_time` is `None` and the key is omitted entirely from the serialized JSON payload via selective field exclusion (`exclude={"arrival_time"}`). Global `exclude_none` is strictly forbidden because domain fields (such as `data.failure_domain`) are legitimately `null`.
- `schema_version` remains `"2"`.

#### The Four Guards
1. **GUARD 1 (Strict Clock Validation)**: When an `ingestion_clock` is provided to any writer (`record_transaction`, `record_upi_side`, `record_account_event`, `record_withdrawal`), it MUST return a valid timezone-aware `datetime`. If the clock returns `None` or a naive datetime, a plain `ValueError` is raised immediately before executing SQL, ensuring no database row is written.
2. **GUARD 2 (Fail-Loud Consumer Validation)**: In `parse_upi_payload`, if `arrival_time` is present but unparseable or naive, or if `extract_run_id(payment_id)` fails, the parser raises `InvalidAccountEventError`. The live Kafka updater logs the topic, partition, offset, and error, and re-raises immediately. The parser NEVER falls back to `recorded_at` or legacy keys when `arrival_time` is present.
3. **GUARD 3 (Golden Byte Compatibility)**: A `clock=None` transaction produces a payload byte-for-byte identical to the pre-16a-3 baseline, containing no `arrival_time` key.
4. **GUARD 4 (Chronological Ordering Warning)**: If `arrival_time` is earlier than the transaction's `event_time`, a warning is logged at `logging.WARNING` recording both timestamps and the `payment_id`. The event is not rejected.

#### Shared Fallback Helper (`arrival_from_row`)
All domain record converters and label visibility handlers resolve arrival timestamps through a single shared helper:
`arrival_from_row(row: Mapping[str, Any], fallback_column: str) -> datetime`:
- Returns `row["arrival_time"]` when non-NULL.
- Falls back to `row[fallback_column]` when `arrival_time` is NULL.
- Enforces timezone awareness, raising `ValueError` naming the row identifier if naive or invalid.
- Used in `account_event_from_row` (fallback `"created_at"`), `withdrawal_from_row` (fallback `"recorded_at"` then `"created_at"`), `payment_side_from_transaction_row` (stamping `arrival_time` into the envelope when non-NULL), and `features/account_labels.py::arrival_of` (delegating to `arrival_from_row(row, "created_at")`).
