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
| Side Event Times | `times` | ZSET | Member: `payment_id:side`, Score: Unix seconds of `event_time` |
| Side Payloads | `records` | HASH | Field: `payment_id:side`, Value: Canonical serialized JSON record |
| Lifecycle Times | `events:times` | ZSET | Member: `event_id`, Score: Unix seconds of `event_time` |
| Lifecycle Payloads | `events:records` | HASH | Field: `event_id`, Value: Canonical serialized JSON record |
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
