# Replay Workspaces

## Architectural Overview

A replay workspace provides a hermetically isolated execution boundary for historical simulations, counterfactual scenarios, and model evaluation runs without risking data corruption, cache pollution, or state leakage into live production databases.

Each workspace encapsulates three independent persistence and streaming layers:
1. **Dedicated PostgreSQL Schema (`postgres_schema`)**: A distinct database schema (e.g., `rpl_<run_id>_<n>`) provisioned and migrated with all production schema migrations (`scripts.run_migrations.apply_pending_migrations`). Tables and indices are completely scoped to this schema via `search_path`, isolating relational state from `public`.
2. **Dedicated Redis Database Index (`redis_db`)**: An isolated numerical database index in Redis (range 1–15, configured via `config/replay.yaml`), guaranteeing that transient simulation keys, watermarks, rate-limits, and caches never overwrite or flush live database entries (typically DB 0).
3. **Dedicated Kafka Topic (`kafka_topic`)**: An isolated event stream (e.g., `rpl_<run_id>_<n>.payments`) with 1 partition to guarantee total chronological ordering, consumed under an isolated consumer group (`kafka_group_id`), ensuring zero offset collisions or message pollution on production topics.

---

## Allocation & Distributed Registry Lock

Workspace allocation is coordinated dynamically by `allocate_workspace`:
- **Distributed Registry Lock**: A distributed lock in live Redis (`replay:workspaces:lock`) prevents race conditions during concurrent allocation. The lock uses an atomic token with an automated TTL to ensure safe cleanup if a process crashes.
- **Dynamic Allocation**: The registry scans the pool range (`redis_workspace_db_min=1` to `redis_workspace_db_max=15`) and selects the lowest available unassigned database index.
- **Live Database Exclusion**: The live Redis database index (parsed directly from `config.redis_url`, defaulting to 0) is strictly excluded from candidate allocation.
- **Concurrency Boundary**: The system supports up to 15 concurrent replay workspaces. If all database numbers are occupied, allocation raises a descriptive `RuntimeError` listing all currently active holder workspaces and their respective DB numbers.

---

## Safety Guards & Lifecycle Teardown

`drop_workspace` tears down resources with strict defensive validation:
1. **Refusal to Flush Live DB**: Any attempt to drop or `FLUSHDB` a workspace targeting the live Redis database index raises an explicit `ValueError`.
2. **Refusal to Drop Protected Schemas**: Any attempt to drop schemas matching `public`, `information_schema`, or `pg_catalog` raises an explicit `ValueError`.
3. **Isolated Teardown Sequence**:
   - The allocated workspace Redis database is flushed via `FLUSHDB` and closed.
   - The workspace schema is dropped with `DROP SCHEMA <schema> CASCADE`.
   - The dedicated Kafka topic is deleted via the admin client.
   - The workspace assignment is removed from the Redis registry hash (`replay:workspaces`) under the distributed lock.
4. **Production State Invariant**: The live PostgreSQL schema (`public`) and live Redis database are never modified or purged during allocation, replay, or teardown.

---

## Replay Wiring & Recording Architecture

### 1. Standalone Decision Request Builder (`MockBankConnectors.build_decision_request`)
- **Motivation & Contract**: In prior iterations, decision request payloads were constructed inline inside the closures returned by `MockBankConnectors.make_hook`. To support deterministic recording, offline simulation, and replay validation without triggering live or mock network calls, steps 1–3 were factored out into `build_decision_request(payment: PaymentView) -> DecisionCall`.
- **Zero Network Side Effects**: `build_decision_request` resolves sending and receiving banks from `account_bank_map`, determines payee classification from `feed_config.payee_type_by_account_prefix`, formats the deterministic idempotency key (`dec:<planned_payment_id>`), and serializes the exact JSON payload. It tracks `seen_accounts` for post-run ledger assertions without invoking any `BankTransport.post` coroutines.
- **Hook Delegation**: `MockBankConnectors.make_hook` delegates directly to `build_decision_request`, ensuring identical request schemas and idempotency tokens across simulated live execution and replay pipelines.

### 2. Run-Scoped Ingestion Watermarks (`run_scoped_watermarks`)
- **Isolation of Historical Horizons**: During historical replays and counterfactual evaluation, world simulation operates across synthetic timestamps that differ from real-time ingestion. Global poller and payment watermarks in Redis reflect live stream progress; using them would leak future or present data into past decision feature contexts.
- **Routing Switch**: The `run_scoped_watermarks: bool = False` flag in `prepare_registry_questions`, `record_decision`, and `InProcessBankClient`:
  - When `False` (default): queries `get_bank_watermark(redis_client, bank_id, run_id=None)` to preserve standard live behavior.
  - When `True`: extracts the 16-hex character `run_id` from `request.payment_id` via `extract_run_id` and queries `get_bank_watermark(redis_client, bank_id, run_id=run_id)`.
  - Propagates seamlessly through `compute_decision_row` to ensure registry prefetch questions evaluate against the specific replay run's watermark horizons.

### 3. Per-Client Feature Logging Override (`decision_feature_logging`)
- **Granular Control**: Feature logging in production is gated by the global configuration toggle `config.decision_feature_logging_enabled`. In multi-bank or multi-workspace replay scenarios, individual clients or test runs require feature logging without globally mutating shared application configuration.
- **Override Priority**: `record_decision` accepts `feature_logging_enabled: bool | None = None`. If an explicit boolean is provided, it takes precedence; otherwise it falls back to `config.decision_feature_logging_enabled`.
- **Client Configuration**: `InProcessBankClient` accepts `decision_feature_logging: bool | None = None` upon initialization and forwards it during `/v1/decisions` dispatch, allowing replay harnesses to record complete feature log rows with run-scoped cutoff dates while leaving production configuration untouched.

### 4. First-Pass Replay Recording (`record_world_for_replay`)
- **Interactive Simulation Interception**: `record_world_for_replay(fast_settings, runner_settings, connectors) -> RecordedWorld` runs `run_interactive_world` with an intercepting hook:
  ```python
  def recording_hook(payment: PaymentView) -> HookDecision:
      call = connectors.build_decision_request(payment)
      recorded.append((payment, call))
      return "allow"
  ```
- **Unaltered Baseline (`"allow"`)**: The recording pass always returns `"allow"`. This ensures the simulation produces an unblocked baseline trajectory identical to a reference run with `allow_all` (matching `run_id`, event counts, and payment sequences).
- **Refusal of Live Bank Modes**: If any bank mode in `connectors.modes` is `"live"`, `record_world_for_replay` raises `ValueError("Live bank mode refused until Level 19")`. Replay recording runs faster than real time or in batch modes; live banking transports are strictly prohibited until live integration is supported in Level 19.

---

## Replay Ordering Architecture (`replay/`)

### 1. Package Boundary & One-Way Import Rule
- **Decoupled Architecture**: The `replay/` package houses pure ordering logic, step representations (`FeedStep`, `DecisionStep`), and replay runner infrastructure.
- **Strict One-Way Dependency**:
  - Operational packages (`api/`, `features/`, `core/`, `world/`) must never import `replay`. This is enforced statically by AST inspection in `tests/unit/test_replay_import_direction.py`.
  - `replay/` belongs to `ashfall_packages` in `config/truth_store.yaml` and is strictly forbidden from importing `evaluation` modules or accessing hidden simulation truth.
  - Classified under `not_engine` in `config/folder_classification.yaml` as an operational testing and replay harness.

### 2. World-Time Replay Ordering Rules (`build_replay_steps`)
During replay, events must be delivered in strict causal sequence so decision evaluations observe the exact historical stream state that existed when the payment was initiated, without future event leakage:
1. **Preceding Feed Delivery**: Every feed record whose world time is at or before a decision's time is placed before that decision.
2. **Own-Sides Placement**: EXCEPT a payment's own sides (the event payment ids whose planned id equals the decision's planned id, via F5 alignment): they are placed immediately after their decision (decide first, then post).
3. **Same-Minute Decisions**: Decisions with the same world minute keep the hook's order; each is followed by its own sides before the next decision.
4. **Deterministic Tie-Breaking**: Remaining ties between feed records are broken deterministically by fixed entity kind order (`account_event` -> `transaction` -> `withdrawal` -> `complaint`), followed by `bank_id`, and finally `record_id`.

### 3. Resumed-Event Guard (`check_no_resumed_events`)
- **Strict Prohibition**: Replay assumes zero held payments because the pass-1 hook always allows; a resumed event would carry its resume minute, not its attempt minute, and leak hold-period information into its decision.
- **Fail-Fast Rejection**: If `events_resumed[i]` is True for any event in the recorded simulation output, `check_no_resumed_events` immediately raises a descriptive `ValueError` naming the payment identifier and minute of the violation.

### 4. Roadmap Note on Execution
Replay ordering defines the pure interleaving sequence of `FeedStep` and `DecisionStep` items. Dynamic runner execution, pipeline drain convergence, and workspace lifecycle management are implemented in `replay/runner.py`.

---

## Replay Runner Execution & Pipeline Convergence (`replay/runner.py`)

### 1. Runner Architecture (`run_replay`)
`run_replay(recorded_world, feed, admin_pg, ...)` drives historical simulations through the real operational stack in causal world-time sequence:
1. **Refusal of Live Modes**: Strictly prohibits live bank modes (`ValueError`), protecting production banking credentials and external endpoints.
2. **Hermetic Workspace Provisioning**: Invokes `allocate_workspace(admin_pg, ...)` to allocate an isolated PostgreSQL schema, dedicated Redis DB, and single-partition Kafka topic. Resources are guaranteed to be torn down via `drop_workspace` in a top-level `finally:` block.
3. **Derived Partition Pre-Creation**: Derives unique calendar days in both local world-time (IST) and UTC directly from the scheduled replay steps (`{s.world_time.date() for s in steps} | {s.world_time.astimezone(timezone.utc).date() for s in steps}`). Pre-creates daily table partitions for `feature_log` and `account_snapshots` under superuser pool connections, ensuring unprivileged tenant roles (`ashfall_bank_api`) without schema DDL grants execute inserts without check violation errors.
4. **World-Clock Step Dispatch**:
   - `FeedStep`: Advances manual world clock to `step.world_time` and dispatches projected record via `connectors.send_feed_record(step.kind, step.record)`. If transaction, tracks partition 0 as produced.
   - `DecisionStep`: Advances manual world clock to `step.world_time`, executes `drain()` to converge the operational pipeline, invokes test injection hook (`before_decision_hook`) if registered, verifies Section 9.4 invariants, and evaluates decision via `InProcessBankClient.post("/v1/decisions", ...)`.

### 2. Drain Rule & Convergence Criteria (`drain`)
Architect decision S2 mandates that all pending outbox messages, Kafka records, and database pollers are fully processed and watermarked before scoring each decision.
- **S2.3 Continuous Polling**: `consume_available(consumer, ...)` is polled on every iteration of the outbox relay loop, preventing deadlocks when the Kafka topic buffer fills while awaiting outbox flushes.
- **Convergence Guarantees**: A drain iteration succeeds only when all three subsystems converge:
  1. **Outbox**: `count_waiting(workspace_conn) == 0` (all transactional outbox rows published to Kafka).
  2. **Kafka Consumer**: For all assigned partitions, `consumer.committed(tp) >= consumer.end_offsets(tp)`.
  3. **Table Pollers**: For every commercial bank, poller cursors reach `MAX(id)` on `bank_account_events`, `bank_withdrawals`, `bank_feedback`, and `bank_complaints`. On caught-up tables, empty-table poller passes advance run-scoped Redis watermarks to the current world clock.

### 3. Timeout as Failure (`ReplayDrainTimeout`)
If pipeline convergence exceeds `deadline = time.monotonic() + drain_timeout_seconds`, `drain` immediately raises `ReplayDrainTimeout` reporting current partition offsets and poller cursors. The exception propagates to trigger clean workspace teardown, ensuring no leaked databases, lingering topics, or corrupted state persist.

### 4. Section 9.4 Architectural Rules (Ruling R1–R3)
- **R1 (Single-Partition Invariant)**: At replay initialization, `check_workspace_topic_partitions` queries broker metadata via `consumer.partitions_for_topic(topic)`. If partition count is not exactly 1, raises `ValueError`. With one partition per topic, partitions that never receive messages cannot hold aggregate watermarks back.
- **R2 (Produced Partition Watermark Check)**: Before evaluating every decision, after `drain()` converges, every partition the runner has produced at least one message to must possess a run-scoped payment watermark (`get_payment_watermark(ws_redis, partition=part, run_id=run_id)`). If missing, `ReplayPipelineError` is raised.
- **R3 (Fail-Closed Initial Decisions)**: Decisions preceding the production of any payment legitimately evaluate with status `"no_watermark"` (failing closed). These are verified and counted in `ReplayResult.decisions_before_first_payment`.

### 5. Redelivery Idempotency (S2.4)
Under at-least-once message delivery, producing duplicate payment messages to the Kafka topic results in bitwise-identical Redis feature state (verified by key count and SHA-256 digest over sorted keys and dumps).

### 6. Test World Measured Metrics
*(Measured on the author's laptop)*
- **Configuration**: Small test world `FastRunSettings(101, 20, 6, 51, 'training')` (from `tests/integration/test_bank_connectors_end_to_end.py:54`).
- **Decisions Sent**: 795 decisions evaluated in world-time sequence.
- **Pre-Payment Decisions (`no_watermark`)**: Exactly 1 decision (`e032d655af2c43f6-p-000000`).
- **Post-Payment Decisions (`incomplete_request`)**: Exactly 794 decisions.
- **Contract Gap Note**: All 794 watermarked decisions have `missing_request_fields == ["receiver_vpa"]`. *Known gap*: `build_decision_request` omits `receiver_vpa`, so payee features evaluate to `None`. Resolution is planned in Level 16c-1 (`decisions_per_event_payment` request contract). Payer-side features (`hour_ist`, `customer_failed_attempts_24h`, `outflow_to_merchant_paise_7d`) evaluate completely.
- **Decisions Arrival Cutoff Behind**: 0 (no decision evaluated with lagging cutoff).
- **Max Arrival Cutoff Gap**: 0.0 seconds.
- **Longest Drain Duration**: 0.039 seconds.
