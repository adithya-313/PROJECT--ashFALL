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
