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
