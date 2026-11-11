# Level 16 Handoff Note

This document summarizes the architecture, operational switches, performance measurements, known limitations, and open issues for Level 16 (Simulation Replay, Label Ingestion, and Point-in-Time Dataset Construction).

---

## 1. Executive Summary & Where Things Stand

Level 16 implements offline simulation replay, comprehensive four-source fraud label ingestion, and point-in-time training/validation dataset construction across sub-levels 16a, 16b, 16c, and 16d:
- **Part 16a (PRs #103–#108)**: Replay ingestion clock and `labels` table; run-scoped partition watermarks; explicit arrival timestamps on payments, account events, and withdrawals.
- **Part 16b (PRs #109–#114)**: Isolated replay workspaces with PostgreSQL schema isolation, Redis DB isolation, and single-partition Kafka topics; world-time step ordering; determinism hash (`feature_log` SHA-256) and history recompute.
- **Part 16c (PRs #115–#120)**: Per-event decision scoring (`decisions_per_event_payment=True`) with counterparty receiver VPA inclusion; fraud labels and daily account snapshots in replay; idle-partition payment watermark advance; dataset builder producing PyArrow tables and Parquet files with cryptographic manifests; evaluation truth join.
- **Part 16d (PRs #121–#124 and current)**: Phone visibility (KYC phone on account opening events); pre-world registry seed and in-world reports through an in-process DPIP registry; in-process nationalsim screener with seeded screen delay; police bulk labels and cross-bank hold request labels at receiving banks; committed Level 16 showcase world (seed 65) with 12 checks; dataset rebuild across splits with superseded marking.

Current status:
- **Showcase**: 12/12 checks passing (`python -m showcase --level 16`).
- **Issue Register**: 104 issues checked (built 72, partial 13, planned 17, deferred 2).

---

## 2. Dynamic Level 16 Switch Table

The table below is generated directly from `config/replay.yaml` via `python scripts/list_level_16_switches.py`:

| Switch Name | Configuration File | Default Live | Replay Value | What OFF Reproduces |
| :--- | :--- | :--- | :--- | :--- |
| `decisions_per_event_payment` | `config/replay.yaml` | `false` | `true` | Level 16b per-batch payment attempt scoring (decisions evaluated per attempt) |
| `idle_partition_advance` | `config/replay.yaml` | `false` | `true` | Level 16c-2 quiet payment partition watermark stalling |
| `kyc_phone_on_account_opened` | `config/replay.yaml` | `false` | `true` | Level 16c feeds without KYC phone payload on account_opened events |
| `world_registry_reporting` | `config/replay.yaml` | `false` | `true` | Level 16c replays without in-process DPIP registry labels |
| `replay_speed` | `config/replay.yaml` | `1.0` | `"max"` | Real-time paced replay execution (with inter-step sleep delays) |

---

## 3. Key Architectural Decisions & Operational Mechanics

### Truth Storage & Evaluation Isolation
Hidden simulation truth (true syndicate membership, mule roles, cash-out transactions) is persisted strictly in per-run JSONL files under `truth/<run_id>/` (via `world/truth_writer.py`), never inside operational PostgreSQL database tables or Redis feature stores. Operational replay workspaces and the live decision API have zero access to truth files. Point-in-time dataset construction operates strictly on operational extracts. Hidden truth is joined exclusively during offline model evaluation via `evaluation/dataset_truth_join.py::join_truth()`.

### Single-Partition Topic Limitation
Replay topics use a single partition (`kafka_topic_partitions: 1` in `config/replay.yaml`) to guarantee total chronological ordering of events during world-clock stepping. Multi-partition Kafka topics are deferred until Level 19, when full distributed partition alignment is addressed.

### Cap-Cut Payments Receive No Decisions
Payments that exceed bank UPI per-payment caps or liquidity thresholds are split or cut by simulator sizing logic. Any payment leg pruned or cancelled before reaching the bank connector hook generates no decision call, no feature vector, and no row in `feature_log`. Datasets reflect only executed transaction decisions.

### Redis Logical-DB Lease Limit
`world/replay_workspace.py` (lines ~257–275) manages workspace isolation by allocating Redis logical databases in the range `1..15` (database `0` is strictly preserved for live production/system traffic). This bounds concurrency to at most 15 simultaneous replay workspaces before lease exhaustion occurs.

### Execution Runtime & Showcase Cost
Replay execution pacing measures ~1.1 seconds per decision on a standard developer laptop:
- Committed showcase world (`seed: 65`, 40 people, 8 shops, 90 days): ~45 minutes per replay.
- `make show LEVEL=16` executes two full replays (Replay A and Replay B) to prove byte-for-byte feature log SHA-256 determinism across isolated workspaces, taking approximately 2.5 hours total runtime.
- Multi-world split datasets (`preset: tiny`, 60 people, 12 shops, 61 days): ~40–50 minutes per world.

### History of Check 16-7 (Four-Path Label Ingestion)
Check 16-7 validates the ingestion of fraud complaint labels across all four operational channels: direct same-bank complaints, cross-bank hold requests, in-process DPIP registry reports, and police bulk crackdowns.
- *Initial implementation*: Direct complaints and hold requests were lumped together under `source == 'complaint'`.
- *Round 16d-3b hardening (Commit 1 / 6de7988)*: Check 16-7 strictly separates direct complaints (`source="complaint"`, `via is None`) from hold-request labels (`source="complaint"`, `via="hold_request"`). Both expected sets are computed and compared independently with explicit count assertions, proving neither path can substitute for the other.

---

## 4. Issue Register Alignment & Future Work

Confirming line references in `config/issues_v2.yaml`:
- **Issue 16-1** (line ~1035): Live-world decisions per attempt and held-payment attempt minute — `status: deferred` to Level 19.
- **Issue 16-2** (line ~1043): Registry wired into replay — `status: built` (complete).
- **Issue 16-3** (line ~1055): Decision API request has no device field — `status: planned` for Level 19. No duplicate entry exists.
- **Issue 16-4** (line ~1062): Full-scale dataset building and faster replay drain — `status: planned` for Level 18a.
- **Issue 16-5** (line ~1069): Idle-partition watermark advance for the live payment consumer — `status: planned` for Level 19.
- **Issue 16-6** (line ~1075): Phone changes during a run (SIM swap, linked-number switch) not modelled — `status: planned` for Level 18a.
- **Issue 16-7** (line ~1082): World generates revoke/dispute for wrong listings — `status: planned` for Level 21.
- **Issue 16-8** (line ~1090): Registry coverage by mule depth (N2) — `status: planned` for Level 18a.
- **Issue 16-9** (line ~1097): Pre-world registry seed share calibration — `status: planned` for Level 28.

All issue statuses in `config/issues_v2.yaml` reflect their true verified state.
