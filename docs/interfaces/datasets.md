# Point-in-Time Dataset Construction (`replay/dataset_builder.py`)

## Architectural Overview

The dataset builder constructs immutable point-in-time training, validation, test, and stress datasets from historical simulation replay workspaces. It extracts relational simulation telemetry, point-in-time feature vectors, daily account snapshots, and complaint fraud labels into structured PyArrow tables and persists them as Parquet files accompanied by a cryptographic metadata manifest.

Point-in-time datasets enforce absolute causal hygiene: all features, labels, and account snapshots are evaluated against a strict world-clock cutoff timestamp ($C$), preventing lookahead bias and future state leakage.

---

## Core Data Structures

### 1. `WorldExtract`
An in-memory frozen snapshot extracted from an isolated replay workspace prior to workspace teardown (`extract_world`):
- `world_id`: Provenance simulation world identifier.
- `split`: Dataset partition (`training`, `validation`, `test`, `stress`).
- `seed`: World builder random seed.
- `run_id`: 16-character fingerprint of the replay run.
- `clock_start`: Simulation world start datetime (used as the chronological baseline).
- `gang_families`: Tuple of active syndicate tactic families present in the world.
- `decisions_per_event_payment`: Boolean flag confirming decisions were evaluated for all event payments.
- `registry_wired`: Boolean flag indicating whether the pre-decision registry was active.
- `registry_wired_reason`: Text explanation for registry wiring state.
- `payment_rows`: Decision feature rows extracted from `feature_log` (real-clock column `created_at` omitted; `receiver_account_id` required).
- `snapshot_rows`: Daily account snapshots extracted from `account_snapshots` (real-clock column `logged_at` omitted).
- `label_rows`: Fraud complaint labels extracted from `labels` (real-clock column `recorded_at` omitted).

### 2. `Dataset`
A point-in-time bundle consisting of:
- `payments`: PyArrow table containing event payment features, decision metadata, and convenience label columns.
- `snapshots`: PyArrow table containing point-in-time customer account snapshot features evaluated at IST midnight boundaries.
- `labels`: PyArrow table containing complaint fraud labels known by the cutoff.
- `manifest`: Structured dictionary detailing dataset provenance, row counts, feature definitions, and cryptographic checksums.

---

## Design Rules (R1–R8)

### R1. Multi-World Concatenation
Extracts across multiple simulation worlds within the same split are concatenated into unified tables. Every row in `payments`, `snapshots`, and `labels` preserves its provenance via an explicit `world_id` column.

### R2. Strict Cutoff Filtering
A single simulation world-clock timestamp ($C$) governs inclusion across all three tables:
- **Payments**: `decision_time <= C`
- **Snapshots**: `snapshot_time <= C`
- **Labels**: `arrival_time <= C`
Events occurring strictly after $C$ are excluded.

### R3. Convenience Label Columns & Exact Counterparty Matching
To facilitate offline model training without manual multi-table joins, the `payments` table includes two point-in-time boolean convenience columns:
1. `label_receiver_reported_by_cutoff`: Evaluates to `True` if any label known by $C$ (`arrival_time <= C`) targets the payment's receiving account (`receiver_account_id`), regardless of final verdict.
2. `label_payment_named_by_cutoff`: Evaluates to `True` if the payment identifier appears in the `payment_ids` array of any label known by $C$.
*Note*: False complaints (`verdict == 'not_fraud'`) count as reported, as a receiving account subject to an investigated complaint is operationally flagged prior to final resolution.

#### Strict Receiver Account Validation (Decision D1)
In fraud decision contexts, `account_id` on the payment row represents the payer/sender account. The payee/receiver account must be explicitly populated via `receiver_account_id` (extracted from the underlying transaction). `build_dataset` requires `receiver_account_id` on every payment row; if missing, it raises a plain `ValueError` without silent fallback to `account_id`.

### R4. Feature Columns Specification
The manifest explicitly enumerates all 52 dense feature columns (`feature_columns`) loaded from `canonical_feature_names()`. No column beginning with the `label_` prefix is ever included in the feature column specification, preventing target leakage during automated vector assembly.

### R5. Preservation of Unwatermarked Rows & Missingness Accounting
Payments logged with status `no_watermark` (e.g. before initial bank watermarks are established) are preserved as first-class training rows. The manifest records:
- `no_watermark_count`: Integer count of `no_watermark` rows in the dataset.
- `feature_none_shares`: Exact proportion of null (`None`) values per feature column (e.g., `payee_registry_match: 1.0` while registry wiring is deferred).
- `label_maturity_note`: Explicit notation that fraud labels arriving near cutoff $C$ contain unadjusted reporting latency.

### R6. Canonical Content Hash (`content_hash`)
A deterministic SHA-256 fingerprint generated across all three tables:
- Rows are sorted canonically:
  - Payments: `(world_id, decision_time, payment_id)`
  - Snapshots: `(world_id, snapshot_time, bank_id, account_id, snapshot_kind)`
  - Labels: `(world_id, arrival_time, label_id, account_id)`
- Column keys are serialized in sorted alphabetical order.
- Real wall-clock timestamps (`created_at`, `logged_at`, `recorded_at`) are excluded.
- Identical simulation replays produce bitwise identical content hashes regardless of host execution speed or row extraction order.

### R7. Fail-Closed Refusals
`build_dataset` refuses invalid dataset requests by raising `ValueError`:
1. `decisions_per_event_payment == False`: Replay must evaluate decisions for all event payments.
2. **Held-Out Tactics in Training**: Any training split dataset containing active held-out tactic families (`slow_drip`, `device_rotation` from `config/evaluation.yaml`) is immediately refused.
3. **Split Mismatch**: Every constituent extract must match the target split (`training`, `validation`, `test`, `stress`).
4. **Empty Sequence**: Building from an empty extract list is prohibited.

### R8. Storage Format, Byte Identity & Manifest Integrity (`write_dataset`)
Datasets are serialized to disk under `<root>/<split>-<cutoff yyyymmddTHHMM>-<content_hash[:12]>`:
```
data/datasets/training-20260915T0000-a1b2c3d4e5f6/
├── payments.parquet
├── snapshots.parquet
├── labels.parquet
└── manifest.json
```
`write_dataset` computes SHA-256 digests for each Parquet file and embeds them into `manifest.json["files"]`. Each constituent world's `clock_start` is stored in ISO-8601 format in the `manifest["worlds"]` list (one entry per world, with `world_id`, `run_id`, `clock_start`).

#### Parquet Byte-Identity Note
While `content_hash` guarantees strict bitwise equivalence of the extracted dataset rows across identical replays, individual `.parquet` files serialized via PyArrow may exhibit minor byte differences across Python sessions due to metadata header timestamps or dictionary page order. Downstream consumers should verify dataset parity via the manifest's `content_hash` rather than raw Parquet file hashes.

## Joining truth (evaluation only)

`evaluation/dataset_truth_join.py::join_truth(dataset_dir, truth_root)` returns the `payments` and `snapshots` tables of a built dataset with hidden-truth columns added. It reads the dataset folder and never writes into it; original columns are unchanged.

**What it adds**
- `payments`: `truth_role_receiver` (role of `receiver_account_id`; `genuine` when no span applies), `truth_owner` (gang id or null), `truth_depth` (mule depth or null).
- `snapshots`: `truth_role`, `truth_owner`, `truth_depth` for `account_id`.

**Minute rule.** Each row is looked up at its own world minute: `ist_to_world_minute(row time, clock_start)`, i.e. whole minutes (floored) since IST midnight of the world's `clock_start` date. The row time is `decision_time` (payments) or `snapshot_time` (snapshots). `run_id` and `clock_start` come from the matching entry in `manifest["worlds"]` (matched by the row's `world_id`). Spans are half-open: `valid_from` counts, `valid_to` does not. A missing `run_id`/`clock_start`, an unknown `world_id`, or a row time before `clock_start` raises `ValueError`.

**Where it may run.** Evaluation, showcase and test code only. `replay/`, `features/` and the other operational packages must not import `evaluation.dataset_truth_join` or `evaluation.truth`; `load_truth` also raises `PermissionError` when called directly from them. The joined tables are for scoring and must never be fed back into training or serving features.
