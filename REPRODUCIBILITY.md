# REPRODUCIBILITY

This file records every artifact needed to reproduce any result from this project —
training run, evaluation, benchmark, or deployment. The discipline starts now, before
anything is trained, so it becomes a habit rather than an afterthought. Fill in each
section every time you produce a result worth keeping. A result that cannot be
reproduced is a result that cannot be trusted.

---

## Git Commit SHA

```
dd400cad150fd5da7c1f1882e64d604ad9b21182
```

<!-- The exact git commit hash (`git rev-parse HEAD`) of the code that produced the result.
     Fill this in immediately after running an experiment or generating a report. -->

---

## DVC Data Hash

```
87c929358f0950a86aac62ce382795e8
```

<!-- MD5 hash of data/enriched/paysim_enriched.parquet as tracked by DVC (from paysim_enriched.parquet.dvc).
     Re-running ingest_paysim.py + inject_adversarial.py with seed=42 is intended to reproduce this hash exactly.
     Updated at the Level 3 correction (see below): the previous hash (265e328c7c7237f0c4c22ef61f648c9d) is the
     pre-correction dataset, kept recoverable in the DVC cache. The Level 3 acceptance history follows:
     Updated at Level 3 acceptance: the prior hash (8c5752ce47acc067fc5517e69ca4062f) was set from a
     parquet produced by an earlier pipeline version (before the final noise injection redesign and
     background rate changes). Two independent clean runs of the current code both produce this hash. -->

---

## MLflow Run ID

```
PLACEHOLDER
```

<!-- The MLflow run ID from the tracking server for the experiment that produced these results.
     Fill this in when MLflow logging is added in Level 3+. -->

---

## Docker Image Tag

```
PLACEHOLDER
```

<!-- The fully-qualified Docker image tag (e.g. ghcr.io/org/repo:sha-abc1234) used for
     inference or evaluation. Fill this in when image publishing is set up in Level 4+. -->

---

## Python Environment Hash

```
PLACEHOLDER
```

<!-- A hash or snapshot of the exact Python environment: run `pip freeze | sha256sum` and
     paste the hash, or attach the full `pip freeze` output as a file. Fill this in
     alongside every experiment so dependency drift is detectable. -->

---

## Adversarial Config Hash

```
92996dee8b0f19a561ff31b2f5de596e
```

<!-- MD5 hash of data/adversarial/adversarial_config.yaml as tracked by DVC (from adversarial_config.yaml.dvc).
     This config controls the random seed (42), the per-pattern noise probabilities (account_takeover 0.45,
     mule_account 0.30) and the injected fraud counts (11,800 each) used to inject adversarial fraud patterns.
     (Previous hash 96c1c93626785172c34d50e561d7c5f5 was the single noise_probability=0.50 config.)
     A change to this hash means the injection parameters changed and the enriched parquet
     must be regenerated to maintain reproducibility. -->

---

## Level 3 Acceptance

- **Enriched Parquet:** 6,362,620 rows; `is_adversarial` (bool, cardinality 2) and `adversarial_pattern_type` (object, 4 values: None/account_takeover/mule_account/merchant_collusion) present
- **AUC Gate:** account_takeover AUC=0.6546 PASS, mule_account AUC=0.5246 PASS, both ≤ 0.70 threshold
- **Pandera Schema:** passes on enriched parquet (21 columns, all dtype and constraint checks green)
- **CI Schema Gate:** `schema-validation` job exists in ci.yml; latest three CI runs on main are `success` (most recent: 2026-09-29)
- **DVC Hash:** `265e328c7c7237f0c4c22ef61f648c9d` re-tracked and committed (prior hash was from earlier pipeline version)
- **Reproducibility:** two independent clean runs of `ingest_paysim.py` + `inject_adversarial.py` with seed=42 both produce MD5=`265e328c7c7237f0c4c22ef61f648c9d`; pipeline is deterministic
- **E-commerce Store:** `scripts/ecommerce_store.py` generates transactions with `ecom_` prefix; schema validation (`data/schemas/ecommerce_transactions.py`) passes on valid batches; live POST /score blocked only by uvloop on Windows (works in Docker/Linux)
- **Merchant Collusion:** deferred to e-commerce store — 38 PaySim labelled rows produce ~11 test-set rows, insufficient for meaningful AUC; architecturally correct deferral

---

## Level 3 Correction

This section supersedes the Level 3 Acceptance figures above wherever they differ (data hash, AUC results,
pattern list). The earlier acceptance text is kept as history.

- **Bug fixed:** the adversarial patterns were not linked to `isFraud` (the old injection labelled rows but never
  set `isFraud`). `scripts/inject_adversarial.py` was rewritten to set `isFraud=1` on injected rows (controlled
  injection, modelled on the documented fraud research in `reports/data/fraud_pattern_research.md`).
- **Prevalence:** total fraud is now ~0.500% (31,813 rows = 8,213 original PaySim + 23,600 injected: 11,800
  `account_takeover` + 11,800 `mule_account`).
- **merchant_collusion** is no longer injected into PaySim (moved to the e-commerce stream).
- **Per-pattern injection noise:** `account_takeover` 0.45, `mule_account` 0.30.
- **AUC gate** retargeted to `isFraud` with a 0.55-0.70 pass band; both signals pass (`account_takeover`
  geo_velocity_spike AUC=0.6740, `mule_account` many_small_inbound_count AUC=0.6248).
- **Schema** now enforces `is_adversarial=True` implies `isFraud=1`.
- **Hashes:** enriched parquet `87c929358f0950a86aac62ce382795e8`; adversarial config
  `92996dee8b0f19a561ff31b2f5de596e`.
