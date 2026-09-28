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
8c5752ce47acc067fc5517e69ca4062f
```

<!-- MD5 hash of data/enriched/paysim_enriched.parquet as tracked by DVC (from paysim_enriched.parquet.dvc).
     Re-running ingest_paysim.py + inject_adversarial.py with seed=42 must reproduce this hash exactly. -->

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
96c1c93626785172c34d50e561d7c5f5
```

<!-- MD5 hash of data/adversarial/adversarial_config.yaml as tracked by DVC (from adversarial_config.yaml.dvc).
     This config controls the random seed (42) and noise probability (0.50) used to inject adversarial
     fraud patterns. A change to this hash means the injection parameters changed and the enriched parquet
     must be regenerated to maintain reproducibility. -->
