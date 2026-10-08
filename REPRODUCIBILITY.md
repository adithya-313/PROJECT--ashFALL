# REPRODUCIBILITY

This file records every artifact needed to reproduce any result from this project —
training run, evaluation, benchmark, or deployment. The discipline starts now, before
anything is trained, so it becomes a habit rather than an afterthought. Fill in each
section every time you produce a result worth keeping. A result that cannot be
reproduced is a result that cannot be trusted.

---

## Git Commit SHA

```
6d7650b68ed5b22144e25e50f23845f3d63a002b
```

The commit this document was produced from (branch `feature/level5-serving-and-ops`, the 5.9 model card commit).
The commit that contains this file has a later SHA (a file cannot contain its own commit hash). The previous
value (`dd400cad...`) was the PR #7 merge and predated all Level 3 fix and Level 5 work.

---

## DVC Data Hash

```
87c929358f0950a86aac62ce382795e8
```

Backing up and restoring the DVC files and MLflow data: see `docs/backup-and-restore.md`.

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
evaluation_xgboost     780831d8c7be4feabff20d894a1abd62  (level5_fraud_evaluation)
calibration_xgboost    cfd6628cc7fc4001a49f4009a3bd4d19  (level5_fraud_calibration)
onnx_export_xgboost    3ba14af13d31406ea8b613936d337fb5  (level5_fraud_onnx)
model_card_xgboost     409ef8299fce413e9c86cf37521be633  (level5_fraud_evaluation)
dvc_artifact_link      5b508c7e19df4bb2b3d08e5042ee3161  (level5_reproducibility, the 5.10 link run)
```

Store: `mlflow.db` (SQLite) and `mlruns/` in the repo root. Both are gitignored and are NOT backed up (see
"Backup gap" below). The serving benchmark runs are `72638a5a66334e418855da439d2cab9b` (concurrency) and
`1edf4646da47405e8e663e3d80bc13ad` (micro-benchmark), both in `level5_fraud_serving`.

---

## Docker Image Tag

```
nvcr.io/nvidia/tritonserver:24.01-py3@sha256:3380720761045fc16ba3bcb96cfa54034531fc302df54ecac6b2a4deeab07bbd
```

This is the Triton Inference Server image that serves the model (see `docker-compose.yml`). The API image
(`project-ashfall-api:latest`) is built locally from `python:3.11-slim` and has no registry tag, so it has no
fully-qualified tag to record. It cannot be pulled from anywhere; rebuild it from the project Dockerfile instead.

---

## Python Environment Hash

```
b54e87f5ed7fdc3ead74326d60645f29bcb8ccea8fb9fa9a4166640636259cc1
```

**Recipe** (so anyone can recompute it): sha256 of the lines of `envs/training_env.txt`, sorted by plain
character order, joined with a newline, with one trailing newline, encoded as UTF-8 (608 lines):

```python
import hashlib
lines = sorted(open('envs/training_env.txt', encoding='utf-8').read().splitlines())
print(hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest())
```

**Caveat (read this):** `envs/training_env.txt` is a `pip freeze` of the whole global Windows Python install on
the training machine, NOT a clean virtual environment. It includes packages unrelated to this project (for
example `cohere`, `darts`, `langchain-core`, `openai`, `djangorestframework`) and one editable install from another
repository. It was taken on Python 3.12.10, while the project targets Python 3.11 (`pyproject.toml`, CI). So the
hash shows which packages were present, not the minimal set needed. The ML libraries that matter are pinned in
`requirements-ml.txt`. A clean-venv freeze on Python 3.11 is a documented follow-up.

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

## Level 3 Acceptance (history, partly superseded)

**Superseded:** the first Level 3 acceptance used a dataset where the injected patterns were not linked to
`isFraud`. Its data hash, AUC figures, reproducibility hash and pattern list were replaced at the Level 3
Correction below, and the old numbers have been removed from this file. The items that still hold:

- **Schema:** the Pandera schema passes on the enriched parquet (re-checked at the correction).
- **CI Schema Gate:** a `schema-validation` job exists in `ci.yml`.
- **E-commerce Store:** `scripts/ecommerce_store.py` generates transactions with an `ecom_` prefix; schema validation
  (`data/schemas/ecommerce_transactions.py`) passes on valid batches; live POST /score is blocked only by uvloop
  on Windows (works in Docker/Linux).
- **Merchant collusion:** deferred to the e-commerce store (see the correction below).

---

## Level 3 Correction

This section supersedes the earlier Level 3 acceptance figures (data hash, AUC results, pattern list), which
have been removed from the Acceptance section above.

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

---

## Level 5: Trained Model and DVC-Tracked Artifacts

**Model:** XGBoost, the winner of the bake-off (XGBoost vs random forest vs CatBoost), tuned with Optuna and then
calibrated with isotonic regression. Trained on 15 features (see the model card).

**The four artifacts** (all DVC-tracked; the `.dvc` pointer files are committed to git, the binaries are not):

| Artifact | DVC pointer | md5 |
|---|---|---|
| `models/xgboost/model.ubj` | `models/xgboost/model.ubj.dvc` | `c5d82437c5bf278eba6e32daa93465dd` |
| `models/xgboost/model.onnx` | `models/xgboost/model.onnx.dvc` | `bbe3570958207ae2f981b07fcfd4ebf3` |
| `models/calibration/isotonic_calibrator.pkl` | `models/calibration/isotonic_calibrator.pkl.dvc` | `df29a6a22de56134961032ad38b483d7` |
| `models/xgboost/shap_model.pkl` | `models/xgboost/shap_model.pkl.dvc` | `4830d819c1cf65fe17e1dc83b62cb245` |

All four describe one model (the SHAP booster and `model.ubj` agree to about 1.2e-07 on 100,000 test rows).
Each `.dvc` file carries a `meta:` entry with the link run ID above, and the link run logs the same hashes as
params (`dvc_md5_<name>`), so the link works in both directions. Link script: `scripts/fraud_dvc_link.py`.

**Key results** (test split): PR-AUC 0.9947, ROC-AUC 0.9999. The model is much better on injected fraud (PR-AUC
about 0.9985 for account takeover and 0.9988 for mule account) than on the original PaySim fraud (0.9444), so the
headline number is flattered by the injected patterns. Details and limits:
`reports/model/model_card.md`; fairness table: `reports/model/fairness_table.md`.

**Complete triplet for this model:** git commit (above) + DVC hashes (this table, plus the data hash above) + MLflow
run IDs (above).

### Backup gap (important)

DVC here is **local-only**: no DVC remote is configured (`.dvc/config` is empty). `dvc add` hashes the files and
copies them into the local cache `.dvc/cache` on this machine. The artifacts are version-tracked and recoverable
from that cache **on this machine only**. They are NOT backed up off-machine, and `dvc push` / `dvc pull` do not
work. The same is true of `mlflow.db` and `mlruns/`. Losing this disk loses the model binaries and the MLflow
history. Configuring a DVC remote is a documented follow-up.

### Branch-switch hazard for `model.ubj`

`model.ubj` was committed to git earlier (commit `69c73ac`, now in `main`). From the 5.10 commit onward it is
DVC-tracked and no longer git-tracked. Old commits and `main` still contain the old file, so switching branches can
change it on disk:

- Checking out a branch that still tracks it (for example `main`) puts that branch's old copy over your file.
  A newer retrained model would be overwritten.
- Checking out this branch from one that tracks it can delete the file from disk.
- Fix: run `dvc checkout`, which restores the file recorded in the `.dvc` pointer from the local cache. With no
  remote, this only works on a machine that has the cache (`.dvc/cache`); a fresh clone cannot fetch the file.
- Habit: after any branch switch or pull, run `dvc status` and `dvc checkout` before using the model.
