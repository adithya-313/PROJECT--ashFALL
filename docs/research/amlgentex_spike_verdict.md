# AMLGentex spike verdict

**What this file is for**
It records a test (a "spike") of the open-source AMLGentex project, to decide if ashFALL should build on it or only borrow ideas from it.
The test ran outside this repo on 2026-10-10 against AMLGentex commit `4b9a69dbd2edcf02fcec8eb3efe54937b6d77f1f` (committed 2026-02-22). No AMLGentex code or data was copied here.
Every answer names the file and line, or the command, that gave it.

## What was done
- Cloned https://github.com/aidotse/AMLGentex twice, both outside the repo: a Windows clone for reading licences, and a Linux (WSL Ubuntu, Python 3.12.3) clone with its own virtual environment for running it.
- Read the licence statements for the code and for generated data.
- Ran the README quick start, step 1 (data generation), two times with different settings.
- Read the code for each question below, and repeated one run to test the seed.
- Earlier the same day, a run on plain Windows failed (details under "CANNOT TELL"). The Linux run worked.

## Project facts

| Question | Answer | Where it came from |
| --- | --- | --- |
| Commit and date | `4b9a69d`, 2026-02-22, 678 commits | `git rev-parse HEAD`, `git log -1` |
| Latest tag or release | none (0 tags, 0 releases) | `git tag`; `gh api repos/aidotse/AMLGentex/releases` |
| Python required | `>=3.10,<3.14` | `pyproject.toml` line 10 |
| LICENSE (first 5 lines) | "Apache License / Version 2.0, January 2004 / http://www.apache.org/licenses/ / (blank) / TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" | `LICENSE` lines 1-5 |

## Licences

| Question | Answer | Evidence |
| --- | --- | --- |
| Code licence | Apache-2.0 | `LICENSE`; `README.md` line 9 (badge) and line 1141; `pyproject.toml` line 11 |
| Licence for generated data | NOT FOUND | No data licence in `LICENSE`, `README.md`, `scripts/README.md`, `experiments/README.md`, `src/data_creation/README.md` or `src/data_tuning/README.md` |
| Hugging Face dataset cards | NOT FOUND: the README links none, and the word "huggingface" does not appear in the repo's text files | `grep -rn -i "huggingface\|hugging face"` |
| Paper statement | The arXiv page (https://arxiv.org/abs/2506.13989) states no data licence; its only licence is for the paper itself | page read on 2026-10-10 |

Conclusions:
- (a) May ashFALL use the code? **YES**. Apache-2.0 (`LICENSE`).
- (b) May ashFALL use the generated data as an outside check? **CANNOT TELL**. No licence for the output was found. Data we generate ourselves with the code is the practical case, but the files do not say who owns or may reuse the output.
- (c) May ashFALL redistribute that data? **CANNOT TELL**, for the same reason.

## Runs

| Run | Config changed | Result |
| --- | --- | --- |
| 1: `experiments/template_experiment/config/data.yaml` | none (the template) | worked in 216 s. `accounts.csv` 10,000 rows; `alert_models.csv` 617; `normal_models.csv` 108,039; `transactions.csv` 39,532; `tx_log.parquet` 678,984 rows. |
| 2: `experiments/run2/config/data.yaml` (a copy) | `data.yaml`: `random_seed` 0 to 1, `total_steps` 100 to 60. `accounts.csv`: 10000 to 5000 accounts. `alertPatterns.csv`: `fan_in` count 10 to 30, `cycle` count 10 to 2. | worked in 131 s. `accounts.csv` 5,000; `alert_models.csv` 760; `normal_models.csv` 77,241; `transactions.csv` 20,548; `tx_log.parquet` 257,282 rows. |
| 2 repeated (`run2b`, same seed) | none | `tx_log.parquet` identical to run 2 (`DataFrame.equals` was True). |

Commands: `python scripts/generate.py --conf_file <data.yaml>`.
The output columns of `tx_log.parquet` are: step, type, amount, nameOrig, nameDest, bankOrig, bankDest, daysInBankOrig, phoneChangesOrig, oldbalanceOrig, newbalanceOrig, daysInBankDest, phoneChangesDest, oldbalanceDest, newbalanceDest, isSAR, patternID, modelType.
Changing the accounts, steps, seed and pattern counts changed the output size and the number of alert patterns, so those parameters do control the output.

## Code questions

| Question | Found? | Where and how it works | To add what ashFALL needs |
| --- | --- | --- | --- |
| a. Accounts, banks, customers | FOUND | `account.py` line 9 (`Account`, with `bank_id` at line 19); `accounts.csv` has a `BANK_ID` column and `count,bank_id` rows in the config; `transaction_graph_generator.py` lines 194-195 keep bank-to-account maps; line 351 `get_typology_members` can choose pattern members from one bank or all banks; `simulator.py` line 558 counts banks and the log has `bankOrig` and `bankDest` | easy: banks exist (the template uses one bank, `bank_a`) |
| b. Devices or shared control | NOT FOUND. The nearest thing is a per-account phone-change counter (`account.py` lines 60-115, `phone_changes`; log columns `phoneChangesOrig` and `phoneChangesDest`). Nothing links several accounts to one device or one actor. | | moderate to hard: needs a new device table and links between accounts, in the spatial step and in `account.py` |
| c. Delayed or noisy labels | PARTLY. Noise: `noise.py` line 9 `flip_labels`, a `missing_labels` function that sets labels to -1, and `apply_train_noise` (line 90) that applies noise to the training labels only. Delay: NOT FOUND. These are random per-account changes with no time in them. | | moderate: label delay needs a time per label; the noise functions work on a table of accounts and are easy to extend |
| d. Holds, freezes, interventions | NOT FOUND (`grep` for freeze, block, intervene, suspend, lien and hold in `src/data_creation` and `src/feature_engineering` finds only rejection-sampling code in `utils.py`) | The simulator builds the transaction schedule first and runs it (`simulator.py` lines 463-552, `run` at line 552); nothing reads a detector's decision during the run | hard: needs a loop where detection can stop money while the simulation runs |
| e. Adaptive fraudsters | NOT FOUND (only an unrelated `seed_strategy` option for choosing seed accounts, `ml_account_selector.py` line 51) | Data is generated before any model runs | hard: needs a new design |
| f. Pattern files | FOUND | `alertPatterns.csv` columns: `count,type,min_accounts,max_accounts,bank_id,is_sar,source_type`. The 8 shapes are fixed in code: `alert_patterns.py` lines 193-207 (`fan_out`, `fan_in`, `cycle`, `bipartite`, `stack`, `scatter_gather`, `gather_scatter`, `random`). An unknown type does nothing (line 209). Timing options `burstiness_bias_alert` and `mean_duration_alert` are in the default section of `data.yaml`. | moderate: a new tactic such as slow drip or fake routine life needs a new `_generate_...` method and a new line in that list, not just a config row. CANNOT TELL whether the timing options alone could give a slow drip. |
| g. Metrics | FOUND | `src/ml/metrics/metrics.py` line 56 `constrained_utility_metric`: alert budget `K` (top K), maximum FPR, or minimum recall. `preprocessor.py` lines 159 and 716: `split_by_pattern` keeps one pattern out of several splits. | free |
| h. Regimes | FOUND | `scripts/train.py` lines 63-80: `centralized`, `federated`, `isolated` | free |
| i. Determinism | FOUND | `simulator.py` lines 36-42 (`random_seed`, seeded `RandomState`); `src/data_creation/README.md` line 241. Two runs with the same seed gave identical output (above). | free |

## What ashFALL needs, in plain words
- Comes free: multiple banks, eight money-flow shapes (fan-in, fan-out, cycle and others), a seed for repeatable runs, an alert-budget metric, pattern-based splits, label noise, and the three regimes.
- Needs configuration only: how many accounts per bank, how many of each pattern and their sizes, amounts, timing, burstiness, and noise fractions.
- Needs code changes: new tactics (slow drip, pseudo-merchant camouflage, fake routine life), a device table and shared control, and label delay.
- Needs a new design: holds and freezes during a run, fraudsters who change tactics after being caught, and UPI-specific actions and tracing.

## Verdict
Verdict: borrow ideas and use as an outside check

Three reasons, from the evidence only:
1. The things that make ashFALL different (devices and shared control, label delay, holds, adaptive fraudsters) were not found, and holds and adaptive fraudsters would need a new design because the data is generated before any detection runs.
2. The pattern shapes are fixed in code (`alert_patterns.py` lines 193-207), so every new tactic means editing AMLGentex itself. Building on it would mean maintaining a changed copy.
3. What it does well is useful without building on it: Apache-2.0 code, repeatable runs, an alert-budget metric, pattern-based splits and three regimes. Those ideas can be borrowed, and its output can serve as an outside check.

## What would be needed to add devices, label delay and holds
- Devices: a device table and links from accounts to devices; pattern code that reuses one device across accounts; features built on those links.
- Label delay: a time for each label (when it becomes known), used in the data and in the train/test split. Today only random flips and missing labels exist (`noise.py`).
- Holds: a step in the simulation where a detector can stop or delay a payment while the run is going, and a way to measure the harm of stopping a genuine payment.

## CANNOT TELL
- The licence of generated data, and whether it may be redistributed (see above).
- Whether data generated here matches real UPI traffic.
- Whether the timing options alone could produce a slow drip.
- Why generation fails on plain Windows. The earlier run first hit a text-encoding error (fixed with `PYTHONUTF8=1`), then an `IndexError` at `transaction_graph_generator.py` line 149. The file it had just written (`spatial/degree.csv`) held lines ending in two carriage returns, which looks like a Windows line-ending problem. This was not repeated in this run.
- The model-training steps (README steps 3-4) were not run.

## How to repeat this
```
git clone https://github.com/aidotse/AMLGentex
cd AMLGentex
git checkout 4b9a69dbd2edcf02fcec8eb3efe54937b6d77f1f
python3 -m venv ../venv
../venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
../venv/bin/pip install networkx==3.4.2 numpy==2.1.0 scipy==1.14.0 pandas==2.2.2 pyarrow pyyaml tqdm==4.66.5 scikit-learn==1.5.1 optuna==3.6.1 matplotlib==3.9.2 torch_geometric==2.6.1 seaborn==0.13.2
../venv/bin/python scripts/generate.py --conf_file experiments/template_experiment/config/data.yaml
```
Use Linux or WSL. Generation takes about 4 minutes for the template.
