# Regression baselines

A baseline is a saved snapshot of something that must not change by accident. The tests in this folder build a fresh
snapshot and compare it with the saved file. If they differ, the test fails and says what changed.

| Saved file | What it protects |
|---|---|
| `model_v1_metrics_baseline.json` | model v1 numbers (PR-AUC, ROC-AUC, row counts) and the DVC hashes of its data and files |
| `calibration_bins_baseline.json` | the calibration bins of model v1 |
| `api_openapi_snapshot.json` | the API routes, and the request/response shapes of `/health` and `/score` |
| `compose_health_baseline.json` | which Docker Compose services have a healthcheck (a config check; nothing is started) |

No Docker, database or Kafka is needed to run them: `pytest -q tests/baselines`.

## When is updating a baseline correct?

Only when the change is on purpose, for example you retrained the model, added an API field, or gave a service a
healthcheck. Then run `python scripts/update_baselines.py`, read the git diff of this folder, and explain it in your
pull request. If the change was not on purpose, undo it instead of updating the baseline.

## Limits

CI cannot re-score the model yet (the model files are not available there). The model baseline only compares the
committed report files with the saved baseline.
