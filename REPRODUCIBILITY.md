# REPRODUCIBILITY

This file records every artifact needed to reproduce any result from this project —
training run, evaluation, benchmark, or deployment. The discipline starts now, before
anything is trained, so it becomes a habit rather than an afterthought. Fill in each
section every time you produce a result worth keeping. A result that cannot be
reproduced is a result that cannot be trusted.

---

## Git Commit SHA

```
PLACEHOLDER
```

<!-- The exact git commit hash (`git rev-parse HEAD`) of the code that produced the result.
     Fill this in immediately after running an experiment or generating a report. -->

---

## DVC Data Hash

```
PLACEHOLDER
```

<!-- The DVC-tracked hash of the input dataset(s) used (`dvc status` or from dvc.lock).
     Fill this in when DVC is wired up in Level 2+. Until then, note the data source manually. -->

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
