# Level 8: the safety net

Level 8 adds protections so that later levels cannot quietly break what already works. This page explains each one in plain
English and how to see them working.

## The protections

| Protection | What it does | Where |
|---|---|---|
| Fast CI test job | runs every test that needs no Docker on each push and pull request | job `pytest-fast` in `.github/workflows/ci.yml` |
| Regression baselines | saved snapshots (model v1 metrics, calibration bins, the API's shape, which services have a healthcheck); a test fails if the current state differs | `tests/baselines/` |
| Backup and restore | the DVC files go to a remote folder and MLflow data is backed up with checksums | [backup-and-restore.md](backup-and-restore.md) |
| Feature switches | four settings for later levels, all off | `core/config.py`, `.env.example` |
| `make check` | runs on your machine what the CI jobs run | `scripts/check.py` |
| No-egress test | proves the stack works with its outgoing network blocked and makes no outside connection attempts | Level 8 showcase, check 4 |

### Baselines: when is updating one correct?

Only when the change is on purpose (for example you retrained the model or added an API field). Run
`python scripts/update_baselines.py`, read the git diff of `tests/baselines/`, and explain it in your pull request. If the change
was not on purpose, undo it instead.

## See it working

```
python -m showcase --level 8            # full run, about 5 minutes
python -m showcase --level 8 --quick    # quick run, about 2.5 minutes (what CI runs)
make show LEVEL=8                       # the same thing, where make is installed
make check                              # or: python scripts/check.py [--no-docker] [--skip-showcases]
```

The five checks: (1) the checks pass and the baselines match; (2) a renamed API field in a temporary copy is caught by the
baseline test, and passes again after the undo; (3) files stored with DVC are deleted together with their cache and restored with
identical SHA-256 hashes; (4) the stack runs with outgoing network blocked and makes zero outbound attempts; (5) the Level 7
showcase still passes in quick mode. Reports are written to `reports/showcase/level_08/`.

### What quick mode does differently (used in CI)

- Check 1 runs the baseline tests and two small test files instead of all of `make check`.
- Check 3 restores generated files; full mode also restores the real seven artifacts (it needs network access and the local remote).
- Check 4 starts only Postgres, Redis and Redpanda and watches for 20 seconds; full mode starts every service except Triton and
  watches for 60 seconds.

## How the no-egress test works

It starts a second copy of the stack under its own project name (`ashfall-noegress`). The copy has no published ports, every
container sits on one Docker network marked `internal` (Docker blocks all traffic to the outside from it), and the real `.env` is
never used. A container on two networks would reach the internet again, so the test fails if any container is on a second network.
Then a small sidecar container with `tcpdump` shares each container's network and counts what it sees. The copy logs with the
local `json-file` driver (the normal stack sends logs to a fluentd address on the host).

Two control experiments prove the counter works: on a normal network a deliberate request to an outside address must show up as
outside packets, and on the internal network an outside name lookup must show up as DNS failures. If a control sees nothing, the
check fails because the counter cannot be trusted.

### Blind spots (measured)

- A request to a literal outside IP address is refused by the kernel ("Network unreachable") and sends no packet. It is only seen
  in the kernel's `OutNoRoutes` counter.
- The name in an outside lookup is not shown; only that a lookup failed.
- Traffic before the sidecar attaches is not seen.
- Only 20 seconds (quick) or 60 seconds (full) are watched. A tool that phones home once a day would not be seen. Settings that
  switch such calls off (Grafana, Loki and Tempo) are therefore also guarded by a test
  (`tests/baselines/test_offline_settings.py`).

### What it found: Grafana

With its defaults, Grafana looked up grafana.com at start-up (it downloads a plugin signature key). Four settings in
`docker-compose.yml` now turn that off: `GF_ANALYTICS_REPORTING_ENABLED`, `GF_ANALYTICS_CHECK_FOR_UPDATES`,
`GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES` and `GF_PLUGINS_PUBLIC_KEY_RETRIEVAL_DISABLED`.

Loki and Tempo have a `-reporting.enabled` option whose default is true (shown by their `-help`). No attempt was seen in the
60-second window, but a usage report on a timer could not have shown up in 60 seconds anyway. So both are now started with
`-reporting.enabled=false` in `docker-compose.yml`, and the config test is the guard, not the observation window. Loki had no
`command` before, so it now lists the image's default argument (`-config.file=/etc/loki/local-config.yaml`) next to the new flag.

Running containers keep their old settings until they are recreated: run `docker compose up -d` to apply the Grafana, Loki and
Tempo settings to your real stack.

## Limits

- The copy builds its own images (`ashfall-noegress-api`, `ashfall-noegress-outbox-relay`) and a sniffer image; they stay on your
  machine after the test and are reused.
- Redpanda: no outbound setting was found, and no attempt was seen.
- Check 4 stops nothing in your real stack. Check 5 runs the Level 7 showcase, which stops and restarts the Redpanda broker for a
  few seconds on purpose.
