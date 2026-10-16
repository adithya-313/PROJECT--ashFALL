# Level 11D-2: Showcase Decisions

## 1. What the Level 11 Showcase Is
The Level 11 showcase (`make show LEVEL=11` or `python -m showcase --level 11`) runs real Level 11 simulation code, verifies 16 automated checks, prints the Section 8 checklist in the terminal, and writes `reports/showcase/level_11/index.html`, `result.json`, and evidence artifacts in `reports/showcase/level_11/evidence/`. It executes fully offline without a database, Docker, or external network services (`NEEDS_DATABASE = False`).

## 2. Core Architectural Decisions

1. **Self-Contained Offline Rich HTML Presentation**:
   The report page at `reports/showcase/level_11/index.html` is rendered from `showcase/templates/level_11_page.html` by `showcase/level_11_page.py`. It uses inline CSS with dark passbook styling and embedded inline SVG charts, with no external web fonts, external scripts, or remote CDN dependencies. All 31 template placeholders are populated from simulation evidence and catalog lookups.

2. **Clean Framework Off-Switch**:
   The showcase runner `showcase/__main__.py` dispatches `write_level_report(result, out_dir)` to write the standard framework report, and then checks:
   `if hasattr(module, "write_custom_page"): module.write_custom_page(result, out_dir)`
   If `write_custom_page` is removed or renamed, the framework falls back cleanly to the generic report without raising errors.

3. **Dual Execution Modes (Quick vs Full)**:
   - **Quick Mode** (`make show LEVEL=11 QUICK=1` or `python -m showcase --level 11 --quick --allow-not-run`): Runs checks 1 through 13 on tiny simulation parameters in ~11 seconds (budget: under 1 minute). Checks 14, 15, and 16 are marked `NOT RUN` with reason `"full mode only"`.
   - **Full Mode** (`make show LEVEL=11` or `python -m showcase --level 11`): Runs all 16 checks, including full live loopback replay, all 6 stress scenario isolation checks, and all 4 golden fingerprint recomputations in ~22 seconds (budget: under 5 minutes).

4. **Pure SVG Vector Telemetry**:
   Charts are constructed deterministically without matplotlib or third-party imaging libraries using `showcase/level_11_charts.py`. Geometries and bounds are loaded from `config/showcase_level_11.yaml`, generating pure SVG for daily transaction volume, rolling success rates, OS adoption curves, account inflow/outflow balances, and the simulation pipeline diagram.

5. **Loopback-Only Live Replay Verification**:
   Check 14 runs `run_live_mode()` against an ephemeral `ThreadingHTTPServer` bound strictly to `127.0.0.1` on an OS-allocated port (`port 0`), verifying byte-for-byte stream fidelity for all 2,915 events before shutting down cleanly.

## 3. Catalog of the 16 Showcase Checks

| # | Check Name | Kind | Proves | Pass Criteria |
|---|---|---|---|---|
| 1 | pre-registration amendments and proof | edge | Amendments recorded openly; no new-engine file predates pre-registration | 11 amendments verified in evaluation.yaml; new-engine file timestamps valid |
| 2 | issue register status counts and copy rejection | negative | 31 research issues categorized; broken copies rejected | 28 built, 2 planned, 1 deferred; copy missing fields raises ValueError |
| 3 | tiny world fast-mode simulation | normal | 60 people, 12 shops, 61 days generate realistic transactions and festival uplifts | Fast mode runs without errors; >= 2,000 attempts; failures > 0; hero chart built |
| 4 | empirical grounding table | normal | Parameter grounding across background, failure, and device realism | All 33 parameters categorized; labels count: 30 assumption, 3 reported |
| 5 | independent seed divergence | edge | Distinct master seeds produce divergent, independent transaction schedules | Seed 101 != Seed 102 events count and SHA-256 digests diverge |
| 6 | bank downtime partial degradation | edge | Bank downtime causes partial degradation without collapsing other banks | Success rate dips during outage; success rate > 0% (partial, not zero) |
| 7 | os update wave adoption curve and device stability | edge | OS wave accelerates device adoption while preserving fraud rule R5 | Peak adoption >= 90%; R5 new-device evaluation within bound |
| 8 | switch outage vs registry outage failure domains | negative | Central switch outage fails payments; deferred registry outage causes zero drift | Switch window success rate == 0%; registry outage event diff == 0 |
| 9 | wedding collection episode money surge | edge | Social money surges generate large inflows without false fraud alerts | Peak day inflow > 5x non-episode baseline; no spurious alerts |
| 10 | bank asymmetry in share and weak kyc | normal | Three financial institutions differ in reach, KYC rigor, and hold delays | Customer share, weak KYC share, and hold delay factors differ across banks |
| 11 | deterministic bit-for-bit repeatability | consistency | Fast mode is bit-for-bit reproducible for identical seeds | Re-run produces identical SHA-256 (matches golden `24fc0e25009d...`) |
| 12 | refusal of oversized world | negative | Worlds exceeding capacity are rejected with helpful guidance | World exceeding memory/runtime raises CapacityError with recommendations |
| 13 | double-entry ledger invariants | consistency | Currency conservation and non-negative balances across accounts | No negative balances; total debits == total credits; tampered ledger caught |
| 14 | live replay event stream fidelity | consistency | Live HTTP streaming mode replays fast-mode events byte-for-byte | All 2,915 events delivered to 127.0.0.1 matching event payload bytes |
| 15 | stress scenario parameter isolation | normal | Stress scenario layers modify only declared parameters | All 6 stress scenarios leave unmentioned configuration parameters identical |
| 16 | golden simulation fingerprints | consistency | Recomputes golden benchmarks in-process | FP, G1, G2, and G3 cryptographic digests match golden baselines |

## 4. Test Suites and Standards Enforcement

The showcase implementation is covered by unit tests adhering to project code standards (functions <= 30 code lines, docstrings >= 300 characters with WHAT, WHY, Example, Args, Returns):
- `tests/unit/showcase/test_level_11_charts.py`: Tests XML well-formedness and SVG element structure for all five chart builders.
- `tests/unit/showcase/test_level_11_checks.py`: Tests Level 11 check metadata, execution list, check bounds, and individual check runners.
- `tests/unit/showcase/test_level_11_page.py`: Tests HTML template placeholder substitution (all 31 replaced, 0 unpopulated) and file output writing.
- `tests/unit/world/test_new_code_comment_standard.py`: Audits that all new showcase modules satisfy function length and technical docstring schema.
