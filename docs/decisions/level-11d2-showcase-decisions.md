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
| 1 | pre-registration amendments and proof | consistency | Amendments to evaluation rules are recorded openly, and no new-engine file predates the pre-registration. | 11 amendments recorded; new-engine files all postdate evaluation.yaml |
| 2 | issue register status counts and copy rejection | negative | The 31 research issues show 28 built, 2 planned and 1 deferred; broken copies without mandatory fields are refused. | 28 built, 2 planned, 1 deferred; broken copies with missing or unlisted IDs rejected |
| 3 | tiny world fast-mode simulation | normal | A tiny world of 60 people and 12 shops over 61 days runs end to end, and after removing weekend, month-start and fixed-date payments, festival days carry the extra payments the plan sets. | Fast mode runs without errors; >= 2,000 attempts; festival uplift measured; hero chart built |
| 4 | empirical grounding table | normal | Every parameter across background, failure and device realism is catalogued with an epistemic label and source citation. | All 33 parameters categorized; labels count: 30 assumption, 3 reported |
| 5 | independent seed divergence | edge | Two different seeds place festivals, outages and OS update waves on different days. | Festival peak days, outage start days, and OS wave update days differ between seeds |
| 6 | bank downtime partial degradation | edge | A multi-hour bank downtime causes partial success rate degradation without collapsing unrelated bank transactions. | Success rate dips during outage; success rate > 0% (partial, not zero) |
| 7 | os update wave adoption curve and device stability | edge | An exaggerated OS update wave raises OS adoption, and new-device flags stay within the k = 4 bound (issue R5). | Peak adoption >= 90%; R5 new-device evaluation within bound |
| 8 | switch outage vs registry outage failure domains | negative | A central switch outage fails all payments inside the window, while deferred registry outages cause zero event drift. | Switch window success rate == 0%; registry outage event diff == 0 |
| 9 | wedding collection episode money surge | edge | An injected wedding collection episode shows up as incoming money on the host account, starting and ending inside its window. | Inflow concentrates inside the collection window; baseline inflow outside is 0 |
| 10 | bank asymmetry in share and weak kyc | normal | The three banks differ in customer share, weak-KYC share and hold-response delay. | Customer share, weak KYC share, and hold delay factors differ across banks |
| 11 | deterministic bit-for-bit repeatability | consistency | Identical seeds produce bit-for-bit identical payment event files matching golden SHA-256 24fc0e25... | Re-run produces identical SHA-256 (matches golden `24fc0e25009d...`) |
| 12 | refusal of oversized world | negative | Proposed worlds exceeding host memory or runtime limits are safely refused with actionable capacity guidance. | World exceeding runtime is refused; estimate and limit reported; nothing written |
| 13 | double-entry ledger invariants | consistency | Double-entry ledgers maintain non-negative balances, entry consistency, and total currency conservation. | No negative balances; total debits == total credits; tampered ledger caught |
| 14 | live replay event stream fidelity | consistency | Live streaming mode replays fast-mode events across loopback HTTP with byte-identical payload fidelity. | All 2,915 events delivered to 127.0.0.1 matching event payload bytes |
| 15 | stress scenario parameter isolation | normal | Every stress scenario layer modifies only its named parameters, leaving unmentioned domain settings identical. | All 6 stress scenarios leave unmentioned configuration parameters identical |
| 16 | golden simulation fingerprints | consistency | The golden fingerprints FP, G1, G2 and G3 recompute in this run and match the stored values. | FP, G1, G2, and G3 cryptographic digests match stored baselines |

## 4. Test Suites and Standards Enforcement

The showcase implementation is covered by unit tests adhering to project code standards (functions <= 30 code lines, docstrings >= 300 characters with WHAT, WHY, Example, Args, Returns):
- `tests/unit/showcase/test_level_11_charts.py`: Tests XML well-formedness and SVG element structure for all five chart builders.
- `tests/unit/showcase/test_level_11_checks.py`: Tests Level 11 check metadata, execution list, check bounds, and individual check runners.
- `tests/unit/showcase/test_level_11_page.py`: Tests HTML template placeholder substitution (all 31 replaced, 0 unpopulated) and file output writing.
- `tests/unit/world/test_new_code_comment_standard.py`: Audits that all new showcase modules satisfy function length and technical docstring schema.

## 5. Check 3: Festival Uplift After Removing Other Calendar Effects

**Problem.** A plain inside-versus-outside comparison of daily counts on the tiny world (seed 101) gives 47.5 payments a day inside festival windows and 48.1 outside. The engine's daily rate is `festival_multiplier x month_start_multiplier x weekend_multiplier`, and on seed 101 the non-festival days happen to include month-start days and weekends. Changing the seed to one where the comparison happens to pass was rejected: that would be choosing the data to fit the claim.

**First attempt, rejected.** Applying the multiplier model to all payments gave a festival excess of only 45-70% of the planned excess at every size (2,000 people: excess 686.7 vs planned 1,505.9). Only 4,000 people passed both tests, with 2,000 and 8,000 failing, so it was not used. The cause: only `everyday` and `lookalike_in` payments are scaled by `daily_multiplier` (`world/calendar_runner.py`). Salary, rent, bill and episode payments sit on fixed dates.

**Method.** Each day's count is split into `c_d = u_d + s_d` using `RunResult.event_purposes`, which has one purpose per event. `run_world` raises an error for any purpose outside `PAYMENT_PURPOSES`, so the split covers every event. `s_d` is the `everyday` and `lookalike_in` count; `u_d` is everything else. `other_d` and `fest_d` come from the engine's own `month_start_multiplier`, `weekend_multiplier` and `festival_multiplier`. Then:

- `base = C_n / S_n`, using non-festival days only (`C_n` = sum of `s_d`, `S_n` = sum of `other_d`);
- `expected_without = base x S_f`, `observed = sum(s_d)` and `planned_excess = sum(base x other_d x (fest_d - 1))`, over festival days;
- `se_excess = sqrt(observed + (S_f / S_n)^2 x C_n)`;
- pass only if `excess > 4 x se_excess` and `|excess - planned_excess| <= 4 x se_excess`.

**Size rule (policy, fixed before looking at results).** Take the smallest `n_people` in `[1000, 2000, 4000, 8000]` (shops = people / 5) whose predicted `planned_excess / sqrt(expected_without)` is at least 8. The prediction uses the plan's multipliers and the tiny world's base rate per person. Predicted ratios on seed 101: 1000 -> 5.11, 2000 -> 7.23, 4000 -> 10.23, 8000 -> 14.47, so the rule picks 4,000 people and 800 shops. The rule is stored under `festival_world` in `config/showcase_level_11.yaml`.

**Measured result (seed 101, 4,000 people, 800 shops, 61 days).** base 2,963.33; observed 77,631; expected_without 74,594.9; excess 3,036.1; planned_excess 2,784.0; se_excess 363.7 (8.3 standard errors); fixed-date payments `u_d` total 15,773. Both tests pass. The tiny world's plain daily averages are still in the evidence as the plain daily view.

## 6. Checks 6 and 7: Measured Outage Counts and the Four-Sigma R5 Bound

**Check 7 (os-update-wave, issue R5).** The bound is `|layer_new - base_new| <= 4 x sqrt(max(base_new, 1))`, four Poisson standard deviations. What the layer changes: under `device_realism` it raises `os_wave_count_per_world` (training) from [1, 2] to [2, 3], shortens `os_wave_length_days` from [14, 42] to [3, 7] days, and raises `os_wave_final_share_of_eligible` from [0.3, 0.7] to [0.9, 1.0]. That matches its `stress_test.exaggeration` text. On seed 101 the base run draws 2 waves of 23 and 33 days, reaching 34% and 38% of phones, for 25 `os_update` events spread over 20 days. The layered run draws 3 waves of 3 to 6 days, reaching 91% to 97%, for 79 `os_update` events on 10 days. (An earlier "12 vs 6" figure was a count of days, not events.) Adoption is measured as the daily share of phones on the newest OS generation. The device schedule is redrawn with the run's own seed stream and checked against the run's applied-event count. Base 28.3% -> 48.3%, layer 28.3% -> 90.0%; largest one-day rise 1.7 vs 10.0 points. New-device flags are 7 vs 7, bound ±10.6.

**Check 6 (bank-outage-long).** The check measures the first bank downtime window only and reports raw counts: outage bank 2 of 3 succeeded, other banks 9 of 10. The baseline (98.8%) is the success rate outside every bank downtime window in the run. The samples are small, so the check claims only what the counts show. At least one outage-bank payment succeeds and at least one fails with `bank_downtime`, and no payment between the other banks fails with `bank_downtime`.

## 7. Checks 9 and 5: Episode Stream and the Page-Built Seed Table

**Check 9.** The injected wedding-collection Episode is planned with the episode random stream, the second value of `spawn_life_streams(101)`, which is the stream fast mode passes to `plan_episode_payments`. A hand-built generator is no longer used. The episode runs from minute 7,200 to minute 28,800, which is days 5 to 19. Measured: Rs 64,994.67 across 39 successful incoming payments on days 5-19, Rs 0 outside, peak day 15 at Rs 8,611.04.

**Check 5.** The `SideBySideChart` dict subclass, which carried HTML through the chart field, has been removed. The check stores plain timing rows in its evidence (timing, seed 101 day, seed 102 day, differs). `showcase/level_11_page.py` builds the table from those rows and passes every cell through `html.escape`. A timing that one seed lacks shows as "none" rather than day 0. The comparison seed is read from `comparison_world.seed` in config.
