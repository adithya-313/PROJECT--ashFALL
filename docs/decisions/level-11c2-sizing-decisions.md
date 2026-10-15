# Level 11C-2 sizing decisions and measurements

Status: approved and implemented in Level 11C-2a.
Author: adithya-313.

## 1. Context

Level 11 Task 17 and Section 6 define the simulation sizing framework:
presets, knobs with ratio scaling, expert configuration files, auto-sizing,
resource estimation before execution, and plain-English refusals.
In this architecture, auto-sizing means fitting world parameters to the host machine's
physical limits (primarily available free memory, available free disk space, and a maximum
run-time limit), rather than shrinking a simulation to fit an arbitrary time budget.

## 2. Finding: the slowdown

During scaling analysis on 2026-10-15, simulations exhibited non-linear slowdowns as population
counts increased. Profiling revealed the root cause in `world/calendar_runner.py`:
- In `_sample_single_everyday`, `rng.choice(shops)` was called for every transaction, repeatedly
  converting the catalog list.
- In `_generate_day_lookalike_payments`, for every lookalike person on every day, a list comprehension
  `others = [p for p in world.people if p.person_id != person.person_id]` reconstructed the entire
  citizen roster, and `rng.choice(others)` was invoked once per payment.

### Linux benchmark comparison (measured 2026-10-15)

| Population | Duration before fix (s) | Duration after fix (s) | Speedup factor |
| :--- | :--- | :--- | :--- |
| 60 people | 0.27 | 0.23 | 1.17x |
| 300 people | 1.07 | 0.83 | 1.29x |
| 1000 people | 5.50 | 2.28 | 2.41x |
| 2000 people | 18.84 | 4.53 | 4.16x |
| 4000 people | 69.28 | 9.77 | 7.09x |

All runs produced identical event, device stamp, and ledger hashes before and after the fix.

### Owner's Windows laptop baseline (measured 2026-10-15)

Specifications: 16.4 GB RAM (1.8 GB free at measurement time), 157.5 GB free disk, 12 CPU cores.
Timings were inflated by active tracemalloc heap tracking:
- Tiny (60 people): 970 payments/s
- Medium (1000 people): 243 payments/s
- Large (2000 people): 124 payments/s

This confirmed that the list rebuilding overhead severely degraded throughput across all platforms.

### Windows laptop CHECK S verification (measured in this session on 2026-10-15)

BEFORE fix on main (commit 7191337):
```
people=300 payments=15151 seconds=4.68 events=b5b59e78b990a6cf1f24e29a513314521a708400c2bf4f7cabca985d04af987f ledger=39ac5487bf81945888c91f7cc69fad93c25d99761fd22e195564132765a04fe1
people=1000 payments=54208 seconds=39.65 events=2bb8f253eeedbcf90d5aef809229db8f010efbedb2dffb3ca03f38684cd5e42d ledger=633419acf378c17c20a222d618e9bc32b61085285ab59ec8fd2bc243930f63e4
```

AFTER fix with index-based sampling:
```
people=300 payments=15151 seconds=1.24 events=b5b59e78b990a6cf1f24e29a513314521a708400c2bf4f7cabca985d04af987f ledger=39ac5487bf81945888c91f7cc69fad93c25d99761fd22e195564132765a04fe1
people=1000 payments=54208 seconds=3.89 events=2bb8f253eeedbcf90d5aef809229db8f010efbedb2dffb3ca03f38684cd5e42d ledger=633419acf378c17c20a222d618e9bc32b61085285ab59ec8fd2bc243930f63e4
```

For 1000 people, run time decreased from 39.65 seconds to 3.89 seconds (over 10x speedup),
while producing bit-for-bit identical event and ledger digests.

## 3. Decision: fix the slowdown with identical results

We replaced repeated list conversions with direct indexing:
- `_sample_single_everyday`: picks directly with `shops[int(rng.integers(0, len(shops)))]`.
- `_sample_lookalike_inbounds`: draws an index offset `pick = int(rng.integers(0, len(people) - 1))`
  and shifts `sender_position = pick + 1 if pick >= person_position else pick`.
With NumPy's Generator, drawing an integer uniformly in `[0, N)` consumes the exact same entropy and
advances the PRNG stream identically to `rng.choice`. Hash parity and golden checks prove zero divergence.

## 4. Decision: calibrate on the running machine

Hardware environments vary by orders of magnitude (the author's laptop ran several times slower
than dedicated Linux test servers). Because this project is open source and designed to run on diverse
workstations, laptops, and CI nodes, sizing metrics must be measured empirically at runtime:
- Throughput (payments per second)
- Frequency (payments per person-day)
- Disk footprint (file bytes per payment)
- Memory allocation (heap bytes per payment)

Calibration on a small 60-person baseline is valid because throughput remains steady as worlds scale up.
Test 7 (`test_speed_stays_steady_as_worlds_grow`) permanently guards this invariant.

## 5. Decision: estimates lean high

To guarantee safety, estimation uses conservative upper bounds rather than average rates:
- Payment frequency is scaled by a +15% safety margin (`payment_count_margin: 1.15`).
- On Linux sweeps across 9 seeds and scales (2026-10-15), measured payments per person-day ranged
  from 0.816 to 0.889 (0.845, 0.819, 0.828, 0.847, 0.862, 0.889, 0.816, 0.851).
  The calibration world measures about 0.80 payments per person-day, so a 15% increase ensures estimates
  reliably exceed real counts.
- Serialized JSON Lines size consistently measures between 351 and 353 bytes per payment event.
- Heap memory overhead measures about 1 KB per payment on both platforms (978 to 1,088 bytes on Windows),
  corresponding to approximately 50 MB per 1,000 citizens for a 61-day horizon.

## 6. Why about 1 KB per payment

In the current calendar runner architecture, each payment lives simultaneously across multiple Python objects:
- `PlannedPayment` schedule record
- `PaymentEvent` execution record
- Two `LedgerEntry` objects (debit and credit legs)
- `DeviceStamp` hardware telemetry record
- Purpose classification string

While columnar storage formats could reduce this footprint tenfold, 1 KB per transaction is lightweight
and easily manageable for all planned simulation scales.

## 7. Known limitation: single CPU core execution

The current simulation calendar engine is single-threaded and executes on a single CPU core.
While running on large server hardware provides substantially larger memory capacity for bigger populations
and allows executing multiple independent simulation worlds in parallel, an individual simulation run will
not automatically accelerate across multiple cores.

## 8. Built in Level 11C-2b

The following operational policies were finalized and built in Level 11C-2b:
- Four standard presets, all spanning 61 days:
  - Tiny: 60 people / 12 shops
  - Small: 300 people / 40 shops
  - Medium: 1000 people / 120 shops
  - Large: 2000 people / 250 shops
- Shop counts follow population using preset ratio knobs.
- Operational thresholds:
  - Maximum time: 30 minutes (overridable via `--max-minutes`)
  - Maximum memory: 50% of available free RAM
  - Maximum disk space: 50% of available free disk space
- Oversized simulations will be refused with clear diagnostic estimates and concrete instructions
  (suggesting the largest preset that fits, an allowable population count, or `--max-minutes`),
  never silently or automatically shrunk.

## 9. Options rejected and why

- **Time budget auto-sizing**: Rejected because time is a secondary planning estimate, not an authoritative
  dimension for economic modeling.
- **Automatic world shrinking**: Rejected because simulation scale decisions belong strictly to human operators;
  silently altering population ratios compromises experimental integrity.
- **Hardcoded developer speed constants**: Rejected because machine performance varies by factors of 5x or more;
  static numbers would cause severe miscalculations on slower laptops or virtual machines.
- **Estimating from mean transaction rates**: Rejected because average rates cause roughly half of all generated
  worlds to exceed estimated boundaries, violating conservative safety guarantees.

## 10. Decisions made for 11C-2b (owner, 2026-10-15)

- **`--estimate-only`**: Check a huge world safely without running it.
- **The seed is always typed (`--seed` or the expert file)**: No hidden default; the person always knows which world they made.
- **`psutil` reads free memory on every system**: Rejected: skipping the memory check with a warning (memory would go unchecked on some machines) and refusing to run (would block all Mac users).
- **The expert file must list exactly five fields**: Unknown or missing fields are refused, so a typo like `n_peopel` is caught instead of silently ignored. Machine limits still apply to expert files.
- **The old command still works and gives byte-identical files**: Confirmed by test 15.
- **Time estimates lean high**: In 11C-2a's test run on the owner's laptop, the estimate was 2.962 s and the real run 1.208 s, because the small calibration world spends more of its time on setup per payment. This only makes refusals more cautious.
