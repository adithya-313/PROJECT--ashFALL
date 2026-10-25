Report-only results for test and stress worlds are never used to tune the world or the threshold (A-026).

# Trivial-World Check Results (A-026)

Evaluation of single-feature AUPRC separability across all 680 planned worlds under Level 12h.
Candidate signals: `devices`, `dwell`, `pass_through`, `payer_count`, `payer_regions`, `routine_life`.
Gate threshold: 0.90 (applied strictly to training and validation splits).

## Gate Splits (Training and Validation)

| Split | Worlds Checked | Failed (>= 0.90) | Best AP Min | Best AP Median | Best AP Max | Dominant Signals |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| training | 320 | 0 | 0.1043 | 0.5528 | 0.8048 | routine_life: 258, dwell: 62 |
| validation | 120 | 0 | 0.1606 | 0.5189 | 0.7867 | routine_life: 101, dwell: 19 |

## Report-Only Splits (Test and Stress)

| Split | Worlds Checked | At/Above 0.90 | Best AP Min | Best AP Median | Best AP Max | Dominant Signals |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| test | 120 | 0 | 0.1665 | 0.5403 | 0.8577 | routine_life: 97, dwell: 23 |
| stress | 120 | 0 | 0.0707 | 0.4171 | 0.7587 | routine_life: 87, dwell: 33 |

## Held-Out Tactic Families (Report-Only: Test and Stress)

| Family | Worlds Evaluated | At/Above 0.90 | Best AP Min | Best AP Median | Best AP Max | Dominant Signals |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| slow_drip | 111 | 2 | 0.0140 | 0.3159 | 1.0000 | routine_life: 104, dwell: 5, pass_through: 2 |
| device_rotation | 129 | 0 | 0.0290 | 0.3683 | 0.7570 | routine_life: 78, dwell: 50, devices: 1 |

## Fraud Share Distribution

| Split | Worlds | Fraud Share Min | Fraud Share Median | Fraud Share Max |
| --- | ---: | ---: | ---: | ---: |
| training | 320 | 0.000595 | 0.003252 | 0.012034 |
| validation | 120 | 0.000479 | 0.003325 | 0.011143 |
| test | 120 | 0.000386 | 0.006141 | 0.094340 |
| stress | 120 | 0.000000 | 0.002314 | 0.036220 |

## Report-only findings (A-026 (c) applies)

The following report-only signals or held-out families reached or exceeded 0.90 AUPRC:

- Held-out family `slow_drip` in `test-b00-0012` (test): best signal `routine_life` achieved AUPRC 1.0000.
- Held-out family `slow_drip` in `stress-b02-0032` (stress): best signal `routine_life` achieved AUPRC 1.0000.

Decision (architect, 2026-10-25): no change to the world. The separating signal is routine_life, not a slow_drip mechanism (dwell or amount); it reaches 1.0 in 2 of 111 slow_drip worlds, where the held-out gang has few mules; and routine life is the signal A-025 (c) already records as untested because fake_routine_life is deferred. No fix can be justified without the test result, so nothing is changed and no test or stress world is regenerated.
