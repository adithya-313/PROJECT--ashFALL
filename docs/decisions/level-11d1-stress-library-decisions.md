# Level 11D-1: Stress-Test Library and World Cards Decisions

## 1. Five Core Architecture Decisions

1. **Stress Tests as Stackable Layers with Labelled Grounding Metadata**:
   Stress tests in `scenarios/stress/` are full Level 11C-3 configuration layers with an optional top-level `stress_test` block containing `real_event`, `exaggeration`, and optional `takes_effect_from_level`. Because they inherit standard layer semantics, stress tests compose cleanly via existing fast-mode, sizing, and layering machinery while carrying cryptographic and narrative provenance.
2. **Every Fast-Mode Run Produces a World Card**:
   Every execution of fast mode writes both `world_card.json` (machine-readable structured schema) and `world_card.md` (human-readable Markdown summary) alongside `manifest.json` and `events.jsonl`. This documents the world configuration, bank differences, regional distribution, applied stress tests, failure breakdown by domain, and fraud base-rate status.
3. **Strict Empirical Grounding of Real Events**:
   Four stress tests exaggerate a reported real event, and two (registry-outage, complaint-backlog) are labelled what-ifs because no published figures were found.
4. **Unified Grounding Table of Realism Parameters**:
   A dedicated module (`world/grounding.py`) audits every parameter across `background_realism.yaml`, `failure_realism.yaml`, and `device_realism.yaml`. It extracts provenance labels, training ranges, and citations, reporting parameter counts categorized by epistemic status (`assumption`, `reported`, `sourced`, `policy`).
5. **Issue R2 Marked Built**:
   Issue R2 ("Stress-test library replaces scenario timelines") is transitioned from `planned` to `built` in `config/issues.yaml`, evidenced by documentation in `scenarios/stress/README.md` and automated validation in `tests/unit/world/test_world_stress.py`.

## 2. Parameter Range Invariant and Zero Handling

Ranges must have low below high (core/config_validator.py check_range), so 'zero' stress values are written as tiny ranges, for example 0 to 0.0001.

## 3. Stress-Test Scenarios Catalog

| File | Changed Settings with Training Ranges | Real Event | Exaggeration | Takes Effect From Level |
| :--- | :--- | :--- | :--- | :--- |
| `bank-outage-long.yaml` | `bank_downtime_duration_minutes`: [480, 720] | May 2024 saw 31 bank downtime instances totalling over 47 hours, about 1.5 hours each (NPCI data via Economic Times, June 2024). | Each bank outage lasts 8 to 12 hours instead of 1 to 2 hours; success rate still dips partly (partial degradation), it does not fall to zero. | now |
| `big-festival.yaml` | `festival_uplift_fraction`: [0.5, 0.6] | Diwali 2025: the Diwali-eve record of 740 to 754 million UPI transactions was about 6 to 9 percent above that October's daily average of 695 million (NPCI-based reports, October 2025). | Festival uplift of 50 to 60 percent instead of the normal 5 to 25 percent, several times the reported 6 to 9 percent. | now |
| `complaint-backlog.yaml` | `complaint_delay_investment_scam_days`: [40, 60]<br>`complaint_delay_job_scam_days`: [40, 60]<br>`complaint_delay_unauthorised_debit_minutes`: [2880, 4320] | No published series on complaint backlogs was found; this is a what-if for complaint intake piling up. | Scam complaints arrive after 40 to 60 days instead of 3 to 15, and unauthorised-debit complaints after 2 to 3 days instead of minutes to hours. | 12 |
| `os-update-wave.yaml` | `os_wave_count_per_world`: [2, 3]<br>`os_wave_length_days`: [3, 7]<br>`os_wave_final_share_of_eligible`: [0.9, 1.0] | OS updates spread gradually: iOS adoption reached about two-thirds of devices months after release (global, February 2026); Android updates roll out in waves (Android update guidance). | Two or three waves where 90 to 100 percent of eligible phones update within 3 to 7 days, instead of one or two waves reaching 30 to 70 percent over 2 to 6 weeks. | now |
| `registry-outage.yaml` | `probability_a_bank_takes_part_in_dpip`: [0.0, 0.01]<br>`registry_entry_delay_days`: [119, 120] | No published outage of the shared fraud registry (mock DPIP and I4C) was found; this is a what-if. | The registry gives ashFALL almost nothing: at most 1 percent chance a bank takes part, and entries arrive after about 120 days. Payments still flow; only shared intelligence is lost. | 13 |
| `switch-outage.yaml` | `npci_incident_success_rate`: [0.0, 0.0001]<br>`npci_incident_count_per_30_days`: [1, 2]<br>`npci_incident_duration_minutes`: [180, 300] | NPCI-side incidents are rare and partial: 95 to 180 minutes, with success rates dropping to about 50 to 80 percent (NPCI uptime dashboard, March to April 2025). | At least one incident a month where NPCI routes almost nothing (0 to 0.01 percent success) for 3 to 5 hours. | now |

## 4. Deferral of new-bank Scenario

new-bank is deferred because it needs world-builder code: three banks are built into the world builder today, so adding a fourth would change build_world and its golden fingerprint (FP). The Level 11 plan marks this stress test as optional.

## 5. World Card Schema and Base-Rate Field

The world card records:
- **Settings**: seed, population (`n_people`), shop count (`n_shops`), horizon days (`n_days`), and range split (`range_name`).
- **Clock Start**: ISO-8601 start timestamp of the simulation calendar.
- **World Fingerprint**: SHA-256 bit-for-bit fingerprint of the world entity graph.
- **Banks**: Per-bank profiles covering kind, customer share, weak KYC share, hold delay factor, and citizen customer count.
- **Regions**: Geographic partition count and citizen residence breakdown.
- **Stress Tests**: List of applied stress test records including real event, exaggeration, effective level, and layer SHA-256.
- **Layers**: Sequence of standard non-stress configuration layers applied.
- **Payments**: Transaction totals for attempts, successes, total failures, and failures breakdown by domain (`customer`, `sender_bank`, `receiver_bank`, `npci_central`, `app_or_network`).
- **Base Rate**: Contains `fraud_share: None`, `fraud_inflation_factor: None`, and note `"Filled in Level 12, once fraud gangs exist."` Because synthetic fraud syndicates and mule networks are introduced in Level 12, genuine baseline payment runs in Level 11 have no ground-truth fraud transactions to evaluate.

## 6. Issue Register and Golden Value Updates

- In `config/issues.yaml`, issue R2 was changed from `status: planned` to `status: built` with evidence pointing to `scenarios/stress/README.md` and `tests/unit/world/test_world_stress.py`.
- Golden summary string in `scripts/golden_values.yaml` updated:
  - Previous: `"OK: 31 issues checked (built 27, planned 3, deferred 1)"`
  - New: `"OK: 31 issues checked (built 28, planned 2, deferred 1)"`
- R2 being built changed two pinned values in tests/unit/test_issue_register.py: counts 27/3/1 became 28/2/1, and the planned-issue example moved from R2 to R7.
