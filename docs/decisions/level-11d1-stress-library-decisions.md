# Level 11D-1: Stress-Test Library and World Cards Decisions

## 1. Five Core Architecture Decisions

1. **Stress Tests as Stackable Layers with Labelled Grounding Metadata**:
   Stress tests in `scenarios/stress/` are full Level 11C-3 configuration layers with an optional top-level `stress_test` block containing `real_event`, `exaggeration`, and optional `takes_effect_from_level`. Because they inherit standard layer semantics, stress tests compose cleanly via existing fast-mode, sizing, and layering machinery while carrying cryptographic and narrative provenance.
2. **Every Fast-Mode Run Produces a World Card**:
   Every execution of fast mode writes both `world_card.json` (machine-readable structured schema) and `world_card.md` (human-readable Markdown summary) alongside `manifest.json` and `events.jsonl`. This documents the world configuration, bank differences, regional distribution, applied stress tests, failure breakdown by domain, and fraud base-rate status.
3. **Strict Empirical Grounding of Real Events**:
   Every stress scenario exaggerates a verified historical or operational event documented in the research dossier or public payment infrastructure reports (e.g., Diwali festive payment surges, major OS version rollout waves, multi-hour core banking downtimes, national switch routing interruptions, delayed bank registry onboarding, and customer dispute resolution backlogs).
4. **Unified Grounding Table of Realism Parameters**:
   A dedicated module (`world/grounding.py`) audits every parameter across `background_realism.yaml`, `failure_realism.yaml`, and `device_realism.yaml`. It extracts provenance labels, training ranges, and citations, reporting parameter counts categorized by epistemic status (`assumption`, `reported`, `sourced`, `policy`).
5. **Issue R2 Marked Built**:
   Issue R2 ("Stress-test library replaces scenario timelines") is transitioned from `planned` to `built` in `config/issues.yaml`, evidenced by documentation in `scenarios/stress/README.md` and automated validation in `tests/unit/world/test_world_stress.py`.

## 2. Parameter Range Invariant and Zero Handling

Ranges must have low below high (core/config_validator.py check_range), so 'zero' stress values are written as tiny ranges, for example 0 to 0.0001.

## 3. Stress-Test Scenarios Catalog

| Scenario File | Real Event | Exaggeration | Takes Effect |
| :--- | :--- | :--- | :--- |
| `scenarios/stress/big-festival.yaml` | Diwali / Dhanteras payment volume spike | 5x normal payment volume with 10x shop traffic | Immediate |
| `scenarios/stress/os-update-wave.yaml` | Major Android / iOS release waves | Two or three waves where 90 to 100 percent of eligible phones update within 3 to 7 days, instead of one or two waves reaching 30 to 70 percent over 2 to 6 weeks | Immediate |
| `scenarios/stress/bank-outage-long.yaml` | Major core banking technical outage (e.g., HDFC/SBI multi-hour incidents) | 24-hour total failure on Bank 0 (large public bank), compared to baseline 1-4 hour incidents | Immediate |
| `scenarios/stress/switch-outage.yaml` | NPCI central switch technical decline incidents | At least one incident a month where NPCI routes almost nothing (0 to 0.01 percent success) for 3 to 5 hours | Immediate |
| `scenarios/stress/registry-outage.yaml` | Bank onboarding delays in central intelligence registries (e.g. DPIP) | The registry gives ashFALL almost nothing: at most 1 percent chance a bank takes part, and entries arrive after about 120 days. Payments still flow; only shared intelligence is lost | Level 13 |
| `scenarios/stress/complaint-backlog.yaml` | Post-festival fraud and unauthorized transaction reporting disputes | Citizens dispute payments for up to 90 days, keeping bank backlogs elevated | Level 16 |

## 4. Deferral of new-bank Scenario

The `new-bank` stress test scenario is deliberately deferred to a later level. While adding a new banking institution into the ecosystem represents a realistic stress condition, full dynamic onboarding of an institution requires bank-failure resolution mechanics, inter-bank settlement adjustments, and core transaction routing reconfiguration that will be authored in later levels.

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
