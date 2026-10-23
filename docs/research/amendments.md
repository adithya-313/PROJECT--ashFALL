# Amendments to pre-registered files

An amendment is a change to a pre-registered config file (such as `config/evaluation.yaml`)
that adds, splits or re-defines a metric or decision block. An amendment is allowed only
before any results exist for the thing it changes. Every entry below records what changed,
why, and whether results existed at the time.

Note: review issues are written R1–R9. The rules baseline's own rules are written
"rule 1" to "rule 4" (see `docs/research/rules_baseline_spec.md`), to avoid confusion.

## A-001

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | L3, N4, A2, E1, R3, R4 |
| File | `config/evaluation.yaml` |
| What changed | Added three new top-level blocks (`money_held_and_returned`, `base_rate_reweighting`, `genuine_harm`) and two new items in the `not_decided_here` list. Nothing existing was changed or removed. |
| Why (held vs returned) | A single "money recovered" number conflates money under lien with money actually given back; splitting shows both. |
| Why (base-rate re-weighting) | Small simulated worlds have a higher fraud share than real life, inflating precision-style numbers; re-weighting shows the honest version. |
| Why (genuine harm) | Harm includes genuine shops restricted downstream and waiting time for wrong liens, not only frozen accounts. |
| Results exist? | No. |

"Money recovered" was never a key in `config/evaluation.yaml`. It appears only in
`docs/plan/section-07-measurement-and-evaluation.md` (line 14) and in the Level 7
showcase code. Nothing was renamed and no frozen code was touched.

No model code exists yet. The `models/` directory holds only v1 PaySim artefacts
(DVC-tracked, excluded by the `earlier_work_note` in `evaluation.yaml`).

## A-002

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | R1, L2 |
| Files | config/world_ranges.yaml, config/research_status.yaml, docs/plan/section-02-research-foundation.md |
| What changed | Wording only, no number or range changed. |
| Why | The 37% was being read as a victim complaint rate. |
| Results exist? | No. |

## A-003

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | G1, G3, G4, G5, G6, L1, L3, L5, E1, E2, E3, N1, N2, N3 |
| Files | `docs/research/mule_typologies.md` |
| What changed | 15 verified findings added as a new table section; no existing row changed, research_status.yaml untouched. Source dates corrected after a second check: Indore 3 Sep 2026, Telangana 30 Apr 2024; Kerala source is Deccan Herald (PTI not found). |
| Why | Task 2: verify the October 2026 findings before parameters depend on them. |
| Results exist? | No. |

## A-004

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | G1, G2, G3, G4, G5, G6, A2, E3 |
| File | config/world_ranges.yaml and tests/unit/test_research_configs.py |
| What changed | 23 new parameters added, 2 labelled reported and 21 assumption; no existing parameter changed; the test count updated from 10 to 33 and a new test protects the original ten. |
| Why | task 3: record ranges before any world code exists. |
| Results exist? | No. |

## A-005

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | L1, L2, L3, L4, L5, A1, A3, N4 |
| Files | config/world_ranges.yaml and tests/unit/test_research_configs.py |
| What changed | 18 new parameters, 1 reported and 17 assumption; no existing parameter changed; count test 33 -> 51; victim_report_delay_minutes is superseded in practice but left untouched. |
| Why | task 3: record ranges before world code exists. |
| Results exist? | No. |

## A-006

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | N1, N2, N3, E1, E2, E3, E4, G2 |
| Files | config/world_ranges.yaml and tests/unit/test_research_configs.py |
| What changed | 22 new parameters, 1 reported and 21 assumption; atm_daily_limit_inr corrected from an assumption of 20000-50000 training to a reported range 20000-100000 training with a source, because bank pages and the RBI FAQ were found; the ten original parameters untouched; count test 51 -> 73 |
| Why | task 3 plus a correction to a guess made earlier in the same level |
| Results exist? | No. |

## A-007

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | R5 |
| File | docs/research/rules_baseline_spec.md |
| What changed | a new design-constraint section added at the end; no existing rule changed; no numbers added |
| Why | OS updates must not trigger a new-device rule |
| Results exist? | No. |

## A-008

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | N1, N2, N3 |
| File | docs/interfaces/national_systems.md |
| What changed | a new subsection of mock fields: identifier reuse, per-bank DPIP participation, hold latency; no existing field changed |
| Why | task 6: the mock national systems must carry what the world will model |
| Results exist? | No. |

## A-009

| Field | Value |
| --- | --- |
| Date | 2026-10-10 |
| Issue IDs | R6 |
| Files | config/issues.yaml, core/issue_register.py, tests/unit/test_issue_register.py |
| What changed | new issue register of 31 issue IDs and a checker that looks up every piece of evidence for real; the Level 10 validator and CI files are untouched; the register is enforced through the existing fast tests |
| Why | task 7: no issue can be forgotten silently |
| Results exist? | No. |

## A-010

| Field | Value |
| --- | --- |
| Date | 2026-10-11 |
| Issue IDs | None |
| Files | `tests/unit/test_level_10_module.py` |
| What changed | The stale "no new-engine code exists" assertion in `tests/unit/test_level_10_module.py` was replaced by the date-ordering check that every entry in `new_engine_dates` is strictly later than `evaluation_date`. |
| Why | Level 11 legitimately added `world/`, which made the old assertion permanently false. The new check proves all new-engine paths postdate `config/evaluation.yaml`. |
| Results exist? | No. |

No model code exists yet (models start in Level 18; `world/` is simulator code).

## A-011

| Field | Value |
| --- | --- |
| Date | 2026-10-13 |
| Issue IDs | A1 |
| Files | config/world_ranges.yaml, tests/unit/test_research_configs.py, config/issues.yaml |
| What changed | 6 new assumption parameters for episode payment patterns; no existing parameter changed; count test 73 -> 79, assumption label count 63 -> 69. |
| Why | Level 11 piece 11B-3c needs amounts and timing to turn episodes into payments; none is published. |
| Results exist? | No. |


## A-012

| Field | Value |
| --- | --- |
| Date | 2026-10-17 |
| Issue IDs | E1 |
| Issue IDs v2 | 12-7 |
| Files | `config/world_ranges.yaml` (`world_fraud_inflation_factor`, reason text and tags only), `docs/research/amendments.md` |
| What changed | The "inflation factor" had three meanings that did not agree: a range in `config/world_ranges.yaml`, a sentence in `config/evaluation.yaml` that takes the factor "from the world card", and two empty fields in `world/world_card.py`. From now on the range in `world_fraud_inflation_factor` is only the TARGET BAND: how much more fraud a small world is aimed to hold than real life. Its ranges are unchanged, and no parameter is added, removed or relabelled, so the 79-parameter and 5/5/69 label counts stand. The MEASURED factor of a world is its complaints per 100,000 payments divided by 1.05, where 1.05 is `real_world_complaints_per_100000_upi_payments`. It is measured in Level 12g and written onto the cards of fraud-on worlds. Fraud-off cards keep their Level 11 values. `base_rate_reweighting` in `config/evaluation.yaml` already takes the factor "from the world card"; from now on that means the measured factor. Today `world/world_card.py` records only `fraud_share: None` and `fraud_inflation_factor: None` with the note "Filled in Level 12"; that is a placeholder, not a measured factor. |
| Why | Reweighting a result from a small world back to real-life rates needs one agreed number. The 1.05 is itself derived and is an upper-bound proxy, because it counts all financial-fraud complaints, not only UPI ones. Dividing by an overestimate makes the measured factor a LOWER BOUND on the true inflation, so world cards must say so. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-013

| Field | Value |
| --- | --- |
| Date | 2026-10-17 |
| Issue IDs | None |
| Issue IDs v2 | 12-6 |
| Files | `config/evaluation.yaml` (new block `trivial_world_check`), `docs/research/amendments.md` |
| What changed | A written test for a "trivial world", one where fraud is so easy to spot that any model would look good. The metric is the best single feature's area under the precision-recall curve (AUPRC). A world FAILS if any single candidate feature reaches AUPRC of 0.90 or more. The report always lists the top 3 features' AUPRC next to the world's fraud share, because AUPRC only means something beside the share it is measured against, and a world approaching the line should be visible before it fails. The check runs on VALIDATION worlds only, never on test or stress worlds (rule 17). The response to a failure is to fix the world, never the threshold. |
| Why | PaySim was rejected in Level 9 because single features separated fraud almost perfectly, but "almost perfectly" had no number. 0.90 marks "almost perfectly". It is a policy decision, not a measurement. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-014

| Field | Value |
| --- | --- |
| Date | 2026-10-17 |
| Issue IDs | G1, G5 |
| Issue IDs v2 | 12-8 |
| Files | `config/tactic_families.yaml` (new), `config/evaluation.yaml` (unchanged, still owns the held-out list), `tests/unit/test_tactic_families.py` (new), `docs/research/amendments.md` |
| What changed | Written definitions of the tactic families, frozen before the first gang world exists. Seven training families (fast_pass_through, layered_chain, fan_in_collector, rented_account_mule, shared_device_ring, app_run_payout, basic_merchant_mule) and one trait (low_balance_after_outflow). Four held-out families, each paired with its nearest training family: slow_drip (fast_pass_through), device_rotation (shared_device_ring), pseudo_merchant_camouflage (basic_merchant_mule) and fake_routine_life (rented_account_mule). Each held-out family lists what differs and what is forbidden in training, so a held-out tactic cannot leak into training worlds by accident. The held-out definitions are project design and not sourced (typology Row 8). The numeric dividing lines (dwell time, amount bands, device lifetimes) are registered as labelled building-block parameters in Level 12b, before any gang world runs. |
| Why | The held-out tactics overlapped training mechanisms and only basic_merchant_mule was named on the training side, so "held out" could not be checked. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-015

| Field | Value |
| --- | --- |
| Date | 2026-10-17 |
| Issue IDs | None |
| Issue IDs v2 | 12-18 |
| Files | `config/evaluation.yaml` (new block `validation_worlds`), `docs/research/amendments.md` |
| What changed | The rule for validation worlds, which `world_splits` already named but nothing defined. They use TRAINING ranges and seeds independent of the training seeds. They are the only place tuning happens (rule 17). Seed disjointness is enforced by the Level 12f sampler. The number of worlds per split stays in `not_decided_here` until 12f commits it from measured run time. |
| Why | Without a split of its own, tuning would happen on training or test worlds and quietly leak into the final judgement. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-016

| Field | Value |
| --- | --- |
| Date | 2026-10-17 |
| Issue IDs | G1, G5 |
| Issue IDs v2 | 12-9 |
| Files | `config/tactic_costs.yaml` (new), `tests/unit/test_tactic_families.py`, `docs/research/amendments.md` |
| What changed | Units for what a tactic costs. Money is in rupees, stored as paise in code (the same as the ledger). Time is in simulated minutes. Risk is a probability per use, a range labelled assumption whose values are set in Level 12b. Two money anchors come from the finding `bengaluru_rented_mule_suppliers` in `docs/research/mule_typologies.md`, both labelled ASSUMPTION because the figures are reported and not confirmed: the account-supplier fee as a share of money moved, range 0.03 to 0.06 (around the reported Rs 4,000 per Rs 1 lakh, which is 4%), and the operator share, range 0.20 to 0.30. They are kept in `config/tactic_costs.yaml`, so `config/world_ranges.yaml` gains no parameter. |
| Why | Tactic costs had no units, so two tactics could not be compared on cost. The source says "the Rs 4,000 per Rs 1 lakh and 20-30% shares not confirmed", so they are ranges, not facts. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-017

| Field | Value |
| --- | --- |
| Date | 2026-10-22 |
| Issue IDs | G3, G4, G5 |
| Issue IDs v2 | 12-8 |
| Files | `config/interactive_runner.yaml`, `config/gang_reactions.yaml` (new), `world/gang_feed.py` (new), `world/gang_reactions.py` (new), `docs/research/amendments.md` |
| What changed | Formal design decisions for reactive criminal syndicates and observation feeds in Level 12d: (a) Observation feed: each gang draws ONCE at run start a delay (whole minutes, uniform in its split's gang_feed_delay_minutes range) and a miss share (in gang_feed_miss_share); each feed item is then dropped with that miss share, else delivered at happened_minute + delay. Gangs see only items about accounts they hold, never truth. (b) Fixed reactions: each gang draws ONCE whether it follows each rule; probability = a value drawn from that rule's world_ranges range. Test (g) is strict for gangs that follow the stop rule. (c) A stopped account: every later gang payment into or out of it and every gang withdrawal from it is cancelled before it runs; a mule's burned_from becomes the minute the lien was seen, if earlier. (d) Planned for later rounds, recorded now: rotation to reserved spare accounts and amount splitting that respects the slow_drip dividing line (D-2); market resale with non-overlapping ownership spans (D-3); an optional gang family field plus a family -> allowed-blocks mapping, with validation worlds treated as training (D-4). (e) New random stream children: 14 gang_observation_feed, 15 gang_reactions (16 reserved for the account market in D-3). |
| Why | A reactive world requires realistic market frictions and information delays: gangs cannot see bank ledgers, encumbrance books, or ground truth, but react to observable friction (liens, holds, declines). Decisions must be registered before implementation. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-018

| Field | Value |
| --- | --- |
| Date | 2026-10-23 |
| Issue IDs | G5 |
| Issue IDs v2 | None |
| Files | `config/interactive_runner.yaml`, `config/account_market.yaml` (new), `world/account_market.py` (new), `world/interactive_runner.py`, `world/interactive_records.py`, `world/interactive_loop.py`, `tests/unit/world/test_account_market.py` (new), `docs/research/amendments.md` |
| What changed | Secondary account market resale across sequential criminal syndicates in Level 12d (G5): (a) Resale scope: only burned rented mule accounts with open-ended role spans (`valid_to` is None) from prior gangs are eligible resale offers for subsequent syndicates; borrowed KYC and business mules are excluded. (b) Non-overlapping spans: for each resale, the seller gang's role span and device control span are closed with `valid_to` set to `resale_minute`, and the buyer gang's span starts at `resale_minute`, ensuring strict non-overlapping temporal ownership in simulation truth. (c) Random stream allocation: child 16 (`account_market`) draws the resale share once per run, and takes exactly one draw per candidate rented mule in role order regardless of offer availability or acceptance. (d) Known gap: leftover illicit money in the resold account from the seller gang may be forwarded by the buyer, capped by the illicit money guard. |
| Why | Police reports document syndicates acquiring accounts from common suppliers who resell burned kits. Ground rule G5 and amendment A-017 pre-registered this mechanism. Strict non-overlapping ownership preserves truth invariant checks and temporal isolation. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-019

| Field | Value |
| --- | --- |
| Date | 2026-10-23 |
| Issue IDs | None |
| Issue IDs v2 | 12-8 |
| Files | `world/tactic_families.py` (new), `world/gang_config.py`, `world/interactive_runner.py`, `tests/unit/world/test_tactic_family_guard.py` (new), `docs/research/amendments.md` |
| What changed | Tactic families on gangs in Level 12d (12-8): (a) A gang may carry an optional family (`GangConfig.family`). The family is written into the plain form, and so into the run id, only when it is set, so gangs without a family keep their run ids. (b) Before the first gang world, the held-out lists in `config/tactic_families.yaml` and `config/evaluation.yaml` are checked to agree and not to overlap the training families. (c) A gang naming a family that is in neither list is refused. (d) A gang naming a held-out family is refused in training worlds, which also covers validation worlds because they use the training ranges (A-015). Test and stress worlds accept held-out families. (e) Known gaps, recorded now: no family-to-block table exists yet, so families are checked by name only; and the dividing line `training_devices_retired_on_schedule` is read but not enforced in code. |
| Why | The 12d brief requires the family mapping to be checked before the first gang world, and held-out families to appear only in test and stress worlds, so a model cannot be trained on a held-out tactic. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-020

| Field | Value |
| --- | --- |
| Date | 2026-10-23 |
| Issue IDs | None |
| Issue IDs v2 | L1, L2, A3, G6, X8 |
| Files | `config/labels.yaml` (new), `world/truth_records.py`, `world/scam_planner.py`, `world/truth_writer.py`, `world/label_records.py` (new), `world/complaints.py` (new), `world/label_builder.py` (new), `config/interactive_runner.yaml`, `docs/research/amendments.md` |
| What changed | Level 12e round E-1, complaints as label events. (a) Label events are built after a run from the finished result, the same way truth files are; the runner, `run_world`, `make_payment`, the events file and random children 0-16 are unchanged. (b) A label event carries only what a bank could know: arrival minute, the complaining account and its bank, the reported account and its bank, the payment ids, and the type the complainant claims. Whether a complaint is true is never written on it; evaluation finds it by matching `payment_id` against the truth's payment stories. (c) `PaymentStory` gains two optional fields, `scam_type` and `scam_step`, so the scam type no longer lives only in description text. (d) Per-world numbers (victim complaint share, false-complaint rate, merchant weight) are drawn once per world inside the split's range. Each delay is drawn per complaint inside the split's range, in whole minutes; day ranges are converted to minutes first. (e) A victim complains at most once, about every successful payment it made to its mule, measured from its last successful payment. A victim whose payments all failed lost nothing and does not complain. Payouts back are never complained about. (f) A false complaint is made by the payer of one successful genuine payment. A payment to a shop is `false_complaint_merchant_weight` times as likely to draw one as a payment to a person, and the expected total equals the configured rate per 100,000 genuine payments. A false complaint claims the remainder scam type and uses its delay range. (g) Ranges are read from the run's layered `world_ranges`, so stress layers such as `complaint-backlog` change label delays with no extra code (task 23). (h) Statistical checks use a minimum sample and a tolerance of a fixed number of standard errors, both in `config/labels.yaml`. (i) New random children: 17 `label_world_settings`, 18 `true_complaints`, 19 `false_complaints`. |
| Why | The 12e brief makes labels events with rates, delays and errors by scam type (L1), keeps the victim complaint share separate from bank reporting (L2), weights false complaints toward merchants (A3), and finishes G6's complaint-delay work. Keeping labels outside the runner protects the frozen anchors (ground rule 27). Keeping truth off the label event follows ground rules 15 and 29. Every wrong registry entry will later come from these false complaints and nowhere else (X8). No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-021

| Field | Value |
| --- | --- |
| Date | 2026-10-23 |
| Issue IDs | None |
| Issue IDs v2 | L2, L5, X8, 12-5, 12-12 |
| Files | `config/labels.yaml`, `docs/interfaces/national_systems.md`, `world/label_records.py`, `world/registry_reports.py` (new), `world/bulk_confirmations.py` (new), `world/label_builder.py`, `config/interactive_runner.yaml`, `showcase/fixtures/level_12/bulk_short_lag.yaml` (new), `docs/research/amendments.md` |
| What changed | Level 12e round E-2. (a) Until bank decisions exist (Levels 13 and 22), a bank treats an account as confirmed at the arrival minute of the first complaint that names it. (b) Each world draws one bank reporting share from `share_reported_to_shared_database`. Each confirmed account is reported with that chance, after a delay drawn per report from `registry_report_delay_minutes` (new, in `config/labels.yaml`). (c) The victim complaint share and the bank reporting share use separate random streams, so changing one never changes the other's outcomes (L2). (d) A wrong registry entry is a report about an account that holds no mule-side role at its confirmation minute. With (a), it can only come from a false complaint; no second error rate exists anywhere (X8). (e) Registry reports are world events in a written contract (`docs/interfaces/national_systems.md`, Level 12e section) that Level 13 ingests (12-5). (f) Bulk confirmations: each world draws the number of crackdowns as a Poisson count with mean events-per-year × days / 365. Each crackdown picks a cut-off minute, a region, a size and a delay. It confirms up to that many mule-side accounts in that region whose role began at or before the cut-off, and it arrives at cut-off + delay. Requested and confirmed counts are both recorded, never silently capped. Accounts opened mid-run have no region and are never swept; their number is reported. (g) Short worlds rarely contain a crackdown, so tests and the showcase use a labelled short-lag fixture layer (12-12). |
| Why | The 12e brief requires the two L2 parameters to vary independently, bulk-label events (L5), wrong registry entries only from 12e (X8), and a written registry-report contract (12-5). Police crackdowns are assumed to confirm only true mules; no source gives a crackdown error rate. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

## A-022

| Field | Value |
| --- | --- |
| Date | 2026-10-23 |
| Issue IDs | None |
| Issue IDs v2 | L4, 12-4 |
| Files | `config/analyst.yaml` (new), `world/analyst.py` (new), `tests/unit/world/test_analyst.py` (new), `docs/research/amendments.md` |
| What changed | Level 12e simulated analyst (L4, 12-4). (a) A pure function decides one case from the truth (is it a mule), evidence strength in [0, 1], queue load (open cases ÷ daily capacity), an optional model score in [0, 1], the world's analyst parameters, and two uniform draws supplied by the caller. It touches no state and no random generator. (b) Accuracy = base − weak-evidence drop × (1 − evidence strength) − overload drop × overload fraction. The overload fraction is 0 when load ≤ 1 and (load − 1) ÷ load above it. Accuracy is floored at `accuracy_floor` (0.5, policy: an analyst is no worse than a coin flip). The unfloored value is returned too, so the floor is visible. (c) With a model score present and the first draw below the follow share, the verdict is "mule" exactly when the score ≥ `score_threshold` (automation bias). Otherwise the verdict is correct when the second draw is below the accuracy. (d) Daily capacity = analysts on shift × shift minutes ÷ minutes per case (L4 "capacity and time per case"). (e) Not yet wired to any queue or run. Level 22 connects it, so 12-4 stays partial. |
| Why | The 12e brief asks for the analyst as a pure function with capacity and time per case, accepting a score now and wired in Level 22. Passing draws in keeps it pure and testable. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No. |

