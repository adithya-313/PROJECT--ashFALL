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

## A-023

| Field | Value |
| --- | --- |
| Date | 2026-10-24 |
| Issue IDs | G4 |
| Issue IDs v2 | 12-8, 12-10, 12-14, 12-18 |
| Files | `config/world_sampler.yaml`, `config/gang_mix.yaml`, `config/tactic_plugins.yaml`, `config/pattern_files.yaml`, `world/world_sampler.py`, `world/sampled_world.py`, `world/gang_mix.py`, `world/tactic_plugins.py`, `world/pattern_files.py`, `world/world_invariants.py`, `world/opening_handles.py`, `world/interactive_loop.py`, `world/cash_withdrawals.py`, `world/cash_out.py`, `world/interactive_runner.py`, `scripts/measure_world_block.py`, `docs/research/amendments.md` |
| What changed | Level 12f. (a) World counts per split, from measurement (12-18): training 320, validation 120, test 120, stress 120, each made of blocks of 40 worlds; every block is its own Latin hypercube with its own stream (`[sampler_seed, split_number, block_number]`) and its own seed run inside the split's seed block. Measured on the owner's laptop (Windows 11, 12 CPUs, Python 3.12, `small` preset, generated gang mix, sampler seed 1): a 61-day world yields about 5 true complaints (training mean 6.1, validation 4.9, test 7.0) and about 20 true mule accounts; 4 workers give about 3 times the serial speed with byte-identical worlds; the whole set takes about 1.9 hours on 4 workers. Validation is 120 because labels are scarce; test and stress are 120 so the spread across worlds and the per-setting sensitivity points rest on more than a handful of worlds. (b) Learning-curve rule for Level 18, written before any model exists: train on 1, 2, 4 and 8 training blocks, score each on the validation split only, and stop adding blocks when doubling them improves the validation score by less than 0.01 (absolute, on the pre-registered metric); 0.01 is a policy choice. (c) Test ranges: report Section 6 says test uses the training ranges and stress the wider ones; `config/world_ranges.yaml` (frozen since Level 11) also has a wider `test` column. Sampled worlds follow Section 6: test worlds read the training column and keep their split name, so the 12d guard still allows held-out families there; the runner refuses any split paired with a column other than training (training, validation, test) or stress (stress). No amendment explains the wider test column; the yaml calls its width the author's choice. It stays in the file, unused by sampled worlds. (d) Stratified parameters (12-14): world-level values pinned per world (number of regions, victim complaint share, false-complaint rate, false-complaint merchant weight, registry-report share) and gang-level centres per world (gang count 2 to 4, route layers, dwell minutes, devices per gang, victims per receiving mule), with each gang drawing its own value within 0.15 of the range width of the centre. Three Section 6 or market parameters are not stratified, because their readers load `world_ranges.yaml` from disk and would ignore a pin: `share_of_accounts_sold_to_several_gangs`, `mule_origin_share_business` (merchant-mule share) and `share_of_scams_investment_or_job` (which shapes complaint delay). Complaint delay itself stays a per-complaint draw. (e) Gangs in every split are built from training-range centres, because route dividing lines apply in every split; stress worlds therefore vary world settings, not gang shape. Held-out tactic families have no building blocks (A-017), so the generator makes training families only and never tags a gang held-out; which level builds held-out mechanisms is undecided (12-8). Every generated gang is `layered_chain` today, because the training layer range starts at 2. (f) Sampled stress worlds run long enough for their longest ranged event (120 days today, from `episode_length_days`), from a `horizon_limits` list in config. (g) Pattern files (task 7) and local plugins (task 8, 12-10): plugins are linted before import, confined to `patterns/plugins/`, receive only the gang's stream, and return only time and money, so no plugin can retire devices (12-8 dividing line). The lint is not a sandbox. Bundles (Level 30) must exclude plugins. (h) Three bugs in 12c and 12d code were found by the generated mix and fixed with regression tests: opening handles were not resolved in cash withdrawals and resales; withdrawals from accounts opened mid-run were refused before the run started; the crypto collector could open after its first payment. All sat on the path from gang-opened accounts to cash-out, which no fixture used. (i) A sweep of 78 generated worlds across all four splits, the account market and injected liens on gang-opened accounts gave no crash and no invariant problem (ledger checks, opening times, truth accounts, ATM daily limits, unique payment ids, label timing). The runner does not record gang observations, so "no new money after an observed lien" (G4) is checked only by the 12d unit test, not by the sweep. |
| Why | Counts sized by what each split must show, from measured labels and run time, not by a time budget, and committed before any model exists so they cannot be tuned on results. Section 6 is the design reference for splits and stratification; where frozen config or missing mechanisms stop us following it, the gap is written down here instead of hidden. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No model results. World statistics only (labels, mules, run time), measured to size the splits. |

## A-024

| Field | Value |
| --- | --- |
| Date | 2026-10-24 |
| Issue IDs | None |
| Issue IDs v2 | 12-18 |
| Files | `docs/research/amendments.md`, `showcase/level_12.py` |
| What changed | Correction to A-023 (i), which says "a sweep of 78 generated worlds". The correct count is 80 clean world runs: 70 distinct world plans (training block 0: 40; validation, test and stress: 10 each, stress counted from its rerun after the horizon fix) plus 10 variant runs of the first 5 training worlds (5 with the account market and gang reactions on, 5 with injected liens). None crashed and none had an invariant problem. The counts per split and every other statement in A-023 are unchanged. A-023 is left as written, because amendments are append-only. |
| Why | Ground rule 1: a wrong number is corrected openly, not silently. The coding agent flagged the mismatch. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No model results. |

## A-025

| Field | Value |
| --- | --- |
| Date | 2026-10-25 |
| Issue IDs | None |
| Issue IDs v2 | 12-8 |
| Files | `config/evaluation.yaml`, `config/tactic_families.yaml`, `config/tactic_blocks.yaml`, `config/gang_mix.yaml`, `world/tactic_families.py`, `world/held_out_options.py`, `world/gang_mix.py` |
| What changed | Level 12g. (a) Active held-out families: `slow_drip` (route options `drip_piece_max_inr`, `drip_spread_days`, `drip_hold_minutes`: pieces below Rs 2,000, each resting more than 60 minutes at every mule, the first hop spread over days) and `device_rotation` (take_control options `rotate_every_days`, `rotations`: one fresh device per mule, replaced on a schedule, so no device controls two accounts). Both are allowed only in test and stress worlds, keyed on the world's split; a world with no split counts as training. (b) Deferred: `fake_routine_life` needs a gang payer funding path, a gang-opened-only target origin and new truth fields, and building it would change how money enters every world, training worlds included; `pseudo_merchant_camouflage` needs convincing merchant traffic and costs the most to build. The Level 12h edge case "a mule imitating a salaried employee" needs the same gang-funded salary credits and is deferred with `fake_routine_life`. (c) What is lost: no held-out family tests the routine-life signal, the signal most likely to make worlds too easy. Until it returns, the defence is Level 12h's trivial-world check, run on worlds that contain rented mules (G1), whose routine life is real. (d) Return conditions: `fake_routine_life` returns when a gang payer funding path exists that keeps gang money separate from fixture money (not by reusing fixture-source registration); `pseudo_merchant_camouflage` returns when convincing merchant traffic exists; each is picked up in Level 29 (failure finder) or Tier 2, whichever comes first. (e) `config/evaluation.yaml` `held_out_tactics.tactics` now lists the two active families; a new `deferred` list holds the other two, with reason, return condition and this amendment. This changes the pre-registered list. (f) Held-out results are reported per family, never pooled. (g) The validator replaces A-017's name-only check: every active name must have at least one implementing block option, every option's family must be active, a deferred name must have no option and must cite an existing amendment, and the status of every name is printed (`python -m world.tactic_families`). (h) The generator gives every test and stress world at least one seen gang and one held-out gang, and never gives one to a training or validation world. (i) The held-out families were designed by the project author, so they test whether the model generalises, not whether the author does. |
| Why | Ground rule 17 and the Level 12g brief allow deferring `fake_routine_life` first, and require at least two real families before Level 12h; two now exist and attack different signals (timing and flow; device control). A bug in the money-entry path would corrupt every world's ledger, training worlds included, for one extra family. Writing down what is lost and what brings it back keeps the deferral honest (ground rule 1). No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No model results. |

## A-026

| Field | Value |
| --- | --- |
| Date | 2026-10-25 |
| Issue IDs | None |
| Issue IDs v2 | 12-6 |
| Files | `config/evaluation.yaml`, `evaluation/trivial_world.py` |
| What changed | Level 12h. Partly replaces A-013 (where it says the check runs on validation worlds only). (a) The trivial-world check GATES on every training and validation world: a world whose best single signal reaches AUPRC 0.90 or more fails, and the response is to fix the world, never the threshold. (b) It also runs on every test and stress world as REPORT-ONLY: results are recorded, with held-out gangs reported per family, and are never used to tune the world or the threshold. Looking at test worlds is allowed; changing anything because of what was seen there is not. (c) If a report-only run shows a family or signal at or above 0.90, the response is fixed in advance: report the family, the signal and the separation; any fix to a mechanism must be justified without the test result (for example "the mechanism was unrealistic"), and that reason is written in a new amendment; after a fix, test and stress worlds are regenerated from fresh seed ranges, so the worlds that were looked at are never the worlds that are scored. This route is open only because no model exists yet; it closes when Level 18 starts. (d) The six candidate signals are fixed before any result: routine life, pass-through, dwell, devices per account, payer count and payer regions, each computed per account from world events in the evaluation package, scored in both directions. Mules are accounts holding a mule role in the truth; genuine accounts hold no gang role at any time (victims included). |
| Why | Rule 17 forbids tuning on test and stress worlds, not looking at them. Recording the response in advance stops a report-only result from quietly turning into tuning. The signals and the scoring are fixed before any result exists. No model code exists yet for the new engine (models start in Level 18); `models/` and `scripts/fraud_*` hold only v1 PaySim work, excluded by `earlier_work_note`. |
| Results exist? | No model results. |

## A-027

| Field | Value |
| --- | --- |
| Date | 2026-10-25 |
| Issue IDs | None |
| Issue IDs v2 | 13-11, 13-12, 13-14, 13-15 |
| Files | `docs/interfaces/national_systems.md`, `docs/interfaces/time.md`, `config/nationalsim.yaml`, `docs/research/amendments.md` |
| What changed | Level 13a. (a) National mock systems contracts updated in Revision 2: all currency amounts specified in integer paise; hold requests carry complaint_id and payment_id; timestamps follow docs/interfaces/time.md (IST aware, Ground Rule 30); identifier hashes use HMAC-SHA256 keyed per run. (b) nationalsim listing delay: `listing_delay_minutes` in `config/nationalsim.yaml` represents the registry's own internal processing time between ingestion receipt and visibility to screening, separate from Level 12's `registry_report_delay_minutes` (which delays report arrival from bank confirmation). (c) `registry_entry_delay_days` (N2 in `config/world_ranges.yaml`) is not used by nationalsim, because using it would delay the same gap twice. It stays as a parameter in `config/world_ranges.yaml` and the reconciliation is open. (d) DPIP Phase-2 transaction scoring is declared out of scope and is not built. |
| Why | Establishes explicit paise and event-time standards for national systems mocks, distinguishes bank reporting delay from registry listing delay, avoids double-counting delay across world ranges and nationalsim, and clarifies that Phase-2 scores are not implemented. |
| Results exist? | No model results. |

## A-028

| Field | Value |
| --- | --- |
| Date | 2026-10-26 |
| Issue IDs | None |
| Issue IDs v2 | N2, 13-5 |
| Files | `evaluation/registry_coverage.py` (new), `tests/unit/evaluation/test_registry_coverage.py` (new), `docs/research/amendments.md` |
| What changed | Level 13b registry coverage evaluation. Coverage is emergent from Level 12 reports: for each mule depth (1, 2, 3+), reported mules / all mules is computed directly from the world's registry reports and truth roles. The configured ranges in `config/world_ranges.yaml` (`registry_coverage_layer1_mule_share` and `registry_coverage_deeper_mule_share`) serve as reference ranges beside the measured values rather than an active filter or sampling mask. Pass rule requires layer-1 coverage > deeper coverage. |
| Why | Using coverage parameters as reference ranges avoids applying a second error rate or artificial filtering on top of Level 12's emergent victim complaint and bank reporting dynamics (X8). |
| Results exist? | No model results. Measured coverage values across synthetic world runs only. |

## A-029

| Field | Value |
| --- | --- |
| Date | 2026-10-26 |
| Issue IDs | None |
| Issue IDs v2 | 13-2, 13-3, 13-16, 13-17 |
| Files | `config/nationalsim.yaml`, `nationalsim/hold_flow.py` (new), `world/hold_replay.py` (new), `docs/research/amendments.md` |
| What changed | Level 13c two-pass hold replay and 1930 administrative hold flow. (a) Two-pass replay: Pass 1 runs the fraud-on world normally and generates victim complaints via `world/label_builder.py`. Each complaint produces at most one hold request keyed by `complaint_id`. Pass 2 re-runs the exact same seed with administrative holds injected as `OutsideEvent(kind="lien")` at their computed landing minutes. (b) Complaints are derived exclusively from Pass 1; complaints are not recomputed after Pass 2. (c) Reinterpretation of `hold_missed_share`: while the parameter text in `config/world_ranges.yaml` describes `hold_missed_share` as "too late because money left", Level 13c separates computed balance depletion from bank operational failure. `too_late` is COMPUTED from reconstructed pass-2 account balances at the landing minute (`balance <= 0`), and `hold_missed_share` is used strictly for "the bank never acts" (administrative inaction where no lien is placed). (d) Outcome classification: evaluated from pass-2 state at the landing minute into four exhaustive outcomes: `applied` (lien placed and full requested funds encumbered, `balance >= amount`), `partial` (lien placed and partial funds encumbered, `0 < balance < amount`), `too_late` (money already left the account before lien landed, `balance <= 0`), and `missed` (bank never acted, no lien placed). (e) Naive one-hop forwarding: receiving banks inspect only their own ledgers. For funds that departed the held account prior to hold landing, the bank issues exactly one new hold request to the institution that received that payment. Forwarding stops at exactly one hop (never transitive). Forwarded holds receive their own latency and evaluated pass-2 outcomes, but are not placed as liens in pass 2 (which would require a third pass). (f) Time-of-day and weekend latency multipliers: `config/nationalsim.yaml` defines `night_hours` ([22, 6] IST), `night_multiplier` (2.0), and `weekend_multiplier` (1.5) under label `assumption` and `derived_from: policy`, scaling latency alongside bank `hold_delay_factor` and adding `unreachable_1930_extra_delay_minutes` when unreachable. |
| Why | Accurately models the operational 1930 national cybercrime reporting hotline and inter-bank administrative hold flow while preserving deterministic two-pass world replay, cleanly separating computed balance depletion (`too_late`) from bank inaction (`missed`), and establishing a transparent baseline for single-hop payment tracebacks without unconstrained multi-hop lien recursion. |
| Results exist? | No model results. |

## A-030

| Field | Value |
| --- | --- |
| Date | 2026-10-27 |
| Issue IDs | None |
| Issue IDs v2 | 13-14, 14-13 |
| Files | `migrations/0005_create_national_outbox.sql` (new), `tests/integration/test_national_outbox_table.py` (new), `tests/integration/test_migration_runner.py`, `config/issues_v2.yaml`, `docs/research/amendments.md` |
| What changed | Level 13c national outbox table and migration sequencing. (a) Dedicated national outbox table: `national_outbox` is created via `migrations/0005_create_national_outbox.sql` for registry events and 1930 hold requests. The existing Level 7 `outbox` table (`migrations/0001`) has `transaction_id UUID NOT NULL REFERENCES transactions(transaction_id)` and a schema envelope locked at version "1", making it structurally impossible to use for national events that have no associated transaction. (b) Reassignment of migration 0005: Migration 0005 was previously reserved in the plan for Level 14 UPI event-time columns. Under Architect Decision, 0005 is reassigned to `national_outbox`, and Level 14's UPI columns are moved to migration 0006. References in `config/issues_v2.yaml` are updated accordingly. (c) Redpanda / Kafka ACL limitation: Redpanda in local development has no ACLs configured. Bank-level message segregation ("a bank cannot consume another bank's messages") is enforced at the bank consumer layer by verifying that the message key equals `<run_id>:<bank_id>` and counting/logging foreign messages without reading their payloads. In a production deployment, broker-level ACLs (e.g. Kafka SASL/SCRAM or mTLS per-principal topic/partition ACLs) would be required to enforce transport-level isolation. |
| Why | Prevents invalid foreign key constraints on non-transactional national events, preserves Level 7 outbox table immutability, reserves clean migration sequencing for UPI columns, and documents development vs production security boundaries for multi-tenant bank messaging. |
| Results exist? | No model results. |

## A-031

| Field | Value |
| --- | --- |
| Date | 2026-10-27 |
| Issue IDs | None |
| Issue IDs v2 | 14-2, 14-8, 14-9, 14-11, 14-12, 14-13, 14-14, 14-15, K3 |
| Files | `migrations/0006_add_upi_columns_and_bank_isolation.sql` (new), `api/v1/bank_scope.py` (new), `api/v1/bank_keys.py` (new), `api/v1/upi_outbox.py` (new), `config/bank_api.yaml` (new), `api/services/outbox_relay.py`, `core/config.py`, `config/issues_v2.yaml`, `docs/research/amendments.md` |
| What changed | Level 14a schema migration, bank isolation, and UPI outbox. (a) 0006 not 0005: Following A-030 which assigned 0005 to `national_outbox`, Level 14 UPI columns, constraints, bank_api_keys table, and RLS policies are applied in `migrations/0006_add_upi_columns_and_bank_isolation.sql`. (b) Non-superuser role and RLS bypass: PostgreSQL superusers bypass Row Level Security policies by engine design. Application and test pool connections connect as the superuser (`postgres`), so tenant isolation is enforced by switching to a non-superuser role (`ashfall_bank_api`) via `SET LOCAL ROLE` and setting `app.bank_id` transaction-locally via `set_config('app.bank_id', bank_id, true)`. (c) Outbox schema "2" and topic routing: UPI payment events use envelope schema_version "2" (`upi.payment.recorded`) and are routed to `config.kafka_upi_topic` (`upi.payments`). The Level 7 consumer (`api.services.idempotent_consumer.parse_event`) strictly asserts `schema_version == "1"` and crashes on any other version, requiring dedicated topic separation for schema "2". (d) UPI idempotency via `(bank_id, payment_id, side)`: Each side of a UPI transfer is uniquely keyed in `transactions` by `uq_transactions_bank_payment_side`. Replayed payment sides return `was_replay=True` with the existing `transaction_id` and do not emit duplicate outbox events. Event IDs are derived deterministically using UUIDv5 with a permanent dedicated namespace. (e) Outbox least-privilege isolation: Outbox rows are not bank-isolated and lack a `bank_id` column. The `ashfall_bank_api` role is granted `INSERT` on `outbox` to stage events alongside transactions in atomic blocks, but is explicitly denied `SELECT` on `outbox` so that commercial banks can never read queued outbox events staged by other banks. |
| Why | Establishes PostgreSQL row-level security that cannot be bypassed on pooled connections, ensures clean backward compatibility with locked Level 7 consumer schemas, guarantees idempotency for cross-bank transfers, and enforces least-privilege security boundaries across tenant banks. |
| Results exist? | No model results. Integration test and unit test evidence only. |

## A-032

| Field | Value |
| --- | --- |
| Date | 2026-10-28 |
| Issue IDs | None |
| Issue IDs v2 | 14-4, 14-9, 14-11, 14-12, 14-16, X3 |
| Files | `api/v1/request_auth.py`, `api/v1/bank_router.py`, `api/v1/signing.py`, `api/v1/webhooks.py`, `config/bank_api.yaml`, `requirements-test.txt`, `tests/contract/test_bank_api_fuzz.py`, `tests/integration/test_bank_endpoints.py`, `tests/integration/test_bank_idempotency.py`, `tests/integration/test_bank_request_auth.py`, `showcase/level_14.py`, `config/issues_v2.yaml`, `docs/research/amendments.md`, `Dockerfile`, `showcase/level08_support.py`, `api/v1/upi_outbox.py`, `tests/unit/test_api_does_not_import_world.py` |
| What changed | Level 14b bank REST API endpoints, raw-path HMAC signing, fuzzing, CI coverage, and showcase part b. (a) Authentication before body parsing: Bank authentication runs before request body deserialization. Requests with missing or invalid credentials receive a structured 401 response (`missing_auth_headers`, `invalid_key`, `bad_signature`, `stale_signature`, `nonce_reused`), never 422 `unprocessable_entity`. (b) Structured error contract: All API errors conform to RFC 7807 intent returning JSON with `{"error": {"code": "...", "message": "..."}}`. (c) Signing secret separate from API key: The HMAC signing secret (`bank_signing_secret_<bank_id>`) is distinct from the hashed bearer API key stored in PostgreSQL (`bank_api_keys`). Compromise of a stored key hash cannot forge cryptographic request signatures. (d) Raw request path signing: Signing and signature verification canonicalize against the ASGI `scope["raw_path"]` bytes, preserving raw percent-encoded characters (such as `%20`) so that URL-encoding differences do not create signature bypass or verification mismatch. (e) Webhook delivery deferred to 14c: Transactional outbox records events for delivery, but the background webhook relay loop is deferred to Level 14c when full event coverage across simulated commercial banks is operational. (f) Schemathesis pin: Schemathesis is pinned to version 4.32.7 in `requirements-test.txt` for deterministic and reproducible property-based API contract fuzzing. (g) Level 13 wire-key tests: Fixed a test field mismatch in Level 13 wire-key tests where `target_account_id` was asserted instead of `account_id`. (h) CI coverage: `.github/workflows/ci.yml` is updated to run the full suite of Level 14 integration and contract fuzz tests against real PostgreSQL in the `integration-tests` job. (i) Default bank mode: Commercial bank API mode defaults to "shadow" in `config/bank_api.yaml` until Level 14c flips it to active decision routing. (j) CI and container fix-ups: The API keeps its own failure-domain list with a guard test (`tests/unit/test_api_does_not_import_world.py`) so it no longer imports the world simulator; the API image (`Dockerfile`) and the Level 8 baseline copy (`showcase/level08_support.py`) now include `config/bank_api.yaml`; and the fuzz test (`tests/contract/test_bank_api_fuzz.py`) is marked `needs_stack`. |
| Why | Guarantees secure, authenticated, and idempotent REST interfaces for commercial bank participants, enforces authentication prior to potentially hostile payload parsing, preserves URL encoding integrity across proxy hops, eliminates unhandled 5xx exceptions under adversarial and randomized inputs, ensures comprehensive CI verification, prevents container boot crashes from missing configuration or simulator dependencies, enables baseline regression tests to locate bank configuration in isolated test workspaces, and prevents fast non-docker CI suites from failing on skipped database fuzz tests. |
| Results exist? | No model results. Integration tests, property-based fuzz tests, and showcase verification only. |

## A-033

| Field | Value |
| --- | --- |
| Date | 2026-10-29 |
| Issue IDs | None |
| Issue IDs v2 | 14-6, 14-8, 14-10, R7, K2, X3 |
| Files | `world/bank_feed.py`, `world/bank_feed_config.py`, `world/bank_connectors.py`, `api/v1/bank_service.py`, `api/v1/bank_clients.py`, `config/bank_feed.yaml`, `config/bank_api.yaml`, `migrations/0008_add_sender_device_columns.sql`, `tests/unit/world/test_bank_feed.py`, `tests/unit/world/test_bank_feed_config.py`, `tests/unit/world/test_bank_connectors.py`, `tests/integration/test_bank_clients.py`, `tests/integration/test_bank_connectors_end_to_end.py`, `showcase/level_14.py`, `showcase/level_14_support_c.py`, `config/issues_v2.yaml`, `tests/unit/test_issue_register_v2.py`, `docs/research/amendments.md` |
| What changed | Level 14c bank-feed projection, mock bank connectors, shadow and live modes, and showcase part c. (a) Bank-feed projection and allow-lists: `world/bank_feed.py` projects simulation output into bank-visible payment sides, account lifecycle and device events, and fraud complaints, strictly validating every record against field allow-lists (`BANK_VISIBLE_PAYMENT_FIELDS`, `BANK_VISIBLE_ACCOUNT_EVENT_FIELDS`, `BANK_VISIBLE_COMPLAINT_FIELDS`, `GENUINE_DEVICE_DETAIL_FIELDS`, `GANG_DEVICE_DETAIL_FIELDS`) and raising `ProjectionLeakError` on unexpected fields (such as `purpose`). (b) R7 failure domain visibility: Failed payments are partitioned strictly by failure domain. Sending banks record all failed attempts (`side='debit'`), whereas receiving banks record failed payments (`side='credit'`) only when the failure domain is `receiver_bank`, ensuring receiver visibility preserves real-world banking boundaries. (c) Assumptions on missing simulator channels and VPAs: Simulator world records lack channel and VPA fields; `channel_value` defaults to `"upi_app"` as an explicit config assumption in `config/bank_feed.yaml`, and `counterparty_vpa` is left empty `""`. (d) Migration 0008 for debit-only sender device columns: Adds optional `sender_device_fingerprint` and `sender_os_generation` columns to `transactions` in `migrations/0008_add_sender_device_columns.sql`. These fields are populated exclusively on debit sides. (e) Per-bank operating modes: Replaces scalar `default_bank_mode` in `config/bank_api.yaml` with a per-bank `bank_modes` mapping (`bank_0`, `bank_1`, `bank_2` configured individually as `"shadow"` or `"live"`). (f) Live mode allow-only behavior until Level 19 (14-6): Because scoring models are not trained or enabled until Level 19 and decision endpoints return `"unscored"`, live mode evaluates incoming responses and falls back to `"allow"`, recording fallback counts (`live_fallbacks`) rather than blocking legitimate simulation flow. (g) Deterministic idempotency keys: Hook decision requests are idempotently keyed by the planned payment ID (`make_decision_idempotency_key`) because the synchronous simulator hook only observes the planned payment ID. (h) Hybrid dispatch architecture: Hook decision requests are dispatched per planned payment during simulation via `asyncio.run_coroutine_threadsafe`, while transaction sides, account events, and complaints are projected and dispatched after simulation completes (`send_feed`), respecting the plan's fallback while guaranteeing causality via global multi-key sorting `(event_time, bank_id, record_kind, original_index)`. (i) Gang device event filtering: Gang planner device event kinds that lack an equivalent commercial bank API lifecycle kind are dropped and counted in `dropped_device_events_by_kind`. (j) In-process and HTTP transport parity: `InProcessBankClient` (direct database pool) and `SignedHttpBankClient` (over `httpx.ASGITransport` using HMAC-SHA256 request signing) implement the `BankTransport` protocol and produce identical stored rows. (k) Webhook delivery deferred to Level 19: Background webhook delivery is deferred to Level 19 because decisions remain unscored throughout Level 14. |
| Why | Preserves tenant isolation and data minimization boundaries between the internal simulation engine and commercial banks, prevents information leaks of simulation ground truth, guarantees deterministic reproducibility between in-process and wire transports, enforces strict causality in record arrival ordering across commercial banks, and establishes verified foundations for Level 15 features and Level 19 ML scoring. |
| Results exist? | No model results. Unit tests, integration tests, and showcase verification only. |




## A-034

| Field | Value |
| --- | --- |
| Date | 2026-11-01 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `config/evaluation.yaml`, `benchmarks/decision_latency.py`, `tests/unit/test_decision_latency.py`, `docs/research/amendments.md` |
| What changed | Latency scope: what the two limits in the `latency` block cover. (a) The registry wait counts only against `max_ms` (100 ms, the whole decision). (b) `feature_computation_max_ms` (50 ms) covers our own feature code only: Redis reads, the graph search over payment neighbours, and arithmetic. The registry wait is not counted against it. (c) Fit check: registry wait + feature computation + all other overhead (request handling, the feature_log write, the response) must fit within 100 ms. (d) Recorded as the new `latency_scope` block in `config/evaluation.yaml`; the pre-registered `latency` block is unchanged. |
| Why | The registry is an outside service. Its wait is limited by its own per-call timeout, not by the 50 ms limit on our own code, which we can speed up. This was decided as a principle before the judged measurement (config commit f9b111d), not because a number failed. |
| Results exist? | No results. A principle recorded before the judged measurement. |

## A-035

| Field | Value |
| --- | --- |
| Date | 2026-11-01 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `config/evaluation.yaml`, `benchmarks/decision_latency.py`, `showcase/level_15_support_c.py`, `docker-compose.yml`, `Makefile`, `tests/unit/test_decision_latency.py`, `docs/research/amendments.md` |
| What changed | Latency method. (a) 100 warm-up calls (discarded), then 3 runs of 1,000 measured calls; each run gives one p99 for the decision and one for the feature computation; the verdict uses the MEDIAN of the three p99 values, and all three and their spread are reported. (b) Every decision is split into registry wait, feature compute, feature_log write and the rest, measured with `time.perf_counter()` inside the process that runs the decisions; for the 10 slowest calls of each run the report says which part took the time. (c) The budget is judged where the service runs: inside the project's Linux API container on the compose network (profile `benchmark`: services `bench` and `nationalsim-bench`); the native Windows run is printed but not judged. (d) Recorded as the new `latency_method` block in `config/evaluation.yaml` (commit f9b111d, before the judged measurement). Judged result, HTTP mock registry with the default added delay, commit 7a1fc0c, 2026-11-01, owner's laptop: decision p99 per run 49.68, 51.18 and 47.40 ms, median 49.7 ms (limit 100); feature compute p99 per run 20.90, 23.13 and 20.94 ms, median 20.9 ms (limit 50); registry wait p99 median 18.6 ms; 0 registry timeouts of 9,000 questions. Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.41, Python 3.11.16, monotonic clock `clock_gettime(CLOCK_MONOTONIC)` with resolution 1 ns. The native Windows clock (`GetTickCount64()`, resolution 15.625 ms, Python 3.12.10) is why the native run is not the verdict. |
| Why | A 300-sample p99 is decided by about 3 samples, so one noisy moment can flip pass and fail. The method was fixed before measuring, so no run could be chosen afterwards. |
| Results exist? | Yes, latency measurements only (no model results). Showcase Level 15 part c, check 15c-3, in full mode judges the container run. |

## A-036

| Field | Value |
| --- | --- |
| Date | 2026-11-01 |
| Issue IDs | None |
| Issue IDs v2 | 15-3, 15-6, 15-7, 15-8 |
| Files | `.github/workflows/ci.yml`, `.gitignore`, `Dockerfile`, `Makefile`, `api/main.py`, `api/v1/bank_router.py`, `api/v1/bank_service.py`, `benchmarks/decision_latency.py`, `benchmarks/profile_decision_features.py`, `config/account_features.yaml`, `config/evaluation.yaml`, `config/nationalsim.yaml`, `core/config.py`, `docker-compose.yml`, `docs/interfaces/account_state_contract.md`, `features/account_keys.py`, `features/account_labels.py`, `features/account_reader.py`, `features/account_records.py`, `features/account_snapshots.py`, `features/account_stream.py`, `features/account_summaries.py`, `features/decision_features.py`, `features/decision_reads.py`, `features/feature_log.py`, `features/feature_replay.py`, `features/feature_set.py`, `features/ingest_watermark.py`, `features/m1_features.py`, `features/m1_intelligence.py`, `features/m1_network.py`, `features/m1_payment.py`, `features/registry_prefetch.py`, `features/registry_screening.py`, `migrations/0010_create_feature_log.sql`, `migrations/0011_reshape_feature_log.sql`, `migrations/0012_add_feature_log_provenance.sql`, `migrations/0013_allow_null_cutoff_on_error.sql`, `migrations/0014_create_account_snapshots.sql`, `nationalsim/app.py`, `nationalsim/bank_client.py`, `requirements.txt`, `scripts/feature_log_size_estimate.py`, `showcase/level08_support.py`, `showcase/level_15.py`, `showcase/level_15_support_c.py`, `tests/baselines/compose_health_baseline.json`, `tests/integration/test_account_snapshots_0014.py`, `tests/integration/test_decision_feature_logging.py`, `tests/integration/test_feature_log_0010.py`, `tests/integration/test_migration_runner.py`, `tests/integration/test_network_label_cutoff.py`, `tests/integration/test_recompute_from_history.py`, `tests/integration/test_registry_pool.py`, `tests/integration/test_registry_prefetch_http.py`, `tests/unit/nationalsim/test_bank_client_tls.py`, `tests/unit/test_account_labels.py`, `tests/unit/test_account_snapshots.py`, `tests/unit/test_account_stream.py`, `tests/unit/test_account_withdrawals_state.py`, `tests/unit/test_decision_features.py`, `tests/unit/test_decision_latency.py`, `tests/unit/test_decision_reads.py`, `tests/unit/test_feature_log.py`, `tests/unit/test_feature_replay.py`, `tests/unit/test_ingest_watermark.py`, `tests/unit/test_m1_intelligence.py`, `tests/unit/test_m1_payment.py`, `tests/unit/test_network_batching.py`, `tests/unit/test_profile_decision_features.py`, `tests/unit/test_registry_prefetch.py`, `tests/unit/test_replay_bad_ids.py`, `docs/research/amendments.md` |
| What changed | Level 15c: decision-time feature logging, recompute, snapshots and decision latency (pull requests #93 to #99). (a) Ingestion watermark: each bank's arrival cutoff is the minimum over its payment stream and its four relational pollers, so a decision records exactly which data it could see. (b) `feature_log` and `account_snapshots` (migrations 0010 to 0014): daily partitions, row-level security by bank, canonical 52-feature arrays, provenance, a null cutoff on error rows. (c) Shadow decision logging behind `decision_feature_logging_enabled`: `record_decision` computes and logs the features and never changes the bank's answer. (d) Recompute from history: a logged row is rebuilt from Postgres history in arrival order into a scratch Redis and must match exactly. (e) Snapshots: weekly full snapshots and activity-based writes, with a size estimate. (f) Latency work: configuration YAML read through the shared content cache; batched account and counterparty reads and a whole-frontier network search; payer snapshot and label lists read once per decision; registry questions asked together with a per-call timeout (never turned into "no match"); one pooled registry connection with the TLS context built once; a timeout that is only "unknown" when the call really took longer, because asyncio timers can fire up to two clock ticks early on Windows; the registry timeout derived by formula (33.19 ms, `config/account_features.yaml`); mock registry delay in a default range (2 to 15 ms) and a stress range (15 to 60 ms). (g) Docker Compose profile `benchmark` (services `bench` and `nationalsim-bench`) so the latency budget is judged in the Linux container (A-035). (h) Effect on the API: when `decision_feature_logging_enabled` is on and a DPIP participant key is configured, the API now starts one shared registry connection pool and screens payers, devices and payees against the registry; without a key the registry features stay unknown, as before. Levels 15a and 15b have no entries in this file; they are not recorded here. |
| Why | Level 15c turns the live account state of 15a and 15b into logged, replayable and measurable decision-time features. The logged vector, not a recomputation at training time, must be what the bank's decision saw (Section 22), late events must never rewrite it, and the whole path has to fit the pre-registered latency limits (A-034, A-035). |
| Results exist? | No model results. Unit tests, integration tests, showcase verification and latency measurements only. |

## A-037

| Field | Value |
| --- | --- |
| Date | 2026-11-01 |
| Issue IDs | None |
| Issue IDs v2 | 15-1, 15-7, 15-8 |
| Files | `.github/workflows/ci.yml`, `config/account_features.yaml`, `docs/interfaces/account_state_contract.md`, `features/account_keys.py`, `features/account_reader.py`, `features/account_records.py`, `features/account_stream.py`, `features/account_updater.py`, `requirements-test.txt`, `showcase/level_15.py`, `showcase/preconditions.py`, `tests/integration/test_account_stream_end_to_end.py`, `tests/unit/test_account_keys_and_records.py`, `tests/unit/test_account_stream.py`, `tests/unit/test_account_updater.py`, `docs/research/amendments.md` |
| What changed | Written 2026-11-01, after the work; covers Level 15a merged 2026-10-29 (PR #84). Level 15a live account state (pull request #84, written into this log afterwards from git and code). (a) Account feature settings and namespaced Redis keys per run, bank and account, with record parsing (`config/account_features.yaml`, `features/account_keys.py`, `features/account_records.py`). (b) An idempotent updater and a point-in-time reader: applying the same record twice gives identical state, and a snapshot only shows records whose event time is inside the 7-day window and whose arrival time is at or before the arrival cutoff (`features/account_updater.py`, `features/account_reader.py`). (c) Streaming: payment sides are read from the Kafka topic `upi.payments` and account events from Postgres, with an overlap re-read so rows that commit late are not skipped (`features/account_stream.py`). (d) Atomic trimming through a Lua script with a cutoff of newest event time minus (retention + allowed lateness), and a key expiry of retention + lateness + margin (the file cites Level 15 architect decision 3). (e) The state contract written down in `docs/interfaces/account_state_contract.md`. (f) Showcase part a (`showcase/level_15.py`) and CI starting Redis for it. |
| Why | Level 15 needs bank-side account state that can be read as it stood at a past decision time, never shows a bank anything that had not arrived by then, and gives the same state when the same events are replayed or arrive twice. |
| Results exist? | No model results. Unit tests, integration tests and showcase verification only. |

## A-038

| Field | Value |
| --- | --- |
| Date | 2026-11-01 |
| Issue IDs | E3, R5, R8, G2 |
| Issue IDs v2 | 15-2, 15-5, 15-6, 15-9, 15-10, 15-11, 15-12 |
| Files | `.env.example`, `api/v1/bank_clients.py`, `api/v1/bank_router.py`, `api/v1/bank_service.py`, `api/v1/models.py`, `api/v1/upi_outbox.py`, `config/account_features.yaml`, `config/bank_feed.yaml`, `config/interactive_runner.yaml`, `config/world_ranges.yaml`, `core/config.py`, `data/schemas/feature_set_m1.yaml`, `docs/interfaces/account_state_contract.md`, `features/account_keys.py`, `features/account_labels.py`, `features/account_reader.py`, `features/account_records.py`, `features/account_stream.py`, `features/account_summaries.py`, `features/account_updater.py`, `features/feature_set.py`, `features/m1_balance.py`, `features/m1_control_region.py`, `features/m1_features.py`, `features/m1_inflow_outflow.py`, `features/m1_intelligence.py`, `features/m1_network.py`, `features/m1_payment.py`, `features/m1_routine_lifecycle.py`, `features/registry_screening.py`, `migrations/0009_add_balance_withdrawals_to_bank_feed.sql`, `nationalsim/app.py`, `nationalsim/bank_client.py`, `nationalsim/registry.py`, `showcase/level_15.py`, `showcase/level_15_support_b.py`, `tests/baselines/api_openapi_snapshot.json`, `tests/integration/test_account_labels_end_to_end.py`, `tests/integration/test_account_summaries_end_to_end.py`, `tests/integration/test_bank_feed_additions_end_to_end.py`, `tests/integration/test_migration_runner.py`, `tests/integration/test_registry_as_of_end_to_end.py`, `tests/unit/test_account_keys_and_records.py`, `tests/unit/test_account_labels.py`, `tests/unit/test_account_lifetime_indexes.py`, `tests/unit/test_account_summaries.py`, `tests/unit/test_account_withdrawals_state.py`, `tests/unit/test_feature_set_m1.py`, `tests/unit/test_m1_balance.py`, `tests/unit/test_m1_control_region.py`, `tests/unit/test_m1_inflow_outflow.py`, `tests/unit/test_m1_intelligence.py`, `tests/unit/test_m1_network.py`, `tests/unit/test_m1_payment.py`, `tests/unit/test_m1_routine_lifecycle.py`, `tests/unit/test_registry_as_of.py`, `tests/unit/world/test_bank_connectors.py`, `tests/unit/world/test_bank_feed.py`, `tests/unit/world/test_bank_feed_additions.py`, `tests/unit/world/test_m1_on_world.py`, `tests/unit/world/test_opening_dates.py`, `tests/unit/world/test_scam_gangs.py`, `tests/unit/world/test_tactic_family_guard.py`, `world/bank_connectors.py`, `world/bank_feed.py`, `world/bank_feed_config.py`, `world/interactive_records.py`, `world/interactive_runner.py`, `world/opening_dates.py`, `docs/research/amendments.md` |
| What changed | Written 2026-11-01, after the work; covers Level 15b merged through 2026-10-31 (PRs #85-#91). Level 15b M1 features (pull requests #85 to #91, written into this log afterwards from git and code). (a) World and bank feed: opening dates drawn for accounts that exist at the start (`account_opening_age_days` in `config/world_ranges.yaml`), balances on payment sides, a withdrawals endpoint and `bank_withdrawals` table (migration 0009), and home regions in the bank feed. (b) The canonical M1 feature list (`data/schemas/feature_set_m1.yaml`, 52 features) with a loader that rejects drift. (c) Long-lived account summaries with closed-day reads and a 62-day horizon (`summary_horizon_days`), with lifetime keys that do not expire. (d) Feature groups: inflow and outflow, balance pattern, routine life, lifecycle, control (device, with three R5 parts, hardware change, SIM swap and app reinstall on the same phone, plus the composite `new_device_r5`) and region (same-bank payers only). (e) Labels and network: confirmed labels come from `confirmed_fraud` feedback on the receiving (credit-side) account, complaints are kept per bank, and network features are computed over this bank's own payment graph (the code cites architect decision 9b). (f) Payment-level features, and registry intelligence features in which an unknown answer stays distinct from no match (the code cites architect rules 1 to 3 and decision 9c); the mock registry can answer as of a past time for replay. (g) Showcase part b. |
| Why | Level 15b turns the account state of 15a into the M1 feature catalog while keeping what each bank may see: only its own payers, its own devices and its own labels, with labels read by arrival time. |
| Results exist? | No model results. Unit tests, integration tests and showcase verification only. |

## A-039

| Field | Value |
| --- | --- |
| Date | 2026-11-12 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `config/evaluation.yaml`, `docs/research/amendments.md` |
| What changed | The weights above, written before any Level 17 results: added `persona_weights` block to `genuine_harm` in `config/evaluation.yaml` (`weights: {retiree: 2, student: 2, gig_worker: 2}`, `default_weight: 1`) and removed weights for vulnerable personas from `not_decided_here`. |
| Why | Genuine harm needs a fixed weighting decided before results; restricting people with little financial slack hurts more. |
| Results exist? | No |

## A-040

| Field | Value |
| --- | --- |
| Date | 2026-11-14 |
| Issue IDs | None |
| Issue IDs v2 | 16-6 |
| Files | `config/world_ranges.yaml`, `config/interactive_runner.yaml`, `config/tactic_blocks.yaml`, `config/gang_mix.yaml`, `config/bank_feed.yaml`, `world/gang_mix.py`, `world/gang_devices.py`, `world/gang_config.py`, `world/gang_planner.py`, `world/bank_feed.py`, `world/world_configs.py`, `tests/unit/test_research_configs.py`, `docs/research/amendments.md` |
| What changed | Generated gangs can draw their take-control flags. (a) Four world ranges, `take_control_share_sim_swap`, `take_control_share_linked_number_switch`, `take_control_share_emulator` and `take_control_share_remote_access`, each training `[0.1, 0.4]` (the last two `[0.05, 0.3]`), test `[0.05, 0.6]` (the last two `[0.02, 0.5]`) and stress `[0.0, 0.9]` (the last two `[0.0, 0.8]`). (b) Stream child 27, `gang_take_control_flags`, draws the flags so the gang-mix stream (child 22) never shifts. (c) A take-control parameter `bind_control_device`: when true, each controlled account gets a gang device event of kind `control_binding` at take-control and at every rotation generation. (d) The gang device kind `control_binding` maps to the bank-visible kind `device_binding_change`. (e) The shares are read with WORLD scope: stress worlds read the stress column and training, validation and test worlds read the training column, because these flags are not route settings. (f) `linked_number_switch` stays bank-invisible in 18a2: a binding event already marks the move to a gang phone, and a phone field on gang events only would make them distinguishable from genuine ones (rule 29). Every new behaviour sits behind `takeover_realism`, false by default, so gangs are unchanged until round 2 turns it on. |
| Why | Issue 16-6 needs generated gangs that occasionally use SIM swaps, linked-number replacement, emulators and remote access, not only the fixed fixture flags, and the bank feed must see those take-overs in the same shape as genuine device binding changes. |
| Results exist? | No |

## A-041

| Field | Value |
| --- | --- |
| Date | 2026-11-14 |
| Issue IDs | None |
| Issue IDs v2 | 16-6 |
| Files | `config/takeover_realism.yaml`, `config/world_ranges.yaml`, `world/gang_mix.py`, `world/world_configs.py`, `tests/unit/test_research_configs.py`, `tests/unit/world/test_gang_mix.py`, `docs/research/amendments.md` |
| What changed | The four take-control share ranges of A-040 now live in `config/takeover_realism.yaml` under `parameters`, not in `config/world_ranges.yaml`, with the same label, unit, training/test/stress ranges, reason, issue ids and derived_from; `config/world_ranges.yaml` is back to 79 settings. `world/gang_mix.py` reads the new file through `load_takeover_shares`, only when `takeover_realism` is true, and the file is deliberately not part of WorldConfigs. Everything else in A-040 stands. |
| Why | `config/world_ranges.yaml` is part of the effective config hashed into the run id, so adding the four entries there changed every fraud-on run id even with the switch off. |
| Results exist? | No |

## A-042

| Field | Value |
| --- | --- |
| Date | 2026-11-17 |
| Issue IDs | 16-4 |
| Issue IDs v2 | 16-4 |
| Files | `scripts/build_level_16_datasets.py`, `scripts/dataset_checkpoints.py`, `scripts/dataset_shards.py`, `tests/unit/test_block_plan.py`, `tests/unit/test_dataset_build_plan.py`, `tests/unit/test_dataset_checkpoints.py`, `tests/unit/test_dataset_shards.py`, `docs/handoff/level_18a1b.md`, `docs/research/amendments.md` |
| What changed | The dataset build now follows the committed Level 12f block plan. (a) A new `--blocks` flag names committed blocks per split (`validation=1-2,test=1-2,stress=1-2,training=1-7`); each block is built with `sample_block`, so world ids keep the `<split>-bNN-NNNN` form and every block is its own Latin hypercube. (b) Without `--blocks`, a per-split count above `block_size` is refused: only a count of at most one block is the committed plan, and the legacy plan of 40 worlds per split has block 0's key values and seeds world for world. (c) `--blocks` cannot be combined with `--worlds-per-split`, and `--splits` must be a subset of the named blocks; `--find-sampler-seed` plans the same blocks the build will. (d) Checkpoint format version 2 adds `key_values_sha256`, the SHA-256 of the world's key values as canonical JSON, to the checkpoint identity; both production callers (the build resume and the assembly check) compute it from the plan's spec, never from the checkpoint file, and a version 1 checkpoint is refused at load. (e) Superseding a prior dataset now requires a shared world seed: a same-split dataset whose manifest lists no common seed is left alone, and a manifest without a worlds list keeps the old same-split rule. (f) The legacy acceptance build (`--preset tiny --sampler-seed 2 --splits training --parallel-workers 2`) is identical to the Level 17 fixture `showcase/fixtures/level_17/datasets/training-20261031T1830-624c2ad40f4c`, and the block plan check with `--preset small --sampler-seed 1 --max-seed-tries 1` fits at sampler seed 1. |
| Why | Issue 16-4 needs Level 18a1b datasets built from the committed Level 12f block plan instead of ad hoc counts, resumable checkpoints that refuse a reused world whose sampled settings changed, and superseding that never hides an earlier dataset of different worlds. |
| Results exist? | No |

## A-043

| Field | Value |
| --- | --- |
| Date | 2026-11-17 |
| Issue IDs | 17-3 |
| Issue IDs v2 | 17-3 |
| Files | `config/evaluation.yaml`, `evaluation/level17_scoring.py`, `evaluation/level17_evaluation.py`, `evaluation/run_level_18_pilot.py`, `evaluation/level18_sanity.py`, `tests/unit/evaluation/test_escalate_once.py`, `tests/unit/evaluation/test_level18_sanity.py`, `docs/research/amendments.md` |
| What changed | Escalation now has a repeats policy, set in `config/evaluation.yaml` under `escalation_repeats`. `daily_escalations` takes `repeat_policy`, either `every_day` (the Level 17 behaviour, re-ranking every bank-day from scratch) or `once_per_world`: an account escalated on one day is under review for the rest of its world, is not a candidate on a later day, and the slot goes to the next account in rank order. Per-bank-day capacity is computed exactly as before, and same-day behaviour is unchanged. `evaluate_world` forwards the policy, and the Level 18 pilot and the Level 18 sanity oracle read the configured value; Level 17 callers keep the `every_day` default and are unchanged. |
| Why | The Level 18a4 oracle on the pilot validation worlds was capped at 69% recall at capacity because 87% (298/343) of escalations repeated an already-escalated account. An escalated account is under review for the rest of the world, as in real review. |
| Disclosure | Found on validation worlds before any test-world evaluation. No success threshold moved. Rules thresholds are tuned on restricted days and genuine harm (`rules_tuning._validation_metrics`), so A-043 changes reported recall, yield and time-to-detection, not the tuned thresholds. Approved by the project engineer 2026-11-17. |
| Results exist? | No |

## A-044

| Field | Value |
| --- | --- |
| Date | 2026-11-18 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `config/rules_labels_extension.yaml` (new), `config/mule_model.yaml`, `decisions/rules_engine.py`, `decisions/rules_config.py`, `decisions/mule_account_model.py`, `evaluation/rules_tuning.py`, `evaluation/run_level_18_pilot.py`, `scripts/handoff_facts.py`, `tests/unit/decisions/test_rules_config.py`, `tests/unit/decisions/test_rules_engine.py`, `tests/unit/evaluation/test_rules_tuning.py`, `tests/unit/evaluation/test_run_level_18_pilot.py`, `tests/unit/test_mule_model_features.py`, `docs/research/amendments.md` |
| What changed | A sixth heuristic rule, `rule_6` (`RB6_LABEL_OR_REGISTRY_NEIGHBOUR`), is defined in `config/rules_labels_extension.yaml` but is loaded only when `rules_plus_labels_and_registry` in `config/mule_model.yaml` is true; the committed value is false. Rule match modes become 'all' (the previous behaviour, default) or 'any'; rule_6 is an 'any' rule over `registry_match_count >= registry_match_min` and `hops_to_complained_account <= complained_hops_max`. `hops_to_complained_account` is built from arrived complaints, not confirmed labels. The extension ships registry_match_min 1 and complained_hops_max 1, both untuned_start, with tuning candidates [1] and [0, 1, 2]. A merged loader adds the extension thresholds and rule_6 after rules 1 to 5 and refuses duplicate threshold keys, rule ids and reason codes; the merged tuning grid is 972 x 1 x 3 = 2916 points. `config/rules_baseline.yaml`, `config/rules_tuning.yaml` and the Level 17 runner are unchanged. |
| Why | The engineer ruling of 2026-11-17 asks for a rule_6 label-or-registry signal to be compared with the XGBoost model in the 2x2 pilot, while keeping the default rules table and every Level 17 result byte-for-byte unchanged. |
| Disclosure | Decided before any full-scale result. No threshold moved; Level 17 unchanged. |
| Results exist? | No |

## A-045

| Field | Value |
| --- | --- |
| Date | 2026-11-18 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `.gitignore`, `docs/research/amendments.md` |
| What changed | `.venv/` is now ignored. Every Level 16 and Level 18 lab dataset built so far carries a `-dirty` code version: block 0 at `71a6d97-dirty`, blocks 1 and later at `d61a65e-dirty`. The stamp was caused only by an untracked `.venv/` folder present in the working tree at build time; no tracked file differed from the recorded commit. No dataset was rebuilt. Datasets built from now on stamp clean. |
| Why | The `-dirty` suffix falsely suggests the recorded commit differs from the code that produced the data, which weakens the provenance every result depends on. |
| Disclosure | Provenance label only. No data, threshold or result changed. |
| Results exist? | No |

## A-046

| Field | Value |
| --- | --- |
| Date | 2026-11-19 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `config/mule_model.yaml`, `decisions/mule_account_model.py`, `decisions/pu_learning.py` (new), `evaluation/run_level_18_pilot.py`, `tests/unit/decisions/test_mule_account_model.py`, `tests/unit/decisions/test_pu_config.py` (new), `tests/unit/test_pu_learning.py` (new), `tests/unit/evaluation/test_run_level_18_pilot.py`, `docs/research/amendments.md` |
| What changed | The pilot account model can now be trained as a positive-unlabelled problem, selected by `pu_method` in `config/mule_model.yaml`: `none` (the committed default, the existing supervised path unchanged), `bagging`, or `elkan_noto`. PU bagging trains `pu_bagging_n_bags` members, each on all positive rows plus `min(unlabelled, ratio * positives)` unlabelled rows drawn without replacement with a seeded generator, and averages member margins before applying the logistic link. Each candidate `(n_estimators, ratio)` in `pu_unlabelled_per_positive_candidates` is scored on validation average precision and the best wins, ties going to fewer estimators then the smaller ratio. Weighted Elkan-Noto trains one unbalanced `g` to estimate P(labelled \| x), takes `c` as the mean of `g` over validation positives, turns each unlabelled row into a target-1 row with weight `((1 - c) / c) * p / (1 - p)` and a target-0 row with weight `1 - weight`, and sets the final class weight from the weight totals; `pu_probability_clip` keeps the factor finite. The manifest records the method, every candidate AP and training time, the chosen values, bag member seeds, and `c`, weight summary and clip count. |
| Why | The bank only holds positive labels and treats every unlabelled account as a negative, so unreported frauds train the model the wrong way. Both methods stay behind a switch that defaults off so the supervised pilot result is unchanged, the method choice is deferred to the full-scale run, and the pilot can compare the candidates on validation. Averaging margins keeps the ensemble SHAP values exact and leaves the members mergeable into one booster for ONNX later. Plain Elkan-Noto probability rescaling is not used because a constant divisor cannot change any ranking metric. |
| Disclosure | Methods added behind a default-off switch before any full-scale result. No committed threshold or existing result changed; the `none` path calls the same code and produces the same `n_estimators_scores` keys as before. |
| Results exist? | No |

## A-047

| Field | Value |
| --- | --- |
| Date | 2026-11-19 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `config/mule_model.yaml`, `docs/research/amendments.md` |
| What changed | `label_maturity_days` in `config/mule_model.yaml` is now 28, up from null with `not_found: true`. The value is the pooled p90 of the measured label delay (27.5687 days) on the 160 full training worlds, rounded up to whole days, so each training world drops its final 28 IST days. Measured with `evaluation/measure_label_delay.py` over `training-20261031T1830-2cdf8145a8d5`, `training-20261031T1830-9e76fa11f9db` and `training-20261031T1830-b16d733022fe`: 3551 accounts with a first illicit credit, 674 labelled, 2877 censored, 12 negative delays, p50 7.56 days, p90 27.57 days, p95 43.37 days. Report `runs/full/label_delay_r7.json`, sha256 `7886a2f6a357d7f5e7964ae086ae5e1547f851552abef02254e0fb553c27d824`. |
| Why | A training row near the end of a world sits in a window where the bank's labels have not arrived yet, so its positives are undercounted; dropping the window up to the pooled p90 delay removes most of that immaturity. |
| Disclosure | Chosen on training worlds only, before any full-scale validation or test result. The p90 is a lower bound because of censoring (674 labelled vs 2877 censored first-credit accounts), and the window is confirmed or switched off by the `maturity_check` stage on validation. |
| Results exist? | No |

## A-048

| Field | Value |
| --- | --- |
| Date | 2026-11-20 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `config/mule_model.yaml`, `evaluation/build_label_trust_curve.py`, `decisions/mule_account_model.py`, `docs/research/amendments.md` |
| What changed | Three blocks added to `config/mule_model.yaml`: `label_trust_curve` (91 shares, day 0 to 90, generated by `evaluation/build_label_trust_curve.py` from the 674 labelled delays pooled over the 160 `by_world` entries of `runs/full/label_delay_r7.json`, sha256 `7886a2f6a357d7f5e7964ae086ae5e1547f851552abef02254e0fb553c27d824`), `label_trust_weighting` (default false) and `maturity_tie_tolerance` (0.002). Stage 1 of the full run becomes a three-way contest between `drop_28`, `all_days` and `all_days_trust_weighted`, replacing the two-variant rule with `MATURITY_SWITCH_IMPROVEMENT`; a variant whose AP is at least the best AP minus the tolerance is tied with the best, so `drop_28` wins only when it leads both others by more than the tolerance, and ties go to the first of `all_days`, `all_days_trust_weighted`, `drop_28`. |
| Why | The 28-day cut removes about 45% of the training rows, including the rows of late-starting mules. Trusting an unlabelled row as a negative in proportion to the share of labels that had arrived by then keeps those rows without treating immature negatives as certain. |
| Disclosure | Chosen before any full-scale validation or test result. The curve uses labelled delays only; censored accounts are missing, so it is optimistic. The switch is off by default and `maturity_check` decides on validation truth average precision. |
| Results exist? | No |

## A-049

| Field | Value |
| --- | --- |
| Date | 2026-11-20 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `docs/research/amendments.md` |
| What changed | Level 18 training worlds are extended from 160 to 320 (blocks 0-7). Night3 built blocks 4-7 on the lab at code version `d61a65e-dirty`: dataset `training-20261031T1830-fb7561164f4f`, 160 worlds (`training-b04-0000` to `training-b07-0039`), seeds 1000160 to 1000319, split `training`. Preset and sampler seed are not recorded in the manifest; consistency was checked via world cards, seed continuity and row counts. None of its 160 run_ids or seeds appears in any existing training, validation, test or stress dataset. |
| Why | Run A's learning curve, scored on validation only, reported `more_training_worlds_needed` = true at 160 training worlds. |
| Stopping rule | 320 is the final Level 18 training size whatever the next learning curve reports. If it flags again, that is recorded as a limitation, not answered with more data. |
| Disclosure | No test or stress world had been scored by any Level 18 full run when this was decided: Run A died in stage 5 while loading validation. The crashed run folders are kept for the record. |
| Results exist? | No |

## A-050

| Field | Value |
| --- | --- |
| Date | 2026-11-21 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/run_level_17.py`, `evaluation/rules_metrics.py`, `evaluation/level17_evaluation.py`, `docs/research/amendments.md` |
| What changed | A world whose card records exactly 0 complaints has no measured inflation, so its E1-re-weighted metrics are undefined: `_world_inflation_factor` returns None, the re-weighted yield is None with the reason "not defined: world has 0 complaints, so the E1 inflation factor is undefined", and the per-scorer summaries compute the re-weighted median over the worlds that defined it and record `worlds_defined` and `worlds_undefined` beside it (a None median with the reason when no world defined it). Measured metrics are unchanged. The success criterion reads the measured yield, so it is unchanged. |
| Why | 14 of the 682 world cards have 0 complaints, and for exactly those 14 the fraud inflation factor is 0: 11 training worlds, 1 validation world, 2 test worlds, 0 of the 120 stress worlds. Stage 5 crashed on test world `test-0028`, whose card records 0 complaints in 14,767 successful payments. Dividing by a zero factor is undefined, and a floor would be a fabricated number. |
| Disclosure | Decided after stage 5 crashed on that world card and before any test metric was produced or read. At the time of the decision `runs/full/run_a320_stage5/` held only `final/<run key>/optuna_trials.json` and `stages/final_mule_model.yaml`, no `result.json`, no `final_validation_and_test.json` and no per-world metric file, and neither log held a line with "recall", "yield" or "test-0" followed by a number (0 of 440 output lines, 0 of 51 error lines). The folder and logs were renamed with the suffix `_zero_complaints_20261121` and kept. |
| Results exist? | No |

## A-051

| Field | Value |
| --- | --- |
| Date | 2026-11-23 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/leakage_checks_5_6.py` (new), `config/leakage_checks.yaml` (new), `tests/unit/evaluation/test_leakage_checks_5_6.py` (new), `docs/research/amendments.md` |
| What changed | Section 23 leakage checks 5 and 6 are added after the Level 18 pilot's test scoring. All three field sets are computed at run time from repository sources: WORLD_FIELDS is the union of the keys returned by `world/fast_mode.py`'s `event_to_record` (called on the module's own example event) and the columns of a real truth file (`runs/full/truth/<run_id>/roles.jsonl`, union of record keys); BANK_VISIBLE is the union of the seven allow-lists in `world/bank_feed.py` (`BANK_VISIBLE_PAYMENT_FIELDS`, `ACCOUNT_OPENED_DETAIL_FIELDS`, `BANK_VISIBLE_WITHDRAWAL_FIELDS`, `BANK_VISIBLE_ACCOUNT_EVENT_FIELDS`, `BANK_VISIBLE_COMPLAINT_FIELDS`, `GENUINE_DEVICE_DETAIL_FIELDS`, `GANG_DEVICE_DETAIL_FIELDS`); SIMULATOR_ONLY is WORLD_FIELDS - BANK_VISIBLE and must contain `purpose` and `sender_is_new_device`. Check 5a scans `features/m1_*.py`, `features/account_summaries.py`, `features/account_snapshots.py`, `features/feature_log.py` and `decisions/mule_account_model.py` for any SIMULATOR_ONLY name used as a string literal or attribute read, and passes only at zero hits; a method invocation such as `datetime.timestamp()` is reported separately and is not a field read. Check 5b requires none of the four training datasets' snapshot or payment tables to carry a SIMULATOR_ONLY column and every `account_model_feature_names()` entry to be a snapshot column. Check 6a scans the same files for a value derived from `account_id`, `bank_id`, `world_id` or `run_id`: wrapping (len, int, str, ...), text transformation (split, rsplit, strip, ...), slicing, arithmetic, an f-string outside a raise, a dict value or a return value. Passing an identifier as a lookup key (function argument, dict key, subscript index) is not a derivation. Check 6b trains a small XGBoost with the committed `train_account_model` defaults and the committed fixed seed from `config/anomaly_detector.yaml` (`random_state` 1717) on training accounts using only id-derived features (id length, numeric suffix, suffix rank within world and bank, prefix code, bank code), scores validation accounts, and fails when validation average precision exceeds `id_leak_max_ap_over_base_rate` (2.0) times the validation mule-role base rate. |
| Why | Section 23 asks whether simulator-only fields or account identifiers reach the account model, and only the bank feed may cross to ashFALL. Checks 5 and 6 read the committed code and the training datasets only, with no test data and no model choice. |
| Disclosure | Added after the test scoring had completed and been read for reporting; the checks read no test data and change no model choice. Results: 5a pass (0 hits), 5b pass (0 simulator-only columns, 42/42 model features present), 6a pass (0 hits), 6b FAIL as a finding: validation average precision 0.6378 against a base rate of 0.0612, 10.4x the base rate. Mule-role accounts carry higher numeric suffixes than other accounts (median 224 vs 129 over 2,589 mule-role and 39,690 other validation accounts), so the identifier numbering itself carries role information. The finding is recorded, not fixed in this change. |
| Results exist? | Yes: `runs/full/run_a320_stage5c/leakage_checks_5_6.json` records the three sets, every hit and the 6b numbers (5a/5b/6a pass, 6b fail). |

## A-052

| Field | Value |
| --- | --- |
| Date | 2026-11-23 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/level18_extra_reporting.py` (new), `evaluation/truth.py`, `tests/unit/evaluation/test_level18_extra_reporting.py` (new), `tests/unit/world/test_gang_truth.py`, `docs/research/amendments.md` |
| What changed | Four reporting metrics are added after the Level 18 Run A single test scoring, computed from the frozen stage 5 checkpoint `runs/full/run_a320_stage5c/final/45c496799170/checkpoint` (code version `0d0ace9`, mule model config sha256 `dcab7aebe67b69334b405e693cabe04485135e0111d5dd3722e5e3985d0b328c`, xgboost cut `0.9631869778454679`): (1) pure model quality, the account-day average precision from `validation_truth_average_precision` plus account-level recall, precision and genuine flag rate at the frozen cut; (2) recall at capacity for the model and for a truth oracle (1.0 for a mule-role account-day, 0.0 otherwise, threshold 0.5), each escalated through `daily_escalations` with the configured repeats policy, at exactly the config's capacity points 0.05, 0.25 and 1.0, with normalized recall = model / oracle (None with a reason when the oracle is 0, values above 1.0 kept); (3) hidden-mule recall by depth with the unreported mule count per depth; (4) recall at capacity and at the cut for the mule accounts of each held-out family in `config/evaluation.yaml` (`slow_drip`, `device_rotation`), read through a new read-only `TruthStore.gang_family` accessor. Aggregates are the median and the 10th-to-90th percentile with worlds used and skipped. A mandatory consistency gate first recomputes every test world's xgboost recall at capacity at the headline point and refuses the run unless it equals that world's stored `result.json` value exactly. The checkpoint is loaded with `PUBaggedAccountModel.load` only after explicit provenance checks (config sha256, training and validation dataset ids, an allow-listed `git diff 0d0ace9 HEAD`, a clean worktree), because the pilot's own checkpoint loader refuses a fit made by another commit. Output: `runs/full/run_a320_stage5c/extra_reporting.json`, with the checkpoint code version, current HEAD, the diff list and the sha256 of `thresholds.json` and every file under `model_bags/`. |
| Why | The single test scoring froze every choice, and these four questions were asked for reporting afterwards. They are answered from the frozen model with the pilot's own scoring, escalation and metric functions, so the numbers mean the same thing as the frozen run's. |
| Disclosure | Added after the single test scoring had completed and been read. It retrains nothing, re-tunes nothing and changes no threshold, so it changes no choice and no verdict. Normalized recall is reported beside recall at capacity, never instead of it. The consistency gate is the proof that the frozen model was scored on the same rows with the same cut. |
| Results exist? | Yes: `runs/full/run_a320_stage5c/extra_reporting.json`. |

## A-053

| Field | Value |
| --- | --- |
| Date | 2026-11-23 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/level18_extra_reporting.py`, `tests/unit/evaluation/test_level18_extra_reporting.py`, `docs/research/amendments.md` |
| What changed | The genuine flag rate and the per-world genuine account count in `evaluation/level18_extra_reporting.py` now read the genuine-role accounts that are never mule accounts, through one new helper `_never_mule_genuine(evidence) = evidence.genuine_active - evidence.mules`. Both previously used `evidence.genuine_active` from `ground_sets`, which lists an account as genuinely active if it carries a genuine-role day on any day, so a recruited mule (genuine before recruitment, mule afterwards) sat in both populations and a flagged mule was counted as an innocent flag. The rate keeps its key `genuine_flag_rate` and its skipped branch, now taken when no never-mule genuine account exists, and `genuine_accounts` counts the same set. A new test builds a world where one of three never-mule genuine accounts is flagged and requires one third. `ground_sets`, `evaluation/level17_scoring.py`, `evaluation/rules_tuning.py`, `evaluation/level17_evaluation.py` and every config are unchanged. |
| Why | The rate answers how often a genuinely innocent account is flagged, and an account that becomes a mule was not innocent at the frozen decision, so dividing by every ever-genuine account mixed mules into the harm denominator. |
| Disclosure | Reporting-only fix made after the single test scoring; it retrains nothing, changes no threshold, no choice and no verdict. The official harm metric `genuine_restricted_days` is unaffected because it counts harm only on genuine-role days, so the shared `ground_sets` and the frozen run's numbers stand. |
| Results exist? | Yes: `runs/full/run_a320_stage5c/extra_reporting.json`, rerun after the fix; the previous file is kept as `runs/full/run_a320_stage5c/extra_reporting_before_a053_20261123.json`. |
+
+## A-054
+
+| Field | Value |
+| --- | --- |
+| Date | 2026-11-23 |
+| Issue IDs | None |
+| Issue IDs v2 | None |
+| Files | `scripts/level18_mlflow_read.py`, `scripts/level18_mlflow_log.py`, `config/level18_failed_attempts.yaml`, `tests/unit/scripts/test_level18_mlflow_read.py`, `tests/unit/scripts/test_level18_mlflow_log.py`, `docs/research/amendments.md` |
+| What changed | Every Level 18 experiment is back-filled into the MLflow experiment `ashFALL Level 18 — mule models` from the saved result files, after the fact. `scripts/level18_mlflow_read.py` holds pure readers that turn a stage file, a maturity check, a learning curve, a PU method search, the recomputed stage 4 file, the final stage 5 folder and the failed-attempt YAML into plain records; it imports no MLflow and no `evaluation` module. `scripts/level18_mlflow_log.py` writes one run per maturity variant, one per learning-curve step, one parent per PU method with one nested child per trial, one per recomputed stage 4 method, one for the final test and stress numbers, and one per failed attempt (ended `FAILED`, or `KILLED` when it was killed, with tags and a file list but no metrics). Every run carries `code_version`, `config_sha256`, `dataset_ids`, `fingerprint_source` (`stage_file`, `partial` or `none`), `source_file`, `source_sha256` and `simulated=true`; a value the files do not record is the literal `not recorded`. The logger is idempotent: before creating a run it searches the experiment for the same `idem_key` and run name and prints SKIP when it finds one. The `idem_key` is a sha256 of the source file bytes plus the sorted-key JSON of the run's name, tags, params and failed-attempt record, so editing a reason re-logs that attempt; it is set only after the run has fully succeeded, so a crashed run stays `FAILED` with a `backfill_error` tag and is logged afresh on the next pass, while `source_sha256` stays as the file's own hash. Failed attempts are listed in `config/level18_failed_attempts.yaml` with their reason, the PR that fixed them and where the record came from. |
+| Why | Every Level 18 experiment, including the failed ones, should be browsable in MLflow with its inputs and provenance, and that record can be built from what the runs already saved. |
+| Disclosure | Read-only back-fill: it retrains nothing, re-scores nothing and changes no file under `runs/`. Nothing reads MLflow for any decision; the only MLflow read is the idempotency lookup, and `core.truth_import_check` stays green because no reader imports an evaluation module. |
+| Results exist? | Yes: the source files under `runs/full`; the back-fill is run after this amendment is merged. |

## A-055

| Field | Value |
| --- | --- |
| Date | 2026-11-23 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/payment_oof_risk.py`, `tests/unit/evaluation/test_payment_oof_risk.py`, `evaluation/payment_model.py`, `tests/unit/evaluation/test_payment_model.py`, `config/mule_payment_feature_order.yaml`, `docs/research/amendments.md` |
| What changed | Pre-registration for the Level 18 payment model `mule_payment_xgb`, written before any Phase 1-3 result exists. It fixes the out-of-fold payee-risk construction that feeds the payment model, the payment model's target and validation objective, the device choice rule, and the hyperparameter search, so none of them can be chosen after seeing a result. The numbered settings below are the complete pre-registered recipe. |
| Why | The payment model consumes an account-model score for the receiver, and that score must come from a model that never saw the row's own world or the payment model would learn from its own training world. Fixing the folds, the frozen recipe, the strict-time and same-bank rule, the target, the objective, the device rule and the search in writing before the runs makes every later number a genuine held-out measurement rather than a choice tuned on the result. |
| Disclosure | Written before Phase 1 ran; no OOF score, device comparison or trial exists yet. It adds new modules and tests and reads existing frozen artifacts (`runs/full/run_a320_stage5c/...`); it changes no existing production file, no committed config value, no threshold and no prior result. Test and stress data are not read in phases 0-3. |
| Results exist? | No: pre-registered before any run. Results are to land under `runs/full/payment_oof/` (out-of-fold payee risk) and `runs/full/payment_model_v1/` (search and frozen model). |

Pre-registered settings, fixed before any Phase 1-3 result:

1. **Folds.** Five folds by whole world. Sort the 320 training `world_id` strings ascending; the world at position `i` (0-based) belongs to fold `i mod 5`. A fold model is trained on the other four folds' worlds and used only to score the held-out fold's worlds.
2. **Fold-model recipe = the frozen Phase 0c Run A recipe exactly.** PU method `bagging`, `n_bags` 50, `unlabelled_per_positive` 1, device `cpu`. Fit parameters (from `runs/full/run_a320_stage5c/final/45c496799170/checkpoint/fit_record.json` `search.best_params`): `max_depth` 8, `learning_rate` 0.14438948595626225, `n_estimators` 118, `min_child_weight` 3.2610095175519205, `subsample` 0.9690047660778822, `colsample_bytree` 0.6934144991295391, `gamma` 3.010102041099164, `reg_alpha` 0.6145078263840675, `reg_lambda` 0.1514174707354069. Trust weighting on, `label_maturity_days` 0, `label_cut` `arrival_time` (from `runs/full/run_a320_stage5c/stages/final_mule_model.yaml`). Features are `account_model_feature_names()` in that order, sha256 `468638cb4d063d5f473ba73d5d99dc2bf45a8534ca77646fd3a967e8126ec061`. Seed 1717 (member seeds 1717 to 1766). Each fold model is discarded after scoring its held-out fold and is never served.
3. **`payee_risk` for a payment** is the account-model score of the receiver's latest account-day (one row per account-day, `snapshot_time` the kept snapshot's time) whose `snapshot_time` is strictly earlier than the payment's `decision_time`, and only when the receiver's bank equals the payment row's `bank_id`. Otherwise it is NaN with a recorded reason: `receiver_bank_unknown` (receiver account absent from snapshots), `receiver_at_other_bank`, or `no_prior_account_day`. Training worlds use their out-of-fold model; validation worlds use the frozen Run A model at `runs/full/run_a320_stage5c/final/45c496799170/checkpoint/model_bags`.
4. **Payment-model target** for training is `label_receiver_reported_by_cutoff` (the bank-visible label already in `payments.parquet`).
5. **Optuna objective** is validation payment-level truth average precision. Truth is the receiver's hidden role at the payment's `decision_time` (`evaluation/dataset_truth_join.py:join_truth` `truth_role_receiver`) being in `MULE_ROLES` (`mule`, `collector`, `cash_out`); this is the existing evaluation reading of the receiver's role on the payment's own day. Truth is read only inside `evaluation/`.
6. **Device rule.** The payment model is fitted once on `cpu` and once on `cuda` with identical fixed parameters and seed, before the search. `cuda` is chosen only if the validation truth AP differs by at most 0.005 in absolute value and the `cuda` wall time is lower on this laptop; otherwise `cpu`. The device is decided once, before the search.
7. **Search.** Reuse `evaluation/hyperparameter_search.py:run_optuna_search` with the same nine numeric parameters from `config/mule_model.yaml` `hyperparameter_space`, the same early stop (`optuna_min_trials_before_stop` 20, `optuna_stall_patience_trials` 10, `optuna_min_improvement` 0.001), at most 40 trials, and seed 1717. This is the same constant stage 3 used: `config/mule_model.yaml` `optuna_seed` 1717 for the sampler and `config/anomaly_detector.yaml` `random_state` 1717 for model fits. The payment model's feature order is frozen and written to `config/mule_payment_feature_order.yaml`, with its sha256 recorded beside the frozen model.
8. **No test, no stress.** Test and stress datasets are not read in any of phases 0-3. The new entry points accept only training and validation dataset folders and raise if a folder's manifest `split` is `test` or `stress`.

## A-056

| Field | Value |
| --- | --- |
| Date | 2026-11-23 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `docs/research/amendments.md`, `evaluation/payment_model.py`, `tests/unit/evaluation/test_payment_model.py`, `config/mule_payment_feature_order.yaml` |
| What changed | Architect ruling for the Level 18 payment model: it fixes the estimator and its target, the fixed parameters for the cpu-versus-cuda device check, the correct feature-list sha256 for the frozen account recipe, the leakage-field sources the input test must use, and the decision to keep `payee_same_bank_risk` with an ablation and a same-bank versus other-bank AP split. |
| Why | Phase 1 finished before the payment-model fit, but A-055 did not name the payment estimator or the fixed device-check values and it recorded the wrong feature-list sha256. Recording the architect's ruling in writing before any payment-model fit keeps the device choice and the search from being tuned on the result. |
| Disclosure | Written after the Phase 1 out-of-fold build had finished (`runs/full/payment_oof/fold_record.json` existed) and before any payment-model fit, device comparison, ablation or search trial existed. It does not edit A-055. |
| Results exist? | No payment-model result at the time of writing; the only prior result is the Phase 1 out-of-fold payee risk. Payment-model results are to land under `runs/full/payment_model_v1/`. |

Rulings, fixed before any payment-model result:

1. **Estimator:** one supervised XGBoost classifier, objective `binary:logistic`, target `label_receiver_reported_by_cutoff`. No PU bagging: positives are about 45% of training rows (the architect's figure; the four training datasets hold 2,523,136 positive of 4,911,546 payment rows = 51.4%), so "unlabelled = maybe positive" handling is not needed.
2. **Fixed device-check parameters:** the Run A frozen account-model best params from Phase 0c (`max_depth` 8, `learning_rate` 0.14438948595626225, `n_estimators` 118, `min_child_weight` 3.2610095175519205, `subsample` 0.9690047660778822, `colsample_bytree` 0.6934144991295391, `gamma` 3.010102041099164, `reg_alpha` 0.6145078263840675, `reg_lambda` 0.1514174707354069), `tree_method` `hist`, and seed 1717. The seed is `config/mule_model.yaml` `optuna_seed` 1717 (lines 147-150), read at `evaluation/run_level_18_pilot.py:530` and passed to `run_optuna_search` at `evaluation/run_level_18_pilot.py:1909`. Derived from: "frozen account recipe, used only as a neutral fixed point for the device check".
3. **A-055 correction:** the frozen feature list is the 42-name `account_model_feature_names()` with sha256 `075183a2b9192ab889029e16c06a01f4f837db1c331218cfb93c2acd6d6efa20` (sha over newline-joined names), not `468638cb…`. A-055 is not edited.
4. **Leakage field lists:** reuse `evaluation/leakage_checks_5_6.py` `simulator_only_fields`, `truth_file_columns` and `bank_visible_fields`. The input-name test fails if any payment-model input is a simulator-only field, a truth field, or not bank-visible.
5. **`payee_same_bank_risk` is kept:** it reads labels by arrival time only (`features/m1_payment.py`, `labelled_accounts_visible`/`complained_accounts_visible` with `arrival_cutoff`), which a real bank may see. The report includes validation truth AP of the best model with `payee_same_bank_risk` set to NaN as an ablation; the ablation is not used for any choice.
6. **Same-bank split:** the report states the frozen best model's validation truth AP separately for payments whose receiver is at the payment row's bank (`payee_risk` finite) and payments where `payee_risk` is NaN.

## A-057

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/payment_model.py`, `tests/unit/evaluation/test_payment_model.py`, `evaluation/payment_oof_risk.py`, `decisions/mule_account_model.py`, `evaluation/hyperparameter_search.py`, `evaluation/level17_scoring.py`, `docs/research/amendments.md` |
| What changed | Corrects the payment model's training target, voids the 23 Nov payment-model search, and fixes the two mypy findings the end-of-round workflow exposed. The target is now derived in `evaluation/payment_model.py`: a payment is positive when `(world_id, receiver_account_id)` is a `(world_id, account_id)` pair in the same dataset's `labels.parquet`. The payments column `label_receiver_reported_by_cutoff` is no longer read. The 23 Nov search (24 trials, best validation truth AP 0.3222) is void and kept only as evidence under `runs/full/payment_model_v1_void_bad_target/`. Mypy: `payment_oof_risk.py` saves and restores the config override through the new `mule_model_config_override()` getter in `decisions/mule_account_model.py`; the truth-scored search functions take the new `HasRole` Protocol (`role: str`) from `evaluation/level17_scoring.py` instead of `ScoredDay`. |
| Why | `replay/dataset_builder.py:394` builds `reported_accounts` from the bare `account_id` with no world id, and `:476` tests `rec_acc in reported_accounts`, so a label received in any world marks every payment to that id in every world as positive — the same bug class as PR #159. The old target's 51.37% positive share against 1.49% validation truth positive share showed the mismatch. The builder and every dataset are frozen and unchanged; Run A does not use this column. |
| Disclosure | Written after the invalid 23 Nov search and before any valid payment-model result. It changes no dataset, no `replay/` file, no committed config value and no prior result; the void run folder is renamed, not deleted. Test and stress data are not read in any phase. |
| Results exist? | No valid payment-model result. The 23 Nov run is void evidence only. Valid results are to land under `runs/full/payment_model_v1/`. |

Facts recorded with the correction:

1. **Finding.** `replay/dataset_builder.py` forms `reported_accounts` from account ids only (`:394`, `reported_accounts.add(r["account_id"])`) and sets the payment column from `rec_acc in reported_accounts` (`:476`). `labels.parquet` keeps `world_id` on every row.
2. **Evidence.** The old column marks 2,523,136 of 4,911,546 training payment rows positive (51.37%); hidden truth marks 27,393 of 1,834,135 validation rows positive (1.49%). The corrected world-scoped target marks 70,821 of 4,911,546 training payment rows positive (1.44%).
3. **Verdicts.** `labels.parquet` holds only verdict `fraud`: training label rows 997, 502, 519 and 1,967 (3,985 total); validation label rows 466 and 956. No "not fraud" verdict exists, so no verdict filter is applied; the builder counts every non-revoked label row that arrived by the cutoff, and `labels.parquet` already holds only those rows.
4. **Fix.** The target is `receiver is not None and (world, receiver) in reported_pairs(dataset)`, where `reported_pairs` is the `(world_id, account_id)` set of that dataset's `labels.parquet`. A null `world_id` or `account_id` in a label row raises `ValueError`. A hand test proves two worlds using `acc_17` with only world A reported leave world B's payments negative; removing the world id from the key makes it fail with `[True, True, True, True] == [True, True, False, False]`.
5. **Void search.** The 23 Nov device check and search are invalid because their target was the buggy column. The folder is renamed to `runs/full/payment_model_v1_void_bad_target/`; `runs/full/payment_model_v1/` is rebuilt by a fresh device check and search after this amendment.
6. **Estimator unchanged.** The single supervised XGBoost classifier with no PU bagging stands by the A-056 ruling; the corrected positive share is 1.44%, recorded in `model_metadata.json` as `training_positives` and `training_rows`.

## A-058

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/payment_test_score.py`, `tests/unit/evaluation/test_payment_test_score.py`, `docs/research/amendments.md` |
| What changed | Pre-registers the once-only final scoring of the frozen Level 18 payment model on the test and stress datasets: the frozen artifacts and their hashes, the `payee_risk` rule for test and stress payments, the hold threshold chosen on validation, the per-world metrics and their cross-world aggregates, the rules-decision status, and the output file with its manifest. No model is fitted or tuned, and no threshold, feature or metric is chosen on test or stress. |
| Why | Task 4 ends by measuring the frozen model on hidden test and stress truth. Writing every rule before the first test or stress row is scored keeps the measurement held out; a threshold or metric chosen after seeing those scores would make the reported numbers a choice instead of a result. |
| Disclosure | Written after PR #171 merged (merge commit `8eb6328`), after the frozen payment model and its validation scores existed, and after the validation-only threshold below was computed. No test or stress payment row had been read for scoring when this was written; the only test/stress reads so far were inventories (folder names, manifest split/world counts, payment column names) and truth-run presence checks, none of which produce a score. It does not edit A-055, A-056 or A-057. |
| Results exist? | No test or stress score exists at the time of writing. Results are to land under `runs/full/payment_model_v1/test_score/result.json`. |

Rulings, fixed before any test or stress score:

1. **Frozen model.** The scored model is `runs/full/payment_model_v1/frozen_model.json` (sha256 `98dffe1fda74342b36d55651deb2168fa144c4886a87cdb10715ff90d3a7864e`) with `runs/full/payment_model_v1/feature_order.yaml` (file sha256 `6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a`; feature-list sha256 `42f52628ad9c6c5cd440fe4b2aa328dd067beea6b929a2368f8dfccda28867c6`). Before any scoring the scorer proves the feature order's `feature_sha256` and feature count (54) equal `model_metadata.json`, and the booster has 54 features and 647 rounds; any mismatch is a stop, not a fix. Seed 1717, device `cuda` as frozen in `model_metadata.json`. No refit, no tuning, no threshold or feature change on test or stress.
2. **`payee_risk` for test and stress payments** follows the validation rule exactly: the frozen Run A account ensemble (`runs/full/run_a320_stage5c/final/45c496799170/checkpoint/model_bags`) scores the receiver's latest account-day (one row per account-day) whose `snapshot_time` is strictly earlier than the payment's `decision_time`, and only when the receiver's bank equals the payment row's `bank_id`. Otherwise the value is NaN with reason `receiver_bank_unknown`, `receiver_at_other_bank` or `no_prior_account_day`. The frozen Run A config (`runs/full/run_a320_stage5c/stages/final_mule_model.yaml`) is active while those scores are produced. Test and stress worlds are scored once each; the same score fills both the model's `payee_risk` input and the same-bank/other-bank split.
3. **Hold threshold, chosen on validation only.** The frozen validation scores are the frozen model's probabilities on the two validation datasets (1,834,135 payments) with their Phase 1 `payee_risk`. With `HOLD_SHARE = 0.005`, the threshold is `numpy.quantile(validation_scores, 1 - HOLD_SHARE, method="linear")` = **0.20182080343366296**. At that threshold 9,171 of 1,834,135 validation payments are held (0.5000177%). The threshold is a constant of this amendment and is applied unchanged to test and stress. A unit test builds the result from fabricated validation and split scores and fails if any test or stress row enters the threshold calculation.
4. **Metrics, per world, then across worlds.** For every test and stress world: payment truth average precision (`average_precision`, discrete tied-group AP) against the receiver's hidden role (`truth_role_receiver` in `MULE_ROLES`) at the payment's decision minute; recall (`TP / positives`) and precision (`TP / held`) at the hold threshold; hold rate (`held / rows`); and truth AP separately for same-bank rows (`payee_risk` finite) and other-bank rows (`payee_risk` NaN). `average_precision` needs at least one positive and one negative; a world or subset that lacks either is recorded as null for that metric and left out of the aggregates, and the row and positive counts are kept. Cross-world aggregates are the median, p10 and p90 (numpy percentile, `method="linear"`) over the non-null per-world values, with the number of worlds used.
5. **Rules comparison.** No payment-level rules decision exists in the datasets or their feature log: the test and stress `manifest.json` `feature_columns` (53 names) and the `payments.parquet` columns hold no rule, hold or decision output (only the `decision_time` timestamp and feature names). The report therefore records `rules: NOT RUN (no payment-level rules decision)` and no rules decision is built.
6. **Scored once.** Every test and stress dataset is scored by exactly one run of the new entry point, which accepts only `test` and `stress` manifests for those folders and raises on training or validation folders, mirroring the earlier guard in reverse. Results are reported as produced, whatever they are.
7. **Output.** `runs/full/payment_model_v1/test_score/result.json` holds a manifest (git hash at run time, model sha256, feature-order file and list sha256, dataset folder names and their manifest `content_hash`, threshold, payment-model seed 1717 and Run A recipe seed), per-world metric rows, cross-world aggregates and the NaN `payee_risk` share with reason counts.

## A-059

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/reason_codes.py`, `tests/unit/evaluation/test_reason_codes.py`, `config/feature_reason_codes.yaml`, `docs/research/amendments.md` |
| What changed | Pre-registers the Level 18 task 5 reason-code rules for the frozen account and payment models: the frozen-artifact checks, the TreeSHAP space, the top-three rule, the one-code-per-schema-group mapping generated from committed sources, the 20,000-row seed-1717 validation samples, the frequency table and the output files. No model is fitted, tuned or changed. |
| Why | A reason code must name the feature groups that actually pushed a held-out row's score up, under rules written before any SHAP value exists. Writing the mapping source, the ranking rule, the sampling rule and the output shape first keeps the reported reasons a measurement of the frozen models rather than a description chosen after seeing them. |
| Disclosure | Written before any SHAP value is computed and before any reason-code output exists. No test or stress row is read in this phase; no model artifact is written. It does not edit A-055, A-056, A-057 or A-058. |
| Results exist? | No. Results are to land under `runs/full/level18b/reason_codes/` and are labelled SIMULATED. |

Rulings, fixed before any SHAP value:

1. **Frozen models and artifact checks.** The account model is the frozen Run A ensemble loaded with `PUBaggedAccountModel.load` from `runs/full/run_a320_stage5c/final/45c496799170/checkpoint/model_bags`, after `load_frozen_recipe()` proves the frozen config sha256 `dcab7aebe67b69334b405e693cabe04485135e0111d5dd3722e5e3985d0b328c`, the 42-name feature order sha256 `075183a2b9192ab889029e16c06a01f4f837db1c331218cfb93c2acd6d6efa20`, the bag count and the contiguous member-seed block. Every file hash recorded under `manifest.checkpoint_sha256` of `runs/full/run_a320_stage5c/extra_reporting.json` (`thresholds.json`, `model_bags/ensemble.json`, `model_bags/member_000.ubj` to `member_049.ubj`) is recomputed from the checkpoint and compared; the set of files present under `model_bags/` must equal the recorded set. The payment model is `runs/full/payment_model_v1/frozen_model.json` (sha256 `98dffe1fda74342b36d55651deb2168fa144c4886a87cdb10715ff90d3a7864e`) with `runs/full/payment_model_v1/feature_order.yaml` (file sha256 `6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a`), checked with the A-058 verifier, then its 54 features and 647 boosted rounds are checked against `model_metadata.json` with the A-058 round check. Any mismatch is a stop, not a fix.
2. **SHAP space and estimator.** TreeSHAP runs in margin space for both models. The account model uses the production `PUBaggedAccountModel.shap_values`, which averages the 50 members' `shap.TreeExplainer(member).shap_values` with weight 1/50 in margin space; no member is weighted differently and no probability-space SHAP is used. The payment model uses `shap.TreeExplainer` on the single frozen booster.
3. **Reason features.** For each row, the features with a strictly positive SHAP contribution are ranked by descending contribution value and the top three are kept (fewer when fewer than three contributions are positive). Ties break by frozen feature order. A zero or negative contribution is never a reason, so a row can carry an empty reason list; that count is reported.
4. **Reason codes from a generated mapping.** One code per feature group in `data/schemas/feature_set_m1.yaml`, formed by upper-casing the group name: `INFLOW`, `OUTFLOW`, `BALANCE_PATTERN`, `ROUTINE_LIFE`, `LIFECYCLE`, `CONTROL`, `REGION`, `NETWORK`, `INTELLIGENCE`, `PAYMENT_LEVEL`. The frozen payment model has one feature the schema does not define, `payee_risk`; it maps to the code `PAYEE_RISK`, taken from its own group in the committed `config/mule_payment_feature_order.yaml`. The whole mapping, with the sha256 of both source files, is written by the module's `build-map` command to `config/feature_reason_codes.yaml` and the run refuses to proceed unless the committed file equals a fresh generation. A row's codes are the mapped codes of its reason features with duplicates removed in rank order, so two reasons in one group carry one code.
5. **Validation sample.** Only the two validation datasets are read; no test or stress row is read. Each model gets exactly 20,000 rows, selected as the 20,000 rows with the smallest `sha256("1717|" + "|".join(key fields))` digest, which is reproducible without depending on a random-number library. Account-day rows are built per world with the production `one_row_per_account_day`, and the key fields are dataset folder name, `world_id`, `bank_id`, `account_id` and the ISO-8601 `snapshot_time`. Payment keys are dataset folder name, `world_id` and `payment_id`. If a split has fewer than 20,000 rows the sample is all of them; a sample request that repeats a key raises. Output rows are ordered by the key fields, not by digest.
6. **Scoring the sample.** Account-day rows are converted to the frozen feature order with the production `_feature_value` rule (None becomes NaN) and scored with `predict_margins` and `predict_probabilities`; the payment rows come from the production `load_split_rows` for validation under the frozen feature order and are converted with `build_matrix` (NaN preserved, frozen categorical encodings), with the margin from the booster's `output_margin=True` prediction and the probability from `predict_proba`.
7. **Frequency table.** For every code: the number of sampled rows whose deduplicated code list contains it, its share of the sample, and the number of rows where it is the first code; plus the number of rows with an empty reason list. The top five codes by rows-with-code are reported per model.
8. **Output.** `runs/full/level18b/reason_codes/account_model.json` and `runs/full/level18b/reason_codes/payment_model.json`, each holding the amendment id, `simulated: true`, the sampling rule, the sample rows (keys, margin, probability, ranked reasons with SHAP value and code, deduplicated codes), the frequency table, and a manifest with the run git hash, the model and artifact sha256 values, the validation dataset folder names and their manifest `content_hash`, seed 1717, sample size, wall seconds and the peak system commit charge. Outputs are descriptive SIMULATED reason codes, not test or stress measurements.
9. **No decision on test or stress, no model change.** This phase only reads validation rows, writes its own JSON, and changes no model output, threshold or frozen artifact. A failed frozen-artifact check or a stale code map stops the phase and is reported; nothing is retried with weakened checks.

## A-060

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/calibration.py`, `tests/unit/evaluation/test_calibration.py`, `docs/research/amendments.md` |
| What changed | Pre-registers the Level 18 task 7 isotonic calibration rules for the frozen account and payment models: the frozen-artifact checks, the row and truth definitions, the validation-only fit, the monotone-ranking assertion, the test-only Brier/ECE/reliability checks and the output files. No model is fitted, tuned or changed. |
| Why | A calibrated probability is only meaningful if the calibrator was fitted without the split that judges it. Writing the truth definition, the binning rule and the output shape before any score exists keeps the reported calibration numbers a measurement of the frozen models and not a result chosen after seeing test. |
| Disclosure | Written before any calibration score is computed and before any calibration output exists. No test or stress row has been read for this phase; no model artifact is written. It does not edit A-055, A-056, A-057, A-058 or A-059. |
| Results exist? | No. Results are to land under `runs/full/level18b/calibration/` and are labelled SIMULATED. |

Rulings, fixed before any calibration value:

1. **Frozen models and artifact checks.** Identical to A-059 ruling 1: the account model is the frozen Run A ensemble at `runs/full/run_a320_stage5c/final/45c496799170/checkpoint/model_bags` after `load_frozen_recipe()` proves the config sha256 `dcab7aebe67b69334b405e693cabe04485135e0111d5dd3722e5e3985d0b328c`, the 42-name feature sha256 `075183a2b9192ab889029e16c06a01f4f837db1c331218cfb93c2acd6d6efa20`, the bag count and seed block, and every file hash recorded under `manifest.checkpoint_sha256` of `runs/full/run_a320_stage5c/extra_reporting.json` is recomputed; the payment model is `runs/full/payment_model_v1/frozen_model.json` (sha256 `98dffe1fda74342b36d55651deb2168fa144c4886a87cdb10715ff90d3a7864e`) with feature order file sha256 `6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a`, its 54 features and 647 rounds checked with the A-058 verifier. Any mismatch stops the phase.
2. **Account-day rows, scores and truth.** Account-days are one row per IST calendar day per account, built per world with the production `one_row_per_account_day`, over both validation dataset folders (fit) and both test dataset folders (check). Scores are the production `predict_account_days` probabilities (the logistic probability of the bagged mean margin). Truth is the hidden role at the kept snapshot's world minute, looked up with `evaluation.truth.load_truth(...).role_at(account_id, minute)` where `minute = core.event_clock.ist_to_world_minute(snapshot_time, clock_start)` and `clock_start` is the world's entry in the dataset manifest, the same rule `evaluation.dataset_truth_join` uses. A role in `MULE_ROLES` (`mule`, `collector`, `cash_out`) is 1; a role in `GENUINE_ROLES` (`genuine`, `victim`) is 0; any other role raises.
3. **Payment rows, scores and truth.** Payments are the production `load_split_rows(..., split, frozen features, OOF root)` rows over both validation folders (fit) and both test folders (check), scored with the frozen `XGBClassifier.predict_proba(matrix)[:, 1]`. Truth is the table's `target` column, the world-scoped receiver-reported flag that `load_split_rows` derives from each dataset's `labels.parquet`.
4. **Fit on validation only, in probability space.** Per model, `sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")` is fitted on the validation raw probabilities against the validation truth, and applied unchanged to test. Validation must hold both classes; otherwise the phase stops. No calibration curve, width, bin count or class weight is chosen from test.
5. **Monotone ranking assertion.** Isotonic calibration may not reorder rows. For validation and for test, ordering the rows by raw probability (ascending, stable) must leave the calibrated probability non-decreasing (`numpy.diff(...) >= 0`), and the fitted `y_thresholds` must be non-decreasing. Any violation raises and stops the phase; no fallback calibrator is used.
6. **Test-only checks and bins.** On test only, the reported numbers are the Brier score `mean((p - y)^2)` and the expected calibration error over 10 equal-count bins: sort rows by probability, split into 10 contiguous bins with `numpy.array_split`, and compute `sum over bins (n_b / N) * |mean(y_b) - mean(p_b)|`. Reliability points for each bin are its index, count, mean probability, observed rate, and minimum and maximum probability. Both the raw and the calibrated probability are reported the same way. Validation contributes only fit diagnostics (rows, class counts, distinct scores, calibrated value range).
7. **Calibrator files.** Each model's calibrator is written as JSON holding `x_thresholds`, `y_thresholds`, `y_min` 0.0, `y_max` 1.0 and the application rule "linear interpolation between the thresholds, clipped to [y_min, y_max]", applied in code with `numpy.interp` followed by a clip. The JSON holds no fitted statistic other than the thresholds.
8. **Output.** `runs/full/level18b/calibration/account_isotonic.json`, `payment_isotonic.json` and `calibration_data.json`, the last holding the amendment id, `simulated: true`, the run git hash, the model and artifact sha256 values, the dataset folder names and their manifest `content_hash`, split row and class counts, the isotonic parameters and calibrator sha256 values, the ranking checks, and the test Brier, ECE and reliability points before and after calibration, plus wall seconds and the peak system commit charge. Outputs are descriptive SIMULATED calibration artifacts, not test or stress measurements and not a production threshold change.
9. **No decision on test or stress, no model change.** This phase only reads validation rows to fit and test rows to check; it changes no model output, frozen artifact, threshold or decision rule, and it writes no tuned parameter. A failed frozen-artifact check, a single-class validation split or a ranking violation stops the phase and is reported; nothing is retried with weakened checks.

## A-060a

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/calibration.py`, `tests/unit/evaluation/test_calibration.py`, `docs/research/amendments.md` |
| What changed | Records the owner's ruling that replaces A-060 ruling 3's payment test check with the Level 18 task 4 production test-scoring path, and fixes the exact reproduction check that must pass before any calibration value is computed. |
| Why | A-060 ruling 3 named `load_split_rows` for both validation and test, but that loader refuses the test split by design (`validate_dataset_split` accepts only training and validation) and the payee-risk OOF root holds training and validation rows only. The owner ruled that the production test path used in task 4 is the same input, so the check must reuse it and prove the rebuilt rows reproduce task 4's recorded totals and truth AP. |
| Disclosure | Written before any test row is read for this phase and before any calibration output exists. The owner gave this ruling in writing after the mismatch was reported; no calibration value had been computed. It does not edit A-055 to A-060. |
| Results exist? | No. Results are to land under `runs/full/level18b/calibration/` and are labelled SIMULATED. |

Rulings:

1. **`load_split_rows` and the OOF module are unchanged.** Their refusal of the test split is correct and stays. No loader is modified or bypassed.
2. **Validation payment fit rows stay as A-060 ruling 3 fixed them.** `load_split_rows` over both validation folders, frozen feature order, `XGBClassifier.predict_proba(matrix)[:, 1]`, truth is the table's `target` column derived from each validation dataset's `labels.parquet`.
3. **Test payment check rows come from the production task 4 path.** The check calls the exact functions `evaluation/payment_test_score.py` used in task 4: `validate_scoring_split` (guard, line 103), `score_split` (line 338) with `join_truth` (`evaluation/dataset_truth_join.py:172`), `_read_world_snapshots` and `score_world_account_days` (`evaluation/payment_oof_risk.py:553` and `:575`), `build_world_payee_risk` (from `evaluation/payment_oof_risk.py`), `attach_payee_risk` (`evaluation/payment_test_score.py:298`), `build_matrix` (`evaluation/payment_model.py`), and `predict_scores` (`evaluation/payment_test_score.py`). The payment truth is the task 4 truth definition: the receiver has a mule role on the payment's decision day, carried by `WorldScores.labels` as `1 if role in MULE_ROLES else 0`.
4. **Same-input reproduction, checked before any calibration value.** The rebuilt test rows must reproduce `runs/full/payment_model_v1/test_score/result.json` exactly: 120 test worlds, 1,846,587 rows, 43,239 positives, every per-world `truth_ap` equal to task 4's recorded value, and the cross-world `truth_ap` aggregate median equal to 0.6320326905308613. A mismatch in any of these stops the phase before a calibrator is fitted or a check number is produced.
5. **Fit and check roles are unchanged.** The calibrator is fitted on validation rows only, in probability space, exactly as A-060 ruling 4 fixed; test is check-only for Brier, ECE and reliability points and no curve, width, bin count or class weight is chosen from test. Because ruling 2 keeps the validation payment truth as `target` while ruling 3 fixes the test payment truth as the hidden receiver role, the payment fit and the payment test check use different truth constructs; this amendment records that difference as the owner's ruling.
6. **Account-model half unchanged.** The account-day rows, scores and truth of A-060 ruling 2 apply as written to both validation (fit) and test (check) folders.
7. **All other A-060 rulings remain in force.** The frozen-artifact checks (ruling 1), the monotone ranking assertion (ruling 5), the test bins (ruling 6), the calibrator files (ruling 7), the outputs (ruling 8) and the no-decision rule (ruling 9) are unchanged.

## A-061

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/onnx_export.py`, `tests/unit/evaluation/test_onnx_export.py`, `docs/research/amendments.md` |
| What changed | Pre-registers the Level 18 task 8 ONNX export rules for the frozen account and payment models: the frozen-artifact checks, the merge of the 50 account bags into one margin-space booster (every leaf times 1/50), the merge proof, the 20,000-row sampling rule, the required converter output contract, the parity checks on validation and test rows and the output files. No model is fitted, tuned or changed. |
| Why | Exporting an ensemble as one booster is only honest if the averaging rule and the failure limits are fixed before the export exists. Writing the leaf scaling, the merge tolerance, the sampling rule and the parity tolerance first keeps the reported numbers a measurement of the frozen models rather than a result adjusted after seeing differences. |
| Disclosure | Written before any merged booster, ONNX file, merge-proof value or parity value exists. No test or stress row has been read for this phase, and no model artifact is written. It does not edit A-055 to A-060a. |
| Results exist? | No. Results are to land under `runs/full/level18b/onnx/` and are labelled SIMULATED. |

Rulings, fixed before any export or parity value:

1. **Frozen models and artifact checks.** The account model is the frozen Run A ensemble at `runs/full/run_a320_stage5c/final/45c496799170/checkpoint/model_bags`: `load_frozen_recipe()` must prove the config sha256 `dcab7aebe67b69334b405e693cabe04485135e0111d5dd3722e5e3985d0b328c`, the 42-name feature sha256 `075183a2b9192ab889029e16c06a01f4f837db1c331218cfb93c2acd6d6efa20`, bag count 50, the seed block 1717 to 1766 and 118 boosting rounds per member; `verify_account_checkpoint(recipe.checkpoint_dir.parent)` must recompute every file hash recorded under `manifest.checkpoint_sha256` of `runs/full/run_a320_stage5c/extra_reporting.json`. The payment model is `runs/full/payment_model_v1/frozen_model.json` (sha256 `98dffe1fda74342b36d55651deb2168fa144c4886a87cdb10715ff90d3a7864e`) with feature order file sha256 `6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a`, its 54 features and 647 rounds checked with the A-058 verifier (`verify_frozen_artifacts`, `verify_booster_rounds`). Any mismatch stops the phase.
2. **Feature orders.** Account matrices carry the 42 `recipe.feature_names` columns in that order built with the production `_feature_value`, and the order must equal `decisions.mule_account_model.account_model_feature_names()`. Payment matrices carry the 54 `feature_order.yaml` features in that order built with the production `build_matrix` and the file's `categorical_encodings`. The ONNX input is float32 with exactly that many columns.
3. **Merged margin-space account booster.** Every member's `save_raw("json")` must share the objective `binary:logistic`, `num_class` 0, `num_feature` 42, `num_parallel_tree` 1 and an identical `base_score` string; otherwise the phase stops, because averaging margins would not be a leaf scaling. The merged tree list is the member trees in `member_000` to `member_049` order and each member's own tree order; in every tree each leaf (a node whose `left_children` entry is -1) has its `split_conditions` value multiplied by 1/50; `tree_info` is all zero; `gbtree_model_param.num_trees` is 5900; `iteration_indptr` is rebuilt as 0 to 5900; the learner parameter block is copied from member 0. The merged booster is written as `account_margin_booster.json` and reloaded with `xgboost.Booster`. No member file is modified and no model is fitted.
4. **Merge proof.** On the 20,000-row account validation sample of ruling 5, the merged booster's margin (`output_margin=True`) must equal `PUBaggedAccountModel.predict_margins` (the production mean of the member margins) with a maximum absolute difference of at most 1e-6. The row count, the row count holding a NaN and the maximum and mean absolute difference are reported. Any value over the limit stops the phase.
5. **Sampling rule.** The A-059 rule is reused unchanged: 20,000 rows selected by the smallest `sha256('1717|' + '|'.join(key fields))` digest, output ordered by the key fields. Account rows use the key fields `(dataset, world_id, bank_id, account_id, snapshot_time)` with `dataset` the dataset folder name; payment rows use `(dataset, world_id, payment_id)`. The same selected keys are used for the native score and the ONNX score. Four independent samples are drawn: account validation, account test, payment validation, payment test. No row is dropped for holding a NaN; the NaN row count is reported.
6. **Row construction.** Account validation rows are the production `one_row_per_account_day` rows over both validation dataset folders. Payment validation rows are the production `load_split_rows(..., "validation", features, OOF root)` table. Account test rows and payment test rows are built with the same production functions the task 4 test path uses: `validate_scoring_split`, `_read_world_snapshots`, `one_row_per_account_day`, `score_world_account_days`, `build_world_payee_risk`, `attach_payee_risk` and `build_matrix`, with the frozen account config active exactly as `score_split` does. Truth files are not read in this phase, because parity is a comparison of scores only.
7. **Export.** `onnxmltools.convert.convert_xgboost` converts the merged account `xgboost.Booster` and the payment booster with `initial_types=[("input", FloatTensorType([None, n_features]))]`, n_features 42 and 54. The converter must return a `TreeEnsembleClassifier` whose graph outputs are exactly `label` and `probabilities`; any other converter output stops the phase, so no post-hoc node is added. The class-1 column `probabilities[:, 1]` is the ONNX score. The opset is the converter's default and is recorded with the converter and package versions.
8. **Parity.** For each of the four samples, the native score is the production probability: `PUBaggedAccountModel.predict_probabilities(matrix)` for the account model and `predict_scores(model, matrix)` (the frozen `predict_proba(matrix)[:, 1]`) for the payment model. The maximum absolute difference between the ONNX score and the native score must be at most 1e-5 per sample; the row count, maximum, mean, over-tolerance count and NaN row count are reported for each. Any sample over the limit stops the phase.
9. **Output.** `runs/full/level18b/onnx/account_margin_booster.json`, `mule_account_xgb.onnx`, `mule_payment_xgb.onnx` and `onnx_export.json`, the last holding the amendment id, `simulated: true`, the run git hash, the model and feature sha256 values, the dataset folder names and manifest `content_hash`, the seed and sampling rule, the converter, onnx, onnxruntime and xgboost versions, the opset, the merge proof, the four parity summaries, wall seconds and the peak system commit charge. Files under `runs/` are gitignored and are not committed.
10. **No model change and no decision on test.** This phase fits nothing, tunes nothing and changes no frozen model, feature order, calibrator, threshold or decision rule. Test rows are used only to measure parity and never to choose a converter setting, a tolerance, a feature order or a sampling rule. A failed artifact check, member disagreement, merge-proof or parity value stops the phase and is reported; nothing is retried with weakened checks.

## A-061a

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/onnx_export.py`, `tests/unit/evaluation/test_onnx_export.py`, `docs/research/amendments.md` |
| What changed | Records two mechanical details A-061 leaves open, fixed before any merge, export or parity value on real dataset rows: merged tree ids are renumbered from 0 in concatenation order, and the exporter proves the ONNX graph has exactly one input named `input`. |
| Why | A-061 ruling 3 requires the written `account_margin_booster.json` to reload and predict for the merge proof, and xgboost 3.2.0 crashes on predict when two loaded trees share an id, which the concatenated members do; A-061 ruling 7 requires the input contract to be proven, and the converter keeps the `FloatTensorType` name `input` verbatim instead of appending `:0`, so the check must pin that name rather than a guessed suffix. |
| Disclosure | Written after the unit tests exist and before any merge, export or parity value on real dataset rows exists. No result file is written by this amendment. |
| Results exist? | No. Results are to land under `runs/full/level18b/onnx/` and are labelled SIMULATED. |

Rulings:

1. **Tree ids in the merged booster.** Merged tree `i` carries `id` `i` for `i` 0 to 5899, in the A-061 ruling 3 concatenation order. No other tree field changes: every leaf `split_conditions` value is still multiplied by 1/50 and internal split values are untouched.
2. **Export contract check.** Each converted graph must hold exactly one input named `input`, the operator `TreeEnsembleClassifier`, and exactly the outputs `label` and `probabilities`; anything else stops the phase as A-061 ruling 7 fixes.
3. **All other A-061 rulings remain in force.** The frozen-artifact checks, the feature orders, the merge preconditions, the merge proof, the sampling rule, the row construction, the parity check, the outputs and the no-decision rule are unchanged.

## A-061b

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/onnx_export.py`, `tests/unit/evaluation/test_onnx_export.py`, `docs/research/amendments.md` |
| What changed | Replaces A-061 ruling 4's single 1e-6 margin gate with the owner's three-part merge gate: a 1e-5 probability-space limit, zero decision flips at the account model's frozen cut, and a ranking check (Spearman correlation at least 0.999999 and identical top-1% index sets). The margin difference is still computed and reported but is no longer gated. |
| Why | The 1e-6 margin limit was an architect guess with no `derived_from`. The merged booster sums 5,900 leaves in float32, and the 20,000-row validation proof measured a deterministic maximum margin difference of 1.310e-05 in two runs (exit 1 each), so the limit cannot be met by any faithful leaf scaling. The new gates measure the properties that matter for serving: the probability error, the decisions the frozen cut makes, and the ranking. |
| Disclosure | Written AFTER the 1e-6 gate failed: the run stopped at `merge_proof` with `merged booster margin differs from the bagged mean margin by 1.310e-05, over the 1e-06 limit` (run exit 1, 80.9 s, reproduced at 77.6 s with the same value and the same merged-booster sha256 `a21ba6033ad957dc75f413eca92348abf11e2835af996f3f7651b15e17f4a954`). No ONNX file or parity value existed when this was written. It changes no frozen model, feature order, sampling rule, row construction or parity tolerance; it only replaces the merge gate and records the margin as a reported number. It does not edit A-055 to A-061a. |
| Results exist? | Partly: the failed margin value exists and is recorded above. No export or parity result exists yet. Results land under `runs/full/level18b/onnx/` and are labelled SIMULATED. |

Rulings:

1. **Merge gate 1 — probability space.** On the A-061 ruling 5 account validation sample of 20,000 rows, the native probability is `PUBaggedAccountModel.predict_probabilities(matrix)` and the merged probability is `1 / (1 + exp(-margin))` from the merged booster's `output_margin=True` margin. The maximum absolute difference must be at most 1e-5, the A-061 parity tolerance. The maximum difference, mean difference, row count and NaN row count are reported.
2. **Margin reported, not gated.** The maximum and mean absolute margin differences between the merged booster and `PUBaggedAccountModel.predict_margins` are computed and reported in `onnx_export.json` and the console summary; no limit applies.
3. **Merge gate 2 — decisions at the frozen cut.** The frozen inclusive cut is read at run time from `runs/full/run_a320_stage5c/extra_reporting.json`, key path `manifest.xgboost_cut`, whose value is `0.9631869778454679` (the ruling's rounded `0.96319`). A row is flagged when its probability is at least the cut, the same inclusive rule as `anomaly_restricted_days` in `evaluation/level17_scoring.py`. The number of rows where the merged and native flags differ must be zero. No cut value is typed into the code.
4. **Merge gate 3 — ranking.** Spearman correlation is the Pearson correlation of midranks (equal values share the average of the ranks they span; scipy is not a declared dependency, so the midranks are computed with numpy). It must be at least 0.999999. The top-1% set is the `ceil(0.01 × rows)` rows ordered by descending probability, with ties broken by ascending row index; the merged and native top-1% index sets must be identical. The correlation, top row count and whether the sets match are reported.
5. **Stop rule.** If any of the three gates fails, the phase stops and the values are reported; no gate is weakened, waived or retried.
6. **All other A-061 and A-061a rulings remain in force.** The frozen-artifact checks, the feature orders, the merge preconditions, the sampling rule, the row construction, the export contract, the parity checks, the outputs and the no-decision rule are unchanged.

## A-062

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `serving/model_repository/mule_account_xgb/config.pbtxt`, `serving/model_repository/mule_payment_xgb/config.pbtxt`, `serving/model_repository/mule_account_xgb/1/.gitkeep`, `serving/model_repository/mule_payment_xgb/1/.gitkeep`, `scripts/handoff_facts.py`, `docs/handoff/level_18b_partial.md`, `docs/research/amendments.md` |
| What changed | Records the Level 18 task 9 Triton model-folder layout and the smoke-load rule for the two Phase 8 ONNX exports: the folder shape copied from `fraud_xgboost`, the input and output contract, the batch limit and batching settings, the 100-row load check and the partial handoff note. |
| Why | Triton only serves a model whose folder, config and version file agree exactly, so the input name, feature counts, output shapes and batch limit have to be written down before anyone trusts a served response. |
| Disclosure | Written 2026-11-24 20:44 UTC, AFTER the 100-row load ran at 20:35 UTC; the round required rules fixed before any result, so this entry is a late record of rules that the files already fixed. No threshold, batch size, batching setting or check was chosen from the load result: the layout is copied from `fraud_xgboost`, `max_batch_size: 128` comes from `TRITON_MAX_BATCH=128` in `.env.example` and `MAX_BATCH` in `serving/fraud_client.py`, the dynamic batching is copied unchanged from `fraud_xgboost`, and the load comparison has no pass threshold (the difference is reported as measured). It changes no model, feature order, calibrator, threshold or decision rule and does not edit A-055 to A-061b. |
| Results exist? | Yes: the load PASS and `runs/full/level18b/triton/triton_load.json`. Results are SIMULATED and land under `runs/`, which is gitignored. |

Rulings:

1. **Folder layout.** `serving/model_repository/mule_account_xgb/` and `mule_payment_xgb/` each hold a `config.pbtxt` and a `1/` version folder whose `model.onnx` is a byte-for-byte copy of `runs/full/level18b/onnx/mule_account_xgb.onnx` (sha256 `e1823d5d35b930e16344ca2fe792f04ae726a10207459c6015ee95f0db2bb4ef`) and `mule_payment_xgb.onnx` (sha256 `7eb86c0e582c49a008fd89858cc220f4936ad8b90ac8566b720ae7aad20371d0`). `*.onnx` is gitignored (`.gitignore` line 14), so the model files are not committed; a tracked `1/.gitkeep` keeps the version folder in git, the same device as `serving/model_repository/fraud_xgboost/1/.gitkeep`.
2. **Input and output contract.** `name` equals the folder name, `backend: "onnxruntime"`, the single input is `input` with `TYPE_FP32` and dims `[42]` (account, frozen Run A order) and `[54]` (payment, `config/mule_payment_feature_order.yaml` order); the outputs are `label` (`TYPE_INT64`, dims `[1]`, model per-row shape declared scalar with `reshape { shape: [ ] }`) and `probabilities` (`TYPE_FP32`, dims `[2]`). Column 1 is the raw model probability; the isotonic calibrator (`account_isotonic.json`, `payment_isotonic.json`) is applied by the client, not by Triton.
3. **Limits.** `max_batch_size: 128` because the repository's single per-request limit is 128 (`TRITON_MAX_BATCH=128` in `.env.example`, `MAX_BATCH` in `serving/fraud_client.py`); dynamic batching `preferred_batch_size: [8, 32]` with `max_queue_delay_microseconds: 1000` copied unchanged from `fraud_xgboost` and not re-tuned; one CPU instance.
4. **Smoke load.** If a Docker engine is available, start the compose `triton` service (profile `serving`) and wait until the repository is live and ready; send 100 float32 rows per model, rng seed 20261124, one NaN feature in each batch, over gRPC `127.0.0.1:8001`; assert the response shapes `probabilities` (100, 2) and `label` (100, 1); compare `probabilities[:, 1]` with the same file on the local onnxruntime and report the maximum absolute difference as measured, with no threshold. If no Docker engine is available, record `Triton load: NOT RUN (Docker engine not running)` and stop there.
5. **Handoff note.** `docs/handoff/level_18b_partial.md` carries the phase numbers already reported, the load result of ruling 4, and the verbatim output of `PYTHONPATH=. py -3.12 -B scripts/handoff_facts.py --part 18b` under its `## Register Changes` heading. The `18b` part in `scripts/handoff_facts.py` lists `evaluation.calibration`, `evaluation.onnx_export` and `evaluation.reason_codes` as interface modules, the records `Calibrator`, `ReasonCode`, `ReasonCodeMap`, `ReasonFeature`, `RowReason` and `WorldRecord`, `config/feature_reason_codes.yaml` and `config/mule_payment_feature_order.yaml` as config files, and no register ids, because Level 18b changes no register entry.
6. **No model change and no decision on test or stress.** This task fits, tunes or changes nothing: no model, feature order, calibrator, threshold or decision rule. No test or stress row is read. Docker availability is not a gate: the load is a smoke check of the committed layout and its absence is reported, not worked around.

## A-062a

| Field | Value |
| --- | --- |
| Date | 2026-11-24 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `docs/research/amendments.md`, `docs/handoff/level_18b_partial.md` |
| What changed | Replaces A-062 ruling 4's synthetic 100-row smoke check with the owner's restated Phase 9 live test: 100 real validation rows per model, a readiness check that the served fraud model (v1) still loads, and the exact start and stop commands for the Triton service alone. |
| Why | A load check on random feature values proves the transport and the graph shapes, but not that the served model consumes the production validation rows the ONNX export was checked on. Scoring 100 validation rows through Triton and comparing with the same file under local onnxruntime closes that gap, and the v1 READY check proves the new folders did not displace the existing model. |
| Disclosure | Written 2026-11-24 21:03 UTC, AFTER both loads ran: the synthetic 100-row load at 20:35 UTC under A-062 and the 100-validation-row load at 21:00 UTC under the owner's restated rule. The owner restated the test in the round message after the first load; this entry records the restated rule, so it is a late record, not a pre-registration. No threshold, row source or setting was chosen from either result: the 1e-5 limit is the A-061 parity tolerance and the rows are the first 100 of the A-061 sorted validation sample on the production feature paths. It changes no model, feature order, calibrator, threshold or decision rule and does not edit A-055 to A-062. |
| Results exist? | Yes: `runs/full/level18b/triton/triton_load.json` records the validation-row load (exit 0, both models under 1e-5, v1 ready). Results are SIMULATED and land under `runs/`, which is gitignored. |

Rulings:

1. **Row source.** Each mule model scores the first 100 rows of the A-061 validation sample, sorted by the A-059 key fields and built with the production feature paths: `account_sample(validation_folders, "validation", feature_names)` for the account model and `_payment_validation_sample(validation_folders, features, encodings, DEFAULT_OOF_ROOT)` for the payment model. The two validation folders are `validation-20261031T1830-5a6d507bde9c` and `validation-20261031T1830-99fed7ecb1d1`. No NaN is injected; the NaN row count of the selected rows is reported (100 of 100 for both models).
2. **Comparison.** The served `probabilities[:, 1]` is compared with the same `1/model.onnx` file under local onnxruntime on the same row matrix. The maximum absolute difference must be at most 1e-5 (the A-061 parity tolerance); any larger value stops the phase. The served `label` shape `(100, 1)` and its agreement with the local ONNX decision are reported.
3. **v1 READY.** `fraud_xgboost` must report ready on the same server before the check passes, so the new folders are proven not to displace the existing model. The server's `successfully loaded` lines for `fraud_xgboost`, `mule_account_xgb` and `mule_payment_xgb` are the pasted readiness evidence.
4. **Start and stop only Triton.** The service is `triton` in `docker-compose.yml` (line 167), image `nvcr.io/nvidia/tritonserver:24.01-py3` (line 168), model repository mount `./serving/model_repository:/models:ro` (line 182). It is started with `docker compose up -d --no-deps triton` and stopped with `docker compose stop triton`; no other service is started. Docker availability is a precondition, not a gate: if the engine is not running, the load says `Triton load: NOT RUN (Docker engine not running)`.
5. **A-062 stays in force otherwise.** Rulings 1 to 3 and 5 to 6 of A-062 (folder layout, input and output contract, limits, handoff note, no model change) are unchanged; only ruling 4's row source and comparison are replaced.

## A-063

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `scripts/level18_console.py`, `tests/unit/scripts/test_level18_console.py`, `scripts/handoff_facts.py`, `docs/handoff/level_18a.md`, `reports/level_18/console/comparison.json`, `reports/level_18/console/hidden_depth.json`, `reports/level_18/console/held_out_families.json`, `reports/level_18/console/calibration.json`, `reports/level_18/console/reason_codes.json`, `reports/level_18/console/payment_results.json`, `reports/level_18/console/leakage.json`, `reports/level_18/console/onnx_parity.json`, `reports/level_18/console/yield_e1.json`, `docs/research/amendments.md` |
| What changed | Pre-registers the Level 18 task 3 console generator: the frozen input files, the output envelope, the nine console files and the exact fields each copies, the definition of the seven leakage checks, the yield/E1 mapping and its one computed number, the no-scoring rule, and the `18a` handoff note. |
| Why | The task 10 model cards and the task 11 showcase may read only these console files, so their field names, source paths and list rules have to be fixed before any console number exists. Otherwise a card or a showcase check could quietly read a different key, a stale result or an invented field. |
| Disclosure | Written before any console file exists (`reports/level_18/console/` is absent at this commit) and before any console number is generated. This task only reads already-written result files: it runs no training, no scoring, no SHAP, no ONNX conversion and no Triton load, and it reads no dataset rows. It changes no frozen artifact, model, feature order, calibrator, threshold or decision rule and does not edit A-055 to A-062a. Ruling 9 is a DECISION that fixes the round's "all seven leakage checks" wording against the checks the repository actually publishes; check 4 has no implementation and is reported as not run rather than invented. |
| Results exist? | No. Results are to land under `reports/level_18/console/` (tracked, not gitignored) and are labelled SIMULATED. |

Rulings:

1. **Inputs (read only).** The generator reads exactly these already-written frozen result files, all repo-relative: `runs/full/run_a320_stage5c/final/45c496799170/result.json`, `runs/full/run_a320_stage5c/extra_reporting.json`, `runs/full/run_a320_stage5c/final/45c496799170/leakage_checks.json`, `runs/full/run_a320_stage5c/leakage_checks_5_6.json`, `runs/full/payment_model_v1/test_score/result.json`, `runs/full/payment_model_v1/diagnosis/diagnosis.json`, `runs/full/level18b/calibration/calibration_data.json`, `runs/full/level18b/reason_codes/account_model.json`, `runs/full/level18b/reason_codes/payment_model.json`, `runs/full/level18b/onnx/onnx_export.json`, and the `runs/full/world_cards/<run_id>/base_rate.json` card of every one of the 120 test worlds listed in `per_world` of the Run A result (the card is found through each world entry's `run_id` field). Every input is opened read-only; the generator never writes under `runs/`.
2. **Output envelope and write rules.** Nine files are written under `reports/level_18/console/`, each with the same three top-level keys: `"simulated": true`, `"manifest": {"generator": "scripts/level18_console.py", "git_hash": <HEAD at generation>, "sources": {<repo-relative input path>: <sha256>}}`, and `"data": {...}`. The git hash comes from `evaluation.payment_oof_risk._git_head()`, the sha256 values from `evaluation.payment_oof_risk._sha256_of_file`, and `sources` lists exactly the inputs that file reads. Files are written with orjson (`OPT_SORT_KEYS | OPT_INDENT_2`) plus a trailing newline, so the bytes are LF-only and deterministic. A missing or unreadable input, an invalid JSON document, or a missing or wrongly typed required key raises a plain `ValueError` naming the input path and the key; there is no default, no fallback and no silently skipped field, and the process exits non-zero. The generator does not catch that error.
3. **`comparison.json`.** `data["test"]` is the whole `across_worlds.test` block of the Run A result copied verbatim: every scorer key present in the source (the four published scorers `anomaly`, `rules`, `rules_and_anomaly`, `xgboost`) and, for each, the five metrics `genuine_restriction_rate`, `median_time_to_detection_hours`, `recall_at_capacity`, `yield_measured`, `yield_reweighted` with their `median`, `p10`, `p90`, `unit`, `world_count` and spread-availability fields. `data["stress"]` is `across_worlds.stress` copied verbatim (its blocks carry the status `not run: no stress worlds supplied`, which is reported as the source records it). `data["decision_cut"]` holds the value of `extra_reporting.json` key `manifest.xgboost_cut` and its source path/key; the cut is described, not applied, and no typed-in cut value appears in the code. `data["test_world_count"]` is the length of the result's `per_world` map. No statistic is recomputed.
4. **`hidden_depth.json`.** `data["hidden_by_depth"]` is `extra_reporting.json` key `test.aggregate.hidden_by_depth` copied verbatim, with its depth keys in sorted order (1 to 5 as published); `data["worlds_processed"]` is `test.worlds_processed`; `data["world_cards_checked"]` is `test.world_cards_checked`. No spread is recomputed.
5. **`held_out_families.json`.** `data["families"]` is `extra_reporting.json` key `test.aggregate.families` copied verbatim (the published `device_rotation` and `slow_drip` blocks, each with `recall_at_capacity`, `recall_at_cut`, `accounts_total`, `worlds_used` and `worlds_absent`); `data["worlds_processed"]` is `test.worlds_processed`.
6. **`calibration.json`.** `data["points"]` is a flat list built from `calibration_data.json` key `models`: for each model under `models` and for the two states `before_calibration` then `after_calibration` of that model's `test` block, one entry per published bin with the bin's own fields (`bin`, `count`, `mean_probability`, `observed_rate`, `min_probability`, `max_probability`) plus `model`, `split` (`test`) and `state` (`raw` for before, `calibrated` for after). `data["summaries"]` carries each model's `rows`, `positives`, `brier` and `ece` for both states. `data["calibrator"]` carries each model's calibrator file name (the published absolute `path` reduced to its file name, so no machine path is committed), its `sha256` and the application rule string from the source. Values are copied, never recomputed.
7. **`reason_codes.json`.** `data["models"]` is keyed by the source file's own `model` field (`account`, `payment`): `feature_count`, `feature_sha256`, `sample_rows` and `seed` from the source manifest, and `frequency` copied verbatim (`rows`, `rows_without_reason`, and every `codes` entry with `code`, `group`, `plain_reason`, `rows`, `share`, `first` in the source's published order). Shares are the source's shares; the code list is not filtered, regrouped or re-sorted beyond the source order.
8. **`payment_results.json`.** `data["threshold"]` and `data["hold_share"]` are the payment result's top-level values. `data["splits"]` holds the `test` and `stress` blocks, each with its whole `aggregate` copied verbatim (the six published metrics `truth_ap`, `recall`, `precision`, `hold_rate`, `same_bank_truth_ap`, `other_bank_truth_ap`, each with `median`, `p10`, `p90`, `worlds_used`), its `nan_payee_risk_share`, and its `nan_reason_counts` when the source publishes them. `data["payee_precision"]` is `diagnosis.json` key `splits.test.payee_precision` verbatim. `data["feature_truth_ap"]` carries `diagnosis.json` keys `splits.test.feature_truth_ap.payee_risk` and `splits.stress.feature_truth_ap.payee_risk` verbatim. `data["frozen_model"]` carries the payment result manifest's model `sha256`, `feature_sha256`, `feature_count`, `rounds` and `seed` and the manifest `git_hash`, with the machine-specific absolute model path left out. `data["leakage"]` carries `diagnosis.json` key `leakage` reduced to its `status`, its `item_status` map and its `run_a_check_6b` block copied verbatim; the per-item evidence lists (`files_scanned`, `citations`, `id_format_checks`) stay in the diagnosis file, which remains the evidence. No number is recomputed.
9. **`leakage.json` and the seven checks.** `data["checks"]` is a list of exactly seven entries in this fixed order: `truth_leak` and `split_leak` from `checks` of `leakage_checks.json`, `check_5a`, `check_5b`, `check_6a` and `check_6b` from `leakage_checks_5_6.json`, and `section_23_check_7` from `checks` of `leakage_checks.json`. Each entry carries `check`, `status` (copied, never assigned), `source` and `detail` (the whole source block verbatim). `data["trivial_world"]` is the `trivial_world` block of `leakage_checks.json` copied verbatim and is reported separately from the seven, because it is a base-rate sanity check on a synthetic world, not a leakage check of the model. `data["section_23_checks_4_to_7"]` is that source block copied verbatim, so its `not run` record is visible; `data["not_implemented"]` names check 4 as not implemented because no repository file evaluates it. DECISION: the round's "all seven leakage checks" is defined as these seven published outcomes; no eighth check is invented and check 4 is reported as not run.
10. **`onnx_parity.json`.** `data["models"]` holds `account` with its `onnx` file name and `sha256`, its `parity` blocks (`validation` and `test`) and its `merge_proof` block copied verbatim, and `payment` with its `onnx` file name and `sha256` and its `parity` blocks. The machine-specific absolute `path` recorded in the run artifact is replaced by the published file name; the sha256 is copied, not recomputed, so the file needs no ONNX bytes at generation time. `data["tolerances"]` and `data["converter"]` are copied verbatim.
11. **`yield_e1.json`.** The E1 re-weighting is reported per `config/evaluation.yaml` block `base_rate_reweighting`, whose inflation factor comes from each world's card (`runs/full/world_cards/<run_id>/base_rate.json`, key `fraud_inflation_factor`). `data["worlds"]` lists, for every one of the 120 test worlds sorted by world key, `world_key`, `run_id`, `fraud_inflation_factor`, `complaints` and `complaints_per_100000` from the card and the world's `yield_rate.measured` and `yield_rate.reweighted` from the `xgboost` scorer block of the Run A result. A card factor is defined when it is strictly positive, the same rule `evaluation/rules_metrics.py` applies when it refuses a non-positive `inflation_factor`; the generator raises `ValueError` if a world's frozen `reweighted` is null exactly when its factor is positive or non-null exactly when its factor is non-positive, so the card and the frozen result are proven to agree. `data["inflation_factor"]` carries `worlds_defined`, `worlds_zero_factor` and `median_defined`: the median of the strictly positive factors computed with `statistics.median` (for an even count, the mean of the two middle values). `data["scorers"]` copies each scorer's `yield_measured` and `yield_reweighted` aggregate blocks verbatim. This median is the only number the generator computes; every other console number is copied from a frozen source.
12. **No scoring, no model change, no decision on test or stress.** The generator launches no training, scoring, SHAP, calibration fit, ONNX conversion or Triton request, reads no dataset rows, and changes no model, feature order, calibrator, threshold or decision rule. Re-running it on the same inputs and the same commit overwrites the nine files with byte-identical `data` and identical `sources`. Its outputs are descriptive SIMULATED console data, not test or stress measurements and not a production threshold change.
13. **Handoff note.** `docs/handoff/level_18a.md` carries the console file list, the source list and the verbatim output of `PYTHONPATH=. py -3.12 -B scripts/handoff_facts.py --part 18a` under its `## Register Changes` heading. A new `18a` part is added to `scripts/handoff_facts.py` with `INTERFACE_MODULES` and `RECORDS_MODULE` both `("scripts.level18_console",)`, empty `RECORD_NAMES`, empty `CONFIG_FILES` and empty `REGISTER_IDS`, because this task adds a read-only generator and changes no register entry.
14. **Tests and sabotage proof.** `tests/unit/scripts/test_level18_console.py` builds a temporary repo-shaped tree of minimal frozen-like JSON documents carrying exactly the required keys and calls the real generator functions (never a copy of their logic). It covers: each file's envelope and copied fields; the `sources` sha256 values equal `hashlib` over the fixture bytes; the lists are read from source keys (four scorers, the published depths, the two families, the seven checks in the fixed order); the yield median over positive factors and the card/result agreement guard; and the failure paths (missing input, malformed JSON and missing key each raise `ValueError`), plus a check that no input file's sha256 changes. Each test has one sabotage proof: the production line it protects is broken, the test is shown failing with the pasted assertion, the line is restored and `git diff` is empty.

## A-063a

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `scripts/level18_console.py`, `docs/research/amendments.md` |
| What changed | Replaces A-063 ruling 2's import of `evaluation.payment_oof_risk._git_head` and `_sha256_of_file` with the generator's own `_git_head()` and `_sha256_of_file()`: the hash comes from `world.run_identity._run_git("git", REPO_ROOT, ["rev-parse", "HEAD"], 5)` and the digest from `hashlib.sha256`. |
| Why | `core/truth_import_check.py` forbids a file outside the allowed reader folders (`evaluation`, `showcase`, `tests`) from importing `evaluation`, because evaluation is a truth reader. A console script that only copies frozen numbers must not inherit truth access, and the guard rejected the first implementation with `scripts/level18_console.py:26: forbidden import from 'evaluation.payment_oof_risk' outside allowed reader folders`. |
| Disclosure | Written AFTER the first generation ran and AFTER the truth-import guard rejected the first implementation. The nine console files were regenerated with the corrected helpers and are byte-identical to the first run (the sha256 of all nine files is unchanged, and so is every console number, every source hash and the recorded git hash). It changes no rule of A-063 about inputs, outputs, lists, the E1 median or the failure policy and does not edit A-055 to A-063. |
| Results exist? | Yes: the nine console files under `reports/level_18/console/`, unchanged by this correction. Results are labelled SIMULATED. |

Rulings:

1. **Local helpers, same rules.** `_git_head()` returns the full HEAD commit of the repository root (the `REPO_ROOT` constant from `world.run_identity`) and raises `ValueError` when git returns nothing; `_sha256_of_file(path)` returns `hashlib.sha256(path.read_bytes()).hexdigest()`. Both keep A-063 ruling 2's rule: no default, no fallback, a plain `ValueError` on failure.
2. **A-063 stays in force otherwise.** Only the sentence naming the evaluation helpers in A-063 ruling 2 is replaced; rulings 1 and 3 to 14 are unchanged.

## A-064

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `scripts/level18_model_cards.py`, `config/mule_model_quality_gate.yaml`, `scripts/mule_model_quality_gate.py`, `tests/unit/scripts/test_level18_model_cards.py`, `tests/unit/scripts/test_mule_model_quality_gate.py`, `.github/workflows/ci.yml`, `scripts/handoff_facts.py`, `docs/handoff/level_18b.md`, `reports/level_18/model_cards/`, `docs/research/amendments.md` |
| What changed | Pre-registers Level 18 task 10: two generated model cards (markdown and JSON) for `mule_account_xgb` and `mule_payment_xgb`, built only from the nine committed console files of A-063, and a new "mule model quality gate" CI job that checks the committed cards against a new threshold config, every threshold carrying `derived_from`. No model is fitted, loaded, scored or changed and model v1's gate is untouched. |
| Why | A model card is only trustworthy if every number in it comes from a committed result and every quality threshold is written down with its reason before the gate exists. Publishing the card fields, the five limitations and the threshold list first means the gate can only measure the frozen results, not choose a standard after seeing them. |
| Disclosure | Written after A-063's console files were committed and merged (PR #178, `e94b62f`) and before any card, threshold config or gate code exists. Every threshold below is set from numbers those already-published console files record; no new measurement, scoring or model load is made for this amendment. It does not edit A-055 to A-063a. |
| Results exist? | No. Results are to land under `reports/level_18/model_cards/` and are labelled SIMULATED. |

Rulings, fixed before any card value or gate exists:

1. **Inputs are the nine console files, read-only.** The generator reads only `reports/level_18/console/calibration.json`, `comparison.json`, `held_out_families.json`, `hidden_depth.json`, `leakage.json`, `onnx_parity.json`, `payment_results.json`, `reason_codes.json`, `yield_e1.json` and calls no evaluation, model, dataset or simulator code. It writes nothing outside `reports/level_18/model_cards/`. Any missing, unreadable, malformed or key-short input raises a plain `ValueError` naming the file or key; nothing is defaulted or skipped. Each card's JSON records `simulated: true`, the generator name, the generation git commit and the sha256 of every console file that card actually read, in the A-063 envelope (`manifest` + `data`), with LF line endings.
2. **Card files and structure.** Four generated files: `reports/level_18/model_cards/mule_account_xgb.json` and `.md`, and `reports/level_18/model_cards/mule_payment_xgb.json` and `.md`. Each JSON card holds, under `data`: `model`, `intended_use` (string), `not_intended_for` (list), `training_data` (label `SIMULATED`, description), `metrics`, `calibration`, `leakage`, `onnx`, `known_limitations` (list of strings) and, for the payment card, `frozen_model` and `hold_share`. Each Markdown card is generated from the same dictionary in the same run and shows the same numbers to six decimal places, the same limitation text and the manifest footer.
3. **Account card field mapping.** `metrics.recall_at_capacity` = `comparison.json` `data.test.xgboost.recall_at_capacity`; `metrics.recall_by_depth` = `hidden_depth.json` `data.hidden_by_depth` (keys `1`..`5`, each with `median`, `p10`, `p90`, `worlds_used`); `metrics.held_out_families` = `held_out_families.json` `data.families`; `metrics.genuine_restriction_rate`, `metrics.median_time_to_detection_hours`, `metrics.yield_measured`, `metrics.yield_reweighted` = the same `comparison.json` `test.xgboost` block; `calibration` = `calibration.json` `data.summaries.account` plus `data.calibrator.account`; `leakage.checks` = the seven entries of `leakage.json` `data.checks` in their committed order plus `data.trivial_world`; `onnx` = `onnx_parity.json` `data.models.account`. Every metric keeps `median`, `p10`, `p90` and world counts exactly as the console file records them.
4. **Payment card field mapping.** `metrics.truth_ap`, `precision`, `recall`, `hold_rate`, `same_bank_truth_ap`, `other_bank_truth_ap` = `payment_results.json` `data.splits.test.aggregate` and `data.splits.stress.aggregate`; `metrics.payee_precision` = `data.payee_precision`; `metrics.feature_truth_ap_payee_risk` = `data.feature_truth_ap`; `metrics.nan_payee_risk_share` and `metrics.nan_reason_counts` = `data.splits.test.*`; `frozen_model` and `hold_share` = `data.frozen_model`, `data.hold_share`; `calibration` = `calibration.json` `data.summaries.payment` plus `data.calibrator.payment`; `leakage` = `payment_results.json` `data.leakage` (five `item_status` entries, `run_a_check_6b`, `status`); `onnx` = `onnx_parity.json` `data.models.payment`.
5. **The five limitations are stated plainly, each anchored to a console value or a code rule.** (a) Per-payee precision is `payee_precision.payee_precision` (0.911046), not the per-world median `payee_precision.reported_per_world_precision_median` (0.994179): the reported per-world number looks better than the payee-level truth. (b) Stress collapse: payment test `truth_ap` median 0.632033 falls to 0.070938 under stress, while the payee-risk feature alone holds stress AP 0.543550 against test AP 0.922653. (c) Other-bank payees have no payee score: `nan_payee_risk_share` is 0.617159 on test, and the `receiver_at_other_bank` count dominates `nan_reason_counts`. (d) Simulator "new account" cleanness: the simulator opens accounts itself and records the opening minute (`world/bank_connectors.py:344`-`369`), so a new account in these results has no prior history; a mule's fresh account opened with a borrowed identity is not represented. (e) Capacity rounds to one per bank-day: recall at capacity uses `max(1, ceil(active_accounts * capacity_pct / 100))` per bank-day (`evaluation/level17_scoring.py:527`), so on a small bank-day one review decision is the whole capacity. The first three limitations embed the console numbers; (d) and (e) are fixed prose anchored to the cited code lines and carry no number.
6. **Threshold config.** `config/mule_model_quality_gate.yaml` holds one `thresholds` list; each entry has `id`, `card` (`mule_account_xgb` or `mule_payment_xgb`), `path` (dot path into that card's `data`), exactly one rule of `at_least`, `at_most` or `equals`, the rule's value, and a non-empty `derived_from` naming the console value and the amendment that published it. The list is fixed now, in this amendment:

| id | card | path | rule | derived_from (console value) |
| --- | --- | --- | --- | --- |
| account_recall_at_capacity_median | account | `metrics.recall_at_capacity.median` | at_least 0.60 | comparison.json test.xgboost median 0.642857, p10 0.477045, 120 worlds; floor sits 0.04 under the median and above p10 |
| account_recall_depth5_median | account | `metrics.recall_by_depth.5.median` | at_least 0.95 | hidden_depth.json depth 5 median 1.0, p10 0.595556, 30 worlds |
| account_genuine_restriction_rate_median | account | `metrics.genuine_restriction_rate.median` | at_most 0.01 | comparison.json median 0.005882, p90 0.008958; rules sit at 0.023529 |
| account_calibrated_ece | account | `calibration.calibrated.ece` | at_most 0.01 | calibration.json account calibrated ECE 0.001843 (raw 0.127137) |
| account_onnx_validation_max_abs_diff | account | `onnx.parity.validation.max_abs_diff` | at_most 1e-5 | onnx_parity.json 1.788418e-07; the A-061b tolerance is 1e-5 |
| account_onnx_test_max_abs_diff | account | `onnx.parity.test.max_abs_diff` | at_most 1e-5 | onnx_parity.json 1.678902e-07 |
| payment_truth_ap_median | payment | `metrics.truth_ap.test.median` | at_least 0.60 | payment_results.json median 0.632033, p10 0.385806, 120 worlds |
| payment_precision_median | payment | `metrics.precision.test.median` | at_least 0.98 | payment_results.json median 0.994179, p10 0.872121 |
| payment_payee_precision | payment | `metrics.payee_precision.payee_precision` | at_least 0.85 | payment_results.json 0.911046 over 1,023 payees and 11,196 held payments |
| payment_other_bank_truth_ap_median | payment | `metrics.other_bank_truth_ap.median` | at_least 0.40 | payment_results.json median 0.467391, p10 0.195569 |
| payment_calibrated_ece | payment | `calibration.calibrated.ece` | at_most 0.02 | calibration.json payment calibrated ECE 0.012821 (raw 0.012284) |
| payment_onnx_validation_max_abs_diff | payment | `onnx.parity.validation.max_abs_diff` | at_most 1e-5 | onnx_parity.json 4.470348e-07 |
| payment_onnx_test_max_abs_diff | payment | `onnx.parity.test.max_abs_diff` | at_most 1e-5 | onnx_parity.json 5.364418e-07 |
| payment_leakage_check_6b_status | payment | `leakage.run_a_check_6b.status` | equals "fail" | payment_results.json status fail, AP over base rate 10.415126 against the 2.0 limit (A-057); the gate keeps the recorded failure visible instead of hiding it |

7. **Gate behaviour.** `scripts/mule_model_quality_gate.py` loads the config with `core.config_validator.load_yaml` and the two committed cards, and for every threshold prints one line `PASS <id> <actual> <rule> <value>` or `FAIL <id> <actual> <rule> <value>`. A missing card, a missing path, a non-numeric value under a numeric rule, a type mismatch under `equals`, an unknown rule, an empty `derived_from`, or fewer thresholds than listed here is a failure, not a skip. Exit code is 0 only when every threshold passes; otherwise it prints the failures and exits 1. It has no flag, environment variable or switch that skips a threshold, ignores a card or overrides a value.
8. **Model v1's gate is untouched.** `scripts/ci_quality_gate.py`, its `.env` threshold, its fixtures and its `Model quality gate` CI job are not modified. The new check is a separate file, config and job named `Mule model quality gate` (job id `mule-model-quality-gate`): checkout, Python 3.11, `pip install pyyaml`, then `python scripts/mule_model_quality_gate.py`.
9. **Handoff.** `docs/handoff/level_18b.md` is the final Level 18b handoff note and carries the verbatim output of `PYTHONPATH=. py -3.12 -B scripts/handoff_facts.py --part 18b`; `--check` must pass. `config/mule_model_quality_gate.yaml` is added to the `18b` `CONFIG_FILES` so the facts block lists it. Phase 9's `docs/handoff/level_18b_partial.md` stays unchanged.
10. **Tests and sabotage proof.** `tests/unit/scripts/test_level18_model_cards.py` builds a temporary repo-shaped tree of minimal console JSON documents carrying the required keys, calls the real generator and asserts: the four files and their envelopes; the numbers equal the fixture console values; the five limitations embed the fixture numbers where ruling 5 says they do; the Markdown shows the same numbers; missing, malformed and key-short inputs raise `ValueError`; inputs are byte-identical after the run. `tests/unit/scripts/test_mule_model_quality_gate.py` calls the real gate against temporary cards and configs and asserts: all thresholds pass on faithful cards; one degraded card fails with its id named and exit 1; a missing path, an empty `derived_from` and a skipped threshold fail. Each test gets one sabotage proof: the production line it protects is broken, the test is shown failing with the pasted assertion, the line is restored byte-identically and `git diff` is empty.
11. **No scoring, no model change, no choice on test or stress.** This task re-reads committed console files only. A failed threshold stops the round and is reported; nothing is retried with a weaker threshold, and no threshold is edited after the gate first runs.

## A-064a

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `.github/workflows/ci.yml`, `docs/research/amendments.md` |
| What changed | A-064 ruling 8's new job gains one environment line on its last step: `PYTHONPATH: ${{ github.workspace }}`. The run command stays exactly `python scripts/mule_model_quality_gate.py`. |
| Why | The first CI run of the new job failed in 6 s with `ModuleNotFoundError: No module named 'core'` (run 45526267169, job 138910298364) on `from core.config_validator import load_yaml`. Running a file under `scripts/` puts `scripts/` first on `sys.path`, so the repository root is not importable, while every other CI step enters the code through `python -m`, which puts the working directory on the path. The only third-party import in that chain is `yaml`, which the job's `pip install pyyaml` step already provides. |
| Disclosure | Written AFTER the first CI run of the new job failed. It changes only the step's path setup: no threshold, card, card value, gate line, exit rule or `derived_from` changes, the gate keeps failing rather than skipping, and model v1's gate and job are still untouched. |
| Results exist? | Yes: the four card files under `reports/level_18/model_cards/`, unchanged by this correction. Results are labelled SIMULATED. |

## A-064b

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `scripts/check.py`, `docs/research/amendments.md` |
| What changed | `scripts/check.py` gains one step, `mule model quality gate` with `ci_job` `mule-model-quality-gate`, running exactly the CI job's command `python scripts/mule_model_quality_gate.py`, and the `Step` record gains an `env` mapping so a step can run under the variables its CI job sets; this step sets `PYTHONPATH` to the repository root. `run_step` passes `{**os.environ, **step.env}` to the subprocess. |
| Why | The guard test `tests/unit/test_check_matches_ci.py` requires every job in `ci.yml` to have a step in `scripts/check.py`. Its first run with the new job failed in the `Fast tests (no Docker)` and `level-showcase` jobs of run 45526700070 with `AssertionError: These CI jobs have no step in scripts/check.py: ['mule-model-quality-gate']`. The local step needs the same `PYTHONPATH` its CI job sets, or it would fail locally for the same import reason A-064a fixed in CI. |
| Disclosure | Written AFTER that CI failure and BEFORE the step was added. It adds no rule about the cards, the 14 thresholds, the gate's output or exit code, or model v1's gate; the new step runs the same gate and fails the same way locally. |
| Results exist? | Yes: the four card files under `reports/level_18/model_cards/`, unchanged by this correction. Results are labelled SIMULATED. |

## A-065

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `showcase/level_18.py`, `showcase/level_18_checks.py`, `tests/unit/showcase/test_level_18.py`, `.github/workflows/ci.yml`, `docs/research/amendments.md` |
| What changed | Pre-registers the Level 18 showcase: seven read-only checks over the committed Level 18 console files of A-063, their names, order, kinds, parts, key numbers, the one calibration chart, the preconditions and the one CI step that proves the Level 18 report exists. No model is fitted, loaded, scored or changed and no dataset row is read. |
| Why | The Level 18 results are frozen and published; the showcase must display them and fail when a console file is missing or contradicts the facts it already records. Writing the check list, the assertions and the key-number formats first keeps the showcase from choosing what to show after seeing the data. |
| Disclosure | Written after PR #179 merged (`b194b87`) and before `showcase/level_18.py` exists. Every value named below is already recorded in the nine committed console files; no new measurement is made for this amendment. It does not edit A-055 to A-064b. |
| Results exist? | No. The showcase report is to land under `reports/showcase/level_18/`; it displays SIMULATED results. |

Rulings, fixed before the showcase module exists:

1. **Inputs are console files, read-only.** The checks read only `reports/level_18/console/comparison.json`, `held_out_families.json`, `hidden_depth.json`, `leakage.json`, `calibration.json`, `onnx_parity.json` and `yield_e1.json`, and only their `data` block. Every load requires `simulated` to be `true`, a `manifest` with a non-empty `generator` and a `sources` mapping, and a `data` mapping; a missing file, a missing key, a non-true `simulated` flag or malformed JSON raises a plain `ValueError` naming the file and key. No evaluation, model, dataset, simulator, ONNX runtime or Triton code is imported or called.
2. **The seven checks, in this fixed order.** Whole-level kinds: 3 normal, 2 edge, 1 negative, 1 consistency.

| # | Name | Kind | Part | Console file |
| --- | --- | --- | --- | --- |
| 18-1 | comparison: recall at capacity with p10 and p90 spreads | normal | a | `comparison.json` |
| 18-2 | held-out families: device rotation and slow drip at capacity and at the cut | normal | a | `held_out_families.json` |
| 18-3 | hidden depth: recall by hidden mule depth 1 to 5 | edge | a | `hidden_depth.json` |
| 18-4 | leakage: all seven published checks, including the recorded 6b failure | negative | b | `leakage.json` |
| 18-5 | calibration: reliability bins before and after isotonic, with a chart | normal | b | `calibration.json` |
| 18-6 | ONNX vs native: parity on validation and test within the 1e-5 tolerance | consistency | b | `onnx_parity.json` |
| 18-7 | yield: measured against E1 re-weighting with the inflation factor | edge | a | `yield_e1.json` |

3. **What each check asserts, and its key number.** All numbers below come from the committed console files. (a) 18-1: `test_world_count` is 120; for each of the four scorers and the five test metrics the spread is ordered (`p10 <= median <= p90`); the xgboost recall-at-capacity median is above both the rules and the rules-and-anomaly median; every fraction metric is within [0, 1] and every detection-hours median is non-negative. Key number: `xgboost 0.642857 (p10 0.477045, p90 0.833333) vs rules 0.368112 and rules+anomaly 0.375000, 120 worlds`. (b) 18-2: exactly `device_rotation` and `slow_drip`; for each, `worlds_used + worlds_absent` equals `worlds_processed` (120) and both recall spreads are ordered; `accounts_total` is positive. Key number: `device_rotation 0.666667 at capacity / 0.930288 at cut (60 used, 60 absent), slow_drip 0.800000 / 0.909091 (59 used, 61 absent)`. (c) 18-3: the depth keys are exactly `1`..`5`; every recall spread is ordered; `worlds_used` equals `world_count`. Key number: `medians depth 1-5: 1.000000, 0.944444, 1.000000, 0.857143, 1.000000; 30 to 120 worlds`. (d) 18-4: exactly seven checks in the A-063 order; exactly one has status `fail` and it is `check_6b`; that block's `average_precision_over_base_rate` exceeds its `max_ap_over_base_rate`; `trivial_world` has status `pass` with `best_direction_ap` below `fail_threshold`; `section_23_check_7` has status `pass`; `section_23_checks_4_to_7` and `not_implemented.check_4` are recorded as `not run`, so the check-4 gap stays visible rather than being dropped. Key number: `7 checks: 6 pass, 1 fail (check_6b 10.415126 > 2.000000); check 4 not run`. (e) 18-5: four series (account and payment, raw and calibrated) with 10 bins each; each series' bin counts sum to its `rows` in `summaries`; calibrated `rows` and `positives` equal raw; every bin satisfies `min_probability <= mean_probability <= max_probability` with `observed_rate` in [0, 1]; account calibrated ECE is below raw ECE. Payment's two ECEs are reported as recorded, not judged. Key number: `account ECE 0.127137 to 0.001843, payment 0.012284 to 0.012821 (1395478 and 1846587 rows)`. (f) 18-6: for account and payment, validation and test: `rows` is positive, `rows_over_tolerance` is 0, `max_abs_diff` is at most `tolerances.parity` and `mean_abs_diff` is at most `max_abs_diff`; the account merge proof has `decision_flips` 0, `top_set_identical` true and `spearman` at least `tolerances.spearman_min`. The tolerance and the Spearman floor are read from the file, never typed into the code. Key number: `account 1.788418e-07 / 1.678902e-07, payment 4.470348e-07 / 5.364418e-07 vs 1.000000e-05; 0 rows over`. (g) 18-7: 120 worlds; `reweighted` is present exactly when `fraud_inflation_factor` is positive; the positive-factor count equals `worlds_defined` and the zero-factor count equals `worlds_zero_factor`; the median recomputed with `statistics.median` over the positive factors equals `median_defined`; the xgboost reweighted median is below its measured median; every present measured and reweighted value is in [0, 1]. Key number: `xgboost 0.923077 measured to 0.272955 reweighted; median factor 39.316691 over 118 of 120 worlds`.
4. **One chart.** 18-5 attaches a `scatter` chart with mean probability on x and observed rate on y, one series each for account raw, account calibrated, payment raw and payment calibrated, 10 points each. No other check attaches a chart.
5. **Preconditions.** The seven files named in ruling 2 must exist. Each missing file is reported by its full path and every check returns NOT RUN with that reason, through the framework's precondition rule. `payment_results.json` and `reason_codes.json` are not read by these seven checks and are not preconditions. Malformed or envelope-less content is a failure, not a NOT RUN.
6. **Parts and quick mode.** Part `a` is checks 18-1, 18-2, 18-3 and 18-7; part `b` is 18-4, 18-5 and 18-6. The coverage rules are checked on the whole level, as the framework does. `get_checks(quick)` returns the same seven checks in both modes, because they only read committed JSON and there is nothing to shrink; the full and quick runs are both measured and reported against the limits (full under 5 minutes, quick under 1 minute).
7. **Reports and exit code.** The framework's writer produces `reports/showcase/level_18/index.html`, `result.json` and `evidence/`, and part runs go to `part_a/` and `part_b/`. The exit code is 0 only when every check passes; a NOT RUN check counts as a failure unless `--allow-not-run` is given. No check is weakened to pass and no assertion is relaxed after the first run; the only thresholds used are the tolerance and Spearman floor read from the console file.
8. **The CI step.** Level 18 is discovered automatically, so the existing `Run the showcase` step (`make show-all QUICK=1`) runs it. One step is added after `Check the Level 7 report exists`: `Check the Level 18 report exists`, running `test -f reports/showcase/level_18/result.json`, because a missing file would mean the level never ran; a failing check already fails the job through the show-all exit code. `scripts/check.py` needs no edit: `level_numbers()` discovers `showcase/level_18.py` and adds its quick step. No other CI job changes.
9. **No scoring, no model change, no choice on test or stress.** Nothing is fitted, loaded or scored and no dataset row is read; the checks copy, order and compare numbers already published in A-063's console files. A failed assertion stops the round and is reported; nothing is retried or relaxed.
10. **Tests and sabotage proof.** `tests/unit/showcase/test_level_18.py` calls the real module and the real check functions and asserts: the seven names, order, kinds and parts; the module attributes the runner needs; the preconditions on a temporary empty folder and on the committed one; the envelope loader refusing a missing file, an unsimulated file and a file without `data`; and, for each check, its key number and one evidence field when run against the committed console. Each test gets one sabotage proof: the production line it protects is broken, the test is shown failing with the pasted assertion, the line is restored byte-identically and `git diff` is empty.

## A-065a

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `showcase/level_18.py`, `showcase/level_18_checks.py`, `tests/unit/showcase/test_level_18.py`, `docs/research/amendments.md` |
| What changed | 18-4 now reads check 4's recorded result from `leakage.json` `data.trivial_world`, the verbatim copy A-063 made of `runs/full/run_a320_stage5c/final/45c496799170/leakage_checks.json` key `checks.trivial_world`. The check requires every one of the seven published entries in `data.checks` and check 4 to carry a recorded `pass` or `fail`; a missing record or a `not run` status raises `ValueError`. It no longer asserts that `data.section_23_checks_4_to_7` or `data.not_implemented.check_4` say `not run`, and neither block is shown as a check result. check_6b is shown as `FINDING — documented limitation (A-051)` with its numbers instead of being counted as a plain failure. New key number: `8 recorded: 7 pass, check_6b FINDING — documented limitation (A-051) 10.415126 > 2.000000; check 4 (trivial world) pass 0.028270 < 0.900000`. |
| Why | The owner's written ruling after the first 18-4 run: check 4 (trivial world) ran and passed in Run A and its result is recorded. A-063 ruling 9 and A-065 ruling 3(d) read the stale `not_implemented.check_4` and `section_23_checks_4_to_7` blocks as "not run", but those are a pre-A-051 placeholder: the later check 5/6 runs and section 23 check 7 are recorded, and check 4's own pass is recorded in the same source file. Leaving check 4 as NOT RUN would misreport a run that happened. |
| Disclosure | Written after PR #180 opened, after the first 18-4 run, and after the owner's ruling, and before the code change. No frozen artifact, model, threshold, score or console file is edited; the console file already holds `data.trivial_world`. It does not edit A-055 to A-065. |
| Results exist? | Yes: `reports/level_18/console/leakage.json` key `data.trivial_world` and its source `runs/full/run_a320_stage5c/final/45c496799170/leakage_checks.json` key `checks.trivial_world`. The showcase report displays SIMULATED results. |

Rulings, fixed before the 18-4 code changes:

1. **Check 4 is read, not invented.** Check 4's status is `data.trivial_world.status` (`pass`) with `best_direction_ap` 0.028269987287482604, `best_feature` `account_age_days` and `fail_threshold` 0.9. Its source is cited as `runs/full/run_a320_stage5c/final/45c496799170/leakage_checks.json:checks.trivial_world`. If that block were missing, the check fails with a plain `ValueError` naming `leakage:data.trivial_world`.
2. **Every record must be recorded.** The seven `data.checks` entries in their A-063 order plus check 4 must each have status `pass` or `fail`. A missing entry, a missing check 4 block, or any status such as `not run` raises `ValueError`. Exactly one record may be `fail` and it must be `check_6b`; its `average_precision_over_base_rate` must exceed its `max_ap_over_base_rate`, and it is shown in the key number as `FINDING — documented limitation (A-051)`. `trivial_world` must pass with `best_direction_ap` below `fail_threshold`, and `section_23_check_7` must pass.
3. **Tests and sabotage proof.** `test_check_18_4_reports_every_leakage_check` asserts the new key number, all seven published statuses and check 4's recorded status. Its sabotage proof removes the `data.trivial_world` record from the console file, shows the 18-4 test failing on the pasted `ValueError`, restores the file byte-identically and shows an empty `git diff`.

## A-066

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/payment_candidates.py`, `tests/unit/evaluation/test_payment_candidates.py`, `docs/research/amendments.md` |
| What changed | Pre-registers the Level 18 task 4b payment-model v2 candidates, their device checks, searches, selection and replacement gate, and the once-only test and stress scoring of the selected winner. Written before any v2 fit, device check, trial or score exists. |
| Why | Payment v1 is frozen at validation truth AP 0.4659191978683695. Three candidate structures must be compared without any choice being made on test or stress, and the replacement rule must be fixed in writing before the first candidate is fitted. |
| Disclosure | Written after PR #180 merged (merge commit `59a8eae`) and before any v2 result. It reads only frozen artifacts and metadata: the v1 `model_metadata.json`, `runs/full/payment_model_v1/device_check.json`, the committed feature order, the dataset manifests, and A-055/A-056/A-057/A-058. No v2 model is fitted, no trial runs, and no test or stress row is read for scoring. It edits no prior amendment, no frozen model, no dataset, no config value and no result under `runs/`. |
| Results exist? | No. Results are to land under `runs/full/payment_model_v2/` (gitignored) and are labelled SIMULATED. |

Rulings, fixed before the first v2 fit:

1. **Same rows, target, OOF risk, feature order and validation as v1.** Training is the four datasets `training-20261031T1830-2cdf8145a8d5`, `training-20261031T1830-9e76fa11f9db`, `training-20261031T1830-b16d733022fe`, `training-20261031T1830-fb7561164f4f`; validation is `validation-20261031T1830-5a6d507bde9c` and `validation-20261031T1830-99fed7ecb1d1` (120 worlds, ids unique across the two). The target is A-057's world-scoped reported pair. `payee_risk` is Phase 1's out-of-fold value from `runs/full/payment_oof/per_world` (validation rows use the frozen Run A model, as in A-055), the feature order is `config/mule_payment_feature_order.yaml` (file sha256 `98ace9793a8dce7cebd69d2bb832941afbf7ca29155a2948b8f096da0d68a34f`, feature-list sha256 `42f52628ad9c6c5cd440fe4b2aa328dd067beea6b929a2368f8dfccda28867c6`, 54 features, `payee_risk` at index 0), and the validation truth is `truth_role_receiver` in `MULE_ROLES` at the payment's decision minute. Expected row counts (training 4,911,546 with 70,821 positives; validation 1,834,135 with 27,393 positives) are re-checked and any difference is a stop. The frozen v1 artifacts are read, never written: `frozen_model.json` sha256 `98dffe1fda74342b36d55651deb2168fa144c4886a87cdb10715ff90d3a7864e` and `feature_order.yaml` sha256 `6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a`, verified with A-058's verifier before any use.
2. **C1 monotone.** One XGBoost classifier on all training rows with `monotone_constraints` a 54-entry vector: `+1` at the `payee_risk` index (0) and `0` elsewhere. Same objective `binary:logistic`, `tree_method` `hist`, seed 1717, and A-056's target and estimator ruling.
3. **C2 segment.** Two boosters: rows whose `payee_risk` is finite (present) and rows where it is NaN. Each segment gets its own Optuna study run separately, and its objective is that segment's validation truth AP. A candidate's validation score is pooled truth AP over every validation row, with each row scored by its own segment's booster.
4. **C3 = C1 + C2.** The C2 structure with the C1 constraint: two segment boosters, each searched separately, each fitted with the 54-entry monotone vector.
5. **Device rule per candidate (A-056).** Before its search, each candidate is fitted with A-056 ruling 2's fixed parameters and seed 1717 once on `cpu` and once on `cuda`; for a segmented candidate every booster is fitted on each device and the candidate's wall time is the total fit time on that device. `cuda` is chosen only when it ran, the pooled validation truth AP differs from `cpu` by at most 0.005 in absolute value, and its total fit time is lower; otherwise `cpu`. `choose_device` from `evaluation/payment_model.py` is reused unchanged.
6. **Search.** Each study reuses `evaluation/hyperparameter_search.py:run_optuna_search` with `config/mule_model.yaml`'s nine-parameter `hyperparameter_space`, `optuna_trials` 40, `optuna_seed` 1717, `optuna_min_trials_before_stop` 20, `optuna_stall_patience_trials` 10 and `optuna_min_improvement` 0.001. C1 runs one study on all training rows; C2 and C3 run one study per segment (`present`, `nan`), so each candidate fits at most 40 trials per study. Every fit uses A-056's fixed estimator settings and model seed 1717.
7. **Candidate score and p10.** A candidate's score is the pooled validation truth AP over all 1,834,135 validation rows using its combined predictions. Its p10 is the 10th percentile (`numpy.percentile`, `method="linear"`) of per-world validation truth AP over the 120 validation worlds, with A-058 ruling 4's null rule (a world without both classes has null AP and is left out of the aggregate; `worlds_used` is recorded). v1's p10 is measured the same way by scoring the frozen v1 model on the same validation rows (validation only, artifact hashes verified first, no fit, no threshold). The v1 measurement is a measurement of a frozen model, not a choice, and is written to `runs/full/payment_model_v2/selection.json` before any candidate search runs.
8. **Selection.** Candidates whose p10 is at least v1's p10 are eligible. Among eligible candidates the highest validation truth AP wins; when a candidate is within 0.002 (inclusive) of the best, the pre-registered priority C1, then C3, then C2 breaks the tie. If no candidate is eligible, no candidate is selected, v1 stays, and the report says so.
9. **Replacement gate.** The selected candidate replaces v1 only if its validation truth AP is at least the literal 0.4659 and its p10 is at least v1's p10; otherwise v1 stays and the report says so. The gate is recorded in `selection.json` with both numbers and the outcome. Nothing is re-searched or re-selected.
10. **Winner scored once on test and stress (A-058 procedure).** After selection, the selected candidate's frozen artifacts are verified against the sha256 values its own `model_metadata.json` recorded, then scored once: validation rows give the score sequence and the hold threshold is re-derived with `HOLD_SHARE` 0.005 using `payment_test_score.hold_threshold`; test folders `test-20261031T1830-4c139026b360` and `test-20261031T1830-4de01fdd1c75` and stress folder `stress-20261229T1830-ed1beb1f7745` are scored with A-058's split guards, `payee_risk` rule, `world_metrics` and `aggregate`. Per-world metrics, cross-world aggregates, the same-bank and other-bank AP, the NaN share and the reason counts are those functions' outputs. Per-payee precision is computed on both test and stress with `payment_diagnosis.dedupe_payees` over the held rows. The stress `payee_risk`-alone AP is the per-world truth AP of the finite `payee_risk` values using `payment_diagnosis.feature_truth_ap`, aggregated by `aggregate`, reported beside the full-model stress AP. No test or stress row enters any choice.
11. **Outputs.** `runs/full/payment_model_v2/C1`, `C2`, `C3` hold each candidate's `device_check.json`, `search_trials.json` (per segment for C2/C3), `search_importances.json`, frozen booster file(s) (`frozen_model.json` for C1; `frozen_model_present.json` and `frozen_model_nan.json` for C2/C3), `feature_order.yaml` and `model_metadata.json` with the artifact sha256 values, validation AP, per-world p10 aggregate and search seconds. `runs/full/payment_model_v2/selection.json` holds v1's AP and p10, every candidate's score, p10, device, trials and eligibility, the tie trace, the gate outcome and, when v1 is replaced, the regeneration list. `runs/full/payment_model_v2/test_score/result.json` holds the scored winner's result and manifest (git hash, artifact sha256 values, dataset ids and content hashes, threshold, seed, wall seconds).
12. **Protections and boundaries.** The frozen v1 model, its feature order, its `test_score/result.json`, the Run A account model, every dataset, every config value and every prior amendment stay unchanged. No ONNX export, Triton model, model card, calibration, reason code or console file is rebuilt in this task; if the gate replaces v1 those items are listed, not regenerated. No test or stress row is read before selection. A failed check, hash mismatch, missing folder or unexpected row count is a stop, not a fix; nothing is retried with weakened checks. The searches run detached on this laptop with a 120-second memory sampler that stops the process when committed memory reaches 85% of the commit limit; peak committed memory and wall time are recorded.
13. **Tests and sabotage proof.** `tests/unit/evaluation/test_payment_candidates.py` calls the real functions and covers: the monotone vector and a fitted booster's non-decreasing response to `payee_risk` (removing the constraint from the fit makes it fail), the segment masks, segment routing, the pooled objective, per-world AP and p10 with null worlds, the tie rule and its priorities, the replacement gate boundaries, artifact-hash re-verification and the winner's scoring guard. Each test gets one sabotage proof: the production line it protects is broken, the test is shown failing, the line is restored byte-identically and `git diff` is empty.

## A-067

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/onnx_export.py`, `evaluation/calibration.py`, `evaluation/reason_codes.py`, `scripts/level18_console.py`, `scripts/level18_model_cards.py`, `config/mule_model_quality_gate.yaml`, `showcase/level_18.py`, `serving/model_repository/mule_payment_xgb/config.pbtxt`, `tests/unit/scripts/test_level18_console.py`, `tests/unit/scripts/test_level18_model_cards.py`, `tests/unit/showcase/test_level_18.py`, `docs/research/amendments.md` |
| What changed | Pre-registers the regeneration of every artifact that depends on the payment model after A-066 selected C1 and its replacement gate passed: the ONNX export and parity check, the Triton model folder, the isotonic calibration, the reason codes, the console files, the payment model card, the quality-gate thresholds and the Level 18 showcase. It also fixes the test edits the regeneration requires. Written before any v2-derived artifact exists. |
| Why | A-066 ruled that C1 replaces v1 and listed the items to regenerate, but fixed nothing about where those regenerated records go, which frozen fields the console may read (the v2 result has no diagnosis), or how the v2 and v1 numbers are shown side by side. Writing those rules first keeps the regenerated files from choosing their own sources after seeing v2. |
| Disclosure | Written after PR #181 merged (merge commit `79b8b64`) and before any v2 ONNX file, calibration, reason code, console file, card, gate threshold or showcase report exists. It reads only frozen artifacts: `runs/full/payment_model_v2/C1/model_metadata.json`, `runs/full/payment_model_v2/selection.json`, `runs/full/payment_model_v2/test_score/result.json`, `runs/full/payment_model_v1/test_score/result.json`, `runs/full/payment_model_v1/diagnosis/diagnosis.json`, and the committed console files. No model is fitted, no dataset split is scored for a new result, and no test or stress row is read for any new choice. It does not edit A-055 to A-066. |
| Results exist? | The frozen v2 result exists at `runs/full/payment_model_v2/test_score/result.json`; the regenerated artifacts below do not exist yet. All results are labelled SIMULATED. |

Rulings, fixed before the first v2-derived artifact:

1. **Frozen v2 inputs.** The replacement model is `runs/full/payment_model_v2/C1` with `model_metadata.json` recording `frozen_model.json` sha256 `d4bc66ce7fcc6c664e3ed67fba09c555f187cc11a1397cb0c5df951132f03bec`, `feature_order.yaml` sha256 `6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a`, feature-list sha256 `42f52628ad9c6c5cd440fe4b2aa328dd067beea6b929a2368f8dfccda28867c6`, 54 features, seed 1717, device `cpu`, and `best_params.all.n_estimators` 104. Every reader calls `payment_candidates.verify_candidate_artifacts` and `payment_candidates.load_candidate_model`, which re-hash the files and check the feature order, the booster's feature count and its round count; a mismatch is a `ValueError`, never a fallback. `selection.json` records v1's measured validation AP 0.4659191819080341 (recorded 0.4659191978683695), v1 p10 0.2811091416884579, C1 validation AP 0.5175284109152372, p10 0.3493015003495941, winner C1, and `gate.replaces_v1` true. The frozen v2 result records threshold 0.1304200652241741, hold share 0.005, validation rows 1,834,135 with 9,171 held, test worlds 120 over 1,846,587 rows and 43,239 positives, and the world blocks used below. Its manifest holds `amendment`, `account_recipe`, `candidate`, `hold_share`, `model` (artifacts, device, feature_count, feature_sha256, path, seed), `scored_dataset_ids`, `threshold`, `validation_dataset_ids` and `wall_seconds`; it records no `git_hash`, no dataset content hashes in the manifest and no `rounds`. A-067 does not invent those: the regenerated console publishes no v2 generation commit (each regenerated file's own envelope carries the commit that regenerated it), the rounds are the metadata's `best_params.all.n_estimators` after `load_candidate_model` checked the booster, and the dataset folders and content hashes stay in the per-split `datasets` blocks the result does record.
2. **"No re-scoring of test or stress" means: no new test or stress result or diagnosis.** The frozen v2 result is the only source of every published payment metric. Test rows are read only by the A-060a calibration reproduction check and the A-061b 20,000-row parity sample, and stress rows by neither. No calibration, reason-code, console, card, gate or showcase file publishes a metric computed from a new test or stress scoring pass.
3. **ONNX export under A-061b's rules.** `evaluation/onnx_export.py` keeps its account half byte-for-byte in behaviour: the Run A recipe, checkpoint checks, 50-bag merge, merge proof and account parity are unchanged. The payment half reads the C1 candidate through the A-067 ruling 1 calls, exports the booster with the same converter and contract (`input` float32 `[None, 54]`, outputs `label` and `probabilities`), and keeps the 20,000-row seed-1717 validation and test samples and the 1e-5 parity tolerance. Outputs go to `runs/full/level18c/onnx/`; `runs/full/level18b/onnx/` is left untouched as v1's record. Parity must pass on both splits or the run stops. The account parity and merge-proof numbers must equal v1's published values exactly (they are the same frozen account artifacts); a different account value stops the run.
4. **Triton folder.** The v1 ONNX file currently at `serving/model_repository/mule_payment_xgb/1/model.onnx` (sha256 `7eb86c0e582c49a008fd89858cc220f4936ad8b90ac8566b720ae7aad20371d0`) is copied, not moved, to `serving/model_archive/mule_payment_xgb_v1/1/model.onnx`, and the copy's sha256 is checked against that value before the v2 file is written in its place. The archive path is outside `serving/model_repository/`, so Triton cannot scan it. `serving/model_repository/mule_payment_xgb/config.pbtxt` changes only its comment: the source path becomes `runs/full/payment_model_v2/C1/frozen_model.json` and `runs/full/level18c/onnx/mule_payment_xgb.onnx`, its sha256 becomes the exported v2 sha256, and the calibrator path becomes `runs/full/level18c/calibration/payment_isotonic.json`. The serving name, backend, batch limits, input/output schema and instance group stay unchanged. A Triton load attempt is made only if Docker is running; otherwise the report says `Triton load: NOT RUN`.
5. **Calibration under A-060 and A-060a.** `evaluation/calibration.py` keeps its account half, and the payment half fits the isotonic calibrator on the C1 validation payment probabilities against `label_receiver_reported_by_cutoff`, proves the monotone ranking rule on validation and test, and reports test Brier and ECE before and after calibration. The A-060a reproduction reference becomes `runs/full/payment_model_v2/test_score/result.json`; the pre-registered totals stay 1,846,587 test rows and 43,239 positives, and every rebuilt test world must equal that result's world list exactly (world id, rows, positives, truth AP) or the run stops before a calibrator is fitted. Outputs go to `runs/full/level18c/calibration/`: `account_isotonic.json`, `payment_isotonic.json` and `calibration_data.json`.
6. **Reason codes under A-059.** `evaluation/reason_codes.py` keeps its account half, and the payment half explains the C1 model on the same 20,000-row seed-1717 validation sample with the same TreeSHAP space, positive-contribution ranking, three-reason limit and committed code map. Outputs go to `runs/full/level18c/reason_codes/account_model.json` and `payment_model.json`.
7. **Console mapping.** `scripts/level18_console.py` reads `runs/full/payment_model_v2/test_score/result.json`, `runs/full/payment_model_v2/C1/model_metadata.json`, `runs/full/payment_model_v1/test_score/result.json` and `runs/full/payment_model_v1/diagnosis/diagnosis.json` for the payment card, and the three regenerated files above plus the unchanged Run A inputs for the rest. `payment_results.json` keeps its envelope and its `manifest.sources` hashes the four payment inputs; its data block becomes: (a) `version` `"v2"`; (b) `threshold` 0.1304200652241741 and `hold_share` 0.005 from the v2 result; (c) `splits.test` and `splits.stress` copied from the v2 result (aggregate, `nan_payee_risk_share`, `nan_reason_counts`); (d) `payee_precision` copied verbatim from the v2 result's `splits.test.payee_precision` (0.9256820319849483 over 1,063 distinct payees, 12,839 held payments, 984 positive payees); (e) `feature_truth_ap` copied from the v1 diagnosis's `splits.<split>.feature_truth_ap.payee_risk` block, because the payee-risk feature is produced by the unchanged frozen Run A model, with an equality assertion that its stress median, p10, p90 and `worlds_used` equal the v2 result's `splits.stress.payee_risk_alone_truth_ap` values (0.5435498675272497, 0.32443482273658947, 0.8114507881207781, 119); a mismatch raises `ValueError`; (f) `frozen_model` with `sha256`, `feature_sha256`, `feature_count` 54, `rounds` 104 and `seed` 1717 from the C1 metadata, and no `git_hash`; (g) `leakage` copied from the v1 diagnosis exactly as before, because the leakage checks read datasets and Run A, not the payment model. A `previous_version` block holds v1's `version` `"v1"`, `git_hash` `dbefd6c7b13032fc2c0c1655e7a644b95d064f14`, `threshold` 0.20182080343366296, `hold_share` 0.005, `frozen_model` (`sha256` `98dffe1fda74342b36d55651deb2168fa144c4886a87cdb10715ff90d3a7864e`, 54 features, 647 rounds, seed 1717), the same two split blocks with its aggregates (test truth AP median 0.6320326905308613, recall 0.247117435203129, precision 0.9941791044776119, hold rate 0.005003022362144852, same-bank 0.8764378360928944, other-bank 0.467390559057543; stress truth AP median 0.07093756449237654, recall 0.17482998395354168, precision 0.09381740143214562, hold rate 0.028381564386260896), its test `payee_precision` block (0.9110459433040078 over 1,023 payees and 11,196 held payments) and its `feature_truth_ap`. The console's other eight files and their sources are unchanged.
8. **Model card.** `scripts/level18_model_cards.py` reads the regenerated `payment_results.json`, `calibration.json` and `onnx_parity.json` as before. The payment card gains `version` `"v2"` and a `previous_version` block copied from the console's block, shown in the Markdown as "Previous version (v1)" with its threshold, rounds, test and stress metrics and payee precision. Its limitations state the three required facts with the published numbers: stress truth AP median 0.122030 against the payee-risk feature alone at 0.543550; the stress hold rate 0.029711 against the 0.005 target; per-payee test precision 0.925682 over 1,063 payees. The stress-hold-rate fact is added to the existing stress limitation, so the card keeps five limitations. The per-payee limitation's "reported per-world median" now reads the test precision median from the console's test aggregate (1.0) instead of the v1 diagnosis's extra field, which v2 does not publish.
9. **Quality gate.** `config/mule_model_quality_gate.yaml` keeps 14 thresholds and every account threshold. The seven payment thresholds are re-derived from the v2 console values with `derived_from` naming the v2 source (A-067): truth AP median floor 0.62 against 0.6493820260824092 (p10 0.4132446979009128); precision median floor 0.98 against 1.0 (p10 0.8078182827535159); payee precision floor 0.90 against 0.9256820319849483 over 1,063 payees and 12,839 held payments; other-bank truth AP median floor 0.42 against 0.47435510648054935 (p10 0.19727324667600954); calibrated ECE ceiling 0.02 when the measured value is at most 0.02, 0.03 when it is at most 0.03, and 0.05 otherwise, with a measured value above 0.05 stopping the round as a failed calibration rather than widening the gate; the two ONNX ceilings stay 1e-5 and cite the regenerated payment diffs; the check-6b status stays `equals: "fail"` and cites the unchanged A-057/A-063 record. No threshold or rule may be changed after the gate is first run.
10. **Showcase.** `showcase/level_18.py` and `showcase/level_18_checks.py` keep their seven checks, names, order, kinds, parts and assertions; only the `WHAT_YOU_SAW` sentence that quotes the payment ECE pair is updated to the regenerated payment Brier/ECE values. All seven checks must pass on the regenerated console. The account key numbers stay as published.
11. **Test edits this regeneration requires.** These are the only existing-test changes permitted, and each keeps its assertions' structure and strength; only the payment fixtures and the payment literals change, to the regenerated values. `tests/unit/scripts/test_level18_console.py`: the fixtures `_payment_result`, `_diagnosis` and `_fixture_root` are replaced by v2-shaped fixtures for the four payment inputs, `test_missing_input_raises`'s expected ONNX path becomes `runs/full/level18c/onnx/onnx_export.json`, and new tests assert the v2 mapping, the `previous_version` block, the absence of a v2 `git_hash` and the stress equality guard. `tests/unit/scripts/test_level18_model_cards.py`: the fixture `_payment_results` gains `version`, `previous_version` and the v2 frozen model, and `test_card_numbers_equal_console_values`, `test_limitations_embed_console_numbers` and `test_markdown_shows_the_same_numbers` carry the regenerated payment literals plus a previous-version assertion. `tests/unit/showcase/test_level_18.py`: `test_check_18_5_reports_the_calibration_and_its_chart` and `test_check_18_6_reports_the_onnx_parity` carry the regenerated payment ECE and parity literals. Every changed test gets one sabotage proof: the production line it protects is broken, the test is shown failing, the line is restored byte-identically and `git diff` is empty.
12. **Protections and boundaries.** The frozen v1 model, feature order, v1 result and v1 diagnosis are read, never written; only the Triton copy of v1's ONNX is created, and `runs/full/level18b/` stays unchanged. The Run A account model and checkpoint, every dataset, every config value, and every prior amendment stay unchanged. No model is retrained or re-searched, and `runs/full/payment_model_v2/` is read only. The regeneration runs detached on this laptop with the 120-second memory sampler, and logs are read from disk after exit. A failed check, hash mismatch, missing folder, unexpected row count or non-reproducing test split is a stop, not a fix; nothing is retried with weakened checks.

## A-068

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `scripts/audit_20261125/audit_b.py`, `scripts/audit_20261125/payment_experiments.py`, `docs/research/amendments.md` |
| What changed | Pre-registers the validation-only improvement experiments of the 2026-11-25 audit round (B1 capped residual stacking, B2 payee-deduplicated sample weights, B3 dropping `payee_same_bank_risk`, B4 an alternative pinned learner, B5 account class weighting) and the one metric every experiment is judged by. Written before any B experiment is fitted or scored. |
| Why | The audit round asks for improvement experiments but forbids adopting any of them; fixing the experiments, their data and their metric before the first fit keeps a later comparison from choosing a flattering variant or metric after seeing the numbers. |
| Disclosure | Written after the Part A audit results (A1-A6) exist and before any Part B model is fitted. It reads only the frozen C1 metadata and `selection.json` for the baseline values, plus `requirements-ml.txt` for the learner pins. No test or stress row is read by any Part B experiment; no frozen model or result file is written; nothing in this amendment changes production code. All results are labelled SIMULATED. |
| Results exist? | No Part B result exists at the time of writing. The C1 baseline values cited below come from the frozen `runs/full/payment_model_v2/selection.json`. |

Rulings, fixed before the first B fit:

1. **Rows, features, parameters and device.** Every payment experiment reuses A-067 ruling 1's four training datasets and two validation datasets, the frozen feature order `config/mule_payment_feature_order.yaml` (54 features, `payee_risk` first), the frozen C1 parameters in `runs/full/payment_model_v2/C1/model_metadata.json` (`best_params.all`, 104 rounds, seed 1717 unless a ruling below changes the learner) and the chosen device `cpu`. Truth is `truth_role_receiver` in `MULE_ROLES` on validation. Test and stress rows are never read in Part B.
2. **The metric.** Each experiment reports (a) the pooled validation truth AP computed by `sklearn.metrics.average_precision_score` over all 1,834,135 validation rows, (b) the worst-10% worlds p10, the 10th percentile of per-world AP over the 120 validation worlds that contain both classes, and (c) as a descriptive column only, the validation hold rate at the frozen v2 threshold 0.1304200652241741. The baseline is C1's recorded validation AP 0.5175284109152372 and p10 0.3493015003495941. No experiment is adopted, and no threshold, gate or published artifact changes.
3. **B1 capped residual stacking.** Two boosters, both with the frozen C1 parameters and seed 1717: P, trained on `payee_risk` alone, and Q, trained on the 53 features without `payee_risk`. The scored logit is `logit(p_P) + clip(logit(q_Q), -c, +c)` for `c` in {0.5, 1, 2}; the reported variant is the one with the highest validation AP, ties broken by higher p10. The clipped constant is reported.
4. **B2 payee-deduplicated weights.** The frozen C1 recipe with training-Row sample weights `1 / count(receiver_account_id)`, so each payee's payments weigh 1 in total.
5. **B3 dropping `payee_same_bank_risk`.** The frozen C1 recipe with that one column removed, the monotone vector adjusted to the remaining columns.
6. **B4 alternative learner.** Runs only if the learner is installed and pinned in `requirements-ml.txt`; LightGBM or CatBoost, same features, short Optuna of at most 15 trials with the same seed. If neither is pinned, the item is NOT RUN with the reason and the installed versions.
7. **B5 account class weighting.** One XGBoost member on the account stage-4 rows with `scale_pos_weight` set to the measured negatives/positives ratio, reported with its source count. If the stage-4 account rows cannot be built through an existing entry point within the 20-minute box, the item is NOT RUN with the reason; no substitute surrogate is fitted.
8. **Time boxes and NOT RUN.** Each item runs in its own box; an item that exceeds its box is stopped and reported NOT RUN or PARTIAL with its last finished number. Every NOT RUN names its reason (time, missing pin, missing entry point).
9. **Protections.** The frozen v1 model, the C1 model, their feature orders, result files, the Run A account model and checkpoint, every dataset, every config value and every prior amendment are read only. All outputs go under `runs/full/audit_20261125/`. No experiment is promoted, merged into a config, or scored on test or stress; the owner decides later.
10. **Environment pins.** Before any B fit the report lists the installed versions of xgboost, lightgbm, catboost, scikit-learn, shap, optuna and numpy and the matching pinned lines, if any, from `requirements-ml.txt`; a learner without a pin cannot be adopted even if it runs.

## A-069

| Field | Value |
| --- | --- |
| Date | 2026-11-25 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `evaluation/audit_train_a.py`, `evaluation/audit_payment_experiments.py`, `docs/research/amendments.md` |
| What changed | Records that the two audit retrain scripts A-068 named under `scripts/audit_20261125/` now live under `evaluation/`, with only the repository-root expression changed from `parents[2]` to `parents[1]`. |
| Why | Level 12 showcase check 7 (truth-import guard, `core/truth_import_check.py`) rejects any `evaluation.*` import from outside the allowed reader folders `[evaluation, showcase, tests]` in `config/truth_store.yaml`; the scripts compile matrices with `evaluation.payment_model` and `evaluation.payment_candidates`, so `scripts/` is a forbidden home for them. Moving them into `evaluation/` satisfies the guard without weakening it. |
| Disclosure | The relocation happened after the Part B results existed and after A-068 was written; the executed code is unchanged, no ruling, dataset, parameter or metric changed, and every recorded Part B number stands. The move does not read test or stress rows. |
| Results exist? | Yes, the Part B results listed in the round report; this amendment changes no result. |

## A-070

| Field | Value |
| --- | --- |
| Date | 2026-11-26 |
| Issue IDs | 154 |
| Issue IDs v2 | None |
| Files | `docs/research/amendments.md` |
| What changed | Pre-registers Run B, the behaviour-only account model, before any Run B row is read by the driver or any Run B result exists: the exact config file and its sha256, the cpu device, the eight dataset folders, the four world counts, the out-root, the worktree commit, issue #154's serving rule verbatim, and the once-only test scoring. |
| Why | Issue #154 fixes the serving choice on validation recall at capacity, but Run B itself, its config, its data, its device, and the point at which test may be read had not been written down. Registering them first keeps the later comparison from choosing a flattering config, dataset, device or metric after seeing numbers. |
| Disclosure | Written after the Run B config file already existed (it is git-ignored) and after Run A's frozen results exist; no Run B row has been read by the driver and no Run B result exists. It reads only `runs/full/run_b_behaviour_only.yaml`, Run A's frozen `runs/full/run_a320_stage5c/extra_reporting.json` dataset block, `runs/pilot18/xgboost_device_parity.json`, `config/evaluation.yaml`, the recall-at-capacity code, and issue #154. No test or stress row is read for any choice here. All results are labelled SIMULATED. |
| Results exist? | No Run B result exists at the time of writing. Run A's frozen test result at `runs/full/run_a320_stage5c/final/45c496799170/result.json` exists and is read only. |

Rulings, fixed before Run B starts:

1. **Config and its hash.** Run B reads `runs/full/run_b_behaviour_only.yaml` in the main folder (git-ignored by `.gitignore` line 57, `runs/`). Its sha256 is `a47911d8c49527304fe4218446299d81dd8b5e420113133446bd7d379480ae96`. It was loaded with the driver's own loader (`decisions.mule_account_model.set_mule_model_config_path` followed by `load_mule_model_config`, the two calls `evaluation/run_level_18_full.py` main makes for `--mule-model-config`), which returned: `data_path = compact`, `use_label_network_features = False`, `rules_plus_labels_and_registry = False`, `xgboost_device = cpu`, `hyperparameter_search = optuna`, `optuna_trials = 40`, `optuna_min_trials_before_stop = 20`, `optuna_stall_patience_trials = 10`, `optuna_min_improvement = 0.001`, `optuna_seed = 1717`, `pu_method = none`, `label_maturity_days = 28`, `label_trust_weighting = False`, `maturity_tie_tolerance = 0.002`, `parity_ap_tolerance = 0.005`, and `label_network_features = ('hops_to_complained_account', 'registry_match_count', 'registry_match_age_days')`. Label-network features are off and rule_6 is off because both switches are `False`. A different hash, a different value, or a file that will not load is a stop, not a fix.
2. **Device cpu.** Run B trains on `cpu`. The account-model GPU parity check failed on the pilot: `runs/pilot18/xgboost_device_parity.json` records xgboost 3.2.0, seed 1717, n_estimators 100, 36,564 training rows and 19,705 validation rows, cpu validation truth AP `0.38843574601915914`, cuda validation truth AP `0.4007876172624497` on both cuda runs, `ap_difference` `0.012351871243290569` against `parity_ap_tolerance` `0.005`, `cuda_runs_bit_identical` true and `passed` false. Run A trained on `cpu`, so a `cuda` Run B would carry a measured 0.0124 validation AP device gap into the Run A-versus-Run B comparison; `cpu` keeps both models on one device.
3. **Data.** The same eight dataset folders Run A used, taken from the `datasets` block of `runs/full/run_a320_stage5c/extra_reporting.json`: training `training-20261031T1830-2cdf8145a8d5`, `training-20261031T1830-9e76fa11f9db`, `training-20261031T1830-b16d733022fe`, `training-20261031T1830-fb7561164f4f`; validation `validation-20261031T1830-5a6d507bde9c`, `validation-20261031T1830-99fed7ecb1d1`; test `test-20261031T1830-4c139026b360`, `test-20261031T1830-4de01fdd1c75`. No stress dataset is passed, so no stress row is read or scored. Every folder is passed by its absolute path under `C:\Users\sinne\Downloads\PROJECT-ashFALL\runs\full\datasets\`; the datasets are not copied. The driver takes every input as a `Path` argument (`--training`, `--validation`, `--test`, `--truth-root`, `--world-cards-root`, `--out-root`) and derives dataset identities from folder names, so absolute paths need no relative base.
4. **Learning curve and out-root.** `--training-world-counts 40,80,160,320`. The out-root is `C:\Users\sinne\Downloads\PROJECT-ashFALL\runs\full\run_b320` (absolute, in the main folder); the driver writes one JSON per stage and the final pilot result under it. Stdout, stderr, sampler lines, exit code and elapsed time are logged under the same out-root.
5. **Worktree and code version.** Run B executes from the detached worktree `C:\Users\sinne\Downloads\ashFALL-runB` at commit `7f35f3e6f2be0465f6afd4ccbde563d567ba1b41`, the merge of PR #183 and the head of `origin/feature/level-18-mule-models` when this amendment was written. No file in the worktree is edited and no git command that changes its HEAD is run while Run B is going; the git-ignored inputs are read from the main folder by absolute path. A stage whose recorded code version is not `7f35f3e` is a stop.
6. **Serving rule (issue #154, verbatim).** "Serve Run A's account model (behaviour + label-network features) if it beats Run B's model (behaviour only) on VALIDATION recall at capacity at the committed 0.5% budget. Otherwise serve Run B's model." "A tie (equal validation recall at capacity) goes to Run B's model, the simpler one that needs no complaint history." "Run B's model stays the documented fallback for a bank with no complaint history." "The choice is made on validation numbers only. Test and stress results are never used to choose between the two."
7. **The validation comparison.** Recall at capacity is computed with the same two calls Run A's published recall at capacity used: per-world escalations from `evaluation.level17_scoring.daily_escalations` (per bank and day, capacity `max(1, ceil(active_accounts * capacity_pct / 100))` at line 527, repeat policy from `load_escalation_repeat_policy()`), then `evaluation.rules_metrics.recall_at_capacity(escalations, mules)`; `evaluation/level18_extra_reporting.py` lines 781-782 makes exactly those calls. The reported validation number is the median across the validation worlds where recall at capacity is defined, at the 0.5% budget issue #154 names; `config/evaluation.yaml` sweeps capacity from 0.05% to 1.0% with a 0.25% headline, so 0.5% is an interior point of the reported curve. A tie means the two validation aggregates are exactly equal; no numeric tolerance is applied, and an exact tie goes to Run B as the issue states. A-071 records both numbers, the rule trace and the decision before any test score exists.
8. **Once-only test scoring.** Run B's test split is scored exactly once, after A-071 is written, through the driver's stage 5, the same stage-5 procedure Run A used. No second test scoring, no re-tuning or threshold change after the test numbers are read, and no test or stress number enters any choice. The Run B test result is reported beside Run A's frozen `runs/full/run_a320_stage5c/final/45c496799170/result.json`, never instead of it. Stress is never scored because no stress dataset is passed.
9. **Protections.** The frozen Run A account model, its checkpoint and result files, the payment v1 and v2 models and results, every dataset, every config value and every prior amendment are read only. No file under `runs/full/run_a320*` is written; all Run B output goes under `runs/full/run_b320/`. Nothing from Run B is promoted, merged into a config or served in this round; the owner decides from A-071 and the test report.
10. **Launch gates and stop rule.** Before launch the run records: commit limit at least 44 GB, `C:` free space at least 40 GB, committed usage at most 17.5 GB, and AC sleep set to never (`powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 0`). The run starts detached and hidden with the 2-minute memory sampler; the sampler stops it at 85% of the commit limit measured at launch (the brief's 38.905 GB at a 45.77 GB limit). If it stops, the last 10 sampler lines are reported, and the stop line is never raised, no world count is cut and no data is dropped.

## A-070a

| Field | Value |
| --- | --- |
| Date | 2026-11-26 |
| Issue IDs | 154 |
| Issue IDs v2 | None |
| Files | `docs/research/amendments.md`; the script named in ruling 2 is added later with its unit test |
| What changed | Amends A-070 rulings 7 and 8 before launch: stage 5 may score test in the same run, with the test outputs sealed until the serving decision is committed and pushed; and the validation recall at capacity of both models is computed after the run by one new read-only script. |
| Why | The driver runs all five stages in one call and has no way to stop before stage 5, so A-070 ruling 8 could not be carried out as written. No stored Run A validation recall at capacity was found: the `validation_recall` values in Run A's stage files and `result.json` are measured at genuine-restriction budgets, not at review capacity. |
| Disclosure | Written on the owner's ruling after the launch checks and before any Run B launch: `runs/full/run_b320/` held only the gate script and its log. Run A's frozen test result already existed. The owner directed that the run start once this amendment is pushed, without waiting for its merge. All results are SIMULATED. |
| Results exist? | No Run B result exists at the time of writing. |

Rulings, fixed before Run B starts:

1. **Sealed test.** Stage 5 of the driver writes Run B's test scores in the same run, as Run A's stage 5 did. Sealed outputs are `runs/full/run_b320/stages/final_validation_and_test.json` and everything under `runs/full/run_b320/final/`. Until A-071 is committed and pushed, only their file paths and sha256 values are recorded; they are not opened, parsed, printed or reported. The one exception is each final fit's `checkpoint/` folder (`thresholds.json`, `fit_record.json` and the model files), which the pilot writes from training and validation data and which ruling 2's script reads. From the driver's stdout and stderr only progress and error lines are read before then. The check that stage 5's recorded code version is `7f35f3e` is made after A-071 is pushed. This replaces A-070 ruling 7's last sentence with "A-071 records both numbers, the rule trace and the decision before any Run B test number is read", and A-070 ruling 8's "after A-071 is written" with "in the same run, and read only after A-071 is pushed". Test is still scored exactly once.
2. **One script, both models.** `evaluation/level18_serving_validation.py` scores Run A's frozen model (`runs/full/run_a320_stage5c`) and Run B's final model on the same two validation folders A-070 ruling 3 names, and reads no test or stress dataset. No stored recall is reused. For each model it uses that model's own run config, that checkpoint's own `xgboost_cut` and tuned rules thresholds, one row per account-day built as `evaluation/level18_extra_reporting.py` builds it, `daily_escalations` in `anomaly` mode at capacity `0.5` with `load_escalation_repeat_policy()`, then `recall_at_capacity`. The aggregate is the median across validation worlds that have at least one mule. Run A's number above Run B's serves Run A; otherwise, including exact equality, Run B is served. The script writes nothing under `runs/full/run_a320*`.
3. **Reading of "0.5%".** The 0.5 in ruling 2 is review capacity, as A-070 ruling 7 reads issue #154. `config/evaluation.yaml` also holds a `main_genuine_restriction_rate_pct` of 0.5; each model's `xgboost_cut` is the one its run tuned at that main budget.
4. **Everything else in A-070 stands.**


## A-071

| Field | Value |
| --- | --- |
| Date | 2026-11-26 |
| Issue IDs | 154 |
| Issue IDs v2 | None |
| Files | `docs/research/amendments.md` |
| What changed | Records the serving decision of issue #154: Run A's account model (behaviour and label-network features) is served; Run B's behaviour-only model stays the documented fallback for a bank with no complaint history. |
| Why | Run A's validation recall at capacity is above Run B's under the rule fixed in A-070 and A-070a. |
| Disclosure | Written after Run B finished and before any Run B test number was opened. Test was already scored by stage 5 in the same run, as A-070a ruling 1 allows; those files are sealed by the hashes below. Run A's frozen test result was already known. All results are SIMULATED. |
| Results exist? | Yes: `runs/full/run_b320/serving_validation.json`, sha256 `da33a12bb385dbd162aadd697407a967c48890ba7e61f33632eecd822b076d78`, written at code version `5ba6133`. |

The comparison, at review capacity 0.5 on `validation-20261031T1830-5a6d507bde9c` and `validation-20261031T1830-99fed7ecb1d1`, repeat policy `once_per_world`:

| Model | Median recall at capacity | Defined worlds | Own score cut | Method | Fit code version |
| --- | --- | --- | --- | --- | --- |
| Run A | 0.6363636363636364 | 120 | 0.9631869778454679 | bagging | `0d0ace9` |
| Run B | 0.5954545454545455 | 120 | 0.9772312809606538 | bagging | `7f35f3e` |

Rule trace: 0.6363636363636364 > 0.5954545454545455, so Run A beats Run B; the tie clause does not apply. **Serve Run A's account model.** World by world, Run A is higher in 59, equal in 55 and lower in 6 of 120 worlds. No test or stress number entered this choice.

Run A was fitted at `0d0ace9` and scored here at `5ba6133`. Between the two, the changes to `decisions/mule_account_model.py`, `evaluation/level17_scoring.py` and `features/account_labels.py` add an accessor, a type protocol and a log level, and change no score.

Sealed Run B outputs, recorded before opening: `runs/full/run_b320/stages/final_validation_and_test.json` sha256 `b2f7b79bee9eeedfd822be6ab22ca1d7cd85e59b99aa9cebbe8cf99f4e433100`; `runs/full/run_b320/final/45c496799170/result.json` sha256 `b184003db2391777f2b02c84fd695bb4cdb60b20d43e522ce930bb34ad0572d5`. All 57 sealed paths and hashes (that stage file and everything under `final/` outside `checkpoint/`) are in `runs/full/run_b320/seal/sealed_sha256.txt`, sha256 `5613a7f3b6c52e45acb7763a2c99318e0b9d2792d7038241d2d8d6c075f13da7`.

## A-072

| Field | Value |
| --- | --- |
| Date | 2026-11-26 |
| Issue IDs | 154 |
| Issue IDs v2 | None |
| Files | `docs/research/amendments.md`, `scripts/level18_console.py`, `scripts/level18_model_cards.py`, `scripts/level18_mlflow_log.py`, their tests, `reports/level_18/` |
| What changed | Records that Run B's leakage check 6b failure is the known A-051 simulator finding. Run B's card shows it as "FINDING — documented limitation (A-051)" with its numbers, never as a pass. Run B is then added to the console files and the account model card as the fallback model for banks with no complaint history. |
| Why | Run B's check 6b result is identical to Run A's: validation average precision 0.6378, 10.415 times the base rate, limit 2.0. Check 6b depends only on the datasets and a fixed seed, not on either account model. Check 6a passes for Run B. |
| Disclosure | Written on the owner's ruling after Run B's checks 5 and 6 had run (`runs/full/run_b320/leakage_checks_5_6.json`) and after Run B's test result was read. The recorded status stays `fail`; no threshold, model or result changes. All results are SIMULATED. |
| Results exist? | Yes: Run B checks 5a, 5b and 6a pass, 6b fails; check 4 recorded `pass` by the pilot. |

## A-073

| Field | Value |
| --- | --- |
| Date | 2026-11-27 |
| Issue IDs | None |
| Issue IDs v2 | None |
| Files | `docs/research/amendments.md`, `scripts/level18_console.py`, `scripts/level18_handoff.py`, their tests, `reports/level_18/`, `docs/handoff/level_18.md` |
| What changed | Corrects a reporting error: Run A's account model WAS scored on 120 stress worlds. The comparison console file now reads `result.json` key `stress.across_worlds` instead of `across_worlds.stress`, and the handoff reports those numbers. |
| Why | The console copied `across_worlds.stress`, which reads "not run: no stress worlds supplied", and not the separate `stress` block. XGBoost, median (p10 to p90): recall at capacity 0.636 (0.454 to 0.859), genuine restriction rate 0.0059 (0.0000 to 0.0118), time to detection 12.47 hours (0.51 to 33.23), measured yield 0.923 (0.763 to 1.000), re-weighted yield 0.216 (0.075 to 1.000). |
| Disclosure | Found after Run A's and Run B's results existed, while writing the pull request to `main`. Reporting only: no model, threshold, score, quality-gate threshold or serving decision changes. Run B was not scored on stress worlds (A-070), and that stays true. All results are SIMULATED. |
| Results exist? | Yes: `runs/full/run_a320_stage5c/final/45c496799170/result.json`, read only. |
