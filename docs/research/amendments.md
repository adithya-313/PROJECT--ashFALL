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
