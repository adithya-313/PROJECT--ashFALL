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
| What changed | 15 verified findings added as a new table section; no existing row changed, research_status.yaml untouched. |
| Why | Task 2: verify the October 2026 findings before parameters depend on them. |
| Results exist? | No. |
