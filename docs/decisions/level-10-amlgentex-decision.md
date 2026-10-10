# Level 10 AMLGentex decision

**What this file is for**
It records what ashFALL decided to do with the open-source AMLGentex project, and why.
The evidence is in `docs/research/amlgentex_spike_verdict.md`.
Label: POLICY = our own choice.

## Decision [POLICY]
- Do not build on AMLGentex.
- Borrow its ideas: the multi-bank setup, the money-flow shapes, the alert-budget metric, pattern-based splits, and the isolated, centralised and federated regimes.
- Use it as an outside check in Level 28.

## Why
The spike verdict found:
- no devices or shared control of accounts;
- label noise that only flips or drops labels, with no delay;
- no holds or freezes;
- no adaptive fraudsters.

ashFALL's labels arrive late and biased, so this is a documented difference between the two.

## Licence handling
- The code is Apache-2.0.
- The licence of the generated data is unknown.
- Use locally generated data privately. Never commit generated data.
- Commit only the generation script, the configs, the seed and the pinned AMLGentex commit hash `4b9a69d`. Anyone can then regenerate the data.

## Open item
Question posted as https://github.com/aidotse/AMLGentex/issues/79 on 2026-10-10. No reply yet; answer will be recorded here.
The answer goes in `config/research_status.yaml` (finding `amlgentex_output_licence_and_release_date`).
