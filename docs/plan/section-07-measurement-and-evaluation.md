7. Measurement and AI evaluation

Success is defined here and committed to `config/evaluation.yaml` in Level 10, before any model exists.

### Engine metrics

| Metric | Definition | Use case |
| --- | --- | --- |
| **Mule recall at capacity** | Share of true mules flagged when only the top N accounts per day can be reviewed | 2 |
| **Yield rate** | Share of escalated cases that turn out to be mules (the term banks use) | 2 |
| **Hidden mule recall** | Recall for deeper mules never reported | 2 |
| **Time to detection** | From a mule's first illicit credit to its first flag (median, 90th percentile) | 2 |
| **Money protected before payment** | Victim money held or abandoned after a pre-payment warning or hold | 1 |
| **Money recovered** | Victim money still held when tracing finished, as a share of stolen money | 2 |
| **Genuine harm** | Genuine accounts restricted and restriction-days (weighted for vulnerable personas); warnings shown to genuine payers | 1, 2 |
| **Analyst load** | Cases per day against capacity | 2 |
| **Decision latency** | p50 and p99 of `POST /v1/decisions` | 1 |
| **Tracing time** | Seconds from complaint to ranked holds | 2 |
| **Evasion cost** (Tier 2) | Extra time, accounts and devices a gang needed to evade, and its profit drop | Both |

### Comparisons

On the same worlds and seeds: the rules baseline; the anomaly detector alone; ashFALL with one bank's data; ashFALL plus registry intelligence; ashFALL plus banks learning together (Tier 2); and the money-flow pipeline on three outside datasets (IBM AML-Data, SAML-D, AMLGentex output).

### Reporting

- Medians with spread (10th–90th percentile) across test worlds, never a single world.
- Sensitivity maps against key unknowns, and a coverage map of what was tested.
- Training, test and stress worlds kept apart; held-out tactic families only in test and stress.
- Every number linked to a run manifest (code version, world files, seeds, model version).

### Success criteria (policy, committed before results)

| Claim | Criterion |
| --- | --- |
| ML beats rules | At equal genuine harm, higher median recall at capacity and yield rate than rules, in test worlds |
| Early warning works | Shorter median time to detection than rules |
| Tracing helps | More money recovered than first-layer holds alone, across complaint delays |
| Least-harm actions help | Fewer genuine restriction-days than freeze-only at equal money protected |
| Shared intelligence helps | Registry raises recall without raising genuine harm |
| Real-time is feasible | Decision p99 within the committed budget |

Failed criteria are reported as failed, with the investigation. Thresholds are policy choices, not predictions.

### AI evaluation

Every AI component is judged by programmatic checks against stored data and hidden truth. No AI component is graded by another AI. You personally review a small sample of each, so the numbers can be trusted.

| Component | What is measured | How | Bar |
| --- | --- | --- | --- |
| **Copilot** (Level 25) | Hallucination rate; grounding (every cited number matches the feature log or trace); latency; cost per explanation | Automatic value-matching on 200 explanations; your review of 20 | Zero invented values in the reviewed sample; all mismatches logged |
| **Retrieval** (Level 26) | Recall@5 and mean reciprocal rank on about 50 labelled questions; time-awareness violations; poisoning attack success | Evaluation harness in CI; property tests; poisoning suite | Retrieval gate passes; zero time-awareness violations; lower attack success with defences |
| **Investigation agent** (Level 27) | Triage lift over the score threshold; grounding rate; cost and time per investigation; impact on decision latency; injection steering rate; retrieval ablation | 50 cases with hidden truth; tool-call logs; load test with the agent running | Lift reported either way; must beat the threshold to be called useful (ground rule 9) |
| **LLM strategist** (Tier 2) | Valid-strategy rate; detection drop caused; evasion cost; share of ideas outside the world | Simulator validation and arms-race runs | Reported as measured |

The combined AI evaluation report is produced in Level 28 and summarised on the showcase page.
