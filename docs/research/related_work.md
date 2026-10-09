# Related work

**What this file is for**
It records the open and commercial projects ashFALL learns from, and how far each claim was checked.
For each project it says what is borrowed, how ashFALL differs, and the status.
Status words: confirmed, changed, reported, unverified.

## PixSim
- What it is: a simulator of Pix fraud (arXiv 2609.30684, 25 Sep 2026). Code: https://github.com/BashirZeimarani/pixsim_public (Apache-2.0).
- Borrow: every parameter is sourced, calibrated, or registered as an assumption.
- Differs: it is for Brazil's Pix, not UPI.
- Status: **confirmed**.

## AMLGentex
- What it is: a generator and benchmark, https://github.com/aidotse/AMLGentex. The code is Apache-2.0.
- Borrow: pattern files, alert-budget metrics, three regimes (centralised, isolated, federated).
- Differs: device tracking, label delay, holds and adaptive fraudsters were not found in the text read.
- Status: code licence **confirmed**. The output dataset licence and the latest release date are **unverified**.
- The spike gave the verdict: borrow ideas and use as an outside check.
- Verdict line: `Verdict: borrow ideas and use as an outside check`. Evidence: `docs/research/amlgentex_spike_verdict.md`.
- The decision and its reasons are in `docs/decisions/level-10-amlgentex-decision.md`.

## IBM AMLSim, Truman and Napier AI Agentic Test Simulator
- IBM AMLSim, Truman (AAMAS 2025) and the Napier AI Agentic Test Simulator come from plan Section 2 only.
- The research tool could not confirm them.
- Status: **unverified**.

## NPCI with NVIDIA
- A collaboration announced 18 Feb 2026 for a payments AI foundation model.
- No primary source describes mule detection.
- Status: **changed**.

## HDFC Bank with QuestDB
- A QuestDB case study, 17 Jul 2026. It is a vendor account of HDFC's streaming platform.
- Status: **reported**.

## Yield paper
- arXiv 2607.17586 (20 Jul 2026): yield 89% versus 61% under the incumbent rules.
- Alert volume also rose from 211 to 302 a month.
- Status: **confirmed**.

## Datasets

| Dataset | Licence or terms | Status |
| --- | --- | --- |
| IBM AML-Data | CDLA-Sharing-1.0 | confirmed |
| SAML-D | CC BY-NC-SA 4.0 (non-commercial) | confirmed |
| AMLGentex output | licence not found | unverified |
| Elliptic | terms not checked | not checked |
| PaySim | used for the v1 audit only | not applicable |
