2. Research foundation and related work

Every design choice traces to a finding below. "Verify" items are re-checked against the primary source in Level 10 before anything depends on them.

### India's shared fraud infrastructure

| Finding | Source and date | Status | Used in |
| --- | --- | --- | --- |
| RBIH found rule-based mule detection gave high false positives and slow turnaround, and built MuleHunter.AI | RBIH and RBI statements, 2024–2025 | Established | Rules baseline |
| MuleHunter.AI used across roughly 23–31 banks | RTI Dec 2025 to media Sep 2026; counts conflict | Verify; quote as a dated range | Positioning |
| I4C Suspect Registry (from Sep 2024) shares suspect identifiers and layer-1 mule accounts with banks | I4C via media, Aug 2026 | Verify figures | Mock registry feed |
| DPIP Smart Registry live from 1 Sep 2026 with 8 banks; salted hashed identifiers; services Ingestion, Screening, Feedback, Revoke, Dispute; transaction scores in phase 2 | RBIH documentation and MediaNama, Sep 2026 | Verify on RBIH's documentation | Mock DPIP |
| NPCI pilot (about a dozen banks): each bank runs NPCI's model locally; NPCI correlates scores across banks | Economic Times and aggregators, 2025–2026 | Verify | Federated mode |
| NPCI with NVIDIA on real-time mule detection; HDFC Bank's in-house streaming platform for mule detection | NVIDIA conference session; QuestDB case study, 2026 | Reported | Architecture choice (streaming) |
| 1930/CFCFRMS hold notices; hold, lien or debit restriction; blanket freeze can be narrowed to a lien on the disputed amount | Legal explainers, 2026 | Verify | Legal actions, tracer |
| RBI draft (2026): temporary debit holds up to 60 days | Economic Times, Sep 2026 | Draft | Action catalogue |
| Banks may have 30–60 seconds to intervene | ET BFSI, 2026 | Reported | Latency budget |

### Mule behaviour and signals

| Finding | Source and date | Status | Used in |
| --- | --- | --- | --- |
| Rapid inbound from unrelated payers then fast dispersal, balances repeatedly near zero; large credits into new or dormant accounts then similar debits soon after | UK FCA multi-firm review, 2023 (updated 2025) | Established | Features, world |
| Absence of routine activity (salary, bills, regular transfers) in mule accounts | Indian practitioner guidance and BCG, 2026 | Reported | Feature |
| Shared devices across accounts; account clusters opened from one device; SIM swaps; emulator and remote-access use | Device-intelligence vendors, 2026 | Reported (vendor) | Shared-control features and world |
| XHelper-style platforms: tens of thousands of mule accounts, OTP forwarding or linked-number replacement, payouts within about 10 minutes, merchant accounts preferred, USDT cash-out | CloudSEK, Mar 2024 | Established for that operation | World building blocks |
| Only 37% of offboarded mule accounts reported to the UK's shared fraud database (2022–23) | FCA, via secondary source | Verify on the FCA page | Bank-to-registry reporting share, not victim complaints (corrected in Level 11, R1) |
| A live mule system's escalations were 89% confirmed ("yield"), up from 61% under rules | arXiv paper, Jul 2026 | Verify | Yield-rate metric |
| Industry claims 85–95% of AML rule alerts are false positives | Vendor reports, 2024–2026 | Context only | Rules baseline narrative |
| Typical amounts, hops, account lifetimes, account-type shares, mule-specific false-positive rates | — | Not found publicly | Ranges and many worlds |

### International approaches

| Finding | Source | Used in |
| --- | --- | --- |
| UK MITS traces stolen money across banks with pooled payment data; over 90% of UK current accounts covered | Vocalink and Pay.UK | Network-tracing design |
| Swift's 13-bank federated experiments roughly doubled detection versus single banks (synthetic data) | Swift, 2024–2025 | Federated mode |
| Netherlands' TMNL consortium restructured after legal constraints | 2024 | Limitations |

### Related open and commercial projects

| Project | What it is | What ashFALL borrows | What it lacks for ashFALL's purpose |
| --- | --- | --- | --- |
| **PixSim** | Open simulator of Pix fraud, recovery and interdiction under analyst capacity (arXiv 2609.30684, Sep 2026) | Its discipline: every parameter sourced, calibrated, or registered as an assumption | Brazil, not UPI; code location to verify |
| **AMLGentex** (AI Sweden) | Apache-2.0 Python generator and benchmark, loosely based on Swish | Pattern files, alert-budget metrics, pattern-based splits, centralised/isolated/federated regimes; also a third outside dataset | No devices or shared control, no delayed or noisy labels, no holds, no adaptive fraudsters |
| **IBM AMLSim** | Apache-2.0 multi-agent laundering generator | Pattern-file idea | Old dependencies; little maintenance |
| **Truman** | LLM-agent laundering simulator (AAMAS 2025) | Agent roles (broker, facilitator, customers) | No public code |
| **Napier AI Agentic Test Simulator** | Commercial, closed beta | Evolving agents as a test method | Closed; results stated as indicative |

No open UPI-specific mule simulator was found, and no simulator found models label delay or under-reporting explicitly.

### Datasets

| Dataset | Type | Licence | Role |
| --- | --- | --- | --- |
| IBM AML-Data (AMLworld) | Synthetic, calibrated, multi-bank | CDLA-Sharing-1.0 | Outside check |
| SAML-D | Synthetic, 28 typologies | CC BY-NC-SA 4.0 (non-commercial) | Outside check |
| AMLGentex output | Synthetic, configurable | Apache-2.0 | Outside check |
| Elliptic (original) | Real Bitcoin graph | Check terms | Only if outside checks prove informative |
| PaySim | Synthetic; its description warns balance columns must not be used | — | Model v1 audit only |

### Trust standards for open-source financial software

FINOS (financial-services open source), the OpenSSF security baseline (OSPS), SLSA-style signed build provenance, SBOMs in CycloneDX or SPDX, and India's CERT-In SBOM guidelines (2025) anchor the trust design in Section 9. Offline operation and privacy checks on shared results were not found as published practice in fraud tools, so ashFALL documents its own approach carefully.

### Competitive landscape

Signzy, BioCatch, Feedzai, NICE Actimize, Outseer and mFilterIt sell mule detection in India, alongside MuleHunter.AI, DPIP and NPCI's pilot. This is why ashFALL is an open engine and testbed, not a competing product.
