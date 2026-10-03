# Fraud Pattern Research — Real-World Basis for Adversarial Injection

**Purpose:** This document records the real-world fraud behaviours that the
adversarial injection (`scripts/inject_adversarial.py`) is designed to mimic.
It exists so the injected patterns are traceable to documented reality, not
invented. Every injection design choice below cites the behaviour it models.

**Scope:** Two patterns injected into the PaySim corpus — `account_takeover`
and `mule_account`. (`merchant_collusion` was moved to the e-commerce stream
and is not injected here.)

---

## 1. Why we researched this

PaySim's native fraud has a single, narrow behaviour: a fraudster drains an
account to zero in one TRANSFER or CASH_OUT. Measured on our data, 97.9% of
native fraud drains the origin to zero, 99.8% satisfies the exact arithmetic
`oldbalanceOrg - amount = newbalanceOrig`, and fraud occurs only in TRANSFER
and CASH_OUT.

That is one *type* of fraud. The fraud-detection literature splits fraud into
two families:

- **Signature-based fraud** (e.g. account takeover): individual transactions
  are themselves anomalous.
- **Behavioural financial crime** (e.g. structuring, layering, mule networks):
  no single transaction is anomalous — only the relationship among
  transactions, accounts, or time windows reveals the attack.

PaySim natively models only the first family. Our engineered patterns add
signals from both families so the trained model learns something closer to
real fraud, rather than only PaySim's drain-to-zero signature.

---

## 2. Account Takeover (ATO) — real-world signals

A fraudster gains control of a legitimate account and then behaves unlike the
real owner. Detection relies on a *sequence* of identity, device, location,
and transaction evidence rather than any single event.

Observable signals practitioners use:

- **New device / browser.** Logins from a device or browser the user has not
  used before.
- **Impossible travel (geo-velocity).** A rapid spike in logins from multiple
  IPs or sudden geolocation jumps — e.g. logging in from one city then another
  far away minutes later — signals credential stuffing or remote access.
- **Transaction amount anomaly.** A sudden large purchase or transfer that is
  abnormally large *relative to the account's own history*; purchase velocity
  often spikes right after a password reset.
- **Beneficiary / profile changes.** New payees, contact-detail changes, and
  withdrawal behaviour that deviate from the established user profile.

**Core shape:** new device + impossible travel + a transaction much larger
than the account's normal, in a tight window after access is gained.

---

## 3. Mule accounts — real-world signals

A mule account receives illicit funds from multiple sources and forwards them
onward, adding distance between the crime and the money. Its function creates
its structure: gather widely, transit quickly, disperse narrowly.

Observable signals, with the concrete thresholds practitioners use:

- **Structuring (many small inbound).** Multiple deposits deliberately sized
  just under the US Bank Secrecy Act reporting threshold of **$10,000** — e.g.
  $9,900, $9,700, $9,500 — in a short window. A representative rule: 3+ deposits
  each between $2,000 and $10,000, aggregating to $15,000+.
- **Rapid pass-through (large outbound).** An outbound transfer within ~48
  hours of an inbound one, where the outbound is **at least ~80%** of the
  inbound amount, leaving the resting balance near zero.
- **Dormant-then-active.** Accounts that jump from dormant to high activity
  after 60+ days of inactivity. (One documented case: dormant 98 days, then
  funds from three sources dispersed through five channels in under an hour.)
- **Many distinct sources.** Five or more distinct sources of funds in seven
  days is a practitioner velocity threshold.

**Core shape:** many small inbound (near/under $10k) → rapid large outbound
(~80%+ of inflow) → balance returns to near zero.

---

## 4. How PaySim native fraud differs

PaySim's built-in fraud is signature-based only: a single transaction that
empties an account via TRANSFER or CASH_OUT. It has no concept of the
multi-transaction, time-windowed, network behaviour that defines mule activity,
and no device/geo/session signals that define account takeover. Our injected
patterns add those signals on top of the native data so the corpus represents
both fraud families rather than one.

---

## 5. Mapping real signals to PaySim columns

| Real-world signal | Pattern | PaySim column | Injected value |
|---|---|---|---|
| New device | ATO | `new_device_flag` | `True` |
| Impossible travel | ATO | `geo_velocity_spike` | `True` |
| Amount >> account norm | ATO | `amount` vs `account_avg_amount` | amount set well above the account average |
| Many small inbound (structuring) | Mule | `many_small_inbound_count` | high count (structuring signature) |
| Large outbound | Mule | `large_outbound_amount` | large relative to `account_avg_amount` |
| Drains to near zero | Both | balance columns | outbound empties the origin account (with noise) |

Injected rows set **both** the engineered behavioural signals **and** a
realistic balance drain. The origin account is nearly emptied on the fraudulent
transaction (a small random residual remains, and the amount/balance arithmetic
is deliberately imperfect) so the injected fraud resembles real cash-out and
pass-through behaviour without reproducing PaySim's perfectly exact drain math.
Injected fraud is drawn only from TRANSFER and CASH_OUT rows, the money-out
types where draining an account is physically meaningful.

---

## 6. Honest limitations — what PaySim cannot represent

The strongest real-world signals for both patterns are **graph and behavioural**,
and PaySim's flat, per-row tabular structure cannot hold them:

- **Mule network topology** — shared counterparties, community structure, and
  convergence on a downstream beneficiary — is invisible at the single-account
  level and only appears across accounts in a transaction graph.
- **ATO behavioural biometrics** — keystroke dynamics, mouse movement, session
  timing — are session-level signals absent from transaction records.
- **Temporal windows** — dormancy-then-activation and 48-hour pass-through
  timing require per-account transaction sequences, which our injection
  approximates with static per-row flags rather than true time series.

These are not defects to hide. The project roadmap adds the graph layer later
(the GNN at Level 14, which replaces the `query_fraud_ring` stub), which is the
appropriate place for the network signals this tabular injection cannot carry.
Until then, injected patterns encode the *per-account transactional* signals
faithfully and omit the *network/temporal* ones explicitly.

---

## 7. Sources

**Regulatory / official:**
- NCUA / FinCEN — SAR Requirements FAQ ($10,000 CTR, structuring definition): https://ncua.gov/regulation-supervision/regulatory-compliance-resources/bank-secrecy-act-resources/frequently-asked-questions-regarding-suspicious-activity-reporting
- FCA (UK) — Firms' use of the National Fraud Database & money mule detection tools: https://www.fca.org.uk/publications/multi-firm-reviews/firms-use-national-fraud-database-money-mule-account-detection-tools

**Practitioner / industry:**
- Adaptive Security — Account Takeover Detection: https://www.adaptivesecurity.com/blog/account-takeover-detection
- TrustDecision — Guide to Account Takeover Fraud: https://trustdecision.com/articles/from-detection-to-prevention-comprehensive-guide-to-account-takeover-fraud
- Veridas — Account Takeover: Fraud, Detection & Prevention: https://veridas.com/en/account-takeover/
- Group-IB — Account Takeover Fraud: https://www.group-ib.com/resources/knowledge-hub/account-takeover/
- Signifyd — What is Account Takeover Fraud: https://www.signifyd.com/blog/account-takeover-fraud/
- SEON — Account Takeover Risk Assessment: https://seon.io/resources/account-takeover-risk-assessment/
- Sanction Scanner — Money Mule Detection: https://www.sanctionscanner.com/blog/money-mule-detection-how-to-identify-and-stop-mule-accounts-1369
- Sanction Scanner — Transaction Monitoring Rules & Scenarios: https://www.sanctionscanner.com/blog/transaction-monitoring-rules-and-scenarios-a-practitioners-guide-to-effective-detection-logic-1371
- Flagright — Role of Money Mules in Money Laundering: https://www.flagright.com/post/uncovering-the-role-of-money-mules-in-money-laundering
- Bureau.id — Money Mule Detection for Banks and Fintechs: https://bureau.id/resources/blog/money-mule-detection
- Finextra — Money Mule Accounts: Recruitment to Detection: https://www.finextra.com/blogposting/31397/money-mule-accounts-from-recruitment-to-detection
- didit.me — Suspicious Financial Activity Red Flags: https://didit.me/blog/suspicious-financial-activity-red-flags-de/
- didit.me — Velocity Rules & Structuring Detection: https://didit.me/blog/velocity-rules-structuring-detection/
- InnReg — Suspicious Activity Reports (SARs) Guide: https://www.innreg.com/blog/suspicious-activity-reports-guide
- Sphinx — What Is a Mule Account: https://sphinxhq.com/blog-posts/what-is-a-mule-account-detection-investigation
- Tranche Two Consultants — Suspicious Transactions (AUSTRAC example): https://tranchetwoconsultants.com/what-is-suspicious-transaction-in-money-laundering/
- Clari5 — Mule Accounts in Indonesia (dormancy-to-activation): https://www.clari5.com/indonesia-mule-accounts-dormancy-activation/

**Academic (peer-reviewed / preprint):**
- arXiv 2606.17555 — An AI Security Agent for Banking (fraud vs behavioural-crime families): https://arxiv.org/pdf/2606.17555
- arXiv 2509.11595 — AMLNet (structuring / layering / integration generation): https://arxiv.org/pdf/2509.11595

*Note: this document paraphrases and synthesizes the cited sources in our own
words; it contains no verbatim reproduction of source text.*
