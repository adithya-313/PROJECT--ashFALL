# Mule typologies

**What this file is for**
It lists the ways money mule accounts behave, with where each claim came from and how far it was checked.
Each row also gives the design name of the building block that the simulated worlds will use.

Status words: **confirmed** = a source says it; **changed** = the source says something different from the first plan;
**reported** = seen in a secondary or vendor source; **unverified** = not found in a source yet.
"Hand checked" says whether the author opened the page.

| # | Pattern | What it looks like | Source and date | Status | Hand checked | Building block |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Fast pass-through into a new or dormant account | "high value payments into a new or (previously) dormant account and similar amounts subsequently debiting the account" (quote found by the research tool) | FCA multi-firm review, 19 Oct 2023 (updated 3 Dec 2025), https://www.fca.org.uk/publications/multi-firm-reviews/proceeds-fraud-detecting-preventing-money-mules | confirmed | no | `pass_through_account` |
| 2 | Cash-out concentrated at the second to fifth mule account, with the peak at the second | Cashing out happens early in the chain of mule accounts. UK data, 140 cases. | FCA, 23 Sep 2026, https://www.fca.org.uk/publications/multi-firm-reviews/money-mules-activity-cashing-out-findings | confirmed | no | `mule_chain_depth` |
| 3 | Many unrelated payers into one account, then fast dispersal | The FCA does not use this exact combined wording, so this is **changed**. A UK National Crime Agency "SARs in Action" issue 22 lists "Fraud transactions from multiple source accounts sent into one account" (reported, found by the research tool). | FCA pages; UK National Crime Agency, SARs in Action, issue 22 | changed | no | `fan_in_account` |
| 4 | Balances repeatedly near zero | No wording was found in the FCA pages. | none found | unverified | no | `low_balance_after_outflow` (design name; needs a registered assumption in the world ranges later) |
| 5 | No routine activity (salary, bills, regular transfers) | Indian practitioner guidance and BCG, 2026. No quantified figure was found. | Indian practitioner guidance and BCG, 2026 | reported | no | `no_routine_activity` |
| 6 | Shared devices, device clusters, SIM swaps, emulators, remote access | Vendor sources only. No quantified figure was found. | Device-intelligence vendors | reported | no | `shared_device_cluster` |
| 7 | XHelper-style app-run mule network | See the notes below the table. | CloudSEK, 5 Mar 2024, https://www.cloudsek.com/blog/shadow-banking-in-your-pocket-exposing-android-app-used-by-money-mules | changed | yes, for the points below | `app_run_payout_window` |

### Notes on row 7 (hand checked: yes)
- Payout orders must finish within a strictly enforced 10-minute window.
- USDT is used.
- The network is described as "exceeding hundreds of thousands" of mule accounts. This is the whole scheme, not one gang.
- Plan Section 2 said "tens of thousands" of accounts and "merchant accounts". The page says hundreds of thousands and a preference for corporate accounts. That is why the status is **changed**.
- Press reports of other account counts conflict.

## Row 8. Held-out tactics (project design, NOT sourced)

These four tactics are our own design. No source is claimed for them.
They are used only in test and stress worlds (see `config/evaluation.yaml`).

| Tactic | Which signal family it attacks |
| --- | --- |
| `slow_drip` | amount and speed |
| `device_rotation` | shared-device signals |
| `pseudo_merchant_camouflage` | account-type signals |
| `fake_routine_life` | the no-routine signal |

`basic_merchant_mule` stays in training.

## October 2026 verification (Level 11)
These findings come from the October 2026 world review. Each was checked against the best source found; nothing here is hand-checked yet.

| Finding id | Review issue | Status | Source link | Source date | What was confirmed |
| --- | --- | --- | --- | --- | --- |
| bengaluru_rented_mule_suppliers | G1, G5 | reported | https://www.deccanherald.com/india/karnataka/bengaluru/bengaluru-four-people-held-for-opening-mule-bank-accounts-supplying-them-to-cyber-criminals-3611364 | Jul 2025 | 357+ accounts and four arrests; the Rs 4,000 per Rs 1 lakh and 20-30% shares not confirmed |
| jaipur_gang_many_accounts | G4 | reported | https://the420.in/jaipur-mule-accounts-students-cyber-fraud-arrest/ | 21 Jul 2026 | Gang spread proceeds across accounts; 13 complaints; no account count |
| indore_51_complaints | G3, E3 | reported | https://www.freepressjournal.in/indore/15-crore-51-complaints-15-states-indore-mba-student-lands-in-cyber-crime-net | 2 Sep 2026 | One account, 51 complaints, 15 states, about Rs 1.5 crore |
| kerala_scam_type_shares | G6 | reported | https://www.deccanherald.com/india/keralam/operation-cy-hunt-police-sting-reveals-newest-recruits-of-cybercrime-rackets-in-kerala-3783625 | Nov 2025 | Trading 34.8% and job 33% of cyber fraud cases (shares of cases, not money) |
| telangana_12_minute_report_25_minute_hold | L1, N3 | changed | https://www.thestatesman.com/technology/quick-action-by-telangana-cyber-security-bureau-saves-citizens-rs-1-crore-1503295631.html | 29 Apr 2024 | Reported in 12 minutes, held in 25; Rs 1,00,10,005; dated 2024, not 2026 |
| maharashtra_report_delay_3_to_15_days | L1 | confirmed | https://newsonair.gov.in/maharashtra-cm-devendra-fadnavis-urges-citizens-to-report-cyber-fraud/ | 26 Aug 2026 | CM's remark that 3-15 day delays hurt recovery; not a measured statistic |
| gujarat_mule_hunt_913 | L5 | reported | https://www.newindianexpress.com/states/gujarat/2026/Jun/01/gujarat-police-busts-cyber-fraud-network-exposes-rs-2289-crore-scamthrough-operation-mule-hunt | 1 Jun 2026 | 913 mule accounts, 565 FIRs, 638 arrests; type of action not stated |
| lok_sabha_2023_cyber_fraud_figures | L3, E2 | confirmed | https://www.pib.gov.in/PressReleasePage.aspx?PRID=2003158 | 6 Feb 2024 | 1,128,265 complaints; Rs 7,488.639 crore reported; Rs 921.5956 crore lien (about 12.3%, derived) |
| ncrp_2025_complaints | E1, E2 | confirmed | https://pib.gov.in/PressReleasePage.aspx?PRID=2226441&reg=3&lang=2 | Feb 2026 | 24,02,579 complaints; Rs 22,495 crore reported |
| upi_volume_2025_monthly | E1 | reported | https://www.npci.org.in/product/upi/product-statistics | 2025 data | About 228,281.85 million payments, about 625.43 million per day; monthly figures from secondary copies |
| complaints_per_upi_payment_derived | E1 | reported | https://pib.gov.in/PressReleasePage.aspx?PRID=2226441&reg=3&lang=2 | derived | About 1 complaint per 95,000 payments; upper-bound proxy (all financial fraud, not only UPI) |
| suspect_registry_lok_sabha_2026 | N2 | confirmed | https://sansad.in/getFile/lsapps/loksabhaquestions/annex/188/AU3751_Uf05QB.pdf?source=lsapps | 11 Aug 2026 | As of 30 Jun 2026: 30.48 lakh identifiers received, 32.08 lakh layer-1 mule accounts shared; bank count not stated |
| mobile_connections_handsets_dot | N1 | changed | https://www.pib.gov.in/PressReleasePage.aspx?PRID=2053530 | 10 Sep 2024 | Over 1 crore connections disconnected, 2.27 lakh handsets blocked; DoT release, cumulative |
| odisha_recovery_2023_2025 | L3 | reported | https://the420.in/odisha-cyberfraud-loss-recovery/ | 29 Sep 2026 | Rs 370.72 crore lost, Rs 9.97 crore recovered in 2023-2025 (about 2.7%, derived) |
| moneylife_1930_access | N3 | changed | https://www.moneylife.in/article/fraud-alert-precautions-are-always-better-to-become-a-victim-of-cyber-frauds/72668.html | 24 Nov 2023 | One test: 15 calls, 2 answered; anecdotal, no wait-time statistics |

Status words: confirmed = official source matches; reported = news source only; changed = the source differs from what the plan assumed.
The 37% finding was corrected separately (see A-002 in amendments.md).

## Searched but not found
- mule account type shares for India
- dormant or new account share
- mule-specific false positive rates
- shared-device and emulator shares
- share of Indian mules not reported to a registry
- analyst review capacity
- published fraud-check latency budgets
