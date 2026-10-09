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

## Searched but not found
- mule account type shares for India
- dormant or new account share
- mule-specific false positive rates
- shared-device and emulator shares
- share of Indian mules not reported to a registry
- analyst review capacity
- published fraud-check latency budgets
