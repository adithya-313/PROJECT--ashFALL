# Rules baseline spec

**What this file is for**
It describes the simple checklist that ashFALL is compared against. The checklist is built in Level 17.
It gives rule names and plain-English meanings only, with no numbers.
The numbers are set later, by the tuning rule at the end of this file.

## Where the idea comes from
RBIH documentation says rule-based detection gives "high false positive rates". Status: **confirmed** (quote found by the research tool).
The "slow turnaround" part of that claim is **unverified**.

## The four rules (a design choice, label: policy)

| Rule | Plain English | Parameter names (no numbers yet) |
| --- | --- | --- |
| R1 | A large credit is followed by an outflow of most of it within a short time window. | `large_credit_amount`, `outflow_share`, `time_window` |
| R2 | A large credit arrives in a new account. | `large_credit_amount`, `new_account_age` |
| R3 | Many different payers send money into one account within a day. | `payer_count`, `one_day_window` |
| R4 | The balance is close to zero after the outflow. | `near_zero_balance` |

## Thresholds are NOT set here
The thresholds are tuned on validation worlds.
They are tuned so that the rules baseline and the ML system have the same genuine-restriction rate.
This follows `config/evaluation.yaml` (`fair_comparison`).
