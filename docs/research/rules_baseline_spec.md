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

## Device rules: a design constraint (added 2026-10-10, review item R5)
The four rules above do not use device information, so there is no "new device"
rule today. If a device rule is added later (Level 17), it must follow this
constraint.
A "new device" means a change of hardware, a changed or swapped SIM, or a fresh
app install on a different phone. An operating system update on the same phone
is NOT a new device, and a rule must not fire on an update alone.
Why: operating system updates reach many ordinary people over days or weeks, so
a rule that fired on them would create a wave of false alarms on genuine
customers. The simulated world includes an OS update wave as a labelled stress
test (review item R2) to check this.
This section sets no numbers. Any numbers for a device rule will come from the
tuning rule above, like the other rules.
