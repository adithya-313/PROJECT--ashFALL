# Stress-Test Scenario Library

A collection of labelled stress-test configuration layers for ashFALL simulation experiments. Every stress test exaggerates a specific real-world event or operational shock and is explicitly labelled with its real event source and exaggeration ratio.

## Stress-Test Scenarios

| Name | Real Event | Exaggeration | Takes Effect |
| :--- | :--- | :--- | :--- |
| `big-festival` | Diwali 2025: the Diwali-eve record of 740 to 754 million UPI transactions was about 6 to 9 percent above that October's daily average of 695 million (NPCI-based reports, October 2025). | Festival uplift of 50 to 60 percent instead of the normal 5 to 25 percent, several times the reported 6 to 9 percent. | Immediately (Level 11) |
| `os-update-wave` | OS updates spread gradually: iOS adoption reached about two-thirds of devices months after release (global, February 2026); Android updates roll out in waves (Android update guidance). | Two or three waves where 90 to 100 percent of eligible phones update within 3 to 7 days, instead of one or two waves reaching 30 to 70 percent over 2 to 6 weeks. | Immediately (Level 11) |
| `bank-outage-long` | May 2024 saw 31 bank downtime instances totalling over 47 hours, about 1.5 hours each (NPCI data via Economic Times, June 2024). | Each bank outage lasts 8 to 12 hours instead of 1 to 2 hours; success rate still dips partly (partial degradation), it does not fall to zero. | Immediately (Level 11) |
| `switch-outage` | NPCI-side incidents are rare and partial: 95 to 180 minutes, with success rates dropping to about 50 to 80 percent (NPCI uptime dashboard, March to April 2025). | At least one incident a month where NPCI routes almost nothing (0 to 0.01 percent success) for 3 to 5 hours. | Immediately (Level 11) |
| `registry-outage` | No published outage of the shared fraud registry (mock DPIP and I4C) was found; this is a what-if. | The registry gives ashFALL almost nothing: at most 1 percent chance a bank takes part, and entries arrive after about 120 days. Payments still flow; only shared intelligence is lost. | Level 13 |
| `complaint-backlog` | No published series on complaint backlogs was found; this is a what-if for complaint intake piling up. | Scam complaints arrive after 40 to 60 days instead of 3 to 15, and unauthorised-debit complaints after 2 to 3 days instead of minutes to hours. | Level 12 |

## How to Use a Stress Test

Apply any stress test overlay layer using the `--layer` argument of `world.fast_mode`:

```powershell
python -B -m world.fast_mode --preset tiny --seed 101 --layer scenarios/stress/big-festival.yaml --out runs/stress-check
```

Stress tests are always labelled: when applied, `fast_mode` outputs an audit line with `STRESS TEST <name>: <exaggeration>`, and indicates if the stress test takes effect from a future simulation level.

## Deferred Scenarios

- `new-bank`: Deferred because it needs world-builder code enhancements (three banks are currently built into the simulation engine) and the project plan marks it as optional.
