Both datasets are synthetic, so agreement with them shows plausibility, not realism. This report is recorded and never used to set or tune any world setting.

## Amount Quantiles Divided by Median

| Quantile | World (shape) | PaySim (shape) | Ratio (World / PaySim) |
| --- | --- | --- | --- |
| 0.1 | 0.1335 | 0.0604 | 2.2110 |
| 0.25 | 0.2863 | 0.1793 | 1.5973 |
| 0.5 | 1.0000 | 1.0000 | 1.0000 |
| 0.75 | 3.7051 | 2.8017 | 1.3225 |
| 0.9 | 10.9521 | 4.9014 | 2.2345 |
| 0.99 | 252.5506 | 21.3029 | 11.8552 |

## Payer Repeat Share

| Metric | World | PaySim | Ratio (World / PaySim) |
| --- | --- | --- | --- |
| Payer repeat share | 0.2526 | 0.0000 | inf |

## Payments Per Payer

| Metric | World | PaySim | Ratio (World / PaySim) |
| --- | --- | --- | --- |
| Median payments per payer | 47.0 | 1.0 | 47.0000 |
| P90 payments per payer | 58.0 | 1.0 | 58.0000 |

## Payment Types

| Type | PaySim Share | World Share | Comparison Note |
| --- | --- | --- | --- |
| CASH_IN | 0.2210 | not comparable | the world has no PaySim types |
| CASH_OUT | 0.3497 | not comparable | the world has no PaySim types |
| DEBIT | 0.0065 | not comparable | the world has no PaySim types |
| PAYMENT | 0.3398 | not comparable | the world has no PaySim types |
| TRANSFER | 0.0829 | not comparable | the world has no PaySim types |

## Differences

- Amount quantile 0.1 / median: World is 0.1335, PaySim is 0.0604 (ratio 2.2110, outside [0.5, 2.0]).
- Amount quantile 0.9 / median: World is 10.9521, PaySim is 4.9014 (ratio 2.2345, outside [0.5, 2.0]).
- Amount quantile 0.99 / median: World is 252.5506, PaySim is 21.3029 (ratio 11.8552, outside [0.5, 2.0]).
- Payer repeat share: World is 0.2526, PaySim is 0.0000 (ratio inf, outside [0.5, 2.0]).
- Median payments per payer: World is 47.0, PaySim is 1.0 (ratio 47.0000, outside [0.5, 2.0]).
- P90 payments per payer: World is 58.0, PaySim is 1.0 (ratio 58.0000, outside [0.5, 2.0]).

## External Datasets Status

IBM AML-Data: pending (not downloaded); register 12-16 records it.
