Both datasets are synthetic, so agreement with them shows plausibility, not realism. This report is recorded and never used to set or tune any world setting.

## Amount Quantiles Divided by Median

IBM amount shape evaluated on US Dollar (37.3% of genuine rows; mixed currencies cannot share one scale).

| Quantile | World (shape) | PaySim (shape) | Ratio (World / PaySim) | IBM (shape) | Ratio (World / IBM) |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.1335 | 0.0604 | 2.2110 | 0.0342 | 3.9049 |
| 0.25 | 0.2863 | 0.1793 | 1.5973 | 0.1777 | 1.6109 |
| 0.5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.75 | 3.7051 | 2.8017 | 1.3225 | 5.8675 | 0.6315 |
| 0.9 | 10.9521 | 4.9014 | 2.2345 | 39.0736 | 0.2803 |
| 0.99 | 252.5506 | 21.3029 | 11.8552 | 3452.4747 | 0.0732 |

## Payer Repeat Share

| Metric | World | PaySim | Ratio (World / PaySim) | IBM | Ratio (World / IBM) |
| --- | --- | --- | --- | --- | --- |
| Payer repeat share | 0.2526 | 0.0000 | inf | 0.8005 | 0.3156 |

## Payments Per Payer

| Metric | World | PaySim | Ratio (World / PaySim) | IBM | Ratio (World / IBM) |
| --- | --- | --- | --- | --- | --- |
| Median payments per payer | 47.0 | 1.0 | 47.0000 | 2.0 | 23.5000 |
| P90 payments per payer | 58.0 | 1.0 | 58.0000 | 29.0 | 2.0000 |

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
- Amount quantile 0.1 / median: World is 0.1335, IBM is 0.0342 (ratio 3.9049, outside [0.5, 2.0]).
- Amount quantile 0.9 / median: World is 10.9521, PaySim is 4.9014 (ratio 2.2345, outside [0.5, 2.0]).
- Amount quantile 0.9 / median: World is 10.9521, IBM is 39.0736 (ratio 0.2803, outside [0.5, 2.0]).
- Amount quantile 0.99 / median: World is 252.5506, PaySim is 21.3029 (ratio 11.8552, outside [0.5, 2.0]).
- Amount quantile 0.99 / median: World is 252.5506, IBM is 3452.4747 (ratio 0.0732, outside [0.5, 2.0]).
- Payer repeat share: World is 0.2526, PaySim is 0.0000 (ratio inf, outside [0.5, 2.0]).
- Payer repeat share: World is 0.2526, IBM is 0.8005 (ratio 0.3156, outside [0.5, 2.0]).
- Median payments per payer: World is 47.0, PaySim is 1.0 (ratio 47.0000, outside [0.5, 2.0]).
- Median payments per payer: World is 47.0, IBM is 2.0 (ratio 23.5000, outside [0.5, 2.0]).
- P90 payments per payer: World is 58.0, PaySim is 1.0 (ratio 58.0000, outside [0.5, 2.0]).

## External Datasets Status

IBM AML-Data: source: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml, file: HI-Small_Trans.csv, md5: 9667f3a6323e6ad88e699acf42db9157, licence: CDLA-Sharing-1.0.
