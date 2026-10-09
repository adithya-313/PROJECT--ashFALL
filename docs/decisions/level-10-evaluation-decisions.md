# Level 10 evaluation decisions

Status: committed before any model for the new engine is trained.
Author: adithya-313. Decisions reviewed with an engineer's advice.
Label key: ASSUMPTION = no public source, our best guess.
           POLICY = our own choice about how to judge results.

## D1. Review capacity  [ASSUMPTION]
- Sweep capacity from 0.05% to 1% of active accounts per day and report a
  recall-at-capacity curve.
- Headline point: 0.25% of active accounts per day.
- Why: real bank review capacity is not public. A curve means the result
  does not depend on one guess.
- Changed from: an earlier single value of 0.5%.
- Note: Level 10 only writes this rule. The curve is drawn in later levels,
  when models and worlds exist.

## D2. Fair comparison  [POLICY]
- Match operating points: tune each system's threshold on validation worlds
  so both have the same genuine-restriction rate, then compare recall.
- Main point 0.5% of genuine accounts restricted; extra points 0.25% and 1%.
- Also show the full trade-off curve.
- Why: a +/-1 point tolerance still allows unequal harm. Matching removes
  the doubt. Tuning on validation worlds, never test worlds, avoids peeking.
- Changed from: an earlier "within 1 percentage point" rule.

## D3. What counts as "ML beats rules"  [POLICY]
All three must hold, in test worlds:
1. Median recall difference of at least 5 percentage points.
2. ML beats rules in at least 70% of test worlds.
3. The 95% bootstrap confidence interval of the difference (resampling
   across worlds) excludes zero.
- Why: each condition blocks a different way of fooling ourselves (small
  gap, gap concentrated in a few worlds, gap due to luck).
- Minimum number of test worlds: 20 [ASSUMPTION]. A suggestion so the 70%
  and bootstrap rules are meaningful. It may be changed only before any
  new-engine model is trained.

## D4. Speed budget  [ASSUMPTION]
- p99 of POST /v1/decisions under 100 ms, measured server-side, at the
  "small" world's payment rate, after a warm-up period, on the author's
  laptop.
- Feature computation sub-budget: 50 ms inside the 100 ms.
- Why: no legal or public limit exists. The "30-60 seconds" figure was a
  conference comment, so a much stricter figure is chosen on purpose.

## D5. Held-out tactics  [POLICY]
- Held out (test and stress worlds only): slow drip, device rotation,
  pseudo-merchant camouflage, fake routine life.
- Kept in training: basic merchant mules (well documented; a bank would
  know them).
- Why: each held-out tactic attacks one signal family, so a failure shows
  which signal is fragile.
- Changed from: an earlier plan that held out plain merchant layering.

## Earlier work
The v1 PaySim model, its data split and its thresholds (0.5, 0.42, 0.03)
existed before this file. They are excluded from these pre-registered
comparisons, which apply to the new engine and its worlds only.

## Proof
Git history must show this file and config/evaluation.yaml committed before
any training code for the new engine. The Level 10 showcase checks this.
