# Tactic plugins

A plugin is one Python file in this folder that adds a tactic block to the gang language. `cool_off.py` is the
example: the gang waits some days before its next step.

## What a plugin may do

- Define one module-level `PLUGIN`, built with `world.tactic_plugins.TacticPlugin`, that describes the block like a
  block of `config/tactic_blocks.yaml`: name, meaning, allowed strengths, parameters (a range or boolean), a
  `risk_per_use` range, label `assumption` and a reason.
- Define a `plan(context)` function. It receives only the step's parameters, the gang's clock minute, the strength
  and the gang's own random generator (`context.rng`), and returns only the minutes it advances the clock and the
  money it costs.

## What a plugin may not do

- Use any other source of randomness (`random`, `secrets`, `numpy.random` functions, `os.urandom`) or the clock.
- Open files, or use `exec`, `eval` or `__import__`.
- Import `evaluation` or any `world` module except `world.tactic_plugins`.
- Make payments, roles or device events. It cannot, because it can only return minutes and money.

The file is checked (linted) before it is imported. The lists are in `config/tactic_plugins.yaml`. A lint is not a
sandbox: read a plugin before you trust it.

## Where plugins live

Plugins load only from this folder and never leave this machine. Contribution bundles must not include this folder.
A plugin's file hash goes into the gang's run id, so changing the code changes the run id.

## Tactic language only

Plugins describe tactics in the tactic language only (ground rule 20: we publish defences, not attacks).
