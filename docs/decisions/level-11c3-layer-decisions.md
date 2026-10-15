# Level 11C-3: Stackable Layers Decisions

## 1. What a Layer Is

A layer is a small YAML overlay document that modifies parameter ranges for named settings during a single simulation run, leaving all other settings untouched (Level 11, task 18). Rather than editing the foundational project configuration files or maintaining duplicated configuration trees, an operator provides one or more layer files to alter continuous parameter bounds for sensitivity analysis or scenario stress-testing.

For example, the `late-festival` layer shifts the training range of Diwali's peak day window:

```yaml
name: late-festival
reason: Check that a late festival window moves the diwali peak.
changes:
  background_realism:
    festival_peak_day_in_world:
      training: [40, 45]
```

When this layer is applied, `festival_peak_day_in_world` shifts its training interval from `[30, 50]` to `[40, 45]`. The other 111 configuration settings across all four domain configuration files (`world_ranges`, `background_realism`, `failure_realism`, and `device_realism`) remain exactly at their baseline values.

## 2. The Four Owner Decisions

The design of the layer subsystem reflects four core decisions established on 2026-10-16:

### Decision 1: Only Existing Settings and Ranges May Be Changed
A layer may change only the `training`, `test`, or `stress` range intervals of settings that already exist in the baseline configurations. Unrecognized configuration names, setting names, or range partition keys are rejected immediately by name with a `ValueError`.
*Reason*: Enforcing that every referenced key exists prevents typos from silently passing through. It guarantees that the simulation engine actively consumes every value declared in a layer overlay.

### Decision 2: Sequential Merge Order (Later Layer Wins on Clashes)
When multiple layers modify the same parameter range, layers stack in the order specified on the command line; a later layer overwrites the value set by an earlier layer. The effective-world summary explicitly prints every transition step.
*Reason*: This is the fundamental definition of "merge order." Displaying both before-and-after values in chronological order ensures complete transparency and auditability, so operators can trace how conflicting layer overrides were resolved.

### Decision 3: Strict Range Nesting Invariant
After stacking all layer overrides, every modified setting must continue to satisfy the invariant `training` inside `test` inside `stress` (`low_outer <= low_inner` and `high_inner <= high_outer`). If any setting violates nesting, execution halts with a `ValueError` naming the configuration, the setting, the specific violation, and the author layer.
*Reason*: This applies the same nested range validation rule enforced by the project validator (`core.config_validator.check_nested_ranges`). A custom layer can never construct an invalid configuration state that the repository itself would reject.

### Decision 4: Mandatory Metadata and Manifest Provenance
Every layer must declare a non-empty `name` and `reason`. When one or more layers are applied in fast mode, the generated `manifest.json` records a `"layers"` array listing the name and SHA-256 digest of each layer in application order.
*Reason*: Requiring names and reasons prevents anonymous, undocumented parameter tweaks. Recording the SHA-256 hash in the run manifest creates a tamper-evident audit trail, ensuring that any layered simulation run can be verified and reproduced bit for bit.

## 3. The Off Switch

When no layers are specified, the simulation engine produces bit-for-bit identical outputs to the baseline system:

1. **Step 1 Verification**: Passing `configs=None` or `configs=read_world_configs()` into `run_fast_world(FastRunSettings(101, 60, 12, 61, 'training'))` produces the exact golden events SHA-256 digest:
   `24fc0e25009d79462d508f040a109620dd4308486cc78aee589393fe921f92fd`
2. **Step 3 Verification**: Running fast mode without layers:
   ```powershell
   python -B -m world.fast_mode --preset tiny --seed 101 --out runs/check-nolayer
   ```
   produces `Events: 2915`, `SHA-256: 24fc0e25009d79462d508f040a109620dd4308486cc78aee589393fe921f92fd`, and a `manifest.json` file with no `"layers"` key, preserving byte-for-byte backwards compatibility.
3. In contrast, running with `--layer runs/late.yaml`:
   ```powershell
   python -B -m world.fast_mode --preset tiny --seed 101 --layer runs/late.yaml --out runs/check-late
   ```
   prints the effective-world transition summary:
   ```
   Layers applied (in order): late-festival
     festival_peak_day_in_world (training): [30, 50] -> [40, 45]   from late-festival
   Everything else: unchanged (111 of 112 settings)
   ```
   and yields a different events SHA-256 digest (`6d1654b6cd7cb68306a9ac083389c7d94e7e4434c7855aa53249a88ff016aead`) and records the layer's SHA-256 digest (`2f2af2812c6aaaf4d0efb3c6717a102bace6c9d7fd12c6800ae9354729e7dea2`) in the manifest.

## 4. Known Limits

1. **Calibration Uses Base Configuration**: Machine calibration measures hardware execution speed using baseline configuration parameters rather than layered parameters. If a layer substantially increases transaction frequency, runtime and memory may be estimated slightly low. The existing +15 percent sizing safety margin covers modest variations.
2. **Range Modification Only**: Layers modify parameter interval bounds; they cannot alter simulation control flow or introduce structural entities (such as adding a fourth bank, custom payment rails, or full registry outage behaviors). Structural and architectural variations require code enhancements and are scheduled for Level 11D or later.
