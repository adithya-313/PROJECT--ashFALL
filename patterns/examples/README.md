# Pattern files

A pattern file is a small YAML file that describes one gang in the tactic language: how many accounts it takes
over, which steps it runs, and how much money, time and risk it may spend. The top level holds one `pattern:` block
with `pattern_version`, `name`, `description` and a `gang:` block in the same shape as the gang fixtures in
`showcase/fixtures/level_12/`. Any other key is refused, so a typo cannot be silently ignored.

## Check one

```
python -c "from world.pattern_files import validate_pattern_file; print(validate_pattern_file('patterns/examples/scam_ring_basic.yaml').summary_text())"
```

A valid file prints a sentence such as "about N mule accounts moving about Rs X per week (ESTIMATE: ...)". An invalid
file raises `PatternFileError`; its message starts with the file name and then the reason, for example
`malformed: missing key 'gang'`, `unsupported version 2`, `invalid gang: ... unknown block ...` or `over budget: ...`.

The preview plans the gang once in a small training world. It does not run the world, so its numbers are an ESTIMATE.

## What to put in a pattern

Patterns describe tactics in the tactic language only (ground rule 20: we publish defences, not attacks). The name and
description are plain text for people; they are never executed.

Examples here:

- `scam_ring_basic.yaml` validates.
- `scam_ring_over_budget.yaml` is the same ring with a budget of 100 paise and is refused with "over budget:".
