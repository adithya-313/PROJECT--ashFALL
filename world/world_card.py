"""
WHAT: Generates and persists standardized world cards describing simulation environments and results.
It summarizes execution settings, bank distributions, regional citizen counts, applied stress scenarios,
and transaction failure metrics into structured JSON and plain-English Markdown formats.

WHY: Provides human-readable and machine-readable audit documentation beside raw simulation event logs.
Operators and downstream evaluation harnesses can quickly inspect town parameters and scenario shocks
without parsing multi-megabyte transaction event streams.

TERM: world card: A standardized summary document recording simulation settings and baseline outcome metrics,
      e.g. recording bank customer shares and total payment failures beside events.jsonl.
TERM: base rate: The unobserved proportion of transactions that represent fraud before detection interventions,
      e.g. the true proportion of fraudulent payments in the overall transaction volume.
TERM: inflation factor: The multiplier estimating total fraudulent activity relative to reported complaints,
      e.g. scaling reported victim losses by an assumed under-reporting multiplier.
TERM: region mix: The geographic allocation of citizens and merchants across distinct simulated districts,
      e.g. distributing sixty citizens across four municipal zones.
"""

import json
from pathlib import Path
from typing import Any

from world.build_world import world_fingerprint
from world.layers import Layer


def _build_banks_summary(world: Any) -> list[dict[str, Any]]:
    """
    WHAT: Extracts configuration parameters and citizen allocations across all banks in the world.
    It counts how many citizens hold accounts at each bank and records their weak KYC shares,
    customer shares, and hold delay multipliers.

    WHY: Documents the structural banking composition of the simulated town economy.
    Verifying that citizen counts accurately reflect bank customer shares enables auditability
    of financial institution dynamics across different simulation seeds.

    Args:
        world: World container holding simulated banks and citizen agents.

    Returns:
        List of dicts with bank index, kind, customer share, weak KYC share, hold delay factor, and people count.

    Example:
        >>> from world.build_world import build_world
        >>> w = build_world(seed=101, n_people=10, n_shops=4)
        >>> banks = _build_banks_summary(w)
        >>> len(banks)
        3
        >>> sum(b["people_count"] for b in banks)
        10
    """
    # 1. Iterate through each bank and count citizen account holders.
    banks_summary: list[dict[str, Any]] = []
    for bank in world.banks:
        count = sum(1 for person in world.people if person.bank_index == bank.bank_index)
        banks_summary.append({
            "bank_index": bank.bank_index,
            "kind": bank.kind,
            "customer_share": bank.customer_share,
            "weak_kyc_share": bank.weak_kyc_share,
            "hold_delay_factor": bank.hold_delay_factor,
            "people_count": count,
        })
    # 2. Hand back the assembled banking partition records.
    return banks_summary


def _build_regions_summary(world: Any) -> dict[str, Any]:
    """
    WHAT: Computes the regional citizen distribution across municipal geographic zones.
    It reads total zone counts from world parameters and tallies citizen residency per zone.
    Zone identifiers are formatted as string numbers to guarantee valid JSON mapping keys.

    WHY: Captures spatial demographic dispersion for auditability and scenario inspection.
    Formatting zone keys as strings ensures consistency across JSON and Markdown outputs.

    Args:
        world: World container holding municipal parameters and citizen records.

    Returns:
        Dictionary containing total zone count and string-keyed citizen count dictionary.

    Example:
        >>> from world.build_world import build_world
        >>> w = build_world(seed=101, n_people=10, n_shops=4)
        >>> reg = _build_regions_summary(w)
        >>> (reg["n_regions"], sum(reg["people_per_region"].values()))
        (4, 10)
    """
    # 1. Extract total zone partition count from initialized parameters.
    n_regions = int(world.parameters_drawn["n_regions"])

    # 2. Initialize tallies and count citizens residing in each zone.
    people_per_region = {str(region_id): 0 for region_id in range(n_regions)}
    for person in world.people:
        zone_key = str(person.region)
        people_per_region[zone_key] = people_per_region.get(zone_key, 0) + 1

    # 3. Assemble and return regional breakdown mapping.
    return {
        "n_regions": n_regions,
        "people_per_region": people_per_region,
    }


def _build_layers_summary(
    layers: list[Layer] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    WHAT: Partitions applied configuration layers into stress tests and standard overlay layers.
    For each stress test, it extracts real-event grounding, exaggeration description, activation level, and SHA-256.
    For non-stress layers, it records their layer names in application sequence.

    WHY: Distinguishes operational stress shocks from standard sensitivity parameter layers.
    Retaining cryptographic hashes and event grounding enables complete audit traceability of simulation inputs.

    Args:
        layers: Optional list of Layer instances applied during the simulation run.

    Returns:
        Tuple of (list of stress test dictionaries, list of non-stress layer names).

    Example:
        >>> from world.layers import Layer
        >>> st = {"real_event": "2024 outage", "exaggeration": "5x duration", "takes_effect_from_level": None}
        >>> l1 = Layer("st1", "test", {}, "hash1", "p1", stress_test=st)
        >>> l2 = Layer("base_l", "test", {}, "hash2", "p2", stress_test=None)
        >>> stress_list, layer_names = _build_layers_summary([l1, l2])
        >>> (len(stress_list), layer_names)
        (1, ['base_l'])
    """
    # 1. Initialize collections for stress scenarios and standard overlays.
    stress_tests: list[dict[str, Any]] = []
    non_stress_layers: list[str] = []
    if not layers:
        return stress_tests, non_stress_layers

    # 2. Iterate through layers and categorize based on presence of stress_test metadata.
    for layer in layers:
        if layer.stress_test is not None:
            stress_tests.append({
                "name": layer.name,
                "real_event": layer.stress_test["real_event"],
                "exaggeration": layer.stress_test["exaggeration"],
                "takes_effect_from_level": layer.stress_test["takes_effect_from_level"],
                "sha256": layer.sha256,
            })
        else:
            non_stress_layers.append(layer.name)

    # 3. Hand back separated stress and non-stress summaries.
    return stress_tests, non_stress_layers


def build_world_card(
    settings: Any,
    output: Any,
    layers: list[Layer] | None = None,
) -> dict[str, Any]:
    """
    WHAT: Constructs a complete world card dictionary summarizing simulation parameters and results.
    It compiles settings, clock start, cryptographic fingerprint, banks, regions, stress tests, layers,
    payment counts by failure domain, and placeholder fraud base-rate records.

    WHY: Encapsulates all environmental, demographic, and operational characteristics of a simulation run.
    Serving as a comprehensive audit artifact, it enables immediate comparison across benchmark runs.

    Args:
        settings: FastRunSettings container specifying seed, population, shops, horizon, and split.
        output: FastRunOutput container holding the generated world, plan, and execution result.
        layers: Optional list of Layer instances applied during world execution.

    Returns:
        Dictionary containing all standardized world card attributes.

    Example:
        >>> from world.fast_mode import FastRunSettings, run_fast_world
        >>> s = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
        >>> out = run_fast_world(s)
        >>> card = build_world_card(s, out)
        >>> (card["settings"]["seed"], len(card["banks"]), card["base_rate"]["fraud_share"])
        (101, 3, None)
    """
    # 1. Summarize banks, regions, and applied configuration layers.
    banks_summary = _build_banks_summary(output.world)
    regions_summary = _build_regions_summary(output.world)
    stress_tests, non_stress_layers = _build_layers_summary(layers)

    # 2. Extract timestamp, world fingerprint, and execution payment counts.
    clock_start = output.plan.clock.to_datetime(0).isoformat()
    fingerprint = world_fingerprint(output.world)
    counts = output.result.counts
    if "failures_by_domain" in counts:
        domain_failures = dict(counts["failures_by_domain"])
    else:
        domain_failures = {
            k.removeprefix("domain_"): v
            for k, v in sorted(counts.items())
            if k.startswith("domain_")
        }
    payments_summary = {
        "attempts": counts["attempts"],
        "successes": counts["successes"],
        "failures": counts["failures"],
        "failures_by_domain": domain_failures,
    }

    # 3. Assemble complete standardized world card dictionary.
    return {
        "settings": {
            "seed": settings.seed,
            "n_people": settings.n_people,
            "n_shops": settings.n_shops,
            "n_days": settings.n_days,
            "range_name": settings.range_name,
        },
        "clock_start": clock_start,
        "world_fingerprint": fingerprint,
        "banks": banks_summary,
        "regions": regions_summary,
        "stress_tests": stress_tests,
        "layers": non_stress_layers,
        "payments": payments_summary,
        "base_rate": {
            "fraud_share": None,
            "fraud_inflation_factor": None,
            "note": "Filled in Level 12, once fraud gangs exist.",
        },
    }


def _render_markdown(card: dict[str, Any]) -> str:
    """
    WHAT: Formats a world card dictionary into a human-readable Markdown report document.
    It structures content into distinct sections covering settings, banks, regions, stress tests,
    layers, payment outcome counts, and base-rate placeholders.

    WHY: Produces an easily readable summary page for simulation operators and reviewers.
    Clearly highlighting stress tests and parameter states simplifies scenario comparison.

    Args:
        card: World card dictionary containing summarized simulation attributes.

    Returns:
        Formatted Markdown text string.

    Example:
        >>> b0 = {"bank_index": 0, "kind": "large", "customer_share": 0.6,
        ...       "weak_kyc_share": 0.1, "hold_delay_factor": 1.0, "people_count": 6}
        >>> sample_card = {
        ...     "settings": {"seed": 101, "n_people": 10, "n_shops": 4, "n_days": 61, "range_name": "training"},
        ...     "clock_start": "2026-09-01T00:00:00", "world_fingerprint": "abc",
        ...     "banks": [b0],
        ...     "regions": {"n_regions": 1, "people_per_region": {"0": 10}},
        ...     "stress_tests": [], "layers": [],
        ...     "payments": {"attempts": 100, "successes": 95, "failures": 5, "failures_by_domain": {"technical": 5}},
        ...     "base_rate": {"fraud_share": None, "fraud_inflation_factor": None, "note": "Filled in Level 12."}
        ... }
        >>> md = _render_markdown(sample_card)
        >>> "# World Card" in md and "Base rate: not yet known (Level 12)" in md
        True
    """
    # 1. Render title, settings, and root environment properties.
    settings = card["settings"]
    lines = [
        "# World Card",
        "",
        "## Settings",
        f"- **Seed**: {settings['seed']}",
        f"- **Citizens**: {settings['n_people']}",
        f"- **Shops**: {settings['n_shops']}",
        f"- **Horizon**: {settings['n_days']} days",
        f"- **Range split**: {settings['range_name']}",
        f"- **Clock start**: {card['clock_start']}",
        f"- **World fingerprint**: `{card['world_fingerprint']}`",
        "",
        "## Banks",
    ]

    # 2. Render financial institutions and demographic distributions.
    for bank in card["banks"]:
        lines.append(
            f"- **Bank {bank['bank_index']}** ({bank['kind']}): {bank['people_count']} citizens, "
            f"customer share {bank['customer_share']}, weak KYC share {bank['weak_kyc_share']}, "
            f"hold delay factor {bank['hold_delay_factor']}"
        )
    lines.extend([
        "",
        "## Regions",
        f"- **Total regions**: {card['regions']['n_regions']}",
    ])
    for zone, count in sorted(card["regions"]["people_per_region"].items(), key=lambda x: int(x[0])):
        lines.append(f"  - Region {zone}: {count} citizens")

    # 3. Render applied scenario stress tests and non-stress configuration layers.
    lines.extend(["", "## Stress Tests"])
    if card["stress_tests"]:
        for st in card["stress_tests"]:
            lines.append(f"### STRESS TEST: {st['name']}")
            lines.append(f"- **Real event**: {st['real_event']}")
            lines.append(f"- **Exaggeration**: {st['exaggeration']}")
            eff = f"Level {st['takes_effect_from_level']}" if st["takes_effect_from_level"] else "Immediate"
            lines.append(f"- **Takes effect**: {eff}")
            lines.append(f"- **Layer SHA-256**: `{st['sha256']}`")
    else:
        lines.append("- None applied.")

    lines.extend(["", "## Layers"])
    if card["layers"]:
        for layer_name in card["layers"]:
            lines.append(f"- {layer_name}")
    else:
        lines.append("- None applied.")

    # 4. Render payment transaction counts and base rate status.
    payments = card["payments"]
    lines.extend([
        "",
        "## Payments",
        f"- **Attempts**: {payments['attempts']}",
        f"- **Successes**: {payments['successes']}",
        f"- **Failures**: {payments['failures']}",
        f"- **Failures by domain**: {payments['failures_by_domain']}",
        "",
        "## Base Rate",
        "- **Base rate: not yet known (Level 12)**",
        f"- **Note**: {card['base_rate']['note']}",
        "",
    ])

    return "\n".join(lines)


def write_world_card(card: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """
    WHAT: Writes both world_card.json and world_card.md files into the specified directory.
    It serializes the dictionary to indented, sorted JSON and writes formatted Markdown.
    Both files are saved using UTF-8 encoding with explicit Unix newlines.

    WHY: Provides consistent export of both structured and human-readable audit cards.
    Placing these files alongside events and manifests guarantees complete simulation metadata.

    Args:
        card: World card dictionary containing all simulation summary properties.
        out_dir: Destination directory path where card files must be written.

    Returns:
        Tuple of (Path to written world_card.json, Path to written world_card.md).

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> sample = {"settings": {"seed": 1, "n_people": 1, "n_shops": 1, "n_days": 1, "range_name": "t"},
        ...           "clock_start": "t0", "world_fingerprint": "fp", "banks": [],
        ...           "regions": {"n_regions": 1, "people_per_region": {}},
        ...           "stress_tests": [], "layers": [],
        ...           "payments": {"attempts": 0, "successes": 0, "failures": 0, "failures_by_domain": {}},
        ...           "base_rate": {"fraud_share": None, "fraud_inflation_factor": None, "note": "n"}}
        >>> with tempfile.TemporaryDirectory() as td:
        ...     jp, mp = write_world_card(sample, Path(td))
        ...     res = (jp.name, mp.name)
        >>> res
        ('world_card.json', 'world_card.md')
    """
    # 1. Serialize card dictionary to sorted, indented JSON with trailing newline.
    json_path = out_dir / "world_card.json"
    json_text = json.dumps(card, sort_keys=True, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8", newline="\n")

    # 2. Render plain-English Markdown summary text and write to disk.
    md_path = out_dir / "world_card.md"
    md_text = _render_markdown(card)
    md_path.write_text(md_text, encoding="utf-8", newline="\n")

    # 3. Hand back paths to both written card artifacts.
    return json_path, md_path
