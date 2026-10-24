"""
WHAT: Edge-case library loader, schema validator, and expectation runner for Level 12h.
It parses YAML edge case specifications from scenarios/edge_cases/, enforces a closed
schema, applies targeted configuration layer overrides with containment nesting checks,
and evaluates scenario outcomes against ground truth.

WHY: Level 12h builds a formal library of edge cases, base-rate variations, and synthetic
boundary conditions to audit the simulation engine and downstream ML detector models
without compromising baseline world distributions or leaking held-out tactic patterns.

TERM: edge case: A specific boundary scenario with defined configuration overrides,
      gang tactics, and expected behavioral invariants.
TERM: runnable case: An edge case executed directly in the simulation engine to verify
      specific behavioral outcomes against truth facts.
TERM: later-level case: An edge case deferred to a subsequent milestone pending engine
      mechanisms like high-volume shops or telephone number recycling.
TERM: deferred case: An edge case permanently deferred under an approved architectural
      amendment such as A-025.
"""

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from core.config_validator import load_yaml
from world.fast_mode import FastRunSettings
from world.gang_config import GangConfig, gang_config_from_plain
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.label_builder import build_label_events, build_registry_reports
from world.label_records import RegistryReport
from world.sampled_world import _check_nesting
from world.world_configs import CONFIG_NAMES, WorldConfigs, read_world_configs

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CASES_DIR: Path = REPO_ROOT / "scenarios" / "edge_cases"
AMENDMENTS_PATH: Path = REPO_ROOT / "docs" / "research" / "amendments.md"
WORLD_SIZING_PATH: Path = REPO_ROOT / "config" / "world_sizing.yaml"

VALID_STATUSES: tuple[str, ...] = ("runnable", "later_level", "deferred")
VALID_SPLITS: tuple[str, ...] = ("training", "test", "stress")
ALLOWED_KEYS: set[str] = {
    "name",
    "title",
    "expected_behaviour",
    "reason",
    "status",
    "active_from_level",
    "deferred_by",
    "split",
    "seed",
    "days",
    "preset",
    "layer_changes",
    "gangs",
    "note",
}


@dataclass(frozen=True)
class EdgeCase:
    """
    WHAT: Immutable specification of an edge-case scenario loaded from a library YAML file.
    It encapsulates metadata, execution boundaries, configuration layer overlays, and gang tactics.

    WHY: Freezing the dataclass guarantees that scenario definitions remain immutable
    across validation, configuration resolution, and simulation run cycles.
    """

    name: str
    title: str
    expected_behaviour: str
    reason: str
    status: str
    active_from_level: int | None = None
    deferred_by: str | None = None
    split: str | None = None
    seed: int | None = None
    days: int | None = None
    preset: str | None = None
    layer_changes: dict[str, Any] = field(default_factory=dict)
    gangs: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None
    file_name: str = ""


def check_mules_forward_after_days(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that slow_drip gang route forwarding waits at least 1,440 minutes and truth marks it held_out.
    WHY: Verifies edge-case 03: slow-drip evasion routing preserves multi-day holding intervals between hops.
    """
    slow_drip_gangs = [g for g in output.result.truth.gangs if g.family == "slow_drip"]
    if not slow_drip_gangs:
        return False, "", "no gang with family 'slow_drip' in truth"
    gang = slow_drip_gangs[0]
    if gang.kind != "held_out":
        return False, "", f"gang kind '{gang.kind}' != 'held_out'"

    route_events = [
        (ev, output.result.truth.payment_stories[idx])
        for ev, idx in zip(output.result.run_result.events, output.result.event_fraud_input_index)
        if idx is not None and output.result.truth.payment_stories[idx].block == "route"
    ]
    scam_victim_events = [
        (ev, output.result.truth.payment_stories[idx])
        for ev, idx in zip(output.result.run_result.events, output.result.event_fraud_input_index)
        if idx is not None
        and output.result.truth.payment_stories[idx].block == "scam"
        and output.result.truth.payment_stories[idx].scam_step == "victim_payment"
    ]
    if not route_events:
        return False, "", "no route payments executed"
    if not scam_victim_events:
        return False, "", "no scam victim payments executed"

    last_scam_min = max(ev.minute for ev, s in scam_victim_events)
    first_route_min = min(ev.minute for ev, s in route_events)
    wait_min = first_route_min - last_scam_min
    if wait_min < 1440:
        return False, f"wait_min={wait_min}", f"route wait {wait_min} < 1440 min"
    return True, f"wait_min={wait_min}", ""


def check_no_victim_ever_complains(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that at least one mule role exists in truth but exactly zero true complaints are filed.
    WHY: Verifies edge-case 04: when victim complaint share is zero, fraudulent activity generates no true complaints.
    """
    mule_roles = [r for r in output.result.truth.roles if r.role == "mule"]
    if not mule_roles:
        return False, "", "truth contains zero mule roles"

    complaints = build_label_events(output)
    fraud_payment_ids = {
        pid for pid, src in zip(output.result.payment_ids, output.result.event_sources) if src == "fraud"
    }
    true_complaints = [
        c for c in complaints if any(pid in fraud_payment_ids for pid in c.payment_ids)
    ]
    if len(true_complaints) > 0:
        return False, f"complaints={len(true_complaints)}", f"found {len(true_complaints)} true complaints"
    return True, "complaints=0", ""


def check_every_complaint_false(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that at least one complaint is filed and 100% of complaints name genuine non-fraud payments.
    WHY: Verifies edge-case 05: elevated false complaint rates produce genuine complaints without any true scam reports.
    """
    complaints = build_label_events(output)
    if not complaints:
        return False, "", "zero complaints generated"

    fraud_payment_ids = {
        pid for pid, src in zip(output.result.payment_ids, output.result.event_sources) if src == "fraud"
    }
    for c in complaints:
        if any(pid in fraud_payment_ids for pid in c.payment_ids):
            return False, "", f"complaint {c.label_id} names a fraud payment"

    return True, "false_ratio=1.0", ""


def check_registry_wrong_listing(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that at least one national registry report is filed for an account holding no active gang role.
    WHY: Verifies edge-case 06: false complaints propagate through reporting pipelines to list innocent citizens.
    """
    complaints = build_label_events(output)
    reports = build_registry_reports(output, complaints)
    if not reports:
        return False, "", "zero registry reports generated"

    wrong_listings: list[RegistryReport] = []
    for report in reports:
        roles_at_minute = [
            r for r in output.result.truth.roles
            if r.account_id == report.account_id
            and r.valid_from <= report.reported_minute
            and (r.valid_to is None or r.valid_to > report.reported_minute)
        ]
        if not roles_at_minute:
            wrong_listings.append(report)

    if not wrong_listings:
        return False, "", "no registry report for an account without a gang role"
    return True, f"wrong_listing_count={len(wrong_listings)}", ""


def check_one_huge_transfer(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that each route hop moves exactly Rs 1,00,000 without split pieces, matching layer count.
    WHY: Verifies edge-case 07: large lump-sum transfers maximize per-transaction limits in single atomic hops.
    """
    route_events = [
        (ev, output.result.truth.payment_stories[idx])
        for ev, idx in zip(output.result.run_result.events, output.result.event_fraud_input_index)
        if idx is not None and output.result.truth.payment_stories[idx].block == "route"
    ]
    if not route_events:
        return False, "", "zero route payments executed"

    gang_steps = [s for g in case.gangs for s in g.get("steps", []) if s.get("block") == "route"]
    expected_layers = gang_steps[0]["params"]["layers"] if gang_steps else 1
    if len(route_events) != expected_layers:
        return False, "", f"route payment count {len(route_events)} != layers {expected_layers}"

    for ev, s in route_events:
        if ev.amount_paise != 10000000:
            return False, "", f"route payment amount {ev.amount_paise} paise != 10,000,000 paise (Rs 1,00,000)"

    return True, "amount_inr=100000", ""


def _check_community_episode(
    output: InteractiveWorldOutput, target_kind: str
) -> tuple[bool, str, str]:
    """
    WHAT: Validates that an authentic pooling episode occurred whose collector and payers hold no gang roles.
    WHY: Avoids duplicating citizen episode validation logic across wedding, medical, and chit fund cases.
    """
    target_episodes = [ep for ep in output.episodes if ep.kind == target_kind]
    if not target_episodes:
        return False, "", f"no {target_kind} episode found"

    ep = target_episodes[0]
    collector_person = next(p for p in output.world.people if p.person_id == ep.person_id)
    collector_roles = [r for r in output.result.truth.roles if r.account_id == collector_person.account_id]
    if collector_roles:
        return False, "", f"collector {collector_person.account_id} holds gang role {collector_roles[0].role}"

    payer_accs = {
        ev.from_account
        for ev in output.result.run_result.events
        if ev.to_account == collector_person.account_id and ep.start_minute <= ev.minute <= ep.end_minute
    }
    payer_roles = [r for r in output.result.truth.roles if r.account_id in payer_accs]
    if payer_roles:
        return False, "", f"payer holds gang role {payer_roles[0].role}"

    return True, f"payer_count={ep.payer_count}", ""


def check_wedding_collection(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that a wedding/medical collection episode occurred with innocent collector and payers.
    WHY: Verifies edge-case 09: life event community collections are distinguished from illicit aggregation.
    """
    return _check_community_episode(output, "wedding_or_medical_collection")


def check_medical_fundraiser(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that an urgent medical collection episode occurred with innocent collector and payers.
    WHY: Verifies edge-case 10: rapid healthcare fundraising is distinguished from fraudulent mule accumulation.
    """
    return _check_community_episode(output, "wedding_or_medical_collection")


def check_chit_fund(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that a chit fund / kitty pooling episode occurred with innocent organizer and contributors.
    WHY: Verifies edge-case 11: informal rotating savings networks are recognized as genuine citizen finance.
    """
    return _check_community_episode(output, "chit_fund_or_kitty")


def check_genuine_shop_downstream_of_mule(output: InteractiveWorldOutput, case: EdgeCase) -> tuple[bool, str, str]:
    """
    WHAT: Asserts that at least one cash-out payment reached a legitimate shop holding no gang role in truth.
    WHY: Verifies edge-case 12: retail merchants downstream of mule spending are not wrongly attributed gang roles.
    """
    cashout_events = [
        (ev, output.result.truth.payment_stories[idx])
        for ev, idx in zip(output.result.run_result.events, output.result.event_fraud_input_index)
        if idx is not None and output.result.truth.payment_stories[idx].block == "cash_out"
    ]
    shop_accs = {s.account_id for s in output.world.shops}
    shop_cashouts = [ev for ev, s in cashout_events if ev.to_account in shop_accs]
    if not shop_cashouts:
        return False, "", "no cash-out payment to a shop"

    for ev in shop_cashouts:
        roles = [r for r in output.result.truth.roles if r.account_id == ev.to_account]
        if roles:
            return False, "", f"shop account {ev.to_account} holds gang role {roles[0].role}"

    return True, f"cashout_shops={len(shop_cashouts)}", ""


# EXPECTATIONS: dict mapping each runnable case name to its check function.
EXPECTATIONS: dict[str, Callable[[InteractiveWorldOutput, EdgeCase], tuple[bool, str, str]]] = {
    "mules_forward_after_days": check_mules_forward_after_days,
    "no_victim_ever_complains": check_no_victim_ever_complains,
    "every_complaint_false": check_every_complaint_false,
    "registry_wrong_listing": check_registry_wrong_listing,
    "one_huge_transfer": check_one_huge_transfer,
    "wedding_collection": check_wedding_collection,
    "medical_fundraiser": check_medical_fundraiser,
    "chit_fund": check_chit_fund,
    "genuine_shop_downstream_of_mule": check_genuine_shop_downstream_of_mule,
}


def load_sizing_presets() -> dict[str, Any]:
    """
    WHAT: Loads the preset templates dictionary defined in config/world_sizing.yaml.
    WHY: Enables validating that runnable edge cases reference recognized world size presets.
    """
    data = load_yaml(WORLD_SIZING_PATH)
    presets = data.get("presets", {})
    return {k: v for k, v in presets.items() if isinstance(v, dict)}


def validate_deferred_amendment(amendment_id: str) -> None:
    """
    WHAT: Verifies that a deferred edge case's amendment ID exists as a section header in amendments.md.
    WHY: Prevents edge cases from citing unapproved, hypothetical, or misspelled amendments.
    """
    if not AMENDMENTS_PATH.exists():
        raise ValueError(f"Amendments document not found at {AMENDMENTS_PATH}")
    content = AMENDMENTS_PATH.read_text(encoding="utf-8")
    target = f"## {amendment_id}"
    for line in content.splitlines():
        if line.strip() == target:
            return
    raise ValueError(f"deferred_by amendment '{amendment_id}' not found as '## {amendment_id}' in {AMENDMENTS_PATH}")


def apply_edge_case_layer(configs: WorldConfigs, layer_changes: dict[str, Any]) -> WorldConfigs:
    """
    WHAT: Deep-copies baseline configs, applies layer changes, and verifies nested containment bounds.
    WHY: Implements the pinned-value nesting rule sampled worlds use, avoiding strict min < max rejections.
    """
    applied = copy.deepcopy(configs)
    for cfg_name, settings_map in layer_changes.items():
        if cfg_name not in CONFIG_NAMES:
            allowed = ", ".join(CONFIG_NAMES)
            raise ValueError(f"unknown config name '{cfg_name}'. Allowed: {allowed}")
        if not isinstance(settings_map, dict) or not settings_map:
            raise ValueError(f"settings mapping for '{cfg_name}' must be a non-empty dict")
        cfg_dict = getattr(applied, cfg_name)
        params = cfg_dict.get("parameters", {})
        for setting_name, ranges_map in settings_map.items():
            if setting_name not in params:
                raise ValueError(f"unknown setting '{setting_name}' in config '{cfg_name}'")
            if not isinstance(ranges_map, dict) or not ranges_map:
                raise ValueError(f"ranges mapping for '{setting_name}' must be a non-empty dict")
            param_block = params[setting_name]
            for r_name, bounds in ranges_map.items():
                if r_name not in ("training", "test", "stress"):
                    raise ValueError(f"invalid range name '{r_name}' for setting '{setting_name}'")
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(f"range for '{setting_name}' ({r_name}) must have 2 numbers")
                low, high = bounds
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (low, high)):
                    raise ValueError(f"range values for '{setting_name}' ({r_name}) must be numeric")
                if low > high:
                    raise ValueError(f"range [{low}, {high}] for '{setting_name}' ({r_name}) has low > high")
                param_block[r_name] = list(bounds)
            # the pinned-value nesting rule sampled worlds use, Level 12f
            _check_nesting(setting_name, param_block)
    return applied


def load_edge_case(path: Path) -> EdgeCase:
    """
    WHAT: Loads and validates a single edge case YAML file against the closed edge-case schema.
    WHY: Enforces complete parameter typing, amendment validation, and layer nesting before execution.
    """
    raw_bytes = path.read_bytes()
    data = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Edge case '{path.name}': content must be a YAML mapping")

    # 1. Closed schema: refuse any unknown keys.
    actual_keys = set(data.keys())
    unknown_keys = sorted(actual_keys - ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(f"Edge case '{path.name}': unknown key '{unknown_keys[0]}'")

    # 2. Check required top-level fields.
    required_keys = {"name", "title", "expected_behaviour", "reason", "status"}
    missing_keys = sorted(required_keys - actual_keys)
    if missing_keys:
        raise ValueError(f"Edge case '{path.name}': missing required key '{missing_keys[0]}'")

    name = str(data["name"]).strip()
    title = str(data["title"]).strip()
    expected_behaviour = str(data["expected_behaviour"]).strip()
    reason = str(data["reason"]).strip()
    status = str(data["status"]).strip()

    if not name or not title or not expected_behaviour or not reason:
        raise ValueError(f"Edge case '{path.name}': required string fields must be non-empty")
    if status not in VALID_STATUSES:
        raise ValueError(f"Edge case '{path.name}': invalid status '{status}'. Allowed: {VALID_STATUSES}")

    active_from_level = data.get("active_from_level")
    deferred_by = data.get("deferred_by")
    split = data.get("split")
    seed = data.get("seed")
    days = data.get("days")
    preset = data.get("preset")
    layer_changes = data.get("layer_changes") or {}
    gangs = data.get("gangs") or []
    note = data.get("note")

    # 3. Status-specific invariants.
    if status == "later_level":
        if active_from_level is None or isinstance(active_from_level, bool) or not isinstance(active_from_level, int):
            raise ValueError(f"Edge case '{path.name}': later_level case requires integer active_from_level")
        if active_from_level <= 0:
            raise ValueError(f"Edge case '{path.name}': active_from_level must be positive")
        if deferred_by is not None:
            raise ValueError(f"Edge case '{path.name}': later_level case must not specify deferred_by")
        if any(v is not None for v in (split, seed, days, preset)):
            raise ValueError(f"Edge case '{path.name}': later_level case must not specify runnable settings")

    elif status == "deferred":
        if deferred_by is None or not isinstance(deferred_by, str) or not deferred_by.strip():
            raise ValueError(f"Edge case '{path.name}': deferred case requires non-empty deferred_by amendment id")
        validate_deferred_amendment(deferred_by.strip())
        if active_from_level is not None:
            raise ValueError(f"Edge case '{path.name}': deferred case must not specify active_from_level")
        if any(v is not None for v in (split, seed, days, preset)):
            raise ValueError(f"Edge case '{path.name}': deferred case must not specify runnable settings")

    elif status == "runnable":
        if split not in VALID_SPLITS:
            raise ValueError(f"Edge case '{path.name}': runnable case requires split in {VALID_SPLITS}")
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Edge case '{path.name}': runnable case requires integer seed")
        if days is None or isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise ValueError(f"Edge case '{path.name}': runnable case requires positive integer days")
        presets = load_sizing_presets()
        if preset not in presets:
            raise ValueError(f"Edge case '{path.name}': runnable case preset '{preset}' not in world_sizing presets")
        if active_from_level is not None or deferred_by is not None:
            raise ValueError(
                f"Edge case '{path.name}': runnable case must not specify active_from_level or deferred_by"
            )

    # 4. Validate layer_changes if present.
    if layer_changes:
        if not isinstance(layer_changes, dict):
            raise ValueError(f"Edge case '{path.name}': layer_changes must be a mapping")
        base_configs = read_world_configs()
        apply_edge_case_layer(base_configs, layer_changes)

    # 5. Validate gangs if present.
    if gangs:
        if not isinstance(gangs, list):
            raise ValueError(f"Edge case '{path.name}': gangs must be a list of plain dicts")
        for g_plain in gangs:
            if not isinstance(g_plain, dict):
                raise ValueError(f"Edge case '{path.name}': each gang must be a plain dict")
            gang_config_from_plain(g_plain)

    return EdgeCase(
        name=name,
        title=title,
        expected_behaviour=expected_behaviour,
        reason=reason,
        status=status,
        active_from_level=active_from_level,
        deferred_by=deferred_by,
        split=split,
        seed=seed,
        days=days,
        preset=preset,
        layer_changes=layer_changes,
        gangs=gangs,
        note=note,
        file_name=path.name,
    )


def load_edge_case_library(folder: Path | None = None) -> list[EdgeCase]:
    """
    WHAT: Loads and validates every YAML edge-case definition within a target folder.
    WHY: Returns a deterministically sorted catalog of validated edge cases for test and CLI execution.
    """
    target_folder = folder if folder is not None else DEFAULT_CASES_DIR
    if not target_folder.exists() or not target_folder.is_dir():
        raise ValueError(f"Edge case directory not found: {target_folder}")

    yaml_files = sorted(target_folder.glob("*.yaml"), key=lambda p: p.name)
    cases = [load_edge_case(p) for p in yaml_files]
    return cases


def validate_case_expectation_registered(case: EdgeCase) -> None:
    """
    WHAT: Asserts that a runnable edge case has an expectation check function registered in EXPECTATIONS.
    WHY: Fails early when a runnable case lacks automated verification logic.
    """
    if case.status == "runnable" and case.name not in EXPECTATIONS:
        raise ValueError(f"runnable case '{case.name}' has no expectation check")


def run_edge_case(case: EdgeCase) -> InteractiveWorldOutput:
    """
    WHAT: Configures and executes an interactive simulation run for a runnable edge case.
    WHY: Prepares applied layer configs, world sizing dimensions, and gang specifications for verification.
    """
    if (
        case.status != "runnable"
        or case.preset is None
        or case.seed is None
        or case.days is None
        or case.split is None
    ):
        raise ValueError(f"Cannot run non-runnable or incomplete edge case '{case.name}' (status: {case.status})")

    base_configs = read_world_configs()
    configs = apply_edge_case_layer(base_configs, case.layer_changes) if case.layer_changes else None

    presets = load_sizing_presets()
    preset_data = presets[case.preset]
    n_people = int(preset_data["n_people"])
    n_shops = int(preset_data["n_shops"])

    # Training or test -> training column; stress -> stress column
    range_col = "stress" if case.split == "stress" else "training"
    fast_settings = FastRunSettings(
        seed=case.seed,
        n_people=n_people,
        n_shops=n_shops,
        n_days=case.days,
        range_name=range_col,
    )

    gang_objs: tuple[GangConfig, ...] = tuple(gang_config_from_plain(g) for g in case.gangs)
    runner_settings = RunnerSettings(fraud_enabled=bool(case.gangs))

    return run_interactive_world(
        fast_settings,
        runner_settings,
        configs=configs,
        gang_configs=gang_objs,
        world_split=case.split,
    )


def main() -> None:
    """
    WHAT: Command-line entry point for the edge case library module.
    WHY: Formats and outputs library status overview and exits 1 if any case violates validation invariants.
    """
    cases = load_edge_case_library()
    run_all = "--run" in sys.argv
    has_errors = False

    for case in cases:
        prefix = case.file_name.split("-")[0] if "-" in case.file_name else "00"
        if case.status == "later_level":
            print(f"{prefix} {case.name}: later_level (active from Level {case.active_from_level})")
        elif case.status == "deferred":
            print(f"{prefix} {case.name}: deferred ({case.deferred_by})")
        elif case.status == "runnable":
            if case.name not in EXPECTATIONS:
                print(f"{prefix} {case.name}: INVALID (runnable case '{case.name}' has no expectation check)")
                has_errors = True
            elif run_all:
                check_fn = EXPECTATIONS[case.name]
                try:
                    output = run_edge_case(case)
                    passed, key_number, detail = check_fn(output, case)
                    if passed:
                        print(f"{prefix} {case.name}: PASS ({key_number})")
                    else:
                        print(f"{prefix} {case.name}: FAIL ({detail})")
                        has_errors = True
                except Exception as exc:
                    print(f"{prefix} {case.name}: FAIL ({exc})")
                    has_errors = True
            else:
                print(f"{prefix} {case.name}: runnable ({case.split})")

    if has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
