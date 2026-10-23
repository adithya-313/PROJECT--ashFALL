"""
WHAT: Generates the gangs of one sampled world. The world sampler sets the centre of each gang-level setting (layers,
forwarding delay, devices, victims per receiving mule, number of gangs); each gang draws its own value around that
centre, and the gangs are built through gang_config_from_plain so every existing gang rule still applies.

WHY: With a single fixture gang every sampled world would vary its labels but never its fraud. Drawing gangs around
stratified centres makes worlds differ in the fraud itself. The draws use their own random stream (child 22 of the
seed, "gang_mix" in config/interactive_runner.yaml) so they never touch the sampler's stream or children 0 to 21.
Family tags describe what a generated gang does and come from committed rules over training family names only; held-out
generation is off because no building blocks exist for held-out families yet.

ISSUE IDs: 12-14, 12-18.
TERM: centre: the value the sampler gives a world for a gang-level setting; its gangs draw around it.
TERM: spread: the share of a parameter's range width a gang may stray from the centre, set in config/gang_mix.yaml.
TERM: capacity: the number of ordinary people a world can supply as mules and victims to its gangs.
"""

from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.gang_config import GangConfig, gang_config_from_plain
from world.run_identity import REPO_ROOT, load_runner_config
from world.tactic_families import load_tactic_families
from world.world_configs import read_world_configs
from world.world_sampler import WorldSpec, key_parameter_range

DEFAULT_MIX_PATH: Path = REPO_ROOT / "config" / "gang_mix.yaml"
STREAM_NAME = "gang_mix"
MINUTES_PER_DAY = 1440


def load_gang_mix_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/gang_mix.yaml (or the given path) and returns its gang_mix block.

    WHY: The spread, ranges, budget and family rules are policy, so they live in YAML and not in code.

    Args:
        path: Optional YAML path.

    Returns:
        The gang_mix dictionary.
    """
    data = load_yaml(DEFAULT_MIX_PATH if path is None else path)
    block: dict[str, Any] = data["gang_mix"]
    return block


def validate_gang_mix_config(mix_cfg: dict[str, Any], training_families: set[str]) -> None:
    """
    WHAT: Refuses a gang mix config whose held-out generation is not false, whose spread is outside (0, 1], whose
    ranges are not ordered whole numbers, or whose family rules name a family that is not a training family.

    WHY: A held-out tag on training behaviour would be a false label, so the guard fails loudly instead of tagging.

    Args:
        mix_cfg: The gang_mix dictionary.
        training_families: The names in the training list of config/tactic_families.yaml.

    Raises:
        ValueError: With a message naming the problem.
    """
    if mix_cfg["held_out_generation"]["value"] is not False:
        raise ValueError("held_out_generation must be false: held-out families have no building blocks yet")
    spread = mix_cfg["spread_share_of_range"]["value"]
    if not 0.0 < spread <= 1.0:
        raise ValueError(f"spread_share_of_range must be above 0 and at most 1, got {spread}")
    for name in ("receiving_mules", "start_day_range"):
        low, high = mix_cfg[name]["range"]
        if not (isinstance(low, int) and isinstance(high, int) and 1 <= low <= high):
            raise ValueError(f"{name} range must be ordered whole numbers of at least 1, got [{low}, {high}]")
    for rule in mix_cfg["family_rules"]["rules"]:
        if rule["family"] not in training_families:
            raise ValueError(f"family rule names '{rule['family']}', which is not a training family")


def gang_mix_rng(seed: int) -> np.random.Generator:
    """
    WHAT: Builds the gang-mix random generator for a world seed: child 22 of the seed, the child number being read from
    random_stream_children in config/interactive_runner.yaml.

    WHY: One private stream keeps the mix repeatable and leaves every other stream, and every run id, as it was.

    Args:
        seed: The world's seed.

    Returns:
        A numpy Generator.
    """
    child = int(load_runner_config()["random_stream_children"][STREAM_NAME])
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(child + 1)[child])


def _draw_around(
    rng: np.random.Generator, centre: float, low: float, high: float, spread: float, is_integer: bool
) -> float | int:
    """Draws uniformly within centre +/- spread x width, clipped to [low, high], rounded for whole numbers."""
    half = spread * (high - low)
    drawn = float(rng.uniform(centre - half, centre + half))
    clipped = max(float(low), min(float(high), drawn))
    if is_integer:
        return max(int(low), min(int(high), int(round(clipped))))
    return clipped


def family_for_gang(rules: list[dict[str, Any]], layers: int) -> str | None:
    """
    WHAT: Returns the family of the first rule that matches a gang's route layers, or None when no rule matches.

    WHY: A family tag must follow from what the gang does; a gang that matches nothing stays untagged.

    Args:
        rules: The family_rules list from config/gang_mix.yaml.
        layers: The gang's route layers.

    Returns:
        A family name or None.
    """
    for rule in rules:
        if "at_least" in rule and layers < rule["at_least"]:
            continue
        if "at_most" in rule and layers > rule["at_most"]:
            continue
        return str(rule["family"])
    return None


def _centres(spec: WorldSpec, sampler_cfg: dict[str, Any]) -> dict[str, float | int]:
    """Returns the spec's gang-scope centres by name."""
    centres: dict[str, float | int] = {}
    for name, value in spec.key_values:
        if sampler_cfg["key_parameters"][name]["scope"] == "gang":
            centres[name] = value
    return centres


def _gang_plain(
    index: int, drawn: dict[str, int], start_minute: int, family: str | None, budget: dict[str, Any]
) -> dict[str, Any]:
    """Builds the plain gang dictionary: acquire, take_control, scam, route, cash_out."""
    acquire_count = drawn["layers"] * drawn["receiving_mules"]
    plain: dict[str, Any] = {
        "gang_id": f"gang_mix_{index}",
        "strength": 1,
        "budget": {
            "money_paise": budget["money_paise"], "time_minutes": budget["time_minutes"],
            "max_risk": budget["max_risk"],
        },
        "start_minute": start_minute,
        "steps": [
            # Route layer 1 is the receiving mules, so every layer needs receiving_mules accounts: acquire exactly that.
            {"block": "acquire", "params": {"count": acquire_count, "origin_mix": "world_ranges"}},
            # Gang phones are shared among the mules in turn; never more phones than mules.
            {"block": "take_control", "params": {"devices": min(drawn["devices"], acquire_count)}},
            {"block": "scam", "params": {
                "receiving_mules": drawn["receiving_mules"], "victims_per_receiving_mule": drawn["victims"],
            }},
            # accounts_per_layer must equal the scam step's receiving_mules (gang validator).
            {"block": "route", "params": {
                "layers": drawn["layers"], "accounts_per_layer": drawn["receiving_mules"],
                "dwell_minutes": drawn["dwell"],
            }},
            {"block": "cash_out", "params": {}},
        ],
    }
    if family is not None:
        plain["family"] = family
    return plain


def generate_gang_mix(
    spec: WorldSpec,
    sampler_cfg: dict[str, Any],
    mix_cfg: dict[str, Any],
    eligible_people: int,
    world_ranges: dict[str, Any] | None = None,
) -> tuple[GangConfig, ...]:
    """
    WHAT: Builds the world's gangs: spec's gang_count gangs, each drawing layers, dwell, devices, victims per receiving
    mule, receiving mules and start day around the spec's centres, with a family from the committed rules. Refuses the
    mix when its mules and victims exceed the people the world has.

    WHY: Gangs differ within a world but stay near its centre, and a world that cannot supply the people is refused
    loudly, never shrunk silently. Building through gang_config_from_plain applies every existing validator.

    Args:
        spec: The planned world.
        sampler_cfg: The world_sampler dictionary.
        mix_cfg: The gang_mix dictionary.
        eligible_people: People who may be mules or victims (not shop accounts).
        world_ranges: The ranges to read gang-scope parameter ranges from; defaults to the files on disk.

    Returns:
        The GangConfig of each gang, in order.

    Raises:
        ValueError: If a family is not a training family, held-out generation is on, or the world lacks the people.
    """
    families = load_tactic_families()
    validate_gang_mix_config(mix_cfg, set(families["training"]))
    ranges = read_world_configs().world_ranges if world_ranges is None else world_ranges
    centres = _centres(spec, sampler_cfg)
    spread = float(mix_cfg["spread_share_of_range"]["value"])
    rng = gang_mix_rng(spec.seed)

    def bounds(name: str) -> tuple[float, float]:
        """Returns the range a gang-scope parameter is drawn inside, as the sampler mapped its centre."""
        return key_parameter_range(sampler_cfg["key_parameters"][name], spec.range_name, ranges)

    drawn_gangs: list[dict[str, int]] = []
    start_days: list[int] = []
    for _ in range(int(centres["gang_count"])):
        drawn = {
            "layers": int(_draw_around(rng, centres["route_layers"], *bounds("route_layers"), spread, True)),
            "dwell": int(_draw_around(rng, centres["dwell_minutes"], *bounds("dwell_minutes"), spread, True)),
            "devices": int(_draw_around(rng, centres["devices_per_gang"], *bounds("devices_per_gang"), spread, True)),
            "victims": int(_draw_around(
                rng, centres["victims_per_receiving_mule"], *bounds("victims_per_receiving_mule"), spread, True
            )),
        }
        low_mules, high_mules = mix_cfg["receiving_mules"]["range"]
        drawn["receiving_mules"] = int(rng.integers(low_mules, high_mules, endpoint=True))
        low_day, high_day = mix_cfg["start_day_range"]["range"]
        start_days.append(int(rng.integers(low_day, high_day, endpoint=True)))
        drawn_gangs.append(drawn)

    needed = 0
    for drawn in drawn_gangs:
        needed += drawn["layers"] * drawn["receiving_mules"] + drawn["receiving_mules"] * drawn["victims"]
    if needed > eligible_people:
        raise ValueError(
            f"the gang mix of {spec.world_id} needs {needed} people as mules and victims, "
            f"but the world has only {eligible_people} eligible people"
        )

    gangs: list[GangConfig] = []
    for index in range(len(drawn_gangs)):
        drawn = drawn_gangs[index]
        family = family_for_gang(mix_cfg["family_rules"]["rules"], drawn["layers"])
        plain = _gang_plain(index, drawn, start_days[index] * MINUTES_PER_DAY, family, mix_cfg["budget"])
        gangs.append(gang_config_from_plain(plain))
    return tuple(gangs)
