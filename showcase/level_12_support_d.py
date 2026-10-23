"""
WHAT: Support utilities and execution caching for Level 12 Showcase Part D checks (16 through 20).
Loads the Part D gang fixtures, and runs the account market world, the training family world, the delayed lien
world, and the held-out family refusal, caching each run so checks that share a world run it once.

WHY: Part D shows a world that can react: gangs buy burned accounts from each other, observe liens late, and are
checked against pre-registered tactic families. Every key number is measured from a real run of the fixtures, and the
fixtures keep the scenarios in files instead of in check code.

TERM: account market: resale of a burned rented mule account from one gang to a later gang.
TERM: feed delay: minutes between a lien being placed and a gang learning of it.
TERM: tactic family: one named way a mule gang moves stolen money, defined in config/tactic_families.yaml.
TERM: cached run: an in-memory simulation result reused across check runners to avoid running it twice.
"""

import tempfile
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from evaluation.truth import TruthStore, load_truth
from showcase.level_12_support_b import get_showcase_b_settings
from world.gang_config import GangConfig, gang_config_from_plain
from world.interactive_records import FraudInputs, HookDecision, InteractiveWorldOutput, PaymentView, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.outside_events import OutsideEvent
from world.truth_writer import write_truth_files

FIXTURE_DIR: Path = Path(__file__).resolve().parent / "fixtures" / "level_12"
MARKET_FIXTURE_PATH: Path = FIXTURE_DIR / "gangs_market_two.yaml"
FAMILIES_FIXTURE_PATH: Path = FIXTURE_DIR / "gangs_families_training.yaml"
HELD_OUT_FIXTURE_PATH: Path = FIXTURE_DIR / "gang_held_out_in_training.yaml"
CODE_VERSION_D: str = "showcase-12d"

_CACHED_MARKET: tuple[InteractiveWorldOutput, TruthStore] | None = None
_CACHED_FAMILIES: tuple[tuple[GangConfig, ...], InteractiveWorldOutput] | None = None
_CACHED_LIEN: tuple[InteractiveWorldOutput, OutsideEvent, int] | None = None
_CACHED_TEMP_DIR: tempfile.TemporaryDirectory[str] | None = None


def load_gangs(path: Path) -> tuple[GangConfig, ...]:
    """
    WHAT: Loads every gang listed under the 'gangs' key of a Part D fixture, or the single gang under 'gang'.
    WHY: Keeps the gang typologies in fixture files; each gang goes through the same validation as any gang config.
    Args:
        path: Filesystem Path of a fixture YAML file.
    Returns:
        The validated GangConfig records in file order.
    Example:
        gangs = load_gangs(MARKET_FIXTURE_PATH)
        assert [g.gang_id for g in gangs] == ["g1", "g2"]
    """
    data = load_yaml(path)
    if "gangs" in data:
        return tuple(gang_config_from_plain(entry) for entry in data["gangs"])
    return (gang_config_from_plain(data),)


def load_market_section(name: str) -> dict[str, Any]:
    """
    WHAT: Returns one configuration section ('account_market', 'gang_reactions' or 'lien') of the market fixture.
    WHY: The market and reaction configs are passed to the runner as dictionaries; they live in the fixture file.
    Args:
        name: Section name in gangs_market_two.yaml.
    Returns:
        The section as a dictionary.
    Raises:
        ValueError: If the fixture has no such section.
    Example:
        assert load_market_section("lien")["minute"] == 3000
    """
    data = load_yaml(MARKET_FIXTURE_PATH)
    if name not in data:
        raise ValueError(f"Fixture {MARKET_FIXTURE_PATH.name} has no section '{name}'.")
    return data[name]


def get_market_run() -> tuple[InteractiveWorldOutput, TruthStore]:
    """
    WHAT: Runs both market fixture gangs on the seed-101 frozen-anchor world with the account market on and
    a resale share of 1, writes the truth files to a temporary folder, and loads a queryable TruthStore.
    WHY: Check 16 needs the resales and the truth owner of the sold accounts from one real run.
    Args:
        None.
    Returns:
        The InteractiveWorldOutput and the TruthStore loaded from its truth files.
    Example:
        output, store = get_market_run()
        assert len(output.result.account_resales) >= 1
    """
    global _CACHED_MARKET, _CACHED_TEMP_DIR
    if _CACHED_MARKET is not None:
        return _CACHED_MARKET
    settings, _ = get_showcase_b_settings()
    output = run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True, account_market_enabled=True),
        gang_configs=load_gangs(MARKET_FIXTURE_PATH),
        account_market_config={"account_market": load_market_section("account_market")},
        code_version=CODE_VERSION_D,
    )
    _CACHED_TEMP_DIR = tempfile.TemporaryDirectory()
    truth_root = Path(_CACHED_TEMP_DIR.name)
    write_truth_files(output, truth_root)
    _CACHED_MARKET = (output, load_truth(truth_root, output.result.run_id))
    return _CACHED_MARKET


def get_families_run() -> tuple[tuple[GangConfig, ...], InteractiveWorldOutput]:
    """
    WHAT: Runs the training-family fixture gangs in the seed-101 training world.
    WHY: Checks 17 and 18 share one run: the first shows no held-out family got in, the second counts by family.
    Args:
        None.
    Returns:
        The gang configs and the InteractiveWorldOutput.
    Example:
        gangs, output = get_families_run()
        assert len(output.result.run_result.events) > 0
    """
    global _CACHED_FAMILIES
    if _CACHED_FAMILIES is not None:
        return _CACHED_FAMILIES
    settings, _ = get_showcase_b_settings()
    gangs = load_gangs(FAMILIES_FIXTURE_PATH)
    output = run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True), gang_configs=gangs, code_version=CODE_VERSION_D
    )
    _CACHED_FAMILIES = (gangs, output)
    return _CACHED_FAMILIES


def get_lien_run() -> tuple[InteractiveWorldOutput, OutsideEvent, int]:
    """
    WHAT: Runs the first market fixture gang with gang reactions on and the fixture lien placed on one of its mules.
    The feed delay comes from the fixture's gang_reactions section.
    WHY: Check 19 shows money flowing through the liened mule until the gang learns of the lien, then stopping.
    Args:
        None.
    Returns:
        The InteractiveWorldOutput, the lien event, and the feed delay D in minutes.
    Raises:
        ValueError: If the fixture delay range is not a single value for the training split.
    Example:
        output, lien, delay = get_lien_run()
        assert lien.kind == "lien" and delay > 0
    """
    global _CACHED_LIEN
    if _CACHED_LIEN is not None:
        return _CACHED_LIEN
    settings, _ = get_showcase_b_settings()
    reactions = {"gang_reactions": load_market_section("gang_reactions")}
    low_delay, high_delay = reactions["gang_reactions"]["gang_feed_delay_minutes"][settings.range_name]
    if low_delay != high_delay:
        raise ValueError("The fixture feed delay must be a single value so the observation minute is exact.")
    section = load_market_section("lien")
    lien = OutsideEvent(
        minute=int(section["minute"]), kind="lien", account_id=str(section["account_id"]),
        amount_paise=int(section["amount_paise"]), encumbrance_id=str(section["encumbrance_id"]),
    )
    output = run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=FraudInputs(outside_events=(lien,)), gang_configs=load_gangs(MARKET_FIXTURE_PATH)[:1],
        gang_reactions_config=reactions, code_version=CODE_VERSION_D,
    )
    _CACHED_LIEN = (output, lien, int(low_delay))
    return _CACHED_LIEN


def run_held_out_in_training() -> tuple[GangConfig, str, int]:
    """
    WHAT: Runs the held-out-family fixture gang in the seed-101 training world and returns the refusal.
    A counting hook and a counting state factory prove that nothing was built or processed before the refusal.
    WHY: Check 20 shows that a training world file naming a held-out family is refused before any planning.
    Args:
        None.
    Returns:
        The gang config, the plain ValueError text, and the number of hook and state factory calls (expected 0).
    Raises:
        AssertionError: If the run is not refused.
    Example:
        gang, message, calls = run_held_out_in_training()
        assert calls == 0 and gang.gang_id in message
    """
    settings, _ = get_showcase_b_settings()
    gang = load_gangs(HELD_OUT_FIXTURE_PATH)[0]
    calls: list[str] = []

    def counting_hook(view: PaymentView) -> HookDecision:
        """Records that a payment reached the bank hook and lets it through."""
        calls.append("hook")
        return "allow"

    def counting_state_factory(prepared: Any) -> Any:
        """Records that the runner state was built, which only happens after the family check."""
        calls.append("state")
        raise AssertionError("the runner state must not be built before the family check")

    try:
        run_interactive_world(
            settings, RunnerSettings(fraud_enabled=True), gang_configs=(gang,), hook=counting_hook,
            state_factory=counting_state_factory, code_version=CODE_VERSION_D,
        )
    except ValueError as error:
        return gang, str(error), len(calls)
    raise AssertionError("a held-out family in a training world was not refused")
