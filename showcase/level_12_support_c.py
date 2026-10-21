"""
WHAT: Support utilities and execution caching for Level 12 Showcase Part C checks.
Loads the Part C gang fixture, caches deterministic simulation execution across checks 11-14,
and provides isolated execution of the pinned bank downtime outage scenario.

WHY: Sharing a single cached simulation run for checks 11 through 14 keeps quick mode fast
while ensuring that all key numbers are measured from real runtime execution artifacts.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: scam gang: An illicit syndicate funded dynamically through victim fraud payments.
TERM: cash-out: Channel through which laundered money exits mule accounts into physical cash or retail.
TERM: cached run: In-memory simulation result reused across multiple check runners to avoid re-execution.
"""

import tempfile
from pathlib import Path

from core.config_validator import load_yaml
from evaluation.truth import TruthStore, load_truth
from showcase.level_12_support_b import get_showcase_b_settings
from world.gang_config import BlockStep, GangBudget, GangConfig, gang_config_from_plain
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.truth_writer import write_truth_files

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
GANG_C_FIXTURE_PATH: Path = REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_victims_cashout.yaml"

# Pinned bank downtime scenario parameters from test_victim_money_route.py
OUTAGE_START_MINUTE = 9145
OUTAGE_GANG_START_MINUTE = 8724
OUTAGE_BANK_INDEX = 1
OUTAGE_ACQUIRE_COUNT = 8

# Module-level singletons for cached simulation runs
_CACHED_C_OUTPUT: InteractiveWorldOutput | None = None
_CACHED_C_STORE: TruthStore | None = None
_CACHED_TEMP_DIR: tempfile.TemporaryDirectory[str] | None = None
_CACHED_OUTAGE_OUTPUT: InteractiveWorldOutput | None = None


def load_showcase_c_fixture(path: Path = GANG_C_FIXTURE_PATH) -> GangConfig:
    """
    WHAT: Loads the Level 12 Part C gang YAML fixture, returning an instantiated GangConfig.
    WHY: Decouples check execution from magic values, ensuring the gang typology originates from config.
    Args:
        path: Filesystem Path pointing to the gang fixture YAML file.
    Returns:
        Validated GangConfig instance configuring the Part C victim scam and cash-out campaign.
    Example:
        gang = load_showcase_c_fixture()
        assert gang.gang_id == "gang_victims_cashout"
    """
    data = load_yaml(path)
    return gang_config_from_plain(data)


def get_showcase_c_run() -> tuple[InteractiveWorldOutput, TruthStore]:
    """
    WHAT: Executes the Level 12 Part C fixture gang on the seed-101 frozen-anchor world and caches results.
    Writes ground truth files into a temporary directory and instantiates a queryable TruthStore.
    WHY: Enforces Decision D2 requirement that checks 11-14 share ONE simulation run to preserve speed.
    Args:
        None.
    Returns:
        Tuple containing InteractiveWorldOutput and TruthStore.
    Example:
        output, store = get_showcase_c_run()
        assert output.result.run_result.event_count > 0
    """
    global _CACHED_C_OUTPUT, _CACHED_C_STORE, _CACHED_TEMP_DIR
    if _CACHED_C_OUTPUT is not None and _CACHED_C_STORE is not None:
        return _CACHED_C_OUTPUT, _CACHED_C_STORE

    gang = load_showcase_c_fixture()
    settings, code_version = get_showcase_b_settings()
    runner_settings = RunnerSettings(fraud_enabled=True)

    _CACHED_TEMP_DIR = tempfile.TemporaryDirectory()
    truth_root = Path(_CACHED_TEMP_DIR.name)
    output = run_interactive_world(
        settings, runner_settings, gang_configs=(gang,), code_version=code_version
    )
    write_truth_files(output, truth_root)
    store = load_truth(truth_root, output.result.run_id)

    _CACHED_C_OUTPUT = output
    _CACHED_C_STORE = store
    return _CACHED_C_OUTPUT, _CACHED_C_STORE


def get_showcase_c_outage_run() -> InteractiveWorldOutput:
    """
    WHAT: Executes the pinned bank downtime outage scenario from test_victim_money_route.py and caches it.
    A scam gang starts at minute 8724 targeting a single victim in bank 1 whose payment hits an outage.
    WHY: Reuses the deterministic outage test case to verify negative forward suppression under Check 15.
    Args:
        None.
    Returns:
        InteractiveWorldOutput documenting failed victim payment and subsequent skipped forward.
    Example:
        out = get_showcase_c_outage_run()
        assert len(out.result.skipped_forwards) > 0
    """
    global _CACHED_OUTAGE_OUTPUT
    if _CACHED_OUTAGE_OUTPUT is not None:
        return _CACHED_OUTAGE_OUTPUT

    settings, code_version = get_showcase_b_settings()
    gang = GangConfig(
        gang_id="gang_victim_outage",
        strength=1,
        budget=GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99),
        source=None,
        steps=(
            BlockStep("acquire", (("count", OUTAGE_ACQUIRE_COUNT),)),
            BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 1))),
            BlockStep("route", (("layers", 2), ("accounts_per_layer", 1))),
        ),
        start_minute=OUTAGE_GANG_START_MINUTE,
    )
    runner_settings = RunnerSettings(fraud_enabled=True)
    _CACHED_OUTAGE_OUTPUT = run_interactive_world(
        settings, runner_settings, gang_configs=(gang,), code_version=code_version
    )
    return _CACHED_OUTAGE_OUTPUT
