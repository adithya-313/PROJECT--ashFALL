# File: world/replay_recording.py
# Purpose: Pre-decision recording of interactive world simulations for replay runs.
"""
WHAT: Pre-decision recording of interactive world simulations for replay runs.
Runs interactive world simulations with a recording hook that builds DecisionCall requests
without invoking bank transports.

WHY: Architect decision 16b D4: replay evaluation requires exact recorded sequences of
PaymentView and DecisionCall instances produced by the world simulation. Running with
an intercepting hook that always allows captures the unaltered baseline run while
refusing live bank modes to prevent accidental external traffic.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from world.bank_connectors import DecisionCall, MockBankConnectors
from world.gang_config import GangConfig
from world.interactive_records import (
    HookDecision,
    InteractiveWorldOutput,
    PaymentView,
)
from world.interactive_runner import (
    FastRunSettings,
    RunnerSettings,
    run_interactive_world,
)
from world.phone_book import PhoneBook, build_phone_book
from world.sampled_world import (
    pinned_configs,
    resolve_sampled_world_gangs,
    run_sampled_world,
)
from world.world_sampler import WorldSpec


@dataclass(frozen=True)
class RecordedWorld:
    """
    WHAT: Holds the interactive simulation output and the ordered sequence of decision calls generated.
    WHY: Enables replay runs to evaluate decisions deterministically without re-running world simulation.
    The phone book is built once here by world.phone_book.build_phone_book on its own seeded stream.
    """

    output: InteractiveWorldOutput
    calls: tuple[tuple[PaymentView, DecisionCall], ...]
    world_id: str | None = None
    split: str | None = None
    seed: int | None = None
    gang_families: tuple[str | None, ...] | None = None
    phone_book: PhoneBook | None = None


def _check_no_live_bank_modes(connectors: MockBankConnectors) -> None:
    """Refuses live bank modes to prevent accidental external traffic during replay recording."""
    for mode in connectors.modes.values():
        if mode == "live":
            raise ValueError("Live bank mode refused until Level 19")


def _make_recording_hook(
    connectors: MockBankConnectors,
    recorded: list[tuple[PaymentView, DecisionCall]],
) -> Callable[[PaymentView], HookDecision]:
    """Builds the intercepting hook that accumulates (PaymentView, DecisionCall) pairs and returns 'allow'."""

    def recording_hook(payment: PaymentView) -> HookDecision:
        call = connectors.build_decision_request(payment)
        recorded.append((payment, call))
        return "allow"

    return recording_hook


def record_world_for_replay(
    fast_settings: FastRunSettings,
    runner_settings: RunnerSettings,
    connectors: MockBankConnectors,
    *,
    gang_configs: tuple[GangConfig, ...] = (),
    world_split: str | None = None,
) -> RecordedWorld:
    """
    WHAT: Runs an interactive world simulation while recording decision requests constructed by connectors.
    WHY: Captures pre-decision payments and DecisionCall payloads for subsequent replay passes,
    refusing live bank modes to prevent accidental external traffic during replay recording.

    Args:
        fast_settings: Simulation settings (seed, population size, shops, days, split).
        runner_settings: Runner settings including the fraud switch.
        connectors: MockBankConnectors used to build DecisionCall requests.
        gang_configs: Optional tuple of GangConfig instances executing when fraud switch is on.
        world_split: Optional world split identifier (e.g. 'training').

    Returns:
        RecordedWorld containing the full InteractiveWorldOutput and recorded (PaymentView, DecisionCall) pairs.

    Raises:
        ValueError: If any bank mode in connectors.modes is 'live'.
    """
    _check_no_live_bank_modes(connectors)

    recorded: list[tuple[PaymentView, DecisionCall]] = []
    hook = _make_recording_hook(connectors, recorded)

    output = run_interactive_world(
        fast_settings,
        runner_settings,
        hook=hook,
        gang_configs=gang_configs,
        world_split=world_split,
    )
    return RecordedWorld(
        output=output,
        calls=tuple(recorded),
        world_id=output.result.run_id,
        split=world_split if world_split is not None else fast_settings.range_name,
        seed=fast_settings.seed,
        gang_families=tuple(g.family for g in gang_configs),
        phone_book=build_phone_book(output, run_seed=fast_settings.seed),
    )


def record_sampled_world_for_replay(
    spec: WorldSpec,
    preset_name: str,
    gang_configs: tuple[GangConfig, ...] | None,
    sampler_cfg: dict[str, Any],
    code_version: str,
    connectors: MockBankConnectors,
) -> RecordedWorld:
    """
    WHAT: Runs a sampled world simulation while recording decision requests constructed by connectors.
    WHY: Implements Decision D4. Enables replay execution across sampled multi-world partitions,
    recording gang families and world metadata without reading from the hidden truth store.

    Args:
        spec: Sampled world specification.
        preset_name: World sizing preset name (e.g. 'tiny').
        gang_configs: Explicit gangs tuple (at least one), or None to generate the mix from spec.
        sampler_cfg: Dictionary loaded from config/world_sampler.yaml.
        code_version: Code version identifier recorded in run_id.
        connectors: MockBankConnectors used to build DecisionCall requests.

    Returns:
        RecordedWorld containing output, calls, world_id, split, seed, and gang_families.

    Raises:
        ValueError: If any bank mode in connectors.modes is 'live'.
    """
    _check_no_live_bank_modes(connectors)

    configs = pinned_configs(spec, sampler_cfg)
    gangs = resolve_sampled_world_gangs(spec, preset_name, gang_configs, sampler_cfg, configs)
    gang_families: tuple[str | None, ...] = tuple(g.family for g in gangs)

    recorded: list[tuple[PaymentView, DecisionCall]] = []
    hook = _make_recording_hook(connectors, recorded)

    output = run_sampled_world(
        spec=spec,
        preset_name=preset_name,
        gang_configs=gangs,
        sampler_cfg=sampler_cfg,
        code_version=code_version,
        hook=hook,
    )
    return RecordedWorld(
        output=output,
        calls=tuple(recorded),
        world_id=spec.world_id,
        split=spec.split,
        seed=spec.seed,
        gang_families=gang_families,
        phone_book=build_phone_book(output, run_seed=spec.seed),
    )
