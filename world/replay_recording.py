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

from dataclasses import dataclass

from world.bank_connectors import DecisionCall, MockBankConnectors
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


@dataclass(frozen=True)
class RecordedWorld:
    """
    WHAT: Holds the interactive simulation output and the ordered sequence of decision calls generated.
    WHY: Enables replay runs to evaluate decisions deterministically without re-running world simulation.
    """

    output: InteractiveWorldOutput
    calls: tuple[tuple[PaymentView, DecisionCall], ...]


def record_world_for_replay(
    fast_settings: FastRunSettings,
    runner_settings: RunnerSettings,
    connectors: MockBankConnectors,
) -> RecordedWorld:
    """
    WHAT: Runs an interactive world simulation while recording decision requests constructed by connectors.
    WHY: Captures pre-decision payments and DecisionCall payloads for subsequent replay passes,
    refusing live bank modes to prevent accidental external traffic during replay recording.

    Args:
        fast_settings: Simulation settings (seed, population size, shops, days, split).
        runner_settings: Runner settings including the fraud switch.
        connectors: MockBankConnectors used to build DecisionCall requests.

    Returns:
        RecordedWorld containing the full InteractiveWorldOutput and recorded (PaymentView, DecisionCall) pairs.

    Raises:
        ValueError: If any bank mode in connectors.modes is 'live'.
    """
    for mode in connectors.modes.values():
        if mode == "live":
            raise ValueError("Live bank mode refused until Level 19")

    recorded: list[tuple[PaymentView, DecisionCall]] = []

    def recording_hook(payment: PaymentView) -> HookDecision:
        call = connectors.build_decision_request(payment)
        recorded.append((payment, call))
        return "allow"

    output = run_interactive_world(fast_settings, runner_settings, hook=recording_hook)
    return RecordedWorld(output=output, calls=tuple(recorded))
