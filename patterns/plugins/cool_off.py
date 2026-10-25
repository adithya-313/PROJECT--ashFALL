"""
WHAT: Example tactic plugin "cool_off": the gang waits a number of days before its next step. It advances the gang's
clock and costs no money; it makes no payments.

WHY: Shows the smallest useful plugin, a timing-only block. Waiting is a training-safe tactic: it uses only the gang's
own random stream (context.rng), no clock, no files and no world modules.

TERM: cool-off: a pause in a gang's campaign before the next step, measured in whole days.
"""

from world.tactic_plugins import PluginContext, PluginStepResult, TacticPlugin

MINUTES_PER_DAY = 1440


def plan(context: PluginContext) -> PluginStepResult:
    """Advances the clock by the given days, or by days drawn from the gang's stream when none are given."""
    bounds = PLUGIN.parameters["days"]["range"]
    if "days" in context.params:
        days = int(context.params["days"])
    else:
        days = int(context.rng.integers(bounds[0], bounds[1], endpoint=True))
    return PluginStepResult(minutes_advanced=days * MINUTES_PER_DAY, money_paise=0)


PLUGIN = TacticPlugin(
    name="cool_off",
    meaning="The gang waits some days before its next step",
    allowed_strengths=(1, 2),
    parameters={"days": {"range": [1, 7]}},
    risk_per_use=(0.0, 0.001),
    label="assumption",
    reason="No published figure; a pause of one to seven days is an assumption for a timing-only example block.",
    plan=plan,
)
