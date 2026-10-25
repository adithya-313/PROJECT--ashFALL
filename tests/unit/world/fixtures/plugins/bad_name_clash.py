# Fixture: a clean plugin whose name equals the built-in block "route". Loading must be refused for the name.
from world.tactic_plugins import PluginContext, PluginStepResult, TacticPlugin


def plan(context: PluginContext) -> PluginStepResult:
    return PluginStepResult(0, 0)


PLUGIN = TacticPlugin("route", "x", (1,), {}, (0.0, 0.0), "assumption", "A name that clashes with a built-in.", plan)
