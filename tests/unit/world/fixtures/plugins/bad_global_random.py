# Fixture: a plugin that uses the global random module. The lint must refuse it before it is imported.
import random

from world.tactic_plugins import PluginContext, PluginStepResult, TacticPlugin

raise RuntimeError("fixture code must never run: the lint should have refused this file")


def plan(context: PluginContext) -> PluginStepResult:
    return PluginStepResult(random.randint(1, 5), 0)


PLUGIN = TacticPlugin("bad_global_random", "x", (1,), {}, (0.0, 0.0), "assumption", "x", plan)
