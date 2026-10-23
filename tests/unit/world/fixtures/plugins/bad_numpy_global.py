# Fixture: a plugin that seeds numpy's global generator through an alias. The lint must refuse it before import.
import numpy as xx

from world.tactic_plugins import PluginContext, PluginStepResult, TacticPlugin

raise RuntimeError("fixture code must never run: the lint should have refused this file")


def plan(context: PluginContext) -> PluginStepResult:
    xx.random.seed(1)
    return PluginStepResult(int(xx.random.default_rng().integers(1, 5)), 0)


PLUGIN = TacticPlugin("bad_numpy_global", "x", (1,), {}, (0.0, 0.0), "assumption", "x", plan)
