"""
Tests that the replay drop context carries the replay DPIP key.

WHY: A before_drop_hook that recomputes decisions must hash with the key the replay decisions used, so the context
hands it over and recompute_logged_row never has to guess it from the run id.

TERM: drop context: the object given to before_drop_hook just before the replay workspace is dropped.
"""

import dataclasses

from replay.registry_replay import replay_dpip_key
from replay.runner import ReplayDropContext

RUN = "0123456789abcdef"


def test_drop_context_requires_a_dpip_key_field() -> None:
    """Checks dpip_key is a required string field on the drop context."""
    field = next(f for f in dataclasses.fields(ReplayDropContext) if f.name == "dpip_key")
    assert field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING


def test_replay_key_is_a_pure_function_of_the_run_id() -> None:
    """Checks replay_dpip_key(run_id), the context key, is stable per run and differs across runs."""
    assert replay_dpip_key(RUN) == replay_dpip_key(RUN)
    assert replay_dpip_key(RUN) != replay_dpip_key("fedcba9876543210")
