"""
WHAT: Automated test suite running docstring examples across Level 11B world modules.
WHY: Guarantees that interactive docstring examples remain accurate, functional, and up to date.
TERM: doctest: An interactive Python example embedded in a docstring verifying expected output,
      e.g. checking function return values in interactive shell sessions.
TERM: example: A concrete worked illustration demonstrating typical input arguments and outputs,
      e.g. showing function invocation with sample numbers.
"""

import doctest
import importlib

DOCTEST_MODULES: list[str] = [
    "world.background_config",
    "world.clock",
    "world.festivals",
    "world.rhythms",
    "world.failures",
    "world.incidents",
    "world.failure_decision",
    "world.device_events",
    "world.device_schedule",
    "world.episodes",
]


def _run_module_doctests(module_name: str) -> tuple[int, int]:
    """
    WHAT: Imports a target world simulation module and executes all embedded doctest examples.
    It counts both the total number of attempted tests and the number of failures encountered.
    WHY: Encapsulates dynamic module inspection and isolation of doctest results.
    This keeps individual module testing uniform without repeating import and execution logic.

    Args:
        module_name: Fully-qualified dot-separated Python module string to test.

    Returns:
        Tuple of (attempted, failed) count of doctest examples executed.

    Example:
        >>> from tests.unit.world.test_world_doctests import _run_module_doctests
        >>> _run_module_doctests("world.background_config")
        (9, 0)
    """
    # Dynamically import the target module using its fully-qualified package path.
    module = importlib.import_module(module_name)
    # Execute all interactive docstring examples without permissive optionflags.
    result = doctest.testmod(module)
    # Pass back the total count of executed docstring examples and failed assertions.
    return (result.attempted, result.failed)


def test_world_doctests_pass() -> None:
    """
    WHAT: Runs doctest verification across all Level 11B world simulation modules.
    It verifies that each module contains working examples and asserts that zero failures occur.
    WHY: Ensures documentation remains synchronized with underlying implementation behavior.
    It guarantees that copy-paste examples in docstrings stay accurate across code updates.

    Args:
        None.

    Returns:
        None.

    Example:
        >>> from tests.unit.world.test_world_doctests import test_world_doctests_pass
        >>> test_world_doctests_pass()
    """
    # Initialize tracking accumulators for attempted doctests, failures, and defective modules.
    total_attempted = 0
    total_failed = 0
    failing_modules: list[str] = []
    n_mods = len(DOCTEST_MODULES)

    # Execute doctests across all configured Level 11B world simulation modules.
    for mod_name in DOCTEST_MODULES:
        attempted, failed = _run_module_doctests(mod_name)
        total_attempted += attempted
        total_failed += failed
        # Flag any module that has failures or completely lacks doctest examples.
        if failed > 0 or attempted == 0:
            failing_modules.append(mod_name)

    # Output the measured summary of executed examples across all checked modules.
    print(f"MEASURED: ran {total_attempted} doctest examples across {n_mods} modules, {total_failed} failed")

    # Assert that all modules executed successfully without failures or missing tests.
    assert not failing_modules, f"Doctest failures or missing examples in modules: {', '.join(failing_modules)}"
