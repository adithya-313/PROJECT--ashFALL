"""
File: tests/lazy_script.py
Purpose: Loads a script from scripts/ (or serving/) only when a test first uses it, instead of when pytest collects
         the test file. If one of the named libraries is missing at that moment, the test is skipped.

Why it exists: Some scripts import heavy machine-learning libraries that only requirements-ml.txt installs. With a
normal import at the top of the test file, the whole file is skipped (or fails to collect) wherever those libraries are
missing, even when the test is deselected with a marker. A lazy load lets CI leave such tests out cleanly with -m.
"""
import importlib.util
from pathlib import Path

import pytest


class LazyScript:
    """Stands in for a script module. The first attribute that is read or set loads the real module."""

    def __init__(self, path: Path, libraries: tuple[str, ...] = ()) -> None:
        # Our own fields go straight into __dict__, so they do not trigger the loading below.
        self.__dict__["_lazy_path"] = Path(path)
        self.__dict__["_lazy_libraries"] = libraries
        self.__dict__["_lazy_module"] = None

    def _lazy_load(self):
        """Load the script once. Skip the running test if a needed library is not installed."""
        if self.__dict__["_lazy_module"] is None:
            for library in self.__dict__["_lazy_libraries"]:
                pytest.importorskip(library)
            path = self.__dict__["_lazy_path"]
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.__dict__["_lazy_module"] = module
        return self.__dict__["_lazy_module"]

    def __getattr__(self, name):
        # pytest itself looks at every name in a test file (for example "_pytestfixturefunction" and "__test__")
        # while collecting. Those lookups must not load the script, or collection would skip the whole file.
        if name.startswith("__") or name.startswith("_pytest") or name == "pytestmark":
            raise AttributeError(name)
        return getattr(self._lazy_load(), name)

    def __setattr__(self, name, value):
        # monkeypatch.setattr(script, "name", value) must change the real module, not this stand-in.
        setattr(self._lazy_load(), name, value)

    def __delattr__(self, name):
        delattr(self._lazy_load(), name)


def lazy_script(path: Path, libraries: tuple[str, ...] = ()) -> LazyScript:
    """Shorter way to build a LazyScript."""
    return LazyScript(path, libraries)
