"""
Unit test suite for decisions package.
"""

from pathlib import Path

# Extend package search path so production decisions package modules are discoverable
# when pytest adds tests/unit to sys.path.
_prod_decisions = Path(__file__).resolve().parents[3] / "decisions"
if _prod_decisions.is_dir() and str(_prod_decisions) not in __path__:
    __path__.append(str(_prod_decisions))

