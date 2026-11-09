"""
WHAT: Tests that the dataset truth join stays out of operational code: a static scan of replay/ and
features/ for imports of the truth modules, and a run-time check that load_truth refuses a caller
whose module name starts with "replay.".

WHY: Hidden truth must never reach the detector or its feature code; this pins that boundary for the
new join module on top of the repository-wide guard in core/truth_import_check.py.

TERM: static scan: reading source with the ast module without running it.
TERM: direct caller: the module whose code line calls load_truth.
"""

import ast
import types
from pathlib import Path

import pytest

from evaluation.truth import load_truth

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("evaluation.dataset_truth_join", "evaluation.truth")


def _forbidden_imports(path: Path) -> list[str]:
    """
    WHAT: Lists forbidden truth imports found in one source file.
    WHY: Covers `import a.b`, `from a.b import c` and `from evaluation import truth`.

    Args:
        path: Python source file.

    Returns:
        Human-readable offending import strings (empty when clean).

    Example:
        # _forbidden_imports(Path("replay/ordering.py")) == []
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        for name in names:
            if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN):
                found.append(f"{path.relative_to(REPO_ROOT)}: {name}")
    return found


def test_replay_and_features_do_not_import_truth_modules() -> None:
    """WHAT: no file under replay/ or features/ imports the truth join or truth loader. WHY: keeps truth out."""
    files = [p for folder in ("replay", "features") for p in (REPO_ROOT / folder).rglob("*.py")]
    assert files, "expected source files under replay/ and features/"
    offenders = [hit for p in files for hit in _forbidden_imports(p)]
    assert offenders == []


def test_load_truth_refuses_a_replay_module_caller(tmp_path: Path) -> None:
    """WHAT: load_truth called from a module named replay.* raises PermissionError. WHY: run-time guard."""
    fake = types.ModuleType("replay.fake_caller")
    exec("def go(fn, root):\n    return fn(root, 'any_run')\n", fake.__dict__)  # noqa: S102
    with pytest.raises(PermissionError):
        fake.go(load_truth, tmp_path)
