"""
Unit tests for truth isolation guard on decisions package.
Verifies runtime caller rejection by evaluation.truth.load_truth and AST static scan of decisions/ files.
"""

import ast
import types
from pathlib import Path

import pytest

from evaluation.truth import load_truth

REPO_ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = ("evaluation",)


def _forbidden_imports(path: Path) -> list[str]:
    """
    PURPOSE:
        Scans a Python source file using AST for any forbidden imports from the evaluation package.

    INPUTS:
        path: Path to source code file.

    KEY LOGIC:
        Inspects ast.Import and ast.ImportFrom nodes. Flags any import whose top-level module
        matches or starts with an element of FORBIDDEN.

    RETURNS:
        list[str]: Offending import location and module strings (empty if clean).
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


def test_decisions_files_do_not_import_evaluation() -> None:
    """Verifies that no Python file under decisions/ imports from the evaluation package."""
    files = list((REPO_ROOT / "decisions").rglob("*.py"))
    assert files, "Expected source files under decisions/"
    offenders = [hit for p in files for hit in _forbidden_imports(p)]
    assert offenders == []


def test_load_truth_refuses_a_decisions_module_caller(tmp_path: Path) -> None:
    """Verifies calling evaluation.truth.load_truth from a decisions.* module raises PermissionError."""
    fake = types.ModuleType("decisions.fake_caller")
    exec("def go(fn, root):\n    return fn(root, 'any_run')\n", fake.__dict__)
    with pytest.raises(PermissionError):
        fake.go(load_truth, tmp_path)

