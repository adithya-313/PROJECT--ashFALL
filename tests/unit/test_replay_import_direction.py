# File: tests/unit/test_replay_import_direction.py
# Purpose: Static AST guard ensuring decoupled import direction for replay.
"""
WHAT: Static AST guard verifying one-way dependency boundaries between replay and core engine packages.
Ensures that no module in api/, features/, world/, or core/ imports replay, and that replay never imports evaluation.

WHY: Architect decision 16b P1: replay is a downstream consumer of simulation outputs and operational APIs.
Core packages and simulation engines must remain independent of replay infrastructure.
"""

import ast
from pathlib import Path


def test_no_operational_package_imports_replay() -> None:
    """Walk every .py file under api/, features/, world/, and core/, failing if any imports replay."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    checked_dirs = ["api", "features", "world", "core"]
    violations: list[str] = []

    for dir_name in checked_dirs:
        pkg_dir = repo_root / dir_name
        py_files = sorted(pkg_dir.rglob("*.py"))
        assert py_files, f"Expected to find .py files under {dir_name}/"

        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "replay" or alias.name.startswith("replay."):
                            violations.append(
                                f"{py_file.relative_to(repo_root)}:{node.lineno}: import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module == "replay" or node.module.startswith("replay.")):
                        violations.append(
                            f"{py_file.relative_to(repo_root)}:{node.lineno}: from {node.module} import ..."
                        )

    assert not violations, f"Modules in api/, features/, world/, core/ must not import replay (found: {violations})"


def test_replay_does_not_import_evaluation() -> None:
    """Walk every .py file under replay/, failing if any imports evaluation."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    replay_dir = repo_root / "replay"
    py_files = sorted(replay_dir.rglob("*.py"))
    assert py_files, "Expected to find .py files under replay/"

    violations: list[str] = []
    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "evaluation" or alias.name.startswith("evaluation."):
                        violations.append(
                            f"{py_file.relative_to(repo_root)}:{node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "evaluation" or node.module.startswith("evaluation.")):
                    violations.append(
                        f"{py_file.relative_to(repo_root)}:{node.lineno}: from {node.module} import ..."
                    )

    assert not violations, f"replay/ modules must not import evaluation (found: {violations})"
