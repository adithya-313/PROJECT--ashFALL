# File: tests/unit/test_api_does_not_import_world.py
# Purpose: Guard tests ensuring the API package remains decoupled from the world simulator.

import ast
from pathlib import Path

from api.v1.upi_outbox import UPI_FAILURE_DOMAINS
from world.failures import FAILURE_DOMAINS


def test_upi_failure_domains_match_the_world() -> None:
    """Verify that the API UPI failure domains match world.failures FAILURE_DOMAINS exactly."""
    assert tuple(FAILURE_DOMAINS) == UPI_FAILURE_DOMAINS


def test_no_api_module_imports_world() -> None:
    """Walk every .py file under api/, parse it with ast, and fail if any Import/ImportFrom names world."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    api_dir = repo_root / "api"
    py_files = sorted(api_dir.rglob("*.py"))
    assert py_files, "Expected to find .py files under api/"

    violations: list[str] = []
    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "world" or alias.name.startswith("world."):
                        violations.append(f"{py_file.relative_to(repo_root)}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "world" or node.module.startswith("world.")):
                    violations.append(f"{py_file.relative_to(repo_root)}:{node.lineno}: from {node.module} import ...")

    assert not violations, f"api/ modules must not import world (found: {violations})"

