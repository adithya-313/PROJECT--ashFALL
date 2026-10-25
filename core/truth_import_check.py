"""
WHAT: Static AST and source code guard enforcing architectural isolation of hidden truth simulation artifacts.
Scans Python files to ensure that operational banking packages never import the evaluation truth loader and
never reference the hidden truth directory environment variable.

WHY: Fraud simulation ground truth contains illicit role labels, gang identities, and transaction narratives.
Under Ground Rule 15 and Level 12b architecture, banking models, feature stores, and API serving pipelines must
never have access to synthetic labels. This static checker runs in CI to guarantee that isolation cannot regress.

TERM: static AST guard: analysis of Python syntax trees without executing runtime code.
TERM: hidden truth: ground-truth records describing illicit roles, device controls, and payment stories.
TERM: operational package: production banking services (api, serving, models, core, features, online_learning, agent).
TERM: allowed reader folders: explicitly permitted directories (evaluation, showcase, tests) allowed to inspect truth.
"""

import ast
import sys
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.run_identity import REPO_ROOT


def _load_truth_policy(repo_root: Path) -> dict[str, Any]:
    """
    WHAT: Loads the truth_store policy configuration from truth_store.yaml.
    Falls back to REPO_ROOT config if the specified repo_root does not contain it.

    WHY: Decouples static rules from hardcoded package names, ensuring policy stays synchronized with YAML config.

    Args:
        repo_root: Filesystem Path pointing to the target repository root.

    Returns:
        Dictionary holding the 'truth_store' configuration section.

    Example:
        # policy = _load_truth_policy(REPO_ROOT)
        # "evaluation" in policy["allowed_reader_folders"]
    """
    # 1. Locate configuration file in target repo_root or fallback to REPO_ROOT.
    target_config = repo_root / "config" / "truth_store.yaml"
    config_path = target_config if target_config.exists() else (REPO_ROOT / "config" / "truth_store.yaml")

    # 2. Parse and return truth store policy section.
    return load_yaml(config_path)["truth_store"]


def _is_skipped_path(relative_path: Path) -> bool:
    """
    WHAT: Determines whether a file path should be excluded from truth isolation scans.
    Skips version control (.git), virtual environments (.venv*), and build reports (reports/).

    WHY: Avoids scanning external library dependencies or ephemeral output directories.

    Args:
        relative_path: Relative filesystem Path from repository root.

    Returns:
        Boolean True if path should be skipped, False otherwise.

    Example:
        # _is_skipped_path(Path(".git/HEAD")) is True
    """
    # 1. Inspect path parts for excluded directory names.
    for part in relative_path.parts:
        if part == ".git" or part == "reports" or part.startswith(".venv"):
            return True
    return False


def _check_file_imports(relative_path: Path, tree: ast.AST) -> list[str]:
    """
    WHAT: Inspects AST Import, ImportFrom, and dynamic import Call nodes for unauthorized imports of 'evaluation'.

    WHY: Files outside allowed reader folders must never import truth loader modules statically or dynamically.

    Args:
        relative_path: Relative path of the file being inspected.
        tree: Parsed AST module tree.

    Returns:
        List of formatted violation strings.

    Example:
        # violations = _check_file_imports(Path("api/main.py"), parsed_ast)
    """
    # 1. Traverse syntax tree to detect forbidden evaluation imports.
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "evaluation" or alias.name.startswith("evaluation."):
                    violations.append(
                        f"{relative_path.as_posix()}:{node.lineno}: "
                        f"forbidden import '{alias.name}' outside allowed reader folders"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "evaluation" or node.module.startswith("evaluation."):
                violations.append(
                    f"{relative_path.as_posix()}:{node.lineno}: "
                    f"forbidden import from '{node.module}' outside allowed reader folders"
                )
        elif isinstance(node, ast.Call):
            is_import_call = False
            if isinstance(node.func, ast.Name) and node.func.id in ("import_module", "__import__"):
                is_import_call = True
            elif isinstance(node.func, ast.Attribute) and node.func.attr in ("import_module", "__import__"):
                is_import_call = True
            if is_import_call and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    if first_arg.value.startswith("evaluation"):
                        violations.append(
                            f"{relative_path.as_posix()}:{node.lineno}: "
                            f"forbidden dynamic import '{first_arg.value}' outside allowed reader folders"
                        )
    return violations


def _check_file_env_var(relative_path: Path, top_folder: str, env_var: str, lines: list[str]) -> list[str]:
    """
    WHAT: Scans source lines of an operational package module for references to the hidden truth env variable.

    WHY: Production banking packages must never read or configure truth storage locations.

    Args:
        relative_path: Relative path of the inspected file.
        top_folder: Top-level package folder name.
        env_var: Target environment variable string defined in truth_store policy.
        lines: Source code lines of the file.

    Returns:
        List of formatted violation strings.

    Example:
        # violations = _check_file_env_var(Path("api/main.py"), "api", "TRUTH_DIR", ["..."])
    """
    # 1. Check line-by-line for occurrences of the forbidden environment variable string.
    violations: list[str] = []
    for line_index, line in enumerate(lines, start=1):
        if env_var in line:
            violations.append(
                f"{relative_path.as_posix()}:{line_index}: "
                f"operational package '{top_folder}' references forbidden truth env var '{env_var}'"
            )
    return violations


def find_truth_violations(repo_root: Path = REPO_ROOT) -> list[str]:
    """
    WHAT: Traverses all Python files in the repository to identify unauthorized truth imports or env references.
    Asserts that only allowed_reader_folders import evaluation.truth, and no operational packages mention truth env var.

    WHY: Core gate ensuring architectural separation between synthetic simulation
    ground truth and operational bank code.

    Args:
        repo_root: Root directory of the repository to audit (defaults to REPO_ROOT).

    Returns:
        List of diagnostic violation messages; empty list if repository is fully compliant.

    Example:
        # violations = find_truth_violations()
        # assert len(violations) == 0
    """
    # 1. Load truth store policy configurations.
    truth_config = _load_truth_policy(repo_root)
    allowed_folders = set(truth_config.get("allowed_reader_folders", ["evaluation", "showcase", "tests"]))
    ashfall_packages = set(truth_config.get("ashfall_packages", []))
    env_var = truth_config.get("env_var", "")

    # 2. Iterate across all Python files in repository.
    violations: list[str] = []
    for file_path in sorted(repo_root.rglob("*.py")):
        relative_path = file_path.relative_to(repo_root)
        if _is_skipped_path(relative_path):
            continue
        top_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        source_text = file_path.read_text(encoding="utf-8")

        # 3. Guard against unauthorized imports outside allowed reader directories.
        if top_folder not in allowed_folders:
            tree = ast.parse(source_text, filename=str(file_path))
            violations.extend(_check_file_imports(relative_path, tree))

        # 4. Guard against environment variable reference leakage into operational packages.
        if top_folder in ashfall_packages and file_path.name != "truth_import_check.py":
            violations.extend(_check_file_env_var(relative_path, top_folder, env_var, source_text.splitlines()))

    return violations


def main() -> int:
    """
    WHAT: CLI entry point for static truth import and environment variable isolation checking.
    Prints detected violations and exits with code 1, or prints confirmation message and exits 0.

    WHY: Integrates directly with CI workflows and local verification recipes.

    Args:
        None.

    Returns:
        Integer exit code: 0 if compliant, 1 if violations exist.

    Example:
        # sys.exit(main())
    """
    # 1. Execute static truth audit across repository.
    violations = find_truth_violations()

    # 2. Report findings and return appropriate exit status.
    for violation in violations:
        print(violation)
    if not violations:
        print("OK: hidden truth isolation verified; no unauthorized imports or env references found.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
