"""
WHAT: Unit tests for static truth isolation guard (core/truth_import_check.py): asserts that the clean repository
passes all isolation rules, violating imports in temporary api and core directories are detected across all import
forms, operational package references to the truth environment variable are caught, and permitted reader folders pass.

WHY: Banking operational code must never depend on or access synthetic simulation truth. Unit tests verify that the
static AST checker reliably prevents forbidden imports and environment leaks before deployment to CI.

TERM: static AST guard: analysis of Python syntax trees without executing runtime code.
TERM: operational package: production banking services (api, serving, models, core, features, online_learning, agent).
TERM: allowed reader folders: explicitly permitted directories (evaluation, showcase, tests) allowed to inspect truth.
TERM: truth isolation: architectural boundary keeping ground truth inaccessible to operational banking code.
"""

from pathlib import Path

from core.truth_import_check import find_truth_violations, main


def test_clean_repository_passes_truth_import_check() -> None:
    """
    WHAT: Asserts that running find_truth_violations against the actual repository returns no violations.

    WHY: Verifies that the production codebase currently satisfies all architectural truth isolation boundaries.

    Args:
        None.

    Returns:
        None. Fails if any violation is detected in the repository.

    Example:
        # assert find_truth_violations() == []
    """
    # 1. Execute repository-wide truth isolation check.
    violations = find_truth_violations()

    # 2. Assert that no unauthorized imports or environment references exist.
    assert violations == []


def test_violating_import_in_temp_api_folder_is_caught(tmp_path: Path) -> None:
    """
    WHAT: Creates a temporary 'api' package containing both 'import evaluation.truth' and 'from evaluation.truth
    import load_truth' and asserts that find_truth_violations catches both occurrences.

    WHY: Operational package 'api' must never import evaluation truth modules via any import syntax.

    Args:
        tmp_path: Pytest temporary directory fixture for isolated file tree.

    Returns:
        None. Fails if violating imports in api are not detected.

    Example:
        # Fake file in api/ importing evaluation.truth is detected.
    """
    # 1. Create temporary api directory and violating source file.
    api_directory = tmp_path / "api"
    api_directory.mkdir(parents=True, exist_ok=True)
    violating_content = (
        "import evaluation.truth\n"
        "from evaluation.truth import load_truth\n"
        "def handler():\n"
        "    return load_truth\n"
    )
    (api_directory / "endpoints.py").write_text(violating_content, encoding="utf-8")

    # 2. Run static audit against temporary tree and assert both violations are caught.
    violations = find_truth_violations(tmp_path)
    assert len(violations) == 2
    assert "api/endpoints.py:1: forbidden import 'evaluation.truth'" in violations[0]
    assert "api/endpoints.py:2: forbidden import from 'evaluation.truth'" in violations[1]


def test_violating_import_in_temp_core_folder_is_caught(tmp_path: Path) -> None:
    """
    WHAT: Creates a temporary 'core' package file importing 'evaluation' and asserts that find_truth_violations
    catches the violation.

    WHY: Core shared operational library must never import evaluation truth modules.

    Args:
        tmp_path: Pytest temporary directory fixture for isolated file tree.

    Returns:
        None. Fails if violating import in core is not detected.

    Example:
        # Fake file in core/ importing evaluation is detected.
    """
    # 1. Create temporary core directory and violating source file.
    core_directory = tmp_path / "core"
    core_directory.mkdir(parents=True, exist_ok=True)
    violating_content = (
        "from evaluation import truth\n"
        "def helper():\n"
        "    return truth\n"
    )
    (core_directory / "service.py").write_text(violating_content, encoding="utf-8")

    # 2. Run static audit against temporary tree and assert violation is detected.
    violations = find_truth_violations(tmp_path)
    assert len(violations) == 1
    assert "core/service.py:1: forbidden import from 'evaluation'" in violations[0]


def test_violating_env_var_in_temp_core_folder_is_caught(tmp_path: Path) -> None:
    """
    WHAT: Creates a temporary 'core' package file referencing 'ASHFALL_TRUTH_DIR' and asserts that
    find_truth_violations detects the environment variable leak.

    WHY: Operational packages must never configure or inspect the hidden truth root directory.

    Args:
        tmp_path: Pytest temporary directory fixture for isolated file tree.

    Returns:
        None. Fails if truth directory environment variable reference is not flagged.

    Example:
        # Fake file in core/ using ASHFALL_TRUTH_DIR is detected.
    """
    # 1. Create temporary core directory and source file containing truth env reference.
    core_directory = tmp_path / "core"
    core_directory.mkdir(parents=True, exist_ok=True)
    violating_content = (
        "import os\n"
        "def read_settings():\n"
        "    return os.environ.get('ASHFALL_TRUTH_DIR', 'runs/truth')\n"
    )
    (core_directory / "settings.py").write_text(violating_content, encoding="utf-8")

    # 2. Run static audit against temporary tree and assert env var leak is detected.
    violations = find_truth_violations(tmp_path)
    assert len(violations) == 1
    assert "core/settings.py:3: operational package 'core' references forbidden truth env var" in violations[0]


def test_allowed_folders_can_import_evaluation(tmp_path: Path) -> None:
    """
    WHAT: Creates temporary files under evaluation, showcase, and tests importing evaluation.truth
    and asserts that find_truth_violations reports zero violations.

    WHY: Allowed reader folders are legitimately permitted to access simulation ground truth.

    Args:
        tmp_path: Pytest temporary directory fixture for isolated file tree.

    Returns:
        None. Fails if allowed folders trigger false-positive violations.

    Example:
        # Clean file in evaluation/ importing evaluation.truth passes.
    """
    # 1. Create valid permitted files in evaluation, showcase, and tests.
    for folder_name in ["evaluation", "showcase", "tests"]:
        folder_path = tmp_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        valid_content = "from evaluation.truth import load_truth\n"
        (folder_path / "consumer.py").write_text(valid_content, encoding="utf-8")

    # 2. Run static audit and assert no violations are reported.
    violations = find_truth_violations(tmp_path)
    assert violations == []


def test_main_cli_returns_clean_exit_code() -> None:
    """
    WHAT: Asserts that main() returns exit code 0 when executed against the clean repository codebase.

    WHY: The command-line entry point 'python -m core.truth_import_check' is invoked directly within
    continuous integration workflows and developer verification recipes; it must exit with code 0 when all
    static truth isolation architectural boundaries are satisfied and no violations are found.

    Args:
        None.

    Returns:
        None. Fails if main() returns non-zero.

    Example:
        # assert main() == 0
    """
    # 1. Invoke main CLI function and assert clean exit code 0.
    exit_code = main()
    assert exit_code == 0


def test_dynamic_import_outside_allowed_folders_is_flagged(tmp_path: Path) -> None:
    """
    WHAT: Creates a temporary script outside allowed folders calling importlib.import_module('evaluation.truth')
    and asserts that find_truth_violations flags the dynamic import attempt.

    WHY: Dynamic imports using importlib or __import__ must not be used to circumvent static truth isolation checks.

    Args:
        tmp_path: Pytest temporary directory fixture for creating test files.

    Returns:
        None. Fails if dynamic evaluation import outside allowed folders is not detected.

    Example:
        # A file under scripts/ dynamically importing evaluation.truth is flagged.
    """
    # 1. Create temporary scripts directory and source file with dynamic import.
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    content = (
        "import importlib\n"
        "def run():\n"
        "    return importlib.import_module('evaluation.truth')\n"
    )
    (scripts_dir / "worker.py").write_text(content, encoding="utf-8")

    # 2. Assert that dynamic import outside allowed folders is flagged.
    violations = find_truth_violations(tmp_path)
    assert len(violations) == 1
    assert "scripts/worker.py:3: forbidden dynamic import 'evaluation.truth'" in violations[0]


def test_dynamic_import_inside_evaluation_is_permitted(tmp_path: Path) -> None:
    """
    WHAT: Creates a temporary file inside the evaluation package directory calling importlib.import_module
    and asserts that find_truth_violations reports no violations.

    WHY: Modules inside allowed reader folders are legitimately permitted to inspect evaluation modules.

    Args:
        tmp_path: Pytest temporary directory fixture for creating test files.

    Returns:
        None. Fails if dynamic import inside evaluation folder triggers false positive.

    Example:
        # A file under evaluation/ dynamically importing evaluation.truth passes cleanly.
    """
    # 1. Create temporary evaluation directory and file with dynamic import.
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    content = (
        "import importlib\n"
        "def run():\n"
        "    return importlib.import_module('evaluation.truth')\n"
    )
    (eval_dir / "runner.py").write_text(content, encoding="utf-8")

    # 2. Assert that dynamic import within evaluation directory is allowed.
    violations = find_truth_violations(tmp_path)
    assert violations == []

