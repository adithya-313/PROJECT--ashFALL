"""
WHAT: Enforces structural comment standards, docstring conventions, and line limits on new world code.
WHY: Ratio checks rewarded filler comments; docstrings carry the explanation while return-restate comments are banned.
TERM: Abstract Syntax Tree: a tree structure showing how code is organized,
      e.g. breaking a function into its name, inputs, and statements.
TERM: Docstring: text placed right after a function header explaining what it does,
      e.g. triple-quoted notes at the top of a file.
TERM: Comment Line: a line starting with a hash symbol explaining intent,
      e.g. # Child streams 0 and 1 are the plan and run streams.
TERM: Return Restatement: an uninformative comment that merely echoes the return statement below it,
      e.g. # Return the result sitting directly above return result.
"""

import ast
import os
from typing import Any

# Master registry of newly authored codebase modules and test suites.
NEW_CODE_FILES: list[str] = [
    # Core simulation engine and calendar planning runtime modules.
    "world/world_plan.py",
    "world/calendar_runner.py",
    # Device lifecycle simulation runtime module.
    "world/device_runtime.py",
    # Episode payment planning module.
    "world/episode_payments.py",
    # Test suites for world plan, calendar runner, and device event wiring.
    "tests/unit/world/test_world_plan.py",
    "tests/unit/world/test_world_calendar_runner.py",
    "tests/unit/world/test_world_device_wiring.py",
    # Episode payment test suite.
    "tests/unit/world/test_world_episode_payments.py",
    # Audit test verifying comment density and docstring standards.
    "tests/unit/world/test_new_code_comment_standard.py",
    # Golden run hashes module and verification recipe test suite.
    "world/run_hashes.py",
    "tests/unit/world/test_run_hashes.py",
    # Verification script and verification unit test suite.
    "scripts/verify_world.py",
    "tests/unit/test_verify_world.py",
    # Level 11A world files conforming to function length and docstring standards.
    "world/devices.py",
    "world/people.py",
    "world/personas.py",
    "world/shops.py",
    "world/money.py",
    "world/config_reader.py",
    "world/placeholders.py",
    "world/regions.py",
    "world/amounts.py",
    "world/banks.py",
    "world/ledger.py",
    "world/payments.py",
    "world/checks.py",
    "world/build_world.py",
    "world/routine.py",
    # Level 11B time and failure files conforming to function length and docstring standards.
    "world/background_config.py",
    "world/clock.py",
    "world/festivals.py",
    "world/rhythms.py",
    "world/failures.py",
    "world/incidents.py",
    "world/failure_decision.py",
    # Level 11B device and episode files conforming to function length and docstring standards.
    "world/device_events.py",
    "world/device_schedule.py",
    "world/episodes.py",
    # Automated test running Level 11B doctests.
    "tests/unit/world/test_world_doctests.py",
    # Level 11C-1 fast mode module and test suite.
    "world/fast_mode.py",
    "tests/unit/world/test_world_fast_mode.py",
    # Level 11C-2a sizing estimates module and test suite.
    "world/estimates.py",
    "tests/unit/world/test_world_estimates.py",
    # Level 11C-2b machine limits, world sizing and sizing test suite.
    "world/machine_limits.py",
    "world/world_sizing.py",
    "tests/unit/world/test_world_sizing.py",
]




NEW_FUNCTIONS_IN_OLD_FILES: list[tuple[str, str]] = [
    ("world/amounts.py", "lookalike_small_amount_paise"),
]


def _count_code_lines(node: Any, source_lines: list[str]) -> int:
    """
    WHAT: Counts executable code lines within an AST function span, excluding docstrings, blank lines,
    and comment lines.
    It identifies the line span of any top-level docstring expression to filter out documentation text.
    It then iterates through all lines between the function header and end line, ignoring comments and whitespace.

    WHY: Standard length checks naively counting physical lines unfairly penalize well-documented functions.
    The 30-line limit is intended to prevent sprawling procedural logic rather than discouraging thorough docstrings.
    Measuring code lines directly ensures developers can write comprehensive explanations without violating line limits.

    Args:
        node: AST FunctionDef or AsyncFunctionDef node whose code lines are being counted.
        source_lines: List of text lines comprising the complete source file.

    Returns:
        Integer count of non-blank, non-comment code lines belonging to the function body and header.

    Example:
        # code_text = ('def sample(x):\n    \"\"\"A docstring.\"\"\"\n'
        #              '    # A comment line\n    \n    y = x + 10\n    return y\n')
        # node = ast.parse(code_text).body[0]
        # lines = code_text.splitlines(keepends=True)
        # count = _count_code_lines(node, lines)
        # assert count == 3
    """
    # Identify the line span of the leading docstring expression if one exists.
    doc_range: tuple[int, int] | None = None
    if node.body:
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            doc_range = (first.lineno, first.end_lineno)

    code_line_count = 0
    # Inspect every line within the function definition's starting and ending line bounds.
    for lineno in range(node.lineno, node.end_lineno + 1):
        # Skip lines falling within the enclosing docstring range.
        if doc_range and doc_range[0] <= lineno <= doc_range[1]:
            continue
        line_content = source_lines[lineno - 1].strip()
        # Skip empty lines, whitespace-only lines, and comment-only lines.
        if not line_content or line_content.startswith("#"):
            continue
        code_line_count += 1

    return code_line_count


def _validate_function_docstring(node: Any, filepath: str) -> list[str]:
    """
    WHAT: Validates character length and required section tags (WHAT, WHY, Example, Args, Returns).
    WHY: Enforces rigorous explanatory standards across all new implementation and test routines.
    Args: node: AST FunctionDef node; filepath: Source path string for diagnostic messages.
    Returns: List of error strings describing any standard non-compliances found.
    Example:
        # errs = _validate_function_docstring(fn_node, 'world/world_plan.py'); assert errs == []
    """
    doc = ast.get_docstring(node)
    # Reject functions that lack docstrings entirely.
    if doc is None:
        return [f"{filepath}:{node.name}: missing docstring"]
    issues: list[str] = []
    # Assert minimum descriptive length so explanations are sufficiently detailed.
    if len(doc) < 300:
        issues.append(f"{filepath}:{node.name}: docstring length {len(doc)} < 300")
    # Verify presence of mandatory explanatory tags.
    for tag in ("WHAT:", "WHY:", "Example:"):
        if tag not in doc:
            issues.append(f"{filepath}:{node.name}: missing '{tag}'")
    # Verify that functions taking parameters document them with an Args section.
    params = [a.arg for a in node.args.args if a.arg != "self"]
    if params and "Args:" not in doc:
        issues.append(f"{filepath}:{node.name}: function takes args {params} but lacks 'Args:'")
    # Verify that functions returning values document them with a Returns section.
    returns_val = any(isinstance(c, ast.Return) and c.value is not None for c in ast.walk(node))
    if returns_val and "Returns:" not in doc:
        issues.append(f"{filepath}:{node.name}: returns value but lacks 'Returns:'")
    return issues


def test_new_code_files_registry() -> None:
    """
    WHAT: Asserts that NEW_CODE_FILES contains at least the 5 required files and all exist on disk.
    WHY: Protects against file omission or registry shrinkage, guaranteeing complete audit coverage.
    Example:
        # Asserts NEW_CODE_FILES has length >= 5 and every path points to an existing file
        # assert len(NEW_CODE_FILES) >= 5 and all(os.path.exists(p) for p in NEW_CODE_FILES)
    """
    expected_minimum = [
        "world/world_plan.py",
        "world/calendar_runner.py",
        "tests/unit/world/test_world_plan.py",
        "tests/unit/world/test_world_calendar_runner.py",
        "tests/unit/world/test_new_code_comment_standard.py",
    ]
    # Check that our registry of audited files has not shrunk below the 5 mandatory files.
    assert len(NEW_CODE_FILES) >= 5, f"Registry shrank below 5 files: {len(NEW_CODE_FILES)}"
    # Verify that every required file is present in the list and exists on the filesystem.
    for req in expected_minimum:
        assert req in NEW_CODE_FILES, f"Missing required file in NEW_CODE_FILES: {req}"
        assert os.path.exists(req), f"Registered file does not exist on disk: {req}"


def test_function_lengths_and_docstrings() -> None:
    """
    WHAT: Traverses all functions in NEW_CODE_FILES to assert code lines <= 30 and full docstring schema.
    WHY: Enforces modular 30-line code boundaries and thorough WHAT/WHY/Example documentation.
    Example:
        # For each function in new files, code lines <= 30 and docstring is complete
        # assert len(errors) == 0
    """
    errors: list[str] = []
    # Parse each registered source file to check function lengths and docstrings.
    for filepath in NEW_CODE_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
        tree = ast.parse("".join(source_lines), filename=filepath)
        # Inspect every function definition in the file.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                code_lines = _count_code_lines(node, source_lines)
                # Enforce the 30-line limit on code lines per function.
                if code_lines > 30:
                    errors.append(f"{filepath}:{node.name}: function code lines {code_lines} > 30")
                # Check that the docstring contains WHAT, WHY, Example, and parameter docs.
                errors.extend(_validate_function_docstring(node, filepath))
    assert not errors, "Function standards violations:\n" + "\n".join(errors)


def test_no_comment_restates_a_return() -> None:
    """
    WHAT: Traverses all files in NEW_CODE_FILES to forbid comments directly above returns that restate the return.
    WHY: Comment-ratio checks incentivized vacuous filler comments; docstrings carry the true architectural explanation.
    Example:
        # Fails if '# Return x' is directly followed by 'return x'
        # violations = []; assert not violations
    """
    violations: list[str] = []
    # Inspect every registered new code file for redundant return restatements.
    for filepath in NEW_CODE_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Scan adjacent line pairs to catch comments preceding return statements.
        for idx in range(len(lines) - 1):
            curr_line = lines[idx].strip()
            next_line = lines[idx + 1].strip()
            # Check if current line is a comment starting with Return after the hash.
            if curr_line.startswith("#"):
                comment_text = curr_line[1:].strip()
                if comment_text.startswith("Return") and next_line.startswith("return"):
                    violations.append(f"{filepath}:{idx + 1}: '{curr_line}' sits directly above '{next_line}'")
    print(f"MEASURED: checked {len(NEW_CODE_FILES)} files for comments restating returns")
    assert not violations, "Found comments restating a return:\n" + "\n".join(violations)


def test_module_docstrings_define_technical_terms() -> None:
    """
    WHAT: Checks that each file in NEW_CODE_FILES defines technical terms in its module docstring.
    WHY: Ensures beginners and domain auditors have clear definitions for specialized terminology.
    Example:
        # Each module docstring contains 'TERM:' explanations
        # assert 'TERM:' in ast.get_docstring(tree)
    """
    missing: list[str] = []
    # Verify that every file includes technical term definitions for beginners.
    for filepath in NEW_CODE_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        doc = ast.get_docstring(tree)
        # Check for the presence of TERM: definitions in the module header docstring.
        if not doc or "TERM:" not in doc:
            missing.append(filepath)
    assert not missing, f"Files missing technical term definitions (TERM:) in module docstring: {missing}"


def test_new_functions_in_old_files_standard() -> None:
    """
    WHAT: Asserts new functions in older files satisfy code line limits (<= 30) and docstring standards.
    WHY: Prevents code bloat and unmaintained documentation when adding helper routines to older modules.
    Example:
        # Every function in NEW_FUNCTIONS_IN_OLD_FILES is <= 30 code lines with full WHAT/WHY/Example tags
        # assert len(errors) == 0
    """
    # Verify the registry contains the required entry and cannot shrink.
    assert ("world/amounts.py", "lookalike_small_amount_paise") in NEW_FUNCTIONS_IN_OLD_FILES
    assert len(NEW_FUNCTIONS_IN_OLD_FILES) >= 1
    errors: list[str] = []
    # Audit each registered function across older codebase files.
    for filepath, fn_name in NEW_FUNCTIONS_IN_OLD_FILES:
        # Parse module AST to find target function definitions.
        with open(filepath, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
        tree = ast.parse("".join(source_lines), filename=filepath)
        nodes = [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name
        ]
        # Check function exists and enforce 30-line limit and docstring rules.
        if not nodes:
            errors.append(f"{filepath}: function '{fn_name}' not found")
        for node in nodes:
            code_lines = _count_code_lines(node, source_lines)
            if code_lines > 30:
                errors.append(f"{filepath}:{fn_name}: function code lines {code_lines} > 30")
            errors.extend(_validate_function_docstring(node, filepath))
    assert not errors, "Standards violations in older file functions:\n" + "\n".join(errors)


