"""
WHAT: Enforces structural comment standards, docstring conventions, and line limits on new world code.
WHY: Ensures code readability, prevents regression in architectural documentation, and validates rule R7 standards.
TERM: Abstract Syntax Tree: In-memory hierarchical representation of Python source syntax parsed for inspection.
TERM: Lexical Tokenizer: Stream scanner breaking Python text into categorized tokens such as comments and names.
TERM: Docstring: String literal appearing as the first statement of a module, class, or function definition.
TERM: Comment Ratio: Metric dividing the count of comment lines by the count of executable code lines.
TERM: Code Line: Source line containing syntax tokens excluding blank lines and multi-line docstring bodies.
"""

import ast
import os
import tokenize
from typing import Any

# =============================================================================
# REGISTRY OF NEW LEVEL 11B-3A FILES SUBJECT TO ENFORCED COMMENT STANDARDS
# =============================================================================
NEW_CODE_FILES: list[str] = [
    "world/world_plan.py",
    "world/calendar_runner.py",
    "tests/unit/world/test_world_plan.py",
    "tests/unit/world/test_world_calendar_runner.py",
    "tests/unit/world/test_new_code_comment_standard.py",
]


# =============================================================================
# HELPER FUNCTIONS: AST INSPECTION AND TOKEN-BASED LEXICAL COUNTING
# =============================================================================
# These utility routines traverse the source files of the ashFALL simulation suite.
# They separate documentation literals from executable statements and measure compliance.
def _collect_docstring_lines(tree: ast.AST) -> set[int]:
    """
    WHAT: Traverses an AST tree to collect all physical line numbers occupied by docstring literals.
    WHY: Distinguishes documentation strings from executable code lines during comment ratio calculations.
    Args: tree: Root Abstract Syntax Tree node of the parsed Python source file.
    Returns: Set of integer 1-indexed line numbers containing module, class, or function docstrings.
    Example:
        # tree = ast.parse('def f():\\n    \"\"\"doc\"\"\"\\n    pass')
        # doc_lines = _collect_docstring_lines(tree); assert 2 in doc_lines
    """
    doc_lines: set[int] = set()
    # Walk all abstract syntax tree nodes to discover docstring expressions.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check if leading statement in body is a string expression literal.
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    for line_num in range(node.body[0].lineno, node.body[0].end_lineno + 1):
                        doc_lines.add(line_num)
    return doc_lines


def _calculate_file_comment_and_code_lines(filepath: str) -> tuple[int, int]:
    """
    WHAT: Computes the count of comment lines and non-docstring code lines for a given source file.
    WHY: Provides the numerator and denominator for validating the >= 0.25 comment-to-code ratio.
    Args: filepath: Relative file path to the Python source file being analyzed.
    Returns: Tuple of (comment_lines_count, code_lines_count).
    Example:
        # comments, code = _calculate_file_comment_and_code_lines('world/world_plan.py')
        # assert comments >= 0 and code > 0
    """
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=filepath)
    doc_lines = _collect_docstring_lines(tree)
    # Tokenize file bytes to locate comments and executable tokens.
    comment_lines: set[int] = set()
    code_lines: set[int] = set()
    with open(filepath, "rb") as f:
        for tok in tokenize.tokenize(f.readline):
            if tok.type == tokenize.COMMENT:
                comment_lines.add(tok.start[0])
            elif tok.type not in (
                tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING
            ):
                for line_idx in range(tok.start[0], tok.end[0] + 1):
                    if line_idx not in doc_lines:
                        code_lines.add(line_idx)
    return len(comment_lines), len(code_lines)


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
    if doc is None:
        return [f"{filepath}:{node.name}: missing docstring"]
    issues: list[str] = []
    # Assert minimum descriptive length.
    if len(doc) < 300:
        issues.append(f"{filepath}:{node.name}: docstring length {len(doc)} < 300")
    # Verify core section tags.
    for tag in ("WHAT:", "WHY:", "Example:"):
        if tag not in doc:
            issues.append(f"{filepath}:{node.name}: missing '{tag}'")
    # Check parameters and return value tags.
    params = [a.arg for a in node.args.args if a.arg != "self"]
    if params and "Args:" not in doc:
        issues.append(f"{filepath}:{node.name}: function takes args {params} but lacks 'Args:'")
    returns_val = any(isinstance(c, ast.Return) and c.value is not None for c in ast.walk(node))
    if returns_val and "Returns:" not in doc:
        issues.append(f"{filepath}:{node.name}: returns value but lacks 'Returns:'")
    return issues


# =============================================================================
# TEST SUITE: ENFORCEMENT OF CODE INTEGRITY, DOCSTRINGS, AND COMMENT DENSITY
# =============================================================================
# The following tests verify that all newly introduced modules maintain rigorous
# architectural commentary, adhere to 30-line function boundaries, and document technical terms.
# -----------------------------------------------------------------------------
# Test 1: Verification of registered file inventory.
# -----------------------------------------------------------------------------
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
    # Verify the registry contains at least the 5 mandatory files.
    assert len(NEW_CODE_FILES) >= 5, f"Registry shrank below 5 files: {len(NEW_CODE_FILES)}"
    for req in expected_minimum:
        assert req in NEW_CODE_FILES, f"Missing required file in NEW_CODE_FILES: {req}"
        assert os.path.exists(req), f"Registered file does not exist on disk: {req}"


# -----------------------------------------------------------------------------
# Test 2: Validation of function length boundaries and docstring schemas.
# -----------------------------------------------------------------------------
def test_function_lengths_and_docstrings() -> None:
    """
    WHAT: Traverses all functions in NEW_CODE_FILES to assert length <= 30 and full docstring schema.
    WHY: Enforces modular 30-line boundaries and thorough WHAT/WHY/Example documentation.
    Example:
        # For each function in new files, end_lineno - lineno + 1 <= 30 and docstring is complete
        # assert len(errors) == 0
    """
    errors: list[str] = []
    # Inspect every registered new source file.
    for filepath in NEW_CODE_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                if length > 30:
                    errors.append(f"{filepath}:{node.name}: function length {length} > 30")
                errors.extend(_validate_function_docstring(node, filepath))
    assert not errors, "Function standards violations:\n" + "\n".join(errors)


# -----------------------------------------------------------------------------
# Test 3: Enforcement of the 25% comment-to-code density threshold.
# -----------------------------------------------------------------------------
def test_comment_to_code_ratio_threshold() -> None:
    """
    WHAT: Computes comment lines / code lines for every file in NEW_CODE_FILES to assert ratio >= 0.25.
    WHY: Guarantees every new file maintains rich inline step-by-step commentary (rule R7 standards).
    Example:
        # For each file, comment_lines / code_lines >= 0.25
        # assert ratio >= 0.25, f"{filepath}: ratio {ratio} < 0.25"
    """
    failures: list[str] = []
    # Calculate comment ratio for each registered file.
    for filepath in NEW_CODE_FILES:
        n_comments, n_code = _calculate_file_comment_and_code_lines(filepath)
        ratio = n_comments / n_code if n_code > 0 else 0.0
        # Assert measured ratio meets or exceeds the 0.25 threshold.
        if ratio < 0.25:
            failures.append(f"{filepath}: ratio {ratio:.3f} ({n_comments}/{n_code}) < 0.25")
    assert not failures, "Comment ratio threshold violations:\n" + "\n".join(failures)


# -----------------------------------------------------------------------------
# Test 4: Verification of technical terminology definitions.
# -----------------------------------------------------------------------------
def test_module_docstrings_define_technical_terms() -> None:
    """
    WHAT: Checks that each file in NEW_CODE_FILES defines technical terms in its module docstring.
    WHY: Ensures beginners and domain auditors have clear definitions for specialized terminology.
    Example:
        # Each module docstring contains 'TERM:' explanations
        # assert 'TERM:' in ast.get_docstring(tree)
    """
    missing: list[str] = []
    # Verify module docstring on each registered file.
    for filepath in NEW_CODE_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        doc = ast.get_docstring(tree)
        if not doc or "TERM:" not in doc:
            missing.append(filepath)
    assert not missing, f"Files missing technical term definitions (TERM:) in module docstring: {missing}"
