"""
WHAT: Enforces structural comment standards, docstring conventions, and line limits on new world code.
WHY: Ensures code readability and prevents regressions in architectural documentation.
TERM: Abstract Syntax Tree: a tree structure showing how code is organized,
      e.g. breaking a function into its name, inputs, and statements.
TERM: Tokenizer: a tool reading source code word by word,
      e.g. separating comments from code words.
TERM: Docstring: text placed right after a function header explaining what it does,
      e.g. triple-quoted notes at the top of a file.
TERM: Comment Ratio: the fraction of lines that explain code,
      e.g. having 1 comment line for every 4 code lines gives a ratio of 0.25.
TERM: Code Line: a line of code doing real work,
      e.g. an assignment like x = 1, ignoring empty lines.
"""

import ast
import os
import tokenize
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
]



NEW_FUNCTIONS_IN_OLD_FILES: list[tuple[str, str]] = [
    ("world/amounts.py", "lookalike_small_amount_paise"),
]


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
    # Walk all syntax tree nodes to discover docstring expressions.
    for node in ast.walk(tree):
        # Check modules, classes, and functions for leading string statements.
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            # Look at the first statement in the body of the definition.
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                # If expression value is a string, record every physical line it covers.
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    for line_num in range(node.body[0].lineno, node.body[0].end_lineno + 1):
                        doc_lines.add(line_num)
    return doc_lines


def _collect_function_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """
    WHAT: Scans an Abstract Syntax Tree to identify starting and ending line intervals of functions.
    WHY: Enables measuring code lines and comments that sit specifically within function bodies.
    Args: tree: Root Abstract Syntax Tree node of the parsed Python source file.
    Returns: List of tuples where each element contains (start_line, end_line) of a function definition.
    Example:
        # tree = ast.parse('def f():\\n    pass')
        # ranges = _collect_function_ranges(tree); assert len(ranges) == 1
    """
    ranges: list[tuple[int, int]] = []
    # Find all function definition nodes in the syntax tree.
    for node in ast.walk(tree):
        # Check if node represents a standard or asynchronous function definition.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Record span from the def keyword line to the last line of the function.
            ranges.append((node.lineno, node.end_lineno))
    return ranges


def _tally_tokens(
    filepath: str, doc_lines: set[int], fn_ranges: list[tuple[int, int]]
) -> tuple[int, int, int, int]:
    """
    WHAT: Scans tokens of a file and tallies comment lines and code lines globally and inside functions.
    WHY: Breaks token scanning into a focused helper under 30 lines to preserve modular design.
    Args: filepath: Path to file; doc_lines: Set of docstring line numbers; fn_ranges: List of function spans.
    Returns: Tuple of (all_comments_count, all_code_count, fn_comments_count, fn_code_count).
    Example:
        # c, code, fn_c, fn_code = _tally_tokens('world/world_plan.py', set(), [])
        # assert code >= 0
    """
    all_c, all_code, fn_c, fn_code = set(), set(), set(), set()
    ignored = (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING)
    # Read the file with the tokenizer to inspect comments and syntax tokens.
    with open(filepath, "rb") as f:
        for tok in tokenize.tokenize(f.readline):
            line = tok.start[0]
            in_fn = any(s <= line <= e for s, e in fn_ranges)
            if tok.type == tokenize.COMMENT:
                all_c.add(line)
                if in_fn:
                    fn_c.add(line)
            elif tok.type not in ignored:
                for idx in range(tok.start[0], tok.end[0] + 1):
                    if idx not in doc_lines:
                        all_code.add(idx)
                        if in_fn:
                            fn_code.add(idx)
    return len(all_c), len(all_code), len(fn_c), len(fn_code)


def _calculate_file_ratios(filepath: str) -> tuple[float, float]:
    """
    WHAT: Computes the whole-file comment ratio and the inside-functions comment ratio for a file.
    WHY: Verifies both global documentation density and localized inline commentary inside functions.
    Args: filepath: Relative file path to the Python source file being analyzed.
    Returns: Tuple of (whole_file_ratio, inside_functions_ratio).
    Example:
        # whole, inside = _calculate_file_ratios('world/world_plan.py')
        # assert whole >= 0.25 and inside >= 0.25
    """
    # Parse source tree to find docstring line spans and function definitions.
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)
    # Collect line numbers for docstrings and line intervals for function bodies.
    doc_lines = _collect_docstring_lines(tree)
    fn_ranges = _collect_function_ranges(tree)
    # Tally comments and code lines across the file and inside functions.
    n_all_c, n_all_code, n_fn_c, n_fn_code = _tally_tokens(filepath, doc_lines, fn_ranges)
    # Compute the ratio for the entire file.
    whole_ratio = n_all_c / n_all_code if n_all_code else 0.0
    # Compute the ratio strictly within function bodies.
    inside_ratio = n_fn_c / n_fn_code if n_fn_code else 0.0
    return whole_ratio, inside_ratio


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
    WHAT: Traverses all functions in NEW_CODE_FILES to assert length <= 30 and full docstring schema.
    WHY: Enforces modular 30-line boundaries and thorough WHAT/WHY/Example documentation.
    Example:
        # For each function in new files, end_lineno - lineno + 1 <= 30 and docstring is complete
        # assert len(errors) == 0
    """
    errors: list[str] = []
    # Parse each registered source file to check function lengths and docstrings.
    for filepath in NEW_CODE_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        # Inspect every function definition in the file.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                # Enforce the 30-line limit per function.
                if length > 30:
                    errors.append(f"{filepath}:{node.name}: function length {length} > 30")
                # Check that the docstring contains WHAT, WHY, Example, and parameter docs.
                errors.extend(_validate_function_docstring(node, filepath))
    assert not errors, "Function standards violations:\n" + "\n".join(errors)


def test_comment_to_code_ratio_threshold() -> None:
    """
    WHAT: Computes comment lines / code lines for every file in NEW_CODE_FILES to assert ratio >= 0.25.
    WHY: Guarantees every new file maintains rich inline step-by-step commentary.
    Example:
        # For each file, comment_lines / code_lines >= 0.25
        # assert ratio >= 0.25, f"{filepath}: ratio {ratio} < 0.25"
    """
    failures: list[str] = []
    # Check the whole-file comment-to-code ratio for each registered file.
    for filepath in NEW_CODE_FILES:
        whole_ratio, _ = _calculate_file_ratios(filepath)
        # Check that the file meets or exceeds the required 25% comment density.
        if whole_ratio < 0.25:
            failures.append(f"{filepath}: whole-file ratio {whole_ratio:.3f} < 0.25")
    assert not failures, "Comment ratio threshold violations:\n" + "\n".join(failures)


def test_inside_function_comment_ratio_threshold() -> None:
    """
    WHAT: Computes comment lines / code lines inside function bodies to assert ratio >= 0.25.
    WHY: Ensures functions are self-explanatory so future readers understand why each step happens.
    Example:
        # For each file, inside_function_comments / inside_function_code >= 0.25
        # assert fn_ratio >= 0.25, f"{filepath}: function ratio {fn_ratio} < 0.25"
    """
    failures: list[str] = []
    # Check the inside-function comment-to-code ratio for each registered file.
    for filepath in NEW_CODE_FILES:
        _, inside_ratio = _calculate_file_ratios(filepath)
        # Check that inline comments inside function bodies meet or exceed 25%.
        if inside_ratio < 0.25:
            failures.append(f"{filepath}: inside-function ratio {inside_ratio:.3f} < 0.25")
    assert not failures, "Inside-function comment ratio threshold violations:\n" + "\n".join(failures)


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
    WHAT: Asserts new functions in older files satisfy line limits (<= 30) and docstring standards.
    WHY: Prevents code bloat and unmaintained documentation when adding helper routines to older modules.
    Example:
        # Every function in NEW_FUNCTIONS_IN_OLD_FILES is <= 30 lines with full WHAT/WHY/Example tags
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
            tree = ast.parse(f.read(), filename=filepath)
        nodes = [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name
        ]
        # Check function exists and enforce 30-line limit and docstring rules.
        if not nodes:
            errors.append(f"{filepath}: function '{fn_name}' not found")
        for node in nodes:
            if node.end_lineno - node.lineno + 1 > 30:
                errors.append(f"{filepath}:{fn_name}: function length > 30")
            errors.extend(_validate_function_docstring(node, filepath))
    assert not errors, "Standards violations in older file functions:\n" + "\n".join(errors)

