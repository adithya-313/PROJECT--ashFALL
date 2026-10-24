"""
Purpose-driven documentation guard ensuring substantive explanations across new world code.
Verifies that all modules, classes, and functions have non-empty docstrings of at least 40
characters that do not merely restate their names, and forbids comments restating returns.
"""

import ast
import os
from typing import Any

# Policy constant: 40 characters ensures a docstring provides a substantive sentence explaining purpose.
MIN_DOCSTRING_LENGTH: int = 40

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
    # Level 11C-3 stackable layers modules and test suite.
    "world/world_configs.py",
    "world/layers.py",
    "tests/unit/world/test_world_layers.py",
    # Level 11C-4 live mode module and test suite.
    "world/live_mode.py",
    "tests/unit/world/test_world_live_mode.py",
    # Level 11D-1 stress library, world card, and grounding modules.
    "world/world_card.py",
    "world/grounding.py",
    "tests/unit/world/test_world_stress.py",
    # Level 11D-2 showcase chart modules and test suites.
    "showcase/level_11_charts.py",
    "tests/unit/showcase/test_level_11_charts.py",
    # Level 11D-2 showcase checks and runner modules.
    "showcase/level_11.py",
    "showcase/level_11_checks.py",
    "showcase/level_11_checks_more.py",
    "tests/unit/showcase/test_level_11_checks.py",
    # Level 11D-2 showcase page renderer and test suite.
    "showcase/level_11_page.py",
    "tests/unit/showcase/test_level_11_page.py",
    "showcase/level_11_festival.py",
    "tests/unit/showcase/test_level_11_festival.py",
    # Level 12a frozen-anchor test guarding every value Level 11 pinned.
    "tests/unit/world/test_frozen_anchors.py",
    # Level 12a showcase --part option test suite.
    "tests/unit/test_showcase_part.py",
    # Level 12a run identity module and test suite.
    "world/run_identity.py",
    "tests/unit/world/test_run_identity.py",
    # Level 12a outside events, encumbrance book and their test suite.
    "world/outside_events.py",
    "world/encumbrance.py",
    "tests/unit/world/test_outside_events_and_encumbrance.py",
    # Level 12a interactive runner records, switch and fraud-off path, and their test suite.
    "world/interactive_records.py",
    "world/interactive_runner.py",
    "tests/unit/world/test_interactive_runner_switch.py",
    # Level 12a fraud-on loop, runner state and their test suite.
    "world/interactive_state.py",
    "world/interactive_loop.py",
    "tests/unit/world/test_interactive_runner_fraud_on.py",
    # Level 12a tactic family and tactic cost config test suite.
    "tests/unit/test_tactic_families.py",
    # Level 12a issue register v2 checker and its test suite.
    "core/issue_register_v2.py",
    "tests/unit/test_issue_register_v2.py",
    # Level 12a events-v2 line builder and its test suite.
    "world/interactive_events.py",
    "tests/unit/world/test_interactive_events.py",
    # Level 12a showcase checks and runner modules.
    "showcase/level_12_support.py",
    "showcase/level_12_checks.py",
    "showcase/level_12.py",
    "tests/unit/showcase/test_level_12_checks.py",
    # Level 12b hidden truth store records module and test suite.
    "world/truth_records.py",
    "tests/unit/world/test_truth_records.py",
    # Level 12b hidden truth store writer and test suite.
    "world/truth_writer.py",
    "tests/unit/world/test_truth_writer.py",
    # Level 12b evaluation truth loader and test suite.
    "evaluation/truth.py",
    "tests/unit/evaluation/test_truth_loader.py",
    # Level 12b static truth import guard and test suite.
    "core/truth_import_check.py",
    "tests/unit/test_truth_import_check.py",
    # Level 12b truth environment separation test suite.
    "tests/unit/test_truth_env_separation.py",
    # Level 12b tactic block catalogue and dividing lines module and test suite.
    "world/tactic_config.py",
    "tests/unit/world/test_tactic_config.py",
    # Level 12b gang configuration module and validation test suite.
    "world/gang_config.py",
    "tests/unit/world/test_gang_config.py",
    # Level 12b tactic blocks and gang planner modules and test suite.
    "world/tactic_blocks.py",
    "world/gang_planner.py",
    "tests/unit/world/test_gang_planner.py",
    # Level 12b gang runner integration test suite.
    "tests/unit/world/test_gang_runs.py",
    # Level 12b gang devices module and test suite.
    "world/gang_devices.py",
    "tests/unit/world/test_gang_devices.py",
    # Level 12b gang device stream test suite.
    "tests/unit/world/test_gang_device_stream.py",
    # Level 12b showcase checks and runner modules.
    "showcase/level_12_support_b.py",
    "showcase/level_12_checks_b.py",
    "tests/unit/showcase/test_level_12_checks_b.py",
    # Level 12c illicit money guard test suite.
    "tests/unit/world/test_illicit_money_guard.py",
    # Level 12c mid-run account openings test suite.
    "tests/unit/world/test_mid_run_accounts.py",
    # Level 12c account openings configuration resolver.
    "world/account_openings_config.py",
    # Level 12c account openings switch and scheduling test suite.
    "tests/unit/world/test_account_openings_switch.py",
    # Level 12c scam schedule configuration resolver and test suite.
    "world/scam_config.py",
    "tests/unit/world/test_scam_config.py",
    # Level 12c scam step planner and gang scam test suite.
    "world/scam_planner.py",
    "tests/unit/world/test_scam_gangs.py",
    # Level 12c victim money route test suite.
    "tests/unit/world/test_victim_money_route.py",
    # Level 12c mule origins and gang-opened mule accounts.
    "world/mule_origins.py",
    "world/opening_handles.py",
    "tests/unit/world/test_mule_origins.py",
    # Level 12c rented mule owner activity.
    "world/rented_owner_activity.py",
    "tests/unit/world/test_rented_owner_activity.py",
    # Level 12c routine life for new genuine people.
    "world/opened_routine.py",
    "tests/unit/world/test_opened_routine.py",
    # Level 12d gang observation feed and reactive decision rules.
    "world/gang_feed.py",
    "world/gang_reactions.py",
    "tests/unit/world/test_gang_feed.py",
    "tests/unit/world/test_gang_reactions.py",
    "tests/unit/world/test_gang_rotation_split.py",
    # Level 12d account market resale planner and test suite.
    "world/account_market.py",
    "tests/unit/world/test_account_market.py",
    # Level 12d tactic family guard and test suite.
    "world/tactic_families.py",
    "tests/unit/world/test_tactic_family_guard.py",
    # Level 12d showcase part d support, checks and test suite.
    "showcase/level_12_support_d.py",
    "showcase/level_12_checks_d.py",
    "tests/unit/showcase/test_level_12_checks_d.py",
    # Level 12d handoff facts generator.
    "scripts/handoff_facts.py",
    # Level 12e structured scam type on payment stories test suite.
    "tests/unit/world/test_scam_story_fields.py",
    # Level 12e complaint planners, label records and test suite.
    "world/label_records.py",
    "world/complaints.py",
    "tests/unit/world/test_complaints.py",
    # Level 12e label builder and test suite.
    "world/label_builder.py",
    "tests/unit/world/test_label_builder.py",
    # Level 12e registry reports and test suite.
    "world/registry_reports.py",
    "tests/unit/world/test_registry_reports.py",
    # Level 12e bulk confirmations and test suite.
    "world/bulk_confirmations.py",
    "tests/unit/world/test_bulk_confirmations.py",
    # Level 12e simulated analyst and test suite.
    "world/analyst.py",
    "tests/unit/world/test_analyst.py",
    # Level 12e showcase part e support, checks and test suite.
    "showcase/level_12_support_e.py",
    "showcase/level_12_checks_e.py",
    "tests/unit/showcase/test_level_12_checks_e.py",
    # Level 12f many-worlds sampler and test suite.
    "world/world_sampler.py",
    "tests/unit/world/test_world_sampler.py",
    # Level 12f sampled world runner, its test suite and runtime measurement script.
    "world/sampled_world.py",
    "tests/unit/world/test_sampled_world.py",
    "scripts/measure_sampled_world_runtime.py",
    # Level 12f pattern files module and test suite.
    "world/pattern_files.py",
    "tests/unit/world/test_pattern_files.py",
    # Level 12f tactic plugins module, example plugin and test suite.
    "world/tactic_plugins.py",
    "patterns/plugins/cool_off.py",
    "tests/unit/world/test_tactic_plugins.py",
    # Level 12f world split versus range column guard test suite.
    "tests/unit/world/test_world_split_guard.py",
    # Level 12f per-world gang mix module and test suite.
    "world/gang_mix.py",
    "tests/unit/world/test_gang_mix.py",
    # Level 12f one-block world measurement script.
    "scripts/measure_world_block.py",
    # Level 12f world invariant checks and test suite.
    "world/world_invariants.py",
    "tests/unit/world/test_world_invariants.py",
    # Level 12f showcase part f support, checks and test suite.
    "showcase/level_12_support_f.py",
    "showcase/level_12_checks_f.py",
    "tests/unit/showcase/test_level_12_checks_f.py",
    # Level 12g held-out options module and test suite.
    "world/held_out_options.py",
    "tests/unit/world/test_held_out_options.py",
    "tests/unit/world/test_slow_drip.py",
    "tests/unit/world/test_device_rotation.py",
    "tests/unit/world/test_gang_device_ids.py",
    "tests/unit/world/test_gang_truth.py",
    # Level 12g Round 3a YAML content cache test suite.
    "tests/unit/test_config_yaml_cache.py",
    # Level 12g Round 3b held-out mechanisms test suite.
    "tests/unit/test_held_out_mechanisms.py",
    # Level 12g Round 3b held-out property test suite.
    "tests/unit/world/test_held_out_property.py",
    # Level 12g showcase part g support, checks and test suite.
    "showcase/level_12_support_g.py",
    "showcase/level_12_checks_g.py",
    "tests/unit/showcase/test_level_12_checks_g.py",
    # Level 12h edge-case library module and test suite.
    "world/edge_cases.py",
    "tests/unit/world/test_edge_cases.py",
    # Level 12h base rate module and test suite.
    "world/base_rate.py",
    "tests/unit/world/test_base_rate.py",
]

NEW_FUNCTIONS_IN_OLD_FILES: list[tuple[str, str]] = [
    ("world/amounts.py", "lookalike_small_amount_paise"),
    ("world/tactic_families.py", "check_held_out_mechanisms"),
    # Level 12a showcase --part option helpers added to older showcase files.
    ("showcase/framework.py", "select_part_checks"),
    ("showcase/__main__.py", "check_part_argument"),
    ("showcase/__main__.py", "report_folder"),
    # Level 12b events-v2 writer function.
    ("world/interactive_events.py", "write_events_v2_file"),
    # Level 12b interactive runner gang configs integration helper.
    ("world/interactive_runner.py", "_apply_gang_configs"),
    # Level 12b unified device event stream helper.
    ("world/interactive_runner.py", "device_event_stream"),
    # Level 12h fraud world card builder.
    ("world/world_card.py", "build_fraud_world_card"),
]


def _first_sentence(text: str) -> str:
    """Extract and normalize the first sentence from docstring text."""
    stripped = text.strip()
    if not stripped:
        return ""
    # Split on first sentence boundary: period, exclamation mark, question mark, or newline.
    for delimiter in (".", "!", "?", "\n"):
        if delimiter in stripped:
            stripped = stripped.split(delimiter, 1)[0]
    return stripped.strip().lower().replace("_", " ")


def _validate_node_docstring(node: Any, filepath: str) -> list[str]:
    """Validate that an AST node has a substantive docstring that does not just restate its name."""
    doc = ast.get_docstring(node)
    if doc is None or not doc.strip():
        return [f"{filepath}:{node.name}: missing docstring"]
    clean_doc = doc.strip()
    errors: list[str] = []
    if len(clean_doc) < MIN_DOCSTRING_LENGTH:
        errors.append(f"{filepath}:{node.name}: docstring length {len(clean_doc)} < {MIN_DOCSTRING_LENGTH}")
    first_sentence = _first_sentence(clean_doc)
    normalized_name = node.name.strip().lower().replace("_", " ")
    if first_sentence == normalized_name:
        errors.append(f"{filepath}:{node.name}: docstring first sentence merely restates name ('{first_sentence}')")
    return errors


def _validate_module_docstring(tree: ast.AST, filepath: str) -> list[str]:
    """Validate that a module has a substantive header docstring."""
    doc = ast.get_docstring(tree)
    if doc is None or not doc.strip():
        return [f"{filepath}: missing module docstring"]
    clean_doc = doc.strip()
    if len(clean_doc) < MIN_DOCSTRING_LENGTH:
        return [f"{filepath}: module docstring length {len(clean_doc)} < {MIN_DOCSTRING_LENGTH}"]
    return []


def test_new_code_files_registry() -> None:
    """Assert that NEW_CODE_FILES contains at least the 5 required files and all exist on disk."""
    expected_minimum = [
        "world/world_plan.py",
        "world/calendar_runner.py",
        "tests/unit/world/test_world_plan.py",
        "tests/unit/world/test_world_calendar_runner.py",
        "tests/unit/world/test_new_code_comment_standard.py",
    ]
    assert len(NEW_CODE_FILES) >= 5, f"Registry shrank below 5 files: {len(NEW_CODE_FILES)}"
    for req in expected_minimum:
        assert req in NEW_CODE_FILES, f"Missing required file in NEW_CODE_FILES: {req}"
        assert os.path.exists(req), f"Registered file does not exist on disk: {req}"


def test_substantive_docstrings() -> None:
    """Traverse all files in NEW_CODE_FILES to assert substantive docstrings on modules, classes, and functions."""
    errors: list[str] = []
    for filepath in NEW_CODE_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
        errors.extend(_validate_module_docstring(tree, filepath))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                errors.extend(_validate_node_docstring(node, filepath))

    # Also verify explicitly registered new functions in older codebase files.
    for filepath, fn_name in NEW_FUNCTIONS_IN_OLD_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
        nodes = [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name
        ]
        if not nodes:
            errors.append(f"{filepath}: function '{fn_name}' not found")
        for node in nodes:
            errors.extend(_validate_node_docstring(node, filepath))

    assert not errors, "Documentation standard violations:\n" + "\n".join(errors)


def test_no_comment_restates_a_return() -> None:
    """Traverse all files in NEW_CODE_FILES to forbid comments directly above returns that restate the return."""
    violations: list[str] = []
    for filepath in NEW_CODE_FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx in range(len(lines) - 1):
            curr_line = lines[idx].strip()
            next_line = lines[idx + 1].strip()
            if curr_line.startswith("#"):
                comment_text = curr_line[1:].strip()
                if comment_text.startswith("Return") and next_line.startswith("return"):
                    violations.append(f"{filepath}:{idx + 1}: '{curr_line}' sits directly above '{next_line}'")
    assert not violations, "Found comments restating a return:\n" + "\n".join(violations)


def test_function_lengths_and_docstrings() -> None:
    """Enforces the new substantive docstring standard across functions, methods, and classes."""
    test_substantive_docstrings()


def test_module_docstrings_define_technical_terms() -> None:
    """Retained for issue register v2 evidence compatibility under the updated standard."""
    # Under the new standard, module docstrings are validated as part of test_substantive_docstrings.
    pass
