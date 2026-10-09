"""
File: tests/unit/test_folder_check.py
Purpose: Tests the top-level folder classification check, using a temporary folder with fake top-level folders.
"""
from core import folder_check
from core.config_validator import load_yaml, validate_file

CLASSIFICATION = {"engine": ["world"], "not_engine": ["core", "docs", "models"]}
ENGINE_PATHS = ["world/", "models/mule_*"]


def make_folders(root, names):
    for name in names:
        (root / name).mkdir()


def test_real_repo_is_fully_classified():
    classification = load_yaml(folder_check.CLASSIFICATION_FILE)["folder_classification"]
    engine_paths = load_yaml(folder_check.NEW_ENGINE_PATHS_FILE)["new_engine_paths"]["paths"]
    folders = folder_check.top_level_folders(folder_check.REPO_ROOT)
    assert folder_check.check_folders(folders, classification, engine_paths) == []


def test_classification_file_has_label_and_reason():
    assert validate_file(folder_check.CLASSIFICATION_FILE) == []


def test_all_folders_listed_passes(tmp_path):
    make_folders(tmp_path, ["core", "docs"])
    folders = folder_check.top_level_folders(tmp_path, use_git=False)
    assert folders == ["core", "docs"]
    assert folder_check.check_folders(folders, CLASSIFICATION, ENGINE_PATHS) == []


def test_unlisted_folder_fails(tmp_path):
    make_folders(tmp_path, ["core", "docs", "sneaky_new_engine"])
    folders = folder_check.top_level_folders(tmp_path, use_git=False)

    errors = folder_check.check_folders(folders, CLASSIFICATION, ENGINE_PATHS)

    assert len(errors) == 1
    assert "sneaky_new_engine" in errors[0] and "not classified" in errors[0]


def test_folder_in_both_lists_fails():
    classification = {"engine": ["world"], "not_engine": ["world", "core"]}
    errors = folder_check.check_folders([], classification, ["world/"])
    assert any("both engine and not_engine" in error for error in errors)


def test_engine_folder_missing_from_new_engine_paths_fails():
    errors = folder_check.check_folders([], CLASSIFICATION, ["models/mule_*"])
    assert any("engine folder 'world' is not in config/new_engine_paths.yaml" in error for error in errors)


def test_watched_path_in_unclassified_folder_fails():
    errors = folder_check.check_folders([], CLASSIFICATION, ["world/", "mystery/part/"])
    assert any("'mystery/part/'" in error and "not classified" in error for error in errors)
