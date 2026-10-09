"""
File: tests/unit/test_level10_order_check.py
Purpose: Tests the pre-registration order check on small FAKE git histories built in a temporary folder.
         The check must FAIL when engine code was committed before config/evaluation.yaml, and pass otherwise.
"""
import os
import subprocess

from core import level10_check
from core.config_validator import validate_file

ENGINE_PATHS = ["world/", "models/mule_*"]


def commit_file(repo, relative_path, date):
    """Create a file in the fake repo and commit it with the given date (ISO text)."""
    file_path = repo / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x\n", encoding="utf-8")
    env = dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    identity = ["-c", "user.name=tester", "-c", "user.email=tester@example.com"]
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(repo)] + identity + ["commit", "-q", "-m", f"add {relative_path}"],
                   check=True, capture_output=True, env=env)


def make_repo(tmp_path):
    """An empty git repo in a temporary folder."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    return tmp_path


def test_engine_code_older_than_evaluation_fails(tmp_path):
    repo = make_repo(tmp_path)
    commit_file(repo, "world/simulator.py", "2026-01-01T10:00:00+00:00")
    commit_file(repo, "config/evaluation.yaml", "2026-02-01T10:00:00+00:00")

    proof = level10_check.preregistration_proof(repo, ENGINE_PATHS)

    assert proof["result"] == "NOT OK"
    assert "world/" in proof["sentence"]


def test_pattern_path_older_than_evaluation_fails(tmp_path):
    repo = make_repo(tmp_path)
    commit_file(repo, "models/mule_v2/model.txt", "2026-01-01T10:00:00+00:00")
    commit_file(repo, "config/evaluation.yaml", "2026-02-01T10:00:00+00:00")

    proof = level10_check.preregistration_proof(repo, ENGINE_PATHS)

    assert proof["result"] == "NOT OK"
    assert "models/mule_*" in proof["sentence"]


def test_engine_code_after_evaluation_passes(tmp_path):
    repo = make_repo(tmp_path)
    commit_file(repo, "config/evaluation.yaml", "2026-01-01T10:00:00+00:00")
    commit_file(repo, "world/simulator.py", "2026-02-01T10:00:00+00:00")

    proof = level10_check.preregistration_proof(repo, ENGINE_PATHS)

    assert proof["result"] == "OK"
    assert "world/" in proof["new_engine_dates"]


def test_no_engine_code_yet_passes(tmp_path):
    repo = make_repo(tmp_path)
    commit_file(repo, "config/evaluation.yaml", "2026-01-01T10:00:00+00:00")

    proof = level10_check.preregistration_proof(repo, ENGINE_PATHS)

    assert proof["result"] == "OK"
    assert proof["new_engine_dates"] == {}


def test_missing_evaluation_commit_fails(tmp_path):
    repo = make_repo(tmp_path)
    commit_file(repo, "README.md", "2026-01-01T10:00:00+00:00")

    proof = level10_check.preregistration_proof(repo, ENGINE_PATHS)

    assert proof["result"] == "NOT OK"


def test_missing_history_is_reported(tmp_path):
    # A folder that is not a git repository has no history at all.
    assert "not a git repository" in level10_check.git_history_problem(tmp_path)


def test_shallow_history_is_reported(tmp_path):
    source = make_repo(tmp_path / "source")
    commit_file(source, "a.txt", "2026-01-01T10:00:00+00:00")
    commit_file(source, "b.txt", "2026-01-02T10:00:00+00:00")
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", source.as_uri(), str(shallow)],
                   check=True, capture_output=True)

    assert "shallow" in level10_check.git_history_problem(shallow)
    assert level10_check.git_history_problem(source) == ""


def test_new_engine_paths_config_is_valid_and_lists_the_expected_paths():
    assert validate_file(level10_check.NEW_ENGINE_PATHS_FILE) == []
    paths = level10_check.load_new_engine_paths()
    for expected in ["world/", "features_m1/", "models/mule_*", "api/v1/"]:
        assert expected in paths
