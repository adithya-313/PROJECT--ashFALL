"""
WHAT: Unit tests for environment variable separation (envs/truth.env.example): asserts that operational configuration
files (docker-compose.yml, .env.example) never define or reference the hidden truth directory variable, while
the dedicated environment template (envs/truth.env.example) properly defines it alongside explanatory rationale.

WHY: Ground Rule 15 and Level 12b architecture mandate strict isolation between operational banking deployments and
synthetic simulation truth. Operational containers boot without truth environment knowledge, ensuring zero leakage.

TERM: operational configuration: settings files (docker-compose.yml, .env.example) powering production banking services.
TERM: truth environment file: dedicated configuration template (envs/truth.env.example) loaded only by evaluation tools.
TERM: hidden truth: ground-truth records describing illicit roles, device controls, and payment stories.
TERM: architectural separation: isolating truth metadata so operational services cannot observe simulation reality.
"""

from world.run_identity import REPO_ROOT

DOCKER_COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
TRUTH_ENV_PATH = REPO_ROOT / "envs" / "truth.env.example"
TRUTH_ENV_VAR = "ASHFALL_TRUTH_DIR"


def test_operational_configurations_exclude_truth_environment_variable() -> None:
    """
    WHAT: Asserts that neither docker-compose.yml nor .env.example contains any reference to ASHFALL_TRUTH_DIR.

    WHY: Operational bank services boot using docker-compose.yml and .env.example; exposing the truth directory
    variable to operational container definitions creates a risk of banking modules discovering simulation labels.
    Verifying that both files omit this variable guarantees separation across local and production orchestrations.

    Args:
        None.

    Returns:
        None. Fails if either operational configuration file mentions ASHFALL_TRUTH_DIR.

    Example:
        # Asserts TRUTH_ENV_VAR is not in docker-compose.yml or .env.example text.
        # test_operational_configurations_exclude_truth_environment_variable()
    """
    # 1. Read source content of docker-compose.yml and verify exclusion.
    docker_compose_text = DOCKER_COMPOSE_PATH.read_text(encoding="utf-8")
    assert TRUTH_ENV_VAR not in docker_compose_text, (
        f"Operational configuration '{DOCKER_COMPOSE_PATH.name}' must not contain '{TRUTH_ENV_VAR}'."
    )

    # 2. Read source content of .env.example and verify exclusion.
    env_example_text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    assert TRUTH_ENV_VAR not in env_example_text, (
        f"Operational template '{ENV_EXAMPLE_PATH.name}' must not contain '{TRUTH_ENV_VAR}'."
    )


def test_dedicated_truth_env_defines_variable_with_separation_rationale() -> None:
    """
    WHAT: Asserts that envs/truth.env.example exists, defines ASHFALL_TRUTH_DIR, and contains non-empty comments
    explaining why this configuration is maintained in a dedicated file outside .env.example.

    WHY: Ground Rule 15 and Level 12b design require evaluation workflows to explicitly configure truth storage
    while documenting the security rationale behind keeping operational and evaluation environments separate.

    Args:
        None.

    Returns:
        None. Fails if truth.env.example is missing, lacks the variable, or lacks documentation.

    Example:
        # Asserts envs/truth.env.example defines ASHFALL_TRUTH_DIR with explanatory header.
        # test_dedicated_truth_env_defines_variable_with_separation_rationale()
    """
    # 1. Verify existence of dedicated truth environment example template.
    assert TRUTH_ENV_PATH.exists(), f"Dedicated truth env file missing at '{TRUTH_ENV_PATH}'."
    truth_env_lines = TRUTH_ENV_PATH.read_text(encoding="utf-8").splitlines()

    # 2. Verify that ASHFALL_TRUTH_DIR is explicitly assigned.
    variable_assignments = [
        line.strip() for line in truth_env_lines
        if line.strip().startswith(f"{TRUTH_ENV_VAR}=")
    ]
    assert len(variable_assignments) == 1, (
        f"Expected exactly one assignment for '{TRUTH_ENV_VAR}', found: {variable_assignments}"
    )
    assert variable_assignments[0] == f"{TRUTH_ENV_VAR}=runs/truth"

    # 3. Verify that non-empty comments explain the architectural separation.
    comment_lines = [
        line.strip() for line in truth_env_lines
        if line.strip().startswith("#") and len(line.strip()) > 1
    ]
    assert len(comment_lines) >= 5, "Expected explanatory comment header documenting environment separation."
    combined_comments = " ".join(comment_lines).lower()
    assert "operational" in combined_comments or "bank" in combined_comments
    assert "evaluation" in combined_comments or "benchmark" in combined_comments
