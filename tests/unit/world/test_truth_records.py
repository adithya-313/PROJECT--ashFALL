"""
WHAT: Unit tests for Level 12b hidden truth records and schema validation functions.
Verifies role taxonomy bounds, half-open temporal validity intervals, mule depth constraints,
mutual exclusion across spans, transaction narrative lengths, and runner switch-off refusal.

WHY: Enforces the hidden truth security and data integrity model. Erroneous, overlapping,
or misattributed truth specifications must be rejected deterministically before runner startup.

TERM: hidden truth: information known to the simulation engine but forbidden from bank feeds.
TERM: time span: a temporal interval [valid_from, valid_to) during which an assignment remains active.
TERM: half-open span: an interval inclusive of valid_from and exclusive of valid_to; valid_to may be None.
TERM: mule depth: distance in payment hops from the initial compromised victim to an illicit account.
TERM: payment story: explanatory metadata attributing a fraud transaction to a gang, block, layer, and description.
"""

import re

import pytest

from core.config_validator import load_yaml
from world.calendar_runner import PlannedPayment
from world.fast_mode import FastRunSettings
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import SWITCH_OFF_MESSAGE, run_interactive_world
from world.run_identity import REPO_ROOT
from world.truth_records import (
    DeviceControlSpan,
    PaymentStory,
    RoleSpan,
    TruthFacts,
    load_truth_config,
    validate_mule_depth,
    validate_no_overlapping_device_control_spans,
    validate_no_overlapping_role_spans,
    validate_payment_stories_length,
    validate_role_in_config,
    validate_role_origin,
    validate_time_span,
    validate_truth_facts,
)

FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
TEST_CODE_VERSION = "test-version"


def test_role_must_be_in_config() -> None:
    """
    WHAT: Asserts that validate_role_in_config accepts allowed roles defined in truth_store.yaml
    and strictly raises ValueError when an unlisted role string is supplied.

    WHY: Truth records must strictly adhere to the project's four canonical role archetypes
    (mule, collector, cash_out, victim) to guarantee consistent evaluation downstream.

    Args:
        None.

    Returns:
        None. Fails if invalid roles are permitted or valid roles rejected.

    Example:
        validate_role_in_config(RoleSpan('acc_1', 'mule', ...), ['mule', ...])  # Passes
    """
    # 1. Load the canonical list of allowed roles from configuration.
    allowed = list(load_truth_config()["truth_store"]["roles"].keys())
    valid_span = RoleSpan("acc_001", "mule", "gang_alpha", 1, 0, 1440)
    invalid_span = RoleSpan("acc_001", "operator", "gang_alpha", 1, 0, 1440)

    # 2. Assert valid role passes validation without raising.
    validate_role_in_config(valid_span, allowed)

    # 3. Assert unlisted role raises ValueError naming the offending role.
    with pytest.raises(ValueError, match="has invalid role 'operator'"):
        validate_role_in_config(invalid_span, allowed)


def test_time_span_valid_bounds() -> None:
    """
    WHAT: Asserts that validate_time_span requires non-negative valid_from and valid_to > valid_from
    when specified, while permitting valid_to to be None for open-ended assignments.

    WHY: Time spans define half-open simulation intervals [valid_from, valid_to); negative start times
    or inverted boundaries contradict linear simulation time.

    Args:
        None.

    Returns:
        None. Fails if invalid interval bounds do not raise ValueError.

    Example:
        validate_time_span(RoleSpan('acc_1', 'mule', 'g1', 1, 0, 100))  # Passes
    """
    # 1. Assert valid bounded and unbounded spans pass verification.
    validate_time_span(RoleSpan("acc_001", "mule", "g1", 1, 0, 100))
    validate_time_span(RoleSpan("acc_001", "mule", "g1", 1, 100, None))

    # 2. Assert negative valid_from is rejected with a descriptive ValueError.
    with pytest.raises(ValueError, match="negative valid_from: -10"):
        validate_time_span(RoleSpan("acc_001", "mule", "g1", 1, -10, 100))

    # 3. Assert inverted or zero-length interval is rejected.
    with pytest.raises(ValueError, match="less than or equal to valid_from"):
        validate_time_span(RoleSpan("acc_001", "mule", "g1", 1, 100, 100))
    with pytest.raises(ValueError, match="less than or equal to valid_from"):
        validate_time_span(RoleSpan("acc_001", "mule", "g1", 1, 100, 50))


def test_mule_depth_requirements() -> None:
    """
    WHAT: Asserts that validate_mule_depth enforces depth >= 1 for the mule role,
    and requires depth to be strictly None for all non-mule roles.

    WHY: Mule depth measures payment forwarding distance from the initial compromised victim;
    non-mules (collectors, cash-outs, victims) do not represent hop layers.

    Args:
        None.

    Returns:
        None. Fails if mule depth constraints are violated without raising.

    Example:
        validate_mule_depth(RoleSpan('acc_1', 'mule', 'g1', 1, 0, 100))  # Passes
    """
    # 1. Assert valid mule and non-mule spans pass verification.
    validate_mule_depth(RoleSpan("acc_001", "mule", "g1", 1, 0, 100))
    validate_mule_depth(RoleSpan("acc_002", "victim", "g1", None, 0, 100))

    # 2. Assert mule with None or non-positive depth raises ValueError.
    with pytest.raises(ValueError, match="requires depth >= 1, got None"):
        validate_mule_depth(RoleSpan("acc_001", "mule", "g1", None, 0, 100))
    with pytest.raises(ValueError, match="requires depth >= 1, got 0"):
        validate_mule_depth(RoleSpan("acc_001", "mule", "g1", 0, 0, 100))

    # 3. Assert non-mule with non-None depth raises ValueError.
    with pytest.raises(ValueError, match="must have depth None, got 2"):
        validate_mule_depth(RoleSpan("acc_002", "cash_out", "g1", 2, 0, 100))


def test_no_overlapping_role_spans() -> None:
    """
    WHAT: Asserts that validate_no_overlapping_role_spans rejects overlapping temporal intervals
    for the same account while permitting abutting intervals and concurrent spans for different accounts.

    WHY: An account cannot simultaneously be operated under multiple conflicting roles or gangs.
    Half-open spans [0, 100) and [100, 200) abut without sharing points.

    Args:
        None.

    Returns:
        None. Fails if overlapping spans for the same account are accepted.

    Example:
        validate_no_overlapping_role_spans([span_a1, span_a2])  # Rejects if overlapping
    """
    # 1. Assert abutting spans and distinct account spans pass without error.
    span_first = RoleSpan("acc_001", "mule", "g1", 1, 0, 100)
    span_second = RoleSpan("acc_001", "mule", "g1", 2, 100, 200)
    span_other_acc = RoleSpan("acc_002", "mule", "g2", 1, 50, 150)
    validate_no_overlapping_role_spans([span_first, span_second, span_other_acc])

    # 2. Assert overlapping spans for the same account raise ValueError.
    span_overlap = RoleSpan("acc_001", "cash_out", "g1", None, 50, 150)
    with pytest.raises(ValueError, match="Overlapping role spans for account 'acc_001'"):
        validate_no_overlapping_role_spans([span_first, span_overlap])

    # 3. Assert overlap with an open-ended span raises ValueError.
    span_unending = RoleSpan("acc_001", "mule", "g1", 1, 0, None)
    span_inside = RoleSpan("acc_001", "cash_out", "g1", None, 50, 100)
    with pytest.raises(ValueError, match="Overlapping role spans for account 'acc_001'"):
        validate_no_overlapping_role_spans([span_unending, span_inside])


def test_no_overlapping_device_control_spans() -> None:
    """
    WHAT: Asserts that validate_no_overlapping_device_control_spans rejects overlapping control spans
    for the exact same (device_id, account_id) pair, while allowing non-overlapping or distinct-pair spans.

    WHY: A specific handset operating a given account cannot simultaneously be claimed by conflicting
    takeover periods with differing compromise indicators.

    Args:
        None.

    Returns:
        None. Fails if overlapping device control records are accepted.

    Example:
        validate_no_overlapping_device_control_spans([dev_span_1, dev_span_2])
    """
    # 1. Assert abutting spans for the same pair and overlapping spans for different devices pass.
    span_1 = DeviceControlSpan("dev_1", "acc_1", "g1", 0, 100, False, False, False, False)
    span_2 = DeviceControlSpan("dev_1", "acc_1", "g1", 100, 200, False, False, False, False)
    span_other_dev = DeviceControlSpan("dev_2", "acc_1", "g1", 50, 150, False, False, False, False)
    validate_no_overlapping_device_control_spans([span_1, span_2, span_other_dev])

    # 2. Assert overlapping spans for the exact same device and account raise ValueError.
    span_overlap = DeviceControlSpan("dev_1", "acc_1", "g2", 50, 150, True, False, False, False)
    with pytest.raises(ValueError, match="Overlapping device control spans for device 'dev_1' and account 'acc_1'"):
        validate_no_overlapping_device_control_spans([span_1, span_overlap])


def test_payment_stories_length_rule() -> None:
    """
    WHAT: Asserts that validate_payment_stories_length accepts stories that are either completely empty
    or exactly equal in length to the count of planned fraud payments, rejecting partial story lists.

    WHY: If transaction stories are supplied, every fraud payment must carry an attribution narrative;
    partial story lists would leave fraudulent payments unexplained.

    Args:
        None.

    Returns:
        None. Fails if length mismatches are permitted.

    Example:
        validate_payment_stories_length([story_1], 1)  # Passes
    """
    # 1. Assert empty stories collection passes for any count of fraud payments.
    validate_payment_stories_length([], 0)
    validate_payment_stories_length([], 3)

    # 2. Assert exact length match passes.
    story = PaymentStory("gang_alpha", "route", 1, "Dispersal transfer")
    validate_payment_stories_length([story], 1)
    validate_payment_stories_length([story, story], 2)

    # 3. Assert length mismatch raises ValueError naming both counts.
    with pytest.raises(ValueError, match="payment_stories has length 1; must be empty or match fraud payments count"):
        validate_payment_stories_length([story], 0)
    with pytest.raises(ValueError, match="payment_stories has length 1; must be empty or match fraud payments count"):
        validate_payment_stories_length([story], 2)


def test_switch_off_refuses_non_empty_truth() -> None:
    """
    WHAT: Asserts that run_interactive_world with the fraud switch off strictly refuses FraudInputs
    containing non-empty truth facts with SWITCH_OFF_MESSAGE.

    WHY: Enforces ground rule 13: when fraud_enabled is False, no fraud parameters of any kind
    (payments, outside events, or truth assertions) may be accepted.

    Args:
        None.

    Returns:
        None. Fails if the run executes rather than raising ValueError.

    Example:
        run_interactive_world(TINY, RunnerSettings(), FraudInputs(truth=TruthFacts(roles=(...)))) raises.
    """
    # 1. Build a non-empty truth facts container with one role span.
    role = RoleSpan("acc_person_000001", "victim", "gang_alpha", None, 0, 1440)
    inputs = FraudInputs(truth=TruthFacts(roles=(role,)))

    # 2. Assert executing with fraud switch off raises the canonical switch-off refusal message.
    with pytest.raises(ValueError, match=SWITCH_OFF_MESSAGE):
        run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), inputs, code_version=TEST_CODE_VERSION)


def test_validate_truth_facts_composite() -> None:
    """
    WHAT: Asserts that validate_truth_facts validates a well-formed composite TruthFacts container holding role spans,
    device control spans, and transaction stories without error, and delegates to config when allowed_roles is None.

    WHY: Verifies the composite facade that the reactive runner invokes before starting simulation execution.

    Args:
        None.

    Returns:
        None. Fails if a valid composite TruthFacts collection raises ValueError.

    Example:
        validate_truth_facts(TruthFacts(), 0)  # Passes
    """
    # 1. Build a valid composite TruthFacts object with one role, device span, and story matching 1 payment.
    role = RoleSpan("acc_person_000001", "mule", "gang_alpha", 1, 0, 1440)
    device_span = DeviceControlSpan(
        "dev_00001", "acc_person_000001", "gang_alpha", 0, 1440, False, False, False, False
    )
    story = PaymentStory("gang_alpha", "route", 1, "Dispersal transfer")
    truth = TruthFacts(roles=(role,), device_control=(device_span,), payment_stories=(story,))

    # 2. Assert validation succeeds with default config loading and payment count 1.
    validate_truth_facts(truth, payment_count=1)


@pytest.mark.parametrize(
    ("bad_truth", "error_match"),
    [
        (
            TruthFacts(
                roles=(RoleSpan("acc_person_000001", "operator", "gang_alpha", 1, 0, 100),),
                payment_stories=(PaymentStory("gang_alpha", "route", 1, "legit story"),),
            ),
            "RoleSpan for account 'acc_person_000001' has invalid role 'operator'",
        ),
        (
            TruthFacts(
                roles=(RoleSpan("acc_person_000001", "mule", "gang_alpha", 1, 100, 100),),
                payment_stories=(PaymentStory("gang_alpha", "route", 1, "legit story"),),
            ),
            "Span for 'acc_person_000001' has valid_to (100) less than or equal to valid_from (100)",
        ),
        (
            TruthFacts(
                roles=(RoleSpan("acc_person_000001", "mule", "gang_alpha", 1, -10, 100),),
                payment_stories=(PaymentStory("gang_alpha", "route", 1, "legit story"),),
            ),
            "Span for 'acc_person_000001' has negative valid_from: -10",
        ),
        (
            TruthFacts(
                roles=(RoleSpan("acc_person_000001", "mule", "gang_alpha", None, 0, 100),),
                payment_stories=(PaymentStory("gang_alpha", "route", 1, "legit story"),),
            ),
            "RoleSpan for mule account 'acc_person_000001' requires depth >= 1, got None",
        ),
        (
            TruthFacts(
                roles=(RoleSpan("acc_person_000001", "victim", "gang_alpha", 2, 0, 100),),
                payment_stories=(PaymentStory("gang_alpha", "route", 1, "legit story"),),
            ),
            "RoleSpan for non-mule account 'acc_person_000001' (role 'victim') must have depth None, got 2",
        ),
        (
            TruthFacts(
                roles=(
                    RoleSpan("acc_person_000001", "mule", "gang_alpha", 1, 0, 100),
                    RoleSpan("acc_person_000001", "collector", "gang_alpha", None, 50, 150),
                ),
                payment_stories=(PaymentStory("gang_alpha", "route", 1, "legit story"),),
            ),
            "Overlapping role spans for account 'acc_person_000001': [0, 100) and [50, 150)",
        ),
        (
            TruthFacts(
                device_control=(
                    DeviceControlSpan(
                        "dev_001", "acc_person_000001", "gang_alpha", 0, 100, False, False, False, False
                    ),
                    DeviceControlSpan(
                        "dev_001", "acc_person_000001", "gang_alpha", 50, 150, True, False, False, False
                    ),
                ),
                payment_stories=(PaymentStory("gang_alpha", "route", 1, "legit story"),),
            ),
            "Overlapping device control spans for device 'dev_001' and account "
            "'acc_person_000001': [0, 100) and [50, 150)",
        ),
        (
            TruthFacts(
                payment_stories=(
                    PaymentStory("gang_alpha", "route", 1, "first story"),
                    PaymentStory("gang_alpha", "route", 2, "extra story"),
                ),
            ),
            "payment_stories has length 2; must be empty or match fraud payments count (1)",
        ),
    ],
)
def test_runner_refuses_invalid_truth_facts(bad_truth: TruthFacts, error_match: str) -> None:
    """
    WHAT: Asserts that run_interactive_world with fraud enabled strictly rejects invalid TruthFacts
    specifications across role taxonomy, temporal bounds, mule depths, span overlaps, and story counts.

    WHY: Truth facts are validated before runner loop execution begins; any malformed or inconsistent truth
    invariant must halt the runner immediately with a descriptive ValueError naming the bad record.

    Args:
        bad_truth: TruthFacts instance containing an intentional schema violation.
        error_match: Substring expected within the raised ValueError message.

    Returns:
        None. Fails if execution proceeds or the raised exception message does not match.

    Example:
        # Calling run_interactive_world with an invalid role raises ValueError.
        # with pytest.raises(ValueError, match="invalid role"): ...
    """
    # 1. Configure a small valid fraud payment to pair with the invalid truth facts.
    fraud_payment = PlannedPayment(
        minute=10,
        from_account="acc_person_000001",
        to_account="acc_person_000002",
        amount_paise=5000,
        creation_index=0,
        purpose="everyday",
    )
    inputs = FraudInputs(payments=(fraud_payment,), truth=bad_truth)

    # 2. Execute simulation and verify deterministic ValueError rejection naming the violation.
    with pytest.raises(ValueError, match=re.escape(error_match)):
        run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), inputs, code_version=TEST_CODE_VERSION)




ALLOWED_ORIGINS = load_truth_config()["truth_store"]["mule_origins"]


def test_origin_is_allowed_only_on_mule_roles_from_the_configured_list() -> None:
    """
    Verifies role origin validation: a configured origin on a mule passes, no origin passes on any role, an
    origin on a victim is refused, and an origin outside the configured list is refused on a mule.
    """
    for origin in ALLOWED_ORIGINS:
        validate_role_origin(RoleSpan("acc_a", "mule", "g1", 1, 0, 100, origin), ALLOWED_ORIGINS)
    validate_role_origin(RoleSpan("acc_a", "victim", "g1", None, 0, None), ALLOWED_ORIGINS)

    with pytest.raises(ValueError, match="origin is only allowed on mule roles, not 'victim'"):
        validate_role_origin(RoleSpan("acc_a", "victim", "g1", None, 0, None, "rented"), ALLOWED_ORIGINS)
    with pytest.raises(ValueError, match="invalid origin 'stolen'"):
        validate_role_origin(RoleSpan("acc_a", "mule", "g1", 1, 0, 100, "stolen"), ALLOWED_ORIGINS)


def test_origin_list_is_read_from_config_and_checked_by_the_composite_validator() -> None:
    """
    Verifies the origin list comes from config: narrowing the list refuses a formerly valid origin, and
    validate_truth_facts applies the origin rule to every role span.
    """
    assert ALLOWED_ORIGINS == ["rented", "borrowed_kyc", "business"]
    span = RoleSpan("acc_a", "mule", "g1", 1, 0, 100, "business")
    with pytest.raises(ValueError, match="invalid origin 'business'"):
        validate_role_origin(span, ["rented"])

    validate_truth_facts(TruthFacts(roles=(span,)), 0)
    bad = RoleSpan("acc_a", "victim", "g1", None, 0, None, "rented")
    with pytest.raises(ValueError, match="origin is only allowed on mule roles"):
        validate_truth_facts(TruthFacts(roles=(bad,)), 0)
