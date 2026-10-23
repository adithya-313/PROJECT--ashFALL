"""
WHAT: Data records and schema validation functions for the Level 12b hidden truth store.
Defines immutable representations for account role spans, device control spans, transaction stories,
and truth fact collections, enforcing temporal boundaries and organizational invariants.

WHY: Fraud simulation requires maintaining ground truth (mule accounts, controlling gangs, attack narratives)
separately from bank-visible transaction logs. Strict pre-execution validation prevents corrupted or
contradictory truth specifications from ever entering the reactive runner loop.

TERM: hidden truth: information known to the simulation engine but forbidden from bank feeds.
TERM: time span: a temporal interval [valid_from, valid_to) during which an assignment remains active.
TERM: half-open span: an interval inclusive of valid_from and exclusive of valid_to; valid_to may be None.
TERM: mule depth: distance in payment hops from the initial compromised victim to an illicit account.
TERM: payment story: explanatory metadata attributing a fraud transaction to a gang, block, layer, and description.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core.config_validator import load_yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
TRUTH_CONFIG_PATH: Path = REPO_ROOT / "config" / "truth_store.yaml"

# The two steps of a scam payment; labels use them to tell what a victim lost from what a mule paid back.
SCAM_STEP_VICTIM_PAYMENT: str = "victim_payment"
SCAM_STEP_PAYOUT_BACK: str = "payout_back"
SCAM_STEPS: tuple[str, str] = (SCAM_STEP_VICTIM_PAYMENT, SCAM_STEP_PAYOUT_BACK)


@dataclass(frozen=True)
class RoleSpan:
    """
    WHAT: A temporal assignment of an account to an illicit fraud role under gang control.
    Records account identifier, role name, controlling gang, mule depth, and validity interval.

    WHY: Accounts (such as rented mules) are genuine before and after rental periods; storing roles
    as half-open time spans captures dynamic identity changes without mutating static account tables.

    Args:
        account_id: The unique identifier of the assigned account, e.g. "acc_person_000001".
        role: The assigned role name, one of "mule", "collector", "cash_out", or "victim".
        gang_id: Identifier of the gang controlling the account during this span.
        depth: Layer number in the laundering chain (>= 1 for mules, None for non-mules).
        valid_from: Starting minute of the role assignment (inclusive, >= 0).
        valid_to: Ending minute of the assignment (exclusive, > valid_from, or None if unending).
        origin: How the gang came to control a mule account, one of the mule_origins in config/truth_store.yaml;
            None when unknown and always None for roles other than "mule".

    Returns:
        A frozen RoleSpan.

    Example:
        RoleSpan("acc_person_000001", "mule", "gang_alpha", 1, 1440, 2880)
    """

    account_id: str
    role: str
    gang_id: str
    depth: int | None
    valid_from: int
    valid_to: int | None
    origin: str | None = None
    active_from: int | None = None
    burned_from: int | None = None


@dataclass(frozen=True)
class DeviceControlSpan:
    """
    WHAT: A temporal record of gang control over a specific device associated with an account.
    Captures operational compromise flags including SIM swap, linked number switch, emulator, and remote access.

    WHY: Ground rule 15 requires tracking compromised hardware spans to distinguish authorized
    account holder logins from gang device manipulation during forensic evaluation.

    Args:
        device_id: Unique hardware or client identifier of the controlled device.
        account_id: Account identifier operated through this device during the span.
        gang_id: Identifier of the gang exercising control over the device.
        valid_from: Starting minute of device control (inclusive, >= 0).
        valid_to: Ending minute of device control (exclusive, > valid_from, or None).
        sim_swap: Boolean indicating whether the mobile number underwent unauthorized SIM reissuance.
        linked_number_switch: Boolean indicating whether the registered UPI mobile number was altered.
        emulator: Boolean indicating whether the transaction executed on an emulated virtual handset.
        remote_access: Boolean indicating presence of remote screen control or overlay software.

    Returns:
        A frozen DeviceControlSpan.

    Example:
        DeviceControlSpan("dev_00001", "acc_person_000001", "gang_alpha", 1440, 2880, False, False, True, False)
    """

    device_id: str
    account_id: str
    gang_id: str
    valid_from: int
    valid_to: int | None
    sim_swap: bool
    linked_number_switch: bool
    emulator: bool
    remote_access: bool


@dataclass(frozen=True)
class PaymentStory:
    """
    WHAT: Explanatory forensic narrative metadata associated with an injected fraud payment.
    Identifies the executing gang, the tactical attack block, routing layer, and narrative description.

    WHY: Connects individual synthetic payment events to the strategic intent of a simulated gang campaign,
    allowing evaluation benchmarks to measure detection performance across distinct attack tactics.

    Args:
        gang_id: Identifier of the gang orchestrating the illicit payment transaction.
        block: Descriptive tactic phase or execution block, e.g. "initial_compromise" or "route".
        layer: Payment hop or layer depth in a multi-hop money movement scheme, or None.
        description: Natural-language narrative summary of the fraudulent transfer.
        scam_type: The scam type of a scam payment (investment, job or the remainder type); None for every other
            payment. Kept here so no code has to read the type out of description text.
        scam_step: SCAM_STEP_VICTIM_PAYMENT for a victim paying a mule, SCAM_STEP_PAYOUT_BACK for a mule paying a
            victim; None for every other payment.

    Returns:
        A frozen PaymentStory.

    Raises:
        ValueError: If scam_step is set without scam_type, or is not one of the two scam step constants.

    Example:
        PaymentStory("gang_alpha", "route", 1, "First-hop dispersion transfer from compromised account")
    """

    gang_id: str
    block: str
    layer: int | None
    description: str
    scam_type: str | None = None
    scam_step: str | None = None

    def __post_init__(self) -> None:
        """
        Checks the scam fields when a story is built, so a story with a step but no type can never exist. Raises
        ValueError for a scam_step that is set without scam_type or that is not one of SCAM_STEPS.
        """
        if self.scam_step is None:
            return
        if self.scam_type is None:
            raise ValueError(f"PaymentStory scam_step '{self.scam_step}' is set but scam_type is None")
        if self.scam_step not in SCAM_STEPS:
            raise ValueError(f"PaymentStory scam_step '{self.scam_step}' must be one of {SCAM_STEPS}")


@dataclass(frozen=True)
class TruthFacts:
    """
    WHAT: Container holding all hidden truth assertions provided to a simulation run.
    Groups account role spans, device control spans, and transaction forensic payment stories.

    WHY: Packages pre-configured or tactic-generated truth facts into a single immutable payload
    embedded inside FraudInputs for runner consumption and truth store generation.

    Args:
        roles: Sequence of account role validity spans across simulation time.
        device_control: Sequence of device takeover and hardware control spans.
        payment_stories: Sequence of transaction narratives corresponding to injected fraud payments.

    Returns:
        A frozen TruthFacts collection.

    Example:
        TruthFacts(roles=(role_span,), device_control=(dev_span,), payment_stories=(story,))
    """

    roles: tuple[RoleSpan, ...] = ()
    device_control: tuple[DeviceControlSpan, ...] = ()
    payment_stories: tuple[PaymentStory, ...] = ()


def load_truth_config(path: Path = TRUTH_CONFIG_PATH) -> dict[str, Any]:
    """
    WHAT: Loads the truth store specification from config/truth_store.yaml.
    Extracts policy constraints including allowed roles, file names, environment variables, and folder scopes.

    WHY: Centralizes schema parameters and permission boundaries in configuration rather than hardcoding.

    Args:
        path: Filesystem path to the truth store YAML configuration file.

    Returns:
        Dictionary mapping truth configuration block names to validated values.

    Example:
        allowed = list(load_truth_config()["truth_store"]["roles"].keys())  # ["mule", "collector", ...]
    """
    # Load and parse the validated truth store configuration block.
    return load_yaml(path)


def validate_role_in_config(role_span: RoleSpan, allowed_roles: Sequence[str]) -> None:
    """
    WHAT: Validates that a role span assigns an allowed role defined in configuration.
    Raises ValueError with a descriptive sentence naming the invalid role and account if not permitted.

    WHY: Enforces strict classification taxonomy; unrecognized or typo-ridden roles cannot enter truth data.

    Args:
        role_span: The RoleSpan instance to validate.
        allowed_roles: Collection of permitted role name strings from truth_store.yaml.

    Returns:
        None.

    Example:
        validate_role_in_config(RoleSpan("acc_a", "mule", "g1", 1, 0, 100), ["mule", "victim"])
    """
    # Assert the assigned role exists in the permitted configuration list.
    if role_span.role not in allowed_roles:
        raise ValueError(
            f"RoleSpan for account '{role_span.account_id}' has invalid role '{role_span.role}'; "
            f"must be one of {', '.join(allowed_roles)}"
        )


def validate_role_origin(role_span: RoleSpan, allowed_origins: Sequence[str]) -> None:
    """
    WHAT: Validates a role span's origin: it may be set only on a "mule" span, and then only to one of the origins
    listed in config/truth_store.yaml under mule_origins.

    WHY: Origin describes how a mule account was obtained, so it means nothing on a victim or collector, and a
    misspelled origin would silently split evaluation groups.

    Args:
        role_span: The RoleSpan instance to validate.
        allowed_origins: Permitted origin names from truth_store.yaml.

    Returns:
        None. Raises ValueError with one sentence naming the account when the origin is not allowed.

    Example:
        validate_role_origin(RoleSpan("acc_a", "mule", "g1", 1, 0, 100, "rented"), ["rented", "business"])
    """
    if role_span.origin is None:
        return
    if role_span.role != "mule":
        raise ValueError(
            f"RoleSpan for account '{role_span.account_id}' has origin '{role_span.origin}' "
            f"but origin is only allowed on mule roles, not '{role_span.role}'."
        )
    if role_span.origin not in allowed_origins:
        raise ValueError(
            f"RoleSpan for account '{role_span.account_id}' has invalid origin '{role_span.origin}'; "
            f"must be one of {', '.join(allowed_origins)}"
        )


def validate_time_span(span: RoleSpan | DeviceControlSpan) -> None:
    """
    WHAT: Validates that a temporal validity span is well-formed: valid_from must be non-negative,
    and valid_to must be None (unending) or strictly greater than valid_from.

    WHY: Time spans in the simulation represent half-open intervals [valid_from, valid_to); inverted
    or negative intervals represent impossible temporal conditions.

    Args:
        span: A RoleSpan or DeviceControlSpan instance possessing valid_from and valid_to attributes.

    Returns:
        None.

    Example:
        validate_time_span(RoleSpan("acc_a", "mule", "g1", 1, 100, 200))
    """
    # 1. Assert starting minute is non-negative.
    if span.valid_from < 0:
        account_name = getattr(span, "account_id", "unknown")
        raise ValueError(f"Span for '{account_name}' has negative valid_from: {span.valid_from}")

    # 2. Assert ending minute is strictly greater than start minute when bounded.
    if span.valid_to is not None and span.valid_to <= span.valid_from:
        raise ValueError(
            f"Span for '{getattr(span, 'account_id', 'unknown')}' has valid_to ({span.valid_to}) "
            f"less than or equal to valid_from ({span.valid_from})"
        )


def validate_mule_depth(role_span: RoleSpan) -> None:
    """
    WHAT: Validates mule depth rules: mule accounts must have depth >= 1, while non-mule roles must have depth None.
    Raises ValueError naming the account and role if the invariant is violated.

    WHY: Mule depth measures distance in payment hops from the initial compromised victim. Non-mules
    (such as victims or collectors) do not occupy intermediary routing layers.

    Args:
        role_span: The RoleSpan instance to validate.

    Returns:
        None.

    Example:
        validate_mule_depth(RoleSpan("acc_a", "mule", "g1", 1, 0, 100))  # Passes
    """
    # 1. Validate that mule role assigns an integer depth >= 1.
    if role_span.role == "mule":
        if role_span.depth is None or role_span.depth < 1:
            raise ValueError(
                f"RoleSpan for mule account '{role_span.account_id}' requires depth >= 1, got {role_span.depth}"
            )
    else:
        # 2. Validate that non-mule roles strictly carry None as their depth attribute.
        if role_span.depth is not None:
            raise ValueError(
                f"RoleSpan for non-mule account '{role_span.account_id}' (role '{role_span.role}') "
                f"must have depth None, got {role_span.depth}"
            )


def validate_role_lifecycle(role_span: RoleSpan) -> None:
    """
    WHAT: Validates mule lifecycle attributes active_from and burned_from on a RoleSpan.
    Asserts both are None or both are set; if set, role must be 'mule' and valid_from <= active_from <= burned_from.

    WHY: Mules progress through defined lifecycle phases (warm-up, active, burned). Non-mules do not possess
    gang lifecycle states, and partial or out-of-order timestamps represent corrupted temporal progression.

    Args:
        role_span: The RoleSpan instance to validate.

    Returns:
        None. Raises ValueError with one plain sentence upon invariant violation.

    Example:
        validate_role_lifecycle(RoleSpan("acc_1", "mule", "g1", 1, 0, None, active_from=100, burned_from=200))
    """
    has_active = role_span.active_from is not None
    has_burned = role_span.burned_from is not None
    if has_active != has_burned:
        raise ValueError(
            f"RoleSpan for account '{role_span.account_id}' must have both active_from "
            f"and burned_from set or both None."
        )
    if has_active:
        if role_span.role != "mule":
            raise ValueError(
                f"RoleSpan for account '{role_span.account_id}' has lifecycle fields but role is '{role_span.role}'; "
                f"lifecycle is only allowed on mule roles."
            )
        assert role_span.active_from is not None and role_span.burned_from is not None
        if not (role_span.valid_from <= role_span.active_from <= role_span.burned_from):
            raise ValueError(
                f"RoleSpan for account '{role_span.account_id}' has invalid lifecycle order: "
                f"valid_from ({role_span.valid_from}) <= active_from ({role_span.active_from}) <= "
                f"burned_from ({role_span.burned_from}) was violated."
            )


def validate_no_overlapping_role_spans(spans: Sequence[RoleSpan]) -> None:
    """
    WHAT: Asserts that no two role spans for the same account overlap in time.
    Raises ValueError identifying the conflicting spans if an overlapping interval is detected.

    WHY: An account can only belong to one owner or role at a time; overlapping ownership spans
    indicate conflicting simulation definitions.

    Args:
        spans: Sequence of RoleSpan records to evaluate for pairwise overlap.

    Returns:
        None.

    Example:
        validate_no_overlapping_role_spans([RoleSpan("acc_a", "mule", "g1", 1, 0, 100), ...])
    """
    # 1. Compare every distinct pair of spans assigned to the same account.
    for i, first in enumerate(spans):
        for second in spans[i + 1:]:
            if first.account_id != second.account_id:
                continue
            # 2. Check half-open interval intersection [s1, e1) and [s2, e2).
            end_first = float("inf") if first.valid_to is None else first.valid_to
            end_second = float("inf") if second.valid_to is None else second.valid_to
            if first.valid_from < end_second and second.valid_from < end_first:
                raise ValueError(
                    f"Overlapping role spans for account '{first.account_id}': "
                    f"[{first.valid_from}, {first.valid_to}) and [{second.valid_from}, {second.valid_to})"
                )


def validate_no_overlapping_device_control_spans(spans: Sequence[DeviceControlSpan]) -> None:
    """
    WHAT: Asserts that no two device-control spans for the same device and account overlap in time.
    Raises ValueError identifying the conflicting spans if an overlap occurs.

    WHY: A specific device operating a specific account cannot simultaneously have multiple overlapping
    control records with contradictory compromise or ownership conditions.

    Args:
        spans: Sequence of DeviceControlSpan records to evaluate for pairwise overlap.

    Returns:
        None.

    Example:
        validate_no_overlapping_device_control_spans([DeviceControlSpan("dev_1", "acc_a", "g1", 0, 100, ...)])
    """
    # 1. Compare every distinct pair of spans assigned to the same device and account.
    for i, first in enumerate(spans):
        for second in spans[i + 1:]:
            if (first.device_id, first.account_id) != (second.device_id, second.account_id):
                continue
            # 2. Check half-open interval intersection [s1, e1) and [s2, e2).
            end_first = float("inf") if first.valid_to is None else first.valid_to
            end_second = float("inf") if second.valid_to is None else second.valid_to
            if first.valid_from < end_second and second.valid_from < end_first:
                raise ValueError(
                    f"Overlapping device control spans for device '{first.device_id}' and "
                    f"account '{first.account_id}': "
                    f"[{first.valid_from}, {first.valid_to}) and [{second.valid_from}, {second.valid_to})"
                )


def validate_payment_stories_length(payment_stories: Sequence[PaymentStory], payment_count: int) -> None:
    """
    WHAT: Asserts that payment_stories is either empty or exactly equal in length to the count of fraud payments.
    Raises ValueError naming both counts if a non-empty mismatch is detected.

    WHY: Each injected fraud payment must either possess a corresponding story or all stories must be omitted;
    partial story lists would leave fraud events unattributed.

    Args:
        payment_stories: Sequence of PaymentStory records provided in truth facts.
        payment_count: Total number of planned fraud payments in FraudInputs.payments.

    Returns:
        None.

    Example:
        validate_payment_stories_length([PaymentStory("g1", "route", 1, "test")], 1)  # Passes
    """
    # Assert stories collection is either completely empty or has exactly one item per fraud payment.
    if len(payment_stories) != 0 and len(payment_stories) != payment_count:
        raise ValueError(
            f"payment_stories has length {len(payment_stories)}; "
            f"must be empty or match fraud payments count ({payment_count})"
        )


def validate_truth_facts(
    truth: TruthFacts, payment_count: int, allowed_roles: Sequence[str] | None = None,
    allowed_origins: Sequence[str] | None = None,
) -> None:
    """
    WHAT: Top-level validator applying all hidden truth invariant checks across role spans, device control spans,
    and payment stories against the provided fraud payment count.

    WHY: Consolidates atomic validation rules into a single pre-run gate invoked before the interactive simulation
    loop processes initial queue items.

    Args:
        truth: The TruthFacts collection to inspect.
        payment_count: Total number of planned fraud payments in the current simulation run.
        allowed_roles: Optional explicit list of allowed role strings; loads from config if None.
        allowed_origins: Optional explicit list of allowed mule origin strings; loads from config if None.

    Returns:
        None.

    Example:
        validate_truth_facts(TruthFacts(), 0)  # Validates empty truth facts against zero payments
    """
    # 1. Resolve permitted role taxonomy from truth_store.yaml if not explicitly supplied.
    truth_config = load_truth_config()
    if allowed_roles is None:
        allowed_roles = list(truth_config["truth_store"]["roles"].keys())
    if allowed_origins is None:
        allowed_origins = list(truth_config["truth_store"]["mule_origins"])

    # 2. Validate individual role spans for schema compliance, non-negative bounds, and mule depth.
    for role_span in truth.roles:
        validate_role_in_config(role_span, allowed_roles)
        validate_role_origin(role_span, allowed_origins)
        validate_time_span(role_span)
        validate_mule_depth(role_span)
        validate_role_lifecycle(role_span)
    validate_no_overlapping_role_spans(truth.roles)

    # 3. Validate device control spans for temporal consistency and mutual exclusion.
    for device_span in truth.device_control:
        validate_time_span(device_span)
    validate_no_overlapping_device_control_spans(truth.device_control)

    # 4. Validate that transaction narratives strictly match the count of fraud payments.
    validate_payment_stories_length(truth.payment_stories, payment_count)
