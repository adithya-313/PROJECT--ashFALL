# File: api/v1/models.py
# Purpose: Request and response models for the /v1 bank endpoints, plus the policy numbers they are built from.
#
# Every model rejects fields it does not know (extra="forbid"). The request models deliberately have NO bank_id
# field: the bank always comes from authentication, so a bank can never name another bank's data.
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast, get_args

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field

from api.v1.bank_scope import load_known_banks
from api.v1.signing import DEFAULT_CONFIG_PATH
from core.config_validator import load_yaml

BankMode = Literal["shadow", "live"]
# The one list of allowed modes: the Literal above, so the config check and the response models cannot drift apart.
BANK_MODES: tuple[str, ...] = get_args(BankMode)
IDENTIFIER_PATTERN: str = r"^[A-Za-z0-9_.:@-]+$"


@dataclass(frozen=True)
class EndpointPolicy:
    """
    Policy numbers of the /v1 endpoints, read from config/bank_api.yaml.

    Fields:
        default_bank_mode: Mode every bank runs in until Level 14c makes it per bank ("shadow" or "live").
        max_amount_paise: Largest accepted payment amount in paise.
        max_payment_ids_per_complaint: Most payment ids one complaint may name.
        max_identifier_length: Longest accepted identifier (payment, account, event or complaint id).
    """

    default_bank_mode: BankMode
    max_amount_paise: int
    max_payment_ids_per_complaint: int
    max_identifier_length: int


def _read_policy_value(block: dict[str, Any], key: str) -> Any:
    """Returns the "value" of one policy entry; raises ValueError if the entry or its value is missing."""
    if key not in block:
        raise ValueError(f"Missing bank API policy setting: {key}")
    entry = block[key]
    if not isinstance(entry, dict) or "value" not in entry:
        raise ValueError(f"Bank API policy setting {key} must be a mapping with a 'value'")
    return entry["value"]


def _read_positive_int(block: dict[str, Any], key: str) -> int:
    """Returns a policy value that must be a positive integer (booleans are refused)."""
    value = _read_policy_value(block, key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Bank API policy setting {key} must be a positive integer, got {value!r}")
    return value


def load_endpoint_policy(config_path: str | Path = DEFAULT_CONFIG_PATH) -> EndpointPolicy:
    """
    WHAT: Reads and validates the four endpoint policy settings from config/bank_api.yaml.

    WHY: The limits that shape request validation live in config, not in code, and a wrong value must stop the
    service at start-up rather than let a bad request through later.

    Args:
        config_path: Path of the bank API YAML file.

    Returns:
        The validated EndpointPolicy.

    Raises:
        ValueError: If a setting is missing, is not a positive integer (or, for the mode, is not shadow or live).
    """
    data = load_yaml(config_path)
    block = data["bank_api"]
    mode = _read_policy_value(block, "default_bank_mode")
    if mode not in BANK_MODES:
        raise ValueError(f"Bank API policy setting default_bank_mode must be one of {BANK_MODES}, got {mode!r}")
    return EndpointPolicy(
        # The check above proved mode is one of the two allowed strings; cast only tells the type checker so.
        default_bank_mode=cast(BankMode, mode),
        max_amount_paise=_read_positive_int(block, "max_amount_paise"),
        max_payment_ids_per_complaint=_read_positive_int(block, "max_payment_ids_per_complaint"),
        max_identifier_length=_read_positive_int(block, "max_identifier_length"),
    )


# Read once when the module loads: the limits below are part of the models' schemas.
POLICY: EndpointPolicy = load_endpoint_policy()


def _require_known_bank(bank_id: str) -> str:
    """Accepts a bank id only if config/bank_api.yaml lists it; the error never repeats the given value."""
    if bank_id not in load_known_banks():
        raise ValueError("unknown bank id")
    return bank_id


Identifier = Annotated[str, Field(min_length=1, max_length=POLICY.max_identifier_length, pattern=IDENTIFIER_PATTERN)]
BankId = Annotated[Identifier, AfterValidator(_require_known_bank)]
# strict=True: only a real JSON integer is accepted, never the text "100" or a boolean.
AmountPaise = Annotated[int, Field(strict=True, ge=1, le=POLICY.max_amount_paise)]
AccountEventKind = Literal[
    "device_binding_change", "sim_swap", "app_reinstall", "os_update", "account_opened", "account_reactivated"
]
FeedbackOutcome = Literal["confirmed_fraud", "not_fraud", "customer_dispute"]


class ApiModel(BaseModel):
    """Base of every /v1 model: unknown fields are an error, so a typo or a smuggled field is never ignored."""

    model_config = ConfigDict(extra="forbid")


class DecisionRequest(ApiModel):
    """A bank asks for a decision on a payment it is about to make (always "unscored" until Level 19)."""

    payment_id: Identifier
    sender_account_id: Identifier
    receiver_bank_id: BankId
    receiver_vpa: Identifier | None = None
    amount_paise: AmountPaise
    event_time: AwareDatetime
    channel: Identifier | None = None
    payee_type: Identifier | None = None


class DecisionResponse(ApiModel):
    """The answer to a decision request."""

    payment_id: str
    decision: Literal["unscored"]
    mode: BankMode
    reason: str


class TransactionSideRequest(ApiModel):
    """One side (debit or credit) of a payment, as the bank booked it. There is no bank_id: it comes from the key."""

    payment_id: Identifier
    side: Literal["debit", "credit"]
    account_id: Identifier
    counterparty_vpa: Identifier | None = None
    counterparty_bank_id: BankId
    sender_bank_id: BankId
    receiver_bank_id: BankId
    amount_paise: AmountPaise
    event_time: AwareDatetime
    status: Literal["success", "failed"]
    failure_domain: Identifier | None = None
    payee_type: Identifier | None = None
    channel: Identifier | None = None


class TransactionResponse(ApiModel):
    """The result of recording one transaction side."""

    transaction_id: str
    event_id: str
    was_replay: bool


class AccountEventRequest(ApiModel):
    """A life-cycle event of one account: a device change, a SIM swap, an app reinstall and so on."""

    event_id: Identifier
    account_id: Identifier
    kind: AccountEventKind
    event_time: AwareDatetime
    details: dict[str, Any] = Field(default_factory=dict)


class AccountEventResponse(ApiModel):
    """The result of recording an account event; duplicate is true when the event id was already stored."""

    event_id: str
    duplicate: bool


class RiskResponse(ApiModel):
    """The risk of one account (always "unscored" until Level 19)."""

    account_id: str
    risk: Literal["unscored"]
    mode: BankMode


class ComplaintRequest(ApiModel):
    """A fraud complaint a bank received from its customer about one or more payments."""

    complaint_id: Identifier
    complainant_account: Identifier
    reported_account: Identifier
    reported_bank_id: BankId
    payment_ids: list[Identifier] = Field(min_length=1, max_length=POLICY.max_payment_ids_per_complaint)
    claimed_type: Identifier
    received_at: AwareDatetime


class ComplaintResponse(ApiModel):
    """The result of storing a complaint."""

    complaint_id: str


class FeedbackRequest(ApiModel):
    """The outcome a bank learned about a payment after the fact."""

    feedback_id: Identifier
    payment_id: Identifier
    outcome: FeedbackOutcome
    note: str | None = Field(default=None, max_length=1000)
    event_time: AwareDatetime


class FeedbackResponse(ApiModel):
    """The result of storing feedback."""

    feedback_id: str
