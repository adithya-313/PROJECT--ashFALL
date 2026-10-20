"""
WHAT: Evaluation truth loader providing secure, authenticated access to simulation ground-truth files.
Verifies cryptographic SHA-256 manifests, guards against calls from operational service packages, and exposes
temporal query methods for account roles, controlling gangs, mule depths, transaction stories, and compromised devices.

WHY: Benchmarking and evaluation algorithms require complete visibility into synthetic ground truth to measure
detection precision, recall, and time-to-detect. Isolating this interface under evaluation/ with caller-frame
permission checks and static import guards ensures banking models and APIs never access simulation labels.

TERM: hidden truth: ground-truth records describing illicit roles, device controls, and payment stories.
TERM: truth store: evaluation interface providing verified forensic simulation reality to benchmarks.
TERM: manifest: metadata file linking file names, row counts, format version, run id, and cryptographic digests.
TERM: half-open span: an interval [valid_from, valid_to) inclusive of start and exclusive of end.
TERM: mule depth: distance in payment hops from the initial compromised victim to an illicit account.
"""

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.run_identity import REPO_ROOT


class TruthStore:
    """
    WHAT: In-memory query interface for verified hidden truth artifacts of a completed simulation run.
    Provides methods to query temporal account roles, gang ownership, mule hierarchy depths, forensic
    transaction narratives, and controlling device identifiers at any specific simulation minute.

    WHY: Decouples downstream evaluation metrics from raw JSON Lines parsing and interval arithmetic.
    Standardizes "genuine" as the default state for any account outside explicit role spans.

    Args:
        roles: Sequence of role record dictionaries parsed from roles.jsonl.
        device_control: Sequence of device control dictionaries parsed from device_control.jsonl.
        payment_stories: Dictionary mapping payment_id strings to payment story records.

    Returns:
        An initialized TruthStore instance.

    Example:
        # store = TruthStore(roles=[...], device_control=[...], payment_stories={...})
        # store.role_at("acc_person_000001", 100) == "genuine"
    """

    def __init__(
        self,
        roles: list[dict[str, Any]],
        device_control: list[dict[str, Any]],
        payment_stories: dict[str, dict[str, Any]],
    ) -> None:
        """
        WHAT: Stores parsed truth records and payment story lookup maps.

        WHY: Enables constant-time transaction story retrieval and linear temporal span scans.

        Args:
            roles: List of parsed role span dictionaries.
            device_control: List of parsed device control dictionaries.
            payment_stories: Mapping from payment_id to forensic story record dictionary.

        Returns:
            None.

        Example:
            # store = TruthStore(roles=[], device_control=[], payment_stories={})
        """
        # Store verified truth collections.
        self._roles = roles
        self._device_control = device_control
        self._payment_stories = payment_stories

    def role_at(self, account_id: str, minute: int) -> str:
        """
        WHAT: Returns the account's operational role at the given simulation minute.
        If the account is not covered by an active role span, returns 'genuine'.

        WHY: Ground rule 15 specifies that genuine status is never stored explicitly; accounts are
        genuine by default outside active criminal takeover or mule recruitment periods.

        Args:
            account_id: Account identifier to inspect.
            minute: Integer simulation minute (>= 0).

        Returns:
            String role name ('mule', 'collector', 'cash_out', 'victim', or 'genuine').

        Example:
            # store.role_at('acc_person_000001', 0) -> 'genuine'
        """
        # 1. Scan role spans for an active half-open interval match.
        for role_span in self._roles:
            if role_span["account_id"] == account_id:
                valid_from = role_span["valid_from"]
                valid_to = role_span["valid_to"]
                if valid_from <= minute and (valid_to is None or minute < valid_to):
                    return str(role_span["role"])

        # 2. Return default genuine status when outside all active role spans.
        return "genuine"

    def owner_at(self, account_id: str, minute: int) -> str | None:
        """
        WHAT: Returns the gang_id controlling the specified account at the given simulation minute,
        or None if the account is genuine or unoccupied at that time.

        WHY: Enables evaluation algorithms to group compromised accounts by illicit syndicate ownership.

        Args:
            account_id: Account identifier to query.
            minute: Integer simulation minute.

        Returns:
            String gang identifier or None.

        Example:
            # store.owner_at('acc_mule_001', 1440) -> 'gang_alpha'
        """
        # 1. Find active role span matching account and temporal minute.
        for role_span in self._roles:
            if role_span["account_id"] == account_id:
                valid_from = role_span["valid_from"]
                valid_to = role_span["valid_to"]
                if valid_from <= minute and (valid_to is None or minute < valid_to):
                    return role_span["gang_id"]

        # 2. Return None if no gang owns the account at this minute.
        return None

    def origin_at(self, account_id: str, minute: int) -> str | None:
        """
        WHAT: Returns how the gang came to control the account (rented, borrowed_kyc or business) if the account
        is an active mule at the given simulation minute and its origin was recorded, otherwise None.

        WHY: Lets evaluation compare detection across mule origins without reading raw truth rows.

        Args:
            account_id: Account identifier to query.
            minute: Integer simulation minute.

        Returns:
            The origin string or None.

        Example:
            # store.origin_at('acc_person_000006', 1440) -> 'rented'
        """
        # 1. Find the active role span and return its recorded origin.
        for role_span in self._roles:
            if role_span["account_id"] == account_id:
                valid_from = role_span["valid_from"]
                valid_to = role_span["valid_to"]
                if valid_from <= minute and (valid_to is None or minute < valid_to):
                    return role_span["origin"]

        # 2. Return None if the account has no active role span.
        return None

    def depth_at(self, account_id: str, minute: int) -> int | None:
        """
        WHAT: Returns the hierarchical mule depth layer for the account at the given simulation minute,
        or None if the account is not an active mule at that time.

        WHY: Enables laundering chain depth analysis (layer 1 = first mule hop after victim).

        Args:
            account_id: Account identifier to query.
            minute: Integer simulation minute.

        Returns:
            Positive integer depth layer or None.

        Example:
            # store.depth_at('acc_mule_001', 1440) -> 1
        """
        # 1. Find active role span and return its recorded mule depth.
        for role_span in self._roles:
            if role_span["account_id"] == account_id:
                valid_from = role_span["valid_from"]
                valid_to = role_span["valid_to"]
                if valid_from <= minute and (valid_to is None or minute < valid_to):
                    return role_span["depth"]

        # 2. Return None if the account has no active role span.
        return None

    def story_of(self, payment_id: str) -> dict[str, Any]:
        """
        WHAT: Retrieves the forensic narrative and tactical attribution dictionary for a payment event.
        Raises KeyError if the payment_id does not exist in the truth store.

        WHY: Links individual transaction events to their simulated fraud narrative and gang campaign.

        Args:
            payment_id: String event identifier (e.g., '66db9aa439b092ee-000000').

        Returns:
            Dictionary containing payment story attributes (gang_id, block, layer, description, origin).

        Raises:
            KeyError: If payment_id is not present in the payment stories collection.

        Example:
            # story = store.story_of('run-000001')
            # story['gang_id'] == 'gang_alpha'
        """
        # 1. Look up payment story by unique event payment_id.
        if payment_id not in self._payment_stories:
            raise KeyError(f"No payment story found for payment_id '{payment_id}'.")
        return self._payment_stories[payment_id]

    def controllers_of(self, account_id: str, minute: int) -> list[str]:
        """
        WHAT: Returns a list of device identifiers actively controlling the specified account at minute.

        WHY: Enables device takeover and hardware credential compromise tracking for fraud evaluation.

        Args:
            account_id: Account identifier to query.
            minute: Integer simulation minute.

        Returns:
            List of device ID strings actively mapped to the account.

        Example:
            # store.controllers_of('acc_victim_01', 500) -> ['dev_00042']
        """
        # 1. Identify all device control spans matching the account active at the given minute.
        active_devices: list[str] = []
        for device_span in self._device_control:
            if device_span["account_id"] == account_id:
                valid_from = device_span["valid_from"]
                valid_to = device_span["valid_to"]
                if valid_from <= minute and (valid_to is None or minute < valid_to):
                    active_devices.append(str(device_span["device_id"]))

        # 2. Return ordered list of active controlling device IDs.
        return active_devices


def _verify_direct_caller() -> None:
    """
    WHAT: Inspects the immediate caller frame to verify it does not belong to an operational package.
    Raises PermissionError if the caller's top-level module name is listed in config ashfall_packages.

    WHY: Defense-in-depth runtime check. While static AST analysis (core/truth_import_check.py) provides
    comprehensive repository-wide guards, this runtime check prevents immediate unauthorized imports.
    Note: this check only inspects the direct caller frame; static linting is the primary enforcement guard.

    Args:
        None.

    Returns:
        None.

    Raises:
        PermissionError: If the direct caller's top-level package is an operational ashFALL package.

    Example:
        # Calling from an 'api' module raises PermissionError.
        # _verify_direct_caller()
    """
    # 1. Walk two frames back to identify the external caller of load_truth.
    current_frame = inspect.currentframe()
    if current_frame is None or current_frame.f_back is None or current_frame.f_back.f_back is None:
        return

    caller_frame = current_frame.f_back.f_back
    caller_module = caller_frame.f_globals.get("__name__", "")
    top_level_package = caller_module.split(".")[0]

    # 2. Compare against prohibited operational packages declared in truth_store policy.
    truth_config = load_yaml(REPO_ROOT / "config" / "truth_store.yaml")["truth_store"]
    prohibited_packages = set(truth_config.get("ashfall_packages", []))
    _allowed_folders = set(truth_config.get("allowed_reader_folders", []))
    if top_level_package in prohibited_packages:
        raise PermissionError(
            f"Operational package '{top_level_package}' is forbidden from loading hidden truth; "
            f"truth access is restricted to evaluation and test frameworks."
        )


def _verify_file_integrity(file_path: Path, filename: str, expected_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """
    WHAT: Verifies file existence, recalculates SHA-256 digest, asserts row count equality, and parses JSON lines.
    Raises ValueError if either the cryptographic digest or row count deviates from manifest metadata.

    WHY: Guarantees tamper detection before ground truth enters evaluation frameworks.

    Args:
        file_path: Filesystem Path pointing to the JSON Lines file to verify.
        filename: Base string file name for diagnostic error messages.
        expected_meta: Dictionary containing 'sha256' string and 'row_count' integer.

    Returns:
        List of parsed JSON dictionary records from the verified file.

    Raises:
        FileNotFoundError: If file_path does not exist on disk.
        ValueError: If calculated SHA-256 or row count does not match expected values.

    Example:
        # rows = _verify_file_integrity(Path('runs/roles.jsonl'), 'roles.jsonl', {'sha256': '...', 'row_count': 1})
        # len(rows) == 1
    """
    # 1. Assert file exists on disk.
    if not file_path.exists():
        raise FileNotFoundError(f"Truth file '{filename}' listed in manifest does not exist at '{file_path}'.")

    # 2. Verify SHA-256 digest.
    file_bytes = file_path.read_bytes()
    actual_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if actual_sha256 != expected_meta["sha256"]:
        raise ValueError(
            f"Cryptographic hash mismatch for '{filename}': expected {expected_meta['sha256']}, got {actual_sha256}."
        )

    # 3. Parse lines and verify row count.
    lines = file_bytes.decode("utf-8").splitlines()
    if len(lines) != expected_meta["row_count"]:
        raise ValueError(
            f"Row count mismatch for '{filename}': expected {expected_meta['row_count']}, got {len(lines)}."
        )

    return [json.loads(line) for line in lines]


def load_truth(truth_root: Path, run_id: str) -> TruthStore:
    """
    WHAT: Loads, validates, and initializes a TruthStore from run artifacts under truth_root / run_id.
    Verifies that the caller is permitted, checks all file digests and row counts against manifest.json,
    and returns a queryable TruthStore container. Refuses tampered files or unpermitted callers.

    WHY: Ground rule 15 and evaluation design require ground truth to be cryptographically verified
    before benchmark ingestion, ensuring evaluation metrics cannot be computed on tampered truth.

    Args:
        truth_root: Filesystem Path pointing to the truth storage root directory.
        run_id: String run identifier designating the target simulation folder.

    Returns:
        Verified TruthStore instance ready for evaluation queries.

    Raises:
        PermissionError: If direct caller resides in an operational package.
        FileNotFoundError: If manifest.json or required artifact files are missing.
        ValueError: If manifest digests or row counts fail verification.

    Example:
        # store = load_truth(Path("runs/truth"), "66db9aa439b092ee")
        # isinstance(store, TruthStore) is True
    """
    # 1. Verify that direct calling package is permitted to access ground truth.
    _verify_direct_caller()

    # 2. Locate and parse cryptographic manifest.
    run_dir = truth_root / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file missing at '{manifest_path}'.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 3. Verify format version and each truth artifact against manifest digests and row counts.
    truth_config = load_yaml(REPO_ROOT / "config" / "truth_store.yaml")["truth_store"]
    expected_format = truth_config.get("format", "truth-v1")
    if manifest.get("format") != expected_format:
        raise ValueError(
            f"Truth format version mismatch: expected '{expected_format}', got '{manifest.get('format')}'."
        )
    files_meta = manifest.get("files", {})
    file_names = truth_config["file_names"]

    roles_name = file_names["roles"]
    roles_rows = _verify_file_integrity(run_dir / roles_name, roles_name, files_meta[roles_name])
    device_name = file_names["device_control"]
    device_rows = _verify_file_integrity(run_dir / device_name, device_name, files_meta[device_name])
    story_name = file_names["payment_stories"]
    story_rows = _verify_file_integrity(run_dir / story_name, story_name, files_meta[story_name])

    # 4. Index stories by payment_id and construct TruthStore.
    stories_map = {story["payment_id"]: story for story in story_rows}
    return TruthStore(roles=roles_rows, device_control=device_rows, payment_stories=stories_map)
