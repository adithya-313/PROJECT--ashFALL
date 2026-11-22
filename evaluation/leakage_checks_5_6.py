"""
WHAT: Section 23 leakage checks 5 and 6 for the Level 18 account model, built from the repository's
own sources at run time.

WHY: Check 5 asks whether a simulator-only field, a value the world computes that no bank feed
carries, reaches the account model's feature code or its training data. Check 6 asks whether an
account's own identity, its id or a value parsed from it, is used as a feature. Both are decided on
the committed code and the four training datasets; no test data is read and no model choice changes.

TERM: simulator-only field: a key produced by the events-v1 record in world/fast_mode.py or a
column of a real truth file under runs/full/truth, minus every field a bank feed may carry
(world/bank_feed.py's BANK_VISIBLE_* and *_DETAIL_FIELDS frozensets). The set is computed here,
never typed.

TERM: identifier derivation: a value computed from account_id, bank_id, world_id or run_id by
wrapping it (len, int, str, ...), transforming its text (split, rsplit, strip, ...), slicing it,
doing arithmetic on it, placing it in an f-string, or storing it as a dict value or a return value.
Passing an identifier to a lookup (a function argument, a dict key, a subscript index) selects rows
and is not a derivation, so the registry hash and every bank-scoped read stay out of the report.
"""

import argparse
import ast
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score

import world.bank_feed as bank_feed
from decisions.anomaly_config import load_anomaly_detector_config
from decisions.mule_account_model import account_model_feature_names, train_account_model
from evaluation.level17_scoring import MULE_ROLES
from world.clock import WORLD_START_DATE, SimClock
from world.device_runtime import DeviceStamp
from world.failures import PaymentEvent
from world.fast_mode import event_to_record

__all__ = [
    "AccountRow",
    "FORBIDDEN_IDENTIFIER_NAMES",
    "ID_FEATURE_NAMES",
    "StaticHit",
    "bank_visible_fields",
    "build_leakage_report",
    "check_training_dataset_columns",
    "event_record_fields",
    "id_leak_passes",
    "load_account_rows",
    "load_leakage_checks_config",
    "main",
    "measure_id_only_signal",
    "scan_identifier_derivations",
    "scan_simulator_field_usage",
    "simulator_only_fields",
    "truth_file_columns",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "leakage_checks.yaml"

FEATURE_SOURCE_PATHS: tuple[str, ...] = (
    "features/account_summaries.py",
    "features/account_snapshots.py",
    "features/feature_log.py",
    "decisions/mule_account_model.py",
)

SIMULATOR_GATE_NAMES: tuple[str, str] = ("purpose", "sender_is_new_device")

FORBIDDEN_IDENTIFIER_NAMES: frozenset[str] = frozenset({"account_id", "bank_id", "world_id", "run_id"})

VALUE_WRAPPER_FUNCTIONS: frozenset[str] = frozenset({"len", "int", "float", "str", "hash", "ord", "bytes"})

STRING_TRANSFORM_METHODS: frozenset[str] = frozenset(
    {
        "split",
        "rsplit",
        "partition",
        "rpartition",
        "strip",
        "lstrip",
        "rstrip",
        "lower",
        "upper",
        "casefold",
        "replace",
        "removeprefix",
        "removesuffix",
        "format",
        "encode",
        "decode",
        "join",
        "zfill",
    }
)

ARITHMETIC_OPERATORS: tuple[type[ast.operator], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
)

ID_FEATURE_NAMES: tuple[str, ...] = (
    "id_length",
    "numeric_suffix",
    "numeric_suffix_rank",
    "prefix_code",
    "bank_code",
)


@dataclass(frozen=True)
class StaticHit:
    """
    WHAT: One place where a scanned source file uses a name the check forbids.

    ATTRIBUTES:
        path: File path as recorded in the report.
        line: One-based line number.
        column: Zero-based column offset.
        name: The forbidden name that was found.
        kind: How the name was used, for example "string literal" or "wrapped in len()".
    """

    path: str
    line: int
    column: int
    name: str
    kind: str


@dataclass(frozen=True)
class AccountRow:
    """
    WHAT: One account's identity, the world it lives in and the run whose truth holds its roles.

    ATTRIBUTES:
        world_id: Dataset world identifier.
        bank_id: Bank that holds the account.
        account_id: Account identifier.
        run_id: Simulation run identifier used to find the truth directory.
    """

    world_id: str
    bank_id: str
    account_id: str
    run_id: str


def _require_nonempty_string(record: dict[str, object], key: str, caller: str) -> str:
    """
    WHAT: Returns a required non-empty string field from a record.

    Raises:
        ValueError: If the field is missing, empty, or not a string.
    """
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{caller} requires a non-empty string '{key}' on every record")
    return value


def _display_path(path: Path) -> str:
    """
    WHAT: Returns a repository-relative path when the file is inside the repository, else its string
    form, so test files outside the tree still get a readable name.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def event_record_fields() -> set[str]:
    """
    WHAT: Returns the keys of the events-v1 record produced by world/fast_mode.py's event_to_record.

    WHY: The world record is the simulator's own output, so its keys are the candidate
    simulator-only fields; calling the real function keeps the list current with the code.

    Returns:
        The 14 event record keys.
    """
    clock = SimClock(WORLD_START_DATE)
    event = PaymentEvent(60, "acc_1", "acc_2", 5000, 0, 1, "success", "ok", None)
    stamp = DeviceStamp("fp_abc", 3, False)
    return set(event_to_record(event, "everyday", stamp, clock))


def truth_file_columns(path: Path) -> set[str]:
    """
    WHAT: Returns the union of keys across the records of one real truth JSONL file.

    WHY: A truth file's schema is its record keys; reading a real file keeps the column list
    current with the simulator instead of a typed copy.

    Args:
        path: A truth JSONL file, for example runs/full/truth/<run_id>/roles.jsonl.

    Returns:
        Every key that appears on any record.

    Raises:
        ValueError: If the file is missing, has no records, or a record is not a JSON object.
    """
    if not path.is_file():
        raise ValueError(f"Truth file not found at {path}")
    columns: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Truth file {path} holds a record that is not a JSON object")
            columns.update(record)
    if not columns:
        raise ValueError(f"Truth file {path} holds no records, so its schema is unknown")
    return columns


def bank_visible_fields() -> frozenset[str]:
    """
    WHAT: Returns the union of every bank-visible allow-list in world/bank_feed.py.

    WHY: Only the bank feed crosses to ashFALL, so a field a bank feed may carry is never
    simulator-only. The allow-lists are imported, not copied, so the check follows the code.

    Returns:
        The union of BANK_VISIBLE_PAYMENT_FIELDS, ACCOUNT_OPENED_DETAIL_FIELDS,
        BANK_VISIBLE_WITHDRAWAL_FIELDS, BANK_VISIBLE_ACCOUNT_EVENT_FIELDS,
        BANK_VISIBLE_COMPLAINT_FIELDS, GENUINE_DEVICE_DETAIL_FIELDS and GANG_DEVICE_DETAIL_FIELDS.
    """
    allow_lists = (
        bank_feed.BANK_VISIBLE_PAYMENT_FIELDS,
        bank_feed.ACCOUNT_OPENED_DETAIL_FIELDS,
        bank_feed.BANK_VISIBLE_WITHDRAWAL_FIELDS,
        bank_feed.BANK_VISIBLE_ACCOUNT_EVENT_FIELDS,
        bank_feed.BANK_VISIBLE_COMPLAINT_FIELDS,
        bank_feed.GENUINE_DEVICE_DETAIL_FIELDS,
        bank_feed.GANG_DEVICE_DETAIL_FIELDS,
    )
    return frozenset().union(*allow_lists)


def simulator_only_fields(world_fields: set[str], bank_visible: frozenset[str]) -> frozenset[str]:
    """
    WHAT: Returns the world and truth fields that no bank feed may carry.

    WHY: This is the set check 5 protects: a feature built from any of these values would use
    information the bank never receives.

    Args:
        world_fields: Keys from the events-v1 record plus the truth file schema.
        bank_visible: Union of the bank-visible allow-lists.

    Returns:
        world_fields - bank_visible, as a frozenset.

    Raises:
        ValueError: If "purpose" or "sender_is_new_device" is not in the result, because the
            derived set does not describe the simulator the check was written for.
    """
    simulator_only = frozenset(world_fields) - bank_visible
    missing = [name for name in SIMULATOR_GATE_NAMES if name not in simulator_only]
    if missing:
        raise ValueError(
            f"Simulator-only set is missing {missing}; refusing to run a check built on the wrong field list"
        )
    return simulator_only


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """
    WHAT: Maps id(child node) to its parent node for one parsed module.

    WHY: The scans decide how a name is used from its parent (a call wrapper, a subscript, an
    f-string, a raise), and ast has no parent pointers of its own.
    """
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _read_ast(path: Path) -> ast.AST:
    """
    WHAT: Parses one Python source file.

    Raises:
        ValueError: If the file cannot be read or parsed, because a scan that skipped a file
            would report a clean result it did not prove.
    """
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise ValueError(f"Cannot parse {path}: {error}") from error


def _hit_order(hit: StaticHit) -> tuple[str, int, int]:
    """Returns the sort key that keeps hits in file and line order."""
    return (hit.path, hit.line, hit.column)


def scan_simulator_field_usage(
    files: Sequence[Path],
    simulator_only: frozenset[str],
) -> tuple[list[StaticHit], list[StaticHit]]:
    """
    WHAT: Finds every use of a simulator-only field name as a string literal or an attribute read.

    WHY: A feature that writes "purpose" or reads record.purpose is using a simulator-only field.
    A method invocation such as datetime.timestamp() is not a field read, so it is returned
    separately as an excluded match rather than silently dropped.

    Args:
        files: Python source files to scan.
        simulator_only: Field names check 5 protects.

    Returns:
        (hits, excluded_method_calls), each sorted by path and line.

    Raises:
        ValueError: If a file cannot be parsed.
    """
    hits: list[StaticHit] = []
    excluded: list[StaticHit] = []
    for path in files:
        tree = _read_ast(path)
        parents = _parent_map(tree)
        display = _display_path(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in simulator_only:
                hits.append(StaticHit(display, node.lineno, node.col_offset, node.value, "string literal"))
            elif isinstance(node, ast.Attribute) and node.attr in simulator_only:
                parent = parents.get(id(node))
                if isinstance(parent, ast.Call) and parent.func is node:
                    excluded.append(StaticHit(display, node.lineno, node.col_offset, node.attr, "method call"))
                else:
                    hits.append(StaticHit(display, node.lineno, node.col_offset, node.attr, "attribute access"))
    hits.sort(key=_hit_order)
    excluded.sort(key=_hit_order)
    return hits, excluded


def _inside_raise(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """
    WHAT: Returns True when a node sits inside a raise statement.

    WHY: An error message may name an identifier; it is never a feature value, so an f-string in a
    raise is excluded from check 6's hits.
    """
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.Raise):
            return True
        current = parents.get(id(current))
    return False


def _identifier_use_kind(node: ast.Name, parents: dict[int, ast.AST]) -> str | None:
    """
    WHAT: Returns how an identifier name is used when that use can produce a value, else None.

    WHY: Selecting rows by an identifier is legitimate (a lookup key); deriving a value from it is
    what check 6 forbids. Keeping the decision in one function makes the rule explicit.

    Args:
        node: A Name node whose id is one of FORBIDDEN_IDENTIFIER_NAMES.
        parents: Parent map for the module.

    Returns:
        A short kind string for a derivation, "excluded: ..." for a reported exclusion, or None
        when the name only selects rows.
    """
    parent = parents.get(id(node))
    if parent is None:
        return None
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id in VALUE_WRAPPER_FUNCTIONS
        and any(argument is node for argument in parent.args)
    ):
        return f"wrapped in {parent.func.id}()"
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr in STRING_TRANSFORM_METHODS:
        owner = parents.get(id(parent))
        if isinstance(owner, ast.Call) and owner.func is parent:
            return f"string method {parent.attr}()"
        return f"attribute {parent.attr}"
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return "subscript of identifier"
    if (
        isinstance(parent, ast.BinOp)
        and (parent.left is node or parent.right is node)
        and isinstance(parent.op, ARITHMETIC_OPERATORS)
    ):
        return "arithmetic on identifier"
    if isinstance(parent, ast.Dict) and any(value is node for value in parent.values):
        return "identifier as dict value"
    if isinstance(parent, ast.DictComp) and parent.value is node:
        return "identifier as dict-comprehension value"
    if isinstance(parent, ast.Return) and parent.value is node:
        return "returned identifier"
    if isinstance(parent, (ast.FormattedValue, ast.JoinedStr)):
        if _inside_raise(node, parents):
            return "excluded: f-string in raise"
        return "identifier in f-string"
    return None


def scan_identifier_derivations(
    files: Sequence[Path],
    identifiers: frozenset[str],
) -> tuple[list[StaticHit], list[StaticHit]]:
    """
    WHAT: Finds every value-producing use of an account identity name in the scanned source files.

    WHY: Check 6 must fail when a feature is derived from the account's own id, and must not fail
    when the id only selects the account's rows.

    Args:
        files: Python source files to scan.
        identifiers: Identity names to inspect, normally FORBIDDEN_IDENTIFIER_NAMES.

    Returns:
        (hits, excluded_contexts), each sorted by path and line. A hit is a derivation; an excluded
        context is a use that names the identifier inside a raise message.

    Raises:
        ValueError: If a file cannot be parsed.
    """
    hits: list[StaticHit] = []
    excluded: list[StaticHit] = []
    for path in files:
        tree = _read_ast(path)
        parents = _parent_map(tree)
        display = _display_path(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or node.id not in identifiers:
                continue
            kind = _identifier_use_kind(node, parents)
            if kind is None:
                continue
            hit = StaticHit(display, node.lineno, node.col_offset, node.id, kind)
            if kind.startswith("excluded:"):
                excluded.append(hit)
            else:
                hits.append(hit)
    hits.sort(key=_hit_order)
    excluded.sort(key=_hit_order)
    return hits, excluded


def check_training_dataset_columns(
    training_dirs: Sequence[Path],
    simulator_only: frozenset[str],
    model_features: Sequence[str],
) -> dict[str, object]:
    """
    WHAT: Proves no training snapshot or payment table carries a simulator-only field and that every
    account-model feature is a snapshot column.

    WHY: Even with clean feature code, a dataset that carried a simulator-only column would let a
    later feature reach it; and a feature the snapshot table lacks would silently become all-null.

    Args:
        training_dirs: The four training dataset folders.
        simulator_only: Field names check 5 protects.
        model_features: account_model_feature_names() at the committed configuration.

    Returns:
        Per-dataset column counts and violations, plus split-wide totals and a status.

    Raises:
        ValueError: If a dataset file is missing or holds no columns.
    """
    datasets: list[dict[str, Any]] = []
    for dataset_dir in training_dirs:
        snapshot_path = dataset_dir / "snapshots.parquet"
        payment_path = dataset_dir / "payments.parquet"
        if not snapshot_path.is_file() or not payment_path.is_file():
            raise ValueError(f"Training dataset {dataset_dir} is missing snapshots.parquet or payments.parquet")
        snapshot_columns = set(pq.read_schema(snapshot_path).names)
        payment_columns = set(pq.read_schema(payment_path).names)
        if not snapshot_columns or not payment_columns:
            raise ValueError(f"Training dataset {dataset_dir} holds an empty table schema")
        snapshot_violations = sorted(snapshot_columns & simulator_only)
        payment_violations = sorted(payment_columns & simulator_only)
        missing_features = sorted(name for name in model_features if name not in snapshot_columns)
        datasets.append(
            {
                "dataset": dataset_dir.name,
                "snapshot_column_count": len(snapshot_columns),
                "payment_column_count": len(payment_columns),
                "simulator_only_snapshot_columns": snapshot_violations,
                "simulator_only_payment_columns": payment_violations,
                "missing_model_features": missing_features,
            }
        )
    violation_total = sum(
        len(dataset["simulator_only_snapshot_columns"]) + len(dataset["simulator_only_payment_columns"])
        for dataset in datasets
    )
    missing_total = sum(len(dataset["missing_model_features"]) for dataset in datasets)
    return {
        "status": "pass" if violation_total == 0 and missing_total == 0 else "fail",
        "dataset_count": len(datasets),
        "model_feature_count": len(model_features),
        "simulator_only_violation_count": violation_total,
        "missing_model_feature_count": missing_total,
        "datasets": datasets,
    }


def load_account_rows(dataset_dirs: Sequence[Path]) -> list[AccountRow]:
    """
    WHAT: Returns one AccountRow per distinct (world, bank, account) in the given datasets.

    WHY: Check 6b learns from accounts, not account-days, so the snapshot table is reduced to its
    distinct identities and the world's run id is attached for truth lookup.

    Args:
        dataset_dirs: Dataset folders holding snapshots.parquet and manifest.json.

    Returns:
        Account rows sorted by world, bank and account.

    Raises:
        ValueError: If a manifest is missing, a world has no run id, or a snapshot names a world
            the manifest does not list.
    """
    rows: list[AccountRow] = []
    for dataset_dir in dataset_dirs:
        manifest_path = dataset_dir / "manifest.json"
        snapshot_path = dataset_dir / "snapshots.parquet"
        if not manifest_path.is_file() or not snapshot_path.is_file():
            raise ValueError(f"Dataset {dataset_dir} is missing manifest.json or snapshots.parquet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        worlds = manifest.get("worlds")
        if not isinstance(worlds, list) or not worlds:
            raise ValueError(f"Dataset {dataset_dir} manifest holds no worlds")
        run_by_world: dict[str, str] = {}
        for world in worlds:
            if not isinstance(world, dict):
                raise ValueError(f"Dataset {dataset_dir} manifest holds a world that is not an object")
            world_id = _require_nonempty_string(world, "world_id", "load_account_rows")
            run_by_world[world_id] = _require_nonempty_string(world, "run_id", "load_account_rows")
        table = pq.read_table(snapshot_path, columns=["world_id", "bank_id", "account_id"])
        unique = table.group_by(["world_id", "bank_id", "account_id"]).aggregate([])
        world_ids = unique.column("world_id").to_pylist()
        bank_ids = unique.column("bank_id").to_pylist()
        account_ids = unique.column("account_id").to_pylist()
        for world_id, bank_id, account_id in zip(world_ids, bank_ids, account_ids, strict=True):
            if world_id not in run_by_world:
                raise ValueError(f"Snapshot world {world_id!r} is not listed in {manifest_path}")
            rows.append(
                AccountRow(
                    world_id=_require_nonempty_string({"world_id": world_id}, "world_id", "load_account_rows"),
                    bank_id=_require_nonempty_string({"bank_id": bank_id}, "bank_id", "load_account_rows"),
                    account_id=_require_nonempty_string({"account_id": account_id}, "account_id", "load_account_rows"),
                    run_id=run_by_world[world_id],
                )
            )
    rows.sort(key=lambda row: (row.world_id, row.bank_id, row.account_id))
    return rows


def _world_mule_accounts(truth_root: Path, run_id: str) -> frozenset[str]:
    """
    WHAT: Returns the accounts whose truth role is a mule role in one run.

    WHY: Check 6b predicts the hidden role, so the label is exactly the role set the evaluation
    package treats as mules: mule, collector and cash_out.

    Args:
        truth_root: Folder holding one directory per run id.
        run_id: Run whose roles.jsonl is read.

    Returns:
        Account ids with at least one mule-role span.

    Raises:
        ValueError: If the roles file is missing or a record lacks account_id or role.
    """
    path = truth_root / run_id / "roles.jsonl"
    if not path.is_file():
        raise ValueError(f"Truth roles file not found at {path}")
    mules: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Truth roles file {path} holds a record that is not a JSON object")
            account_id = _require_nonempty_string(record, "account_id", "_world_mule_accounts")
            role = _require_nonempty_string(record, "role", "_world_mule_accounts")
            if role in MULE_ROLES:
                mules.add(account_id)
    return frozenset(mules)


def _account_targets(rows: Sequence[AccountRow], truth_root: Path) -> list[int]:
    """
    WHAT: Returns 1 for a mule-role account and 0 otherwise, aligned with the given rows.

    WHY: The id-only model must predict the hidden role per account, so every account row needs a
    target even when the account appears in more than one bank of the same world.
    """
    mules_by_run: dict[str, frozenset[str]] = {}
    targets: list[int] = []
    for row in rows:
        if row.run_id not in mules_by_run:
            mules_by_run[row.run_id] = _world_mule_accounts(truth_root, row.run_id)
        targets.append(1 if row.account_id in mules_by_run[row.run_id] else 0)
    return targets


def _split_account_id(account_id: str) -> tuple[str, int | None]:
    """
    WHAT: Splits an account id into its non-digit prefix and its trailing digits as a number.

    WHY: Prefix and numeric suffix are the two readable pieces of an id, so they are the id-only
    features check 6b tests.

    Returns:
        (prefix, number); number is None when the id does not end in a digit.
    """
    index = len(account_id)
    while index > 0 and "0" <= account_id[index - 1] <= "9":
        index -= 1
    prefix = account_id[:index]
    digits = account_id[index:]
    return prefix, int(digits) if digits else None


def _fit_category_codes(rows: Sequence[AccountRow]) -> tuple[dict[str, int], dict[str, int]]:
    """
    WHAT: Assigns each account prefix and bank a stable integer code from the training rows.

    WHY: XGBoost needs numbers, and fitting the codes on training only keeps a validation category
    from influencing the encoder.

    Returns:
        (prefix codes, bank codes), each mapping value to its position in sorted order.
    """
    prefixes = sorted({_split_account_id(row.account_id)[0] for row in rows})
    banks = sorted({row.bank_id for row in rows})
    return ({prefix: index for index, prefix in enumerate(prefixes)}, {bank: index for index, bank in enumerate(banks)})


def _id_only_matrix(
    rows: Sequence[AccountRow],
    prefix_codes: dict[str, int],
    bank_codes: dict[str, int],
) -> numpy.ndarray:
    """
    WHAT: Builds the id-only feature matrix in ID_FEATURE_NAMES order.

    WHY: The matrix must contain only what an id exposes: its length, its numeric suffix, that
    suffix's rank inside its world and bank, its prefix and its bank. A category absent from the
    training fit stays NaN so XGBoost treats it as missing rather than inventing a code.

    Args:
        rows: Account rows to encode.
        prefix_codes: Prefix-to-code map fitted on training rows.
        bank_codes: Bank-to-code map fitted on training rows.

    Returns:
        float32 matrix with one row per account row.
    """
    parsed = [_split_account_id(row.account_id) for row in rows]
    group_suffixes: dict[tuple[str, str], set[int]] = {}
    for row, (_, suffix) in zip(rows, parsed, strict=True):
        if suffix is not None:
            group_suffixes.setdefault((row.world_id, row.bank_id), set()).add(suffix)
    ranks = {
        group: {suffix: rank for rank, suffix in enumerate(sorted(suffixes))}
        for group, suffixes in group_suffixes.items()
    }
    matrix = numpy.full((len(rows), len(ID_FEATURE_NAMES)), numpy.nan, dtype=numpy.float32)
    for index, (row, (prefix, suffix)) in enumerate(zip(rows, parsed, strict=True)):
        matrix[index, 0] = len(row.account_id)
        if suffix is not None:
            matrix[index, 1] = suffix
            matrix[index, 2] = ranks[(row.world_id, row.bank_id)][suffix]
        if prefix in prefix_codes:
            matrix[index, 3] = prefix_codes[prefix]
        if row.bank_id in bank_codes:
            matrix[index, 4] = bank_codes[row.bank_id]
    return matrix


def id_leak_passes(average_precision: float, base_rate: float, max_ap_over_base_rate: float) -> bool:
    """
    WHAT: Applies check 6b's pass rule: the id-only average precision must not exceed the configured
    multiple of the base rate.

    WHY: With no id signal, average precision sits near the base rate; a far higher value means the
    identifiers themselves carry the role.

    Args:
        average_precision: Validation average precision of the id-only model.
        base_rate: Share of validation accounts whose truth role is a mule role.
        max_ap_over_base_rate: Allowed multiple, read from config/leakage_checks.yaml.

    Returns:
        True when average_precision <= max_ap_over_base_rate * base_rate.

    Raises:
        ValueError: If a value is non-finite or outside its meaningful range.
    """
    if not math.isfinite(average_precision) or average_precision < 0.0:
        raise ValueError(f"average_precision must be a non-negative finite number; got {average_precision!r}")
    if not math.isfinite(base_rate) or not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be a finite number in (0.0, 1.0); got {base_rate!r}")
    if not math.isfinite(max_ap_over_base_rate) or max_ap_over_base_rate <= 0.0:
        raise ValueError(f"max_ap_over_base_rate must be a positive finite number; got {max_ap_over_base_rate!r}")
    return average_precision <= max_ap_over_base_rate * base_rate


def measure_id_only_signal(
    training_rows: Sequence[AccountRow],
    validation_rows: Sequence[AccountRow],
    truth_root: Path,
    seed: int,
    max_ap_over_base_rate: float,
) -> dict[str, object]:
    """
    WHAT: Trains a small XGBoost on id-only features and measures how well it separates mule-role
    accounts on validation.

    WHY: Check 6b is a measurement, not a code scan: if the ids alone predict the hidden role far
    better than the base rate, the model can learn the answer from identity.

    Args:
        training_rows: Training account rows.
        validation_rows: Validation account rows; never test rows.
        truth_root: Folder holding per-run truth directories.
        seed: Fixed random seed for the model.
        max_ap_over_base_rate: Allowed average-precision multiple of the base rate.

    Returns:
        Validation average precision, base rate, their ratio, counts and the pass status.

    Raises:
        ValueError: If either split is empty or a split holds only one target class.
    """
    if not training_rows:
        raise ValueError("measure_id_only_signal needs at least one training account row")
    if not validation_rows:
        raise ValueError("measure_id_only_signal needs at least one validation account row")
    training_targets = _account_targets(training_rows, truth_root)
    validation_targets = _account_targets(validation_rows, truth_root)
    if len(set(training_targets)) < 2:
        raise ValueError("Training accounts must hold both mule-role and other accounts")
    if len(set(validation_targets)) < 2:
        raise ValueError("Validation accounts must hold both mule-role and other accounts")
    prefix_codes, bank_codes = _fit_category_codes(training_rows)
    training_matrix = _id_only_matrix(training_rows, prefix_codes, bank_codes)
    validation_matrix = _id_only_matrix(validation_rows, prefix_codes, bank_codes)
    model, scale_pos_weight = train_account_model(training_matrix, training_targets, seed)
    scores = model.predict_proba(validation_matrix)[:, 1]
    average_precision = float(average_precision_score(validation_targets, scores))
    base_rate = sum(validation_targets) / len(validation_targets)
    ratio = average_precision / base_rate
    passed = id_leak_passes(average_precision, base_rate, max_ap_over_base_rate)
    return {
        "status": "pass" if passed else "fail",
        "feature_names": list(ID_FEATURE_NAMES),
        "training_account_count": len(training_rows),
        "training_mule_account_count": sum(training_targets),
        "validation_account_count": len(validation_rows),
        "validation_mule_account_count": sum(validation_targets),
        "validation_base_rate": base_rate,
        "validation_average_precision": average_precision,
        "average_precision_over_base_rate": ratio,
        "max_ap_over_base_rate": max_ap_over_base_rate,
        "seed": seed,
        "scale_pos_weight": scale_pos_weight,
    }


def load_leakage_checks_config(path: Path = CONFIG_PATH) -> float:
    """
    WHAT: Reads check 6b's allowed average-precision multiple from config/leakage_checks.yaml.

    WHY: The pass rule is policy, so the number lives in config with its label, source and reason.

    Args:
        path: The config file; defaults to config/leakage_checks.yaml.

    Returns:
        The allowed multiple of the base rate.

    Raises:
        ValueError: If the file or block is missing, the block lacks its label, derived_from or
            reason, or the value is not a positive finite number.
    """
    if not path.is_file():
        raise ValueError(f"Leakage check config not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Leakage check config {path} must hold a mapping at the root")
    block = raw.get("id_leak_max_ap_over_base_rate")
    if not isinstance(block, dict):
        raise ValueError(f"Leakage check config {path} is missing the block 'id_leak_max_ap_over_base_rate'")
    for field in ("label", "derived_from", "reason"):
        text = block.get(field)
        if not isinstance(text, str) or not text:
            raise ValueError(f"Block 'id_leak_max_ap_over_base_rate' in {path} is missing a non-empty {field!r}")
    value = block.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"id_leak_max_ap_over_base_rate in {path} must be a positive finite number; got {value!r}")
    return float(value)


def _feature_source_files() -> list[Path]:
    """
    WHAT: Returns the feature modules check 5 and check 6 scan.

    WHY: These are the files that compute the account model's features and build its training rows.

    Raises:
        ValueError: If a configured file is missing.
    """
    files = sorted(REPO_ROOT.glob("features/m1_*.py"))
    if not files:
        raise ValueError("No features/m1_*.py files were found to scan")
    for relative in FEATURE_SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ValueError(f"Feature source file {path} was not found")
        files.append(path)
    return files


def _truth_root_from_datasets(dataset_dir: Path) -> Path:
    """
    WHAT: Returns the truth root beside a dataset folder: runs/full/truth for runs/full/datasets/<name>.

    WHY: The dataset manifest lists run ids, and each run id names a directory under the truth root.
    """
    return Path(dataset_dir).resolve().parents[1] / "truth"


def _default_truth_roles_path(training_dir: Path, truth_root: Path) -> Path:
    """
    WHAT: Returns the roles.jsonl of the first world in a training dataset's manifest.

    WHY: The world field set needs one real truth file's schema; tying it to the training data keeps
    the reported set tied to the worlds the model actually saw.

    Raises:
        ValueError: If the manifest is missing or lists no world with a run id.
    """
    manifest_path = training_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Training dataset {training_dir} is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    worlds = manifest.get("worlds")
    if not isinstance(worlds, list) or not worlds or not isinstance(worlds[0], dict):
        raise ValueError(f"Training dataset {training_dir} manifest holds no world")
    run_id = _require_nonempty_string(worlds[0], "run_id", "_default_truth_roles_path")
    return truth_root / run_id / "roles.jsonl"


def build_leakage_report(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    max_ap_over_base_rate: float,
    truth_root: Path | None = None,
) -> dict[str, Any]:
    """
    WHAT: Runs checks 5a, 5b, 6a and 6b and returns the report recorded in the run folder.

    WHY: One function builds the whole record so the field sets, the static scans and the
    measurement all describe the same training and validation data.

    Args:
        training_dirs: The four training dataset folders.
        validation_dirs: The validation dataset folders.
        max_ap_over_base_rate: Check 6b's allowed average-precision multiple.
        truth_root: Truth folder; when None it is taken beside the training datasets.

    Returns:
        A report with statuses, the three field sets, every static hit and the 6b numbers.

    Raises:
        ValueError: If a split is empty, the field-set gate fails, or a helper input is malformed.
    """
    training_paths = [Path(path) for path in training_dirs]
    validation_paths = [Path(path) for path in validation_dirs]
    if not training_paths or not validation_paths:
        raise ValueError("build_leakage_report needs at least one training and one validation dataset")
    resolved_truth_root = Path(truth_root) if truth_root is not None else _truth_root_from_datasets(training_paths[0])
    truth_roles_path = _default_truth_roles_path(training_paths[0], resolved_truth_root)
    world_fields = event_record_fields() | truth_file_columns(truth_roles_path)
    bank_visible = bank_visible_fields()
    simulator_only = simulator_only_fields(world_fields, bank_visible)
    source_files = _feature_source_files()
    hits_5a, excluded_5a = scan_simulator_field_usage(source_files, simulator_only)
    check_5b = check_training_dataset_columns(training_paths, simulator_only, account_model_feature_names())
    hits_6a, excluded_6a = scan_identifier_derivations(source_files, FORBIDDEN_IDENTIFIER_NAMES)
    seed = load_anomaly_detector_config().random_state
    training_rows = load_account_rows(training_paths)
    validation_rows = load_account_rows(validation_paths)
    check_6b = measure_id_only_signal(training_rows, validation_rows, resolved_truth_root, seed, max_ap_over_base_rate)
    check_5a = {
        "status": "pass" if not hits_5a else "fail",
        "files_scanned": [_display_path(path) for path in source_files],
        "hit_count": len(hits_5a),
        "hits": [asdict(hit) for hit in hits_5a],
        "excluded_method_calls": [asdict(hit) for hit in excluded_5a],
    }
    check_6a = {
        "status": "pass" if not hits_6a else "fail",
        "files_scanned": [_display_path(path) for path in source_files],
        "hit_count": len(hits_6a),
        "hits": [asdict(hit) for hit in hits_6a],
        "excluded_contexts": [asdict(hit) for hit in excluded_6a],
    }
    statuses = [check_5a["status"], check_5b["status"], check_6a["status"], check_6b["status"]]
    return {
        "status": "pass" if all(status == "pass" for status in statuses) else "fail",
        "fields": {
            "world_fields": sorted(world_fields),
            "bank_visible_fields": sorted(bank_visible),
            "simulator_only_fields": sorted(simulator_only),
            "truth_schema_file": _display_path(truth_roles_path),
            "bank_visible_sources": [
                "BANK_VISIBLE_PAYMENT_FIELDS",
                "ACCOUNT_OPENED_DETAIL_FIELDS",
                "BANK_VISIBLE_WITHDRAWAL_FIELDS",
                "BANK_VISIBLE_ACCOUNT_EVENT_FIELDS",
                "BANK_VISIBLE_COMPLAINT_FIELDS",
                "GENUINE_DEVICE_DETAIL_FIELDS",
                "GANG_DEVICE_DETAIL_FIELDS",
            ],
        },
        "check_5a": check_5a,
        "check_5b": check_5b,
        "check_6a": check_6a,
        "check_6b": check_6b,
        "config": {"max_ap_over_base_rate": max_ap_over_base_rate},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command-line entry point that runs checks 5 and 6 and writes the JSON report.

    Args:
        argv: Optional argument list; defaults to sys.argv.

    Returns:
        0 when every check passes, 2 when 5a, 5b or 6a fails (a real leak), 1 when only 6b fails
        (a finding to report).
    """
    parser = argparse.ArgumentParser(description="Run Section 23 leakage checks 5 and 6")
    parser.add_argument(
        "--training", type=Path, action="append", required=True, help="training dataset folder; repeatable"
    )
    parser.add_argument(
        "--validation", type=Path, action="append", required=True, help="validation dataset folder; repeatable"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument(
        "--truth-root", type=Path, default=None, help="truth folder; defaults beside the training datasets"
    )
    args = parser.parse_args(argv)
    max_ap_over_base_rate = load_leakage_checks_config()
    report = build_leakage_report(args.training, args.validation, max_ap_over_base_rate, args.truth_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    statuses = {name: report[name]["status"] for name in ("check_5a", "check_5b", "check_6a", "check_6b")}
    print(" ".join(f"{name.removeprefix('check_')} {status}" for name, status in statuses.items()))
    if any(statuses[name] == "fail" for name in ("check_5a", "check_5b", "check_6a")):
        return 2
    if statuses["check_6b"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
