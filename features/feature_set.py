# File: features/feature_set.py
# Purpose: Level 15b M1 live mule feature set definition, schema validation, and declarative loading.
# Issue: Level 15b long-history summaries, cash-out state, and M1 feature list (plan Task 1, R8, G1, G2, E3, R5, R7).
"""
WHAT: Provides declarative data structures and validation loaders for the Level 15 M1 feature set,
specifying exact feature names, functional groups, bank-visible sources, sliding horizons,
empty-history fallback semantics, and tracking excluded simulator features.

WHY: Fraud detection feature engineering requires an explicit, audited catalog of available
behavioral signals before scoring models are trained. Enforcing strict schema validation
guarantees that every engineered feature maps to realistic commercial bank ledger visibility,
prevents future-leakage, and rejects unsupported or hallucinated signals.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

# Canonical list of approved M1 feature grouping categories per architecture plan.
KNOWN_GROUPS: frozenset[str] = frozenset({
    "inflow",
    "outflow",
    "balance_pattern",
    "routine_life",
    "lifecycle",
    "control",
    "network",
    "intelligence",
    "region",
    "payment_level",
})


@dataclass(frozen=True)
class M1Feature:
    """
    Immutable representation of a single M1 candidate fraud detection feature.

    Attributes:
        name: Unique programmatic feature identifier.
        group: Functional category (e.g. 'inflow', 'control', 'routine_life').
        definition: Plain-English explanation of what behavioral signal the feature captures.
        bank_visible_source_records: Specific bank-visible data sources used to compute the value.
        window_or_horizon: Sliding look-back duration (e.g. '1h', '24h', '62d', 'account lifetime').
        empty_means: Semantic interpretation when no history exists ('zero', 'no data', 'not enough history').
        issue_ids: Associated project problem and typology issue codes (e.g. G1, G2, E3, R5, R7).
    """

    name: str
    group: str
    definition: str
    bank_visible_source_records: str
    window_or_horizon: str
    empty_means: str
    issue_ids: tuple[str, ...]


@dataclass(frozen=True)
class RemovedFeature:
    """
    Candidate feature explicitly excluded from M1 modeling with documented technical rationale.

    Attributes:
        name: Name of the candidate feature.
        reason: Plain-English reason why the signal is excluded or non-viable in this simulator.
    """

    name: str
    reason: str


@dataclass(frozen=True)
class M1FeatureSet:
    """
    Validated catalog of Level 15 M1 live mule features and documented exclusions.

    Attributes:
        features: Tuple of active M1Feature definitions.
        removed: Tuple of RemovedFeature exclusion records.
        gap_notes: Tuple of informational modeling gap notes.
    """

    features: tuple[M1Feature, ...]
    removed: tuple[RemovedFeature, ...]
    gap_notes: tuple[str, ...]

    def get_feature(self, name: str) -> M1Feature:
        """Looks up a feature by name; raises ValueError if not found."""
        for feat in self.features:
            if feat.name == name:
                return feat
        raise ValueError(f"Feature {name!r} not found in M1 feature set")

    def features_by_group(self, group: str) -> tuple[M1Feature, ...]:
        """Returns all features belonging to the specified functional group."""
        if group not in KNOWN_GROUPS:
            raise ValueError(f"Unknown group {group!r}; must be one of {sorted(KNOWN_GROUPS)}")
        return tuple(f for f in self.features if f.group == group)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Returns a tuple of all active feature names in catalog order."""
        return tuple(f.name for f in self.features)


def load_feature_set_m1(config_path: Path | None = None) -> M1FeatureSet:
    """
    PURPOSE:
        Loads and validates data/schemas/feature_set_m1.yaml into an immutable M1FeatureSet.

    INPUTS:
        config_path: Optional explicit Path to feature_set_m1.yaml. If None, resolves to default repository path.

    KEY LOGIC:
        1. Reads YAML file from disk with UTF-8 encoding.
        2. Validates top-level structure ('features' and 'removed' lists).
        3. Enforces uniqueness of names within active features.
        4. Validates that every feature specifies a group from KNOWN_GROUPS.
        5. Validates non-empty string values for definition and bank-visible source records.
        6. Enforces that no feature name appears in both active 'features' and 'removed' lists.
        7. Returns immutable M1FeatureSet.

    SIDE EFFECTS / ERRORS:
        FileNotFoundError: If the YAML file does not exist on disk.
        ValueError: If file content is missing required blocks, violates schema constraints,
            or contains duplicate names or unknown groups.

    CONSTRAINTS / COMPLEXITY:
        O(N) validation over the feature list; safe to call during startup configuration loading.
    """
    root_dir = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = root_dir / "data" / "schemas" / "feature_set_m1.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"M1 feature schema file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if not isinstance(raw_cfg, dict):
        raise ValueError(f"M1 feature set in {config_path} must be a YAML mapping")

    raw_features = raw_cfg.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError("Missing or empty 'features' list in M1 feature schema")

    raw_removed = raw_cfg.get("removed", [])
    if not isinstance(raw_removed, list):
        raise ValueError("'removed' must be a list of excluded features")

    raw_gap_notes = raw_cfg.get("gap_notes", [])
    if not isinstance(raw_gap_notes, list):
        raise ValueError("'gap_notes' must be a list of strings")
    gap_notes = tuple(str(n).strip() for n in raw_gap_notes if str(n).strip())

    seen_feature_names: set[str] = set()
    parsed_features: list[M1Feature] = []

    for idx, item in enumerate(raw_features):
        if not isinstance(item, dict):
            raise ValueError(f"Feature entry at index {idx} must be a dictionary")

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Feature entry at index {idx} missing non-empty 'name'")
        name = name.strip()

        if name in seen_feature_names:
            raise ValueError(f"Duplicate feature name {name!r} detected in M1 feature set")
        seen_feature_names.add(name)

        group = item.get("group")
        if not isinstance(group, str) or group not in KNOWN_GROUPS:
            raise ValueError(
                f"Feature {name!r} has invalid group {group!r}; must be one of {sorted(KNOWN_GROUPS)}"
            )

        definition = item.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"Feature {name!r} missing non-empty 'definition'")

        source = item.get("bank_visible_source_records") or item.get("source_records")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Feature {name!r} missing non-empty 'bank_visible_source_records'")

        window = item.get("window_or_horizon") or item.get("window")
        if not isinstance(window, str) or not window.strip():
            raise ValueError(f"Feature {name!r} missing non-empty 'window_or_horizon'")

        empty_means = item.get("empty_means")
        if not isinstance(empty_means, str) or not empty_means.strip():
            raise ValueError(f"Feature {name!r} missing non-empty 'empty_means'")

        raw_issue_ids = item.get("issue_ids", [])
        if not isinstance(raw_issue_ids, list):
            raise ValueError(f"Feature {name!r} 'issue_ids' must be a list of strings")
        issue_ids = tuple(str(i).strip() for i in raw_issue_ids)

        parsed_features.append(
            M1Feature(
                name=name,
                group=group,
                definition=definition.strip(),
                bank_visible_source_records=source.strip(),
                window_or_horizon=window.strip(),
                empty_means=empty_means.strip(),
                issue_ids=issue_ids,
            )
        )

    seen_removed_names: set[str] = set()
    parsed_removed: list[RemovedFeature] = []

    for idx, item in enumerate(raw_removed):
        if not isinstance(item, dict):
            raise ValueError(f"Removed feature entry at index {idx} must be a dictionary")

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Removed feature entry at index {idx} missing non-empty 'name'")
        name = name.strip()

        if name in seen_removed_names:
            raise ValueError(f"Duplicate removed feature name {name!r}")
        seen_removed_names.add(name)

        if name in seen_feature_names:
            raise ValueError(f"Feature {name!r} appears in both active 'features' and 'removed' lists")

        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Removed feature {name!r} missing non-empty 'reason'")

        parsed_removed.append(RemovedFeature(name=name, reason=reason.strip()))

    return M1FeatureSet(
        features=tuple(parsed_features),
        removed=tuple(parsed_removed),
        gap_notes=gap_notes,
    )
