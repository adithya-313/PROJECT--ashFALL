"""
WHAT: One schema-complete mule model config body for tests that write a config file.

WHY: The loader refuses a config missing any required parameter block, so a test about one value
must still supply the other blocks or it fails before reaching the code it protects.
"""

from pathlib import Path

LABEL_FEATURES = (
    "hops_to_complained_account",
    "registry_match_count",
    "registry_match_age_days",
)
FEEDBACK_FEATURES = (
    "hops_to_confirmed_label",
    "shared_counterparties_with_flagged_7d",
)


def _block(label: str, value: str) -> str:
    """Returns one parameter block whose value line is written verbatim."""
    return f"  {label}:\n    label: policy\n{value}    derived_from: test\n    reason: test\n"


def _name_list(names: tuple[str, ...]) -> str:
    """Returns the YAML value lines for a list of feature names."""
    return "    value:\n" + "".join(f"      - {name}\n" for name in names)


def label_trust_blocks_text(*, label_trust_weighting: bool = False) -> str:
    """
    Returns the three label-trust parameter blocks a schema-complete config must carry.

    The curve is a valid non-decreasing ramp of 91 shares from 0.0 to 1.0, so tests exercise
    parsing rather than the measured values.
    """
    curve_values = ", ".join(str(round(day / 90, 6)) for day in range(91))
    return (
        _block("label_trust_curve", f"    value: [{curve_values}]\n")
        + _block("label_trust_weighting", f"    value: {str(label_trust_weighting).lower()}\n")
        + _block("maturity_tie_tolerance", "    value: 0.002\n")
    )


def mule_model_config_text(
    *,
    label_cut: str = "arrival_time",
    label_maturity_days: int | None = 0,
    feature_fill_flag_share: float = 0.999,
    label_features: tuple[str, ...] = LABEL_FEATURES,
    feedback_features: tuple[str, ...] = FEEDBACK_FEATURES,
) -> str:
    """Returns a config body carrying every required block and the given values."""
    maturity = (
        "    value: null\n    not_found: true\n"
        if label_maturity_days is None
        else f"    value: {label_maturity_days}\n"
    )
    return (
        "parameters:\n"
        + _block("label_cut", f"    value: {label_cut}\n")
        + _block("label_maturity_days", maturity)
        + label_trust_blocks_text()
        + _block("feature_fill_flag_share", f"    value: {feature_fill_flag_share}\n")
        + _block("use_label_network_features", "    value: true\n")
        + _block("label_network_features", _name_list(label_features))
        + _block("feedback_network_features_not_used", _name_list(feedback_features))
        + _block("hyperparameter_search", '    value: "off"\n')
        + _block("optuna_trials", "    value: 40\n")
        + _block("optuna_seed", "    value: 1717\n")
        + _block("optuna_min_trials_before_stop", "    value: 20\n")
        + _block("optuna_stall_patience_trials", "    value: 10\n")
        + _block("optuna_min_improvement", "    value: 0.001\n")
        + _block(
            "hyperparameter_space",
            "    value:\n"
            "      max_depth: {min: 3, max: 10, log: false}\n"
            "      learning_rate: {min: 0.02, max: 0.3, log: true}\n"
            "      n_estimators: {min: 100, max: 800, log: false}\n"
            "      min_child_weight: {min: 1, max: 20, log: true}\n"
            "      subsample: {min: 0.5, max: 1.0, log: false}\n"
            "      colsample_bytree: {min: 0.5, max: 1.0, log: false}\n"
            "      gamma: {min: 0.0, max: 5.0, log: false}\n"
            "      reg_alpha: {min: 0.001, max: 10.0, log: true}\n"
            "      reg_lambda: {min: 0.1, max: 10.0, log: true}\n",
        )
    )


def write_mule_model_config(
    path: Path,
    *,
    label_cut: str = "arrival_time",
    label_maturity_days: int | None = 0,
    feature_fill_flag_share: float = 0.999,
    label_features: tuple[str, ...] = LABEL_FEATURES,
    feedback_features: tuple[str, ...] = FEEDBACK_FEATURES,
) -> Path:
    """Writes the schema-complete config body with the given values and returns the path."""
    path.write_text(
        mule_model_config_text(
            label_cut=label_cut,
            label_maturity_days=label_maturity_days,
            feature_fill_flag_share=feature_fill_flag_share,
            label_features=label_features,
            feedback_features=feedback_features,
        ),
        encoding="utf-8",
    )
    return path
