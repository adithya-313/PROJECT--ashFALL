"""
WHAT: Positive-unlabelled training for the Level 18 account model: PU bagging and weighted
Elkan-Noto, both built on train_account_model.

WHY: The bank only ever holds positive labels (complaints, police and registry reports) and treats
every unlabelled account as a negative, which is wrong for the many frauds that were never reported.
PU bagging trains each member on all positives plus a small random sample of unlabelled rows, and
weighted Elkan-Noto weighs every unlabelled row as both a positive and a negative with the weight
implied by an estimate of P(labelled | x).

TERM: unlabelled row: a training row whose target is 0 because no bank-visible label named it.
TERM: margin: the model's raw score before the logistic link; SHAP explains this space for an
XGBoost classifier with a logistic objective.

Every function validates the shapes and values it depends on and raises ValueError; nothing is
guessed and no failure is swallowed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy
import shap
import xgboost

from decisions.mule_account_model import train_account_model

__all__ = ["PUBaggedAccountModel", "train_elkan_noto_model", "train_pu_bagged_model"]

_SIDECAR_NAME = "ensemble.json"


@dataclass(frozen=True)
class PUBaggedAccountModel:
    """
    WHAT: An unordered ensemble of account models, each trained on all positives plus one small
    unlabelled sample.

    WHY: A single model must arbitrarily treat every unlabelled account as a negative; averaging
    many members trained on different unlabelled samples stops any one arbitrary sample from
    deciding the score. Margins (not probabilities) are averaged so the ensemble still explains
    exactly with SHAP, because the logistic link is applied once to the mean margin.

    Attributes:
        members: Fitted XGBClassifier members, in seed order.
        member_seeds: Random seed used to sample and fit each member; member i uses seed + i.
        n_bags: Number of members; equals len(members).
        n_estimators: Boosting rounds passed to every member.
        unlabelled_per_positive: Unlabelled rows each member draws per positive row.
    """

    members: tuple[xgboost.XGBClassifier, ...]
    member_seeds: tuple[int, ...]
    n_bags: int
    n_estimators: int
    unlabelled_per_positive: int

    def predict_margins(self, matrix: list[list[float]]) -> list[float]:
        """
        WHAT: Returns the mean raw margin over the members for every row of matrix.

        WHY: Averaging margins keeps the ensemble's SHAP values exact: the mean of each member's
        additive margin explanation is an additive explanation of the mean margin.

        Args:
            matrix: Feature rows in the order the members were trained on.

        Returns:
            One mean margin per input row, in row order.

        Raises:
            ValueError: If no member has been fitted.
        """
        if not self.members:
            raise ValueError("PUBaggedAccountModel requires at least one member")
        member_margins = [
            numpy.asarray(member.predict(matrix, output_margin=True), dtype=float) for member in self.members
        ]
        mean_margins = numpy.mean(numpy.vstack(member_margins), axis=0)
        return [float(margin) for margin in mean_margins]

    def predict_probabilities(self, matrix: list[list[float]]) -> list[float]:
        """
        WHAT: Returns the logistic probability of the mean margin for every row of matrix.

        WHY: Callers such as predict_account_days and the score-cut tuner expect a fraud
        probability, while the ensemble itself averages in margin space.

        Args:
            matrix: Feature rows in the order the members were trained on.

        Returns:
            One probability per input row, in row order.

        Raises:
            ValueError: If no member has been fitted.
        """
        margins = numpy.asarray(self.predict_margins(matrix), dtype=float)
        probabilities = 1.0 / (1.0 + numpy.exp(-margins))
        return [float(probability) for probability in probabilities]

    def shap_values(self, matrix: list[list[float]]) -> numpy.ndarray:
        """
        WHAT: Returns the mean SHAP values over the members for every row of matrix.

        WHY: The installed SHAP explains an XGBoost logistic classifier in margin space by default,
        which is the space the ensemble averages in, so the mean of the members' explanations is an
        exact explanation of the ensemble's margin.

        Args:
            matrix: Feature rows in the order the members were trained on.

        Returns:
            A (rows, features) array of mean margin-space SHAP values.

        Raises:
            ValueError: If no member has been fitted.
        """
        if not self.members:
            raise ValueError("PUBaggedAccountModel requires at least one member")
        member_values = [
            numpy.asarray(shap.TreeExplainer(member).shap_values(matrix), dtype=float) for member in self.members
        ]
        return numpy.mean(numpy.stack(member_values), axis=0)

    def save(self, directory: Path, metadata: dict[str, Any]) -> None:
        """
        WHAT: Writes every member to member_NNN.ubj and one ensemble.json sidecar into directory.

        WHY: A member file alone does not say how many members there are or how each was sampled,
        so loading must read those facts from one sidecar.

        Args:
            directory: Destination folder; created when absent.
            metadata: Must carry training_dataset_ids, recorded in the sidecar.

        Raises:
            ValueError: If metadata does not carry the required key.
        """
        required = {"training_dataset_ids"}
        missing = sorted(required - set(metadata))
        if missing:
            raise ValueError(f"PUBaggedAccountModel.save metadata is missing required keys: {missing}")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for index, member in enumerate(self.members):
            member.save_model(str(directory / f"member_{index:03d}.ubj"))
        sidecar = {
            "n_bags": self.n_bags,
            "member_seeds": list(self.member_seeds),
            "unlabelled_per_positive": self.unlabelled_per_positive,
            "n_estimators": self.n_estimators,
            "training_dataset_ids": metadata["training_dataset_ids"],
        }
        (directory / _SIDECAR_NAME).write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "PUBaggedAccountModel":
        """
        WHAT: Rebuilds the ensemble from the sidecar and the member files written by save.

        Args:
            directory: Folder holding ensemble.json and member_NNN.ubj files.

        Returns:
            The reconstructed ensemble with the recorded seeds and sampling settings.

        Raises:
            ValueError: If the sidecar is missing or malformed, or a member file is absent or
                disagrees with the recorded member count.
        """
        directory = Path(directory)
        sidecar_path = directory / _SIDECAR_NAME
        if not sidecar_path.is_file():
            raise ValueError(f"PU bagging sidecar is missing: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict):
            raise ValueError(f"PU bagging sidecar {sidecar_path} must hold a mapping")

        n_bags = sidecar.get("n_bags")
        if not isinstance(n_bags, int) or isinstance(n_bags, bool) or n_bags <= 0:
            raise ValueError(f"PU bagging sidecar {sidecar_path} needs a positive integer 'n_bags'")
        member_seeds = sidecar.get("member_seeds")
        if (
            not isinstance(member_seeds, list)
            or len(member_seeds) != n_bags
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in member_seeds)
        ):
            raise ValueError(f"PU bagging sidecar {sidecar_path} needs one integer seed per member in 'member_seeds'")
        unlabelled_per_positive = sidecar.get("unlabelled_per_positive")
        if (
            not isinstance(unlabelled_per_positive, int)
            or isinstance(unlabelled_per_positive, bool)
            or unlabelled_per_positive <= 0
        ):
            raise ValueError(f"PU bagging sidecar {sidecar_path} needs a positive integer 'unlabelled_per_positive'")
        n_estimators = sidecar.get("n_estimators")
        if not isinstance(n_estimators, int) or isinstance(n_estimators, bool) or n_estimators <= 0:
            raise ValueError(f"PU bagging sidecar {sidecar_path} needs a positive integer 'n_estimators'")

        members: list[xgboost.XGBClassifier] = []
        for index in range(n_bags):
            member_path = directory / f"member_{index:03d}.ubj"
            if not member_path.is_file():
                raise ValueError(f"PU bagging member file is missing: {member_path}")
            member = xgboost.XGBClassifier()
            member.load_model(str(member_path))
            members.append(member)
        return cls(
            members=tuple(members),
            member_seeds=tuple(member_seeds),
            n_bags=n_bags,
            n_estimators=n_estimators,
            unlabelled_per_positive=unlabelled_per_positive,
        )


def train_pu_bagged_model(
    matrix: list[list[float]],
    targets: list[int],
    seed: int,
    n_estimators: int,
    n_bags: int,
    unlabelled_per_positive: int,
) -> PUBaggedAccountModel:
    """
    WHAT: Trains a PU bagging ensemble, one member per bag.

    WHY: Each member sees every positive but only a small sample of the unlabelled rows, so the
    rows an unlabelled account is accidentally treated as a negative for change from bag to bag;
    averaging the members' margins removes that arbitrary choice.

    Key logic:
        Bag i draws min(unlabelled, unlabelled_per_positive * positives) unlabelled rows without
        replacement with numpy.random.default_rng(seed + i), prepends all positive rows, and fits
        with train_account_model(..., seed + i, n_estimators), which computes that bag's own
        positives / negatives class weight. The drawn unlabelled indices are sorted so a bag that
        happens to cover every unlabelled row has exactly the full-data row order.

    Args:
        matrix: Feature rows in account_model_feature_names() order.
        targets: Binary labels aligned with matrix; 0 marks an unlabelled row.
        seed: Base random seed; member i uses seed + i for both sampling and fitting.
        n_estimators: Boosting rounds passed to every member.
        n_bags: Number of members to train.
        unlabelled_per_positive: Unlabelled rows drawn per positive row in each bag.

    Returns:
        The fitted PUBaggedAccountModel.

    Raises:
        ValueError: If matrix and targets differ in length, n_bags or unlabelled_per_positive is
            not positive, or there is no positive or no unlabelled row to sample from.
    """
    if len(matrix) != len(targets):
        raise ValueError(f"matrix length {len(matrix)} does not match targets length {len(targets)}")
    if n_bags <= 0:
        raise ValueError(f"n_bags must be positive; got {n_bags}")
    if unlabelled_per_positive <= 0:
        raise ValueError(f"unlabelled_per_positive must be positive; got {unlabelled_per_positive}")

    positive_indices = [index for index, target in enumerate(targets) if target == 1]
    unlabelled_indices = [index for index, target in enumerate(targets) if target == 0]
    if not positive_indices:
        raise ValueError("PU bagging requires at least one positive row")
    if not unlabelled_indices:
        raise ValueError("PU bagging requires at least one unlabelled row")

    sample_size = min(len(unlabelled_indices), unlabelled_per_positive * len(positive_indices))
    unlabelled_array = numpy.asarray(unlabelled_indices, dtype=numpy.int64)

    members: list[xgboost.XGBClassifier] = []
    member_seeds: list[int] = []
    for bag_index in range(n_bags):
        member_seed = seed + bag_index
        generator = numpy.random.default_rng(member_seed)
        sampled = generator.choice(unlabelled_array, size=sample_size, replace=False)
        ordered_indices = positive_indices + sorted(int(index) for index in sampled)
        bag_rows = [matrix[index] for index in ordered_indices]
        bag_targets = [targets[index] for index in ordered_indices]
        member, _ = train_account_model(bag_rows, bag_targets, member_seed, n_estimators)
        members.append(member)
        member_seeds.append(member_seed)

    return PUBaggedAccountModel(
        members=tuple(members),
        member_seeds=tuple(member_seeds),
        n_bags=n_bags,
        n_estimators=n_estimators,
        unlabelled_per_positive=unlabelled_per_positive,
    )


def _positive_probability_mean(probabilities: list[float], targets: list[int]) -> float:
    """
    WHAT: Returns c, the mean of g's positive probability over the rows whose target is 1.

    WHY: With g trained with scale_pos_weight 1.0, g(x) estimates P(labelled | x); its mean over
    labelled validation rows estimates the constant label frequency c the Elkan-Noto weights need.

    Args:
        probabilities: g's positive probability for one row per validation target.
        targets: Binary validation targets aligned with probabilities.

    Returns:
        The mean probability over target-1 rows.

    Raises:
        ValueError: If the lengths differ, there is no target-1 row, or the mean is zero.
    """
    if len(probabilities) != len(targets):
        raise ValueError(f"probabilities length {len(probabilities)} does not match targets length {len(targets)}")
    selected = [probability for probability, target in zip(probabilities, targets, strict=True) if target == 1]
    if not selected:
        raise ValueError("Elkan-Noto requires at least one positive validation row")
    c = float(numpy.mean(selected))
    if c == 0.0:
        raise ValueError("Elkan-Noto c is zero; g assigns no probability to labelled validation rows")
    return c


def _elkan_noto_weights(
    probabilities: list[float],
    c: float,
    probability_clip: float,
) -> tuple[list[float], int]:
    """
    WHAT: Returns the Elkan-Noto weight per probability and how many weights the upper clip changed.

    WHY: Elkan-Noto treats each unlabelled row as a positive with weight ((1 - c) / c) * p / (1 - p)
    and as a negative with weight 1 - weight, where p bounds P(labelled | x, labelled or not). The
    probability clip keeps p / (1 - p) finite, and the weight clip keeps every weight meaningful.

    Args:
        probabilities: g's positive probability for each unlabelled training row.
        c: Constant label frequency estimated on validation positives; must be in (0, 1].
        probability_clip: Cap on p strictly inside (0, 1).

    Returns:
        (weights, clip_count) where each weight lies in [0, 1] and clip_count is the number of
        raw weights above 1 that the upper clip reduced.

    Raises:
        ValueError: If c is not in (0, 1] or probability_clip is not in (0, 1).
    """
    if not 0.0 < c <= 1.0:
        raise ValueError(f"Elkan-Noto c must be in (0.0, 1.0]; got {c}")
    if not 0.0 < probability_clip < 1.0:
        raise ValueError(f"probability_clip must be in (0.0, 1.0); got {probability_clip}")

    weights: list[float] = []
    clip_count = 0
    for probability in probabilities:
        clipped_probability = min(probability, probability_clip)
        weight = ((1.0 - c) / c) * clipped_probability / (1.0 - clipped_probability)
        if weight > 1.0:
            weight = 1.0
            clip_count += 1
        if weight < 0.0:
            weight = 0.0
        weights.append(weight)
    return weights, clip_count


def train_elkan_noto_model(
    train_matrix: list[list[float]],
    train_targets: list[int],
    validation_matrix: list[list[float]],
    validation_targets: list[int],
    seed: int,
    n_estimators: int,
    probability_clip: float,
) -> tuple[xgboost.XGBClassifier, float, dict[str, float], int]:
    """
    WHAT: Trains one account model with weighted Elkan-Noto positive-unlabelled learning.

    WHY: The supervised model treats every unlabelled row as certainly negative. Elkan-Noto instead
    estimates P(labelled | x) with an unbalanced model g, turns it into a per-row positive weight,
    and reuses every unlabelled row as both a weighted positive and a weighted negative, so the
    final model sees the right expected class balance.

    Key logic:
        g = train_account_model(..., scale_pos_weight_override=1.0). Training g unbalanced is what
        makes g(x) an estimate of P(labelled | x) rather than an estimate already inflated by the
        negatives / positives class weight. c is the mean g over validation positives. Each
        unlabelled training row then enters twice, once as target 1 with weight w and once as
        target 0 with weight 1 - w, and the final model's class weight is the total negative weight
        divided by the total positive weight.

    Args:
        train_matrix: Training feature rows in account_model_feature_names() order.
        train_targets: Binary training labels aligned with train_matrix.
        validation_matrix: Validation feature rows used only to estimate c.
        validation_targets: Binary validation labels aligned with validation_matrix.
        seed: Random seed for g and the final model.
        n_estimators: Boosting rounds for g and the final model.
        probability_clip: Cap on g(x) so p / (1 - p) stays finite.

    Returns:
        (final model, c, weight summary with min / median / max of the unlabelled weights, clip
        count) where clip_count is the number of weights the upper clip reduced.

    Raises:
        ValueError: If a matrix and its targets differ in length, there is no positive validation
            row, g gives c zero, the weight totals would make the class weight invalid, or
            probability_clip is outside (0, 1).
    """
    if len(train_matrix) != len(train_targets):
        raise ValueError(f"train_matrix length {len(train_matrix)} does not match targets length {len(train_targets)}")
    if len(validation_matrix) != len(validation_targets):
        raise ValueError(
            f"validation_matrix length {len(validation_matrix)} does not match targets length {len(validation_targets)}"
        )
    if not 0.0 < probability_clip < 1.0:
        raise ValueError(f"probability_clip must be in (0.0, 1.0); got {probability_clip}")

    g, _ = train_account_model(
        train_matrix,
        train_targets,
        seed,
        n_estimators,
        scale_pos_weight_override=1.0,
    )
    g_validation = [float(value) for value in g.predict_proba(validation_matrix)[:, 1]]
    c = _positive_probability_mean(g_validation, validation_targets)

    g_train = [float(value) for value in g.predict_proba(train_matrix)[:, 1]]
    positive_rows = [row for row, target in zip(train_matrix, train_targets, strict=True) if target == 1]
    unlabelled_rows = [row for row, target in zip(train_matrix, train_targets, strict=True) if target == 0]
    unlabelled_probabilities = [
        probability for probability, target in zip(g_train, train_targets, strict=True) if target == 0
    ]
    unlabelled_weights, clip_count = _elkan_noto_weights(unlabelled_probabilities, c, probability_clip)

    expanded_rows = list(positive_rows)
    expanded_targets = [1] * len(positive_rows)
    sample_weights = [1.0] * len(positive_rows)
    for row, weight in zip(unlabelled_rows, unlabelled_weights, strict=True):
        expanded_rows.append(row)
        expanded_targets.append(1)
        sample_weights.append(weight)
        expanded_rows.append(row)
        expanded_targets.append(0)
        sample_weights.append(1.0 - weight)

    positive_weight_total = sum(
        weight for weight, target in zip(sample_weights, expanded_targets, strict=True) if target == 1
    )
    negative_weight_total = sum(
        weight for weight, target in zip(sample_weights, expanded_targets, strict=True) if target == 0
    )
    if positive_weight_total <= 0.0 or negative_weight_total <= 0.0:
        raise ValueError(
            "Elkan-Noto weights leave a class with no total weight: "
            f"positive {positive_weight_total}, negative {negative_weight_total}"
        )
    class_weight = negative_weight_total / positive_weight_total

    final_model, _ = train_account_model(
        expanded_rows,
        expanded_targets,
        seed,
        n_estimators,
        sample_weight=sample_weights,
        scale_pos_weight_override=class_weight,
    )

    weight_summary = {
        "min": float(numpy.min(unlabelled_weights)),
        "median": float(numpy.median(unlabelled_weights)),
        "max": float(numpy.max(unlabelled_weights)),
    }
    return final_model, c, weight_summary, clip_count
