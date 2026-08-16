"""Leakage-controlled linear probes (clean-room reconstruction).

This module is a new implementation based on the surviving methodological
description and callers of the lost project.  It is not the original source.

The central invariant is simple: every learned transformation, hyperparameter,
and threshold is fitted without access to the corresponding outer-test items.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from .crowd_data import CROWD_EMOTIONS


SEED = 20240804
PROBABILITY_CLIP = 1e-12
DEFAULT_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_BLOCK_MULTIPLIER_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
# Reconstruction decision: the original threshold grid did not survive.
DEFAULT_THRESHOLD_GRID = tuple(float(v) for v in np.arange(0.05, 1.0, 0.05))

FloatMatrix = NDArray[np.float64]
EstimatorFactory = Callable[[float], "TransformedMultilabelLogistic"]
MulticlassEstimatorFactory = Callable[[float], "TransformedMulticlassLogistic"]
MulticlassBlockEstimatorFactory = Callable[
    [float, float], "TransformedMulticlassLogistic"
]


def _as_float_matrix(X: ArrayLike) -> FloatMatrix:
    """Validate a dense two-dimensional feature matrix."""

    return np.asarray(
        check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64),
        dtype=np.float64,
    )


def _as_binary_matrix(y: ArrayLike) -> NDArray[np.int64]:
    """Validate a non-empty, two-dimensional binary target matrix."""

    target = np.asarray(y)
    if target.ndim != 2 or target.shape[1] == 0:
        raise ValueError("y must be a non-empty 2D multilabel indicator matrix")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("y must contain only binary values 0 and 1")
    return target.astype(np.int64, copy=False)


def _binary_log_loss_bits(y_true: ArrayLike, probability: ArrayLike) -> float:
    """Mean binary cross-entropy across items and labels, measured in bits."""

    target = _as_binary_matrix(y_true)
    prob = np.asarray(probability, dtype=np.float64)
    if prob.shape != target.shape:
        raise ValueError(
            f"probability shape {prob.shape} does not match target shape {target.shape}"
        )
    prob = np.clip(prob, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    loss = -(target * np.log2(prob) + (1 - target) * np.log2(1.0 - prob))
    return float(np.mean(loss))


def _validated_probabilities(
    probability: ArrayLike,
    *,
    expected_shape: tuple[int, int],
    context: str,
) -> FloatMatrix:
    prob = np.asarray(probability, dtype=np.float64)
    if prob.shape != expected_shape:
        raise ValueError(
            f"{context} returned probability shape {prob.shape}; expected {expected_shape}"
        )
    if not np.isfinite(prob).all():
        raise ValueError(f"{context} returned non-finite probabilities")
    if ((prob < 0.0) | (prob > 1.0)).any():
        raise ValueError(f"{context} returned probabilities outside [0, 1]")
    return prob


def _validated_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in class_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("class_names must be non-empty, nonblank, and unique")
    return names


def _as_multiclass_target(
    y: ArrayLike,
    class_names: Sequence[str],
) -> NDArray[np.str_]:
    raw = np.asarray(y, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("y must be a non-empty one-dimensional multiclass target")
    if pd.isna(raw).any():
        raise ValueError("multiclass y must not contain missing values")
    target = raw.astype(str)
    unknown = sorted(set(target) - set(class_names))
    if unknown:
        raise ValueError(f"multiclass y contains unknown classes: {unknown}")
    return target


def _validated_multiclass_probabilities(
    probability: ArrayLike,
    *,
    expected_shape: tuple[int, int],
    context: str,
) -> FloatMatrix:
    prob = _validated_probabilities(
        probability,
        expected_shape=expected_shape,
        context=context,
    )
    if not np.allclose(prob.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError(f"{context} returned probability rows that do not sum to one")
    return prob


def _multiclass_log_loss_bits(
    y_true: ArrayLike,
    probability: ArrayLike,
    *,
    class_names: Sequence[str],
) -> float:
    names = _validated_class_names(class_names)
    target = _as_multiclass_target(y_true, names)
    prob = _validated_multiclass_probabilities(
        probability,
        expected_shape=(len(target), len(names)),
        context="multiclass loss",
    )
    index = {label: position for position, label in enumerate(names)}
    true_probability = prob[
        np.arange(len(target)),
        np.asarray([index[label] for label in target], dtype=np.int64),
    ]
    return float(-np.log2(np.clip(true_probability, PROBABILITY_CLIP, 1.0)).mean())


class BlockTransformer(TransformerMixin, BaseEstimator):
    """Standardize feature blocks independently and apply fixed multipliers.

    Separate train-fitted scalers prevent a high-dimensional block from
    inheriting the raw scale of another block.  ``block_dims=None`` treats the
    complete matrix as one block.

    The constructor API is a reconstruction decision.  Independent per-block
    ``StandardScaler`` fitting and train-only use are directly attested.
    """

    def __init__(
        self,
        *,
        block_dims: tuple[int, ...] | None = None,
        block_multipliers: tuple[float, ...] | None = None,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        self.block_dims = block_dims
        self.block_multipliers = block_multipliers
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "BlockTransformer":
        del y
        features = _as_float_matrix(X)
        n_features = int(features.shape[1])

        dims = self.block_dims if self.block_dims is not None else (n_features,)
        if not dims or any(not isinstance(dim, int) or dim <= 0 for dim in dims):
            raise ValueError("block_dims must contain positive integers")
        if sum(dims) != n_features:
            raise ValueError(
                f"block_dims sum to {sum(dims)}, but X contains {n_features} features"
            )

        multipliers = (
            self.block_multipliers
            if self.block_multipliers is not None
            else tuple(1.0 for _ in dims)
        )
        if len(multipliers) != len(dims):
            raise ValueError("block_multipliers must have one value per block")
        if not np.isfinite(np.asarray(multipliers, dtype=float)).all():
            raise ValueError("block_multipliers must be finite")
        if (np.asarray(multipliers, dtype=float) < 0).any():
            raise ValueError("block_multipliers must be non-negative")

        self.n_features_in_ = n_features
        self.block_dims_ = tuple(dims)
        self.block_multipliers_ = tuple(float(value) for value in multipliers)
        self.block_slices_: tuple[slice, ...] = tuple(
            slice(start, start + width)
            for start, width in _block_offsets(self.block_dims_)
        )
        self.scalers_: list[StandardScaler] = []
        for block_slice in self.block_slices_:
            scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
            scaler.fit(features[:, block_slice])
            self.scalers_.append(scaler)
        return self

    def transform(self, X: ArrayLike) -> FloatMatrix:
        check_is_fitted(self, ("scalers_", "block_slices_", "n_features_in_"))
        features = _as_float_matrix(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X contains {features.shape[1]} features; expected {self.n_features_in_}"
            )
        blocks = [
            scaler.transform(features[:, block_slice]) * multiplier
            for scaler, block_slice, multiplier in zip(
                self.scalers_,
                self.block_slices_,
                self.block_multipliers_,
                strict=True,
            )
        ]
        return np.concatenate(blocks, axis=1)


def _block_offsets(dims: Sequence[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for width in dims:
        offsets.append((start, int(width)))
        start += int(width)
    return offsets


class TransformedMultilabelLogistic(ClassifierMixin, BaseEstimator):
    """One L2-logistic model per label after train-fitted block scaling.

    A label that is constant in a training fold receives its clipped empirical
    prevalence.  This behaviour is attested by the surviving decoder script and
    prevents small inner folds from crashing.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        block_dims: tuple[int, ...] | None = None,
        block_multipliers: tuple[float, ...] | None = None,
        threshold: float = 0.5,
        class_weight: str | dict[int, float] | None = None,
        solver: str = "lbfgs",
        max_iter: int = 5000,
        random_state: int = SEED,
        probability_clip: float = PROBABILITY_CLIP,
    ) -> None:
        self.C = C
        self.block_dims = block_dims
        self.block_multipliers = block_multipliers
        self.threshold = threshold
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.probability_clip = probability_clip

    def fit(self, X: ArrayLike, y: ArrayLike) -> "TransformedMultilabelLogistic":
        if self.C <= 0:
            raise ValueError("C must be strictly positive")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")
        if not 0.0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must lie strictly between 0 and 0.5")

        features = _as_float_matrix(X)
        target = _as_binary_matrix(y)
        if features.shape[0] != target.shape[0]:
            raise ValueError("X and y contain different numbers of items")

        self.transformer_ = BlockTransformer(
            block_dims=self.block_dims,
            block_multipliers=self.block_multipliers,
        ).fit(features)
        transformed = self.transformer_.transform(features)

        self.n_features_in_ = features.shape[1]
        self.n_labels_ = target.shape[1]
        self.classes_ = [np.asarray([0, 1], dtype=np.int64) for _ in range(self.n_labels_)]
        self.estimators_: list[LogisticRegression | None] = []
        self.constant_probabilities_: list[float | None] = []

        for label_index in range(self.n_labels_):
            label = target[:, label_index]
            if np.unique(label).size < 2:
                probability = float(
                    np.clip(label.mean(), self.probability_clip, 1.0 - self.probability_clip)
                )
                self.estimators_.append(None)
                self.constant_probabilities_.append(probability)
                continue

            estimator = LogisticRegression(
                C=float(self.C),
                penalty="l2",
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
            estimator.fit(transformed, label)
            self.estimators_.append(estimator)
            self.constant_probabilities_.append(None)
        return self

    def predict_proba(self, X: ArrayLike) -> FloatMatrix:
        check_is_fitted(
            self,
            ("transformer_", "estimators_", "constant_probabilities_", "n_labels_"),
        )
        transformed = self.transformer_.transform(X)
        probability = np.empty((transformed.shape[0], self.n_labels_), dtype=np.float64)
        for label_index, (estimator, constant) in enumerate(
            zip(self.estimators_, self.constant_probabilities_, strict=True)
        ):
            if estimator is None:
                if constant is None:  # pragma: no cover - internal invariant
                    raise RuntimeError("constant-label estimator has no probability")
                probability[:, label_index] = constant
            else:
                positive_index = int(np.flatnonzero(estimator.classes_ == 1)[0])
                probability[:, label_index] = estimator.predict_proba(transformed)[
                    :, positive_index
                ]
        return np.clip(
            probability,
            self.probability_clip,
            1.0 - self.probability_clip,
        )

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")
        return (self.predict_proba(X) >= self.threshold).astype(np.int64)


class TransformedMulticlassLogistic(ClassifierMixin, BaseEstimator):
    """L2 multinomial logistic probe after train-fitted block scaling.

    The class axis is explicit and fixed rather than inherited from sklearn's
    lexical ordering. Missing training classes are rejected because no
    smoothing rule for this probe survives in the historical protocol.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        class_names: tuple[str, ...] = CROWD_EMOTIONS,
        block_dims: tuple[int, ...] | None = None,
        block_multipliers: tuple[float, ...] | None = None,
        class_weight: str | dict[str, float] | None = None,
        solver: str = "lbfgs",
        max_iter: int = 5000,
        random_state: int = SEED,
    ) -> None:
        self.C = C
        self.class_names = class_names
        self.block_dims = block_dims
        self.block_multipliers = block_multipliers
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "TransformedMulticlassLogistic":
        if self.C <= 0 or not np.isfinite(self.C):
            raise ValueError("C must be finite and strictly positive")
        names = _validated_class_names(self.class_names)
        features = _as_float_matrix(X)
        target = _as_multiclass_target(y, names)
        if features.shape[0] != target.shape[0]:
            raise ValueError("X and y contain different numbers of items")
        missing = sorted(set(names) - set(target))
        if missing:
            raise ValueError(f"multiclass training data misses classes: {missing}")

        self.transformer_ = BlockTransformer(
            block_dims=self.block_dims,
            block_multipliers=self.block_multipliers,
        ).fit(features)
        transformed = self.transformer_.transform(features)
        estimator = LogisticRegression(
            C=float(self.C),
            penalty="l2",
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(transformed, target)
        except ConvergenceWarning as error:
            raise RuntimeError(
                f"multiclass logistic probe did not converge for C={self.C}"
            ) from error
        self.estimator_ = estimator
        self.n_features_in_ = features.shape[1]
        self.classes_ = np.asarray(names, dtype=str)
        estimator_classes = set(estimator.classes_.astype(str))
        if estimator_classes != set(names):  # pragma: no cover - defensive seal
            raise RuntimeError("fitted estimator class axis is incomplete")
        return self

    def predict_proba(self, X: ArrayLike) -> FloatMatrix:
        check_is_fitted(self, ("transformer_", "estimator_", "classes_"))
        transformed = self.transformer_.transform(X)
        raw = self.estimator_.predict_proba(transformed)
        index = {
            str(label): position
            for position, label in enumerate(self.estimator_.classes_)
        }
        probability = raw[:, [index[label] for label in self.classes_]]
        return _validated_multiclass_probabilities(
            probability,
            expected_shape=(transformed.shape[0], len(self.classes_)),
            context="multiclass estimator",
        )

    def predict(self, X: ArrayLike) -> NDArray[np.str_]:
        probability = self.predict_proba(X)
        return self.classes_[np.argmax(probability, axis=1)]


def make_dense_multiclass_factory(
    *,
    class_names: tuple[str, ...] = CROWD_EMOTIONS,
    block_dims: tuple[int, ...] | None = None,
    block_multipliers: tuple[float, ...] | None = None,
    class_weight: str | dict[str, float] | None = None,
    solver: str = "lbfgs",
    max_iter: int = 5000,
    random_state: int = SEED,
) -> MulticlassEstimatorFactory:
    """Return a fixed-class multinomial estimator factory."""

    def factory(C: float) -> TransformedMulticlassLogistic:
        return TransformedMulticlassLogistic(
            C=C,
            class_names=class_names,
            block_dims=block_dims,
            block_multipliers=block_multipliers,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

    return factory


def make_dense_multilabel_factory(
    *,
    block_dims: tuple[int, ...] | None = None,
    block_multipliers: tuple[float, ...] | None = None,
    class_weight: str | dict[int, float] | None = None,
    solver: str = "lbfgs",
    max_iter: int = 5000,
    random_state: int = SEED,
) -> EstimatorFactory:
    """Return the estimator factory consumed by nested selection.

    The exact original factory signature is unknown; accepting ``C`` as the
    sole runtime argument keeps selection explicit and readily testable.
    """

    def factory(C: float) -> TransformedMultilabelLogistic:
        return TransformedMultilabelLogistic(
            C=C,
            block_dims=block_dims,
            block_multipliers=block_multipliers,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

    return factory


@dataclass(frozen=True)
class MultilabelSelection:
    """Inner-fold model-selection result for one outer training partition."""

    C: float
    threshold: float
    log_loss_bits: float
    macro_f1: float
    inner_oof_probabilities: FloatMatrix


@dataclass(frozen=True)
class NestedMultilabelResult:
    """Complete nested out-of-fold predictions and per-fold selections."""

    oof: pd.DataFrame
    selections: pd.DataFrame


@dataclass(frozen=True)
class MulticlassSelection:
    """Inner-OOF regularization selection on one outer training partition."""

    C: float
    log_loss_bits: float
    macro_f1: float
    inner_oof_probabilities: FloatMatrix


@dataclass(frozen=True)
class MulticlassBlockSelection:
    """Joint inner-OOF selection of regularization and first-block weight.

    With one feature block, ``block_multiplier`` is fixed to ``1.0``. With two
    blocks it denotes the effective multiplier tuple ``(block_multiplier,
    1.0)``. The second block is the reference scale.
    """

    C: float
    block_multiplier: float
    log_loss_bits: float
    macro_f1: float
    inner_oof_probabilities: FloatMatrix


@dataclass(frozen=True)
class NestedMulticlassResult:
    """Complete multiclass OOF predictions and per-fold selections."""

    oof: pd.DataFrame
    selections: pd.DataFrame


def _prepare_multiclass_inner_selection(
    X: ArrayLike,
    y: ArrayLike,
    *,
    validation_folds: ArrayLike,
    groups: ArrayLike,
    class_names: Sequence[str],
    selection_metric: Literal["log_loss", "macro_f1"],
) -> tuple[
    FloatMatrix,
    NDArray[np.str_],
    tuple[str, ...],
    NDArray[Any],
    list[Any],
    NDArray[np.str_],
]:
    """Validate and align the invariants shared by multiclass selectors."""

    names = _validated_class_names(class_names)
    features = _as_float_matrix(X)
    target = _as_multiclass_target(y, names)
    if target.shape[0] != features.shape[0]:
        raise ValueError("X and y contain different numbers of items")
    if selection_metric not in {"log_loss", "macro_f1"}:
        raise ValueError("selection_metric must be 'log_loss' or 'macro_f1'")

    folds = np.asarray(validation_folds)
    if folds.ndim != 1 or folds.shape[0] != features.shape[0]:
        raise ValueError("validation_folds must contain one value per item")
    if pd.isna(folds).any():
        raise ValueError("validation_folds must not contain missing values")
    try:
        unique_folds = sorted(pd.unique(folds).tolist())
    except TypeError as error:
        raise ValueError("validation_folds must contain comparable values") from error
    if len(unique_folds) < 2:
        raise ValueError("at least two non-empty inner validation folds are required")

    raw_groups = np.asarray(groups, dtype=object)
    if raw_groups.shape != folds.shape:
        raise ValueError("groups must contain one value per item")
    if pd.isna(raw_groups).any():
        raise ValueError("groups must not contain missing values")
    return (
        features,
        target,
        names,
        folds,
        unique_folds,
        raw_groups.astype(str),
    )


def _score_multiclass_inner_candidate(
    features: FloatMatrix,
    target: NDArray[np.str_],
    *,
    names: tuple[str, ...],
    folds: NDArray[Any],
    unique_folds: Sequence[Any],
    group_values: NDArray[np.str_],
    estimator_factory: Callable[[], TransformedMulticlassLogistic],
    context: str,
) -> tuple[FloatMatrix, float, float]:
    """Fit one candidate in every inner fold and score pooled OOF output."""

    inner_oof = np.full((len(target), len(names)), np.nan, dtype=np.float64)
    for validation_fold in unique_folds:
        validation_mask = folds == validation_fold
        training_mask = ~validation_mask
        if not validation_mask.any() or not training_mask.any():
            raise ValueError(f"inner fold {validation_fold!r} is empty")
        overlap = set(group_values[training_mask]) & set(group_values[validation_mask])
        if overlap:
            raise ValueError(
                f"group leakage in inner fold {validation_fold!r}: "
                f"{sorted(overlap)[:3]}"
            )
        missing = sorted(set(names) - set(target[training_mask]))
        if missing:
            raise ValueError(
                f"missing training classes in inner fold {validation_fold!r}, "
                f"{context}: {missing}"
            )
        estimator = estimator_factory()
        estimator.fit(features[training_mask], target[training_mask])
        inner_oof[validation_mask] = _validated_multiclass_probabilities(
            estimator.predict_proba(features[validation_mask]),
            expected_shape=(int(validation_mask.sum()), len(names)),
            context=(
                f"inner multiclass estimator for {context}, "
                f"fold={validation_fold!r}"
            ),
        )
    if not np.isfinite(inner_oof).all():  # pragma: no cover - defensive seal
        raise RuntimeError("inner multiclass OOF prediction matrix is incomplete")

    loss = _multiclass_log_loss_bits(target, inner_oof, class_names=names)
    prediction = np.asarray(names)[np.argmax(inner_oof, axis=1)]
    macro_f1 = float(
        f1_score(
            target,
            prediction,
            labels=list(names),
            average="macro",
            zero_division=0,
        )
    )
    return inner_oof, loss, macro_f1


def select_multiclass_C(
    X: ArrayLike,
    y: ArrayLike,
    *,
    validation_folds: ArrayLike,
    groups: ArrayLike,
    class_names: Sequence[str] = CROWD_EMOTIONS,
    estimator_factory: MulticlassEstimatorFactory | None = None,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    selection_metric: Literal["log_loss", "macro_f1"] = "log_loss",
) -> MulticlassSelection:
    """Select multinomial regularization from pooled inner-OOF predictions."""

    features, target, names, folds, unique_folds, group_values = (
        _prepare_multiclass_inner_selection(
            X,
            y,
            validation_folds=validation_folds,
            groups=groups,
            class_names=class_names,
            selection_metric=selection_metric,
        )
    )
    C_values = sorted({float(value) for value in C_grid})
    if not C_values or any(value <= 0 or not np.isfinite(value) for value in C_values):
        raise ValueError("C_grid must contain finite, strictly positive values")

    factory = estimator_factory or make_dense_multiclass_factory(
        class_names=tuple(names)
    )
    probabilities: dict[float, FloatMatrix] = {}
    losses: dict[float, float] = {}
    macro_f1_scores: dict[float, float] = {}
    for C in C_values:
        inner_oof, loss, macro_f1 = _score_multiclass_inner_candidate(
            features,
            target,
            names=names,
            folds=folds,
            unique_folds=unique_folds,
            group_values=group_values,
            estimator_factory=lambda C=C: factory(C),
            context=f"C={C}",
        )
        probabilities[C] = inner_oof
        losses[C] = loss
        macro_f1_scores[C] = macro_f1

    if selection_metric == "log_loss":
        best_C = min(C_values, key=lambda value: (losses[value], value))
    else:
        best_C = min(
            C_values,
            key=lambda value: (-macro_f1_scores[value], losses[value], value),
        )
    return MulticlassSelection(
        C=best_C,
        log_loss_bits=losses[best_C],
        macro_f1=macro_f1_scores[best_C],
        inner_oof_probabilities=probabilities[best_C].copy(),
    )


def select_multiclass_C_block_multiplier(
    X: ArrayLike,
    y: ArrayLike,
    *,
    validation_folds: ArrayLike,
    groups: ArrayLike,
    class_names: Sequence[str] = CROWD_EMOTIONS,
    block_dims: tuple[int, ...] | None = None,
    estimator_factory: MulticlassBlockEstimatorFactory | None = None,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    block_multiplier_grid: Sequence[float] = DEFAULT_BLOCK_MULTIPLIER_GRID,
    selection_metric: Literal["log_loss", "macro_f1"] = "log_loss",
) -> MulticlassBlockSelection:
    """Jointly select ``C`` and a first-block multiplier by pooled inner OOF.

    The selector intentionally supports only the protocol's one- and two-block
    cases. One block is always assigned multiplier ``(1.0,)`` and therefore
    does not create redundant candidates. For two blocks, candidate ``m``
    means ``(m, 1.0)``; the second block is the fixed reference.

    Reconstruction tie-break: retain the existing preference for the smaller
    ``C`` first, then prefer ``m`` closest to 1 on a log scale, then smaller
    ``m``. In macro-F1 mode, pooled log loss precedes those complexity ties.
    """

    features, target, names, folds, unique_folds, group_values = (
        _prepare_multiclass_inner_selection(
            X,
            y,
            validation_folds=validation_folds,
            groups=groups,
            class_names=class_names,
            selection_metric=selection_metric,
        )
    )
    dims = block_dims if block_dims is not None else (features.shape[1],)
    if len(dims) not in {1, 2}:
        raise ValueError("joint block selection requires exactly one or two blocks")
    if any(not isinstance(dim, int) or dim <= 0 for dim in dims):
        raise ValueError("block_dims must contain positive integers")
    if sum(dims) != features.shape[1]:
        raise ValueError(
            f"block_dims sum to {sum(dims)}, but X contains "
            f"{features.shape[1]} features"
        )

    C_values = sorted({float(value) for value in C_grid})
    if not C_values or any(value <= 0 or not np.isfinite(value) for value in C_values):
        raise ValueError("C_grid must contain finite, strictly positive values")
    if len(dims) == 1:
        multiplier_values = (1.0,)
    else:
        multiplier_values = tuple(
            sorted({float(value) for value in block_multiplier_grid})
        )
        if not multiplier_values or any(
            value <= 0 or not np.isfinite(value) for value in multiplier_values
        ):
            raise ValueError(
                "block_multiplier_grid must contain finite, strictly positive values"
            )

    default_factory = estimator_factory is None
    scores: dict[tuple[float, float], tuple[FloatMatrix, float, float]] = {}
    for C in C_values:
        for multiplier in multiplier_values:
            if default_factory:
                def factory(
                    C: float = C,
                    multiplier: float = multiplier,
                ) -> TransformedMulticlassLogistic:
                    return make_dense_multiclass_factory(
                        class_names=tuple(names),
                        block_dims=tuple(dims),
                        block_multipliers=(1.0,)
                        if len(dims) == 1
                        else (multiplier, 1.0),
                    )(C)
            else:
                if estimator_factory is None:  # pragma: no cover - narrowed above
                    raise RuntimeError("missing multiclass block estimator factory")
                def factory(
                    C: float = C,
                    multiplier: float = multiplier,
                ) -> TransformedMulticlassLogistic:
                    return estimator_factory(C, multiplier)
            scores[(C, multiplier)] = _score_multiclass_inner_candidate(
                features,
                target,
                names=names,
                folds=folds,
                unique_folds=unique_folds,
                group_values=group_values,
                estimator_factory=factory,
                context=f"C={C}, block_multiplier={multiplier}",
            )

    def complexity_tie(pair: tuple[float, float]) -> tuple[float, float, float]:
        C, multiplier = pair
        return C, abs(float(np.log(multiplier))), multiplier

    if selection_metric == "log_loss":
        best_pair = min(
            scores,
            key=lambda pair: (scores[pair][1], *complexity_tie(pair)),
        )
    else:
        best_pair = min(
            scores,
            key=lambda pair: (
                -scores[pair][2],
                scores[pair][1],
                *complexity_tie(pair),
            ),
        )
    probability, loss, macro_f1 = scores[best_pair]
    return MulticlassBlockSelection(
        C=best_pair[0],
        block_multiplier=best_pair[1],
        log_loss_bits=loss,
        macro_f1=macro_f1,
        inner_oof_probabilities=probability.copy(),
    )


def run_nested_multiclass_oof(
    X: ArrayLike,
    y: ArrayLike,
    *,
    item_ids: ArrayLike,
    class_names: Sequence[str] = CROWD_EMOTIONS,
    outer_folds: ArrayLike | pd.DataFrame,
    inner_folds: pd.DataFrame,
    groups: ArrayLike | None = None,
    estimator_factory: MulticlassEstimatorFactory | None = None,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    selection_metric: Literal["log_loss", "macro_f1"] = "log_loss",
) -> NestedMulticlassResult:
    """Generate leakage-controlled multinomial item-level OOF probabilities."""

    names = _validated_class_names(class_names)
    features = _as_float_matrix(X)
    target = _as_multiclass_target(y, names)
    raw_ids = np.asarray(item_ids, dtype=object)
    if raw_ids.ndim != 1 or raw_ids.shape[0] != features.shape[0]:
        raise ValueError("item_ids must contain one value per item")
    if pd.isna(raw_ids).any():
        raise ValueError("item_ids must not contain missing values")
    ids = raw_ids.astype(str)
    if target.shape[0] != features.shape[0]:
        raise ValueError("X and y contain different numbers of items")
    if np.unique(ids).size != ids.size:
        raise ValueError("item_ids must be unique")

    outer = _align_outer_folds(ids, outer_folds)
    inner = inner_folds.copy()
    required_inner = {"outer_fold", "item_id", "validation_fold"}
    if not required_inner.issubset(inner.columns):
        raise ValueError(f"inner_folds must contain columns {sorted(required_inner)}")
    if inner[list(required_inner)].isna().any().any():
        raise ValueError("inner_folds contains missing assignments")
    inner["item_id"] = inner["item_id"].astype(str)
    group_values = _align_group_values(
        ids,
        groups=groups,
        outer_folds=outer_folds,
        inner_folds=inner,
    )
    if pd.isna(outer).any():
        raise ValueError("outer_folds must not contain missing values")
    try:
        unique_outer_folds = sorted(pd.unique(outer).tolist())
    except TypeError as error:
        raise ValueError("outer_folds must contain comparable values") from error

    factory = estimator_factory or make_dense_multiclass_factory(
        class_names=tuple(names)
    )
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    for outer_fold in unique_outer_folds:
        test_mask = outer == outer_fold
        training_mask = ~test_mask
        if not test_mask.any() or not training_mask.any():
            raise ValueError(f"outer fold {outer_fold!r} is empty")
        overlap = set(group_values[training_mask]) & set(group_values[test_mask])
        if overlap:
            raise ValueError(
                f"group leakage in outer fold {outer_fold!r}: {sorted(overlap)[:3]}"
            )
        missing = sorted(set(names) - set(target[training_mask]))
        if missing:
            raise ValueError(
                f"missing training classes in outer fold {outer_fold!r}: {missing}"
            )

        train_ids = ids[training_mask]
        inner_rows = inner[inner["outer_fold"].astype(str) == str(outer_fold)]
        if inner_rows["item_id"].duplicated().any():
            raise ValueError(f"duplicate inner assignments for outer fold {outer_fold!r}")
        fold_by_id = dict(
            zip(
                inner_rows["item_id"],
                inner_rows["validation_fold"],
                strict=True,
            )
        )
        missing_ids = sorted(set(train_ids) - set(fold_by_id))
        unexpected_ids = sorted(set(fold_by_id) - set(train_ids))
        if missing_ids or unexpected_ids:
            raise ValueError(
                f"inner assignments do not match outer-train items for fold "
                f"{outer_fold!r}; missing={missing_ids[:3]}, "
                f"unexpected={unexpected_ids[:3]}"
            )
        validation_folds = np.asarray([fold_by_id[item_id] for item_id in train_ids])
        selection = select_multiclass_C(
            features[training_mask],
            target[training_mask],
            validation_folds=validation_folds,
            groups=group_values[training_mask],
            class_names=names,
            estimator_factory=factory,
            C_grid=C_grid,
            selection_metric=selection_metric,
        )
        estimator = factory(selection.C)
        estimator.fit(features[training_mask], target[training_mask])
        probability = _validated_multiclass_probabilities(
            estimator.predict_proba(features[test_mask]),
            expected_shape=(int(test_mask.sum()), len(names)),
            context=f"outer multiclass estimator for fold={outer_fold!r}",
        )
        prediction = np.asarray(names)[np.argmax(probability, axis=1)]
        frame_data: dict[str, Any] = {
            "item_id": ids[test_mask],
            "outer_fold": outer_fold,
            "group_id": group_values[test_mask],
            "y_true": target[test_mask],
            "y_pred": prediction,
        }
        for class_index, class_name in enumerate(names):
            frame_data[f"prob__{class_name}"] = probability[:, class_index]
        prediction_frames.append(pd.DataFrame(frame_data))
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "C": selection.C,
                "inner_log_loss_bits": selection.log_loss_bits,
                "inner_macro_f1": selection.macro_f1,
                "selection_metric": selection_metric,
                "n_train": int(training_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )

    oof = pd.concat(prediction_frames, ignore_index=True)
    if len(oof) != len(ids) or oof["item_id"].nunique() != len(ids):
        raise RuntimeError("outer multiclass OOF does not cover every item once")
    oof = oof.sort_values("item_id", kind="stable").reset_index(drop=True)
    selections = pd.DataFrame(selection_rows).sort_values(
        "outer_fold", kind="stable"
    ).reset_index(drop=True)
    return NestedMulticlassResult(oof=oof, selections=selections)


def select_multilabel_C_threshold(
    X: ArrayLike,
    y: ArrayLike,
    *,
    validation_folds: ArrayLike,
    groups: ArrayLike,
    estimator_factory: EstimatorFactory | None = None,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    selection_metric: Literal["macro_f1", "log_loss"] = "macro_f1",
) -> MultilabelSelection:
    """Select ``C`` and one global threshold from inner OOF predictions.

    In ``macro_f1`` mode, the `(C, threshold)` pair maximizes macro-F1. In
    ``log_loss`` mode, `C` minimizes binary log loss in bits and the threshold
    subsequently maximizes macro-F1. These rules are documented reconstruction
    decisions; the existence of both selection objectives is attested.
    """

    features = _as_float_matrix(X)
    target = _as_binary_matrix(y)
    folds = np.asarray(validation_folds)
    if folds.ndim != 1 or folds.shape[0] != features.shape[0]:
        raise ValueError("validation_folds must contain one value per item")
    if target.shape[0] != features.shape[0]:
        raise ValueError("X and y contain different numbers of items")
    if selection_metric not in {"macro_f1", "log_loss"}:
        raise ValueError("selection_metric must be 'macro_f1' or 'log_loss'")
    if pd.isna(folds).any():
        raise ValueError("validation_folds must not contain missing values")
    try:
        unique_folds = sorted(pd.unique(folds).tolist())
    except TypeError as error:
        raise ValueError("validation_folds must contain mutually comparable values") from error
    if len(unique_folds) < 2:
        raise ValueError("at least two non-empty inner validation folds are required")

    raw_groups = np.asarray(groups)
    if pd.isna(np.asarray(groups, dtype=object)).any():
        raise ValueError("groups must not contain missing values")
    group_values = raw_groups.astype(str)
    if group_values.shape != folds.shape:
        raise ValueError("groups must contain one value per item")

    C_values = sorted({float(value) for value in C_grid})
    if not C_values or any(value <= 0 or not np.isfinite(value) for value in C_values):
        raise ValueError("C_grid must contain finite, strictly positive values")
    thresholds = sorted({float(value) for value in threshold_grid})
    if not thresholds or any(
        value <= 0 or value >= 1 or not np.isfinite(value) for value in thresholds
    ):
        raise ValueError("threshold_grid values must lie strictly between 0 and 1")

    factory = estimator_factory or make_dense_multilabel_factory()
    probability_by_C: dict[float, FloatMatrix] = {}
    loss_by_C: dict[float, float] = {}
    threshold_scores_by_C: dict[float, dict[float, float]] = {}

    for C in C_values:
        inner_oof = np.full(target.shape, np.nan, dtype=np.float64)
        for validation_fold in unique_folds:
            validation_mask = folds == validation_fold
            training_mask = ~validation_mask
            if not validation_mask.any() or not training_mask.any():
                raise ValueError(f"inner fold {validation_fold!r} is empty")
            overlap = set(group_values[training_mask]) & set(group_values[validation_mask])
            if overlap:
                sample = sorted(overlap)[:3]
                raise ValueError(
                    f"group leakage in inner fold {validation_fold!r}: {sample}"
                )
            estimator = factory(C)
            estimator.fit(features[training_mask], target[training_mask])
            predicted = _validated_probabilities(
                estimator.predict_proba(features[validation_mask]),
                expected_shape=(int(validation_mask.sum()), target.shape[1]),
                context=f"inner estimator for C={C}, fold={validation_fold!r}",
            )
            inner_oof[validation_mask] = predicted
        if not np.isfinite(inner_oof).all():  # pragma: no cover - defensive invariant
            raise RuntimeError("inner OOF prediction matrix is incomplete")
        probability_by_C[C] = inner_oof
        loss_by_C[C] = _binary_log_loss_bits(target, inner_oof)
        threshold_scores_by_C[C] = {
            threshold: float(
                f1_score(
                    target,
                    (inner_oof >= threshold).astype(np.int64),
                    average="macro",
                    zero_division=0,
                )
            )
            for threshold in thresholds
        }

    def best_threshold_for_C(C: float) -> float:
        scores = threshold_scores_by_C[C]
        # Highest macro-F1; then closest to 0.5; then the lower threshold.
        return min(
            thresholds,
            key=lambda threshold: (
                -scores[threshold],
                abs(threshold - 0.5),
                threshold,
            ),
        )

    if selection_metric == "log_loss":
        best_C = min(C_values, key=lambda value: (loss_by_C[value], value))
        best_threshold = best_threshold_for_C(best_C)
    else:
        best_pair = min(
            (
                (C, best_threshold_for_C(C))
                for C in C_values
            ),
            key=lambda pair: (
                -threshold_scores_by_C[pair[0]][pair[1]],
                loss_by_C[pair[0]],
                pair[0],
                abs(pair[1] - 0.5),
                pair[1],
            ),
        )
        best_C, best_threshold = best_pair
    best_probability = probability_by_C[best_C]
    threshold_scores = threshold_scores_by_C[best_C]
    return MultilabelSelection(
        C=best_C,
        threshold=best_threshold,
        log_loss_bits=loss_by_C[best_C],
        macro_f1=threshold_scores[best_threshold],
        inner_oof_probabilities=best_probability.copy(),
    )


def run_nested_multilabel_oof(
    X: ArrayLike,
    y: ArrayLike,
    *,
    item_ids: ArrayLike,
    label_names: Sequence[str],
    outer_folds: ArrayLike | pd.DataFrame,
    inner_folds: pd.DataFrame,
    groups: ArrayLike | None = None,
    estimator_factory: EstimatorFactory | None = None,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    selection_metric: Literal["macro_f1", "log_loss"] = "macro_f1",
) -> NestedMultilabelResult:
    """Generate leakage-controlled item-level predictions under nested CV.

    The accepted split-table columns mirror the preserved CSV files:

    - outer: ``item_id``, ``test_fold``;
    - inner: ``outer_fold``, ``item_id``, ``validation_fold``.
    """

    features = _as_float_matrix(X)
    target = _as_binary_matrix(y)
    ids = np.asarray(item_ids).astype(str)
    names = tuple(str(name) for name in label_names)
    if ids.ndim != 1 or ids.shape[0] != features.shape[0]:
        raise ValueError("item_ids must contain one value per item")
    if np.unique(ids).size != ids.size:
        raise ValueError("item_ids must be unique")
    if target.shape != (features.shape[0], len(names)):
        raise ValueError("y shape must equal (n_items, len(label_names))")
    if len(set(names)) != len(names):
        raise ValueError("label_names must be unique")

    outer = _align_outer_folds(ids, outer_folds)
    inner = inner_folds.copy()
    required_inner = {"outer_fold", "item_id", "validation_fold"}
    if not required_inner.issubset(inner.columns):
        raise ValueError(f"inner_folds must contain columns {sorted(required_inner)}")
    inner["item_id"] = inner["item_id"].astype(str)

    group_values = _align_group_values(
        ids,
        groups=groups,
        outer_folds=outer_folds,
        inner_folds=inner,
    )

    factory = estimator_factory or make_dense_multilabel_factory()
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []

    if pd.isna(outer).any():
        raise ValueError("outer_folds must not contain missing values")
    try:
        unique_outer_folds = sorted(pd.unique(outer).tolist())
    except TypeError as error:
        raise ValueError("outer_folds must contain mutually comparable values") from error

    for outer_fold in unique_outer_folds:
        test_mask = outer == outer_fold
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"outer fold {outer_fold!r} is empty")
        if group_values is not None:
            overlap = set(group_values[train_mask]) & set(group_values[test_mask])
            if overlap:
                sample = sorted(overlap)[:3]
                raise ValueError(f"group leakage in outer fold {outer_fold!r}: {sample}")

        train_ids = ids[train_mask]
        inner_rows = inner[inner["outer_fold"].astype(str) == str(outer_fold)]
        if inner_rows["item_id"].duplicated().any():
            raise ValueError(f"duplicate inner assignments for outer fold {outer_fold!r}")
        fold_by_id = dict(
            zip(
                inner_rows["item_id"],
                inner_rows["validation_fold"],
                strict=True,
            )
        )
        missing = sorted(set(train_ids) - set(fold_by_id))
        unexpected = sorted(set(fold_by_id) - set(train_ids))
        if missing or unexpected:
            raise ValueError(
                f"inner assignments do not match outer-train items for fold {outer_fold!r}; "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}"
            )
        validation_folds = np.asarray([fold_by_id[item_id] for item_id in train_ids])

        selection = select_multilabel_C_threshold(
            features[train_mask],
            target[train_mask],
            validation_folds=validation_folds,
            groups=group_values[train_mask],
            estimator_factory=factory,
            C_grid=C_grid,
            threshold_grid=threshold_grid,
            selection_metric=selection_metric,
        )
        estimator = factory(selection.C)
        estimator.fit(features[train_mask], target[train_mask])
        probability = _validated_probabilities(
            estimator.predict_proba(features[test_mask]),
            expected_shape=(int(test_mask.sum()), target.shape[1]),
            context=f"outer estimator for fold={outer_fold!r}",
        )
        prediction = (probability >= selection.threshold).astype(np.int64)

        frame_data: dict[str, Any] = {
            "item_id": ids[test_mask],
            "outer_fold": outer_fold,
            # New serialization choice: preserves the fold-specific decision rule.
            "threshold": selection.threshold,
        }
        frame_data["group_id"] = group_values[test_mask]
        for label_index, label_name in enumerate(names):
            frame_data[f"y_true__{label_name}"] = target[test_mask, label_index]
            frame_data[f"prob__{label_name}"] = probability[:, label_index]
            frame_data[f"pred__{label_name}"] = prediction[:, label_index]
        prediction_frames.append(pd.DataFrame(frame_data))
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "C": selection.C,
                "threshold": selection.threshold,
                "inner_log_loss_bits": selection.log_loss_bits,
                "inner_macro_f1": selection.macro_f1,
                "selection_metric": selection_metric,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )

    oof = pd.concat(prediction_frames, ignore_index=True)
    if oof["item_id"].nunique() != ids.size or len(oof) != ids.size:
        raise RuntimeError("outer OOF predictions do not cover every item exactly once")
    oof = oof.sort_values("item_id", kind="stable").reset_index(drop=True)
    selections = pd.DataFrame(selection_rows).sort_values(
        "outer_fold", kind="stable"
    ).reset_index(drop=True)
    return NestedMultilabelResult(oof=oof, selections=selections)


def _align_outer_folds(
    item_ids: NDArray[np.str_], outer_folds: ArrayLike | pd.DataFrame
) -> NDArray[Any]:
    if isinstance(outer_folds, pd.DataFrame):
        required = {"item_id", "test_fold"}
        if not required.issubset(outer_folds.columns):
            raise ValueError(f"outer_folds must contain columns {sorted(required)}")
        frame = outer_folds.loc[:, ["item_id", "test_fold"]].copy()
        frame["item_id"] = frame["item_id"].astype(str)
        if frame["item_id"].duplicated().any():
            raise ValueError("outer_folds contains duplicate item assignments")
        fold_by_id = dict(zip(frame["item_id"], frame["test_fold"], strict=True))
        missing = sorted(set(item_ids) - set(fold_by_id))
        unexpected = sorted(set(fold_by_id) - set(item_ids))
        if missing or unexpected:
            raise ValueError(
                "outer_folds does not match item_ids; "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}"
            )
        return np.asarray([fold_by_id[item_id] for item_id in item_ids])

    folds = np.asarray(outer_folds)
    if folds.ndim != 1 or folds.shape[0] != item_ids.shape[0]:
        raise ValueError("outer_folds must contain one value per item")
    return folds


def _align_group_values(
    item_ids: NDArray[np.str_],
    *,
    groups: ArrayLike | None,
    outer_folds: ArrayLike | pd.DataFrame,
    inner_folds: pd.DataFrame,
) -> NDArray[np.str_]:
    """Recover and cross-check the canonical item-to-group mapping."""

    mappings: list[tuple[str, dict[str, str]]] = []
    if isinstance(outer_folds, pd.DataFrame) and "group_id" in outer_folds.columns:
        outer_groups = outer_folds.loc[:, ["item_id", "group_id"]].copy()
        if outer_groups["group_id"].isna().any():
            raise ValueError("outer_folds contains missing group_id values")
        outer_groups["item_id"] = outer_groups["item_id"].astype(str)
        outer_groups["group_id"] = outer_groups["group_id"].astype(str)
        if outer_groups["item_id"].duplicated().any():
            raise ValueError("outer_folds contains duplicate group assignments")
        mappings.append(
            (
                "outer_folds",
                dict(zip(outer_groups["item_id"], outer_groups["group_id"], strict=True)),
            )
        )

    if "group_id" in inner_folds.columns:
        inner_groups = inner_folds.loc[:, ["item_id", "group_id"]].copy()
        if inner_groups["group_id"].isna().any():
            raise ValueError("inner_folds contains missing group_id values")
        inner_groups["item_id"] = inner_groups["item_id"].astype(str)
        inner_groups["group_id"] = inner_groups["group_id"].astype(str)
        group_counts = inner_groups.groupby("item_id", sort=False)["group_id"].nunique()
        if (group_counts > 1).any():
            bad_id = str(group_counts[group_counts > 1].index[0])
            raise ValueError(f"inner_folds assigns multiple groups to item {bad_id!r}")
        deduplicated = inner_groups.drop_duplicates("item_id")
        mappings.append(
            (
                "inner_folds",
                dict(
                    zip(
                        deduplicated["item_id"],
                        deduplicated["group_id"],
                        strict=True,
                    )
                ),
            )
        )

    if groups is not None and pd.isna(np.asarray(groups, dtype=object)).any():
        raise ValueError("groups contains missing group_id values")
    explicit = None if groups is None else np.asarray(groups).astype(str)
    if explicit is not None:
        if explicit.shape != item_ids.shape:
            raise ValueError("groups must contain one value per item")
        mappings.append(("groups", dict(zip(item_ids, explicit, strict=True))))
    if not mappings:
        raise ValueError(
            "group IDs are required: pass groups or include group_id in a split table"
        )

    canonical_name, canonical = mappings[0]
    missing = sorted(set(item_ids) - set(canonical))
    unexpected = sorted(set(canonical) - set(item_ids))
    if missing or unexpected:
        raise ValueError(
            f"{canonical_name} group mapping does not match item_ids; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for mapping_name, mapping in mappings[1:]:
        missing = sorted(set(item_ids) - set(mapping))
        unexpected = sorted(set(mapping) - set(item_ids))
        if missing or unexpected:
            raise ValueError(
                f"{mapping_name} group mapping does not match item_ids; "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}"
            )
        disagreement = [
            item_id
            for item_id in item_ids
            if canonical[item_id] != mapping[item_id]
        ]
        if disagreement:
            item_id = str(disagreement[0])
            raise ValueError(
                f"group mismatch for item {item_id!r}: "
                f"{canonical_name}={canonical[item_id]!r}, "
                f"{mapping_name}={mapping[item_id]!r}"
            )
    return np.asarray([canonical[item_id] for item_id in item_ids], dtype=str)


__all__ = [
    "BlockTransformer",
    "DEFAULT_BLOCK_MULTIPLIER_GRID",
    "DEFAULT_C_GRID",
    "DEFAULT_THRESHOLD_GRID",
    "MulticlassBlockSelection",
    "MulticlassSelection",
    "MultilabelSelection",
    "NestedMulticlassResult",
    "NestedMultilabelResult",
    "SEED",
    "TransformedMulticlassLogistic",
    "TransformedMultilabelLogistic",
    "make_dense_multiclass_factory",
    "make_dense_multilabel_factory",
    "run_nested_multiclass_oof",
    "run_nested_multilabel_oof",
    "select_multiclass_C",
    "select_multiclass_C_block_multiplier",
    "select_multilabel_C_threshold",
]
