"""Metrics reconstructed from item-level out-of-fold probabilities.

This is a clean-room implementation, not recovered original source.  Its
attested core is: metrics are reconstructed per item from ``prob__*`` columns;
log loss is reported in bits; ECE uses ten equal-width bins; uncertainty uses
paired writer/conversation-group bootstrap resampling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


SEED = 20240804
PROBABILITY_CLIP = 1e-15


@dataclass(frozen=True)
class MetricReconstruction:
    """Metrics reconstructed from one item-level OOF table.

    ``classwise`` scores are one-vs-rest.  ``reliability`` omits empty bins.
    The exact original return type and column names are not recoverable.
    """

    overall: pd.Series
    classwise: pd.DataFrame
    per_item: pd.DataFrame
    reliability: pd.DataFrame


@dataclass(frozen=True)
class BootstrapDelta:
    """Paired group-bootstrap result with percentile confidence interval."""

    observed_a: float
    observed_b: float
    observed_delta: float
    ci_low: float
    ci_high: float
    standard_error: float
    confidence_level: float
    n_bootstrap: int
    n_groups: int
    seed: int
    samples: NDArray[np.float64]


def expected_calibration_error(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE with the bin boundaries used by surviving code."""

    table = reliability_table(confidence, correct, n_bins=n_bins)
    return float((table["weight"] * table["absolute_gap"]).sum())


def reliability_table(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return the non-empty equal-width reliability bins underlying ECE.

    The boundaries are attested: the first bin is ``[0, 1/B]`` and every
    later bin is ``(b/B, (b+1)/B]``.  Column names are reconstruction choices.
    """

    conf = np.asarray(confidence, dtype=np.float64)
    outcome = np.asarray(correct)
    if conf.ndim != 1 or outcome.ndim != 1 or conf.shape != outcome.shape:
        raise ValueError("confidence and correct must be aligned one-dimensional arrays")
    if conf.size == 0:
        raise ValueError("reliability requires at least one item")
    if n_bins <= 0:
        raise ValueError("n_bins must be strictly positive")
    _validate_unit_interval(conf, name="confidence")
    if not np.isin(outcome, (0, 1)).all():
        raise ValueError("correct must contain only 0/1 values")
    binary_outcome = outcome.astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        if bin_index == 0:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf > lower) & (conf <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(binary_outcome[mask].mean())
        mean_confidence = float(conf[mask].mean())
        rows.append(
            {
                "bin_index": bin_index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "weight": float(count / conf.size),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": abs(accuracy - mean_confidence),
            }
        )
    return pd.DataFrame(rows)


def reconstruct_multiclass_metrics(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
) -> MetricReconstruction:
    """Reconstruct multiclass metrics from fixed-order ``prob__<label>`` columns."""

    frame, names, probability = _validated_oof_probabilities(oof, labels=labels)
    if "y_true" not in frame.columns:
        raise ValueError("multiclass OOF table must contain y_true")
    truth = frame["y_true"].astype(str).to_numpy()
    label_to_index = {label: index for index, label in enumerate(names)}
    unknown = sorted(set(truth) - set(names))
    if unknown:
        raise ValueError(f"y_true contains unknown labels: {unknown[:3]}")
    row_sums = probability.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("multiclass probability rows must sum to one")

    true_index = np.asarray([label_to_index[value] for value in truth], dtype=int)
    pred_index = probability.argmax(axis=1)
    prediction = np.asarray([names[index] for index in pred_index], dtype=str)
    if "y_pred" in frame.columns:
        stored = frame["y_pred"].astype(str).to_numpy()
        if not np.array_equal(stored, prediction):
            raise ValueError("stored y_pred disagrees with argmax(prob__*)")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=list(names),
        zero_division=0,
    )
    item_loss = _multiclass_loss_from_arrays(true_index, probability)
    one_hot = np.eye(len(names), dtype=np.float64)[true_index]
    confidence = probability.max(axis=1)
    correct = (pred_index == true_index).astype(float)
    true_probability = probability[np.arange(len(frame)), true_index]
    item_brier = np.sum((probability - one_hot) ** 2, axis=1)
    if len(names) > 1:
        other_probability = probability.copy()
        other_probability[np.arange(len(frame)), true_index] = -np.inf
        true_probability_gap = true_probability - other_probability.max(axis=1)
        two_largest = np.partition(probability, -2, axis=1)[:, -2:]
        top_two_margin = two_largest.max(axis=1) - two_largest.min(axis=1)
    else:
        true_probability_gap = np.full(len(frame), np.nan, dtype=np.float64)
        top_two_margin = np.full(len(frame), np.nan, dtype=np.float64)

    identity_columns = [
        column
        for column in ("item_id", "outer_fold", "group_id")
        if column in frame.columns
    ]
    per_item = frame[identity_columns].copy()
    per_item["y_true"] = truth
    per_item["y_pred"] = prediction
    per_item["correct"] = correct.astype(bool)
    per_item["confidence"] = confidence
    per_item["true_class_probability"] = true_probability
    per_item["nll_bits"] = item_loss
    per_item["brier"] = item_brier
    per_item["true_class_probability_gap"] = true_probability_gap
    per_item["top_two_probability_margin"] = top_two_margin

    classwise_rows: list[dict[str, Any]] = []
    for index, label in enumerate(names):
        binary_truth = (true_index == index).astype(int)
        binary_probability = probability[:, index]
        classwise_rows.append(
            {
                "label": label,
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "ap": _average_precision_or_zero(binary_truth, binary_probability),
                "bce_bits": _binary_log_loss_bits(binary_truth, binary_probability),
                "brier": float(np.mean((binary_probability - binary_truth) ** 2)),
                "ece": expected_calibration_error(binary_probability, binary_truth),
            }
        )

    overall = pd.Series(
        {
            "n_items": int(len(frame)),
            "n_labels": int(len(names)),
            "accuracy": float(correct.mean()),
            "macro_f1": float(np.mean(f1)),
            "macro_ap": float(np.mean([row["ap"] for row in classwise_rows])),
            "log_loss_bits": float(item_loss.mean()),
            # Multiclass Brier: sum across classes, then mean across items.
            "brier": float(item_brier.mean()),
            "ece": expected_calibration_error(confidence, correct),
        },
        dtype=object,
    )
    return MetricReconstruction(
        overall=overall,
        classwise=pd.DataFrame(classwise_rows),
        per_item=per_item,
        reliability=reliability_table(confidence, correct),
    )


def reconstruct_multilabel_metrics(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
    threshold: float | None = None,
) -> MetricReconstruction:
    """Reconstruct multilabel metrics with global or row-wise OOF thresholds.

    Pass one global ``threshold``, or omit it when the OOF table contains the
    row-wise ``threshold`` selected inside each outer fold.  The latter column
    is a new serialization rule that makes the reconstructed nested runner and
    metric module composable.
    """

    frame, names, probability = _validated_oof_probabilities(oof, labels=labels)
    item_threshold = _resolve_multilabel_thresholds(frame, threshold=threshold)
    truth_columns = [f"y_true__{label}" for label in names]
    missing_truth = [column for column in truth_columns if column not in frame.columns]
    if missing_truth:
        raise ValueError(f"multilabel OOF table is missing columns: {missing_truth}")
    truth = _validated_binary_columns(
        frame,
        truth_columns,
        name="multilabel truth columns",
    )
    prediction = (probability >= item_threshold[:, np.newaxis]).astype(int)
    for index, label in enumerate(names):
        stored_column = f"pred__{label}"
        if stored_column in frame.columns:
            stored = _validated_binary_columns(
                frame,
                [stored_column],
                name=stored_column,
            )[:, 0]
            if not np.array_equal(stored, prediction[:, index]):
                raise ValueError(
                    f"stored {stored_column} disagrees with the applicable threshold"
                )

    classwise_rows: list[dict[str, Any]] = []
    for index, label in enumerate(names):
        binary_truth = truth[:, index]
        binary_prediction = prediction[:, index]
        binary_probability = probability[:, index]
        precision, recall, f1, support = precision_recall_fscore_support(
            binary_truth,
            binary_prediction,
            labels=[1],
            zero_division=0,
        )
        classwise_rows.append(
            {
                "label": label,
                "support": int(support[0]),
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "ap": _average_precision_or_zero(binary_truth, binary_probability),
                "bce_bits": _binary_log_loss_bits(binary_truth, binary_probability),
                "brier": float(np.mean((binary_probability - binary_truth) ** 2)),
                "ece": expected_calibration_error(binary_probability, binary_truth),
            }
        )

    exact_match = np.all(prediction == truth, axis=1)
    cell_correct = prediction == truth
    item_loss = _multilabel_loss_from_arrays(truth, probability)
    item_brier = np.mean((probability - truth) ** 2, axis=1)
    identity_columns = [
        column
        for column in ("item_id", "outer_fold", "group_id")
        if column in frame.columns
    ]
    per_item = frame[identity_columns].copy()
    per_item["threshold"] = item_threshold
    per_item["exact_match"] = exact_match
    per_item["hamming_accuracy"] = cell_correct.mean(axis=1)
    per_item["bce_bits"] = item_loss
    per_item["brier"] = item_brier
    unique_thresholds = np.unique(item_threshold)
    overall = pd.Series(
        {
            "n_items": int(len(frame)),
            "n_labels": int(len(names)),
            "threshold": (
                float(unique_thresholds[0]) if unique_thresholds.size == 1 else np.nan
            ),
            "threshold_mode": "global" if unique_thresholds.size == 1 else "row_wise",
            "n_thresholds": int(unique_thresholds.size),
            "accuracy": float(exact_match.mean()),
            "hamming_accuracy": float(cell_correct.mean()),
            "macro_f1": float(np.mean([row["f1"] for row in classwise_rows])),
            "macro_ap": float(np.mean([row["ap"] for row in classwise_rows])),
            "log_loss_bits": float(item_loss.mean()),
            "brier": float(item_brier.mean()),
            # Reconstruction decision: flatten item-label cells for overall ECE.
            "ece": expected_calibration_error(probability.ravel(), truth.ravel()),
        },
        dtype=object,
    )
    return MetricReconstruction(
        overall=overall,
        classwise=pd.DataFrame(classwise_rows),
        per_item=per_item,
        reliability=reliability_table(probability.ravel(), truth.ravel()),
    )


def summarize_multiclass_predictions(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
) -> MetricReconstruction:
    """Reconstruct the attested summary API from multiclass OOF probabilities.

    Only the historical function name is attested by the surviving pytest
    node IDs.  This clean-room signature and return type are new.
    """

    return reconstruct_multiclass_metrics(oof, labels=labels)


def reconstruct_multiclass(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
) -> MetricReconstruction:
    """Compatibility name inferred from surviving test names."""

    return reconstruct_multiclass_metrics(oof, labels=labels)


def reconstruct_multilabel(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
    threshold: float | None = None,
) -> MetricReconstruction:
    """Compatibility name inferred from surviving test names."""

    return reconstruct_multilabel_metrics(oof, labels=labels, threshold=threshold)


def multiclass_itemwise_log_loss_bits(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
) -> NDArray[np.float64]:
    """Return one paired multiclass log-loss value per item."""

    frame, names, probability = _validated_oof_probabilities(oof, labels=labels)
    if "y_true" not in frame.columns:
        raise ValueError("multiclass OOF table must contain y_true")
    label_to_index = {label: index for index, label in enumerate(names)}
    truth = frame["y_true"].astype(str).to_numpy()
    unknown = sorted(set(truth) - set(names))
    if unknown:
        raise ValueError(f"y_true contains unknown labels: {unknown[:3]}")
    row_sums = probability.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("multiclass probability rows must sum to one")
    indices = np.asarray([label_to_index[value] for value in truth], dtype=int)
    return _multiclass_loss_from_arrays(indices, probability)


def multilabel_itemwise_log_loss_bits(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
) -> NDArray[np.float64]:
    """Return per-item BCE averaged across the fixed label set."""

    frame, names, probability = _validated_oof_probabilities(oof, labels=labels)
    truth_columns = [f"y_true__{label}" for label in names]
    missing = [column for column in truth_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"multilabel OOF table is missing columns: {missing}")
    truth = _validated_binary_columns(
        frame,
        truth_columns,
        name="multilabel truth columns",
    )
    return _multilabel_loss_from_arrays(truth, probability)


def paired_group_bootstrap_delta(
    values_a: ArrayLike,
    values_b: ArrayLike,
    groups: ArrayLike,
    *,
    item_ids_a: ArrayLike,
    item_ids_b: ArrayLike,
    statistic: Callable[[NDArray[Any]], float] | None = None,
    n_bootstrap: int = 2000,
    seed: int = SEED,
    confidence_level: float = 0.95,
) -> BootstrapDelta:
    """Bootstrap ``statistic(A) - statistic(B)`` using paired group resamples.

    Each replicate samples the observed groups uniformly with replacement. All
    items belonging to a selected group are concatenated, with multiplicity,
    and the identical item indices are applied to A and B.
    """

    a = np.asarray(values_a)
    b = np.asarray(values_b)
    raw_groups = np.asarray(groups)
    raw_ids_a = np.asarray(item_ids_a)
    raw_ids_b = np.asarray(item_ids_b)
    if pd.isna(np.asarray(groups, dtype=object)).any():
        raise ValueError("groups must not contain missing values")
    if pd.isna(np.asarray(item_ids_a, dtype=object)).any() or pd.isna(
        np.asarray(item_ids_b, dtype=object)
    ).any():
        raise ValueError("item IDs must not contain missing values")
    group_values = raw_groups.astype(str)
    ids_a = raw_ids_a.astype(str)
    ids_b = raw_ids_b.astype(str)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("values_a and values_b must have an item dimension")
    if a.shape[0] != b.shape[0] or a.shape[0] != group_values.shape[0]:
        raise ValueError("A, B, and groups must contain the same number of items")
    if ids_a.ndim != 1 or ids_b.ndim != 1:
        raise ValueError("item IDs must be one-dimensional")
    if ids_a.shape[0] != a.shape[0] or ids_b.shape[0] != b.shape[0]:
        raise ValueError("item IDs must align with their corresponding values")
    if np.unique(ids_a).size != ids_a.size or np.unique(ids_b).size != ids_b.size:
        raise ValueError("item IDs must be unique within each paired sample")
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("paired samples must contain item IDs in identical order")
    if group_values.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    if a.shape[0] == 0:
        raise ValueError("bootstrap requires at least one item")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    stat = statistic or (lambda values: float(np.mean(values)))
    observed_a = float(stat(a))
    observed_b = float(stat(b))
    if not np.isfinite([observed_a, observed_b]).all():
        raise ValueError("statistic must be finite on the observed samples")

    unique_groups = pd.unique(group_values)
    if unique_groups.size < 2:
        raise ValueError("paired group bootstrap requires at least two groups")
    indices_by_group = {
        group: np.flatnonzero(group_values == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for replicate in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate(
            [indices_by_group[group] for group in sampled_groups]
        )
        delta = float(stat(a[sampled_indices])) - float(stat(b[sampled_indices]))
        if not np.isfinite(delta):
            raise ValueError(f"statistic is non-finite in bootstrap replicate {replicate}")
        samples[replicate] = delta

    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapDelta(
        observed_a=observed_a,
        observed_b=observed_b,
        observed_delta=observed_a - observed_b,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        standard_error=float(samples.std(ddof=1)) if n_bootstrap > 1 else 0.0,
        confidence_level=float(confidence_level),
        n_bootstrap=int(n_bootstrap),
        n_groups=int(unique_groups.size),
        seed=int(seed),
        samples=samples,
    )


def _validated_oof_probabilities(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...], NDArray[np.float64]]:
    if not isinstance(oof, pd.DataFrame):
        raise TypeError("oof must be a pandas DataFrame")
    if oof.empty:
        raise ValueError("OOF table must not be empty")
    if "item_id" not in oof.columns:
        raise ValueError("OOF table must contain item_id")
    if oof["item_id"].isna().any():
        raise ValueError("OOF table must not contain missing item_id values")
    if oof["item_id"].astype(str).duplicated().any():
        raise ValueError("OOF table must contain exactly one row per item_id")
    names = tuple(str(label) for label in labels)
    if not names or len(set(names)) != len(names):
        raise ValueError("labels must be non-empty and unique")
    probability_columns = [f"prob__{label}" for label in names]
    missing = [column for column in probability_columns if column not in oof.columns]
    if missing:
        raise ValueError(f"OOF table is missing probability columns: {missing}")
    probability = oof[probability_columns].to_numpy(dtype=np.float64)
    _validate_unit_interval(probability, name="probabilities")
    return oof.copy(), names, probability


def _validate_unit_interval(values: NDArray[np.float64], *, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{name} must lie in [0, 1]")


def _validated_binary_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> NDArray[np.int64]:
    """Validate raw values before casting so fractional labels cannot truncate."""

    raw = frame[list(columns)].to_numpy()
    if not np.isin(raw, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return raw.astype(np.int64)


def _resolve_multilabel_thresholds(
    frame: pd.DataFrame,
    *,
    threshold: float | None,
) -> NDArray[np.float64]:
    """Resolve a global threshold or the serialized outer-fold thresholds."""

    if threshold is None:
        if "threshold" not in frame.columns:
            raise ValueError(
                "supply threshold or include a row-wise threshold column in the OOF table"
            )
        values = frame["threshold"].to_numpy(dtype=np.float64)
    else:
        value = float(threshold)
        values = np.full(len(frame), value, dtype=np.float64)
        if "threshold" in frame.columns:
            stored = frame["threshold"].to_numpy(dtype=np.float64)
            if not np.allclose(stored, values, atol=0.0, rtol=0.0):
                raise ValueError(
                    "OOF threshold column disagrees with the supplied global threshold"
                )
    if not np.isfinite(values).all() or ((values <= 0.0) | (values >= 1.0)).any():
        raise ValueError("threshold values must lie strictly between 0 and 1")
    return values


def _multiclass_loss_from_arrays(
    true_index: NDArray[np.int64], probability: NDArray[np.float64]
) -> NDArray[np.float64]:
    true_probability = np.clip(
        probability[np.arange(len(true_index)), true_index],
        PROBABILITY_CLIP,
        1.0,
    )
    return -np.log2(true_probability)


def _multilabel_loss_from_arrays(
    truth: NDArray[np.int64], probability: NDArray[np.float64]
) -> NDArray[np.float64]:
    clipped = np.clip(probability, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    cell_loss = -(
        truth * np.log2(clipped) + (1 - truth) * np.log2(1.0 - clipped)
    )
    return cell_loss.mean(axis=1)


def _binary_log_loss_bits(truth: ArrayLike, probability: ArrayLike) -> float:
    y = np.asarray(truth, dtype=int)
    p = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(p, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return float(-np.mean(y * np.log2(clipped) + (1 - y) * np.log2(1 - clipped)))


def _average_precision_or_zero(truth: ArrayLike, probability: ArrayLike) -> float:
    y = np.asarray(truth, dtype=int)
    if int(y.sum()) == 0:
        return 0.0
    return float(average_precision_score(y, np.asarray(probability, dtype=float)))


__all__ = [
    "BootstrapDelta",
    "MetricReconstruction",
    "expected_calibration_error",
    "multiclass_itemwise_log_loss_bits",
    "multilabel_itemwise_log_loss_bits",
    "paired_group_bootstrap_delta",
    "reconstruct_multiclass",
    "reconstruct_multiclass_metrics",
    "reconstruct_multilabel",
    "reconstruct_multilabel_metrics",
    "reliability_table",
    "summarize_multiclass_predictions",
]
