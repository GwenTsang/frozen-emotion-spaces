from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from frozen_emotion_spaces.crowd_data import (
    APPRAISAL_NAMES,
    CROWD_EMOTIONS,
    build_crowd_manifests,
)
from frozen_emotion_spaces.metrics import reconstruct_multilabel_metrics
from frozen_emotion_spaces.probes import (
    BlockTransformer,
    TransformedMulticlassLogistic,
    TransformedMultilabelLogistic,
    make_dense_multiclass_factory,
    make_dense_multilabel_factory,
    run_nested_multiclass_oof,
    run_nested_multilabel_oof,
    select_multiclass_C,
    select_multiclass_C_block_multiplier,
    select_multilabel_C_threshold,
)
from frozen_emotion_spaces.splits import read_split_bundle


CROWD_ARCHIVE = Path(
    os.environ.get("FES_CROWD_ARCHIVE", "datasets/crowd-enVent2023.zip")
)
PRESERVED_SPLITS = Path(__file__).resolve().parents[1] / "splits"


def test_block_transformer_uses_train_statistics_and_block_multipliers() -> None:
    train = np.array([[0.0, 10.0], [2.0, 14.0]])
    test = np.array([[3.0, 16.0]])

    transformer = BlockTransformer(
        block_dims=(1, 1),
        block_multipliers=(2.0, 0.5),
    ).fit(train)

    np.testing.assert_allclose(
        transformer.transform(train),
        np.array([[-2.0, -0.5], [2.0, 0.5]]),
    )
    # This is non-zero precisely because test statistics were not fitted.
    np.testing.assert_allclose(transformer.transform(test), np.array([[4.0, 1.0]]))


def test_block_transformer_rejects_inconsistent_dimensions() -> None:
    with pytest.raises(ValueError, match="sum"):
        BlockTransformer(block_dims=(1, 1)).fit(np.ones((4, 3)))


def test_block_transformer_zero_multiplier_suppresses_a_block() -> None:
    X = np.array([[0.0, 10.0], [2.0, 14.0]])

    transformed = BlockTransformer(
        block_dims=(1, 1),
        block_multipliers=(1.0, 0.0),
    ).fit_transform(X)

    np.testing.assert_allclose(transformed[:, 1], 0.0)


def test_estimators_are_sklearn_cloneable_and_clones_are_unfitted() -> None:
    transformer = BlockTransformer(
        block_dims=(1, 2),
        block_multipliers=(0.3, 1.0),
    ).fit(np.arange(12, dtype=float).reshape(4, 3))
    estimator = TransformedMultilabelLogistic(
        C=0.1,
        block_dims=(1, 2),
        block_multipliers=(0.3, 1.0),
    ).fit(
        np.arange(18, dtype=float).reshape(6, 3),
        np.column_stack([np.arange(6) % 2, (np.arange(6) + 1) % 2]),
    )

    transformer_clone = clone(transformer)
    estimator_clone = clone(estimator)
    with pytest.raises(NotFittedError):
        transformer_clone.transform(np.ones((1, 3)))
    with pytest.raises(NotFittedError):
        estimator_clone.predict_proba(np.ones((1, 3)))


def test_multilabel_logistic_handles_constant_labels() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1]])

    model = TransformedMultilabelLogistic(C=1.0).fit(X, y)
    probability = model.predict_proba(X)

    assert probability.shape == y.shape
    assert np.all((probability > 0.0) & (probability < 1.0))
    np.testing.assert_allclose(probability[:, 1], model.probability_clip)
    np.testing.assert_allclose(probability[:, 2], 1.0 - model.probability_clip)
    assert probability[0, 0] < probability[-1, 0]


def test_multilabel_logistic_preserves_one_label_shape() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([[0], [0], [1], [1]])

    probability = TransformedMultilabelLogistic().fit(X, y).predict_proba(X)

    assert probability.shape == (4, 1)


def test_multilabel_logistic_revalidates_mutable_threshold_at_prediction() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([[0], [0], [1], [1]])
    model = TransformedMultilabelLogistic().fit(X, y)
    model.threshold = 2.0

    with pytest.raises(ValueError, match="threshold"):
        model.predict(X)


def test_factory_keeps_blocks_separate() -> None:
    factory = make_dense_multilabel_factory(
        block_dims=(1, 2),
        block_multipliers=(0.3, 1.0),
    )
    estimator = factory(0.1)

    assert estimator.C == 0.1
    assert estimator.block_dims == (1, 2)
    assert estimator.block_multipliers == (0.3, 1.0)


def test_multiclass_estimator_is_cloneable_and_preserves_class_order() -> None:
    class_names = ("zeta", "alpha", "middle")
    X = np.tile(np.eye(3), (4, 1)) + np.arange(12)[:, None] * 0.001
    y = np.tile(np.asarray(class_names), 4)
    estimator = TransformedMulticlassLogistic(
        class_names=class_names,
        C=1.0,
    ).fit(X, y)

    probability = estimator.predict_proba(X)
    assert tuple(estimator.classes_) == class_names
    assert probability.shape == (12, 3)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    with pytest.raises(NotFittedError):
        clone(estimator).predict_proba(X)


def test_multiclass_factory_keeps_balanced_weight_explicit() -> None:
    estimator = make_dense_multiclass_factory(
        class_names=("a", "b", "c"),
        class_weight="balanced",
    )(0.1)

    assert estimator.C == 0.1
    assert estimator.class_weight == "balanced"


class _IndexedMulticlassProbabilityEstimator:
    def __init__(self, probability: np.ndarray) -> None:
        self.probability = probability

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.probability[X[:, 0].astype(int)]


def test_multiclass_selection_tie_breaks_to_smaller_C() -> None:
    class_names = ("a", "b", "c")
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = np.tile(np.asarray(class_names), 6)
    probability = np.full((18, 3), 0.05)
    probability[np.arange(18), np.arange(18) % 3] = 0.90

    selection = select_multiclass_C(
        X,
        y,
        validation_folds=np.repeat(np.arange(3), 6),
        groups=np.asarray([f"g-{index}" for index in range(18)]),
        class_names=class_names,
        estimator_factory=lambda C: _IndexedMulticlassProbabilityEstimator(
            probability
        ),
        C_grid=(1.0, 0.1),
    )

    assert selection.C == 0.1
    assert selection.inner_oof_probabilities.shape == (18, 3)


def _multiclass_selection_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X = np.column_stack([np.arange(18, dtype=float), np.zeros(18)])
    y = np.tile(np.asarray(["a", "b", "c"]), 6)
    folds = np.repeat(np.arange(3), 6)
    groups = np.asarray([f"joint-{index}" for index in range(18)])
    return X, y, folds, groups


def test_joint_multiclass_selection_fixes_single_block_multiplier_to_one() -> None:
    X, y, folds, groups = _multiclass_selection_fixture()
    probability = np.full((18, 3), 0.05)
    probability[np.arange(18), np.arange(18) % 3] = 0.90
    calls: list[tuple[float, float]] = []

    def factory(C: float, multiplier: float):
        calls.append((C, multiplier))
        return _IndexedMulticlassProbabilityEstimator(probability)

    selection = select_multiclass_C_block_multiplier(
        X[:, :1],
        y,
        validation_folds=folds,
        groups=groups,
        class_names=("a", "b", "c"),
        block_dims=(1,),
        estimator_factory=factory,
        C_grid=(1.0, 0.1),
        block_multiplier_grid=(0.1, 3.0, 10.0),
    )

    assert selection.C == 0.1
    assert selection.block_multiplier == 1.0
    assert {multiplier for _, multiplier in calls} == {1.0}
    assert selection.inner_oof_probabilities.shape == (18, 3)


def test_joint_multiclass_selection_two_block_tie_break_is_deterministic() -> None:
    X, y, folds, groups = _multiclass_selection_fixture()
    probability = np.full((18, 3), 0.05)
    probability[np.arange(18), np.arange(18) % 3] = 0.90

    selection = select_multiclass_C_block_multiplier(
        X,
        y,
        validation_folds=folds,
        groups=groups,
        class_names=("a", "b", "c"),
        block_dims=(1, 1),
        estimator_factory=lambda C, multiplier: (
            _IndexedMulticlassProbabilityEstimator(probability)
        ),
        C_grid=(1.0, 0.1),
        block_multiplier_grid=(3.0, 0.3, 1.0),
    )

    # Identical scores retain the existing smaller-C tie-break, then prefer
    # the least rescaled first block.
    assert selection.C == 0.1
    assert selection.block_multiplier == 1.0


def test_joint_multiclass_default_factory_preserves_nonlexical_class_order() -> None:
    rng = np.random.default_rng(83)
    class_names = ("zeta", "alpha", "middle")
    y = np.tile(np.asarray(class_names), 6)
    class_index = np.tile(np.arange(3), 6)
    X = 4.0 * np.eye(3)[class_index] + rng.normal(scale=0.01, size=(18, 3))

    selection = select_multiclass_C_block_multiplier(
        X,
        y,
        validation_folds=np.repeat(np.arange(3), 6),
        groups=np.asarray([f"ordered-{index}" for index in range(18)]),
        class_names=class_names,
        block_dims=(2, 1),
        C_grid=(10.0,),
        block_multiplier_grid=(0.3,),
    )

    probability = selection.inner_oof_probabilities
    prediction = np.asarray(class_names)[np.argmax(probability, axis=1)]
    assert selection.block_multiplier == 0.3
    np.testing.assert_array_equal(prediction, y)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_joint_multiclass_selection_objectives_can_choose_different_pairs() -> None:
    X, y, folds, groups = _multiclass_selection_fixture()
    perfectly_ranked = np.full((18, 3), 0.33)
    perfectly_ranked[np.arange(18), np.arange(18) % 3] = 0.34
    low_loss_one_error = np.full((18, 3), 0.01)
    low_loss_one_error[np.arange(18), np.arange(18) % 3] = 0.98
    low_loss_one_error[0] = np.asarray([0.01, 0.98, 0.01])
    by_multiplier = {0.3: perfectly_ranked, 3.0: low_loss_one_error}

    common = {
        "validation_folds": folds,
        "groups": groups,
        "class_names": ("a", "b", "c"),
        "block_dims": (1, 1),
        "estimator_factory": lambda C, multiplier: (
            _IndexedMulticlassProbabilityEstimator(by_multiplier[multiplier])
        ),
        "C_grid": (1.0,),
        "block_multiplier_grid": (0.3, 3.0),
    }
    information = select_multiclass_C_block_multiplier(
        X,
        y,
        selection_metric="log_loss",
        **common,
    )
    classification = select_multiclass_C_block_multiplier(
        X,
        y,
        selection_metric="macro_f1",
        **common,
    )

    assert information.block_multiplier == 3.0
    assert classification.block_multiplier == 0.3


def test_joint_multiclass_selection_rejects_group_leakage() -> None:
    X, y, folds, groups = _multiclass_selection_fixture()
    groups[6] = groups[0]

    with pytest.raises(ValueError, match="group leakage"):
        select_multiclass_C_block_multiplier(
            X,
            y,
            validation_folds=folds,
            groups=groups,
            class_names=("a", "b", "c"),
            block_dims=(1, 1),
            C_grid=(1.0,),
            block_multiplier_grid=(1.0,),
        )


def test_joint_multiclass_selection_validates_two_block_protocol() -> None:
    X, y, folds, groups = _multiclass_selection_fixture()

    with pytest.raises(ValueError, match="one or two blocks"):
        select_multiclass_C_block_multiplier(
            X,
            y,
            validation_folds=folds,
            groups=groups,
            class_names=("a", "b", "c"),
            block_dims=(1, 1, 0),
            C_grid=(1.0,),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        select_multiclass_C_block_multiplier(
            X,
            y,
            validation_folds=folds,
            groups=groups,
            class_names=("a", "b", "c"),
            block_dims=(1, 1),
            C_grid=(1.0,),
            block_multiplier_grid=(0.0, 1.0),
        )


def test_multiclass_selection_rejects_missing_inner_training_class() -> None:
    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.asarray(["a", "b", "c"] * 3)

    with pytest.raises(ValueError, match="missing training classes"):
        select_multiclass_C(
            X,
            y,
            validation_folds=np.asarray([0, 1, 2] * 3),
            groups=np.asarray([f"g-{index}" for index in range(9)]),
            class_names=("a", "b", "c"),
            C_grid=(1.0,),
        )


def _synthetic_multiclass_nested_problem():
    rng = np.random.default_rng(31)
    class_names = ("zeta", "alpha", "middle")
    y = np.tile(np.asarray(class_names), 9)
    class_index = np.tile(np.arange(3), 9)
    X = np.eye(3)[class_index] + rng.normal(scale=0.1, size=(27, 3))
    item_ids = np.asarray([f"multi-{index:02d}" for index in range(27)])
    groups = np.asarray([f"group-{index:02d}" for index in range(27)])
    outer = np.repeat(np.arange(3), 9)
    inner_rows = []
    for outer_fold in range(3):
        train_indices = np.flatnonzero(outer != outer_fold)
        for index in train_indices:
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "item_id": item_ids[index],
                    "group_id": groups[index],
                    "validation_fold": (index // 3) % 3,
                }
            )
    outer_table = pd.DataFrame(
        {
            "item_id": item_ids,
            "group_id": groups,
            "test_fold": outer,
        }
    )
    return X, y, item_ids, class_names, outer_table, pd.DataFrame(inner_rows)


@pytest.mark.parametrize("selection_metric", ["log_loss", "macro_f1"])
def test_nested_multiclass_runner_aligns_and_covers_oof(selection_metric: str) -> None:
    X, y, item_ids, names, outer, inner = _synthetic_multiclass_nested_problem()
    result = run_nested_multiclass_oof(
        X,
        y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer.sample(frac=1, random_state=7),
        inner_folds=inner.sample(frac=1, random_state=9),
        C_grid=(0.1, 1.0),
        selection_metric=selection_metric,
    )

    assert result.oof["item_id"].nunique() == len(item_ids)
    assert len(result.oof) == len(item_ids)
    assert len(result.selections) == 3
    assert list(result.oof.filter(like="prob__").columns) == [
        f"prob__{name}" for name in names
    ]
    np.testing.assert_allclose(
        result.oof.filter(like="prob__").sum(axis=1),
        1.0,
    )


@pytest.mark.skipif(
    not (CROWD_ARCHIVE.exists() and PRESERVED_SPLITS.exists()),
    reason="released crowd archive or preserved split evidence unavailable",
)
def test_real_crowd_multiclass_runner_covers_preserved_five_by_three_splits() -> None:
    crowd = build_crowd_manifests(CROWD_ARCHIVE)
    splits = read_split_bundle(PRESERVED_SPLITS)
    generation = crowd.generation
    result = run_nested_multiclass_oof(
        generation.loc[:, APPRAISAL_NAMES].to_numpy(dtype=np.float64),
        generation["y_writer"].to_numpy(),
        item_ids=generation["item_id"].to_numpy(),
        class_names=CROWD_EMOTIONS,
        outer_folds=splits.crowd_full_outer,
        inner_folds=splits.crowd_full_inner,
        C_grid=(0.1,),
    )

    assert len(result.oof) == 6_600
    assert result.oof["item_id"].nunique() == 6_600
    assert len(result.selections) == 5
    np.testing.assert_allclose(result.oof.filter(like="prob__").sum(axis=1), 1.0)


@pytest.mark.parametrize("selection_metric", ["macro_f1", "log_loss"])
def test_inner_selection_returns_complete_deterministic_oof(
    selection_metric: str,
) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(18, 3))
    y = np.column_stack(
        [
            (X[:, 0] + 0.2 * X[:, 1] > 0).astype(int),
            (X[:, 2] > -0.2).astype(int),
        ]
    )
    validation_folds = np.tile(np.arange(3), 6)
    groups = np.asarray([f"group-{index}" for index in range(len(X))])

    first = select_multilabel_C_threshold(
        X,
        y,
        validation_folds=validation_folds,
        groups=groups,
        C_grid=(0.1, 1.0),
        threshold_grid=(0.3, 0.5, 0.7),
        selection_metric=selection_metric,
    )
    second = select_multilabel_C_threshold(
        X,
        y,
        validation_folds=validation_folds,
        groups=groups,
        C_grid=(0.1, 1.0),
        threshold_grid=(0.3, 0.5, 0.7),
        selection_metric=selection_metric,
    )

    assert first.C in {0.1, 1.0}
    assert first.threshold in {0.3, 0.5, 0.7}
    assert first.inner_oof_probabilities.shape == y.shape
    assert np.isfinite(first.inner_oof_probabilities).all()
    assert first.C == second.C
    assert first.threshold == second.threshold
    np.testing.assert_allclose(
        first.inner_oof_probabilities,
        second.inner_oof_probabilities,
    )


def test_inner_selection_rejects_group_leakage() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.column_stack([np.arange(6) % 2, (np.arange(6) + 1) % 2])
    folds = np.array([0, 0, 1, 1, 2, 2])
    groups = np.array(["leak", "a", "leak", "b", "c", "d"])

    with pytest.raises(ValueError, match="group leakage"):
        select_multilabel_C_threshold(
            X,
            y,
            validation_folds=folds,
            groups=groups,
            C_grid=(1.0,),
            threshold_grid=(0.5,),
        )


def test_inner_selection_requires_nonmissing_groups() -> None:
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([[0], [1], [0], [1]])
    folds = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="missing"):
        select_multilabel_C_threshold(
            X,
            y,
            validation_folds=folds,
            groups=np.array(["a", None, "c", "d"], dtype=object),
            C_grid=(1.0,),
            threshold_grid=(0.5,),
        )


class _IndexedProbabilityEstimator:
    def __init__(self, probability_by_C: dict[float, np.ndarray], C: float) -> None:
        self.probability_by_C = probability_by_C
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_IndexedProbabilityEstimator":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        indices = X[:, 0].astype(int)
        return self.probability_by_C[self.C][indices]


def test_selection_objectives_can_choose_different_C_values() -> None:
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.array([[1], [1], [0], [0]])
    probabilities = {
        0.1: np.array([[0.51], [0.51], [0.49], [0.49]]),
        1.0: np.array([[0.90], [0.40], [0.10], [0.10]]),
    }
    factory = lambda C: _IndexedProbabilityEstimator(probabilities, C)
    common = {
        "validation_folds": np.array([0, 1, 0, 1]),
        "groups": np.array(["a", "b", "c", "d"]),
        "estimator_factory": factory,
        "C_grid": (0.1, 1.0),
        "threshold_grid": (0.5,),
    }

    macro = select_multilabel_C_threshold(
        X,
        y,
        selection_metric="macro_f1",
        **common,
    )
    information = select_multilabel_C_threshold(
        X,
        y,
        selection_metric="log_loss",
        **common,
    )

    assert macro.C == 0.1
    assert information.C == 1.0


def test_inner_selection_rejects_invalid_factory_probabilities() -> None:
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.array([[1], [1], [0], [0]])
    bad = {1.0: np.full((4, 1), 2.0)}

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        select_multilabel_C_threshold(
            X,
            y,
            validation_folds=np.array([0, 1, 0, 1]),
            groups=np.array(["a", "b", "c", "d"]),
            estimator_factory=lambda C: _IndexedProbabilityEstimator(bad, C),
            C_grid=(1.0,),
            threshold_grid=(0.5,),
        )


def _synthetic_nested_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    pd.DataFrame,
]:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(18, 4))
    y = np.column_stack(
        [
            (X[:, 0] - X[:, 1] > 0).astype(int),
            (X[:, 2] + X[:, 3] > 0).astype(int),
        ]
    )
    item_ids = np.asarray([f"item-{index:02d}" for index in range(len(X))])
    groups = np.asarray([f"group-{index:02d}" for index in range(len(X))])
    outer = np.arange(len(X)) % 3
    inner_rows: list[dict[str, object]] = []
    for outer_fold in range(3):
        train_indices = np.flatnonzero(outer != outer_fold)
        for position, index in enumerate(train_indices):
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "item_id": item_ids[index],
                    "validation_fold": position % 2,
                }
            )
    return X, y, item_ids, groups.tolist(), outer, pd.DataFrame(inner_rows)


def test_nested_runner_covers_group_disjoint_oof_once() -> None:
    X, y, item_ids, groups, outer, inner = _synthetic_nested_problem()

    result = run_nested_multilabel_oof(
        X,
        y,
        item_ids=item_ids,
        label_names=("joy", "sadness"),
        outer_folds=outer,
        inner_folds=inner,
        groups=groups,
        C_grid=(0.1, 1.0),
        threshold_grid=(0.3, 0.5, 0.7),
    )

    assert len(result.oof) == len(item_ids)
    assert result.oof["item_id"].nunique() == len(item_ids)
    assert set(result.oof["item_id"]) == set(item_ids)
    assert len(result.selections) == 3
    assert set(result.selections["selection_metric"]) == {"macro_f1"}
    assert {
        "item_id",
        "outer_fold",
        "group_id",
        "threshold",
        "y_true__joy",
        "prob__joy",
        "pred__joy",
        "y_true__sadness",
        "prob__sadness",
        "pred__sadness",
    }.issubset(result.oof.columns)
    assert result.oof[["prob__joy", "prob__sadness"]].to_numpy().min() > 0
    assert result.oof[["prob__joy", "prob__sadness"]].to_numpy().max() < 1
    for row in result.selections.itertuples(index=False):
        fold_rows = result.oof[result.oof["outer_fold"] == row.outer_fold]
        assert fold_rows["threshold"].nunique() == 1
        assert fold_rows["threshold"].iloc[0] == pytest.approx(row.threshold)
        expected = (
            fold_rows[["prob__joy", "prob__sadness"]].to_numpy() >= row.threshold
        ).astype(int)
        observed = fold_rows[["pred__joy", "pred__sadness"]].to_numpy()
        np.testing.assert_array_equal(observed, expected)

    reconstructed = reconstruct_multilabel_metrics(
        result.oof,
        labels=("joy", "sadness"),
    )
    assert reconstructed.overall["n_items"] == len(item_ids)


def test_nested_runner_aligns_shuffled_split_tables_by_item_id() -> None:
    X, y, item_ids, groups, outer, inner = _synthetic_nested_problem()
    outer_table = pd.DataFrame({"item_id": item_ids, "test_fold": outer}).sample(
        frac=1.0,
        random_state=3,
    )
    shuffled_inner = inner.sample(frac=1.0, random_state=5)

    result = run_nested_multilabel_oof(
        X,
        y,
        item_ids=item_ids,
        label_names=("joy", "sadness"),
        outer_folds=outer_table,
        inner_folds=shuffled_inner,
        groups=groups,
        C_grid=(1.0,),
        threshold_grid=(0.5,),
    )

    assert set(result.oof["item_id"]) == set(item_ids)


def test_nested_runner_infers_and_cross_checks_groups_from_split_tables() -> None:
    X, y, item_ids, groups, outer, inner = _synthetic_nested_problem()
    group_by_id = dict(zip(item_ids, groups, strict=True))
    outer_table = pd.DataFrame(
        {"item_id": item_ids, "test_fold": outer, "group_id": groups}
    )
    inner = inner.assign(group_id=inner["item_id"].map(group_by_id))

    result = run_nested_multilabel_oof(
        X,
        y,
        item_ids=item_ids,
        label_names=("joy", "sadness"),
        outer_folds=outer_table,
        inner_folds=inner,
        C_grid=(1.0,),
        threshold_grid=(0.5,),
    )
    assert set(result.oof["group_id"]) == set(groups)

    mismatched = inner.copy()
    mismatched.loc[mismatched.index[0], "group_id"] = "wrong-group"
    with pytest.raises(ValueError, match="multiple groups|group mismatch"):
        run_nested_multilabel_oof(
            X,
            y,
            item_ids=item_ids,
            label_names=("joy", "sadness"),
            outer_folds=outer_table,
            inner_folds=mismatched,
            C_grid=(1.0,),
            threshold_grid=(0.5,),
        )


def test_nested_runner_rejects_missing_group_ids_in_split_tables() -> None:
    X, y, item_ids, groups, outer, inner = _synthetic_nested_problem()
    group_by_id = dict(zip(item_ids, groups, strict=True))
    outer_table = pd.DataFrame(
        {"item_id": item_ids, "test_fold": outer, "group_id": groups}
    )
    inner_with_groups = inner.assign(group_id=inner["item_id"].map(group_by_id))

    missing_outer = outer_table.copy()
    missing_outer.loc[0, "group_id"] = None
    with pytest.raises(ValueError, match="outer_folds contains missing group_id"):
        run_nested_multilabel_oof(
            X,
            y,
            item_ids=item_ids,
            label_names=("joy", "sadness"),
            outer_folds=missing_outer,
            inner_folds=inner_with_groups,
            C_grid=(1.0,),
            threshold_grid=(0.5,),
        )

    missing_inner = inner_with_groups.copy()
    missing_inner.loc[missing_inner.index[0], "group_id"] = None
    with pytest.raises(ValueError, match="inner_folds contains missing group_id"):
        run_nested_multilabel_oof(
            X,
            y,
            item_ids=item_ids,
            label_names=("joy", "sadness"),
            outer_folds=outer_table,
            inner_folds=missing_inner,
            C_grid=(1.0,),
            threshold_grid=(0.5,),
        )


def test_nested_runner_refuses_missing_group_information() -> None:
    X, y, item_ids, _, outer, inner = _synthetic_nested_problem()

    with pytest.raises(ValueError, match="group IDs are required"):
        run_nested_multilabel_oof(
            X,
            y,
            item_ids=item_ids,
            label_names=("joy", "sadness"),
            outer_folds=outer,
            inner_folds=inner,
            C_grid=(1.0,),
            threshold_grid=(0.5,),
        )


def test_nested_runner_rejects_outer_group_leakage() -> None:
    X, y, item_ids, groups, outer, inner = _synthetic_nested_problem()
    groups[0] = groups[1]

    with pytest.raises(ValueError, match="group leakage in outer fold"):
        run_nested_multilabel_oof(
            X,
            y,
            item_ids=item_ids,
            label_names=("joy", "sadness"),
            outer_folds=outer,
            inner_folds=inner,
            groups=groups,
            C_grid=(1.0,),
            threshold_grid=(0.5,),
        )
