from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frozen_emotion_spaces.counterfactual import (
    PILOT_FORMAT,
    _fit_space_transform,
    _inverse_squared_knn_predict,
    _sample_complete_groups,
    run_counterfactual_pilot,
    validate_counterfactual_pilot,
)
from frozen_emotion_spaces.experiment_c import run_crowd_representation_probe
from frozen_emotion_spaces.counterfactual_index import (
    validate_counterfactual_index,
    write_counterfactual_index,
)
from frozen_emotion_spaces.counterfactual_observed import (
    validate_observed_counterfactual_analysis,
    write_observed_counterfactual_analysis,
)


def _data_and_splits() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], pd.DataFrame, pd.DataFrame
]:
    names = ("red", "green", "blue")
    item_ids = np.asarray([f"design-{index:03d}" for index in range(45)])
    targets = np.tile(np.asarray(names), 15)
    rng = np.random.default_rng(42)
    features = rng.normal(scale=0.35, size=(45, 3))
    for class_index, name in enumerate(names):
        features[targets == name, class_index] += 3.0
    outer_assignment = np.repeat(np.arange(3), 15)
    groups = np.asarray([f"group-{index:03d}" for index in range(45)])
    outer = pd.DataFrame(
        {"item_id": item_ids, "group_id": groups, "test_fold": outer_assignment}
    )
    inner_rows: list[dict[str, object]] = []
    for outer_fold in range(3):
        train_indices = np.flatnonzero(outer_assignment != outer_fold)
        for position, item_index in enumerate(train_indices):
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "item_id": item_ids[item_index],
                    "group_id": groups[item_index],
                    "validation_fold": (position // 3) % 3,
                }
            )
    return item_ids, targets, features, names, outer, pd.DataFrame(inner_rows)


def test_counterfactual_A_pilot_is_deterministic_and_atomic(tmp_path: Path) -> None:
    item_ids, targets, features, names, outer, inner = _data_and_splits()
    source = run_crowd_representation_probe(
        tmp_path / "A-source",
        representation="A",
        appraisals=features,
        appraisal_names=("a", "b", "c"),
        y=targets,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        C_grid=(0.1,),
        selection_metric="log_loss",
    )
    first = run_counterfactual_pilot(
        tmp_path / "pilot-one",
        source_run=source.directory,
        features=features,
        item_ids=item_ids,
        outer_folds=outer,
        space="A_STANDARDIZED",
        n_sites=3,
        n_constellations_per_fold=3,
        n_repetitions=2,
        max_samples_per_cell=4,
        seed=123,
    )
    second = run_counterfactual_pilot(
        tmp_path / "pilot-two",
        source_run=source.directory,
        features=features,
        item_ids=item_ids,
        outer_folds=outer,
        space="A_STANDARDIZED",
        n_sites=3,
        n_constellations_per_fold=3,
        n_repetitions=2,
        max_samples_per_cell=4,
        seed=123,
    )

    assert first.metadata["pilot_format"] == PILOT_FORMAT
    assert len(first.constellations) == 9
    assert len(first.learnability) == 36
    assert len(first.regressions) == 18
    pd.testing.assert_frame_equal(first.constellations, second.constellations)
    pd.testing.assert_frame_equal(first.learnability, second.learnability)
    pd.testing.assert_frame_equal(first.regressions, second.regressions)
    assert validate_counterfactual_pilot(first.directory).metadata == first.metadata
    observed = write_observed_counterfactual_analysis(
        tmp_path / "observed-vs-pilot",
        pilot_directory=first.directory,
        source_run=source.directory,
        features=features,
        item_ids=item_ids,
        outer_folds=outer,
    )
    assert len(observed.scores) == 3
    assert validate_observed_counterfactual_analysis(
        observed.directory
    ).metadata == observed.metadata
    assert observed.scores["contrast_favorable_percentile"].between(0, 1).all()
    assert (first.constellations["representation_sum"] >= 0).all()
    assert set(first.learnability["learner"]) == {
        "approximate_prototype",
        "knn_inverse_squared",
    }
    index_path = tmp_path / "counterfactual-index.json"
    rows = write_counterfactual_index(index_path, runs_root=tmp_path)
    assert len(rows) == 2
    assert validate_counterfactual_index(index_path, runs_root=tmp_path) == rows

    grouped = run_counterfactual_pilot(
        tmp_path / "pilot-grouped",
        source_run=source.directory,
        features=features,
        item_ids=item_ids,
        outer_folds=outer,
        space="A_STANDARDIZED",
        n_sites=3,
        n_constellations_per_fold=2,
        n_repetitions=2,
        sampling_scheme="fixed_group_budget",
        sample_group_budget=8,
        seed=321,
    )
    assert (grouped.learnability["n_sample_groups"] == 8).all()
    assert (grouped.regressions["regression_model"] == "C_R_plus_sample_items").all()
    assert set(grouped.regressions["status"]) <= {
        "insufficient_residual_df", "constant_outcome", "rank_deficient"
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_counterfactual_pilot(
            first.directory,
            source_run=source.directory,
            features=features,
            item_ids=item_ids,
            outer_folds=outer,
            space="A_STANDARDIZED",
            n_sites=3,
            n_constellations_per_fold=1,
            n_repetitions=1,
        )


def test_space_transform_is_train_only_and_H_PCA_has_locked_width() -> None:
    train = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    test = np.asarray([[1000.0, -1000.0]])
    transformed_train, transformed_test, record = _fit_space_transform(
        train, test, space="A_STANDARDIZED", pca_components=None
    )
    assert np.allclose(record["scaler_mean"], [2.0, 3.0])
    assert np.allclose(transformed_train.mean(axis=0), 0.0)
    assert transformed_test.shape == (1, 2)
    assert record["pca_components"].shape == (0, 2)

    larger_train = np.column_stack((train, [0.0, 1.0, 0.0]))
    larger_test = np.asarray([[10.0, 11.0, 1.0]])
    reduced_train, reduced_test, pca_record = _fit_space_transform(
        larger_train, larger_test, space="H_PCA", pca_components=2
    )
    assert reduced_train.shape == (3, 2)
    assert reduced_test.shape == (1, 2)
    assert pca_record["pca_components"].shape == (2, 3)


def test_inverse_squared_knn_handles_exact_matches_and_is_deterministic() -> None:
    train = np.asarray([[0.0], [2.0], [10.0]])
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    test = np.asarray([[0.0], [1.9], [9.8]])
    prediction = _inverse_squared_knn_predict(
        train, labels, test, n_classes=3, n_neighbors=2
    )
    assert prediction.tolist() == [0, 1, 2]
    boundary_tie = _inverse_squared_knn_predict(
        np.asarray([[-1.0], [1.0], [3.0]]),
        np.asarray([2, 1, 0], dtype=np.int64),
        np.asarray([[0.0]]),
        n_classes=3,
        n_neighbors=1,
    )
    assert boundary_tie.tolist() == [2]
    with pytest.raises(ValueError, match="n_neighbors"):
        _inverse_squared_knn_predict(
            train, labels, test, n_classes=3, n_neighbors=4
        )


def test_complete_group_sampling_keeps_every_member_and_all_cells() -> None:
    assignments = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    groups = np.asarray(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    selected = _sample_complete_groups(
        assignments,
        groups,
        n_sites=3,
        group_budget=2,
        rng=np.random.default_rng(9),
    )
    selected_groups = set(groups[selected])
    assert len(selected_groups) == 2
    assert set(np.flatnonzero(np.isin(groups, list(selected_groups)))) == set(selected)
    assert set(assignments[selected]) == {0, 1, 2}
