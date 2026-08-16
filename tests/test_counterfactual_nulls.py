from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frozen_emotion_spaces.counterfactual_nulls import (
    MECHANISMS,
    NULL_FORMAT,
    _attempt_cell_centroid_null,
    _attempt_permutation_null,
    _fold_null_draws,
    run_observed_matched_nulls,
    validate_observed_matched_nulls,
)
from frozen_emotion_spaces.experiment_a import _sha256_file
from frozen_emotion_spaces.experiment_c import run_crowd_representation_probe
from frozen_emotion_spaces.geometry import (
    assign_domain_points_to_nearest_sites,
    contrast_representation_scores,
)


def _make_frames(
    item_ids: np.ndarray,
    groups: np.ndarray,
    outer_assignment: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return outer, pd.DataFrame(inner_rows)


def _synthetic_data() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, np.ndarray
]:
    names = ("red", "green", "blue")
    item_ids = np.asarray([f"null-{index:03d}" for index in range(45)])
    targets = np.tile(np.asarray(names), 15)
    rng = np.random.default_rng(42)
    features = rng.normal(scale=0.35, size=(45, 3))
    for class_index, name in enumerate(names):
        features[targets == name, class_index] += 3.0
    outer_assignment = np.repeat(np.arange(3), 15)
    groups = np.asarray([f"group-{index:03d}" for index in range(45)])
    return item_ids, targets, features, names, outer_assignment, groups


def _make_source(
    directory: Path,
    *,
    item_ids: np.ndarray,
    targets: np.ndarray,
    features: np.ndarray,
    names: tuple[str, ...],
    outer_assignment: np.ndarray,
    groups: np.ndarray,
):
    outer, inner = _make_frames(item_ids, groups, outer_assignment)
    appraisal_names = tuple(f"a{index}" for index in range(features.shape[1]))
    source = run_crowd_representation_probe(
        directory,
        representation="A",
        appraisals=features,
        appraisal_names=appraisal_names,
        y=targets,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        C_grid=(0.1,),
        selection_metric="log_loss",
    )
    return source, outer


def test_matched_nulls_are_deterministic_atomic_and_validated(tmp_path: Path) -> None:
    item_ids, targets, features, names, outer_assignment, groups = _synthetic_data()
    source, outer = _make_source(
        tmp_path / "source",
        item_ids=item_ids, targets=targets, features=features, names=names,
        outer_assignment=outer_assignment, groups=groups,
    )
    first = run_observed_matched_nulls(
        tmp_path / "nulls-one",
        source_run=source.directory, features=features, item_ids=item_ids,
        outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=4,
        max_attempts_per_draw=50, seed=777,
    )
    second = run_observed_matched_nulls(
        tmp_path / "nulls-two",
        source_run=source.directory, features=features, item_ids=item_ids,
        outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=4,
        max_attempts_per_draw=50, seed=777,
    )

    assert first.metadata["null_format"] == NULL_FORMAT
    assert first.metadata["status"].endswith("not_confirmatory")
    assert first.metadata["numpy_bit_generator"] == "PCG64"
    assert len(first.draws) == 3 * len(MECHANISMS) * 4
    assert len(first.fold_summaries) == 3 * len(MECHANISMS)
    pd.testing.assert_frame_equal(first.draws, second.draws)
    pd.testing.assert_frame_equal(first.fold_summaries, second.fold_summaries)
    assert first.metadata == second.metadata
    validated = validate_observed_matched_nulls(first.directory)
    assert validated.metadata == first.metadata
    pd.testing.assert_frame_equal(validated.draws, first.draws)

    assert (first.draws["attempts"] >= 1).all()
    assert (first.draws["contrast_sum"] >= 0).all()
    assert (first.draws["representation_sum"] >= 0).all()
    assert first.draws["cell_support_entropy_normalized"].between(0, 1).all()
    summaries = first.fold_summaries
    assert summaries["contrast_p_plus1"].between(0, 1).all()
    assert summaries["representation_p_plus1"].between(0, 1).all()
    assert summaries["joint_domination_fraction"].between(0, 1).all()
    assert (summaries["fidelity_precondition"] == "satisfied").all()
    permutation = first.draws[first.draws["mechanism"] == "label_permutation"]
    assert permutation["initial_site_item_ids_json"].isna().all()
    cell = first.draws[first.draws["mechanism"] == "counterfactual_cell_centroids"]
    assert cell["initial_site_item_ids_json"].notna().all()
    assert (cell["initial_contrast_sum"] >= 0).all()


def test_matched_nulls_refuse_overwrite(tmp_path: Path, monkeypatch) -> None:
    item_ids, targets, features, names, outer_assignment, groups = _synthetic_data()
    source, outer = _make_source(
        tmp_path / "source",
        item_ids=item_ids, targets=targets, features=features, names=names,
        outer_assignment=outer_assignment, groups=groups,
    )
    run_observed_matched_nulls(
        tmp_path / "nulls",
        source_run=source.directory, features=features, item_ids=item_ids,
        outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=2, seed=5,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_observed_matched_nulls(
            tmp_path / "nulls",
            source_run=source.directory, features=features, item_ids=item_ids,
            outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=2, seed=5,
        )

    original_rename = Path.rename

    def concurrent_publish(path: Path, target: Path):
        Path(target).mkdir()
        raise OSError(39, "directory not empty")

    monkeypatch.setattr(Path, "rename", concurrent_publish)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_observed_matched_nulls(
            tmp_path / "raced-null",
            source_run=source.directory, features=features, item_ids=item_ids,
            outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=1, seed=6,
        )
    monkeypatch.setattr(Path, "rename", original_rename)


def test_permutation_null_preserves_supports_exactly() -> None:
    rng_data = np.random.default_rng(7)
    train = rng_data.normal(size=(30, 3))
    encoded = np.repeat(np.arange(3), (12, 10, 8)).astype(np.int64)
    rng = np.random.default_rng(99)
    attempt = None
    for _ in range(100):
        attempt = _attempt_permutation_null(train, encoded, n_classes=3, rng=rng)
        if attempt.rejection is None:
            break
    assert attempt is not None and attempt.rejection is None
    assert attempt.permuted_labels is not None
    np.testing.assert_array_equal(
        np.bincount(attempt.permuted_labels, minlength=3),
        np.bincount(encoded, minlength=3),
    )
    assert not np.array_equal(attempt.permuted_labels, encoded)
    for label in range(3):
        np.testing.assert_allclose(
            attempt.sites[label], train[attempt.permuted_labels == label].mean(axis=0)
        )
    assignments = assign_domain_points_to_nearest_sites(
        attempt.sites, domain_points=train, require_nonempty_cells=False
    )
    assert (np.bincount(assignments, minlength=3) > 0).all()


def test_cell_centroid_null_replaces_sites_with_cell_centroids() -> None:
    rng_data = np.random.default_rng(11)
    train = rng_data.normal(size=(60, 4))
    rng = np.random.default_rng(123)
    attempt = None
    for _ in range(100):
        attempt = _attempt_cell_centroid_null(train, n_sites=3, rng=rng)
        if attempt.rejection is None:
            break
    assert attempt is not None and attempt.rejection is None
    initial_sites = train[attempt.initial_site_indices]
    initial_assignments = assign_domain_points_to_nearest_sites(
        initial_sites, domain_points=train, require_nonempty_cells=False
    )
    for cell in range(3):
        np.testing.assert_allclose(
            attempt.sites[cell],
            train[initial_assignments == cell].mean(axis=0),
        )
    initial_scores = contrast_representation_scores(initial_sites, domain_points=train)
    assert attempt.initial_contrast_sum == pytest.approx(initial_scores.contrast_sum)
    assert attempt.initial_representation_sum == pytest.approx(
        initial_scores.representation_sum
    )
    reinduced = assign_domain_points_to_nearest_sites(
        attempt.sites, domain_points=train, require_nonempty_cells=False
    )
    assert (np.bincount(reinduced, minlength=3) > 0).all()


def test_fold_nulls_never_observe_outer_test(tmp_path: Path) -> None:
    item_ids, targets, features, names, outer_assignment, groups = _synthetic_data()
    source_one, outer_one = _make_source(
        tmp_path / "source-one",
        item_ids=item_ids, targets=targets, features=features, names=names,
        outer_assignment=outer_assignment, groups=groups,
    )
    run_one = run_observed_matched_nulls(
        tmp_path / "nulls-one",
        source_run=source_one.directory, features=features, item_ids=item_ids,
        outer_folds=outer_one, space="A_STANDARDIZED", n_draws_per_fold=4, seed=31,
    )

    # Permute every row of the fold-0 test block: item identity, targets,
    # groups, and features move together, so the mapping stays consistent
    # while every outer-test row of fold 0 changes.
    fold_zero = np.flatnonzero(outer_assignment == 0)
    reversed_zero = fold_zero[::-1]
    ids_two = item_ids.copy()
    targets_two = targets.copy()
    groups_two = groups.copy()
    features_two = features.copy()
    ids_two[fold_zero] = item_ids[reversed_zero]
    targets_two[fold_zero] = targets[reversed_zero]
    groups_two[fold_zero] = groups[reversed_zero]
    features_two[fold_zero] = features[reversed_zero]
    source_two, outer_two = _make_source(
        tmp_path / "source-two",
        item_ids=ids_two, targets=targets_two, features=features_two, names=names,
        outer_assignment=outer_assignment, groups=groups_two,
    )
    run_two = run_observed_matched_nulls(
        tmp_path / "nulls-two",
        source_run=source_two.directory, features=features_two, item_ids=ids_two,
        outer_folds=outer_two, space="A_STANDARDIZED", n_draws_per_fold=4, seed=31,
    )

    for frame_one, frame_two in (
        (run_one.draws, run_two.draws),
        (run_one.fold_summaries, run_two.fold_summaries),
    ):
        fold_one = frame_one[frame_one["outer_fold"] == 0].reset_index(drop=True)
        fold_two = frame_two[frame_two["outer_fold"] == 0].reset_index(drop=True)
        pd.testing.assert_frame_equal(fold_one, fold_two)


def test_fold_engine_matches_run_and_needs_no_test_rows(tmp_path: Path) -> None:
    item_ids, targets, features, names, outer_assignment, groups = _synthetic_data()
    source, outer = _make_source(
        tmp_path / "source",
        item_ids=item_ids, targets=targets, features=features, names=names,
        outer_assignment=outer_assignment, groups=groups,
    )
    artifact = run_observed_matched_nulls(
        tmp_path / "nulls",
        source_run=source.directory, features=features, item_ids=item_ids,
        outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=3,
        max_attempts_per_draw=50, seed=17,
    )
    from frozen_emotion_spaces.counterfactual import (
        _fit_space_transform,
        _fold_seed_component,
    )
    from frozen_emotion_spaces.counterfactual_nulls import _encode_targets

    for fold in (0, 1, 2):
        train_mask = outer_assignment != fold
        corrupted = features.copy()
        corrupted[~train_mask] = np.nan
        train, _, _ = _fit_space_transform(
            corrupted[train_mask],
            space="A_STANDARDIZED", pca_components=None,
        )
        encoded = _encode_targets(targets[train_mask], names)
        draws, _, observed = _fold_null_draws(
            train, encoded, item_ids[train_mask],
            seed=17, fold_code=_fold_seed_component(np.int64(fold)),
            n_classes=3, n_draws=3, max_attempts=50,
        )
        recorded = artifact.draws[artifact.draws["outer_fold"] == fold]
        assert len(recorded) == len(draws)
        for expected, row in zip(
            sorted(draws, key=lambda row: (row["mechanism"], row["draw_number"])),
            recorded.sort_values(["mechanism", "draw_number"]).itertuples(index=False),
            strict=True,
        ):
            assert expected["contrast_sum"] == pytest.approx(row.contrast_sum)
            assert expected["representation_sum"] == pytest.approx(row.representation_sum)
            assert expected["attempts"] == int(row.attempts)
        assert np.isfinite(observed["contrast_sum"])


def test_semantic_corruption_is_detected(tmp_path: Path) -> None:
    item_ids, targets, features, names, outer_assignment, groups = _synthetic_data()
    source, outer = _make_source(
        tmp_path / "source",
        item_ids=item_ids, targets=targets, features=features, names=names,
        outer_assignment=outer_assignment, groups=groups,
    )
    artifact = run_observed_matched_nulls(
        tmp_path / "nulls",
        source_run=source.directory, features=features, item_ids=item_ids,
        outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=3,
        max_attempts_per_draw=50, seed=23,
    )

    def rehash(directory: Path, filename: str) -> None:
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        path = directory / filename
        metadata["files"][filename] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    corrupted_draws = tmp_path / "corrupted-draws"
    shutil.copytree(artifact.directory, corrupted_draws)
    draws = pd.read_parquet(corrupted_draws / "draws.parquet")
    draws.loc[0, "contrast_sum"] = float(draws.loc[0, "contrast_sum"]) + 1.0
    draws.to_parquet(corrupted_draws / "draws.parquet", index=False, engine="pyarrow")
    rehash(corrupted_draws, "draws.parquet")
    with pytest.raises(ValueError, match="inconsistent"):
        validate_observed_matched_nulls(corrupted_draws)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_observed_matched_nulls(corrupted_draws, verify_hashes=False)

    corrupted_summary = tmp_path / "corrupted-summary"
    shutil.copytree(artifact.directory, corrupted_summary)
    summaries = pd.read_parquet(corrupted_summary / "fold_summaries.parquet")
    summaries.loc[0, "joint_domination_fraction"] = 0.5
    summaries.to_parquet(
        corrupted_summary / "fold_summaries.parquet", index=False, engine="pyarrow"
    )
    rehash(corrupted_summary, "fold_summaries.parquet")
    with pytest.raises(ValueError, match="inconsistent"):
        validate_observed_matched_nulls(corrupted_summary)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_observed_matched_nulls(corrupted_summary, verify_hashes=False)

    corrupted_status = tmp_path / "corrupted-status"
    shutil.copytree(artifact.directory, corrupted_status)
    metadata_path = corrupted_status / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "confirmatory_hcr4"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="identity"):
        validate_observed_matched_nulls(corrupted_status)

    corrupted_dimensions = tmp_path / "corrupted-dimensions"
    shutil.copytree(artifact.directory, corrupted_dimensions)
    metadata_path = corrupted_dimensions / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pca_components"] = 2
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="A space dimensions"):
        validate_observed_matched_nulls(corrupted_dimensions)

    corrupted_numeric_type = tmp_path / "corrupted-numeric-type"
    shutil.copytree(artifact.directory, corrupted_numeric_type)
    metadata_path = corrupted_numeric_type / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_dimension"] = str(metadata["input_dimension"])
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="valid input_dimension"):
        validate_observed_matched_nulls(corrupted_numeric_type)


def _coincident_centroid_data() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, np.ndarray
]:
    """Blue rows are bitwise copies of the red rows, so the observed red and
    blue class centroids are identical after any rowwise transform and the
    blue ordinary cell is always empty.  The red coordinates cancel exactly
    inside every five-item fold block; the constant green corner guarantees
    duplicate-site rejections for the cell-centroid null."""

    names = ("red", "blue", "green")
    item_ids = np.asarray([f"coincident-{index:03d}" for index in range(45)])
    targets = np.repeat(np.asarray(names), 15)
    red_blocks = (
        (np.array([1.25, -1.25, 0.75, -0.75, 0.0]), np.array([0.5, -0.5, 2.0, -2.0, 0.0])),
        (np.array([1.5, -1.5, 0.25, -0.25, 0.0]), np.array([1.0, -1.0, 3.0, -3.0, 0.0])),
        (np.array([2.25, -2.25, 1.75, -1.75, 0.0]), np.array([0.25, -0.25, 1.5, -1.5, 0.0])),
    )
    features = np.zeros((45, 2))
    for block_index, (x_values, y_values) in enumerate(red_blocks):
        block = np.column_stack([x_values, y_values])
        features[block_index * 5:(block_index + 1) * 5] = block
        features[15 + block_index * 5:15 + (block_index + 1) * 5] = block
    features[30:45] = 10.0
    outer_assignment = np.tile(np.repeat(np.arange(3), 5), 3)
    groups = np.asarray([f"group-{index:03d}" for index in range(45)])
    return item_ids, targets, features, names, outer_assignment, groups


def test_empty_observed_cells_block_monte_carlo_comparisons(tmp_path: Path) -> None:
    item_ids, targets, features, names, outer_assignment, groups = (
        _coincident_centroid_data()
    )
    source, outer = _make_source(
        tmp_path / "source",
        item_ids=item_ids, targets=targets, features=features, names=names,
        outer_assignment=outer_assignment, groups=groups,
    )
    artifact = run_observed_matched_nulls(
        tmp_path / "nulls",
        source_run=source.directory, features=features, item_ids=item_ids,
        outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=3,
        max_attempts_per_draw=500, seed=41,
    )
    summaries = artifact.fold_summaries
    assert (summaries["fidelity_precondition"] == "empty_observed_cells").all()
    assert (summaries["observed_n_empty_cells"] == 1).all()
    assert np.isfinite(summaries["observed_contrast_sum"]).all()
    assert summaries["observed_representation_sum"].isna().all()
    assert summaries["contrast_p_plus1"].isna().all()
    assert summaries["representation_p_plus1"].isna().all()
    assert summaries["joint_domination_fraction"].isna().all()
    assert len(artifact.draws) == 3 * len(MECHANISMS) * 3
    validated = validate_observed_matched_nulls(artifact.directory)
    assert validated.metadata == artifact.metadata


def test_rejections_and_attempts_are_counted_explicitly(tmp_path: Path) -> None:
    item_ids, targets, features, names, outer_assignment, groups = (
        _coincident_centroid_data()
    )
    source, outer = _make_source(
        tmp_path / "source",
        item_ids=item_ids, targets=targets, features=features, names=names,
        outer_assignment=outer_assignment, groups=groups,
    )
    artifact = run_observed_matched_nulls(
        tmp_path / "nulls",
        source_run=source.directory, features=features, item_ids=item_ids,
        outer_folds=outer, space="A_STANDARDIZED", n_draws_per_fold=4,
        max_attempts_per_draw=500, seed=53,
    )
    summaries = artifact.fold_summaries
    attempts = summaries["n_attempts_total"]
    assert (attempts == summaries["n_draws_accepted"] + summaries["n_draws_rejected"]).all()
    # The duplicated red/blue rows and the constant green corner force
    # duplicate-site or empty-cell rejections with this locked seed.
    assert artifact.metadata["rejected_attempts"] > 0
    cell = summaries[summaries["mechanism"] == "counterfactual_cell_centroids"]
    breakdown = cell[
        [
            "n_rejected_duplicate_sites", "n_rejected_empty_initial_cells",
            "n_rejected_empty_reinduced_cells",
        ]
    ].sum(axis=1)
    assert (breakdown == cell["n_draws_rejected"]).all()
    permutation = summaries[summaries["mechanism"] == "label_permutation"]
    breakdown = permutation[
        ["n_rejected_identity_arrangement", "n_rejected_empty_induced_cells"]
    ].sum(axis=1)
    assert (breakdown == permutation["n_draws_rejected"]).all()
