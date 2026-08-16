"""Locate observed class-centroid sites inside a counterfactual pilot."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)

from .counterfactual import validate_counterfactual_pilot
from .experiment_a import _dataframe_digest, _sha256_array, _sha256_file
from .experiment_c import _aligned_outer, validate_crowd_representation_probe
from .geometry import (
    assign_domain_points_to_nearest_sites,
    contrast_representation_scores,
    contrast_sum_pairwise_distances,
    mean_pairwise_site_distance,
)


ANALYSIS_FORMAT = "frozen-emotion-spaces-observed-counterfactual-analysis-v1"
ANALYSIS_FILES = ("scores.parquet", "metadata.json")


@dataclass(frozen=True)
class ObservedCounterfactualArtifact:
    directory: Path
    metadata: dict[str, Any]
    scores: pd.DataFrame


def write_observed_counterfactual_analysis(
    output_directory: str | Path,
    *,
    pilot_directory: str | Path,
    source_run: str | Path,
    features: np.ndarray,
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
) -> ObservedCounterfactualArtifact:
    """Score train-derived class centroids against each fold's random sites."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite observed counterfactual analysis: {output}"
        )
    pilot = validate_counterfactual_pilot(pilot_directory)
    source = validate_crowd_representation_probe(source_run)
    if _sha256_file(source.directory / "metadata.json") != pilot.metadata[
        "source_run_metadata_sha256"
    ]:
        raise ValueError("pilot and observed analysis source runs disagree")
    matrix = np.asarray(features, dtype=np.float64)
    ids = np.asarray([str(value) for value in item_ids], dtype=str)
    if matrix.shape != (
        int(source.metadata["n_items"]), int(source.metadata["n_features"])
    ):
        raise ValueError("observed-analysis features disagree with source dimensions")
    if ids.shape != (len(matrix),) or np.unique(ids).size != len(ids):
        raise ValueError("observed-analysis item IDs are invalid")
    if _sha256_array(matrix) != source.metadata["feature_matrix_sha256"]:
        raise ValueError("observed-analysis feature hash disagrees with source run")
    if _dataframe_digest(outer_folds) != source.metadata["outer_split_sha256"]:
        raise ValueError("observed-analysis outer split disagrees with source run")
    class_names = tuple(str(value) for value in source.metadata["class_names"])
    if len(class_names) != int(pilot.metadata["n_sites"]):
        raise ValueError("observed class count and counterfactual site count disagree")

    target_by_id = source.oof.set_index("item_id", verify_integrity=True)["y_true"]
    try:
        targets = np.asarray([str(target_by_id.at[item_id]) for item_id in ids])
    except KeyError as error:
        raise ValueError("observed-analysis item IDs disagree with source OOF") from error
    outer, groups, unique_outer = _aligned_outer(ids, outer_folds)
    with np.load(Path(pilot_directory) / "transforms.npz", allow_pickle=False) as archive:
        transforms = {name: np.asarray(archive[name]) for name in archive.files}
    transform_by_fold = {
        str(value): index
        for index, value in enumerate(transforms["outer_fold"].astype(str))
    }

    rows: list[dict[str, Any]] = []
    counterfactuals = pilot.constellations
    for outer_fold in unique_outer:
        fold_key = str(outer_fold)
        if fold_key not in transform_by_fold:
            raise ValueError(f"pilot transformations lack fold {outer_fold!r}")
        train_mask = outer != outer_fold
        test_mask = ~train_mask
        if set(groups[train_mask]) & set(groups[test_mask]):
            raise ValueError(f"outer fold {outer_fold!r} leaks groups")
        transform_index = transform_by_fold[fold_key]
        train = _apply_transform(matrix[train_mask], transforms, transform_index)
        test = _apply_transform(matrix[test_mask], transforms, transform_index)
        target_train = targets[train_mask]
        target_test = targets[test_mask]
        sites = np.vstack(
            [train[target_train == name].mean(axis=0) for name in class_names]
        )
        train_assignment = assign_domain_points_to_nearest_sites(
            sites, domain_points=train, require_nonempty_cells=False
        )
        test_assignment = assign_domain_points_to_nearest_sites(
            sites, domain_points=test, require_nonempty_cells=False
        )
        train_support = np.bincount(train_assignment, minlength=len(class_names))
        empty = np.flatnonzero(train_support == 0)
        contrast = contrast_sum_pairwise_distances(sites, domain_points=train)
        contrast_mean = mean_pairwise_site_distance(sites, domain_points=train)
        representation = np.nan
        representation_mean = np.nan
        if empty.size == 0:
            design = contrast_representation_scores(sites, domain_points=train)
            representation = design.representation_sum
            representation_mean = design.mean_site_to_cell_centroid_distance
        fold_random = counterfactuals.loc[
            counterfactuals["outer_fold"].astype(str) == fold_key
        ]
        if len(fold_random) != int(pilot.metadata["n_constellations_per_fold"]):
            raise ValueError("counterfactual fold coverage is incomplete")
        random_contrast = fold_random["contrast_sum"].to_numpy(dtype=float)
        random_representation = fold_random["representation_sum"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "outer_fold": outer_fold,
            "site_family": "observed_class_centroids",
            "dimension": train.shape[1],
            "n_train": len(train),
            "n_test": len(test),
            "n_empty_train_cells": int(empty.size),
            "empty_train_cell_indices": ",".join(str(value) for value in empty),
            "contrast_sum": contrast,
            "mean_pairwise_site_distance": contrast_mean,
            "representation_sum": representation,
            "mean_site_to_cell_centroid_distance": representation_mean,
            "contrast_favorable_percentile": _smoothed_fraction(
                random_contrast <= contrast
            ),
            "representation_favorable_percentile": (
                _smoothed_fraction(random_representation >= representation)
                if np.isfinite(representation)
                else np.nan
            ),
            "observed_dominates_random_fraction": (
                _smoothed_fraction(
                    (random_contrast <= contrast)
                    & (random_representation >= representation)
                )
                if np.isfinite(representation)
                else np.nan
            ),
            "random_dominates_observed_fraction": (
                _smoothed_fraction(
                    (random_contrast >= contrast)
                    & (random_representation <= representation)
                )
                if np.isfinite(representation)
                else np.nan
            ),
            **_fidelity(
                target_train,
                train_assignment,
                class_names=class_names,
                prefix="train",
            ),
            **_fidelity(
                target_test,
                test_assignment,
                class_names=class_names,
                prefix="test",
            ),
        }
        rows.append(row)
    scores = pd.DataFrame(rows).sort_values("outer_fold", kind="stable").reset_index(
        drop=True
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        score_path = temporary / "scores.parquet"
        scores.to_parquet(score_path, index=False, engine="pyarrow", compression="zstd")
        metadata = {
            "analysis_format": ANALYSIS_FORMAT,
            "status": "exploratory_observed_vs_counterfactual_not_confirmatory",
            "site_family": "observed_class_centroids",
            "space": pilot.metadata["space"],
            "n_items": len(ids),
            "n_folds": len(unique_outer),
            "n_sites": len(class_names),
            "counterfactuals_per_fold": int(
                pilot.metadata["n_constellations_per_fold"]
            ),
            "source_run_metadata_sha256": _sha256_file(
                source.directory / "metadata.json"
            ),
            "pilot_metadata_sha256": _sha256_file(
                Path(pilot_directory) / "metadata.json"
            ),
            "feature_matrix_sha256": _sha256_array(matrix),
            "outer_split_sha256": _dataframe_digest(outer_folds),
            "implementation_sha256": _sha256_file(Path(__file__)),
            "percentile_definition": (
                "add-one smoothed favorable empirical fraction: contrast random<=observed; "
                "representation random>=observed"
            ),
            "files": {
                "scores.parquet": {
                    "sha256": _sha256_file(score_path),
                    "bytes": score_path.stat().st_size,
                }
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_observed_counterfactual_analysis(output)


def validate_observed_counterfactual_analysis(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> ObservedCounterfactualArtifact:
    root = Path(directory)
    missing = [name for name in ANALYSIS_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"partial observed-counterfactual analysis: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("observed-counterfactual metadata is unreadable") from error
    if (
        metadata.get("analysis_format") != ANALYSIS_FORMAT
        or metadata.get("status")
        != "exploratory_observed_vs_counterfactual_not_confirmatory"
        or metadata.get("site_family") != "observed_class_centroids"
    ):
        raise ValueError("observed-counterfactual identity is invalid")
    for field in (
        "source_run_metadata_sha256", "pilot_metadata_sha256",
        "feature_matrix_sha256", "outer_split_sha256", "implementation_sha256",
    ):
        if not _is_sha256(metadata.get(field)):
            raise ValueError(f"observed-counterfactual metadata lacks {field}")
    record = metadata.get("files", {}).get("scores.parquet")
    score_path = root / "scores.parquet"
    if not isinstance(record, Mapping) or score_path.stat().st_size != record.get("bytes"):
        raise ValueError("observed-counterfactual score file size mismatch")
    if verify_hashes and _sha256_file(score_path) != record.get("sha256"):
        raise ValueError("observed-counterfactual score file hash mismatch")
    try:
        scores = pd.read_parquet(score_path)
    except Exception as error:
        raise ValueError("observed-counterfactual scores are unreadable") from error
    required = {
        "outer_fold", "site_family", "dimension", "n_train", "n_test",
        "n_empty_train_cells", "contrast_sum", "representation_sum",
        "contrast_favorable_percentile", "representation_favorable_percentile",
        "observed_dominates_random_fraction", "random_dominates_observed_fraction",
        "train_macro_f1", "train_accuracy", "train_adjusted_rand",
        "train_normalized_mutual_info", "test_macro_f1", "test_accuracy",
        "test_adjusted_rand", "test_normalized_mutual_info",
    }
    if (
        not required.issubset(scores)
        or len(scores) != int(metadata.get("n_folds", 0))
        or scores["outer_fold"].astype(str).duplicated().any()
        or not (scores["site_family"] == "observed_class_centroids").all()
    ):
        raise ValueError("observed-counterfactual score coverage is invalid")
    numeric = scores[list(required - {"outer_fold", "site_family"})].apply(
        pd.to_numeric, errors="coerce"
    )
    always_finite = numeric.drop(columns=["representation_sum"])
    if not np.isfinite(always_finite.to_numpy()).all():
        raise ValueError("observed-counterfactual scores contain non-finite values")
    for column in (
        "contrast_favorable_percentile", "representation_favorable_percentile",
        "observed_dominates_random_fraction", "random_dominates_observed_fraction",
        "train_macro_f1", "train_accuracy", "train_normalized_mutual_info",
        "test_macro_f1", "test_accuracy", "test_normalized_mutual_info",
    ):
        values = numeric[column].dropna()
        if ((values < -1e-12) | (values > 1.0 + 1e-12)).any():
            raise ValueError(f"observed-counterfactual {column} is outside [0,1]")
    return ObservedCounterfactualArtifact(root, dict(metadata), scores)


def _apply_transform(
    matrix: np.ndarray,
    transforms: Mapping[str, np.ndarray],
    index: int,
) -> np.ndarray:
    scaled = (
        matrix - transforms["scaler_mean"][index]
    ) / transforms["scaler_scale"][index]
    components = transforms["pca_components"][index]
    if components.shape[0]:
        scaled = (scaled - transforms["pca_mean"][index]) @ components.T
    if not np.isfinite(scaled).all():
        raise ValueError("observed-analysis transformation produced non-finite values")
    return np.asarray(scaled, dtype=np.float64)


def _fidelity(
    targets: np.ndarray,
    assignments: np.ndarray,
    *,
    class_names: tuple[str, ...],
    prefix: str,
) -> dict[str, float]:
    prediction = np.asarray(class_names)[assignments]
    target_index = {name: index for index, name in enumerate(class_names)}
    encoded_target = np.asarray([target_index[value] for value in targets])
    return {
        f"{prefix}_macro_f1": float(
            f1_score(
                targets, prediction, labels=list(class_names), average="macro",
                zero_division=0,
            )
        ),
        f"{prefix}_accuracy": float(accuracy_score(targets, prediction)),
        f"{prefix}_adjusted_rand": float(
            adjusted_rand_score(encoded_target, assignments)
        ),
        f"{prefix}_normalized_mutual_info": float(
            normalized_mutual_info_score(encoded_target, assignments)
        ),
    }


def _smoothed_fraction(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool)
    return float((1 + values.sum()) / (len(values) + 1))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "ANALYSIS_FILES",
    "ANALYSIS_FORMAT",
    "ObservedCounterfactualArtifact",
    "validate_observed_counterfactual_analysis",
    "write_observed_counterfactual_analysis",
]
