"""Train-only diagnostics for geometry stored by an A/H/AH probe run."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from .experiment_a import (
    _dataframe_digest,
    _ordered_pair_digest,
    _sha256_array,
    _sha256_file,
)
from .experiment_c import _aligned_outer, validate_crowd_representation_probe
from .geometry import (
    assign_domain_points_by_power_distance,
    assign_domain_points_to_nearest_sites,
    contrast_representation_scores,
    contrast_sum_pairwise_distances,
    mean_pairwise_site_distance,
)


ANALYSIS_FORMAT = "frozen-emotion-spaces-observed-geometry-reconstruction-v1"
ANALYSIS_FILES = ("scores.parquet", "metadata.json")


@dataclass(frozen=True)
class ObservedGeometryArtifact:
    directory: Path
    metadata: dict[str, Any]
    scores: pd.DataFrame


def score_observed_run_geometry(
    run_directory: str | Path,
    *,
    features: np.ndarray,
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
) -> pd.DataFrame:
    """Score stored sites on each corresponding outer-training domain.

    The source-faithful Representation score is undefined when an ordinary
    nearest-site cell is empty. Such rows are retained with an explicit status
    and NaN Representation rather than silently dropping a class or changing
    the definition.
    """

    artifact = validate_crowd_representation_probe(run_directory)
    metadata = artifact.metadata
    matrix = np.asarray(features, dtype=np.float64)
    ids = np.asarray([str(value) for value in item_ids], dtype=str)
    expected_shape = (int(metadata["n_items"]), int(metadata["n_features"]))
    if matrix.shape != expected_shape or ids.shape != (expected_shape[0],):
        raise ValueError("features/item_ids shape disagrees with the run metadata")
    if np.unique(ids).size != ids.size:
        raise ValueError("item_ids must be unique")
    if _sha256_array(matrix) != metadata["feature_matrix_sha256"]:
        raise ValueError("feature matrix hash disagrees with the fitted run")
    if _dataframe_digest(outer_folds) != metadata["outer_split_sha256"]:
        raise ValueError("outer split hash disagrees with the fitted run")
    target_by_id = artifact.oof.set_index("item_id", verify_integrity=True)["y_true"]
    try:
        targets = np.asarray([str(target_by_id.at[item_id]) for item_id in ids])
    except KeyError as error:
        raise ValueError("item_ids disagree with the fitted run") from error
    if _ordered_pair_digest(ids, targets) != metadata["ordered_item_target_sha256"]:
        raise ValueError("ordered item/target identity disagrees with the fitted run")

    outer, _, unique_outer = _aligned_outer(ids, outer_folds)
    class_names = tuple(str(value) for value in metadata["class_names"])
    block_dims = tuple(int(value) for value in metadata["block_dims"])
    with np.load(Path(run_directory) / "geometry.npz", allow_pickle=False) as archive:
        geometry = {name: np.asarray(archive[name]) for name in archive.files}
    geometry_by_fold = {
        str(fold): index for index, fold in enumerate(geometry["outer_fold"].astype(str))
    }
    rows: list[dict[str, object]] = []
    for outer_fold in unique_outer:
        fold_key = str(outer_fold)
        if fold_key not in geometry_by_fold:
            raise ValueError(f"geometry archive lacks outer fold {outer_fold!r}")
        geometry_index = geometry_by_fold[fold_key]
        training_mask = outer != outer_fold
        domain = _transform_from_serialized_geometry(
            matrix[training_mask],
            mean=geometry["scaler_mean"][geometry_index],
            scale=geometry["scaler_scale"][geometry_index],
            block_dims=block_dims,
            block_multipliers=geometry["block_multipliers"][geometry_index],
        )
        target_train = targets[training_mask]
        for family, sites in (
            ("class_centroids", geometry["class_centroids"][geometry_index]),
            ("linear_decoder_sites", geometry["sites"][geometry_index]),
        ):
            assignments = assign_domain_points_to_nearest_sites(
                sites,
                domain_points=domain,
                require_nonempty_cells=False,
            )
            counts = np.bincount(assignments, minlength=len(class_names))
            empty = np.flatnonzero(counts == 0)
            contrast_sum = contrast_sum_pairwise_distances(
                sites, domain_points=domain
            )
            contrast_mean = mean_pairwise_site_distance(
                sites, domain_points=domain
            )
            representation_sum = np.nan
            representation_mean = np.nan
            status = "undefined_empty_cells" if empty.size else "defined"
            if not empty.size:
                scores = contrast_representation_scores(sites, domain_points=domain)
                representation_sum = scores.representation_sum
                representation_mean = scores.mean_site_to_cell_centroid_distance
            fidelity = _partition_fidelity(
                target_train,
                assignments,
                class_names=class_names,
            )
            row: dict[str, object] = {
                "outer_fold": outer_fold,
                "site_family": family,
                "domain": "outer_train_only",
                "partition_rule": "ordinary_voronoi",
                "dimension": domain.shape[1],
                "n_domain_items": domain.shape[0],
                "n_empty_cells": int(empty.size),
                "empty_cell_indices": ",".join(str(value) for value in empty),
                "representation_status": status,
                "contrast_sum": contrast_sum,
                "mean_pairwise_site_distance": contrast_mean,
                "representation_sum": representation_sum,
                "mean_site_to_cell_centroid_distance": representation_mean,
                **fidelity,
                "power_macro_f1": np.nan,
                "power_adjusted_rand": np.nan,
                "power_normalized_mutual_info": np.nan,
            }
            if family == "linear_decoder_sites":
                power_assignment = assign_domain_points_by_power_distance(
                    sites,
                    geometry["power_weights"][geometry_index],
                    domain_points=domain,
                )
                power_fidelity = _partition_fidelity(
                    target_train,
                    power_assignment,
                    class_names=class_names,
                )
                row.update(
                    {
                        "power_macro_f1": power_fidelity["ordinary_macro_f1"],
                        "power_adjusted_rand": power_fidelity["ordinary_adjusted_rand"],
                        "power_normalized_mutual_info": power_fidelity[
                            "ordinary_normalized_mutual_info"
                        ],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["outer_fold", "site_family"], kind="stable"
    ).reset_index(drop=True)


def write_observed_geometry_analysis(
    output_directory: str | Path,
    *,
    run_directory: str | Path,
    features: np.ndarray,
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
) -> ObservedGeometryArtifact:
    """Atomically publish train-only observed-constellation diagnostics."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite observed geometry: {output}")
    run = validate_crowd_representation_probe(run_directory)
    matrix = np.asarray(features, dtype=np.float64)
    scores = score_observed_run_geometry(
        run.directory,
        features=matrix,
        item_ids=item_ids,
        outer_folds=outer_folds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        score_path = temporary / "scores.parquet"
        scores.to_parquet(score_path, index=False, engine="pyarrow", compression="zstd")
        metadata = {
            "analysis_format": ANALYSIS_FORMAT,
            "status": "new_replication_diagnostic_not_historical_recovery",
            "domain": "outer_train_only",
            "ordinary_score_definition": (
                "IgorDouven/Concept_Learning calc(), commit "
                "2325717f68f9eecbc85cfa7d7e5ada0dc7e95679"
            ),
            "representation": run.metadata["representation"],
            "n_items": int(run.metadata["n_items"]),
            "n_features": int(run.metadata["n_features"]),
            "source_run_metadata_sha256": _sha256_file(run.directory / "metadata.json"),
            "feature_matrix_sha256": _sha256_array(matrix),
            "outer_split_sha256": _dataframe_digest(outer_folds),
            "implementation_sha256": _sha256_file(Path(__file__)),
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
    return validate_observed_geometry_analysis(output)


def validate_observed_geometry_analysis(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> ObservedGeometryArtifact:
    root = Path(directory)
    missing = [name for name in ANALYSIS_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"partial observed geometry analysis; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("observed geometry metadata is unreadable") from error
    if metadata.get("analysis_format") != ANALYSIS_FORMAT:
        raise ValueError("unknown observed geometry analysis format")
    if (
        metadata.get("status")
        != "new_replication_diagnostic_not_historical_recovery"
        or metadata.get("domain") != "outer_train_only"
        or not isinstance(metadata.get("n_items"), int)
        or int(metadata["n_items"]) <= 0
        or not isinstance(metadata.get("n_features"), int)
        or int(metadata["n_features"]) <= 0
    ):
        raise ValueError("observed geometry identity metadata is invalid")
    for field in (
        "source_run_metadata_sha256", "feature_matrix_sha256",
        "outer_split_sha256", "implementation_sha256",
    ):
        if not _is_sha256(metadata.get(field)):
            raise ValueError(f"observed geometry metadata lacks {field}")
    files = metadata.get("files")
    record = files.get("scores.parquet") if isinstance(files, Mapping) else None
    score_path = root / "scores.parquet"
    if not isinstance(record, Mapping) or score_path.stat().st_size != record.get("bytes"):
        raise ValueError("observed geometry score file size mismatch")
    if verify_hashes and _sha256_file(score_path) != record.get("sha256"):
        raise ValueError("observed geometry score file hash mismatch")
    try:
        scores = pd.read_parquet(score_path, engine="pyarrow")
    except Exception as error:
        raise ValueError("observed geometry score parquet is unreadable") from error
    required = {
        "outer_fold", "site_family", "domain", "partition_rule", "dimension",
        "n_domain_items", "n_empty_cells", "representation_status",
        "contrast_sum", "mean_pairwise_site_distance", "representation_sum",
        "mean_site_to_cell_centroid_distance", "ordinary_macro_f1",
        "ordinary_adjusted_rand", "ordinary_normalized_mutual_info",
        "power_macro_f1", "power_adjusted_rand", "power_normalized_mutual_info",
    }
    if scores.empty or not required.issubset(scores):
        raise ValueError("observed geometry score schema is incomplete")
    if scores.duplicated(["outer_fold", "site_family"]).any():
        raise ValueError("observed geometry contains duplicate fold/site rows")
    if set(scores["site_family"]) != {"class_centroids", "linear_decoder_sites"}:
        raise ValueError("observed geometry has an unknown or incomplete site axis")
    if not (scores.groupby("outer_fold")["site_family"].nunique() == 2).all():
        raise ValueError("observed geometry fold/site axis is incomplete")
    numeric_columns = (
        "dimension", "n_domain_items", "n_empty_cells", "contrast_sum",
        "mean_pairwise_site_distance", "representation_sum",
        "mean_site_to_cell_centroid_distance", "ordinary_macro_f1",
        "ordinary_adjusted_rand", "ordinary_normalized_mutual_info",
        "power_macro_f1", "power_adjusted_rand", "power_normalized_mutual_info",
    )
    numeric = scores[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if (
        not (numeric["dimension"].astype(int) == int(metadata["n_features"])).all()
        or (numeric["n_domain_items"] <= 0).any()
        or (numeric["n_domain_items"] > int(metadata["n_items"])).any()
        or (numeric["n_empty_cells"] < 0).any()
        or not np.isfinite(
            numeric[["dimension", "n_domain_items", "n_empty_cells",
                     "contrast_sum", "mean_pairwise_site_distance"]].to_numpy()
        ).all()
        or (numeric[["contrast_sum", "mean_pairwise_site_distance"]] < 0).any().any()
    ):
        raise ValueError("observed geometry dimensions or scores are invalid")
    if not (scores["domain"] == "outer_train_only").all() or not (
        scores["partition_rule"] == "ordinary_voronoi"
    ).all():
        raise ValueError("observed geometry domain or partition rule is invalid")
    defined = scores["representation_status"] == "defined"
    if (
        not set(scores["representation_status"]).issubset(
            {"defined", "undefined_empty_cells"}
        )
        or
        (defined != (numeric["n_empty_cells"] == 0)).any()
        or not np.isfinite(
            numeric.loc[
                defined,
                ["representation_sum", "mean_site_to_cell_centroid_distance"],
            ].to_numpy()
        ).all()
        or (numeric.loc[
            defined, ["representation_sum", "mean_site_to_cell_centroid_distance"]
        ] < 0).any().any()
        or numeric.loc[
            ~defined, ["representation_sum", "mean_site_to_cell_centroid_distance"]
        ].notna().any().any()
    ):
        raise ValueError("observed geometry Representation status is inconsistent")
    for column in (
        "ordinary_macro_f1", "ordinary_normalized_mutual_info",
        "power_macro_f1", "power_normalized_mutual_info",
    ):
        values = numeric[column].dropna()
        if ((values < -1e-12) | (values > 1.0 + 1e-12)).any():
            raise ValueError(f"observed geometry fidelity {column} is outside [0, 1]")
    for column in ("ordinary_adjusted_rand", "power_adjusted_rand"):
        values = numeric[column].dropna()
        if ((values < -1) | (values > 1)).any():
            raise ValueError(f"observed geometry fidelity {column} is outside [-1, 1]")
    centroid_rows = scores["site_family"] == "class_centroids"
    power_columns = [
        "power_macro_f1", "power_adjusted_rand", "power_normalized_mutual_info"
    ]
    if numeric.loc[centroid_rows, power_columns].notna().any().any() or not np.isfinite(
        numeric.loc[~centroid_rows, power_columns].to_numpy()
    ).all():
        raise ValueError("observed geometry power fidelity is inconsistent")
    return ObservedGeometryArtifact(root, dict(metadata), scores)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _transform_from_serialized_geometry(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    block_dims: tuple[int, ...],
    block_multipliers: np.ndarray,
) -> np.ndarray:
    if features.shape[1] != sum(block_dims):
        raise ValueError("feature width and serialized block dimensions disagree")
    if mean.shape != (features.shape[1],) or scale.shape != mean.shape:
        raise ValueError("serialized scaler shape is invalid")
    if len(block_dims) != len(block_multipliers):
        raise ValueError("serialized block multipliers have invalid shape")
    transformed = (features - mean) / scale
    start = 0
    for width, multiplier in zip(block_dims, block_multipliers, strict=True):
        transformed[:, start : start + width] *= float(multiplier)
        start += width
    if not np.isfinite(transformed).all():
        raise ValueError("serialized transformation produced non-finite coordinates")
    return transformed


def _partition_fidelity(
    targets: np.ndarray,
    assignments: np.ndarray,
    *,
    class_names: tuple[str, ...],
) -> dict[str, float]:
    predicted = np.asarray(class_names)[assignments]
    target_index = {name: index for index, name in enumerate(class_names)}
    encoded_target = np.asarray([target_index[value] for value in targets])
    return {
        "ordinary_macro_f1": float(
            f1_score(
                targets,
                predicted,
                labels=list(class_names),
                average="macro",
                zero_division=0,
            )
        ),
        "ordinary_adjusted_rand": float(
            adjusted_rand_score(encoded_target, assignments)
        ),
        "ordinary_normalized_mutual_info": float(
            normalized_mutual_info_score(encoded_target, assignments)
        ),
    }


__all__ = [
    "ANALYSIS_FILES",
    "ANALYSIS_FORMAT",
    "ObservedGeometryArtifact",
    "score_observed_run_geometry",
    "validate_observed_geometry_analysis",
    "write_observed_geometry_analysis",
]
