"""Prospective counterfactual Contrast/Representation pilot.

This is not recovered historical code.  The site scores reproduce the
definitions in IgorDouven/Concept_Learning ``calc()`` at commit
2325717f68f9eecbc85cfa7d7e5ada0dc7e95679.  The outer-fold evaluation,
PCA, capped sampling, metrics, and artifact schema are new safeguards and
adaptations for Crowd-enVENT.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

from .experiment_a import _dataframe_digest, _sha256_array, _sha256_file
from .experiment_c import _aligned_outer, validate_crowd_representation_probe
from .geometry import (
    assign_domain_points_to_nearest_sites,
    contrast_representation_scores,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Space = Literal["A_STANDARDIZED", "H_PCA"]
SamplingScheme = Literal["per_cell_capped_items", "fixed_group_budget"]

PILOT_FORMAT = "frozen-emotion-spaces-counterfactual-pilot-reconstruction-v1"
PILOT_FILES = (
    "constellations.parquet",
    "learnability.parquet",
    "regressions.parquet",
    "transforms.npz",
    "metadata.json",
)


@dataclass(frozen=True)
class CounterfactualPilotArtifact:
    directory: Path
    metadata: dict[str, Any]
    constellations: pd.DataFrame
    learnability: pd.DataFrame
    regressions: pd.DataFrame


def run_counterfactual_pilot(
    output_directory: str | Path,
    *,
    source_run: str | Path,
    features: np.ndarray,
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    space: Space,
    n_sites: int = 13,
    n_constellations_per_fold: int = 20,
    n_repetitions: int = 5,
    pca_components: int | None = None,
    max_samples_per_cell: int = 25,
    sampling_scheme: SamplingScheme = "per_cell_capped_items",
    sample_group_budget: int | None = None,
    seed: int = 20240804,
    max_attempts_per_constellation: int = 100,
) -> CounterfactualPilotArtifact:
    """Run a small train-defined, outer-test learnability pilot.

    Sites are sampled only from an outer-training domain.  Their induced
    labels are extended to the untouched outer-test domain using the same
    sites.  Each learning repetition samples at least one and at most
    ``max_samples_per_cell`` items from every induced training cell, fits the
    approximate-prototype and inverse-squared KNN learners, and evaluates both
    only on the group-disjoint outer test.
    """

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual pilot: {output}")
    artifact = validate_crowd_representation_probe(source_run)
    expected_representation = "A" if space == "A_STANDARDIZED" else "H"
    if artifact.metadata["representation"] != expected_representation:
        raise ValueError(
            f"space {space} requires a {expected_representation} source run"
        )
    if space == "A_STANDARDIZED" and pca_components is not None:
        raise ValueError("A_STANDARDIZED must not declare PCA components")
    if space == "H_PCA" and pca_components is None:
        raise ValueError("H_PCA requires pca_components")
    if sampling_scheme not in {"per_cell_capped_items", "fixed_group_budget"}:
        raise ValueError("unknown counterfactual sampling scheme")
    if sampling_scheme == "per_cell_capped_items":
        if sample_group_budget is not None:
            raise ValueError("per-cell item sampling must not declare a group budget")
    elif (
        not isinstance(sample_group_budget, int)
        or isinstance(sample_group_budget, bool)
        or sample_group_budget < 1
    ):
        raise ValueError("fixed group sampling requires a positive group budget")
    for name, value, minimum in (
        ("n_sites", n_sites, 2),
        ("n_constellations_per_fold", n_constellations_per_fold, 1),
        ("n_repetitions", n_repetitions, 1),
        ("max_samples_per_cell", max_samples_per_cell, 1),
        ("max_attempts_per_constellation", max_attempts_per_constellation, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    matrix = np.asarray(features, dtype=np.float64)
    ids = np.asarray([str(value) for value in item_ids], dtype=str)
    if matrix.shape != (int(artifact.metadata["n_items"]), int(artifact.metadata["n_features"])):
        raise ValueError("features disagree with source-run dimensions")
    if ids.shape != (matrix.shape[0],) or np.unique(ids).size != ids.size:
        raise ValueError("item_ids must be unique and aligned with features")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain non-finite values")
    if _sha256_array(matrix) != artifact.metadata["feature_matrix_sha256"]:
        raise ValueError("feature matrix hash disagrees with source run")
    if _dataframe_digest(outer_folds) != artifact.metadata["outer_split_sha256"]:
        raise ValueError("outer split hash disagrees with source run")

    outer, groups, unique_outer = _aligned_outer(ids, outer_folds)
    if n_sites >= min(int(np.sum(outer != fold)) for fold in unique_outer):
        raise ValueError("n_sites must be smaller than every outer-training domain")

    constellation_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    transform_records: list[dict[str, np.ndarray]] = []
    total_rejections = 0
    output_dimension: int | None = None

    for outer_fold in unique_outer:
        train_mask = outer != outer_fold
        test_mask = ~train_mask
        if set(groups[train_mask]) & set(groups[test_mask]):
            raise ValueError(f"outer fold {outer_fold!r} leaks groups")
        train, test, transform = _fit_space_transform(
            matrix[train_mask],
            matrix[test_mask],
            space=space,
            pca_components=pca_components,
        )
        if output_dimension is None:
            output_dimension = train.shape[1]
        elif output_dimension != train.shape[1]:  # pragma: no cover - fixed config
            raise RuntimeError("transformed dimensions vary across folds")
        transform_records.append({"outer_fold": np.asarray(str(outer_fold)), **transform})
        train_ids = ids[train_mask]
        train_groups = groups[train_mask]
        if (
            sampling_scheme == "fixed_group_budget"
            and int(sample_group_budget or 0) > np.unique(train_groups).size
        ):
            raise ValueError("group budget exceeds an outer-training group count")

        for constellation_number in range(n_constellations_per_fold):
            fold_code = _fold_seed_component(outer_fold)
            site_rng = np.random.default_rng(
                np.random.SeedSequence([seed, fold_code, constellation_number, 0])
            )
            accepted: tuple[IntArray, IntArray, IntArray] | None = None
            attempts = 0
            while attempts < max_attempts_per_constellation:
                attempts += 1
                site_indices = np.asarray(
                    site_rng.choice(len(train), size=n_sites, replace=False),
                    dtype=np.int64,
                )
                sites = train[site_indices]
                if np.unique(sites, axis=0).shape[0] != n_sites:
                    total_rejections += 1
                    continue
                train_assignments = assign_domain_points_to_nearest_sites(
                    sites, domain_points=train, require_nonempty_cells=False
                )
                train_support = np.bincount(train_assignments, minlength=n_sites)
                if (train_support == 0).any():
                    total_rejections += 1
                    continue
                test_assignments = assign_domain_points_to_nearest_sites(
                    sites, domain_points=test, require_nonempty_cells=False
                )
                accepted = (site_indices, train_assignments, test_assignments)
                break
            if accepted is None:
                raise RuntimeError(
                    f"could not sample a valid constellation for fold {outer_fold!r} "
                    f"after {max_attempts_per_constellation} attempts"
                )
            site_indices, train_assignments, test_assignments = accepted
            sites = train[site_indices]
            train_support = np.bincount(train_assignments, minlength=n_sites)
            test_support = np.bincount(test_assignments, minlength=n_sites)
            design = contrast_representation_scores(sites, domain_points=train)
            constellation_id = f"{outer_fold}:{constellation_number:05d}"
            constellation_rows.append(
                {
                    "constellation_id": constellation_id,
                    "outer_fold": outer_fold,
                    "constellation_number": constellation_number,
                    "attempts": attempts,
                    "site_item_ids_json": json.dumps(train_ids[site_indices].tolist()),
                    "train_cell_supports_json": json.dumps(train_support.tolist()),
                    "test_cell_supports_json": json.dumps(test_support.tolist()),
                    "n_empty_test_cells": int(np.sum(test_support == 0)),
                    "contrast_sum": design.contrast_sum,
                    "mean_pairwise_site_distance": design.mean_pairwise_site_distance,
                    "representation_sum": design.representation_sum,
                    "mean_site_to_cell_centroid_distance": (
                        design.mean_site_to_cell_centroid_distance
                    ),
                }
            )

            for repetition in range(n_repetitions):
                repetition_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, fold_code, constellation_number, repetition, 1]
                    )
                )
                if sampling_scheme == "per_cell_capped_items":
                    sample_indices = _sample_each_cell(
                        train_assignments,
                        n_sites=n_sites,
                        maximum_per_cell=max_samples_per_cell,
                        rng=repetition_rng,
                    )
                else:
                    sample_indices = _sample_complete_groups(
                        train_assignments,
                        train_groups,
                        n_sites=n_sites,
                        group_budget=int(sample_group_budget or 0),
                        rng=repetition_rng,
                    )
                sample_X = train[sample_indices]
                sample_y = train_assignments[sample_indices]
                prototype_prediction = _approximate_prototype_predict(
                    sample_X, sample_y, test, n_sites=n_sites
                )
                neighbor_k = max(1, min(len(sample_X), int(round(np.sqrt(len(sample_X))))))
                knn_prediction = _inverse_squared_knn_predict(
                    sample_X,
                    sample_y,
                    test,
                    n_classes=n_sites,
                    n_neighbors=neighbor_k,
                )
                sample_group_count = int(np.unique(train_groups[sample_indices]).size)
                for learner, prediction, k_value in (
                    ("approximate_prototype", prototype_prediction, 0),
                    ("knn_inverse_squared", knn_prediction, neighbor_k),
                ):
                    learning_rows.append(
                        {
                            "constellation_id": constellation_id,
                            "outer_fold": outer_fold,
                            "constellation_number": constellation_number,
                            "repetition": repetition,
                            "learner": learner,
                            "n_sample_items": len(sample_indices),
                            "n_sample_groups": sample_group_count,
                            "neighbor_k": k_value,
                            **_partition_metrics(
                                test_assignments, prediction, n_classes=n_sites
                            ),
                        }
                    )

    constellations = pd.DataFrame(constellation_rows).sort_values(
        ["outer_fold", "constellation_number"], kind="stable"
    ).reset_index(drop=True)
    learnability = pd.DataFrame(learning_rows).sort_values(
        ["outer_fold", "constellation_number", "repetition", "learner"],
        kind="stable",
    ).reset_index(drop=True)
    regressions = _standardized_regressions(
        constellations,
        learnability,
        adjust_sample_items=sampling_scheme == "fixed_group_budget",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        constellation_path = temporary / "constellations.parquet"
        learning_path = temporary / "learnability.parquet"
        regression_path = temporary / "regressions.parquet"
        transform_path = temporary / "transforms.npz"
        constellations.to_parquet(
            constellation_path, index=False, engine="pyarrow", compression="zstd"
        )
        learnability.to_parquet(
            learning_path, index=False, engine="pyarrow", compression="zstd"
        )
        regressions.to_parquet(
            regression_path, index=False, engine="pyarrow", compression="zstd"
        )
        _write_transforms(transform_path, transform_records)
        file_records = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                constellation_path, learning_path, regression_path, transform_path
            )
        }
        metadata = {
            "pilot_format": PILOT_FORMAT,
            "status": "prospective_pilot_new_replication_not_historical_recovery",
            "dataset": artifact.metadata["dataset"],
            "space": space,
            "source_representation": artifact.metadata["representation"],
            "source_run_metadata_sha256": _sha256_file(artifact.directory / "metadata.json"),
            "source_feature_matrix_sha256": artifact.metadata["feature_matrix_sha256"],
            "source_outer_split_sha256": artifact.metadata["outer_split_sha256"],
            "source_embedding_layer_sha256": artifact.metadata[
                "embedding_layer_sha256"
            ],
            "n_items": len(ids),
            "input_dimension": matrix.shape[1],
            "output_dimension": int(output_dimension or 0),
            "pca_components": pca_components,
            "n_sites": n_sites,
            "n_outer_folds": len(unique_outer),
            "n_constellations_per_fold": n_constellations_per_fold,
            "n_repetitions": n_repetitions,
            "max_samples_per_cell": max_samples_per_cell,
            "sampling_scheme": sampling_scheme,
            "sample_group_budget": sample_group_budget,
            "max_attempts_per_constellation": max_attempts_per_constellation,
            "rejected_site_draws": total_rejections,
            "seed": seed,
            "site_score_definition": (
                "IgorDouven/Concept_Learning calc(), commit "
                "2325717f68f9eecbc85cfa7d7e5ada0dc7e95679"
            ),
            "evaluation_domain": "preserved_group_disjoint_outer_test_only",
            "sampling_rule": (
                "uniform integer 1..min(cell support,max_samples_per_cell), "
                "then item sampling without replacement inside every train cell"
                if sampling_scheme == "per_cell_capped_items"
                else "uniform fixed-size sample of complete outer-training groups, "
                "rejected and redrawn until every induced cell is represented"
            ),
            "implementation_sha256": {
                "counterfactual.py": _sha256_file(Path(__file__)),
                "geometry.py": _sha256_file(Path(__file__).with_name("geometry.py")),
            },
            "numpy_version": np.__version__,
            "scipy_version": distribution_version("scipy"),
            "scikit_learn_version": distribution_version("scikit-learn"),
            "numpy_bit_generator": "PCG64",
            "numpy_build_dependencies": np.__config__.CONFIG.get(
                "Build Dependencies", {}
            ),
            "files": file_records,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_counterfactual_pilot(output)


def validate_counterfactual_pilot(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> CounterfactualPilotArtifact:
    root = Path(directory)
    missing = [name for name in PILOT_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"partial counterfactual pilot; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("counterfactual metadata is unreadable") from error
    if (
        metadata.get("pilot_format") != PILOT_FORMAT
        or metadata.get("status")
        != "prospective_pilot_new_replication_not_historical_recovery"
        or metadata.get("space") not in {"A_STANDARDIZED", "H_PCA"}
    ):
        raise ValueError("counterfactual pilot identity is invalid")
    for field in (
        "source_run_metadata_sha256", "source_feature_matrix_sha256",
        "source_outer_split_sha256",
    ):
        if not _is_sha256(metadata.get(field)):
            raise ValueError(f"counterfactual metadata lacks {field}")
    if metadata["space"] == "H_PCA" and not _is_sha256(
        metadata.get("source_embedding_layer_sha256")
    ):
        raise ValueError("H_PCA metadata lacks its embedding-layer identity")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("counterfactual metadata lacks file records")
    for filename in PILOT_FILES[:-1]:
        record = files.get(filename)
        path = root / filename
        if not isinstance(record, Mapping) or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"counterfactual file size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"counterfactual file hash mismatch: {filename}")
    try:
        constellations = pd.read_parquet(root / "constellations.parquet")
        learnability = pd.read_parquet(root / "learnability.parquet")
        regressions = pd.read_parquet(root / "regressions.parquet")
    except Exception as error:
        raise ValueError("counterfactual parquet is unreadable") from error
    required_constellations = {
        "constellation_id", "outer_fold", "constellation_number", "attempts",
        "site_item_ids_json", "train_cell_supports_json", "test_cell_supports_json",
        "n_empty_test_cells", "contrast_sum", "mean_pairwise_site_distance",
        "representation_sum", "mean_site_to_cell_centroid_distance",
    }
    required_learning = {
        "constellation_id", "outer_fold", "constellation_number", "repetition",
        "learner", "n_sample_items", "n_sample_groups", "neighbor_k",
        "mutual_info_nats", "normalized_mutual_info", "adjusted_rand",
        "macro_f1", "accuracy",
    }
    required_regression = {
        "outer_fold", "learner", "outcome", "n_constellations", "rank",
        "beta_contrast", "beta_representation", "r_squared", "status",
    }
    if (
        not required_constellations.issubset(constellations)
        or not required_learning.issubset(learnability)
        or not required_regression.issubset(regressions)
    ):
        raise ValueError("counterfactual parquet schema is incomplete")
    n_folds = int(metadata["n_outer_folds"])
    n_constellations = int(metadata["n_constellations_per_fold"])
    n_repetitions = int(metadata["n_repetitions"])
    expected_constellations = n_folds * n_constellations
    if (
        len(constellations) != expected_constellations
        or constellations["constellation_id"].duplicated().any()
        or len(learnability) != expected_constellations * n_repetitions * 2
        or learnability.duplicated(
            ["constellation_id", "repetition", "learner"]
        ).any()
        or set(learnability["learner"])
        != {"approximate_prototype", "knn_inverse_squared"}
    ):
        raise ValueError("counterfactual row coverage is invalid")
    if set(learnability["constellation_id"]) != set(constellations["constellation_id"]):
        raise ValueError("counterfactual learning/constellation identities disagree")
    n_sites = int(metadata["n_sites"])
    maximum_per_cell = int(metadata["max_samples_per_cell"])
    sampling_scheme = metadata.get("sampling_scheme", "per_cell_capped_items")
    sample_group_budget = metadata.get("sample_group_budget")
    if sampling_scheme not in {"per_cell_capped_items", "fixed_group_budget"}:
        raise ValueError("counterfactual sampling identity is invalid")
    if sampling_scheme == "fixed_group_budget" and (
        not isinstance(sample_group_budget, int) or sample_group_budget < 1
    ):
        raise ValueError("counterfactual fixed group budget is invalid")
    constellation_identity = constellations.set_index(
        "constellation_id", verify_integrity=True
    )[["outer_fold", "constellation_number"]]
    for row in constellations.itertuples(index=False):
        try:
            sites = json.loads(row.site_item_ids_json)
            train_support = np.asarray(
                json.loads(row.train_cell_supports_json), dtype=np.int64
            )
            test_support = np.asarray(
                json.loads(row.test_cell_supports_json), dtype=np.int64
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("counterfactual site/support JSON is invalid") from error
        if (
            not isinstance(sites, list)
            or len(sites) != n_sites
            or len(set(str(value) for value in sites)) != n_sites
            or train_support.shape != (n_sites,)
            or test_support.shape != (n_sites,)
            or (train_support <= 0).any()
            or (test_support < 0).any()
            or int(np.sum(test_support == 0)) != int(row.n_empty_test_cells)
            or int(train_support.sum() + test_support.sum()) != int(metadata["n_items"])
        ):
            raise ValueError("counterfactual site/support content is inconsistent")
    for row in learnability.itertuples(index=False):
        identity = constellation_identity.loc[row.constellation_id]
        if (
            str(row.outer_fold) != str(identity["outer_fold"])
            or int(row.constellation_number) != int(identity["constellation_number"])
        ):
            raise ValueError("counterfactual learning identity is inconsistent")
        if row.learner == "approximate_prototype":
            if int(row.neighbor_k) != 0:
                raise ValueError("prototype rows must use neighbor_k=0")
        elif int(row.neighbor_k) != max(
            1, int(round(np.sqrt(int(row.n_sample_items))))
        ):
            raise ValueError("KNN neighbor count disagrees with its sample size")
        if not n_sites <= int(row.n_sample_items) or not (
            1 <= int(row.n_sample_groups) <= int(row.n_sample_items)
        ):
            raise ValueError("counterfactual learning sample budget is invalid")
        if sampling_scheme == "per_cell_capped_items" and int(
            row.n_sample_items
        ) > n_sites * maximum_per_cell:
            raise ValueError("counterfactual per-cell sample exceeds its cap")
        if sampling_scheme == "fixed_group_budget" and int(
            row.n_sample_groups
        ) != int(sample_group_budget):
            raise ValueError("counterfactual group sample disagrees with fixed budget")
    expected_repetitions = set(range(n_repetitions))
    coverage = learnability.groupby(["constellation_id", "learner"])["repetition"]
    if any(set(values.astype(int)) != expected_repetitions for _, values in coverage):
        raise ValueError("counterfactual repetition coverage is invalid")
    design_numeric = constellations[
        [
            "attempts", "n_empty_test_cells", "contrast_sum",
            "mean_pairwise_site_distance", "representation_sum",
            "mean_site_to_cell_centroid_distance",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    learning_numeric = learnability[
        [
            "n_sample_items", "n_sample_groups", "neighbor_k",
            "mutual_info_nats", "normalized_mutual_info", "adjusted_rand",
            "macro_f1", "accuracy",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        not np.isfinite(design_numeric.to_numpy()).all()
        or not np.isfinite(learning_numeric.to_numpy()).all()
        or (design_numeric[["contrast_sum", "mean_pairwise_site_distance",
                            "representation_sum",
                            "mean_site_to_cell_centroid_distance"]] < 0).any().any()
        or (learning_numeric[["n_sample_items", "n_sample_groups"]] <= 0).any().any()
    ):
        raise ValueError("counterfactual numeric values are invalid")
    for column in ("normalized_mutual_info", "macro_f1", "accuracy"):
        if (
            (learning_numeric[column] < -1e-12)
            | (learning_numeric[column] > 1.0 + 1e-12)
        ).any():
            raise ValueError(f"counterfactual {column} is outside [0, 1]")
    if (
        (learning_numeric["adjusted_rand"] < -1.0 - 1e-12).any()
        or (learning_numeric["adjusted_rand"] > 1.0 + 1e-12).any()
    ):
        raise ValueError("counterfactual adjusted Rand is outside [-1, 1]")
    outcomes = {"mutual_info_nats", "normalized_mutual_info", "macro_f1"}
    expected_regressions = n_folds * 2 * len(outcomes)
    if (
        len(regressions) != expected_regressions
        or regressions.duplicated(["outer_fold", "learner", "outcome"]).any()
        or set(regressions["learner"])
        != {"approximate_prototype", "knn_inverse_squared"}
        or set(regressions["outcome"]) != outcomes
        or not (regressions["n_constellations"] == n_constellations).all()
        or not set(regressions["status"]).issubset(
            {"ok", "constant_outcome", "rank_deficient", "insufficient_residual_df"}
        )
    ):
        raise ValueError("counterfactual regression coverage is invalid")
    if sampling_scheme == "fixed_group_budget":
        if not {"regression_model", "beta_sample_items"}.issubset(regressions):
            raise ValueError("group-budget regression lacks its sample-size adjustment")
        if not (
            regressions["regression_model"] == "C_R_plus_sample_items"
        ).all():
            raise ValueError("group-budget regression model identity is invalid")
    _validate_transforms(root / "transforms.npz", metadata=metadata)
    return CounterfactualPilotArtifact(
        root, dict(metadata), constellations, learnability, regressions
    )


def _fit_space_transform(
    train: FloatArray,
    test: FloatArray | None = None,
    *,
    space: Space,
    pca_components: int | None,
) -> tuple[FloatArray, FloatArray | None, dict[str, np.ndarray]]:
    """Fit the outer-train-only space transform.

    ``test=None`` is the explicit train-only mode used by the matched-null
    generator: no held-out row is transformed or inspected, so a null
    distribution cannot depend on outer-test data.
    """

    scaler = StandardScaler().fit(train)
    scaled_train = np.asarray(scaler.transform(train), dtype=np.float64)
    scaled_test = (
        None if test is None else np.asarray(scaler.transform(test), dtype=np.float64)
    )
    if space == "A_STANDARDIZED":
        transformed_train = scaled_train
        transformed_test = scaled_test
        components = np.empty((0, train.shape[1]), dtype=np.float64)
        pca_mean = np.empty((0,), dtype=np.float64)
        explained = np.empty((0,), dtype=np.float64)
    else:
        if (
            not isinstance(pca_components, int)
            or isinstance(pca_components, bool)
            or not 1 <= pca_components <= min(scaled_train.shape)
        ):
            raise ValueError("pca_components is invalid for an outer-training fold")
        reducer = PCA(n_components=pca_components, svd_solver="full")
        transformed_train = np.asarray(reducer.fit_transform(scaled_train), dtype=np.float64)
        transformed_test = (
            None
            if scaled_test is None
            else np.asarray(reducer.transform(scaled_test), dtype=np.float64)
        )
        components = np.asarray(reducer.components_, dtype=np.float64)
        pca_mean = np.asarray(reducer.mean_, dtype=np.float64)
        explained = np.asarray(reducer.explained_variance_ratio_, dtype=np.float64)
    if not np.isfinite(transformed_train).all() or (
        transformed_test is not None and not np.isfinite(transformed_test).all()
    ):
        raise ValueError("space transformation produced non-finite coordinates")
    return transformed_train, transformed_test, {
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "pca_components": components,
        "pca_mean": pca_mean,
        "pca_explained_variance_ratio": explained,
    }


def _sample_each_cell(
    assignments: IntArray,
    *,
    n_sites: int,
    maximum_per_cell: int,
    rng: np.random.Generator,
) -> IntArray:
    sampled: list[IntArray] = []
    for label in range(n_sites):
        members = np.flatnonzero(assignments == label)
        if members.size == 0:
            raise ValueError("cannot learn a constellation with an empty training cell")
        maximum = min(maximum_per_cell, members.size)
        sample_size = int(rng.integers(1, maximum + 1))
        sampled.append(
            np.asarray(rng.choice(members, size=sample_size, replace=False), dtype=np.int64)
        )
    return np.concatenate(sampled)


def _sample_complete_groups(
    assignments: IntArray,
    groups: NDArray[np.str_],
    *,
    n_sites: int,
    group_budget: int,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> IntArray:
    """Sample exactly ``group_budget`` complete groups with class coverage."""

    group_values = np.unique(np.asarray(groups, dtype=str))
    if not 1 <= group_budget <= len(group_values):
        raise ValueError("group budget is incompatible with the training groups")
    for _ in range(max_attempts):
        selected_groups = rng.choice(group_values, size=group_budget, replace=False)
        selected = np.flatnonzero(np.isin(groups, selected_groups))
        if np.unique(assignments[selected]).size == n_sites:
            return np.asarray(selected, dtype=np.int64)
    raise RuntimeError(
        "could not obtain complete induced-cell coverage from the fixed group budget"
    )


def _approximate_prototype_predict(
    train: FloatArray,
    labels: IntArray,
    test: FloatArray,
    *,
    n_sites: int,
) -> IntArray:
    prototypes = np.vstack([train[labels == label].mean(axis=0) for label in range(n_sites)])
    return _nearest_rows(prototypes, test)


def _inverse_squared_knn_predict(
    train: FloatArray,
    labels: IntArray,
    test: FloatArray,
    *,
    n_classes: int,
    n_neighbors: int,
) -> IntArray:
    if not 1 <= n_neighbors <= len(train):
        raise ValueError("n_neighbors must lie between one and the training size")
    squared = _squared_distances(test, train)
    # Stable sorting makes a K-boundary distance tie resolve by original
    # training index, rather than by argpartition's platform-dependent subset.
    neighbor_indices = np.argsort(squared, axis=1, kind="stable")[:, :n_neighbors]
    prediction = np.empty(len(test), dtype=np.int64)
    for row in range(len(test)):
        indices = neighbor_indices[row]
        distances = squared[row, indices]
        neighbor_labels = labels[indices]
        zero = distances <= 1e-24
        if zero.any():
            weights = np.ones(int(zero.sum()), dtype=np.float64)
            vote_labels = neighbor_labels[zero]
        else:
            weights = 1.0 / distances
            vote_labels = neighbor_labels
        votes = np.bincount(vote_labels, weights=weights, minlength=n_classes)
        prediction[row] = int(np.argmax(votes))
    return prediction


def _partition_metrics(
    truth: IntArray,
    prediction: IntArray,
    *,
    n_classes: int,
) -> dict[str, float]:
    labels = list(range(n_classes))
    return {
        "mutual_info_nats": float(mutual_info_score(truth, prediction)),
        "normalized_mutual_info": float(normalized_mutual_info_score(truth, prediction)),
        "adjusted_rand": float(adjusted_rand_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(truth, prediction)),
    }


def _standardized_regressions(
    constellations: pd.DataFrame,
    learnability: pd.DataFrame,
    *,
    adjust_sample_items: bool = False,
) -> pd.DataFrame:
    outcomes = ("mutual_info_nats", "normalized_mutual_info", "macro_f1")
    averaged = learnability.groupby(
        ["outer_fold", "constellation_id", "learner"], as_index=False
    )[[*outcomes, "n_sample_items"]].mean()
    merged = averaged.merge(
        constellations[["constellation_id", "contrast_sum", "representation_sum"]],
        on="constellation_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (fold, learner), group in merged.groupby(["outer_fold", "learner"], sort=True):
        predictor_names = ["contrast_sum", "representation_sum"]
        if adjust_sample_items:
            predictor_names.append("n_sample_items")
        predictors = group[predictor_names].to_numpy(dtype=float)
        predictor_scale = predictors.std(axis=0, ddof=0)
        predictor_z = np.divide(
            predictors - predictors.mean(axis=0),
            predictor_scale,
            out=np.zeros_like(predictors),
            where=predictor_scale > 0,
        )
        design = np.column_stack((np.ones(len(group)), predictor_z))
        for outcome in outcomes:
            values = group[outcome].to_numpy(dtype=float)
            outcome_scale = float(values.std(ddof=0))
            rank = int(np.linalg.matrix_rank(design))
            if outcome_scale == 0:
                rows.append(
                    {
                        "outer_fold": fold,
                        "learner": learner,
                        "outcome": outcome,
                        "n_constellations": len(group),
                        "rank": rank,
                        "beta_contrast": np.nan,
                        "beta_representation": np.nan,
                        "beta_sample_items": np.nan,
                        "r_squared": np.nan,
                        "regression_model": (
                            "C_R_plus_sample_items" if adjust_sample_items else "C_R"
                        ),
                        "status": "constant_outcome",
                    }
                )
                continue
            if rank < design.shape[1] or len(group) <= rank:
                rows.append(
                    {
                        "outer_fold": fold,
                        "learner": learner,
                        "outcome": outcome,
                        "n_constellations": len(group),
                        "rank": rank,
                        "beta_contrast": np.nan,
                        "beta_representation": np.nan,
                        "beta_sample_items": np.nan,
                        "r_squared": np.nan,
                        "regression_model": (
                            "C_R_plus_sample_items" if adjust_sample_items else "C_R"
                        ),
                        "status": (
                            "rank_deficient"
                            if rank < design.shape[1]
                            else "insufficient_residual_df"
                        ),
                    }
                )
                continue
            target = (values - values.mean()) / outcome_scale
            coefficient, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
            fitted = design @ coefficient
            denominator = float(np.sum((target - target.mean()) ** 2))
            r_squared = 1.0 - float(np.sum((target - fitted) ** 2)) / denominator
            rows.append(
                {
                    "outer_fold": fold,
                    "learner": learner,
                    "outcome": outcome,
                    "n_constellations": len(group),
                    "rank": int(rank),
                    "beta_contrast": float(coefficient[1]),
                    "beta_representation": float(coefficient[2]),
                    "beta_sample_items": (
                        float(coefficient[3]) if adjust_sample_items else np.nan
                    ),
                    "r_squared": r_squared,
                    "regression_model": (
                        "C_R_plus_sample_items" if adjust_sample_items else "C_R"
                    ),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["outer_fold", "learner", "outcome"], kind="stable"
    ).reset_index(drop=True)


def _write_transforms(path: Path, records: Sequence[dict[str, np.ndarray]]) -> None:
    if not records:
        raise ValueError("at least one fold transform is required")
    fields = tuple(records[0])
    if any(tuple(record) != fields for record in records):
        raise ValueError("fold transforms have inconsistent fields")
    np.savez_compressed(
        path, **{field: np.stack([record[field] for record in records]) for field in fields}
    )


def _validate_transforms(path: Path, *, metadata: Mapping[str, Any]) -> None:
    required = {
        "outer_fold", "scaler_mean", "scaler_scale", "pca_components",
        "pca_mean", "pca_explained_variance_ratio",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError("counterfactual transform schema is invalid")
            arrays = {name: np.asarray(archive[name]) for name in required}
    except (OSError, ValueError) as error:
        raise ValueError("counterfactual transforms are unreadable") from error
    n_folds = int(metadata["n_outer_folds"])
    input_dimension = int(metadata["input_dimension"])
    components = int(metadata["pca_components"] or 0)
    expected = {
        "outer_fold": (n_folds,),
        "scaler_mean": (n_folds, input_dimension),
        "scaler_scale": (n_folds, input_dimension),
        "pca_components": (n_folds, components, input_dimension),
        "pca_mean": (n_folds, input_dimension) if components else (n_folds, 0),
        "pca_explained_variance_ratio": (n_folds, components),
    }
    if any(arrays[name].shape != shape for name, shape in expected.items()):
        raise ValueError("counterfactual transform shapes disagree with metadata")
    if len(set(arrays["outer_fold"].astype(str))) != n_folds:
        raise ValueError("counterfactual transform fold identities are not unique")
    for name, array in arrays.items():
        if name != "outer_fold" and not np.isfinite(array.astype(float)).all():
            raise ValueError(f"counterfactual transform {name} is non-finite")
    if (arrays["scaler_scale"] <= 0).any():
        raise ValueError("counterfactual scaler has a non-positive scale")
    if components:
        for fold_components in arrays["pca_components"]:
            if not np.allclose(
                fold_components @ fold_components.T,
                np.eye(components),
                atol=1e-8,
            ):
                raise ValueError("counterfactual PCA components are not orthonormal")
        ratios = arrays["pca_explained_variance_ratio"]
        if (
            (ratios < -1e-12).any()
            or (ratios > 1.0 + 1e-12).any()
            or (ratios.sum(axis=1) > 1.0 + 1e-10).any()
            or int(metadata["output_dimension"]) != components
        ):
            raise ValueError("counterfactual PCA variance/dimension is invalid")
    elif int(metadata["output_dimension"]) != input_dimension:
        raise ValueError("counterfactual standardized-space dimension is invalid")


def _nearest_rows(sites: FloatArray, points: FloatArray) -> IntArray:
    return np.argmin(_squared_distances(points, sites), axis=1).astype(np.int64)


def _squared_distances(left: FloatArray, right: FloatArray) -> FloatArray:
    squared = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    np.maximum(squared, 0.0, out=squared)
    return squared


def _fold_seed_component(value: Any) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "CounterfactualPilotArtifact",
    "PILOT_FILES",
    "PILOT_FORMAT",
    "run_counterfactual_pilot",
    "validate_counterfactual_pilot",
]
