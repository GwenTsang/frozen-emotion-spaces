"""Mechanism-matched null distributions for the observed emotion categories.

This module implements the two H-CR4 matched nulls required by the locked
protocol (``PROTOCOL_CONTRAST_REPRESENTATION.md`` section 10):

- ``label_permutation``: permute the outer-training labels, which preserves
  every class support exactly, compute the permuted-class centroids, and score
  Contrast/Representation on the Voronoi partition those centroids induce.
- ``counterfactual_cell_centroids``: draw 13 sites with the same itemwise
  uniform-without-replacement mechanism as the pilot, induce outer-training
  cells, replace each site by the centroid of its induced cell, re-induce the
  partition, and score the centroid constellation.

Everything is defined on outer-training partitions only.  No outer-test item
influences any transformation, site, centroid, or score: the space transform
is fitted through the existing ``_fit_space_transform`` contract without a
test slice so that no outer-test row can enter the null definition.

The output is strictly descriptive.  It is not a confirmatory H-CR4 test:
the protocol requires a fidelity criterion fixed in advance to be met before
observed percentiles are interpreted, and current centroid-Voronoi fidelity
is low.  This module therefore refuses to produce Monte-Carlo comparisons
whenever the observed class centroids induce an empty ordinary cell; it never
substitutes a power-diagram or otherwise repaired score.

This is new reconstruction code.  None of it is recovered historical source.
"""

from __future__ import annotations

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
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)

from .counterfactual import _fit_space_transform, _fold_seed_component
from .experiment_a import _dataframe_digest, _sha256_array, _sha256_file
from .experiment_c import _aligned_outer, validate_crowd_representation_probe
from .geometry import (
    assign_domain_points_to_nearest_sites,
    contrast_representation_scores,
    contrast_sum_pairwise_distances,
    mean_pairwise_site_distance,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Space = Literal["A_STANDARDIZED", "H_PCA"]

NULL_FORMAT = "frozen-emotion-spaces-matched-nulls-reconstruction-v1"
NULL_FILES = ("draws.parquet", "fold_summaries.parquet", "metadata.json")
MECHANISMS = ("label_permutation", "counterfactual_cell_centroids")
# Stream codes 0 and 1 are used by the pilot for site/repetition draws; the
# null mechanisms use distinct codes so no stream is ever reused.
_MECHANISM_CODES = {"label_permutation": 2, "counterfactual_cell_centroids": 3}
PRECONDITION_SATISFIED = "satisfied"
PRECONDITION_EMPTY = "empty_observed_cells"

_REJECTION_FIELDS = (
    "n_rejected_identity_arrangement",
    "n_rejected_empty_induced_cells",
    "n_rejected_duplicate_sites",
    "n_rejected_empty_initial_cells",
    "n_rejected_empty_reinduced_cells",
)


@dataclass(frozen=True)
class MatchedNullArtifact:
    directory: Path
    metadata: dict[str, Any]
    draws: pd.DataFrame
    fold_summaries: pd.DataFrame


@dataclass(frozen=True)
class _NullAttempt:
    """One null-generation attempt; exactly one rejection reason or none."""

    rejection: str | None
    sites: FloatArray | None = None
    permuted_labels: IntArray | None = None
    initial_site_indices: IntArray | None = None
    initial_contrast_sum: float = np.nan
    initial_representation_sum: float = np.nan


def run_observed_matched_nulls(
    output_directory: str | Path,
    *,
    source_run: str | Path,
    features: np.ndarray,
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    space: Space,
    pca_components: int | None = None,
    n_draws_per_fold: int = 1000,
    seed: int = 20240804,
    max_attempts_per_draw: int = 100,
) -> MatchedNullArtifact:
    """Generate both mechanism-matched nulls on outer-training folds only.

    The number of sites is fixed to the number of canonical source classes
    (13 for the locked crowd target).  Each accepted draw is scored with the
    source-faithful Contrast and Representation sums.  Observed class
    centroids are scored once per fold; plus-one Monte-Carlo comparisons and
    the joint domination fraction are reported per fold and mechanism only
    when the observed ordinary cells are all nonempty.
    """

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite matched-null artifact: {output}")
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
    for name, value, minimum in (
        ("n_draws_per_fold", n_draws_per_fold, 1),
        ("max_attempts_per_draw", max_attempts_per_draw, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    matrix = np.asarray(features, dtype=np.float64)
    ids = np.asarray([str(value) for value in item_ids], dtype=str)
    if matrix.shape != (
        int(artifact.metadata["n_items"]), int(artifact.metadata["n_features"])
    ):
        raise ValueError("matched-null features disagree with source-run dimensions")
    if ids.shape != (matrix.shape[0],) or np.unique(ids).size != ids.size:
        raise ValueError("item_ids must be unique and aligned with features")
    if not np.isfinite(matrix).all():
        raise ValueError("matched-null features contain non-finite values")
    if _sha256_array(matrix) != artifact.metadata["feature_matrix_sha256"]:
        raise ValueError("feature matrix hash disagrees with source run")
    if _dataframe_digest(outer_folds) != artifact.metadata["outer_split_sha256"]:
        raise ValueError("outer split hash disagrees with source run")
    class_names = tuple(str(value) for value in artifact.metadata["class_names"])
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError("matched nulls require at least two canonical classes")
    target_by_id = artifact.oof.set_index("item_id", verify_integrity=True)["y_true"]
    try:
        targets = np.asarray([str(target_by_id.at[item_id]) for item_id in ids])
    except KeyError as error:
        raise ValueError("matched-null item IDs disagree with source OOF") from error

    outer, groups, unique_outer = _aligned_outer(ids, outer_folds)
    if n_classes >= min(int(np.sum(outer != fold)) for fold in unique_outer):
        raise ValueError("site count must be smaller than every outer-training domain")

    draw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_rejections = 0
    output_dimension: int | None = None

    for outer_fold in unique_outer:
        train_mask = outer != outer_fold
        test_mask = ~train_mask
        if set(groups[train_mask]) & set(groups[test_mask]):
            raise ValueError(f"outer fold {outer_fold!r} leaks groups")
        # Train-only transform mode: no outer-test row is transformed or
        # inspected, so the null distribution is train-defined by construction.
        train, _, _ = _fit_space_transform(
            matrix[train_mask],
            space=space,
            pca_components=pca_components,
        )
        if output_dimension is None:
            output_dimension = train.shape[1]
        elif output_dimension != train.shape[1]:  # pragma: no cover - fixed config
            raise RuntimeError("transformed dimensions vary across folds")
        encoded_targets = _encode_targets(targets[train_mask], class_names)
        fold_draws, fold_stats, observed = _fold_null_draws(
            train,
            encoded_targets,
            ids[train_mask],
            seed=seed,
            fold_code=_fold_seed_component(outer_fold),
            n_classes=n_classes,
            n_draws=n_draws_per_fold,
            max_attempts=max_attempts_per_draw,
        )
        for row in fold_draws:
            row["draw_id"] = f"{outer_fold}:{row['draw_id']}"
            row["outer_fold"] = outer_fold
        draw_rows.extend(fold_draws)
        total_rejections += sum(stats["rejected"] for stats in fold_stats.values())
        for mechanism in MECHANISMS:
            summary_rows.append(
                _fold_summary_row(
                    outer_fold,
                    mechanism,
                    fold_draws=[row for row in fold_draws if row["mechanism"] == mechanism],
                    stats=fold_stats[mechanism],
                    observed=observed,
                    n_train=len(train),
                    dimension=train.shape[1],
                    n_classes=n_classes,
                )
            )

    draws = pd.DataFrame(draw_rows).sort_values(
        ["outer_fold", "mechanism", "draw_number"], kind="stable"
    ).reset_index(drop=True)
    fold_summaries = pd.DataFrame(summary_rows).sort_values(
        ["outer_fold", "mechanism"], kind="stable"
    ).reset_index(drop=True)
    column_order = [
        "draw_id", "outer_fold", "mechanism", "draw_number", "attempts",
        "contrast_sum", "representation_sum", "mean_pairwise_site_distance",
        "mean_site_to_cell_centroid_distance", "cell_supports_json",
        "cell_support_entropy_bits", "cell_support_entropy_normalized",
        "initial_site_item_ids_json", "initial_contrast_sum",
        "initial_representation_sum",
    ]
    draws = draws[column_order]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        draws_path = temporary / "draws.parquet"
        summaries_path = temporary / "fold_summaries.parquet"
        draws.to_parquet(draws_path, index=False, engine="pyarrow", compression="zstd")
        fold_summaries.to_parquet(
            summaries_path, index=False, engine="pyarrow", compression="zstd"
        )
        file_records = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (draws_path, summaries_path)
        }
        metadata = {
            "null_format": NULL_FORMAT,
            "status": (
                "descriptive_mechanism_matched_null_new_replication_not_confirmatory"
            ),
            "inference_ceiling": (
                "descriptive only; a confirmatory H-CR4 reading requires a fidelity "
                "criterion fixed in advance to be met, and these comparisons never "
                "substitute a power-diagram or otherwise repaired score when the "
                "observed ordinary cells are empty"
            ),
            "dataset": artifact.metadata["dataset"],
            "space": space,
            "source_representation": artifact.metadata["representation"],
            "source_run_metadata_sha256": _sha256_file(
                artifact.directory / "metadata.json"
            ),
            "source_feature_matrix_sha256": artifact.metadata["feature_matrix_sha256"],
            "source_outer_split_sha256": artifact.metadata["outer_split_sha256"],
            "source_embedding_layer_sha256": artifact.metadata[
                "embedding_layer_sha256"
            ],
            "class_names": list(class_names),
            "n_items": len(ids),
            "input_dimension": matrix.shape[1],
            "output_dimension": int(output_dimension or 0),
            "pca_components": pca_components,
            "n_sites": n_classes,
            "n_outer_folds": len(unique_outer),
            "mechanisms": list(MECHANISMS),
            "n_draws_per_fold": n_draws_per_fold,
            "max_attempts_per_draw": max_attempts_per_draw,
            "rejected_attempts": total_rejections,
            "seed": seed,
            "evaluation_domain": "outer_training_partitions_only_no_outer_test_use",
            "site_score_definition": (
                "IgorDouven/Concept_Learning calc(), commit "
                "2325717f68f9eecbc85cfa7d7e5ada0dc7e95679"
            ),
            "favorable_direction": (
                "higher Contrast and lower Representation are favorable; a null "
                "draw jointly dominates the observed constellation when "
                "contrast_null >= contrast_observed and "
                "representation_null <= representation_observed"
            ),
            "p_value_definition": (
                "plus-one Monte-Carlo: (1 + count of accepted null draws at least "
                "as favorable as observed) / (1 + accepted draws), computed "
                "separately for Contrast and Representation; the joint domination "
                "fraction is the raw share of jointly dominating draws"
            ),
            "implementation_sha256": {
                "counterfactual_nulls.py": _sha256_file(Path(__file__)),
                "counterfactual.py": _sha256_file(
                    Path(__file__).with_name("counterfactual.py")
                ),
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
        try:
            temporary.rename(output)
        except OSError as error:
            if output.exists():
                raise FileExistsError(
                    f"refusing to overwrite matched-null artifact: {output}"
                ) from error
            raise
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_observed_matched_nulls(output)


def validate_observed_matched_nulls(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> MatchedNullArtifact:
    """Semantically revalidate a matched-null artifact, not just its hashes."""

    root = Path(directory)
    missing = [name for name in NULL_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"partial matched-null artifact; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("matched-null metadata is unreadable") from error
    if (
        metadata.get("null_format") != NULL_FORMAT
        or metadata.get("status")
        != "descriptive_mechanism_matched_null_new_replication_not_confirmatory"
        or metadata.get("space") not in {"A_STANDARDIZED", "H_PCA"}
        or metadata.get("mechanisms") != list(MECHANISMS)
    ):
        raise ValueError("matched-null identity is invalid")
    for field in (
        "source_run_metadata_sha256", "source_feature_matrix_sha256",
        "source_outer_split_sha256",
    ):
        if not _is_sha256(metadata.get(field)):
            raise ValueError(f"matched-null metadata lacks {field}")
    if metadata["space"] == "H_PCA" and not _is_sha256(
        metadata.get("source_embedding_layer_sha256")
    ):
        raise ValueError("matched-null H_PCA metadata lacks its embedding identity")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("matched-null metadata lacks file records")
    for filename in NULL_FILES[:-1]:
        record = files.get(filename)
        path = root / filename
        if not isinstance(record, Mapping) or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"matched-null file size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"matched-null file hash mismatch: {filename}")
    try:
        draws = pd.read_parquet(root / "draws.parquet")
        fold_summaries = pd.read_parquet(root / "fold_summaries.parquet")
    except Exception as error:
        raise ValueError("matched-null parquet is unreadable") from error

    positive_integer_fields = (
        "n_outer_folds", "n_sites", "n_draws_per_fold", "n_items",
        "input_dimension", "output_dimension", "max_attempts_per_draw",
    )
    for field in positive_integer_fields:
        value = metadata.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"matched-null metadata lacks valid {field}")
    for field in ("seed", "rejected_attempts"):
        value = metadata.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"matched-null metadata lacks valid {field}")
    n_folds = metadata["n_outer_folds"]
    n_sites = metadata["n_sites"]
    n_draws = metadata["n_draws_per_fold"]
    input_dimension = metadata["input_dimension"]
    output_dimension = metadata["output_dimension"]
    pca_components = metadata.get("pca_components")
    if metadata["space"] == "A_STANDARDIZED":
        if (
            pca_components is not None
            or output_dimension != input_dimension
            or metadata.get("source_embedding_layer_sha256") is not None
        ):
            raise ValueError("matched-null A space dimensions are inconsistent")
    elif (
        not isinstance(pca_components, int)
        or isinstance(pca_components, bool)
        or not 1 <= pca_components <= input_dimension
        or output_dimension != pca_components
    ):
        raise ValueError("matched-null H PCA dimensions are inconsistent")
    class_names = tuple(str(value) for value in metadata.get("class_names", ()))
    if len(class_names) != n_sites or n_sites < 2:
        raise ValueError("matched-null class axis is inconsistent")
    n_pairs = n_sites * (n_sites - 1) // 2

    required_draws = {
        "draw_id", "outer_fold", "mechanism", "draw_number", "attempts",
        "contrast_sum", "representation_sum", "mean_pairwise_site_distance",
        "mean_site_to_cell_centroid_distance", "cell_supports_json",
        "cell_support_entropy_bits", "cell_support_entropy_normalized",
        "initial_site_item_ids_json", "initial_contrast_sum",
        "initial_representation_sum",
    }
    required_summaries = {
        "outer_fold", "mechanism", "n_train", "dimension", "n_draws_accepted",
        "n_attempts_total", "n_draws_rejected", "rejection_rate",
        *_REJECTION_FIELDS,
        "observed_label_supports_json", "observed_label_entropy_bits",
        "observed_label_entropy_normalized", "observed_cell_supports_json",
        "observed_cell_entropy_bits", "observed_cell_entropy_normalized",
        "observed_n_empty_cells", "observed_contrast_sum",
        "observed_mean_pairwise_site_distance", "observed_representation_sum",
        "observed_mean_site_to_cell_centroid_distance",
        "observed_train_macro_f1", "observed_train_accuracy",
        "observed_train_adjusted_rand", "observed_train_normalized_mutual_info",
        "fidelity_precondition", "n_contrast_at_least_observed",
        "n_representation_at_most_observed", "n_jointly_dominating",
        "contrast_p_plus1", "representation_p_plus1", "joint_domination_fraction",
    }
    if not required_draws.issubset(draws) or not required_summaries.issubset(
        fold_summaries
    ):
        raise ValueError("matched-null parquet schema is incomplete")
    if (
        len(draws) != n_folds * len(MECHANISMS) * n_draws
        or draws["draw_id"].duplicated().any()
        or set(draws["mechanism"]) != set(MECHANISMS)
        or len(fold_summaries) != n_folds * len(MECHANISMS)
        or fold_summaries.duplicated(["outer_fold", "mechanism"]).any()
        or set(fold_summaries["mechanism"]) != set(MECHANISMS)
    ):
        raise ValueError("matched-null row coverage is invalid")
    coverage = draws.groupby(["outer_fold", "mechanism"])["draw_number"]
    if any(set(values.astype(int)) != set(range(n_draws)) for _, values in coverage):
        raise ValueError("matched-null draw coverage is invalid")

    summary_by_key = fold_summaries.set_index(
        ["outer_fold", "mechanism"], verify_integrity=True
    )
    for row in draws.itertuples(index=False):
        key = (row.outer_fold, row.mechanism)
        if key not in summary_by_key.index:
            raise ValueError("matched-null draw lacks its fold summary")
        n_train = int(summary_by_key.at[key, "n_train"])
        try:
            supports = np.asarray(json.loads(row.cell_supports_json), dtype=np.int64)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("matched-null cell supports are invalid") from error
        if (
            supports.shape != (n_sites,)
            or (supports <= 0).any()
            or int(supports.sum()) != n_train
        ):
            raise ValueError("matched-null cell supports are inconsistent")
        if not np.isclose(
            float(row.cell_support_entropy_bits), _support_entropy_bits(supports),
            rtol=1e-12, atol=1e-12,
        ) or not np.isclose(
            float(row.cell_support_entropy_normalized),
            _support_entropy_bits(supports) / np.log2(n_sites),
            rtol=1e-12, atol=1e-12,
        ):
            raise ValueError("matched-null support entropy is inconsistent")
        if (
            not np.isfinite(row.contrast_sum)
            or not np.isfinite(row.representation_sum)
            or row.contrast_sum < 0
            or row.representation_sum < 0
            or not np.isclose(
                float(row.mean_pairwise_site_distance) * n_pairs,
                float(row.contrast_sum), rtol=1e-9, atol=1e-12,
            )
            or not np.isclose(
                float(row.mean_site_to_cell_centroid_distance) * n_sites,
                float(row.representation_sum), rtol=1e-9, atol=1e-12,
            )
            or int(row.attempts) < 1
        ):
            raise ValueError("matched-null draw scores are inconsistent")
        if row.mechanism == "label_permutation":
            if (
                row.initial_site_item_ids_json is not None
                and not (
                    isinstance(row.initial_site_item_ids_json, float)
                    and np.isnan(row.initial_site_item_ids_json)
                )
            ):
                raise ValueError("permutation draws must not record site items")
            if np.isfinite(row.initial_contrast_sum) or np.isfinite(
                row.initial_representation_sum
            ):
                raise ValueError("permutation draws must not record initial scores")
        else:
            try:
                site_ids = json.loads(row.initial_site_item_ids_json)
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError("cell-centroid draw site items are invalid") from error
            if (
                not isinstance(site_ids, list)
                or len(site_ids) != n_sites
                or len(set(str(value) for value in site_ids)) != n_sites
                or not np.isfinite(row.initial_contrast_sum)
                or not np.isfinite(row.initial_representation_sum)
                or row.initial_contrast_sum < 0
                or row.initial_representation_sum < 0
            ):
                raise ValueError("cell-centroid draw provenance is inconsistent")

    for row in fold_summaries.itertuples(index=False):
        subset = draws.loc[
            (draws["outer_fold"] == row.outer_fold)
            & (draws["mechanism"] == row.mechanism)
        ]
        n_accepted = len(subset)
        attempts_total = int(subset["attempts"].sum())
        rejected = attempts_total - n_accepted
        breakdown = {field: int(getattr(row, field)) for field in _REJECTION_FIELDS}
        if row.mechanism == "label_permutation":
            applicable = (
                "n_rejected_identity_arrangement", "n_rejected_empty_induced_cells",
            )
        else:
            applicable = (
                "n_rejected_duplicate_sites", "n_rejected_empty_initial_cells",
                "n_rejected_empty_reinduced_cells",
            )
        if (
            n_accepted != int(row.n_draws_accepted)
            or attempts_total != int(row.n_attempts_total)
            or rejected != int(row.n_draws_rejected)
            or sum(breakdown[field] for field in applicable) != rejected
            or any(breakdown[field] != 0 for field in _REJECTION_FIELDS if field not in applicable)
            or not np.isclose(
                float(row.rejection_rate),
                rejected / attempts_total if attempts_total else 0.0,
                rtol=1e-12, atol=1e-12,
            )
        ):
            raise ValueError("matched-null attempt/rejection accounting is inconsistent")
        try:
            label_supports = np.asarray(
                json.loads(row.observed_label_supports_json), dtype=np.int64
            )
            cell_supports = np.asarray(
                json.loads(row.observed_cell_supports_json), dtype=np.int64
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("matched-null observed supports are invalid") from error
        if (
            label_supports.shape != (n_sites,)
            or (label_supports <= 0).any()
            or int(label_supports.sum()) != int(row.n_train)
            or cell_supports.shape != (n_sites,)
            or (cell_supports < 0).any()
            or int(cell_supports.sum()) != int(row.n_train)
            or int(np.sum(cell_supports == 0)) != int(row.observed_n_empty_cells)
            or not np.isclose(
                float(row.observed_label_entropy_bits),
                _support_entropy_bits(label_supports), rtol=1e-12, atol=1e-12,
            )
            or not np.isclose(
                float(row.observed_cell_entropy_bits),
                _support_entropy_bits(cell_supports), rtol=1e-12, atol=1e-12,
            )
            or int(row.n_train) <= n_sites
            or int(row.dimension) < 1
        ):
            raise ValueError("matched-null observed supports are inconsistent")
        if not np.isclose(
            float(row.observed_mean_pairwise_site_distance) * n_pairs,
            float(row.observed_contrast_sum), rtol=1e-9, atol=1e-12,
        ):
            raise ValueError("matched-null observed Contrast is inconsistent")
        for column in (
            "observed_train_macro_f1", "observed_train_accuracy",
            "observed_train_normalized_mutual_info",
        ):
            value = float(getattr(row, column))
            if not -1e-12 <= value <= 1.0 + 1e-12:
                raise ValueError(f"matched-null {column} is outside [0, 1]")
        ari = float(row.observed_train_adjusted_rand)
        if not -1.0 - 1e-12 <= ari <= 1.0 + 1e-12:
            raise ValueError("matched-null observed adjusted Rand is outside [-1, 1]")
        if row.fidelity_precondition == PRECONDITION_SATISFIED:
            n_c = int(np.sum(subset["contrast_sum"] >= float(row.observed_contrast_sum)))
            n_r = int(
                np.sum(subset["representation_sum"] <= float(row.observed_representation_sum))
            )
            n_j = int(
                np.sum(
                    (subset["contrast_sum"] >= float(row.observed_contrast_sum))
                    & (subset["representation_sum"] <= float(row.observed_representation_sum))
                )
            )
            if (
                not np.isfinite(row.observed_representation_sum)
                or int(row.observed_n_empty_cells) != 0
                or n_c != int(row.n_contrast_at_least_observed)
                or n_r != int(row.n_representation_at_most_observed)
                or n_j != int(row.n_jointly_dominating)
                or not np.isclose(
                    float(row.contrast_p_plus1), (1 + n_c) / (1 + n_accepted),
                    rtol=1e-12, atol=1e-15,
                )
                or not np.isclose(
                    float(row.representation_p_plus1), (1 + n_r) / (1 + n_accepted),
                    rtol=1e-12, atol=1e-15,
                )
                or not np.isclose(
                    float(row.joint_domination_fraction), n_j / n_accepted,
                    rtol=1e-12, atol=1e-15,
                )
            ):
                raise ValueError("matched-null Monte-Carlo comparisons are inconsistent")
        elif row.fidelity_precondition == PRECONDITION_EMPTY:
            if (
                np.isfinite(row.observed_representation_sum)
                or np.isfinite(row.contrast_p_plus1)
                or np.isfinite(row.representation_p_plus1)
                or np.isfinite(row.joint_domination_fraction)
                or int(row.observed_n_empty_cells) < 1
                or int(row.n_contrast_at_least_observed) != -1
                or int(row.n_representation_at_most_observed) != -1
                or int(row.n_jointly_dominating) != -1
            ):
                raise ValueError("matched-null fidelity precondition state is invalid")
        else:
            raise ValueError("matched-null fidelity precondition is unknown")
    observed_keys = (
        "observed_label_supports_json", "observed_cell_supports_json",
        "observed_contrast_sum", "observed_representation_sum",
        "observed_n_empty_cells", "fidelity_precondition", "n_train", "dimension",
    )
    for _, fold_rows in fold_summaries.groupby("outer_fold"):
        for key in observed_keys:
            values = fold_rows[key].astype(str)
            if values.nunique() != 1:
                raise ValueError("matched-null observed rows disagree across mechanisms")
    return MatchedNullArtifact(root, dict(metadata), draws, fold_summaries)


def _fold_null_draws(
    train: FloatArray,
    encoded_targets: IntArray,
    train_ids: NDArray[np.str_],
    *,
    seed: int,
    fold_code: int,
    n_classes: int,
    n_draws: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    """Draw both nulls for one outer fold from train-only arrays.

    The signature intentionally has no outer-test parameter: the null
    definition cannot observe the held-out partition.
    """

    observed = _observed_fold_geometry(train, encoded_targets, n_classes=n_classes)
    draw_rows: list[dict[str, Any]] = []
    stats = {
        mechanism: {"attempts": 0, "rejected": 0, **{field: 0 for field in _REJECTION_FIELDS}}
        for mechanism in MECHANISMS
    }
    for mechanism in MECHANISMS:
        for draw_number in range(n_draws):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, fold_code, draw_number, _MECHANISM_CODES[mechanism]]
                )
            )
            accepted: _NullAttempt | None = None
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                if mechanism == "label_permutation":
                    attempt = _attempt_permutation_null(
                        train, encoded_targets, n_classes=n_classes, rng=rng
                    )
                else:
                    attempt = _attempt_cell_centroid_null(
                        train, n_sites=n_classes, rng=rng
                    )
                if attempt.rejection is None:
                    accepted = attempt
                    break
                stats[mechanism][f"n_rejected_{attempt.rejection}"] += 1
            stats[mechanism]["attempts"] += attempts
            stats[mechanism]["rejected"] += attempts - (1 if accepted else 0)
            if accepted is None:
                raise RuntimeError(
                    f"could not draw a valid {mechanism} null "
                    f"after {max_attempts} attempts"
                )
            scores = contrast_representation_scores(accepted.sites, domain_points=train)
            supports = np.bincount(scores.assignments, minlength=n_classes)
            entropy = _support_entropy_bits(supports)
            draw_rows.append(
                {
                    "draw_id": f"{mechanism}:{draw_number:05d}",
                    "mechanism": mechanism,
                    "draw_number": draw_number,
                    "attempts": attempts,
                    "contrast_sum": scores.contrast_sum,
                    "representation_sum": scores.representation_sum,
                    "mean_pairwise_site_distance": scores.mean_pairwise_site_distance,
                    "mean_site_to_cell_centroid_distance": (
                        scores.mean_site_to_cell_centroid_distance
                    ),
                    "cell_supports_json": json.dumps(supports.tolist()),
                    "cell_support_entropy_bits": entropy,
                    "cell_support_entropy_normalized": entropy / np.log2(n_classes),
                    "initial_site_item_ids_json": (
                        json.dumps(train_ids[accepted.initial_site_indices].tolist())
                        if accepted.initial_site_indices is not None
                        else None
                    ),
                    "initial_contrast_sum": accepted.initial_contrast_sum,
                    "initial_representation_sum": accepted.initial_representation_sum,
                }
            )
    return draw_rows, stats, observed


def _attempt_permutation_null(
    train: FloatArray,
    encoded_targets: IntArray,
    *,
    n_classes: int,
    rng: np.random.Generator,
) -> _NullAttempt:
    """Permute train labels; class supports are preserved by construction.

    Any permutation that leaves the label vector unchanged is rejected: the
    null must never contain the observed arrangement itself.  Permuted-class
    centroids whose induced ordinary partition has an empty cell are also
    rejected, because the source-faithful Representation score is undefined
    there.
    """

    permutation = rng.permutation(len(train))
    permuted = np.asarray(encoded_targets[permutation], dtype=np.int64)
    if np.array_equal(permuted, encoded_targets):
        return _NullAttempt(rejection="identity_arrangement")
    sites = _class_centroids(train, permuted, n_classes)
    assignments = assign_domain_points_to_nearest_sites(
        sites, domain_points=train, require_nonempty_cells=False
    )
    if (np.bincount(assignments, minlength=n_classes) == 0).any():
        return _NullAttempt(rejection="empty_induced_cells")
    return _NullAttempt(rejection=None, sites=sites, permuted_labels=permuted)


def _attempt_cell_centroid_null(
    train: FloatArray,
    *,
    n_sites: int,
    rng: np.random.Generator,
) -> _NullAttempt:
    """Draw itemwise sites like the pilot, then replace them by cell centroids."""

    site_indices = np.asarray(
        rng.choice(len(train), size=n_sites, replace=False), dtype=np.int64
    )
    sites = train[site_indices]
    if np.unique(sites, axis=0).shape[0] != n_sites:
        return _NullAttempt(rejection="duplicate_sites")
    initial_assignments = assign_domain_points_to_nearest_sites(
        sites, domain_points=train, require_nonempty_cells=False
    )
    if (np.bincount(initial_assignments, minlength=n_sites) == 0).any():
        return _NullAttempt(rejection="empty_initial_cells")
    initial_scores = contrast_representation_scores(sites, domain_points=train)
    centroids = initial_scores.centroids
    reinduced = assign_domain_points_to_nearest_sites(
        centroids, domain_points=train, require_nonempty_cells=False
    )
    if (np.bincount(reinduced, minlength=n_sites) == 0).any():
        return _NullAttempt(rejection="empty_reinduced_cells")
    return _NullAttempt(
        rejection=None,
        sites=centroids,
        initial_site_indices=site_indices,
        initial_contrast_sum=initial_scores.contrast_sum,
        initial_representation_sum=initial_scores.representation_sum,
    )


def _observed_fold_geometry(
    train: FloatArray,
    encoded_targets: IntArray,
    *,
    n_classes: int,
) -> dict[str, Any]:
    """Score observed class centroids on their induced train partition."""

    label_supports = np.bincount(encoded_targets, minlength=n_classes)
    if (label_supports == 0).any():
        raise ValueError("an outer-training partition lacks a canonical class")
    sites = _class_centroids(train, encoded_targets, n_classes)
    contrast = contrast_sum_pairwise_distances(sites, domain_points=train)
    contrast_mean = mean_pairwise_site_distance(sites, domain_points=train)
    assignments = assign_domain_points_to_nearest_sites(
        sites, domain_points=train, require_nonempty_cells=False
    )
    cell_supports = np.bincount(assignments, minlength=n_classes)
    n_empty = int(np.sum(cell_supports == 0))
    representation = np.nan
    representation_mean = np.nan
    if n_empty == 0:
        scores = contrast_representation_scores(sites, domain_points=train)
        representation = scores.representation_sum
        representation_mean = scores.mean_site_to_cell_centroid_distance
    return {
        "label_supports": label_supports,
        "cell_supports": cell_supports,
        "n_empty_cells": n_empty,
        "contrast_sum": contrast,
        "mean_pairwise_site_distance": contrast_mean,
        "representation_sum": representation,
        "mean_site_to_cell_centroid_distance": representation_mean,
        "train_macro_f1": float(
            f1_score(
                encoded_targets, assignments, labels=list(range(n_classes)),
                average="macro", zero_division=0,
            )
        ),
        "train_accuracy": float(accuracy_score(encoded_targets, assignments)),
        "train_adjusted_rand": float(
            adjusted_rand_score(encoded_targets, assignments)
        ),
        "train_normalized_mutual_info": float(
            normalized_mutual_info_score(encoded_targets, assignments)
        ),
    }


def _fold_summary_row(
    outer_fold: Any,
    mechanism: str,
    *,
    fold_draws: list[dict[str, Any]],
    stats: dict[str, int],
    observed: Mapping[str, Any],
    n_train: int,
    dimension: int,
    n_classes: int,
) -> dict[str, Any]:
    n_accepted = len(fold_draws)
    attempts_total = int(stats["attempts"])
    rejected = int(stats["rejected"])
    label_entropy = _support_entropy_bits(observed["label_supports"])
    cell_entropy = _support_entropy_bits(observed["cell_supports"])
    satisfied = int(observed["n_empty_cells"]) == 0
    if satisfied:
        null_contrast = np.asarray(
            [row["contrast_sum"] for row in fold_draws], dtype=np.float64
        )
        null_representation = np.asarray(
            [row["representation_sum"] for row in fold_draws], dtype=np.float64
        )
        n_c = int(np.sum(null_contrast >= observed["contrast_sum"]))
        n_r = int(np.sum(null_representation <= observed["representation_sum"]))
        n_j = int(
            np.sum(
                (null_contrast >= observed["contrast_sum"])
                & (null_representation <= observed["representation_sum"])
            )
        )
        contrast_p = (1 + n_c) / (1 + n_accepted)
        representation_p = (1 + n_r) / (1 + n_accepted)
        joint = n_j / n_accepted
    else:
        n_c = n_r = n_j = -1
        contrast_p = representation_p = joint = np.nan
    return {
        "outer_fold": outer_fold,
        "mechanism": mechanism,
        "n_train": n_train,
        "dimension": dimension,
        "n_draws_accepted": n_accepted,
        "n_attempts_total": attempts_total,
        "n_draws_rejected": rejected,
        **{field: int(stats[field]) for field in _REJECTION_FIELDS},
        "rejection_rate": rejected / attempts_total if attempts_total else 0.0,
        "observed_label_supports_json": json.dumps(
            observed["label_supports"].tolist()
        ),
        "observed_label_entropy_bits": label_entropy,
        "observed_label_entropy_normalized": label_entropy / np.log2(n_classes),
        "observed_cell_supports_json": json.dumps(observed["cell_supports"].tolist()),
        "observed_cell_entropy_bits": cell_entropy,
        "observed_cell_entropy_normalized": cell_entropy / np.log2(n_classes),
        "observed_n_empty_cells": int(observed["n_empty_cells"]),
        "observed_contrast_sum": observed["contrast_sum"],
        "observed_mean_pairwise_site_distance": observed[
            "mean_pairwise_site_distance"
        ],
        "observed_representation_sum": observed["representation_sum"],
        "observed_mean_site_to_cell_centroid_distance": observed[
            "mean_site_to_cell_centroid_distance"
        ],
        "observed_train_macro_f1": observed["train_macro_f1"],
        "observed_train_accuracy": observed["train_accuracy"],
        "observed_train_adjusted_rand": observed["train_adjusted_rand"],
        "observed_train_normalized_mutual_info": observed[
            "train_normalized_mutual_info"
        ],
        "fidelity_precondition": (
            PRECONDITION_SATISFIED if satisfied else PRECONDITION_EMPTY
        ),
        "n_contrast_at_least_observed": n_c,
        "n_representation_at_most_observed": n_r,
        "n_jointly_dominating": n_j,
        "contrast_p_plus1": contrast_p,
        "representation_p_plus1": representation_p,
        "joint_domination_fraction": joint,
    }


def _class_centroids(
    train: FloatArray, encoded_labels: IntArray, n_classes: int
) -> FloatArray:
    return np.vstack(
        [train[encoded_labels == label].mean(axis=0) for label in range(n_classes)]
    )


def _encode_targets(
    targets: np.ndarray, class_names: tuple[str, ...]
) -> IntArray:
    index = {name: position for position, name in enumerate(class_names)}
    try:
        return np.asarray([index[str(value)] for value in targets], dtype=np.int64)
    except KeyError as error:
        raise ValueError("targets contain a label outside the canonical axis") from error


def _support_entropy_bits(supports: np.ndarray) -> float:
    total = float(np.sum(supports))
    if total <= 0:
        raise ValueError("support entropy requires a non-empty partition")
    probabilities = np.asarray(supports, dtype=np.float64) / total
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "MatchedNullArtifact",
    "MECHANISMS",
    "NULL_FILES",
    "NULL_FORMAT",
    "run_observed_matched_nulls",
    "validate_observed_matched_nulls",
]
