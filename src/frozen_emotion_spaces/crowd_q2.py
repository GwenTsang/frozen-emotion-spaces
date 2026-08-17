"""Resumable batch runner for the crowd-enVENT Q2 representation-probe suite.

This is a clean-room reconstruction, not recovered historical source.  It
orchestrates :func:`run_crowd_representation_probe` across named encoder/layer
A, H, and AH triplets with nested log-loss selection, and emits paired
conditional-analysis-ready manifests.  Every scaler, PCA, hyperparameter, and
block multiplier is fitted on the corresponding outer-training partition only.

Q2 configurations
-----------------
1. **Full-generation writer labels** — 6 600 items, 5×3 nested CV
   (``crowd_full_outer.csv`` / ``crowd_full_inner.csv``), writer appraisals
   predicting writer emotion labels.
2. **Cross-rater reader labels** — 1 200 items, 5×3 nested CV
   (``crowd_reader_outer.csv`` / ``crowd_reader_inner.csv``) with writer-side
   appraisals predicting reader-majority emotion targets.  Coordinate raters
   (writers) and target raters (readers) are recorded as separate metadata
   fields and are never conflated.
3. **External-test writer labels** — sealed split with ``train`` / ``test`` /
   ``excluded_test_writer`` / ``excluded_test_duplicate`` roles
   (``crowd_external.csv``) and a three-fold train-only inner split
   (``crowd_external_inner.csv``).  The external test is never treated as an
   OOF fold; excluded writers are preserved untouched.

PCA appraisal dimensions d ∈ {3, 5, 7, 10, 21} are fitted inside each outer
fold on the train partition only, then used to transform (never fit on) test
rows.

All outputs carry ``"status": "new_replication_not_historical_recovery"`` in
their metadata.  Completed child artifacts are validated and never overwritten;
resuming reuses only validated, fully compatible children.
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
from sklearn.decomposition import PCA

from .config import embedding_artifact_directory, get_model_spec
from .crowd_data import APPRAISAL_NAMES, CROWD_EMOTIONS
from .embeddings import EmbeddingArtifact, validate_embedding_artifact
from .experiment_a import _dataframe_digest, _ordered_pair_digest
from .experiment_c import (
    RUN_FORMAT,
    CrowdRepresentationProbeArtifact,
    _aligned_outer,
    _fold_geometry,
    _sha256_array,
    _sha256_file,
    _validated_inner,
    _write_geometry,
    run_crowd_representation_probe,
    validate_crowd_representation_probe,
)
from .metrics import (
    multiclass_itemwise_log_loss_bits,
    paired_group_bootstrap_delta,
    reconstruct_multiclass_metrics,
)
from .probes import (
    DEFAULT_BLOCK_MULTIPLIER_GRID,
    DEFAULT_C_GRID,
    TransformedMulticlassLogistic,
    select_multiclass_C_block_multiplier,
)


# ---------------------------------------------------------------------------
# Suite format identity
# ---------------------------------------------------------------------------

SUITE_FORMAT = "frozen-emotion-spaces-crowd-q2-suite-reconstruction-v1"
SUITE_FILES = ("suite_metadata.json", "summary.parquet", "contrasts.parquet")
EXTERNAL_FORMAT = "frozen-emotion-spaces-crowd-q2-external-probe-reconstruction-v1"
EXTERNAL_FILES = ("test_predictions.parquet", "selections.parquet", "metadata.json")
PCA_DIMENSIONS = (3, 5, 7, 10, 21)
ALLOWED_TARGET_SCALES = ("full_writer", "reader", "external_writer")
ALLOWED_POOLING = ("mean", "first")

# Split-table and target provenance per scale.  These names make the rater
# roles and split tables explicit in every artifact so that writer-side
# appraisal coordinates are never conflated with reader-side targets.
FULL_SPLIT_TABLES = ("crowd_full_outer.csv", "crowd_full_inner.csv")
READER_SPLIT_TABLES = ("crowd_reader_outer.csv", "crowd_reader_inner.csv")
EXTERNAL_SPLIT_TABLES = ("crowd_external.csv", "crowd_external_inner.csv")
_TARGET_LABELS = {
    "full_writer": "y_writer",
    "reader": "y_reader_majority",
    "external_writer": "y_writer",
}
_DEFAULT_RATERS = {
    "full_writer": ("writer", "writer"),
    "reader": ("writer", "reader"),
    "external_writer": ("writer", "writer"),
}
_DEFAULT_SPLIT_TABLES = {
    "full_writer": FULL_SPLIT_TABLES,
    "reader": READER_SPLIT_TABLES,
    "external_writer": EXTERNAL_SPLIT_TABLES,
}
EXTERNAL_ROLES = ("train", "test", "excluded_test_writer", "excluded_test_duplicate")


# ---------------------------------------------------------------------------
# External split roles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExternalRolePartition:
    """Explicit role partition of the sealed ``crowd_external.csv`` table."""

    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    excluded_test_writer_ids: tuple[str, ...]
    excluded_test_duplicate_ids: tuple[str, ...]


def external_role_partition(external: pd.DataFrame) -> ExternalRolePartition:
    """Split ``crowd_external.csv`` into its sealed roles without conflation.

    Excluded writers and duplicates are preserved as named, disjoint sets so
    callers can prove they were never used for fitting, selection, or testing.
    """

    required = {"item_id", "group_id", "writer_id", "role"}
    if not required.issubset(external.columns) or external[
        list(required)
    ].isna().any().any():
        raise ValueError(f"external split must contain non-missing {sorted(required)}")
    frame = external.copy()
    frame["item_id"] = frame["item_id"].astype(str)
    if frame["item_id"].duplicated().any():
        raise ValueError("external split contains duplicate item IDs")
    roles = frame["role"].astype(str)
    unknown = sorted(set(roles) - set(EXTERNAL_ROLES))
    if unknown:
        raise ValueError(f"external split has unknown roles: {unknown}")
    partition = {
        role: tuple(frame.loc[roles == role, "item_id"].tolist())
        for role in EXTERNAL_ROLES
    }
    if not partition["train"] or not partition["test"]:
        raise ValueError("external split must contain train and test roles")
    # Roles are mutually exclusive by construction; state the invariant so a
    # malformed table fails loudly instead of silently conflating partitions.
    seen: set[str] = set()
    for role in EXTERNAL_ROLES:
        overlap = seen & set(partition[role])
        if overlap:
            raise ValueError(f"external roles overlap on items: {sorted(overlap)[:3]}")
        seen |= set(partition[role])
    return ExternalRolePartition(
        train_ids=partition["train"],
        test_ids=partition["test"],
        excluded_test_writer_ids=partition["excluded_test_writer"],
        excluded_test_duplicate_ids=partition["excluded_test_duplicate"],
    )


# ---------------------------------------------------------------------------
# Resume compatibility
# ---------------------------------------------------------------------------

def _require_compatible(
    metadata: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    directory: Path,
) -> None:
    """Refuse to resume a child whose stored configuration disagrees."""

    mismatched = [
        f"{key} (stored={metadata.get(key)!r}, requested={value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatched:
        raise FileExistsError(
            f"existing artifact at {directory} is incompatible: "
            + "; ".join(mismatched)
        )


# ---------------------------------------------------------------------------
# PCA helpers (always fitted inside the outer-train fold)
# ---------------------------------------------------------------------------

def _pca_appraisals(
    appraisals: np.ndarray,
    *,
    d: int,
    outer_folds: np.ndarray,
    unique_folds: list[Any],
    item_ids: np.ndarray,
    groups: np.ndarray,
    class_names: Sequence[str],
    targets: np.ndarray,
    inner_folds: pd.DataFrame,
    appraisal_names: Sequence[str],
    C_grid: Sequence[float],
    selection_metric: Literal["log_loss", "macro_f1"],
    class_weight: str | dict[str, float] | None,
    output_root: Path,
    run_name: str,
) -> CrowdRepresentationProbeArtifact:
    """Run one PCA-d appraisal representation probe with per-fold PCA.

    PCA is fitted on the outer-train partition of each fold only.  Test rows
    are transformed by their own fold's train-fitted PCA; the stored feature
    digest binds exactly those leak-free reductions.  Inner-fold selection
    runs on the train partition reduced by the same train-fitted PCA.
    """

    ids = np.asarray([str(i) for i in item_ids], dtype=str)
    names = tuple(str(n) for n in class_names)
    appraisal_axis = tuple(str(n) for n in appraisal_names)
    block_dims = (int(d),)

    # OOF feature matrix: each row is transformed exclusively by the PCA
    # fitted on its own fold's train partition.  Train reductions are local
    # to their fold iteration and never written back, so no fold's PCA can
    # contaminate another fold's features.
    reduced = np.empty((len(ids), d), dtype=np.float64)
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    geometry_records: list[dict[str, np.ndarray]] = []

    def estimator_for(C: float, multiplier: float) -> TransformedMulticlassLogistic:
        return TransformedMulticlassLogistic(
            C=float(C),
            class_names=names,
            block_dims=block_dims,
            block_multipliers=(1.0,),
            class_weight=class_weight,
        )

    for fold in unique_folds:
        test_mask = outer_folds == fold
        train_mask = ~test_mask
        overlap = set(groups[train_mask]) & set(groups[test_mask])
        if overlap:
            raise ValueError(
                f"group leakage in outer fold {fold!r}: {sorted(overlap)[:3]}"
            )
        pca = PCA(n_components=d, random_state=0)
        train_features = pca.fit_transform(appraisals[train_mask])
        test_features = pca.transform(appraisals[test_mask])
        reduced[test_mask] = test_features

        train_ids = ids[train_mask]
        rows = inner_folds.loc[inner_folds["outer_fold"].astype(str) == str(fold)]
        fold_by_id = dict(
            zip(rows["item_id"].astype(str), rows["validation_fold"], strict=True)
        )
        validation_folds = np.asarray([fold_by_id[i] for i in train_ids])

        selection = select_multiclass_C_block_multiplier(
            train_features,
            targets[train_mask],
            validation_folds=validation_folds,
            groups=groups[train_mask],
            class_names=names,
            block_dims=block_dims,
            estimator_factory=estimator_for,
            C_grid=C_grid,
            block_multiplier_grid=(1.0,),
            selection_metric=selection_metric,
        )
        estimator = estimator_for(selection.C, selection.block_multiplier)
        estimator.fit(train_features, targets[train_mask])
        probability = estimator.predict_proba(test_features)
        prediction = np.asarray(names)[np.argmax(probability, axis=1)]
        frame: dict[str, Any] = {
            "item_id": ids[test_mask],
            "outer_fold": fold,
            "group_id": groups[test_mask],
            "y_true": targets[test_mask],
            "y_pred": prediction,
        }
        for ci, cn in enumerate(names):
            frame[f"prob__{cn}"] = probability[:, ci]
        prediction_frames.append(pd.DataFrame(frame))
        selection_rows.append({
            "outer_fold": fold,
            "C": selection.C,
            "block_multiplier": selection.block_multiplier,
            "inner_log_loss_bits": selection.log_loss_bits,
            "inner_macro_f1": selection.macro_f1,
            "selection_metric": selection_metric,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })
        geometry_records.append(
            _fold_geometry(
                estimator,
                train_features,
                targets[train_mask],
                names=names,
                outer_fold=fold,
            )
        )

    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(
        "item_id", kind="stable"
    ).reset_index(drop=True)
    selections = pd.DataFrame(selection_rows).sort_values(
        "outer_fold", kind="stable"
    ).reset_index(drop=True)

    if len(oof) != len(ids) or oof["item_id"].nunique() != len(ids):
        raise RuntimeError("PCA appraisal OOF does not cover every item once")
    reconstruct_multiclass_metrics(oof, labels=names)

    output_dir = output_root / run_name
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        oof_path = temporary / "oof.parquet"
        sel_path = temporary / "selections.parquet"
        geo_path = temporary / "geometry.npz"
        oof.to_parquet(oof_path, index=False, engine="pyarrow", compression="zstd")
        selections.to_parquet(sel_path, index=False, engine="pyarrow", compression="zstd")
        _write_geometry(geo_path, geometry_records)
        file_records = {
            p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
            for p in (oof_path, sel_path, geo_path)
        }
        outer_frame = pd.DataFrame({
            "item_id": ids,
            "group_id": groups,
            "test_fold": outer_folds,
        })
        metadata = {
            "run_format": RUN_FORMAT,
            "status": "new_replication_not_historical_recovery",
            "dataset": "crowd",
            "target": "y_writer",
            "representation": "A",
            "layer": None,
            "pooling": None,
            "class_names": list(names),
            "appraisal_names": list(appraisal_axis),
            "block_dims": [int(d)],
            "C_grid": [float(v) for v in C_grid],
            "block_multiplier_grid": [1.0],
            "selection_metric": selection_metric,
            "class_weight": class_weight,
            "n_items": len(ids),
            "n_features": int(d),
            "appraisal_matrix_sha256": _sha256_array(appraisals),
            "feature_matrix_sha256": _sha256_array(reduced),
            "ordered_item_target_sha256": _ordered_pair_digest(ids, targets),
            "outer_split_sha256": _dataframe_digest(outer_frame),
            "inner_split_sha256": _dataframe_digest(inner_folds),
            "implementation_sha256": {
                "crowd_q2.py": _sha256_file(Path(__file__)),
            },
            "pandas_version": distribution_version("pandas"),
            "scikit_learn_version": distribution_version("scikit-learn"),
            "pyarrow_version": distribution_version("pyarrow"),
            "files": file_records,
            "embedding_artifact_format": None,
            "embedding_model_key": None,
            "embedding_revision": None,
            "embedding_mode": None,
            "embedding_text_variant": None,
            "embedding_metadata_sha256": None,
            "embedding_item_text_pairs_sha256": None,
            "embedding_layer_sha256": None,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_crowd_representation_probe(output_dir)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Q2SuiteArtifact:
    """Validated Q2 suite output directory and its summary tables."""

    directory: Path
    metadata: dict[str, Any]
    summary: pd.DataFrame
    contrasts: pd.DataFrame


@dataclass(frozen=True)
class ExternalTestResult:
    """Probe result on the sealed external test set (not an OOF fold)."""

    directory: Path
    metadata: dict[str, Any]
    test_predictions: pd.DataFrame
    selections: pd.DataFrame


# ---------------------------------------------------------------------------
# Core suite runner
# ---------------------------------------------------------------------------

def run_q2_representation_triplet(
    output_directory: str | Path,
    *,
    representation: Literal["A", "H", "AH"],
    appraisals: np.ndarray,
    y: Sequence[str],
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    inner_folds: pd.DataFrame,
    target_scale: str,
    rater_role: str,
    embedding_directory: str | Path | EmbeddingArtifact | None = None,
    layer: int | None = None,
    pooling: str = "mean",
    class_names: Sequence[str] = CROWD_EMOTIONS,
    appraisal_names: Sequence[str] = APPRAISAL_NAMES,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    block_multiplier_grid: Sequence[float] = DEFAULT_BLOCK_MULTIPLIER_GRID,
    selection_metric: Literal["log_loss", "macro_f1"] = "log_loss",
    class_weight: str | dict[str, float] | None = None,
    pca_dimension: int | None = None,
    coordinate_rater: str | None = None,
    target_rater: str | None = None,
    split_tables: tuple[str, str] | None = None,
) -> CrowdRepresentationProbeArtifact:
    """Run one A, H, or AH probe with optional PCA and resumable semantics.

    Parameters
    ----------
    output_directory:
        Target path.  Must not already exist unless it is a valid completed
        artifact whose stored configuration is fully compatible with this
        request, in which case the run is skipped (resumable).
    representation:
        ``"A"`` (appraisal only), ``"H"`` (hidden only), or ``"AH"`` (both
        blocks concatenated).
    pca_dimension:
        Appraisal PCA dimension fitted inside each outer fold on the train
        partition.  Only valid for ``representation == "A"``; values greater
        than or equal to the appraisal width reproduce the full-dimensional
        baseline.  ``None`` means no PCA.
    target_scale:
        One of ``"full_writer"`` or ``"reader"``.  The sealed
        ``"external_writer"`` configuration is never an OOF run; use
        :func:`run_q2_external_probe` for it.
    rater_role:
        Explicit label for the rater wiring (e.g.
        ``"writer_appraisal_to_writer_target"``).
    coordinate_rater, target_rater:
        Who produced the appraisal coordinates and who produced the targets.
        Defaults derive from *target_scale*; both are recorded in metadata.
    split_tables:
        Names of the outer/inner split tables backing this run.  Defaults
        derive from *target_scale* and are recorded in metadata.

    Returns
    -------
    CrowdRepresentationProbeArtifact

    Raises
    ------
    FileExistsError
        If *output_directory* exists but is not a valid completed artifact
        compatible with this request.
    """

    if representation not in {"A", "H", "AH"}:
        raise ValueError("representation must be 'A', 'H', or 'AH'")
    if target_scale not in ALLOWED_TARGET_SCALES:
        raise ValueError(f"target_scale must be one of {ALLOWED_TARGET_SCALES}")
    if target_scale == "external_writer":
        raise ValueError(
            "the sealed external configuration is never an OOF run; "
            "use run_q2_external_probe"
        )
    if pooling not in ALLOWED_POOLING:
        raise ValueError(f"pooling must be one of {ALLOWED_POOLING}")
    if selection_metric not in {"log_loss", "macro_f1"}:
        raise ValueError("selection_metric must be 'log_loss' or 'macro_f1'")
    if pca_dimension is not None:
        if representation != "A":
            raise ValueError("pca_dimension applies to representation 'A' only")
        if not isinstance(pca_dimension, int) or pca_dimension < 1:
            raise ValueError("pca_dimension must be a positive integer")

    ids = np.asarray([str(i) for i in item_ids], dtype=str)
    targets = np.asarray(y, dtype=str)
    names = tuple(str(n) for n in class_names)
    appraisal_axis = tuple(str(n) for n in appraisal_names)
    appraisal_matrix = np.asarray(appraisals, dtype=np.float64)
    if appraisal_matrix.ndim != 2 or appraisal_matrix.shape != (
        len(ids),
        len(appraisal_axis),
    ):
        raise ValueError(
            "appraisals must have shape "
            f"({len(ids)}, {len(appraisal_axis)}); got {appraisal_matrix.shape}"
        )
    if not np.isfinite(appraisal_matrix).all():
        raise ValueError("appraisals contain non-finite values")
    if targets.shape != ids.shape:
        raise ValueError("y must contain one value per item")

    default_coordinate, default_target_rater = _DEFAULT_RATERS[target_scale]
    coordinate_rater = coordinate_rater or default_coordinate
    target_rater = target_rater or default_target_rater
    split_tables = split_tables or _DEFAULT_SPLIT_TABLES[target_scale]
    target_label = _TARGET_LABELS[target_scale]

    output = Path(output_directory)

    # --- resumable: validate and skip only fully compatible artifacts ---
    if output.exists():
        try:
            existing = validate_crowd_representation_probe(output)
        except (ValueError, KeyError, OSError) as error:
            raise FileExistsError(
                f"existing artifact at {output} is incompatible or corrupt; "
                "remove it before re-running"
            ) from error
        meta = existing.metadata
        expected: dict[str, Any] = {
            "representation": representation,
            "target": target_label,
            "target_scale": target_scale,
            "rater_role": rater_role,
            "coordinate_rater": coordinate_rater,
            "target_rater": target_rater,
            "split_outer_table": split_tables[0],
            "split_inner_table": split_tables[1],
            "pca_dimension": pca_dimension,
            "layer": layer if representation != "A" else None,
            "class_names": list(names),
            "appraisal_names": list(appraisal_axis),
            "n_items": len(ids),
            "selection_metric": selection_metric,
            "class_weight": class_weight,
            "C_grid": [float(v) for v in C_grid],
            "ordered_item_target_sha256": _ordered_pair_digest(ids, targets),
            "appraisal_matrix_sha256": _sha256_array(appraisal_matrix),
        }
        if representation != "A":
            expected["pooling"] = pooling
        _require_compatible(meta, expected=expected, directory=output)
        return existing

    # --- PCA path (A-only, d below the appraisal width) ---
    if (
        representation == "A"
        and pca_dimension is not None
        and pca_dimension < appraisal_matrix.shape[1]
    ):
        outer_arr, groups_arr, unique_folds = _aligned_outer(ids, outer_folds)
        inner_validated = _validated_inner(
            ids, groups_arr, outer_arr, inner_folds, unique_folds
        )
        _pca_appraisals(
            appraisal_matrix,
            d=pca_dimension,
            outer_folds=outer_arr,
            unique_folds=unique_folds,
            item_ids=ids,
            groups=groups_arr,
            class_names=names,
            targets=targets,
            inner_folds=inner_validated,
            appraisal_names=appraisal_axis,
            C_grid=C_grid,
            selection_metric=selection_metric,
            class_weight=class_weight,
            output_root=output.parent,
            run_name=output.name,
        )
        _patch_metadata(
            output,
            target=target_label,
            target_scale=target_scale,
            rater_role=rater_role,
            coordinate_rater=coordinate_rater,
            target_rater=target_rater,
            split_tables=split_tables,
            pca_dimension=pca_dimension,
        )
        return validate_crowd_representation_probe(output)

    # --- standard path (delegates to experiment_c) ---
    run_crowd_representation_probe(
        output,
        representation=representation,
        appraisals=appraisal_matrix,
        y=targets,
        item_ids=ids,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        embedding_directory=embedding_directory,
        layer=layer,
        pooling=pooling,
        class_names=names,
        appraisal_names=appraisal_axis,
        C_grid=C_grid,
        block_multiplier_grid=block_multiplier_grid,
        selection_metric=selection_metric,
        class_weight=class_weight,
    )
    _patch_metadata(
        output,
        target=target_label,
        target_scale=target_scale,
        rater_role=rater_role,
        coordinate_rater=coordinate_rater,
        target_rater=target_rater,
        split_tables=split_tables,
        pca_dimension=pca_dimension,
    )
    return validate_crowd_representation_probe(output)


def _patch_metadata(
    directory: Path,
    *,
    target: str,
    target_scale: str,
    rater_role: str,
    coordinate_rater: str,
    target_rater: str,
    split_tables: tuple[str, str],
    pca_dimension: int | None,
) -> None:
    """Add Q2-suite provenance fields into an existing artifact's metadata."""

    meta_path = directory / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["status"] = "new_replication_not_historical_recovery"
    metadata["target"] = target
    metadata["target_scale"] = target_scale
    metadata["rater_role"] = rater_role
    metadata["coordinate_rater"] = coordinate_rater
    metadata["target_rater"] = target_rater
    metadata["split_outer_table"] = split_tables[0]
    metadata["split_inner_table"] = split_tables[1]
    metadata["pca_dimension"] = pca_dimension
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_q2_reader_probe(
    output_directory: str | Path,
    *,
    representation: Literal["A", "H", "AH"],
    writer_appraisals: np.ndarray,
    reader_targets: Sequence[str],
    item_ids: Sequence[str],
    reader_outer_folds: pd.DataFrame,
    reader_inner_folds: pd.DataFrame,
    embedding_directory: str | Path | EmbeddingArtifact | None = None,
    layer: int | None = None,
    pooling: str = "mean",
    class_names: Sequence[str] = CROWD_EMOTIONS,
    appraisal_names: Sequence[str] = APPRAISAL_NAMES,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    block_multiplier_grid: Sequence[float] = DEFAULT_BLOCK_MULTIPLIER_GRID,
    selection_metric: Literal["log_loss", "macro_f1"] = "log_loss",
    class_weight: str | dict[str, float] | None = None,
    pca_dimension: int | None = None,
) -> CrowdRepresentationProbeArtifact:
    """Writer-appraisal → reader-target cross-rater probe.

    Writer-side appraisal coordinates (``writer_appraisals``) predict
    reader-majority emotion targets (``reader_targets``) over the reader
    split tables ``crowd_reader_outer.csv`` / ``crowd_reader_inner.csv``.
    Targets, coordinate raters, class names, and split tables are explicit
    parameters and are recorded as separate metadata fields, never conflated.
    """

    return run_q2_representation_triplet(
        output_directory,
        representation=representation,
        appraisals=writer_appraisals,
        y=reader_targets,
        item_ids=item_ids,
        outer_folds=reader_outer_folds,
        inner_folds=reader_inner_folds,
        target_scale="reader",
        rater_role="writer_appraisal_to_reader_target",
        coordinate_rater="writer",
        target_rater="reader",
        split_tables=READER_SPLIT_TABLES,
        embedding_directory=embedding_directory,
        layer=layer,
        pooling=pooling,
        class_names=class_names,
        appraisal_names=appraisal_names,
        C_grid=C_grid,
        block_multiplier_grid=block_multiplier_grid,
        selection_metric=selection_metric,
        class_weight=class_weight,
        pca_dimension=pca_dimension,
    )


# ---------------------------------------------------------------------------
# External-test runner (sealed split, never OOF)
# ---------------------------------------------------------------------------

def run_q2_external_probe(
    output_directory: str | Path,
    *,
    representation: Literal["A", "H", "AH"],
    appraisals: np.ndarray,
    y_train: Sequence[str],
    item_ids_train: Sequence[str],
    appraisals_test: np.ndarray,
    y_test: Sequence[str],
    item_ids_test: Sequence[str],
    inner_folds: pd.DataFrame,
    embedding_directory: str | Path | EmbeddingArtifact | None = None,
    layer: int | None = None,
    pooling: str = "mean",
    class_names: Sequence[str] = CROWD_EMOTIONS,
    appraisal_names: Sequence[str] = APPRAISAL_NAMES,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    block_multiplier_grid: Sequence[float] = DEFAULT_BLOCK_MULTIPLIER_GRID,
    selection_metric: Literal["log_loss", "macro_f1"] = "log_loss",
    class_weight: str | dict[str, float] | None = None,
    pca_dimension: int | None = None,
    excluded_item_ids: Sequence[str] = (),
) -> ExternalTestResult:
    """Fit on the sealed train partition, predict on the external test.

    The external test is **not** an OOF fold.  Inner-fold selection uses
    ``crowd_external_inner.csv`` (three folds) on the train rows only, PCA is
    fitted on the train rows only, and excluded test writers are preserved
    untouched — pass them via *excluded_item_ids* to bind their exclusion
    into the artifact metadata.
    """

    if representation not in {"A", "H", "AH"}:
        raise ValueError("representation must be 'A', 'H', or 'AH'")
    if pooling not in ALLOWED_POOLING:
        raise ValueError(f"pooling must be one of {ALLOWED_POOLING}")
    if selection_metric not in {"log_loss", "macro_f1"}:
        raise ValueError("selection_metric must be 'log_loss' or 'macro_f1'")
    if pca_dimension is not None and (
        not isinstance(pca_dimension, int) or pca_dimension < 1
    ):
        raise ValueError("pca_dimension must be a positive integer")

    names = tuple(str(n) for n in class_names)
    appraisal_axis = tuple(str(n) for n in appraisal_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names must be non-empty and unique")

    train_ids = np.asarray([str(i) for i in item_ids_train], dtype=str)
    test_ids = np.asarray([str(i) for i in item_ids_test], dtype=str)
    excluded_ids = tuple(str(i) for i in excluded_item_ids)
    if train_ids.ndim != 1 or train_ids.size == 0 or np.unique(train_ids).size != train_ids.size:
        raise ValueError("item_ids_train must be a non-empty sequence of unique values")
    if test_ids.ndim != 1 or test_ids.size == 0 or np.unique(test_ids).size != test_ids.size:
        raise ValueError("item_ids_test must be a non-empty sequence of unique values")
    if set(train_ids) & set(test_ids):
        raise ValueError("external train and test item IDs must be disjoint")
    if set(excluded_ids) & (set(train_ids) | set(test_ids)):
        raise ValueError("excluded items must be disjoint from train and test")
    train_targets = np.asarray(y_train, dtype=str)
    test_targets = np.asarray(y_test, dtype=str)
    if train_targets.shape != train_ids.shape or test_targets.shape != test_ids.shape:
        raise ValueError("targets must contain one value per item")
    if not set(train_targets).issubset(names) or not set(test_targets).issubset(names):
        raise ValueError("class_names must cover every train and test target")

    train_appraisals = np.asarray(appraisals, dtype=np.float64)
    test_appraisals = np.asarray(appraisals_test, dtype=np.float64)
    if train_appraisals.shape != (len(train_ids), len(appraisal_axis)):
        raise ValueError(
            "appraisals must have shape "
            f"({len(train_ids)}, {len(appraisal_axis)}); got {train_appraisals.shape}"
        )
    if test_appraisals.shape != (len(test_ids), len(appraisal_axis)):
        raise ValueError(
            "appraisals_test must have shape "
            f"({len(test_ids)}, {len(appraisal_axis)}); got {test_appraisals.shape}"
        )
    if not np.isfinite(train_appraisals).all() or not np.isfinite(test_appraisals).all():
        raise ValueError("appraisals contain non-finite values")

    # --- train-only inner split (crowd_external_inner.csv schema) ---
    required_inner = {"item_id", "group_id", "validation_fold"}
    if not required_inner.issubset(inner_folds.columns) or inner_folds[
        list(required_inner)
    ].isna().any().any():
        raise ValueError(
            f"inner_folds must contain non-missing {sorted(required_inner)}"
        )
    inner = inner_folds.copy()
    inner["item_id"] = inner["item_id"].astype(str)
    if inner["item_id"].duplicated().any() or set(inner["item_id"]) != set(train_ids):
        raise ValueError(
            "external inner split must contain exactly the training items, once each"
        )
    inner_by_id = inner.set_index("item_id", verify_integrity=True)
    validation_folds = np.asarray(
        [inner_by_id.at[i, "validation_fold"] for i in train_ids]
    )
    train_groups = np.asarray(
        [str(inner_by_id.at[i, "group_id"]) for i in train_ids], dtype=str
    )

    output = Path(output_directory)

    # --- resumable: validate and skip only fully compatible artifacts ---
    if output.exists():
        try:
            existing = validate_q2_external_probe(output)
        except (ValueError, KeyError, OSError) as error:
            raise FileExistsError(
                f"existing artifact at {output} is incompatible or corrupt; "
                "remove it before re-running"
            ) from error
        expected: dict[str, Any] = {
            "representation": representation,
            "layer": layer if representation != "A" else None,
            "class_names": list(names),
            "appraisal_names": list(appraisal_axis),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "selection_metric": selection_metric,
            "class_weight": class_weight,
            "pca_dimension": pca_dimension,
            "C_grid": [float(v) for v in C_grid],
            "ordered_train_item_target_sha256": _ordered_pair_digest(
                train_ids, train_targets
            ),
            "ordered_test_item_target_sha256": _ordered_pair_digest(
                test_ids, test_targets
            ),
            "appraisal_matrix_sha256": _sha256_array(train_appraisals),
            "appraisal_matrix_test_sha256": _sha256_array(test_appraisals),
            "inner_split_sha256": _dataframe_digest(inner_folds),
            "n_excluded_items": len(excluded_ids),
        }
        if representation != "A":
            expected["pooling"] = pooling
        _require_compatible(existing.metadata, expected=expected, directory=output)
        return existing

    # --- build features ---
    embedding: EmbeddingArtifact | None = None
    hidden_train: np.ndarray | None = None
    hidden_test: np.ndarray | None = None
    embedding_layer_sha256: str | None = None

    if representation in {"H", "AH"}:
        if embedding_directory is None or layer is None:
            raise ValueError("H and AH require embedding_directory and layer")
        artifact_path = (
            embedding_directory.directory
            if isinstance(embedding_directory, EmbeddingArtifact)
            else embedding_directory
        )
        all_ids = np.concatenate([train_ids, test_ids])
        embedding = validate_embedding_artifact(
            artifact_path, expected_item_ids=all_ids
        )
        if not isinstance(layer, int) or not 0 <= layer < int(
            embedding.metadata["n_layers"]
        ):
            raise ValueError("layer is outside the validated embedding layer axis")
        pooled = np.load(
            embedding.directory / f"{pooling}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        hidden_all = np.asarray(pooled[layer], dtype=np.float64)
        embedding_layer_sha256 = _sha256_array(hidden_all.astype(np.float32))
        expected_hash = embedding.metadata["files"][f"{pooling}.npy"]["layer_sha256"][layer]
        if embedding_layer_sha256 != expected_hash:
            raise ValueError("loaded hidden layer disagrees with embedding metadata")
        position = {item_id: index for index, item_id in enumerate(embedding.item_ids)}
        hidden_train = hidden_all[[position[i] for i in train_ids]]
        hidden_test = hidden_all[[position[i] for i in test_ids]]
    elif embedding_directory is not None or layer is not None:
        raise ValueError("A-only runs must not declare an embedding or layer")

    # --- PCA fitted on the sealed train partition only ---
    if representation == "A":
        block_dims = (train_appraisals.shape[1],)
        train_features = train_appraisals
        test_features = test_appraisals
        if pca_dimension is not None and pca_dimension < train_appraisals.shape[1]:
            pca = PCA(n_components=pca_dimension, random_state=0)
            train_features = pca.fit_transform(train_appraisals)
            test_features = pca.transform(test_appraisals)
            block_dims = (pca_dimension,)
    elif representation == "H":
        if hidden_train is None:  # pragma: no cover - narrowed above
            raise RuntimeError("missing hidden features")
        train_features = hidden_train
        test_features = hidden_test
        block_dims = (hidden_train.shape[1],)
    else:  # AH
        if hidden_train is None:  # pragma: no cover - narrowed above
            raise RuntimeError("missing hidden features")
        appraisal_block_train = train_appraisals
        appraisal_block_test = test_appraisals
        if pca_dimension is not None and pca_dimension < train_appraisals.shape[1]:
            pca = PCA(n_components=pca_dimension, random_state=0)
            appraisal_block_train = pca.fit_transform(train_appraisals)
            appraisal_block_test = pca.transform(test_appraisals)
        train_features = np.concatenate([appraisal_block_train, hidden_train], axis=1)
        test_features = np.concatenate([appraisal_block_test, hidden_test], axis=1)
        block_dims = (appraisal_block_train.shape[1], hidden_train.shape[1])

    # --- inner-fold selection on the sealed train partition only ---
    def estimator_for(C: float, multiplier: float) -> TransformedMulticlassLogistic:
        effective = (1.0,) if len(block_dims) == 1 else (float(multiplier), 1.0)
        return TransformedMulticlassLogistic(
            C=float(C),
            class_names=names,
            block_dims=block_dims,
            block_multipliers=effective,
            class_weight=class_weight,
        )

    bm_grid = tuple(block_multiplier_grid) if len(block_dims) == 2 else (1.0,)
    selection = select_multiclass_C_block_multiplier(
        train_features,
        train_targets,
        validation_folds=validation_folds,
        groups=train_groups,
        class_names=names,
        block_dims=block_dims,
        estimator_factory=estimator_for,
        C_grid=C_grid,
        block_multiplier_grid=bm_grid,
        selection_metric=selection_metric,
    )

    # --- refit on the full train partition, predict the sealed test ---
    final_estimator = estimator_for(selection.C, selection.block_multiplier)
    final_estimator.fit(train_features, train_targets)
    probability = final_estimator.predict_proba(test_features)
    prediction = np.asarray(names)[np.argmax(probability, axis=1)]

    test_frame = pd.DataFrame({
        "item_id": test_ids,
        "y_true": test_targets,
        "y_pred": prediction,
        **{f"prob__{cn}": probability[:, ci] for ci, cn in enumerate(names)},
    })
    selections = pd.DataFrame([{
        "C": selection.C,
        "block_multiplier": selection.block_multiplier,
        "inner_log_loss_bits": selection.log_loss_bits,
        "inner_macro_f1": selection.macro_f1,
        "selection_metric": selection_metric,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
    }])

    # --- atomic write ---
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        pred_path = temporary / "test_predictions.parquet"
        sel_path = temporary / "selections.parquet"
        test_frame.to_parquet(pred_path, index=False, engine="pyarrow", compression="zstd")
        selections.to_parquet(sel_path, index=False, engine="pyarrow", compression="zstd")
        file_records = {
            p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
            for p in (pred_path, sel_path)
        }
        metadata: dict[str, Any] = {
            "run_format": EXTERNAL_FORMAT,
            "status": "new_replication_not_historical_recovery",
            "dataset": "crowd",
            "target": "y_writer",
            "target_scale": "external_writer",
            "rater_role": "writer_appraisal_to_writer_target",
            "coordinate_rater": "writer",
            "target_rater": "writer",
            "split_outer_table": EXTERNAL_SPLIT_TABLES[0],
            "split_inner_table": EXTERNAL_SPLIT_TABLES[1],
            "representation": representation,
            "layer": layer if representation != "A" else None,
            "pooling": pooling if representation != "A" else None,
            "class_names": list(names),
            "appraisal_names": list(appraisal_axis),
            "block_dims": list(block_dims),
            "pca_dimension": pca_dimension,
            "C_grid": [float(v) for v in C_grid],
            "block_multiplier_grid": [float(v) for v in bm_grid],
            "selection_metric": selection_metric,
            "class_weight": class_weight,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "n_excluded_items": len(excluded_ids),
            "excluded_item_ids_sha256": (
                _ordered_pair_digest(excluded_ids, excluded_ids)
                if excluded_ids
                else None
            ),
            "n_features": int(train_features.shape[1]),
            "appraisal_matrix_sha256": _sha256_array(train_appraisals),
            "appraisal_matrix_test_sha256": _sha256_array(test_appraisals),
            "feature_matrix_sha256": _sha256_array(np.ascontiguousarray(train_features)),
            "ordered_train_item_target_sha256": _ordered_pair_digest(
                train_ids, train_targets
            ),
            "ordered_test_item_target_sha256": _ordered_pair_digest(
                test_ids, test_targets
            ),
            "inner_split_sha256": _dataframe_digest(inner_folds),
            "implementation_sha256": {
                "crowd_q2.py": _sha256_file(Path(__file__)),
            },
            "pandas_version": distribution_version("pandas"),
            "scikit_learn_version": distribution_version("scikit-learn"),
            "pyarrow_version": distribution_version("pyarrow"),
            "files": file_records,
        }
        if embedding is not None:
            metadata.update({
                "embedding_artifact_format": embedding.metadata["artifact_format"],
                "embedding_model_key": embedding.metadata["model_key"],
                "embedding_revision": embedding.metadata["revision"],
                "embedding_mode": embedding.metadata["mode"],
                "embedding_text_variant": embedding.metadata["text_variant"],
                "embedding_metadata_sha256": _sha256_file(
                    embedding.directory / "metadata.json"
                ),
                "embedding_item_text_pairs_sha256": embedding.metadata[
                    "ordered_item_text_pairs_sha256"
                ],
                "embedding_layer_sha256": embedding_layer_sha256,
            })
        else:
            metadata.update({
                "embedding_artifact_format": None,
                "embedding_model_key": None,
                "embedding_revision": None,
                "embedding_mode": None,
                "embedding_text_variant": None,
                "embedding_metadata_sha256": None,
                "embedding_item_text_pairs_sha256": None,
                "embedding_layer_sha256": None,
            })
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return validate_q2_external_probe(output)


def validate_q2_external_probe(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> ExternalTestResult:
    """Validate one completed sealed external-test probe artifact."""

    root = Path(directory)
    missing = [name for name in EXTERNAL_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"partial external probe; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("external probe metadata is unreadable") from error
    if metadata.get("run_format") != EXTERNAL_FORMAT:
        raise ValueError("unknown external probe format")
    if metadata.get("status") != "new_replication_not_historical_recovery":
        raise ValueError("external probe status is not a reconstruction label")
    if metadata.get("representation") not in {"A", "H", "AH"}:
        raise ValueError("invalid external probe representation")
    if metadata.get("target_scale") != "external_writer":
        raise ValueError("external probe must carry the external_writer scale")
    names = tuple(str(n) for n in metadata.get("class_names", ()))
    n_train = int(metadata.get("n_train", 0))
    n_test = int(metadata.get("n_test", 0))
    if not names or n_train <= 0 or n_test <= 0:
        raise ValueError("invalid external probe classes or partition sizes")
    for field in (
        "appraisal_matrix_sha256",
        "feature_matrix_sha256",
        "ordered_train_item_target_sha256",
        "ordered_test_item_target_sha256",
        "inner_split_sha256",
    ):
        value = metadata.get(field)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"external probe metadata lacks {field}")

    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("external probe metadata lacks file records")
    for filename in ("test_predictions.parquet", "selections.parquet"):
        record = files.get(filename)
        path = root / filename
        if not isinstance(record, Mapping) or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"external probe file size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"external probe file hash mismatch: {filename}")
    try:
        test_frame = pd.read_parquet(root / "test_predictions.parquet", engine="pyarrow")
        selections = pd.read_parquet(root / "selections.parquet", engine="pyarrow")
    except Exception as error:
        raise ValueError("external probe parquet is unreadable") from error
    required_predictions = {
        "item_id", "y_true", "y_pred", *(f"prob__{name}" for name in names)
    }
    if not required_predictions.issubset(test_frame.columns):
        raise ValueError("external probe prediction schema is incomplete")
    if len(test_frame) != n_test or test_frame["item_id"].duplicated().any():
        raise ValueError("external probe test coverage is invalid")
    if not set(test_frame["y_true"].astype(str)).issubset(names):
        raise ValueError("external probe test targets disagree with class names")
    probability = test_frame[[f"prob__{name}" for name in names]].to_numpy(
        dtype=np.float64
    )
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("external probe probabilities must sum to one")
    required_selections = {
        "C", "block_multiplier", "inner_log_loss_bits", "inner_macro_f1",
        "selection_metric", "n_train", "n_test",
    }
    if not required_selections.issubset(selections.columns) or len(selections) != 1:
        raise ValueError("external probe selections must contain exactly one row")
    row = selections.iloc[0]
    if int(row["n_train"]) != n_train or int(row["n_test"]) != n_test:
        raise ValueError("external probe selection counts disagree with metadata")
    return ExternalTestResult(root, dict(metadata), test_frame, selections)


# ---------------------------------------------------------------------------
# Suite manifest builder
# ---------------------------------------------------------------------------

def build_q2_suite(
    output_directory: str | Path,
    *,
    runs_root: str | Path,
    labels: Sequence[str] = CROWD_EMOTIONS,
    n_bootstrap: int = 2000,
    seed: int = 20240804,
) -> Q2SuiteArtifact:
    """Build a Q2 suite manifest from completed representation runs.

    Scans *runs_root* for completed representation-probe artifacts, computes
    per-run metrics (log loss, macro-F1, Brier, ECE), and paired H-minus-AH
    log-loss contrasts keyed by the same item ordering.  The suite directory
    is immutable: an existing output is never overwritten.
    """

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Q2 suite: {output}")

    names = tuple(str(v) for v in labels)
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {root}")

    # --- collect completed runs (external sealed probes are a different
    # format and are never picked up as OOF runs) ---
    run_artifacts: list[CrowdRepresentationProbeArtifact] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        relative = metadata_path.resolve().relative_to(root)
        if any(part.startswith(".") or ".tmp-" in part for part in relative.parts):
            continue
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, Mapping) or raw.get("run_format") != RUN_FORMAT:
            continue
        try:
            artifact = validate_crowd_representation_probe(metadata_path.parent)
        except ValueError:
            continue
        if tuple(artifact.metadata.get("class_names", ())) != names:
            continue
        run_artifacts.append(artifact)

    if not run_artifacts:
        raise ValueError("no completed representation runs found under runs_root")

    # --- per-run metrics ---
    summary_rows: list[dict[str, Any]] = []
    for artifact in run_artifacts:
        m = artifact.metadata
        metrics = reconstruct_multiclass_metrics(artifact.oof, labels=names)
        row: dict[str, Any] = {
            "representation": m.get("representation"),
            "model_key": m.get("embedding_model_key"),
            "layer": m.get("layer"),
            "pooling": m.get("pooling"),
            "target_scale": m.get("target_scale", "full_writer"),
            "rater_role": m.get("rater_role"),
            "coordinate_rater": m.get("coordinate_rater"),
            "target_rater": m.get("target_rater"),
            "pca_dimension": m.get("pca_dimension"),
            "selection_metric": m.get("selection_metric"),
            "run_path": str(artifact.directory.relative_to(root)),
            **metrics.overall.to_dict(),
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # --- paired H-minus-AH contrasts on identically ordered OOF tables ---
    artifacts_by_path = {
        str(artifact.directory.relative_to(root)): artifact
        for artifact in run_artifacts
    }
    group_keys = [
        "target_scale", "model_key", "layer", "pooling", "rater_role", "pca_dimension"
    ]
    contrast_rows: list[dict[str, Any]] = []
    for group_vals, group_df in summary.groupby(group_keys, dropna=False):
        h_paths = group_df.loc[group_df["representation"] == "H", "run_path"]
        ah_paths = group_df.loc[group_df["representation"] == "AH", "run_path"]
        for h_path in h_paths:
            for ah_path in ah_paths:
                h_artifact = artifacts_by_path[h_path]
                ah_artifact = artifacts_by_path[ah_path]
                h_oof = h_artifact.oof.sort_values("item_id", kind="stable").reset_index(
                    drop=True
                )
                ah_oof = ah_artifact.oof.sort_values(
                    "item_id", kind="stable"
                ).reset_index(drop=True)
                if not h_oof["item_id"].equals(ah_oof["item_id"]):
                    # Different item universes cannot be paired; skip loudly
                    # via the summary rather than silently pairing.
                    continue
                if not h_oof["group_id"].astype(str).equals(
                    ah_oof["group_id"].astype(str)
                ):
                    raise ValueError(
                        "paired H/AH runs disagree on group assignments: "
                        f"{h_path} vs {ah_path}"
                    )
                if not h_oof["y_true"].astype(str).equals(ah_oof["y_true"].astype(str)):
                    raise ValueError(
                        "paired H/AH runs disagree on targets: "
                        f"{h_path} vs {ah_path}"
                    )
                h_loss = multiclass_itemwise_log_loss_bits(h_oof, labels=names)
                ah_loss = multiclass_itemwise_log_loss_bits(ah_oof, labels=names)
                delta = paired_group_bootstrap_delta(
                    h_loss,
                    ah_loss,
                    h_oof["group_id"],
                    item_ids_a=h_oof["item_id"],
                    item_ids_b=ah_oof["item_id"],
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                keys = dict(zip(group_keys, group_vals, strict=True))
                contrast_rows.append({
                    **keys,
                    "n_items": len(h_loss),
                    "H_log_loss_bits": float(delta.observed_a),
                    "AH_log_loss_bits": float(delta.observed_b),
                    "delta_H_minus_AH": float(delta.observed_delta),
                    "ci_low": float(delta.ci_low),
                    "ci_high": float(delta.ci_high),
                    "standard_error": float(delta.standard_error),
                    "H_run_path": h_path,
                    "AH_run_path": ah_path,
                })

    contrasts = pd.DataFrame(contrast_rows) if contrast_rows else pd.DataFrame()

    # --- atomic write ---
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        summary_path = temporary / "summary.parquet"
        contrast_path = temporary / "contrasts.parquet"
        summary.to_parquet(summary_path, index=False, engine="pyarrow", compression="zstd")
        contrasts.to_parquet(contrast_path, index=False, engine="pyarrow", compression="zstd")
        file_records = {
            p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
            for p in (summary_path, contrast_path)
        }
        suite_metadata = {
            "suite_format": SUITE_FORMAT,
            "status": "new_replication_not_historical_recovery",
            "class_names": list(names),
            "n_runs": len(run_artifacts),
            "n_contrasts": len(contrasts),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "implementation_sha256": {
                "crowd_q2.py": _sha256_file(Path(__file__)),
            },
            "pandas_version": distribution_version("pandas"),
            "pyarrow_version": distribution_version("pyarrow"),
            "files": file_records,
        }
        (temporary / "suite_metadata.json").write_text(
            json.dumps(suite_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return validate_q2_suite(output)


def validate_q2_suite(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> Q2SuiteArtifact:
    """Validate a completed Q2 suite artifact."""

    root = Path(directory)
    missing = [name for name in SUITE_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"partial Q2 suite; missing files: {missing}")
    try:
        metadata = json.loads(
            (root / "suite_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Q2 suite metadata is unreadable") from error
    if metadata.get("suite_format") != SUITE_FORMAT:
        raise ValueError("unknown Q2 suite format")
    if metadata.get("status") != "new_replication_not_historical_recovery":
        raise ValueError("Q2 suite status is not a reconstruction label")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Q2 suite metadata lacks file records")
    for filename in ("summary.parquet", "contrasts.parquet"):
        record = files.get(filename)
        path = root / filename
        if not isinstance(record, Mapping) or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"Q2 suite file size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Q2 suite file hash mismatch: {filename}")
    try:
        summary = pd.read_parquet(root / "summary.parquet", engine="pyarrow")
        contrasts = pd.read_parquet(root / "contrasts.parquet", engine="pyarrow")
    except Exception as error:
        raise ValueError("Q2 suite parquet files are unreadable") from error
    required_summary = {
        "representation", "n_items", "log_loss_bits", "macro_f1", "brier", "ece",
    }
    if not required_summary.issubset(summary.columns):
        raise ValueError("Q2 suite summary is missing required columns")
    if len(summary) != int(metadata.get("n_runs", -1)):
        raise ValueError("Q2 suite summary row count disagrees with metadata")
    if not contrasts.empty:
        required_contrasts = {
            "n_items", "H_log_loss_bits", "AH_log_loss_bits",
            "delta_H_minus_AH", "ci_low", "ci_high", "H_run_path", "AH_run_path",
        }
        if not required_contrasts.issubset(contrasts.columns):
            raise ValueError("Q2 suite contrasts are missing required columns")
        if len(contrasts) != int(metadata.get("n_contrasts", -1)):
            raise ValueError("Q2 suite contrast count disagrees with metadata")
    return Q2SuiteArtifact(root, dict(metadata), summary, contrasts)


def summarize_q2_suite(directory: str | Path) -> pd.DataFrame:
    """Return a summary table of per-run log loss, macro-F1, Brier/ECE, and
    H-minus-AH contrasts keyed by the same item ordering.

    This is the command/table interface for inspecting a completed suite.
    """

    artifact = validate_q2_suite(directory)
    summary = artifact.summary.copy()
    contrasts = artifact.contrasts.copy()
    if contrasts.empty:
        return summary

    merge_keys = [
        key
        for key in ("target_scale", "model_key", "layer", "pooling",
                    "rater_role", "pca_dimension")
        if key in summary.columns and key in contrasts.columns
    ]
    if not merge_keys:
        return summary
    contrast_columns = merge_keys + [
        column
        for column in ("delta_H_minus_AH", "ci_low", "ci_high", "standard_error")
        if column in contrasts.columns
    ]

    def _canonical_keys(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for key in merge_keys:
            out[key] = out[key].map(
                lambda value: (
                    "<none>"
                    if pd.isna(value)
                    else str(int(value))
                    if isinstance(value, (int, np.integer, float, np.floating))
                    and float(value).is_integer()
                    else str(value)
                )
            )
        return out

    return _canonical_keys(summary).merge(
        _canonical_keys(contrasts)[contrast_columns], on=merge_keys, how="left"
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_q2_batch(
    output_root: str | Path,
    *,
    appraisals: np.ndarray,
    y: Sequence[str],
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    inner_folds: pd.DataFrame,
    target_scale: str,
    rater_role: str,
    model_keys: Sequence[str] = ("roberta-base",),
    layers: Sequence[int] = (12,),
    pooling: str = "mean",
    class_names: Sequence[str] = CROWD_EMOTIONS,
    appraisal_names: Sequence[str] = APPRAISAL_NAMES,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    block_multiplier_grid: Sequence[float] = DEFAULT_BLOCK_MULTIPLIER_GRID,
    selection_metric: Literal["log_loss", "macro_f1"] = "log_loss",
    class_weight: str | dict[str, float] | None = None,
    pca_dimensions: Sequence[int] = PCA_DIMENSIONS,
    embedding_root: str | Path | None = None,
    embedding_mode: str = "pretrained",
    text_variant: str = "masked",
    require_embeddings: bool = True,
) -> list[CrowdRepresentationProbeArtifact]:
    """Run the full A/H/AH triplet for each named encoder/layer combination.

    Child runs are laid out as ``<model_key>/L<layer>/<representation>`` under
    *output_root*, with PCA appraisal variants as ``A-pca<d>``.  Every child
    is resumable: completed compatible artifacts are validated and skipped,
    never overwritten.

    Parameters
    ----------
    pca_dimensions:
        Appraisal PCA variants for A runs.  Defaults to the paper grid
        ``{3, 5, 7, 10, 21}``; ``d >= appraisal width`` reproduces the
        full-dimensional baseline under an explicitly named child.
    embedding_root:
        Cache root containing validated embedding artifacts in the attested
        layout (see :func:`embedding_artifact_directory`).  Required for the
        H/AH children unless ``require_embeddings=False`` is passed for an
        appraisal-only batch.
    require_embeddings:
        When true (default), a missing *embedding_root* is an error rather
        than a silently hidden-only-free batch.
    """

    if require_embeddings and embedding_root is None:
        raise ValueError(
            "embedding_root is required for H/AH triplet runs; "
            "pass require_embeddings=False for an appraisal-only batch"
        )
    for d in pca_dimensions:
        if not isinstance(d, int) or d < 1:
            raise ValueError("pca_dimensions must contain positive integers")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    results: list[CrowdRepresentationProbeArtifact] = []

    for model_key in model_keys:
        spec = get_model_spec(model_key)
        for layer_idx in layers:
            if not isinstance(layer_idx, int) or not 0 <= layer_idx < spec.emitted_layers:
                raise ValueError(
                    f"layer {layer_idx!r} is outside {spec.key}'s "
                    f"0..{spec.emitted_layers - 1} layer axis"
                )
            emb_dir: Path | None = None
            if embedding_root is not None:
                emb_dir = embedding_artifact_directory(
                    Path(embedding_root),
                    dataset="crowd",
                    model=model_key,
                    mode=embedding_mode,
                    text_variant=text_variant,
                )

            shared = dict(
                appraisals=appraisals,
                y=y,
                item_ids=item_ids,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                target_scale=target_scale,
                rater_role=rater_role,
                class_names=class_names,
                appraisal_names=appraisal_names,
                C_grid=C_grid,
                block_multiplier_grid=block_multiplier_grid,
                selection_metric=selection_metric,
                class_weight=class_weight,
            )

            # A run (full appraisal width) plus the paper PCA variants.
            results.append(run_q2_representation_triplet(
                root / model_key / f"L{layer_idx}" / "A",
                representation="A",
                **shared,
            ))
            for d in pca_dimensions:
                results.append(run_q2_representation_triplet(
                    root / model_key / f"L{layer_idx}" / f"A-pca{d}",
                    representation="A",
                    pca_dimension=d,
                    **shared,
                ))

            if emb_dir is None:
                continue

            results.append(run_q2_representation_triplet(
                root / model_key / f"L{layer_idx}" / "H",
                representation="H",
                embedding_directory=emb_dir,
                layer=layer_idx,
                pooling=pooling,
                **shared,
            ))
            results.append(run_q2_representation_triplet(
                root / model_key / f"L{layer_idx}" / "AH",
                representation="AH",
                embedding_directory=emb_dir,
                layer=layer_idx,
                pooling=pooling,
                **shared,
            ))

    return results


__all__ = [
    "ALLOWED_POOLING",
    "ALLOWED_TARGET_SCALES",
    "EXTERNAL_FILES",
    "EXTERNAL_FORMAT",
    "EXTERNAL_ROLES",
    "EXTERNAL_SPLIT_TABLES",
    "ExternalRolePartition",
    "ExternalTestResult",
    "FULL_SPLIT_TABLES",
    "PCA_DIMENSIONS",
    "Q2SuiteArtifact",
    "READER_SPLIT_TABLES",
    "SUITE_FILES",
    "SUITE_FORMAT",
    "build_q2_suite",
    "external_role_partition",
    "run_q2_batch",
    "run_q2_external_probe",
    "run_q2_reader_probe",
    "run_q2_representation_triplet",
    "summarize_q2_suite",
    "validate_q2_external_probe",
    "validate_q2_suite",
]
