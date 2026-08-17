"""Conditional appraisal/hidden-state probe with serialized fold geometry.

This is a clean-room replication runner, not recovered historical source.  It
implements the smallest crowd-enVENT comparison needed for the dissertation
question: appraisal features (``A``), one frozen hidden layer (``H``), and the
two blocks together (``AH``).  Every scaler, hyperparameter, block multiplier,
and affine decoder is fitted on the corresponding outer-training partition.
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

from .crowd_data import APPRAISAL_NAMES, CROWD_EMOTIONS
from .embeddings import EmbeddingArtifact, validate_embedding_artifact
from .experiment_a import (
    _dataframe_digest,
    _ordered_pair_digest,
    _sha256_array,
    _sha256_file,
)
from .metrics import reconstruct_multiclass_metrics
from .probes import (
    DEFAULT_BLOCK_MULTIPLIER_GRID,
    DEFAULT_C_GRID,
    TransformedMulticlassLogistic,
    select_multiclass_C_block_multiplier,
)


Representation = Literal["A", "H", "AH"]
RUN_FORMAT = "frozen-emotion-spaces-crowd-representation-probe-reconstruction-v1"
RUN_FILES = ("oof.parquet", "selections.parquet", "geometry.npz", "metadata.json")


@dataclass(frozen=True)
class CrowdRepresentationProbeArtifact:
    directory: Path
    metadata: dict[str, Any]
    oof: pd.DataFrame
    selections: pd.DataFrame


def run_crowd_representation_probe(
    output_directory: str | Path,
    *,
    representation: Representation,
    appraisals: np.ndarray,
    y: Sequence[str],
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
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
) -> CrowdRepresentationProbeArtifact:
    """Fit a complete nested OOF A/H/AH probe and publish it atomically."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite representation run: {output}")
    if representation not in {"A", "H", "AH"}:
        raise ValueError("representation must be 'A', 'H', or 'AH'")
    if selection_metric not in {"log_loss", "macro_f1"}:
        raise ValueError("selection_metric must be 'log_loss' or 'macro_f1'")
    if pooling not in {"mean", "first"}:
        raise ValueError("pooling must be 'mean' or 'first'")

    ids = np.asarray([str(item_id) for item_id in item_ids], dtype=str)
    if ids.ndim != 1 or ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError("item_ids must be a non-empty sequence of unique values")
    targets = np.asarray(y, dtype=str)
    names = tuple(str(name) for name in class_names)
    appraisal_axis = tuple(str(name) for name in appraisal_names)
    if targets.shape != ids.shape:
        raise ValueError("y must contain one value per item")
    if not names or len(set(names)) != len(names) or not set(targets).issubset(names):
        raise ValueError("class_names must be unique and cover y")
    if len(appraisal_axis) != len(set(appraisal_axis)) or not appraisal_axis:
        raise ValueError("appraisal_names must be non-empty and unique")

    appraisal_matrix = np.asarray(appraisals, dtype=np.float64)
    if appraisal_matrix.shape != (len(ids), len(appraisal_axis)):
        raise ValueError(
            "appraisals must have shape "
            f"({len(ids)}, {len(appraisal_axis)}); got {appraisal_matrix.shape}"
        )
    if not np.isfinite(appraisal_matrix).all():
        raise ValueError("appraisals contain non-finite values")

    embedding: EmbeddingArtifact | None = None
    hidden: np.ndarray | None = None
    embedding_layer_sha256: str | None = None
    if representation in {"H", "AH"}:
        if embedding_directory is None or layer is None:
            raise ValueError("H and AH require embedding_directory and layer")
        artifact_path = (
            embedding_directory.directory
            if isinstance(embedding_directory, EmbeddingArtifact)
            else embedding_directory
        )
        embedding = validate_embedding_artifact(artifact_path)
        positions = {item_id: index for index, item_id in enumerate(embedding.item_ids)}
        missing = [item_id for item_id in ids if item_id not in positions]
        if missing:
            raise ValueError(
                f"embedding artifact is missing {len(missing)} requested items"
            )
        row_index = np.array([positions[item_id] for item_id in ids])
        if not isinstance(layer, int) or not 0 <= layer < int(
            embedding.metadata["n_layers"]
        ):
            raise ValueError("layer is outside the validated embedding layer axis")
        pooled = np.load(
            embedding.directory / f"{pooling}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        hidden_full = np.asarray(pooled[layer], dtype=np.float64)
        embedding_layer_sha256 = _sha256_array(hidden_full.astype(np.float32))
        expected = embedding.metadata["files"][f"{pooling}.npy"]["layer_sha256"][
            layer
        ]
        if embedding_layer_sha256 != expected:
            raise ValueError("loaded hidden layer disagrees with embedding metadata")
        hidden = np.ascontiguousarray(hidden_full[row_index])
    elif embedding_directory is not None or layer is not None:
        raise ValueError("A-only runs must not declare an embedding or layer")

    if representation == "A":
        features = appraisal_matrix
        block_dims = (appraisal_matrix.shape[1],)
    elif representation == "H":
        if hidden is None:  # pragma: no cover - narrowed above
            raise RuntimeError("missing hidden features")
        features = hidden
        block_dims = (hidden.shape[1],)
    else:
        if hidden is None:  # pragma: no cover - narrowed above
            raise RuntimeError("missing hidden features")
        features = np.concatenate((appraisal_matrix, hidden), axis=1)
        block_dims = (appraisal_matrix.shape[1], hidden.shape[1])

    outer, groups, unique_outer = _aligned_outer(ids, outer_folds)
    inner = _validated_inner(ids, groups, outer, inner_folds, unique_outer)
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    geometry_records: list[dict[str, np.ndarray]] = []

    def estimator_for(C: float, multiplier: float) -> TransformedMulticlassLogistic:
        effective = (1.0,) if len(block_dims) == 1 else (float(multiplier), 1.0)
        return TransformedMulticlassLogistic(
            C=float(C),
            class_names=names,
            block_dims=block_dims,
            block_multipliers=effective,
            class_weight=class_weight,
        )

    for outer_fold in unique_outer:
        test_mask = outer == outer_fold
        train_mask = ~test_mask
        overlap = set(groups[train_mask]) & set(groups[test_mask])
        if overlap:
            raise ValueError(
                f"group leakage in outer fold {outer_fold!r}: {sorted(overlap)[:3]}"
            )
        train_ids = ids[train_mask]
        rows = inner.loc[inner["outer_fold"].astype(str) == str(outer_fold)]
        fold_by_id = dict(
            zip(rows["item_id"].astype(str), rows["validation_fold"], strict=True)
        )
        validation_folds = np.asarray([fold_by_id[item_id] for item_id in train_ids])

        selection = select_multiclass_C_block_multiplier(
            features[train_mask],
            targets[train_mask],
            validation_folds=validation_folds,
            groups=groups[train_mask],
            class_names=names,
            block_dims=block_dims,
            estimator_factory=estimator_for,
            C_grid=C_grid,
            block_multiplier_grid=block_multiplier_grid,
            selection_metric=selection_metric,
        )
        estimator = estimator_for(selection.C, selection.block_multiplier)
        estimator.fit(features[train_mask], targets[train_mask])
        probability = estimator.predict_proba(features[test_mask])
        prediction = np.asarray(names)[np.argmax(probability, axis=1)]
        frame: dict[str, Any] = {
            "item_id": ids[test_mask],
            "outer_fold": outer_fold,
            "group_id": groups[test_mask],
            "y_true": targets[test_mask],
            "y_pred": prediction,
        }
        for class_index, class_name in enumerate(names):
            frame[f"prob__{class_name}"] = probability[:, class_index]
        prediction_frames.append(pd.DataFrame(frame))
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "C": selection.C,
                "block_multiplier": selection.block_multiplier,
                "inner_log_loss_bits": selection.log_loss_bits,
                "inner_macro_f1": selection.macro_f1,
                "selection_metric": selection_metric,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )
        geometry_records.append(
            _fold_geometry(
                estimator,
                features[train_mask],
                targets[train_mask],
                names=names,
                outer_fold=outer_fold,
            )
        )

    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(
        "item_id", kind="stable"
    ).reset_index(drop=True)
    selections = pd.DataFrame(selection_rows).sort_values(
        "outer_fold", kind="stable"
    ).reset_index(drop=True)
    if len(oof) != len(ids) or oof["item_id"].nunique() != len(ids):
        raise RuntimeError("representation OOF does not cover every item once")
    reconstruct_multiclass_metrics(oof, labels=names)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        oof_path = temporary / "oof.parquet"
        selections_path = temporary / "selections.parquet"
        geometry_path = temporary / "geometry.npz"
        oof.to_parquet(oof_path, index=False, engine="pyarrow", compression="zstd")
        selections.to_parquet(
            selections_path, index=False, engine="pyarrow", compression="zstd"
        )
        _write_geometry(geometry_path, geometry_records)
        file_records = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (oof_path, selections_path, geometry_path)
        }
        embedding_metadata_sha256 = None
        embedding_identity: dict[str, Any] = {
            "embedding_artifact_format": None,
            "embedding_model_key": None,
            "embedding_revision": None,
            "embedding_mode": None,
            "embedding_text_variant": None,
            "embedding_metadata_sha256": None,
            "embedding_item_text_pairs_sha256": None,
            "embedding_layer_sha256": None,
        }
        if embedding is not None:
            embedding_metadata_sha256 = _sha256_file(embedding.directory / "metadata.json")
            embedding_identity = {
                "embedding_artifact_format": embedding.metadata["artifact_format"],
                "embedding_model_key": embedding.metadata["model_key"],
                "embedding_revision": embedding.metadata["revision"],
                "embedding_mode": embedding.metadata["mode"],
                "embedding_text_variant": embedding.metadata["text_variant"],
                "embedding_metadata_sha256": embedding_metadata_sha256,
                "embedding_item_text_pairs_sha256": embedding.metadata[
                    "ordered_item_text_pairs_sha256"
                ],
                "embedding_layer_sha256": embedding_layer_sha256,
            }
        metadata = {
            "run_format": RUN_FORMAT,
            "dataset": "crowd",
            "target": "y_writer",
            "representation": representation,
            "layer": layer,
            "pooling": pooling if representation != "A" else None,
            "class_names": list(names),
            "appraisal_names": list(appraisal_axis),
            "block_dims": list(block_dims),
            "C_grid": [float(value) for value in C_grid],
            "block_multiplier_grid": (
                [1.0]
                if len(block_dims) == 1
                else [float(value) for value in block_multiplier_grid]
            ),
            "selection_metric": selection_metric,
            "class_weight": class_weight,
            "n_items": len(ids),
            "n_features": int(features.shape[1]),
            "appraisal_matrix_sha256": _sha256_array(appraisal_matrix),
            "feature_matrix_sha256": _sha256_array(features),
            "ordered_item_target_sha256": _ordered_pair_digest(ids, targets),
            "outer_split_sha256": _dataframe_digest(outer_folds),
            "inner_split_sha256": _dataframe_digest(inner_folds),
            "implementation_sha256": {
                filename: _sha256_file(Path(__file__).with_name(filename))
                for filename in ("experiment_c.py", "probes.py", "metrics.py")
            },
            "pandas_version": distribution_version("pandas"),
            "scikit_learn_version": distribution_version("scikit-learn"),
            "pyarrow_version": distribution_version("pyarrow"),
            "files": file_records,
            **embedding_identity,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_crowd_representation_probe(output)


def validate_crowd_representation_probe(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> CrowdRepresentationProbeArtifact:
    """Validate one completed representation-probe artifact."""

    root = Path(directory)
    missing = [name for name in RUN_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"partial representation run; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("representation metadata is unreadable") from error
    if metadata.get("run_format") != RUN_FORMAT:
        raise ValueError("unknown representation run format")
    representation = metadata.get("representation")
    if representation not in {"A", "H", "AH"}:
        raise ValueError("invalid representation identity")
    names = tuple(str(name) for name in metadata.get("class_names", ()))
    block_dims = tuple(int(value) for value in metadata.get("block_dims", ()))
    n_items = int(metadata.get("n_items", 0))
    n_features = int(metadata.get("n_features", 0))
    if (
        not names
        or len(set(names)) != len(names)
        or len(block_dims) not in {1, 2}
        or any(value <= 0 for value in block_dims)
        or sum(block_dims) != n_features
        or n_items <= 0
    ):
        raise ValueError("invalid representation dimensions/classes/count")
    expected_blocks = 2 if representation == "AH" else 1
    if len(block_dims) != expected_blocks:
        raise ValueError("representation and block dimensions disagree")
    if representation == "A":
        if any(metadata.get(field) is not None for field in (
            "layer", "pooling", "embedding_model_key", "embedding_revision",
            "embedding_metadata_sha256", "embedding_layer_sha256",
        )):
            raise ValueError("A-only metadata must not claim a hidden representation")
    else:
        if not isinstance(metadata.get("layer"), int) or metadata.get("pooling") not in {
            "mean", "first"
        }:
            raise ValueError("hidden representation metadata lacks layer/pooling")
        for field in (
            "embedding_metadata_sha256",
            "embedding_item_text_pairs_sha256",
            "embedding_layer_sha256",
        ):
            if not _is_sha256(metadata.get(field)):
                raise ValueError(f"hidden representation metadata lacks {field}")
    for field in (
        "appraisal_matrix_sha256",
        "feature_matrix_sha256",
        "ordered_item_target_sha256",
        "outer_split_sha256",
        "inner_split_sha256",
    ):
        if not _is_sha256(metadata.get(field)):
            raise ValueError(f"representation metadata lacks {field}")

    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("representation metadata lacks file records")
    for filename in ("oof.parquet", "selections.parquet", "geometry.npz"):
        record = files.get(filename)
        path = root / filename
        if not isinstance(record, Mapping) or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"representation file size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"representation file hash mismatch: {filename}")
    try:
        oof = pd.read_parquet(root / "oof.parquet", engine="pyarrow")
        selections = pd.read_parquet(root / "selections.parquet", engine="pyarrow")
    except Exception as error:
        raise ValueError("representation parquet is unreadable") from error
    required_oof = {
        "item_id", "outer_fold", "group_id", "y_true", "y_pred",
        *(f"prob__{name}" for name in names),
    }
    required_selections = {
        "outer_fold", "C", "block_multiplier", "inner_log_loss_bits",
        "inner_macro_f1", "selection_metric", "n_train", "n_test",
    }
    if not required_oof.issubset(oof) or not required_selections.issubset(selections):
        raise ValueError("representation parquet schema is incomplete")
    if len(oof) != n_items or oof["item_id"].duplicated().any():
        raise ValueError("representation OOF coverage is invalid")
    if selections.empty or selections["outer_fold"].astype(str).duplicated().any():
        raise ValueError("representation selections must have one row per fold")
    if set(oof["outer_fold"].astype(str)) != set(selections["outer_fold"].astype(str)):
        raise ValueError("representation OOF and selection folds disagree")
    if oof.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("representation OOF separates an outer group")
    numeric = selections[
        ["C", "block_multiplier", "inner_log_loss_bits", "inner_macro_f1", "n_train", "n_test"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (
        numeric[["C", "block_multiplier", "n_train", "n_test"]] <= 0
    ).any().any():
        raise ValueError("representation selections contain invalid numeric values")
    if ((numeric["inner_macro_f1"] < 0) | (numeric["inner_macro_f1"] > 1)).any():
        raise ValueError("representation selections contain invalid macro-F1")
    try:
        C_grid = np.asarray(metadata["C_grid"], dtype=np.float64)
        multiplier_grid = np.asarray(
            metadata["block_multiplier_grid"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("representation hyperparameter grids are invalid") from error
    if (
        C_grid.ndim != 1
        or multiplier_grid.ndim != 1
        or C_grid.size == 0
        or multiplier_grid.size == 0
        or not np.isfinite(C_grid).all()
        or not np.isfinite(multiplier_grid).all()
        or (C_grid <= 0).any()
        or (multiplier_grid <= 0).any()
        or np.unique(C_grid).size != C_grid.size
        or np.unique(multiplier_grid).size != multiplier_grid.size
    ):
        raise ValueError("representation hyperparameter grids are invalid")
    if not np.isin(numeric["C"].to_numpy(), C_grid).all():
        raise ValueError("selected C is outside the declared grid")
    if not np.isin(
        numeric["block_multiplier"].to_numpy(), multiplier_grid
    ).all():
        raise ValueError("selected block multiplier is outside the declared grid")
    if representation != "AH" and not np.array_equal(multiplier_grid, [1.0]):
        raise ValueError("one-block runs must declare only multiplier 1")
    if not (selections["selection_metric"] == metadata.get("selection_metric")).all():
        raise ValueError("representation selection objective disagrees with metadata")
    if not np.array_equal(
        numeric["n_test"].astype(int).to_numpy(),
        selections["outer_fold"].map(oof.groupby("outer_fold").size()).to_numpy(),
    ):
        raise ValueError("representation selection test counts disagree with OOF")
    if not (
        numeric["n_train"].astype(int) + numeric["n_test"].astype(int) == n_items
    ).all():
        raise ValueError("representation selection train/test counts disagree")
    reconstruct_multiclass_metrics(oof, labels=names)
    _validate_geometry(
        root / "geometry.npz",
        n_folds=len(selections),
        n_classes=len(names),
        n_features=n_features,
        n_blocks=len(block_dims),
        expected_folds=set(selections["outer_fold"].astype(str)),
    )
    return CrowdRepresentationProbeArtifact(root, dict(metadata), oof, selections)


def _aligned_outer(
    item_ids: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    required = {"item_id", "group_id", "test_fold"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError(f"outer_folds must contain non-missing {sorted(required)}")
    work = frame.copy()
    work["item_id"] = work["item_id"].astype(str)
    if work["item_id"].duplicated().any() or set(work["item_id"]) != set(item_ids):
        raise ValueError("outer_folds item IDs do not match item_ids exactly")
    by_id = work.set_index("item_id", verify_integrity=True)
    outer = np.asarray([by_id.at[item, "test_fold"] for item in item_ids])
    groups = np.asarray([str(by_id.at[item, "group_id"]) for item in item_ids])
    try:
        unique = sorted(pd.unique(outer).tolist())
    except TypeError as error:
        raise ValueError("outer fold values must be comparable") from error
    if len(unique) < 2:
        raise ValueError("at least two outer folds are required")
    return outer, groups, unique


def _validated_inner(
    item_ids: np.ndarray,
    groups: np.ndarray,
    outer: np.ndarray,
    frame: pd.DataFrame,
    unique_outer: Sequence[Any],
) -> pd.DataFrame:
    required = {"outer_fold", "item_id", "group_id", "validation_fold"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError(f"inner_folds must contain non-missing {sorted(required)}")
    inner = frame.copy()
    inner["item_id"] = inner["item_id"].astype(str)
    canonical_group = dict(zip(item_ids, groups, strict=True))
    unexpected = sorted(set(inner["item_id"]) - set(item_ids))
    if unexpected:
        raise ValueError(f"inner_folds contains unexpected items: {unexpected[:3]}")
    mismatch = inner["group_id"].astype(str) != inner["item_id"].map(canonical_group)
    if mismatch.any():
        raise ValueError("inner and outer group IDs disagree")
    for fold in unique_outer:
        rows = inner.loc[inner["outer_fold"].astype(str) == str(fold)]
        expected = set(item_ids[outer != fold])
        if rows["item_id"].duplicated().any() or set(rows["item_id"]) != expected:
            raise ValueError(f"inner assignments do not equal outer-train for fold {fold!r}")
    if set(inner["outer_fold"].astype(str)) != {str(value) for value in unique_outer}:
        raise ValueError("inner_folds outer-fold axis is incomplete")
    return inner


def _fold_geometry(
    estimator: TransformedMulticlassLogistic,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    names: tuple[str, ...],
    outer_fold: Any,
) -> dict[str, np.ndarray]:
    transformed = estimator.transformer_.transform(X_train)
    class_index = {str(label): index for index, label in enumerate(estimator.estimator_.classes_)}
    coef = np.asarray(
        estimator.estimator_.coef_[[class_index[name] for name in names]],
        dtype=np.float64,
    )
    intercept = np.asarray(
        estimator.estimator_.intercept_[[class_index[name] for name in names]],
        dtype=np.float64,
    )
    centered_coef = coef - coef.mean(axis=0, keepdims=True)
    centered_intercept = intercept - intercept.mean()
    sites = centered_coef / 2.0
    power_weights = centered_intercept + np.sum(centered_coef**2, axis=1) / 4.0
    centroids = np.vstack(
        [transformed[np.asarray(y_train) == name].mean(axis=0) for name in names]
    )
    scaler_mean = np.concatenate(
        [np.asarray(scaler.mean_, dtype=np.float64) for scaler in estimator.transformer_.scalers_]
    )
    scaler_scale = np.concatenate(
        [np.asarray(scaler.scale_, dtype=np.float64) for scaler in estimator.transformer_.scalers_]
    )
    return {
        "outer_fold": np.asarray(str(outer_fold)),
        "coef": centered_coef,
        "intercept": centered_intercept,
        "sites": sites,
        "power_weights": power_weights,
        "class_centroids": centroids,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "block_multipliers": np.asarray(
            estimator.transformer_.block_multipliers_, dtype=np.float64
        ),
    }


def _write_geometry(path: Path, records: Sequence[dict[str, np.ndarray]]) -> None:
    if not records:
        raise ValueError("at least one geometry record is required")
    fields = tuple(records[0])
    if any(tuple(record) != fields for record in records):
        raise ValueError("geometry records have inconsistent fields")
    arrays = {field: np.stack([record[field] for record in records]) for field in fields}
    np.savez_compressed(path, **arrays)


def _validate_geometry(
    path: Path,
    *,
    n_folds: int,
    n_classes: int,
    n_features: int,
    n_blocks: int,
    expected_folds: set[str],
) -> None:
    required = {
        "outer_fold", "coef", "intercept", "sites", "power_weights",
        "class_centroids", "scaler_mean", "scaler_scale", "block_multipliers",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != required:
                raise ValueError("geometry archive schema is invalid")
            arrays = {name: np.asarray(data[name]) for name in required}
    except (OSError, ValueError) as error:
        raise ValueError("geometry archive is unreadable or unsafe") from error
    expected_shapes = {
        "outer_fold": (n_folds,),
        "coef": (n_folds, n_classes, n_features),
        "intercept": (n_folds, n_classes),
        "sites": (n_folds, n_classes, n_features),
        "power_weights": (n_folds, n_classes),
        "class_centroids": (n_folds, n_classes, n_features),
        "scaler_mean": (n_folds, n_features),
        "scaler_scale": (n_folds, n_features),
        "block_multipliers": (n_folds, n_blocks),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("geometry archive shapes disagree with metadata")
    if set(arrays["outer_fold"].astype(str)) != expected_folds:
        raise ValueError("geometry and selection folds disagree")
    for name, array in arrays.items():
        if name != "outer_fold" and not np.isfinite(array.astype(float)).all():
            raise ValueError(f"geometry field {name} contains non-finite values")
    if (arrays["scaler_scale"] <= 0).any() or (arrays["block_multipliers"] <= 0).any():
        raise ValueError("geometry scaling parameters must be positive")
    if not np.allclose(arrays["sites"], arrays["coef"] / 2.0):
        raise ValueError("geometry sites disagree with identified coefficients")
    if not np.allclose(arrays["coef"].sum(axis=1), 0.0, atol=1e-10):
        raise ValueError("geometry coefficients are not in a sum-zero gauge")
    if not np.allclose(arrays["intercept"].sum(axis=1), 0.0, atol=1e-10):
        raise ValueError("geometry intercepts are not in a sum-zero gauge")
    expected_weights = arrays["intercept"] + np.sum(arrays["coef"] ** 2, axis=2) / 4.0
    if not np.allclose(arrays["power_weights"], expected_weights):
        raise ValueError("geometry power weights disagree with coefficients")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "CrowdRepresentationProbeArtifact",
    "RUN_FILES",
    "RUN_FORMAT",
    "run_crowd_representation_probe",
    "validate_crowd_representation_probe",
]
