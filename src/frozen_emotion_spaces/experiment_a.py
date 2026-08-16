"""Atomic per-layer runner for reconstructed crowd Experiment A.

This module deliberately schedules one encoder layer per artifact so expensive
jobs can be distributed across independent machines.  The historical module
name and broad experiment are attested; the API and artifact schema below are
new, provenance-explicit reconstruction contracts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import get_model_spec
from .crowd_data import CROWD_EMOTIONS
from .embeddings import (
    EmbeddingArtifact,
    validate_embedding_artifact,
)
from .metrics import reconstruct_multiclass_metrics
from .probes import (
    DEFAULT_C_GRID,
    make_dense_multiclass_factory,
    run_nested_multiclass_oof,
)


RUN_FORMAT = "frozen-emotion-spaces-crowd-layer-probe-reconstruction-v1"
RUN_FILES = ("oof.parquet", "selections.parquet", "metadata.json")


@dataclass(frozen=True)
class CrowdLayerProbeArtifact:
    directory: Path
    metadata: dict[str, Any]
    oof: pd.DataFrame
    selections: pd.DataFrame


def run_crowd_layer_probe(
    output_directory: str | Path,
    *,
    embedding_directory: str | Path | EmbeddingArtifact,
    layer: int,
    y: Sequence[str],
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    inner_folds: pd.DataFrame,
    class_names: Sequence[str] = CROWD_EMOTIONS,
    pooling: str = "mean",
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    selection_metric: str | None = None,
    class_weight: str | dict[str, float] | None = None,
) -> CrowdLayerProbeArtifact:
    """Fit one nested layerwise probe and atomically publish raw OOF output."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite crowd layer run: {output}")
    if selection_metric not in {"log_loss", "macro_f1"}:
        raise ValueError(
            "selection_metric must be explicitly set to 'log_loss' or 'macro_f1'"
        )
    ids = [str(item_id) for item_id in item_ids]
    artifact_path = (
        embedding_directory.directory
        if isinstance(embedding_directory, EmbeddingArtifact)
        else embedding_directory
    )
    artifact = validate_embedding_artifact(
        artifact_path,
        expected_item_ids=ids,
    )
    if pooling not in {"mean", "first"}:
        raise ValueError("pooling must be 'mean' or 'first'")
    if not isinstance(layer, int) or not 0 <= layer < int(artifact.metadata["n_layers"]):
        raise ValueError("layer is outside the validated embedding layer axis")
    pooled_array = np.load(
        artifact.directory / f"{pooling}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    features = np.asarray(pooled_array[layer], dtype=np.float32)
    loaded_ids = artifact.item_ids.copy()
    embedding_layer_sha256 = _sha256_array(features)
    expected_layer_sha256 = artifact.metadata["files"][f"{pooling}.npy"][
        "layer_sha256"
    ][layer]
    if embedding_layer_sha256 != expected_layer_sha256:
        raise ValueError("loaded embedding layer disagrees with artifact metadata")
    names = tuple(str(name) for name in class_names)
    factory = make_dense_multiclass_factory(
        class_names=names,
        class_weight=class_weight,
    )
    result = run_nested_multiclass_oof(
        features,
        y,
        item_ids=loaded_ids,
        class_names=names,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        estimator_factory=factory,
        C_grid=C_grid,
        selection_metric=selection_metric,
    )
    # A final reconstruction from serialized fields checks the probability axis
    # and itemwise metric contract before anything reaches disk.
    reconstruct_multiclass_metrics(result.oof, labels=names)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        oof_path = temporary / "oof.parquet"
        selection_path = temporary / "selections.parquet"
        result.oof.to_parquet(
            oof_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        result.selections.to_parquet(
            selection_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        source_metadata = artifact.directory / "metadata.json"
        file_records = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (oof_path, selection_path)
        }
        metadata = {
            "run_format": RUN_FORMAT,
            "dataset": "crowd",
            "target": "y_writer",
            "layer": int(layer),
            "pooling": pooling,
            "class_names": list(names),
            "C_grid": [float(value) for value in C_grid],
            "selection_metric": selection_metric,
            "class_weight": class_weight,
            "n_items": len(ids),
            "embedding_artifact_format": artifact.metadata["artifact_format"],
            "embedding_model_key": artifact.metadata["model_key"],
            "embedding_revision": artifact.metadata["revision"],
            "embedding_mode": artifact.metadata["mode"],
            "embedding_text_variant": artifact.metadata["text_variant"],
            "embedding_metadata_sha256": _sha256_file(source_metadata),
            "embedding_item_text_pairs_sha256": artifact.metadata[
                "ordered_item_text_pairs_sha256"
            ],
            "embedding_layer_sha256": embedding_layer_sha256,
            "ordered_item_target_sha256": _ordered_pair_digest(ids, y),
            "outer_split_sha256": _dataframe_digest(outer_folds),
            "inner_split_sha256": _dataframe_digest(inner_folds),
            "implementation_sha256": {
                filename: _sha256_file(Path(__file__).with_name(filename))
                for filename in ("experiment_a.py", "probes.py", "metrics.py")
            },
            "pandas_version": distribution_version("pandas"),
            "scikit_learn_version": distribution_version("scikit-learn"),
            "pyarrow_version": distribution_version("pyarrow"),
            "files": file_records,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_crowd_layer_probe(output)


def validate_crowd_layer_probe(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> CrowdLayerProbeArtifact:
    """Reject incomplete, corrupt, or internally inconsistent layer runs."""

    root = Path(directory)
    missing = [filename for filename in RUN_FILES if not (root / filename).is_file()]
    if missing:
        raise ValueError(f"partial crowd layer run; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("crowd layer metadata is unreadable") from error
    if metadata.get("run_format") != RUN_FORMAT:
        raise ValueError("unknown crowd layer run format")
    try:
        names = tuple(str(name) for name in metadata["class_names"])
        C_grid = tuple(float(value) for value in metadata["C_grid"])
        layer = int(metadata["layer"])
        n_items = int(metadata["n_items"])
        spec = get_model_spec(str(metadata["embedding_model_key"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("crowd layer metadata has invalid identity fields") from error
    if (
        not names
        or len(set(names)) != len(names)
        or not C_grid
        or any(value <= 0 or not np.isfinite(value) for value in C_grid)
        or not 0 <= layer < spec.emitted_layers
        or n_items <= 0
    ):
        raise ValueError("crowd layer metadata has invalid classes/grid/layer/count")
    if metadata.get("embedding_revision") != spec.revision:
        raise ValueError("crowd layer metadata checkpoint revision is not pinned")
    if metadata.get("pooling") not in {"mean", "first"}:
        raise ValueError("crowd layer metadata has an invalid pooling rule")
    selection_metric = metadata.get("selection_metric")
    if selection_metric not in {"log_loss", "macro_f1"}:
        raise ValueError("crowd layer metadata has an invalid selection metric")
    class_weight = metadata.get("class_weight")
    if class_weight not in {None, "balanced"} and not isinstance(class_weight, Mapping):
        raise ValueError("crowd layer metadata has invalid class weighting")
    digest_fields = (
        "embedding_metadata_sha256",
        "embedding_item_text_pairs_sha256",
        "embedding_layer_sha256",
        "ordered_item_target_sha256",
        "outer_split_sha256",
        "inner_split_sha256",
    )
    implementation = metadata.get("implementation_sha256")
    if any(not _is_sha256(metadata.get(field)) for field in digest_fields) or not isinstance(
        implementation, Mapping
    ) or any(not _is_sha256(implementation.get(name)) for name in (
        "experiment_a.py",
        "probes.py",
        "metrics.py",
    )):
        raise ValueError("crowd layer metadata lacks valid provenance digests")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("crowd layer metadata lacks file records")
    for filename in ("oof.parquet", "selections.parquet"):
        record = files.get(filename)
        path = root / filename
        if not isinstance(record, Mapping) or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"crowd layer file size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"crowd layer file hash mismatch: {filename}")
    try:
        oof = pd.read_parquet(root / "oof.parquet", engine="pyarrow")
        selections = pd.read_parquet(
            root / "selections.parquet", engine="pyarrow"
        )
    except Exception as error:
        raise ValueError("crowd layer parquet is unreadable") from error
    required_oof = {
        "item_id",
        "outer_fold",
        "group_id",
        "y_true",
        "y_pred",
        *(f"prob__{name}" for name in names),
    }
    if not names or not required_oof.issubset(oof.columns):
        raise ValueError("crowd layer OOF schema is incomplete")
    if len(oof) != n_items or oof["item_id"].duplicated().any():
        raise ValueError("crowd layer OOF item coverage is invalid")
    if selections.empty or not {
        "outer_fold",
        "C",
        "inner_log_loss_bits",
        "inner_macro_f1",
        "selection_metric",
    }.issubset(selections.columns):
        raise ValueError("crowd layer selections schema is incomplete")
    if selections["outer_fold"].isna().any() or selections["outer_fold"].astype(
        str
    ).duplicated().any():
        raise ValueError("crowd layer must contain one selection per outer fold")
    oof_folds = set(oof["outer_fold"].astype(str))
    selection_folds = set(selections["outer_fold"].astype(str))
    if oof_folds != selection_folds:
        raise ValueError("crowd layer OOF and selection folds disagree")
    if oof.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("crowd layer OOF separates a group across outer folds")
    if not set(oof["y_true"].astype(str)).issubset(names) or not set(
        oof["y_pred"].astype(str)
    ).issubset(names):
        raise ValueError("crowd layer OOF contains unknown target classes")
    numeric_selection = selections[
        ["C", "inner_log_loss_bits", "inner_macro_f1", "n_train", "n_test"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_selection.to_numpy()).all():
        raise ValueError("crowd layer selections contain non-finite values")
    if (
        (numeric_selection["inner_log_loss_bits"] < 0).any()
        or ((numeric_selection["inner_macro_f1"] < 0) | (numeric_selection["inner_macro_f1"] > 1)).any()
        or (numeric_selection[["n_train", "n_test"]] < 0).any().any()
        or not np.equal(
            numeric_selection[["n_train", "n_test"]],
            np.floor(numeric_selection[["n_train", "n_test"]]),
        ).all().all()
    ):
        raise ValueError("crowd layer selections contain invalid metric/count values")
    if (numeric_selection["C"] <= 0).any() or not all(
        any(np.isclose(value, candidate, rtol=0, atol=0) for candidate in C_grid)
        for value in numeric_selection["C"]
    ):
        raise ValueError("crowd layer selected C is outside the configured grid")
    if not selections["selection_metric"].astype(str).eq(selection_metric).all():
        raise ValueError("crowd layer selection rows disagree with metadata objective")
    for row, numeric in zip(
        selections.itertuples(index=False),
        numeric_selection.itertuples(index=False),
        strict=True,
    ):
        n_test = int((oof["outer_fold"].astype(str) == str(row.outer_fold)).sum())
        if numeric.n_test != n_test or numeric.n_train != n_items - n_test:
            raise ValueError("crowd layer selection train/test counts disagree with OOF")
    reconstruct_multiclass_metrics(oof, labels=names)
    return CrowdLayerProbeArtifact(
        directory=root,
        metadata=dict(metadata),
        oof=oof,
        selections=selections,
    )


def _dataframe_digest(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical.columns = canonical.columns.astype(str)
    canonical = canonical.reindex(sorted(canonical.columns), axis=1)
    canonical = canonical.astype(str).sort_values(
        list(canonical.columns), kind="stable"
    )
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_pair_digest(left: Sequence[str], right: Sequence[str]) -> str:
    if len(left) != len(right):
        raise ValueError("item_ids and targets must have equal length")
    digest = hashlib.sha256()
    for first, second in zip(left, right, strict=True):
        for value in (str(first), str(second)):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "RUN_FILES",
    "RUN_FORMAT",
    "CrowdLayerProbeArtifact",
    "run_crowd_layer_probe",
    "validate_crowd_layer_probe",
]
