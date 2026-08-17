"""Atomic per-layer runner for reconstructed EmoTwiCS Experiment B.

This module fits one-vs-rest L2 logistic probes per requested encoder layer
under conversation-disjoint outer/inner folds on the EmoTwiCS cluster-label
target matrix.  It deliberately schedules one layer per artifact so expensive
jobs can be distributed across independent machines, mirroring the crowd
Experiment A pattern.

The all-layer aggregate output enables macro-F1 and macro-AP trajectory plots
without re-reading individual layer artifacts.  This is a clean-room
reconstruction — the historical module name and broad experiment are attested
but the API and artifact schema below are new, provenance-explicit contracts.
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

from .config import get_model_spec
from .emotwics_data import CLUSTER_COLUMNS, EMOTION_CLUSTERS
from .embeddings import (
    EmbeddingArtifact,
    validate_embedding_artifact,
)
from .metrics import reconstruct_multilabel_metrics
from .probes import (
    DEFAULT_C_GRID,
    DEFAULT_THRESHOLD_GRID,
    make_dense_multilabel_factory,
    run_nested_multilabel_oof,
)


RUN_FORMAT = "frozen-emotion-spaces-emotwics-layer-probe-reconstruction-v1"
RUN_FILES = ("oof.parquet", "selections.parquet", "metadata.json")


@dataclass(frozen=True)
class EmoTwiCSLayerProbeArtifact:
    """Validated per-layer EmoTwiCS multilabel probe artifact."""

    directory: Path
    metadata: dict[str, Any]
    oof: pd.DataFrame
    selections: pd.DataFrame


@dataclass(frozen=True)
class EmoTwiCSAllLayerSummary:
    """Machine-readable aggregate of all completed layer artifacts.

    Suitable for plotting macro-F1 and macro-AP trajectories across layers.
    Each row corresponds to one layer and carries the reconstructed metrics
    plus the layer identity and provenance digests needed for reproducibility.
    """

    layers: pd.DataFrame


def run_emotwics_layer_probe(
    output_directory: str | Path,
    *,
    embedding_directory: str | Path | EmbeddingArtifact,
    layer: int,
    y: np.ndarray,
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    inner_folds: pd.DataFrame,
    label_names: Sequence[str] = EMOTION_CLUSTERS,
    pooling: str = "mean",
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    selection_metric: str | None = None,
) -> EmoTwiCSLayerProbeArtifact:
    """Fit one nested layerwise multilabel probe and atomically publish OOF output.

    Parameters
    ----------
    output_directory:
        Destination for the atomic artifact directory.  Must not already exist.
    embedding_directory:
        Path or validated artifact for the frozen encoder output.
    layer:
        Zero-based layer index into the embedding artifact.
    y:
        Binary label matrix of shape ``(n_items, n_labels)``.
    item_ids:
        String item identifiers aligned with ``y`` rows.
    outer_folds:
        Outer split table with columns ``item_id``, ``group_id``, ``test_fold``.
    inner_folds:
        Inner split table with columns ``outer_fold``, ``item_id``,
        ``validation_fold``, and optionally ``group_id``.
    label_names:
        Ordered label names matching the columns of ``y``.
    pooling:
        ``"mean"`` or ``"first"`` — which pooling file to load from the artifact.
    C_grid:
        Candidate L2 regularisation strengths.
    threshold_grid:
        Candidate probability thresholds for the macro-F1 selector.
    selection_metric:
        ``"log_loss"`` or ``"macro_f1"`` — must be set explicitly.
    """

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite EmoTwiCS layer run: {output}")
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

    names = tuple(str(name) for name in label_names)
    factory = make_dense_multilabel_factory()
    result = run_nested_multilabel_oof(
        features,
        y,
        item_ids=loaded_ids,
        label_names=names,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        estimator_factory=factory,
        C_grid=C_grid,
        threshold_grid=threshold_grid,
        selection_metric=selection_metric,
    )
    # A final reconstruction from serialized fields checks the probability axis
    # and itemwise metric contract before anything reaches disk.
    reconstruct_multilabel_metrics(result.oof, labels=names)

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
            "dataset": "emotwics",
            "target": "emotion_clusters",
            "layer": int(layer),
            "pooling": pooling,
            "label_names": list(names),
            "C_grid": [float(value) for value in C_grid],
            "threshold_grid": [float(value) for value in threshold_grid],
            "selection_metric": selection_metric,
            "n_items": len(ids),
            "n_labels": len(names),
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
            "label_matrix_sha256": _sha256_array(np.asarray(y, dtype=np.float64)),
            "ordered_item_target_sha256": _ordered_pair_digest(
                ids,
                # Produce a stable per-item label fingerprint so the digest
                # captures the target matrix ordering without an unbounded
                # serialisation of the full binary matrix.
                [
                    ",".join(
                        str(int(v)) for v in row
                    )
                    for row in np.asarray(y, dtype=np.int64)
                ],
            ),
            "outer_split_sha256": _dataframe_digest(outer_folds),
            "inner_split_sha256": _dataframe_digest(inner_folds),
            "implementation_sha256": {
                filename: _sha256_file(Path(__file__).with_name(filename))
                for filename in ("experiment_b.py", "probes.py", "metrics.py")
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
    return validate_emotwics_layer_probe(output)


def validate_emotwics_layer_probe(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> EmoTwiCSLayerProbeArtifact:
    """Reject incomplete, corrupt, or internally inconsistent EmoTwiCS layer runs."""

    root = Path(directory)
    missing = [filename for filename in RUN_FILES if not (root / filename).is_file()]
    if missing:
        raise ValueError(f"partial EmoTwiCS layer run; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("EmoTwiCS layer metadata is unreadable") from error
    if metadata.get("run_format") != RUN_FORMAT:
        raise ValueError("unknown EmoTwiCS layer run format")
    try:
        names = tuple(str(name) for name in metadata["label_names"])
        C_grid = tuple(float(value) for value in metadata["C_grid"])
        threshold_grid = tuple(float(value) for value in metadata.get("threshold_grid", ()))
        layer = int(metadata["layer"])
        n_items = int(metadata["n_items"])
        n_labels = int(metadata["n_labels"])
        spec = get_model_spec(str(metadata["embedding_model_key"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("EmoTwiCS layer metadata has invalid identity fields") from error
    if (
        not names
        or len(set(names)) != len(names)
        or n_labels != len(names)
        or not C_grid
        or any(value <= 0 or not np.isfinite(value) for value in C_grid)
        or not threshold_grid
        or any(
            value <= 0 or value >= 1 or not np.isfinite(value)
            for value in threshold_grid
        )
        or not 0 <= layer < spec.emitted_layers
        or n_items <= 0
    ):
        raise ValueError("EmoTwiCS layer metadata has invalid labels/grid/layer/count")
    if metadata.get("embedding_revision") != spec.revision:
        raise ValueError("EmoTwiCS layer metadata checkpoint revision is not pinned")
    if metadata.get("pooling") not in {"mean", "first"}:
        raise ValueError("EmoTwiCS layer metadata has an invalid pooling rule")
    selection_metric = metadata.get("selection_metric")
    if selection_metric not in {"log_loss", "macro_f1"}:
        raise ValueError("EmoTwiCS layer metadata has an invalid selection metric")
    digest_fields = (
        "embedding_metadata_sha256",
        "embedding_item_text_pairs_sha256",
        "embedding_layer_sha256",
        "label_matrix_sha256",
        "ordered_item_target_sha256",
        "outer_split_sha256",
        "inner_split_sha256",
    )
    implementation = metadata.get("implementation_sha256")
    if any(not _is_sha256(metadata.get(field)) for field in digest_fields) or not isinstance(
        implementation, Mapping
    ) or any(not _is_sha256(implementation.get(name)) for name in (
        "experiment_b.py",
        "probes.py",
        "metrics.py",
    )):
        raise ValueError("EmoTwiCS layer metadata lacks valid provenance digests")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("EmoTwiCS layer metadata lacks file records")
    for filename in ("oof.parquet", "selections.parquet"):
        record = files.get(filename)
        path = root / filename
        if not isinstance(record, Mapping) or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"EmoTwiCS layer file size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"EmoTwiCS layer file hash mismatch: {filename}")
    try:
        oof = pd.read_parquet(root / "oof.parquet", engine="pyarrow")
        selections = pd.read_parquet(
            root / "selections.parquet", engine="pyarrow"
        )
    except Exception as error:
        raise ValueError("EmoTwiCS layer parquet is unreadable") from error
    required_oof = {
        "item_id",
        "outer_fold",
        "group_id",
        "threshold",
        *(f"y_true__{name}" for name in names),
        *(f"prob__{name}" for name in names),
        *(f"pred__{name}" for name in names),
    }
    if not names or not required_oof.issubset(oof.columns):
        raise ValueError("EmoTwiCS layer OOF schema is incomplete")
    if len(oof) != n_items or oof["item_id"].duplicated().any():
        raise ValueError("EmoTwiCS layer OOF item coverage is invalid")
    if selections.empty or not {
        "outer_fold",
        "C",
        "threshold",
        "inner_log_loss_bits",
        "inner_macro_f1",
        "selection_metric",
    }.issubset(selections.columns):
        raise ValueError("EmoTwiCS layer selections schema is incomplete")
    if selections["outer_fold"].isna().any() or selections["outer_fold"].astype(
        str
    ).duplicated().any():
        raise ValueError("EmoTwiCS layer must contain one selection per outer fold")
    oof_folds = set(oof["outer_fold"].astype(str))
    selection_folds = set(selections["outer_fold"].astype(str))
    if oof_folds != selection_folds:
        raise ValueError("EmoTwiCS layer OOF and selection folds disagree")
    oof_threshold = pd.to_numeric(oof["threshold"], errors="coerce")
    if oof_threshold.isna().any() or (
        (oof_threshold <= 0) | (oof_threshold >= 1)
    ).any():
        raise ValueError("EmoTwiCS layer OOF thresholds are outside (0, 1)")
    selection_threshold_by_fold = {
        str(row.outer_fold): float(row.threshold)
        for row in selections.itertuples(index=False)
    }
    for fold_value, fold_frame in oof.groupby(oof["outer_fold"].astype(str)):
        applied = pd.unique(oof_threshold.loc[fold_frame.index])
        selected = selection_threshold_by_fold[str(fold_value)]
        if len(applied) != 1 or not np.isclose(
            float(applied[0]), selected, rtol=0, atol=0
        ):
            raise ValueError(
                "EmoTwiCS layer OOF threshold disagrees with the inner-fold selection"
            )
    if oof.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("EmoTwiCS layer OOF separates a group across outer folds")
    numeric_selection = selections[
        ["C", "threshold", "inner_log_loss_bits", "inner_macro_f1", "n_train", "n_test"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_selection.to_numpy()).all():
        raise ValueError("EmoTwiCS layer selections contain non-finite values")
    if (
        (numeric_selection["inner_log_loss_bits"] < 0).any()
        or ((numeric_selection["inner_macro_f1"] < 0) | (numeric_selection["inner_macro_f1"] > 1)).any()
        or (numeric_selection[["n_train", "n_test"]] < 0).any().any()
        or not np.equal(
            numeric_selection[["n_train", "n_test"]],
            np.floor(numeric_selection[["n_train", "n_test"]]),
        ).all().all()
    ):
        raise ValueError("EmoTwiCS layer selections contain invalid metric/count values")
    if (numeric_selection["C"] <= 0).any() or not all(
        any(np.isclose(value, candidate, rtol=0, atol=0) for candidate in C_grid)
        for value in numeric_selection["C"]
    ):
        raise ValueError("EmoTwiCS layer selected C is outside the configured grid")
    if (
        (numeric_selection["threshold"] <= 0).any()
        or (numeric_selection["threshold"] >= 1).any()
        or not all(
            any(np.isclose(value, candidate, rtol=0, atol=0) for candidate in threshold_grid)
            for value in numeric_selection["threshold"]
        )
    ):
        raise ValueError(
            "EmoTwiCS layer selected threshold is outside the configured grid"
        )
    if not selections["selection_metric"].astype(str).eq(selection_metric).all():
        raise ValueError(
            "EmoTwiCS layer selection rows disagree with metadata objective"
        )
    for row, numeric in zip(
        selections.itertuples(index=False),
        numeric_selection.itertuples(index=False),
        strict=True,
    ):
        n_test = int((oof["outer_fold"].astype(str) == str(row.outer_fold)).sum())
        if numeric.n_test != n_test or numeric.n_train != n_items - n_test:
            raise ValueError(
                "EmoTwiCS layer selection train/test counts disagree with OOF"
            )
    reconstruct_multilabel_metrics(oof, labels=names)
    return EmoTwiCSLayerProbeArtifact(
        directory=root,
        metadata=dict(metadata),
        oof=oof,
        selections=selections,
    )


def build_all_layer_summary(
    layer_artifacts: Sequence[EmoTwiCSLayerProbeArtifact],
) -> EmoTwiCSAllLayerSummary:
    """Build a machine-readable aggregate suitable for layer-trajectory plots.

    Each row carries the layer index, pooled metric summaries, and the
    per-artifact metadata digests required to identify the exact input
    provenance for that layer.  The output never overwrites individual
    layer artifacts.
    """

    if not layer_artifacts:
        raise ValueError("at least one layer artifact is required")
    layers_seen = [int(artifact.metadata["layer"]) for artifact in layer_artifacts]
    if len(set(layers_seen)) != len(layers_seen):
        raise ValueError("duplicate layer artifacts would make the trajectory ambiguous")
    provenance_keys = (
        "dataset",
        "target",
        "embedding_model_key",
        "embedding_revision",
        "pooling",
        "selection_metric",
        "n_items",
    )
    reference = layer_artifacts[0].metadata
    reference_labels = tuple(str(name) for name in reference["label_names"])
    for artifact in layer_artifacts[1:]:
        m = artifact.metadata
        disagreements = [
            key
            for key in provenance_keys
            if m.get(key) != reference.get(key)
        ]
        if tuple(str(name) for name in m["label_names"]) != reference_labels:
            disagreements.append("label_names")
        if disagreements:
            raise ValueError(
                "layer artifacts mix incompatible provenance; "
                f"disagreeing fields: {disagreements}"
            )
    rows: list[dict[str, Any]] = []
    for artifact in layer_artifacts:
        m = artifact.metadata
        names = tuple(str(name) for name in m["label_names"])
        metrics = reconstruct_multilabel_metrics(artifact.oof, labels=names)
        rows.append(
            {
                "layer": int(m["layer"]),
                "run_format": str(m["run_format"]),
                "dataset": str(m["dataset"]),
                "target": str(m["target"]),
                "pooling": str(m["pooling"]),
                "model_key": str(m["embedding_model_key"]),
                "revision": str(m["embedding_revision"]),
                "selection_metric": str(m["selection_metric"]),
                "n_items": int(m["n_items"]),
                "n_labels": int(m["n_labels"]),
                "macro_f1": float(metrics.overall["macro_f1"]),
                "macro_ap": float(metrics.overall["macro_ap"]),
                "log_loss_bits": float(metrics.overall["log_loss_bits"]),
                "brier": float(metrics.overall["brier"]),
                "hamming_accuracy": float(metrics.overall["hamming_accuracy"]),
                "accuracy": float(metrics.overall["accuracy"]),
                "embedding_layer_sha256": str(m["embedding_layer_sha256"]),
                "run_path": str(artifact.directory),
            }
        )
    frame = pd.DataFrame(rows).sort_values("layer", kind="stable").reset_index(
        drop=True
    )
    return EmoTwiCSAllLayerSummary(layers=frame)


def resumable_run_emotwics_layer_probe(
    output_directory: str | Path,
    *,
    embedding_directory: str | Path | EmbeddingArtifact,
    layer: int,
    y: np.ndarray,
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    inner_folds: pd.DataFrame,
    label_names: Sequence[str] = EMOTION_CLUSTERS,
    pooling: str = "mean",
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    selection_metric: str | None = None,
) -> EmoTwiCSLayerProbeArtifact:
    """Run or validate one EmoTwiCS layer probe with resumable semantics.

    If ``output_directory`` already exists, validate it instead of re-running.
    Raise ``ValueError`` if the existing artifact is partial or corrupt rather
    than silently overwriting it.
    """

    output = Path(output_directory)
    if output.exists():
        existing = validate_emotwics_layer_probe(output)
        _assert_run_request_matches(
            existing.metadata,
            embedding_directory=embedding_directory,
            layer=layer,
            y=y,
            item_ids=item_ids,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            label_names=label_names,
            pooling=pooling,
            C_grid=C_grid,
            threshold_grid=threshold_grid,
            selection_metric=selection_metric,
        )
        return existing
    return run_emotwics_layer_probe(
        output,
        embedding_directory=embedding_directory,
        layer=layer,
        y=y,
        item_ids=item_ids,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        label_names=label_names,
        pooling=pooling,
        C_grid=C_grid,
        threshold_grid=threshold_grid,
        selection_metric=selection_metric,
    )


# ---------------------------------------------------------------------------
# Private helpers (mirroring experiment_a conventions)
# ---------------------------------------------------------------------------

def _assert_run_request_matches(
    metadata: Mapping[str, Any],
    *,
    embedding_directory: str | Path | EmbeddingArtifact,
    layer: int,
    y: np.ndarray,
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    inner_folds: pd.DataFrame,
    label_names: Sequence[str],
    pooling: str,
    C_grid: Sequence[float],
    threshold_grid: Sequence[float],
    selection_metric: str | None,
) -> None:
    """Reject a completed artifact whose provenance disagrees with the request.

    Resumability must never silently accept an artifact produced by a
    different layer, pooling rule, objective, split revision, or target
    matrix; every cheaply recomputable provenance digest is compared.
    """

    ids = [str(item_id) for item_id in item_ids]
    target = np.asarray(y)
    if target.ndim != 2 or target.shape[0] != len(ids):
        raise ValueError("y must be a two-dimensional matrix aligned with item_ids")
    target = target.astype(np.int64)
    artifact_path = (
        embedding_directory.directory
        if isinstance(embedding_directory, EmbeddingArtifact)
        else embedding_directory
    )
    embedding = validate_embedding_artifact(
        artifact_path,
        expected_item_ids=ids,
        verify_hashes=False,
    )
    pooled_array = np.load(
        embedding.directory / f"{pooling}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    layer_index = layer if isinstance(layer, int) else -1
    if 0 <= layer_index < int(embedding.metadata["n_layers"]):
        layer_sha256 = _sha256_array(
            np.asarray(pooled_array[layer_index], dtype=np.float32)
        )
    else:
        layer_sha256 = None
    expected: dict[str, Any] = {
        "layer": layer_index,
        "pooling": pooling,
        "selection_metric": selection_metric,
        "label_names": [str(name) for name in label_names],
        "C_grid": [float(value) for value in C_grid],
        "threshold_grid": [float(value) for value in threshold_grid],
        "n_items": len(ids),
        "n_labels": int(target.shape[1]),
        "embedding_model_key": embedding.metadata["model_key"],
        "embedding_revision": embedding.metadata["revision"],
        "embedding_mode": embedding.metadata["mode"],
        "embedding_text_variant": embedding.metadata["text_variant"],
        "embedding_metadata_sha256": _sha256_file(
            embedding.directory / "metadata.json"
        ),
        "embedding_layer_sha256": layer_sha256,
        "label_matrix_sha256": _sha256_array(np.asarray(y, dtype=np.float64)),
        "ordered_item_target_sha256": _ordered_pair_digest(
            ids,
            [",".join(str(int(v)) for v in row) for row in target],
        ),
        "outer_split_sha256": _dataframe_digest(outer_folds),
        "inner_split_sha256": _dataframe_digest(inner_folds),
    }
    mismatched = [
        field
        for field, expected_value in expected.items()
        if expected_value is None or metadata.get(field) != expected_value
    ]
    if mismatched:
        raise ValueError(
            "existing EmoTwiCS layer artifact does not match the run request; "
            f"mismatched fields: {mismatched}"
        )


def _dataframe_digest(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    for column in canonical.columns:
        canonical[column] = canonical[column].astype(str)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_pair_digest(left: Sequence[str], right: Sequence[str]) -> str:
    if len(left) != len(right):
        raise ValueError("left and right must have equal length")
    digest = hashlib.sha256()
    for left_value, right_value in zip(left, right, strict=True):
        digest.update(left_value.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(right_value.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
    "EmoTwiCSAllLayerSummary",
    "EmoTwiCSLayerProbeArtifact",
    "RUN_FILES",
    "RUN_FORMAT",
    "build_all_layer_summary",
    "resumable_run_emotwics_layer_probe",
    "run_emotwics_layer_probe",
    "validate_emotwics_layer_probe",
]
