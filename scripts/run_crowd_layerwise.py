#!/usr/bin/env python
"""Resumable all-layer/multi-encoder crowd-enVENT Q1 layer-probe batch runner.

This script drives the reconstructed per-layer runner
:func:`frozen_emotion_spaces.experiment_a.run_crowd_layer_probe` across one or
more stable named embedding series and every requested encoder layer, then
publishes an immutable aggregate manifest plus a compact machine-readable
summary for the layer-trajectory analysis.  Completed child artifacts are
revalidated and reused on re-invocation (resume); partial, corrupt, or
configuration-incompatible children are rejected and are never overwritten.

Every child remains an atomic ``CrowdLayerProbeArtifact``: nesting, group
leakage controls, embedding validation, hyperparameter selection
(outer-train/inner-fold only), and atomic publication are inherited from the
per-layer runner unchanged.  This batch layer adds no transforms and no model
selection of its own.

Like every artifact format in this repository, the aggregate formats below are
new clean-room reconstruction contracts, not recovered historical source.

Series names
------------
A series is named ``model_key[/mode[/text_variant[/max_length]]]`` and is
resolved under ``--cache-root`` using the canonical cache layout
(``embeddings/crowd/<model>/<revision>/<mode>/<variant>/maxlen-<N>``).  The
canonical recorded name is always the full
``model_key/mode/text_variant/maxlen-N`` form.

Usage examples
--------------
All thirteen layers of the primary RoBERTa series::

    python scripts/run_crowd_layerwise.py \\
        --archive datasets/crowd-enVent2023.zip \\
        --splits splits \\
        --cache-root work/cache-fast \\
        --series roberta-base \\
        --output-root runs/crowd-layerwise \\
        --layers 0-12 \\
        --pooling mean \\
        --selection-metric log_loss

Two encoders, resuming after interruption (completed children are reused)::

    python scripts/run_crowd_layerwise.py \\
        --archive datasets/crowd-enVent2023.zip \\
        --splits splits \\
        --cache-root cache \\
        --series roberta-base --series deberta-v3-base/pretrained/masked/256 \\
        --output-root runs/crowd-layerwise \\
        --layers 0,6,12 \\
        --pooling mean \\
        --selection-metric log_loss \\
        --C-grid 0.0001,0.001,0.01,0.1,1,10,100

Render the trajectory from the aggregate summary only::

    python scripts/plot_crowd_layerwise.py \\
        --summary runs/crowd-layerwise/summary.json \\
        --output runs/crowd-layerwise/trajectory.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from frozen_emotion_spaces import experiment_a
from frozen_emotion_spaces.config import (
    PRIMARY_MAX_LENGTH,
    embedding_artifact_directory,
    get_model_spec,
)
from frozen_emotion_spaces.crowd_data import CROWD_EMOTIONS, build_crowd_manifests
from frozen_emotion_spaces.embeddings import (
    EmbeddingArtifact,
    validate_embedding_artifact,
)
from frozen_emotion_spaces.experiment_a import (
    CrowdLayerProbeArtifact,
    run_crowd_layer_probe,
    validate_crowd_layer_probe,
)
# The aggregate binds children with the per-layer runner's own provenance
# digests so the two definitions can never drift apart.
from frozen_emotion_spaces.experiment_a import (
    _dataframe_digest,
    _ordered_pair_digest,
    _sha256_file,
)
from frozen_emotion_spaces.metrics import reconstruct_multiclass_metrics
from frozen_emotion_spaces.probes import DEFAULT_C_GRID
from frozen_emotion_spaces.splits import read_split_bundle


BATCH_FORMAT = "frozen-emotion-spaces-crowd-layerwise-batch-reconstruction-v1"
SUMMARY_FORMAT = "frozen-emotion-spaces-crowd-layerwise-summary-reconstruction-v1"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"
PROVENANCE = {
    "origin": "clean-room reconstruction written for the crowd Q1 rerun pipeline",
    "historical_source_recovered": False,
    "generator": "scripts/run_crowd_layerwise.py",
}
SUMMARY_ROW_REQUIRED = (
    "series",
    "layer",
    "oof_macro_f1",
    "oof_log_loss_bits",
    "run_path",
    "run_metadata_sha256",
    "embedding_metadata_sha256",
    "embedding_layer_sha256",
    "ordered_item_target_sha256",
    "outer_split_sha256",
    "inner_split_sha256",
    "fold_selections",
)


@dataclass(frozen=True)
class SeriesSpec:
    """One stable named embedding series in canonical cache-layout form."""

    model_key: str
    mode: str
    text_variant: str
    max_length: int

    @property
    def name(self) -> str:
        return (
            f"{self.model_key}/{self.mode}/{self.text_variant}"
            f"/maxlen-{self.max_length}"
        )


@dataclass(frozen=True)
class LayerwiseRunRecord:
    """Invocation-local status of one (series, layer) child run."""

    series: str
    layer: int
    directory: Path
    status: str  # "completed" (fresh) or "validated" (resumed)


@dataclass(frozen=True)
class CrowdLayerwiseBatch:
    """Published aggregate locations, content, and per-child statuses."""

    output_root: Path
    manifest_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    records: tuple[LayerwiseRunRecord, ...]


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "grid must be comma-separated numbers"
        ) from error
    if not grid:
        raise argparse.ArgumentTypeError("grid must not be empty")
    if any(not np.isfinite(number) or number <= 0 for number in grid):
        raise argparse.ArgumentTypeError("grid values must be strictly positive")
    return grid


def _parse_layers(value: str) -> tuple[int, ...]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise argparse.ArgumentTypeError("layers must not be empty")
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise argparse.ArgumentTypeError(
                    f"invalid layer range: {part!r}"
                )
            try:
                start, stop = int(bounds[0]), int(bounds[1])
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid layer range: {part!r}"
                ) from error
            if start > stop:
                raise argparse.ArgumentTypeError(
                    f"layer range is descending: {part!r}"
                )
            layers.extend(range(start, stop + 1))
        else:
            try:
                layers.append(int(part))
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid layer: {part!r}"
                ) from error
    if any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be non-negative")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must not contain duplicates")
    return tuple(sorted(layers))


def _parse_series_spec(value: str) -> SeriesSpec:
    parts = value.split("/")
    if not 1 <= len(parts) <= 4:
        raise argparse.ArgumentTypeError(
            "series must be model_key[/mode[/text_variant[/max_length]]]"
        )
    try:
        spec = get_model_spec(parts[0])
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    mode = parts[1] if len(parts) > 1 else "pretrained"
    if mode not in {"pretrained", "random"}:
        raise argparse.ArgumentTypeError("series mode must be pretrained|random")
    text_variant = parts[2] if len(parts) > 2 else "masked"
    if text_variant in {"", ".", ".."}:
        raise argparse.ArgumentTypeError("series text_variant is invalid")
    try:
        max_length = int(parts[3]) if len(parts) > 3 else PRIMARY_MAX_LENGTH
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "series max_length must be an integer"
        ) from error
    if max_length <= 2:
        raise argparse.ArgumentTypeError("series max_length must exceed two")
    return SeriesSpec(
        model_key=spec.key,
        mode=mode,
        text_variant=text_variant,
        max_length=max_length,
    )


def _validate_series_name(name: str) -> str:
    """Return ``name`` as a safe relative POSIX path or reject it."""

    text = str(name)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe embedding series name: {name!r}")
    return path.as_posix()


def _assert_compatible_child(
    child: CrowdLayerProbeArtifact,
    *,
    expected: Mapping[str, Any],
) -> None:
    """Reject a resumed child whose stored contract differs from this batch."""

    mismatches: dict[str, dict[str, Any]] = {}
    for key, want in expected.items():
        got = child.metadata.get(key, "<absent>")
        if got != want:
            mismatches[key] = {"stored": got, "requested": want}
    if mismatches:
        raise ValueError(
            "existing crowd layer run is incompatible with the requested batch: "
            f"{child.directory} ({json.dumps(mismatches, sort_keys=True)})"
        )


def _summary_row(
    child: CrowdLayerProbeArtifact,
    *,
    series: str,
    run_path: str,
    run_metadata_sha256: str,
) -> dict[str, Any]:
    """Project one validated child into its compact summary record."""

    metadata = child.metadata
    names = tuple(str(name) for name in metadata["class_names"])
    overall = reconstruct_multiclass_metrics(child.oof, labels=names).overall
    fold_selections = []
    for row in child.selections.itertuples(index=False):
        fold = row.outer_fold
        fold_selections.append(
            {
                "outer_fold": (
                    int(fold) if isinstance(fold, (int, np.integer)) else str(fold)
                ),
                "C": float(row.C),
                "inner_log_loss_bits": float(row.inner_log_loss_bits),
                "inner_macro_f1": float(row.inner_macro_f1),
            }
        )
    return {
        "series": series,
        "layer": int(metadata["layer"]),
        "oof_macro_f1": float(overall["macro_f1"]),
        "oof_log_loss_bits": float(overall["log_loss_bits"]),
        "run_path": run_path,
        "run_metadata_sha256": run_metadata_sha256,
        "embedding_metadata_sha256": str(metadata["embedding_metadata_sha256"]),
        "embedding_layer_sha256": str(metadata["embedding_layer_sha256"]),
        "ordered_item_target_sha256": str(metadata["ordered_item_target_sha256"]),
        "outer_split_sha256": str(metadata["outer_split_sha256"]),
        "inner_split_sha256": str(metadata["inner_split_sha256"]),
        "fold_selections": fold_selections,
    }


def _require_identical_aggregate(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FileExistsError(
            f"refusing to overwrite unreadable aggregate artifact: {path}"
        ) from error
    if existing != payload:
        raise FileExistsError(
            "refusing to overwrite aggregate artifact with different content: "
            f"{path}"
        )


def _publish_immutable_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically publish JSON; identical re-publication is an idempotent no-op.

    Returns True when the file was newly written.  Any pre-existing divergent
    (or unreadable) content is refused: aggregate evidence is never replaced.
    """

    if path.exists():
        _require_identical_aggregate(path, payload)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require_identical_aggregate(path, payload)
            return False
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


def run_layerwise_batch(
    output_root: str | Path,
    *,
    embedding_series: Mapping[str, str | Path],
    layers: Sequence[int],
    y: Sequence[str],
    item_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    inner_folds: pd.DataFrame,
    class_names: Sequence[str] = CROWD_EMOTIONS,
    pooling: str = "mean",
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    selection_metric: str,
) -> CrowdLayerwiseBatch:
    """Run or revalidate every (series, layer) child, then publish aggregates.

    Children are atomic per-layer artifacts; an interrupted batch resumes by
    revalidating completed children against the requested configuration and
    running only the missing ones.  The aggregate manifest and summary are
    published only when every requested child is complete and valid.
    """

    if selection_metric not in {"log_loss", "macro_f1"}:
        raise ValueError(
            "selection_metric must be explicitly set to 'log_loss' or 'macro_f1'"
        )
    if pooling not in {"mean", "first"}:
        raise ValueError("pooling must be 'mean' or 'first'")
    root = Path(output_root)
    names = tuple(str(name) for name in class_names)
    ids = [str(item_id) for item_id in item_ids]
    labels = [str(value) for value in y]
    grid = tuple(float(value) for value in C_grid)
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("C_grid must contain only positive finite values")
    ordered_layers = tuple(sorted({int(layer) for layer in layers}))
    if not ordered_layers or any(layer < 0 for layer in ordered_layers):
        raise ValueError("layers must be non-empty non-negative integers")
    if not embedding_series:
        raise ValueError("at least one embedding series is required")

    series_artifacts: dict[str, EmbeddingArtifact] = {}
    embedding_metadata_digests: dict[str, str] = {}
    for raw_name, directory in embedding_series.items():
        name = _validate_series_name(raw_name)
        if name in series_artifacts:
            raise ValueError(f"duplicate embedding series name: {name!r}")
        artifact = validate_embedding_artifact(
            Path(directory),
            expected_item_ids=ids,
        )
        n_layers = int(artifact.metadata["n_layers"])
        out_of_range = [layer for layer in ordered_layers if layer >= n_layers]
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} are outside the validated layer axis of "
                f"series {name!r} (n_layers={n_layers})"
            )
        series_artifacts[name] = artifact
        embedding_metadata_digests[name] = _sha256_file(
            artifact.directory / "metadata.json"
        )

    target_digest = _ordered_pair_digest(ids, labels)
    outer_digest = _dataframe_digest(outer_folds)
    inner_digest = _dataframe_digest(inner_folds)

    records: list[LayerwiseRunRecord] = []
    children: dict[tuple[str, int], CrowdLayerProbeArtifact] = {}
    for name, artifact in series_artifacts.items():
        embedding_metadata = artifact.metadata
        layer_digests = embedding_metadata["files"][f"{pooling}.npy"][
            "layer_sha256"
        ]
        for layer in ordered_layers:
            directory = root / name / pooling / f"layer-{layer}"
            expected = {
                "dataset": "crowd",
                "target": "y_writer",
                "layer": layer,
                "pooling": pooling,
                "class_names": list(names),
                "C_grid": [float(value) for value in grid],
                "selection_metric": selection_metric,
                "class_weight": None,
                "n_items": len(ids),
                "embedding_model_key": str(embedding_metadata["model_key"]),
                "embedding_revision": str(embedding_metadata["revision"]),
                "embedding_mode": str(embedding_metadata["mode"]),
                "embedding_text_variant": str(embedding_metadata["text_variant"]),
                "embedding_metadata_sha256": embedding_metadata_digests[name],
                "embedding_item_text_pairs_sha256": str(
                    embedding_metadata["ordered_item_text_pairs_sha256"]
                ),
                "embedding_layer_sha256": str(layer_digests[layer]),
                "ordered_item_target_sha256": target_digest,
                "outer_split_sha256": outer_digest,
                "inner_split_sha256": inner_digest,
            }
            if directory.exists():
                child = validate_crowd_layer_probe(directory)
                _assert_compatible_child(child, expected=expected)
                status = "validated"
            else:
                child = run_crowd_layer_probe(
                    directory,
                    embedding_directory=artifact.directory,
                    layer=layer,
                    y=labels,
                    item_ids=ids,
                    outer_folds=outer_folds,
                    inner_folds=inner_folds,
                    class_names=names,
                    pooling=pooling,
                    C_grid=grid,
                    selection_metric=selection_metric,
                )
                status = "completed"
            children[(name, layer)] = child
            records.append(
                LayerwiseRunRecord(
                    series=name,
                    layer=layer,
                    directory=directory,
                    status=status,
                )
            )

    series_entries = [
        {
            "name": name,
            "model_key": str(artifact.metadata["model_key"]),
            "repository": str(artifact.metadata["repository"]),
            "revision": str(artifact.metadata["revision"]),
            "mode": str(artifact.metadata["mode"]),
            "text_variant": str(artifact.metadata["text_variant"]),
            "max_length": int(artifact.metadata["max_length"]),
            "n_layers": int(artifact.metadata["n_layers"]),
            "n_items": int(artifact.metadata["n_items"]),
            "embedding_metadata_sha256": embedding_metadata_digests[name],
            "ordered_item_text_pairs_sha256": str(
                artifact.metadata["ordered_item_text_pairs_sha256"]
            ),
        }
        for name, artifact in series_artifacts.items()
    ]
    run_entries: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for record in records:
        child = children[(record.series, record.layer)]
        run_path = record.directory.relative_to(root).as_posix()
        run_metadata_sha256 = _sha256_file(record.directory / "metadata.json")
        run_entries.append(
            {
                "series": record.series,
                "layer": record.layer,
                "run_path": run_path,
                "run_metadata_sha256": run_metadata_sha256,
                "embedding_layer_sha256": str(
                    child.metadata["embedding_layer_sha256"]
                ),
            }
        )
        summary_rows.append(
            _summary_row(
                child,
                series=record.series,
                run_path=run_path,
                run_metadata_sha256=run_metadata_sha256,
            )
        )

    shared = {
        "provenance": dict(PROVENANCE),
        "dataset": "crowd",
        "target": "y_writer",
        "pooling": pooling,
        "selection_metric": selection_metric,
        "C_grid": [float(value) for value in grid],
        "layers": [int(layer) for layer in ordered_layers],
        "class_names": list(names),
        "n_items": len(ids),
        "ordered_item_target_sha256": target_digest,
        "outer_split_sha256": outer_digest,
        "inner_split_sha256": inner_digest,
    }
    manifest = {
        "batch_format": BATCH_FORMAT,
        **shared,
        "series": series_entries,
        "runs": run_entries,
        "implementation_sha256": {
            "run_crowd_layerwise.py": _sha256_file(Path(__file__)),
            **{
                filename: _sha256_file(
                    Path(experiment_a.__file__).with_name(filename)
                )
                for filename in ("experiment_a.py", "probes.py", "metrics.py")
            },
        },
        "pandas_version": distribution_version("pandas"),
        "scikit_learn_version": distribution_version("scikit-learn"),
        "pyarrow_version": distribution_version("pyarrow"),
    }
    summary = {
        "summary_format": SUMMARY_FORMAT,
        **shared,
        "series": [entry["name"] for entry in series_entries],
        "rows": summary_rows,
    }
    manifest_path = root / MANIFEST_FILENAME
    summary_path = root / SUMMARY_FILENAME
    _publish_immutable_json(manifest_path, manifest)
    _publish_immutable_json(summary_path, summary)
    return CrowdLayerwiseBatch(
        output_root=root,
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=manifest,
        summary=summary,
        records=tuple(records),
    )


def load_layerwise_summary(path: str | Path) -> dict[str, Any]:
    """Load and validate an aggregate summary without touching any run tree."""

    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("crowd layerwise summary is unreadable") from error
    if not isinstance(loaded, Mapping) or loaded.get("summary_format") != SUMMARY_FORMAT:
        raise ValueError("unknown crowd layerwise summary format")
    if loaded.get("pooling") not in {"mean", "first"} or loaded.get(
        "selection_metric"
    ) not in {"log_loss", "macro_f1"}:
        raise ValueError("crowd layerwise summary has invalid batch fields")
    rows = loaded.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("crowd layerwise summary must contain a non-empty rows list")
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or not all(
            key in row for key in SUMMARY_ROW_REQUIRED
        ):
            raise ValueError(
                f"crowd layerwise summary row {position} has an incomplete schema"
            )
        if (
            not isinstance(row["series"], str)
            or not row["series"]
            or not isinstance(row["run_path"], str)
            or not row["run_path"]
        ):
            raise ValueError(
                f"crowd layerwise summary row {position} has invalid identity fields"
            )
        if not isinstance(row["layer"], int) or row["layer"] < 0:
            raise ValueError(
                f"crowd layerwise summary row {position} has an invalid layer"
            )
        metrics = (row["oof_macro_f1"], row["oof_log_loss_bits"])
        if any(
            not isinstance(value, (int, float)) or not np.isfinite(value)
            for value in metrics
        ):
            raise ValueError(
                f"crowd layerwise summary row {position} has non-finite metrics"
            )
        folds = row["fold_selections"]
        if not isinstance(folds, list) or not folds:
            raise ValueError(
                f"crowd layerwise summary row {position} lacks fold selections"
            )
        for fold in folds:
            C = fold.get("C") if isinstance(fold, Mapping) else None
            if (
                not isinstance(fold, Mapping)
                or "outer_fold" not in fold
                or not isinstance(C, (int, float))
                or not np.isfinite(C)
                or C <= 0
            ):
                raise ValueError(
                    "crowd layerwise summary row "
                    f"{position} has an invalid fold selection"
                )
    return dict(loaded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_crowd_layerwise",
        description=(
            "Run the resumable all-layer/multi-encoder crowd Q1 layerwise "
            "probe batch from cached frozen embedding artifacts (clean-room "
            "reconstruction)."
        ),
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="cache root holding canonically laid out embedding artifacts",
    )
    parser.add_argument(
        "--series",
        action="append",
        type=_parse_series_spec,
        required=True,
        metavar="MODEL[/MODE[/VARIANT[/MAXLEN]]]",
        help=(
            "stable named embedding series resolved under --cache-root "
            "(repeatable); defaults: pretrained/masked/"
            f"{PRIMARY_MAX_LENGTH}"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=_parse_layers,
        required=True,
        metavar="SPEC",
        help="comma-separated layers and ranges, e.g. '0-12' or '0,6,12'",
    )
    parser.add_argument("--pooling", choices=("mean", "first"), default="mean")
    parser.add_argument(
        "--selection-metric",
        choices=("log_loss", "macro_f1"),
        required=True,
    )
    parser.add_argument(
        "--C-grid",
        type=_parse_float_grid,
        default=DEFAULT_C_GRID,
        metavar="C1,C2,...",
        help="comma-separated L2 regularisation strengths (default: built-in grid)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    crowd = build_crowd_manifests(arguments.archive)
    generation = crowd.generation
    splits = read_split_bundle(arguments.splits)
    embedding_series: dict[str, Path] = {}
    for spec in arguments.series:
        if spec.name in embedding_series:
            raise ValueError(f"duplicate embedding series: {spec.name}")
        embedding_series[spec.name] = embedding_artifact_directory(
            arguments.cache_root,
            dataset="crowd",
            model=spec.model_key,
            mode=spec.mode,
            text_variant=spec.text_variant,
            max_length=spec.max_length,
        )
    batch = run_layerwise_batch(
        arguments.output_root,
        embedding_series=embedding_series,
        layers=arguments.layers,
        y=generation["y_writer"].astype(str).tolist(),
        item_ids=generation["item_id"].astype(str).tolist(),
        outer_folds=splits.crowd_full_outer,
        inner_folds=splits.crowd_full_inner,
        pooling=arguments.pooling,
        C_grid=arguments.C_grid,
        selection_metric=arguments.selection_metric,
    )
    for record in batch.records:
        print(
            f"[{record.status}] series={record.series} "
            f"layer={record.layer} {record.directory}"
        )
    print(f"manifest: {batch.manifest_path}")
    print(f"summary: {batch.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
