"""External metadata index for reconstructed crowd layer-probe runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .experiment_a import RUN_FORMAT, validate_crowd_layer_probe


RUN_INDEX_FORMAT = "frozen-emotion-spaces-crowd-run-index-reconstruction-v1"
RUN_INDEX_FIELDS = (
    "index_format",
    "run_format",
    "dataset",
    "target",
    "model_key",
    "revision",
    "mode",
    "text_variant",
    "layer",
    "pooling",
    "selection_metric",
    "class_weight",
    "n_items",
    "run_path",
    "metadata_sha256",
    "embedding_metadata_sha256",
    "embedding_layer_sha256",
    "ordered_item_target_sha256",
    "outer_split_sha256",
    "inner_split_sha256",
)


def build_crowd_run_index(runs_root: str | Path) -> list[dict[str, Any]]:
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"crowd runs root does not exist: {root}")
    metadata_paths = sorted(root.rglob("metadata.json"))
    if not metadata_paths:
        raise ValueError("crowd runs root contains no completed runs")
    rows: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        run = validate_crowd_layer_probe(metadata_path.parent)
        metadata = run.metadata
        relative = run.directory.resolve().relative_to(root)
        rows.append(
            {
                "index_format": RUN_INDEX_FORMAT,
                "run_format": RUN_FORMAT,
                "dataset": str(metadata["dataset"]),
                "target": str(metadata["target"]),
                "model_key": str(metadata["embedding_model_key"]),
                "revision": str(metadata["embedding_revision"]),
                "mode": str(metadata["embedding_mode"]),
                "text_variant": str(metadata["embedding_text_variant"]),
                "layer": int(metadata["layer"]),
                "pooling": str(metadata["pooling"]),
                "selection_metric": str(metadata["selection_metric"]),
                "class_weight": metadata["class_weight"],
                "n_items": int(metadata["n_items"]),
                "run_path": relative.as_posix(),
                "metadata_sha256": _sha256_file(metadata_path),
                "embedding_metadata_sha256": metadata[
                    "embedding_metadata_sha256"
                ],
                "embedding_layer_sha256": metadata["embedding_layer_sha256"],
                "ordered_item_target_sha256": metadata[
                    "ordered_item_target_sha256"
                ],
                "outer_split_sha256": metadata["outer_split_sha256"],
                "inner_split_sha256": metadata["inner_split_sha256"],
            }
        )
    rows.sort(key=_row_sort_key)
    if len({row["run_path"] for row in rows}) != len(rows):  # pragma: no cover
        raise ValueError("crowd run index contains duplicate paths")
    return rows


def write_crowd_run_index(
    index_path: str | Path,
    *,
    runs_root: str | Path,
) -> list[dict[str, Any]]:
    target = Path(index_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite crowd run index: {target}")
    rows = build_crowd_run_index(runs_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite crowd run index: {target}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def validate_crowd_run_index(
    index_path: str | Path,
    *,
    runs_root: str | Path,
) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("crowd run index is unreadable") from error
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("crowd run index must be a non-empty JSON list")
    rows: list[dict[str, Any]] = []
    expected_schema = tuple(sorted(RUN_INDEX_FIELDS))
    for position, raw in enumerate(loaded):
        if not isinstance(raw, Mapping) or tuple(sorted(raw)) != expected_schema:
            raise ValueError(f"crowd run index row {position} has invalid schema")
        row = dict(raw)
        relative = Path(str(row["run_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"crowd run index row {position} has unsafe path")
        if row["index_format"] != RUN_INDEX_FORMAT or row["run_format"] != RUN_FORMAT:
            raise ValueError(f"crowd run index row {position} has unknown format")
        rows.append(row)
    if rows != sorted(rows, key=_row_sort_key):
        raise ValueError("crowd run index rows are not in canonical order")
    rebuilt = build_crowd_run_index(runs_root)
    if rows != rebuilt:
        raise ValueError("crowd run index disagrees with current run metadata/content")
    return rows


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["model_key"]),
        str(row["revision"]),
        str(row["mode"]),
        str(row["text_variant"]),
        str(row["pooling"]),
        int(row["layer"]),
        str(row["selection_metric"]),
        json.dumps(row["class_weight"], sort_keys=True),
        str(row["run_path"]),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "RUN_INDEX_FIELDS",
    "RUN_INDEX_FORMAT",
    "build_crowd_run_index",
    "validate_crowd_run_index",
    "write_crowd_run_index",
]
