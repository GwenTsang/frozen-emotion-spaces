"""External no-replace index for reconstructed A/H/AH probe artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .experiment_c import RUN_FORMAT, validate_crowd_representation_probe


INDEX_FORMAT = "frozen-emotion-spaces-representation-run-index-reconstruction-v1"
INDEX_FIELDS = (
    "index_format",
    "run_format",
    "representation",
    "model_key",
    "revision",
    "layer",
    "pooling",
    "selection_metric",
    "class_weight",
    "n_items",
    "run_path",
    "metadata_sha256",
    "feature_matrix_sha256",
    "outer_split_sha256",
    "inner_split_sha256",
    "oof_sha256",
    "selections_sha256",
    "geometry_sha256",
)


def build_representation_run_index(runs_root: str | Path) -> list[dict[str, Any]]:
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"representation runs root does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        relative_metadata = metadata_path.resolve().relative_to(root)
        if any(
            part.startswith(".") or ".tmp-" in part
            for part in relative_metadata.parts
        ):
            continue
        try:
            identity = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unreadable metadata below runs root: {metadata_path}") from error
        if not isinstance(identity, Mapping) or identity.get("run_format") != RUN_FORMAT:
            continue
        try:
            artifact = validate_crowd_representation_probe(metadata_path.parent)
        except ValueError as error:
            raise ValueError(f"invalid representation run at {metadata_path.parent}") from error
        metadata = artifact.metadata
        relative = artifact.directory.resolve().relative_to(root)
        files = metadata["files"]
        rows.append(
            {
                "index_format": INDEX_FORMAT,
                "run_format": RUN_FORMAT,
                "representation": metadata["representation"],
                "model_key": metadata["embedding_model_key"],
                "revision": metadata["embedding_revision"],
                "layer": metadata["layer"],
                "pooling": metadata["pooling"],
                "selection_metric": metadata["selection_metric"],
                "class_weight": metadata["class_weight"],
                "n_items": int(metadata["n_items"]),
                "run_path": relative.as_posix(),
                "metadata_sha256": _sha256_file(metadata_path),
                "feature_matrix_sha256": metadata["feature_matrix_sha256"],
                "outer_split_sha256": metadata["outer_split_sha256"],
                "inner_split_sha256": metadata["inner_split_sha256"],
                "oof_sha256": files["oof.parquet"]["sha256"],
                "selections_sha256": files["selections.parquet"]["sha256"],
                "geometry_sha256": files["geometry.npz"]["sha256"],
            }
        )
    if not rows:
        raise ValueError("representation runs root contains no completed runs")
    return sorted(rows, key=_sort_key)


def write_representation_run_index(
    index_path: str | Path,
    *,
    runs_root: str | Path,
) -> list[dict[str, Any]]:
    target = Path(index_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite representation run index: {target}")
    rows = build_representation_run_index(runs_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite representation run index: {target}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def validate_representation_run_index(
    index_path: str | Path,
    *,
    runs_root: str | Path,
) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("representation run index is unreadable") from error
    schema = tuple(sorted(INDEX_FIELDS))
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("representation run index must be a non-empty list")
    for position, raw in enumerate(loaded):
        if not isinstance(raw, Mapping) or tuple(sorted(raw)) != schema:
            raise ValueError(f"representation index row {position} has invalid schema")
        relative = Path(str(raw["run_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"representation index row {position} has unsafe path")
        if raw["index_format"] != INDEX_FORMAT or raw["run_format"] != RUN_FORMAT:
            raise ValueError(f"representation index row {position} has unknown format")
    rebuilt = build_representation_run_index(runs_root)
    if loaded != rebuilt:
        raise ValueError("representation run index disagrees with current run content")
    return [dict(row) for row in loaded]


def _sort_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row["representation"]),
        str(row["model_key"]),
        str(row["layer"]),
        str(row["pooling"]),
        str(row["run_path"]),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "INDEX_FIELDS",
    "INDEX_FORMAT",
    "build_representation_run_index",
    "validate_representation_run_index",
    "write_representation_run_index",
]
