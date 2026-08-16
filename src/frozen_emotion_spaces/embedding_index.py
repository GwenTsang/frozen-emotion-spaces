"""External cryptographic index for reconstructed embedding artifacts.

The historical project had ``manifests/embedding_index.json`` as a JSON list,
but its complete row schema is lost.  This clean-room format keeps that outer
shape while versioning every row and binding each artifact's otherwise
self-authenticating metadata file from outside the artifact directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .config import embedding_artifact_directory
from .embeddings import ARTIFACT_FORMAT, validate_embedding_artifact


INDEX_FORMAT = "frozen-emotion-spaces-embedding-index-reconstruction-v1"
INDEX_FILENAME = "embedding_index.json"
INDEX_FIELDS = (
    "index_format",
    "artifact_format",
    "dataset",
    "model_key",
    "mode",
    "text_variant",
    "revision",
    "max_length",
    "n_items",
    "artifact_path",
    "metadata_sha256",
    "ordered_item_text_pairs_sha256",
    "tokenizer_is_fast",
)


def build_embedding_index(cache_root: str | Path) -> list[dict[str, Any]]:
    """Validate every artifact below ``cache_root`` and return sorted rows."""

    root = Path(cache_root).resolve()
    embeddings_root = root / "embeddings"
    if not embeddings_root.is_dir():
        raise FileNotFoundError(f"embedding cache directory does not exist: {embeddings_root}")
    partial = [
        path
        for path in embeddings_root.rglob("maxlen-*")
        if path.is_dir() and not (path / "metadata.json").is_file()
    ]
    if partial:
        raise ValueError(f"embedding cache contains a partial artifact: {partial[0]}")
    metadata_paths = sorted(embeddings_root.rglob("metadata.json"))
    if not metadata_paths:
        raise ValueError("embedding cache contains no complete artifacts")

    rows: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    for metadata_path in metadata_paths:
        artifact = validate_embedding_artifact(metadata_path.parent)
        metadata = artifact.metadata
        expected = embedding_artifact_directory(
            root,
            dataset=str(metadata["dataset"]),
            model=str(metadata["model_key"]),
            mode=str(metadata["mode"]),
            text_variant=str(metadata["text_variant"]),
            max_length=int(metadata["max_length"]),
        ).resolve()
        if artifact.directory.resolve() != expected:
            raise ValueError(
                f"embedding artifact is outside its canonical metadata path: "
                f"{artifact.directory}"
            )
        relative = artifact.directory.resolve().relative_to(root).as_posix()
        if relative in observed_paths:  # pragma: no cover - filesystem invariant
            raise ValueError(f"duplicate embedding artifact path: {relative}")
        observed_paths.add(relative)
        rows.append(
            {
                "index_format": INDEX_FORMAT,
                "artifact_format": ARTIFACT_FORMAT,
                "dataset": str(metadata["dataset"]),
                "model_key": str(metadata["model_key"]),
                "mode": str(metadata["mode"]),
                "text_variant": str(metadata["text_variant"]),
                "revision": str(metadata["revision"]),
                "max_length": int(metadata["max_length"]),
                "n_items": int(metadata["n_items"]),
                "artifact_path": relative,
                "metadata_sha256": _sha256_file(metadata_path),
                "ordered_item_text_pairs_sha256": str(
                    metadata["ordered_item_text_pairs_sha256"]
                ),
                "tokenizer_is_fast": bool(metadata["tokenizer_is_fast"]),
            }
        )
    rows.sort(key=_row_sort_key)
    return rows


def write_embedding_index(
    index_path: str | Path,
    *,
    cache_root: str | Path,
) -> list[dict[str, Any]]:
    """Atomically publish a new index and refuse to replace prior evidence."""

    target = Path(index_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite embedding index: {target}")
    rows = build_embedding_index(cache_root)
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
                f"refusing to overwrite embedding index: {target}"
            ) from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return rows


def validate_embedding_index(
    index_path: str | Path,
    *,
    cache_root: str | Path,
) -> list[dict[str, Any]]:
    """Validate syntax, schema, canonical order, and every bound artifact."""

    path = Path(index_path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("embedding index is unreadable") from error
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("embedding index must be a non-empty JSON list")
    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(loaded):
        if not isinstance(raw, Mapping) or tuple(sorted(raw)) != tuple(
            sorted(INDEX_FIELDS)
        ):
            raise ValueError(f"embedding index row {position} has an invalid schema")
        row = dict(raw)
        if row["index_format"] != INDEX_FORMAT or row["artifact_format"] != ARTIFACT_FORMAT:
            raise ValueError(f"embedding index row {position} has an unknown format")
        relative = Path(str(row["artifact_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"embedding index row {position} has an unsafe path")
        rows.append(row)
    if rows != sorted(rows, key=_row_sort_key):
        raise ValueError("embedding index rows are not in canonical order")
    rebuilt = build_embedding_index(cache_root)
    if rows != rebuilt:
        raise ValueError("embedding index disagrees with current artifact metadata/content")
    return rows


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["dataset"]),
        str(row["model_key"]),
        str(row["revision"]),
        str(row["mode"]),
        str(row["text_variant"]),
        int(row["max_length"]),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "INDEX_FIELDS",
    "INDEX_FILENAME",
    "INDEX_FORMAT",
    "build_embedding_index",
    "validate_embedding_index",
    "write_embedding_index",
]
