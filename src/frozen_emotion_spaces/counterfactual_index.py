"""External no-replace index for prospective counterfactual pilot artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .counterfactual import PILOT_FORMAT, validate_counterfactual_pilot


INDEX_FORMAT = "frozen-emotion-spaces-counterfactual-index-reconstruction-v1"
INDEX_FIELDS = (
    "index_format",
    "pilot_format",
    "space",
    "source_representation",
    "source_run_metadata_sha256",
    "source_feature_matrix_sha256",
    "source_outer_split_sha256",
    "source_embedding_layer_sha256",
    "n_items",
    "input_dimension",
    "output_dimension",
    "pca_components",
    "n_sites",
    "n_constellations_per_fold",
    "n_repetitions",
    "max_samples_per_cell",
    "seed",
    "run_path",
    "metadata_sha256",
    "constellations_sha256",
    "learnability_sha256",
    "regressions_sha256",
    "transforms_sha256",
)


def build_counterfactual_index(runs_root: str | Path) -> list[dict[str, Any]]:
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"counterfactual runs root does not exist: {root}")
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
            raise ValueError(
                f"unreadable metadata below counterfactual root: {metadata_path}"
            ) from error
        if not isinstance(identity, Mapping) or identity.get("pilot_format") != PILOT_FORMAT:
            continue
        try:
            artifact = validate_counterfactual_pilot(metadata_path.parent)
        except ValueError as error:
            raise ValueError(f"invalid counterfactual run at {metadata_path.parent}") from error
        metadata = artifact.metadata
        files = metadata["files"]
        rows.append(
            {
                "index_format": INDEX_FORMAT,
                "pilot_format": PILOT_FORMAT,
                "space": metadata["space"],
                "source_representation": metadata["source_representation"],
                "source_run_metadata_sha256": metadata["source_run_metadata_sha256"],
                "source_feature_matrix_sha256": metadata[
                    "source_feature_matrix_sha256"
                ],
                "source_outer_split_sha256": metadata["source_outer_split_sha256"],
                "source_embedding_layer_sha256": metadata[
                    "source_embedding_layer_sha256"
                ],
                "n_items": int(metadata["n_items"]),
                "input_dimension": int(metadata["input_dimension"]),
                "output_dimension": int(metadata["output_dimension"]),
                "pca_components": metadata["pca_components"],
                "n_sites": int(metadata["n_sites"]),
                "n_constellations_per_fold": int(
                    metadata["n_constellations_per_fold"]
                ),
                "n_repetitions": int(metadata["n_repetitions"]),
                "max_samples_per_cell": int(metadata["max_samples_per_cell"]),
                "seed": int(metadata["seed"]),
                "run_path": artifact.directory.resolve().relative_to(root).as_posix(),
                "metadata_sha256": _sha256_file(metadata_path),
                "constellations_sha256": files["constellations.parquet"]["sha256"],
                "learnability_sha256": files["learnability.parquet"]["sha256"],
                "regressions_sha256": files["regressions.parquet"]["sha256"],
                "transforms_sha256": files["transforms.npz"]["sha256"],
            }
        )
    if not rows:
        raise ValueError("counterfactual runs root contains no completed pilot")
    return sorted(rows, key=_sort_key)


def write_counterfactual_index(
    index_path: str | Path,
    *,
    runs_root: str | Path,
) -> list[dict[str, Any]]:
    target = Path(index_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual index: {target}")
    rows = build_counterfactual_index(runs_root)
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
                f"refusing to overwrite counterfactual index: {target}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def validate_counterfactual_index(
    index_path: str | Path,
    *,
    runs_root: str | Path,
) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("counterfactual index is unreadable") from error
    schema = tuple(sorted(INDEX_FIELDS))
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("counterfactual index must be a non-empty list")
    for position, row in enumerate(loaded):
        if not isinstance(row, Mapping) or tuple(sorted(row)) != schema:
            raise ValueError(f"counterfactual index row {position} has invalid schema")
        relative = Path(str(row["run_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"counterfactual index row {position} has unsafe path")
        if row["index_format"] != INDEX_FORMAT or row["pilot_format"] != PILOT_FORMAT:
            raise ValueError(f"counterfactual index row {position} has unknown format")
    rebuilt = build_counterfactual_index(runs_root)
    if loaded != rebuilt:
        raise ValueError("counterfactual index disagrees with current pilot content")
    return [dict(row) for row in loaded]


def _sort_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row["space"]),
        str(row["output_dimension"]),
        str(row["n_constellations_per_fold"]),
        str(row["n_repetitions"]),
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
    "build_counterfactual_index",
    "validate_counterfactual_index",
    "write_counterfactual_index",
]
