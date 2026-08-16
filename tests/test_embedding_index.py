from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from frozen_emotion_spaces.config import embedding_artifact_directory
from frozen_emotion_spaces.embedding_index import (
    INDEX_FORMAT,
    build_embedding_index,
    validate_embedding_index,
    write_embedding_index,
)


def _canonical_cache(tmp_path: Path, source: Path, metadata: dict) -> Path:
    cache = tmp_path / "cache"
    destination = embedding_artifact_directory(
        cache,
        dataset=metadata["dataset"],
        model=metadata["model_key"],
        mode=metadata["mode"],
        text_variant=metadata["text_variant"],
        max_length=metadata["max_length"],
    )
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)
    return cache


def test_embedding_index_round_trip_and_external_metadata_hash(
    roberta_artifact,
    tmp_path: Path,
) -> None:
    _, artifact, _, _ = roberta_artifact
    cache = _canonical_cache(tmp_path, artifact.directory, artifact.metadata)
    rows = build_embedding_index(cache)

    assert len(rows) == 1
    assert rows[0]["index_format"] == INDEX_FORMAT
    assert len(rows[0]["metadata_sha256"]) == 64
    index_path = tmp_path / "manifests" / "embedding_index.json"
    written = write_embedding_index(index_path, cache_root=cache)
    assert written == rows
    assert validate_embedding_index(index_path, cache_root=cache) == rows

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_embedding_index(index_path, cache_root=cache)


def test_embedding_index_detects_metadata_tampering(
    roberta_artifact,
    tmp_path: Path,
) -> None:
    _, artifact, _, _ = roberta_artifact
    cache = _canonical_cache(tmp_path, artifact.directory, artifact.metadata)
    index_path = tmp_path / "embedding_index.json"
    rows = write_embedding_index(index_path, cache_root=cache)
    artifact_path = cache / rows[0]["artifact_path"]
    metadata_path = artifact_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset"] = "changed-after-indexing"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_embedding_index(index_path, cache_root=cache)


def test_embedding_index_rejects_unindexed_partial_canonical_artifact(
    tmp_path: Path,
) -> None:
    partial = (
        tmp_path
        / "cache"
        / "embeddings"
        / "crowd"
        / "roberta-base"
        / "revision"
        / "pretrained"
        / "masked"
        / "maxlen-256"
    )
    partial.mkdir(parents=True)

    with pytest.raises(ValueError, match="partial artifact"):
        build_embedding_index(tmp_path / "cache")
