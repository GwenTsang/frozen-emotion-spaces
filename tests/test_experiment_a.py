from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frozen_emotion_spaces.embeddings import extract_to_artifact
from frozen_emotion_spaces.experiment_a import (
    RUN_FORMAT,
    run_crowd_layer_probe,
    validate_crowd_layer_probe,
)
from frozen_emotion_spaces.run_index import (
    validate_crowd_run_index,
    write_crowd_run_index,
)


def _nested_tables(item_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.asarray([f"group-{index:02d}" for index in range(len(item_ids))])
    outer_assignment = np.repeat(np.arange(3), 9)
    outer = pd.DataFrame(
        {
            "item_id": item_ids,
            "group_id": groups,
            "test_fold": outer_assignment,
        }
    )
    rows = []
    for outer_fold in range(3):
        for index in np.flatnonzero(outer_assignment != outer_fold):
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "item_id": item_ids[index],
                    "group_id": groups[index],
                    "validation_fold": (index // 3) % 3,
                }
            )
    return outer, pd.DataFrame(rows)


def test_crowd_layer_probe_atomic_parquet_round_trip(
    roberta_artifact,
    tmp_path: Path,
) -> None:
    encoder, _, _, _ = roberta_artifact
    names = ("zeta", "alpha", "middle")
    y = np.tile(np.asarray(names), 9)
    item_ids = np.asarray([f"run-{index:02d}" for index in range(27)])
    texts = [f"{label} emotional example number {index}" for index, label in enumerate(y)]
    embedding = extract_to_artifact(
        tmp_path / "embedding",
        item_ids=item_ids,
        texts=texts,
        model="roberta-base",
        dataset="crowd",
        text_variant="masked",
        max_length=32,
        batch_size=9,
        encoder=encoder,
        local_files_only=True,
    )
    outer, inner = _nested_tables(item_ids)
    runs_root = tmp_path / "runs"
    output = runs_root / "run"
    run = run_crowd_layer_probe(
        output,
        embedding_directory=embedding,
        layer=0,
        pooling="mean",
        y=y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        C_grid=(0.1,),
        selection_metric="macro_f1",
    )

    assert run.metadata["run_format"] == RUN_FORMAT
    assert len(run.oof) == 27
    assert len(run.selections) == 3
    assert len(run.metadata["embedding_metadata_sha256"]) == 64
    assert validate_crowd_layer_probe(output).metadata == run.metadata
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_crowd_layer_probe(
            output,
            embedding_directory=embedding,
            layer=0,
            y=y,
            item_ids=item_ids,
            class_names=names,
            outer_folds=outer,
            inner_folds=inner,
            C_grid=(0.1,),
            selection_metric="macro_f1",
        )
    index_path = tmp_path / "crowd_run_index.json"
    rows = write_crowd_run_index(index_path, runs_root=runs_root)
    assert validate_crowd_run_index(index_path, runs_root=runs_root) == rows
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["layer"] = 12
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        validate_crowd_run_index(index_path, runs_root=runs_root)


def test_crowd_layer_probe_detects_parquet_corruption(
    roberta_artifact,
    tmp_path: Path,
) -> None:
    encoder, _, _, _ = roberta_artifact
    names = ("zeta", "alpha", "middle")
    y = np.tile(np.asarray(names), 9)
    item_ids = np.asarray([f"corrupt-{index:02d}" for index in range(27)])
    embedding = extract_to_artifact(
        tmp_path / "embedding",
        item_ids=item_ids,
        texts=[f"{label} example {index}" for index, label in enumerate(y)],
        model="roberta-base",
        dataset="crowd",
        text_variant="masked",
        max_length=32,
        batch_size=9,
        encoder=encoder,
        local_files_only=True,
    )
    outer, inner = _nested_tables(item_ids)
    output = tmp_path / "run"
    run_crowd_layer_probe(
        output,
        embedding_directory=embedding,
        layer=0,
        y=y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        C_grid=(0.1,),
        selection_metric="macro_f1",
    )
    path = output / "oof.parquet"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_crowd_layer_probe(output)
