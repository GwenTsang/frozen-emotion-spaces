from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frozen_emotion_spaces.embeddings import extract_to_artifact
from frozen_emotion_spaces.experiment_c import (
    RUN_FORMAT,
    run_crowd_representation_probe,
    validate_crowd_representation_probe,
)
from frozen_emotion_spaces.representation_index import (
    validate_representation_run_index,
    write_representation_run_index,
)
from frozen_emotion_spaces.observed_geometry import (
    score_observed_run_geometry,
    validate_observed_geometry_analysis,
    write_observed_geometry_analysis,
)
from frozen_emotion_spaces.conditional_analysis import (
    validate_conditional_analysis,
    write_conditional_analysis,
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
    rows: list[dict[str, object]] = []
    for outer_fold in range(3):
        train_indices = np.flatnonzero(outer_assignment != outer_fold)
        for position, index in enumerate(train_indices):
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "item_id": item_ids[index],
                    "group_id": groups[index],
                    "validation_fold": (position // 3) % 3,
                }
            )
    return outer, pd.DataFrame(rows)


def _toy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    names = ("zeta", "alpha", "middle")
    item_ids = np.asarray([f"conditional-{index:02d}" for index in range(27)])
    y = np.tile(np.asarray(names), 9)
    rng = np.random.default_rng(20240804)
    appraisal = rng.normal(size=(27, 3))
    for class_index, name in enumerate(names):
        appraisal[y == name, class_index] += 2.0
    return item_ids, y, appraisal, names


def _rewrite_file_record(run: Path, filename: str) -> None:
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    path = run / filename
    metadata["files"][filename] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_A_representation_probe_atomic_round_trip(tmp_path: Path) -> None:
    item_ids, y, appraisal, names = _toy_data()
    outer, inner = _nested_tables(item_ids)
    output = tmp_path / "A-run"
    artifact = run_crowd_representation_probe(
        output,
        representation="A",
        appraisals=appraisal,
        appraisal_names=("a", "b", "c"),
        y=y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        C_grid=(0.1,),
        selection_metric="log_loss",
    )

    assert artifact.metadata["run_format"] == RUN_FORMAT
    assert artifact.metadata["representation"] == "A"
    assert artifact.metadata["embedding_model_key"] is None
    assert len(artifact.oof) == 27
    assert len(artifact.selections) == 3
    assert set(artifact.selections["block_multiplier"]) == {1.0}
    assert validate_crowd_representation_probe(output).metadata == artifact.metadata
    geometry_scores = score_observed_run_geometry(
        output,
        features=appraisal,
        item_ids=item_ids,
        outer_folds=outer,
    )
    assert len(geometry_scores) == 6
    assert set(geometry_scores["site_family"]) == {
        "class_centroids",
        "linear_decoder_sites",
    }
    assert (
        geometry_scores.loc[
            geometry_scores["site_family"] == "class_centroids", "n_empty_cells"
        ]
        == 0
    ).all()
    observed = write_observed_geometry_analysis(
        tmp_path / "observed",
        run_directory=output,
        features=appraisal,
        item_ids=item_ids,
        outer_folds=outer,
    )
    assert validate_observed_geometry_analysis(observed.directory).metadata == observed.metadata
    bad_observed = tmp_path / "observed-negative-contrast"
    shutil.copytree(observed.directory, bad_observed)
    bad_scores = pd.read_parquet(bad_observed / "scores.parquet")
    bad_scores.loc[0, "contrast_sum"] = -1.0
    bad_scores.to_parquet(bad_observed / "scores.parquet", index=False)
    _rewrite_file_record(bad_observed, "scores.parquet")
    with pytest.raises(ValueError, match="dimensions or scores are invalid"):
        validate_observed_geometry_analysis(bad_observed)
    with np.load(output / "geometry.npz", allow_pickle=False) as geometry:
        assert geometry["coef"].shape == (3, 3, 3)
        assert geometry["class_centroids"].shape == (3, 3, 3)
        assert np.allclose(geometry["sites"], geometry["coef"] / 2.0)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_crowd_representation_probe(
            output,
            representation="A",
            appraisals=appraisal,
            appraisal_names=("a", "b", "c"),
            y=y,
            item_ids=item_ids,
            class_names=names,
            outer_folds=outer,
            inner_folds=inner,
            C_grid=(0.1,),
            selection_metric="log_loss",
        )
    index_path = tmp_path / "representation-index.json"
    shutil.copytree(output, tmp_path / ".A-run.tmp-orphan")
    rows = write_representation_run_index(index_path, runs_root=tmp_path)
    assert len(rows) == 1
    assert validate_representation_run_index(index_path, runs_root=tmp_path) == rows


def test_AH_representation_serializes_selected_geometry(
    roberta_artifact,
    tmp_path: Path,
) -> None:
    encoder, _, _, _ = roberta_artifact
    item_ids, y, appraisal, names = _toy_data()
    texts = [f"{label} conditional example {index}" for index, label in enumerate(y)]
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
    output = tmp_path / "AH-run"
    artifact = run_crowd_representation_probe(
        output,
        representation="AH",
        appraisals=appraisal,
        appraisal_names=("a", "b", "c"),
        y=y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        embedding_directory=embedding,
        layer=0,
        pooling="mean",
        C_grid=(0.1,),
        block_multiplier_grid=(0.3, 1.0),
        selection_metric="log_loss",
    )

    assert artifact.metadata["representation"] == "AH"
    assert artifact.metadata["block_dims"] == [3, 768]
    assert set(artifact.selections["block_multiplier"]).issubset({0.3, 1.0})
    with np.load(output / "geometry.npz", allow_pickle=False) as geometry:
        assert geometry["coef"].shape == (3, 3, 771)
        assert geometry["block_multipliers"].shape == (3, 2)
        assert np.allclose(geometry["block_multipliers"][:, 1], 1.0)

    A_output = tmp_path / "A-companion"
    H_output = tmp_path / "H-companion"
    run_crowd_representation_probe(
        A_output,
        representation="A",
        appraisals=appraisal,
        appraisal_names=("a", "b", "c"),
        y=y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        C_grid=(0.1,),
        selection_metric="log_loss",
    )
    run_crowd_representation_probe(
        H_output,
        representation="H",
        appraisals=appraisal,
        appraisal_names=("a", "b", "c"),
        y=y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        embedding_directory=embedding,
        layer=0,
        pooling="mean",
        C_grid=(0.1,),
        selection_metric="log_loss",
    )
    conditional = write_conditional_analysis(
        tmp_path / "conditional-analysis",
        A_run=A_output,
        H_run=H_output,
        AH_run=output,
        labels=names,
        n_bootstrap=20,
    )
    assert conditional.bootstrap["n_bootstrap"] == 20
    assert validate_conditional_analysis(conditional.directory).metadata == conditional.metadata
    bad_conditional = tmp_path / "conditional-bad-delta"
    shutil.copytree(conditional.directory, bad_conditional)
    bootstrap_path = bad_conditional / "bootstrap_H_minus_AH.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["observed_delta"] += 1.0
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_file_record(bad_conditional, "bootstrap_H_minus_AH.json")
    with pytest.raises(ValueError, match="arithmetically inconsistent"):
        validate_conditional_analysis(bad_conditional)

    H_AH_fields = {
        "embedding_artifact_format": "different-format",
        "embedding_model_key": "deberta-v3-base",
        "embedding_revision": "different-revision",
        "embedding_mode": "random",
        "embedding_text_variant": "original",
        "embedding_metadata_sha256": "a" * 64,
        "embedding_item_text_pairs_sha256": "b" * 64,
        "embedding_layer_sha256": "c" * 64,
        "layer": 1,
        "pooling": "first",
    }
    for position, (field, value) in enumerate(H_AH_fields.items()):
        mismatched = tmp_path / f"AH-mismatch-hidden-{position}"
        shutil.copytree(output, mismatched)
        metadata_path = mismatched / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[field] = value
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=f"H/AH runs disagree on {field}"):
            write_conditional_analysis(
                tmp_path / f"rejected-hidden-{position}",
                A_run=A_output,
                H_run=H_output,
                AH_run=mismatched,
                labels=names,
                n_bootstrap=5,
            )

    A_AH_fields = {
        "appraisal_matrix_sha256": "d" * 64,
        "appraisal_names": ["changed-a", "b", "c"],
    }
    for position, (field, value) in enumerate(A_AH_fields.items()):
        mismatched = tmp_path / f"AH-mismatch-appraisal-{position}"
        shutil.copytree(output, mismatched)
        metadata_path = mismatched / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[field] = value
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=f"A/AH runs disagree on {field}"):
            write_conditional_analysis(
                tmp_path / f"rejected-appraisal-{position}",
                A_run=A_output,
                H_run=H_output,
                AH_run=mismatched,
                labels=names,
                n_bootstrap=5,
            )

    for position, (field, value) in enumerate(
        (("C_grid", [0.1, 1.0]), ("class_weight", "balanced"))
    ):
        mismatched = tmp_path / f"AH-mismatch-common-{position}"
        shutil.copytree(output, mismatched)
        metadata_path = mismatched / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[field] = value
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=f"run AH disagrees on {field}"):
            write_conditional_analysis(
                tmp_path / f"rejected-common-{position}",
                A_run=A_output,
                H_run=H_output,
                AH_run=mismatched,
                labels=names,
                n_bootstrap=5,
            )

    A_fold_mismatch = tmp_path / "A-fold-mismatch"
    shutil.copytree(A_output, A_fold_mismatch)
    fold_oof = pd.read_parquet(A_fold_mismatch / "oof.parquet")
    first = fold_oof.index[fold_oof["outer_fold"] == 0][0]
    second = fold_oof.index[fold_oof["outer_fold"] == 1][0]
    fold_oof.loc[[first, second], "outer_fold"] = fold_oof.loc[
        [second, first], "outer_fold"
    ].to_numpy()
    fold_oof.to_parquet(A_fold_mismatch / "oof.parquet", index=False)
    _rewrite_file_record(A_fold_mismatch, "oof.parquet")
    with pytest.raises(ValueError, match="OOF tables disagree on outer_fold"):
        write_conditional_analysis(
            tmp_path / "rejected-fold-mismatch",
            A_run=A_fold_mismatch,
            H_run=H_output,
            AH_run=output,
            labels=names,
            n_bootstrap=5,
        )


def test_representation_probe_rejects_inner_group_mismatch(tmp_path: Path) -> None:
    item_ids, y, appraisal, names = _toy_data()
    outer, inner = _nested_tables(item_ids)
    inner.loc[0, "group_id"] = "wrong-group"
    with pytest.raises(ValueError, match="group IDs disagree"):
        run_crowd_representation_probe(
            tmp_path / "bad",
            representation="A",
            appraisals=appraisal,
            appraisal_names=("a", "b", "c"),
            y=y,
            item_ids=item_ids,
            class_names=names,
            outer_folds=outer,
            inner_folds=inner,
            C_grid=(0.1,),
            selection_metric="log_loss",
        )


def test_representation_probe_detects_geometry_corruption(tmp_path: Path) -> None:
    item_ids, y, appraisal, names = _toy_data()
    outer, inner = _nested_tables(item_ids)
    output = tmp_path / "corrupt"
    run_crowd_representation_probe(
        output,
        representation="A",
        appraisals=appraisal,
        appraisal_names=("a", "b", "c"),
        y=y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        C_grid=(0.1,),
        selection_metric="log_loss",
    )
    path = output / "geometry.npz"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        final = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final[0] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_crowd_representation_probe(output)


def test_representation_validator_rejects_semantic_tampering(tmp_path: Path) -> None:
    item_ids, y, appraisal, names = _toy_data()
    outer, inner = _nested_tables(item_ids)
    output = tmp_path / "semantic-source"
    run_crowd_representation_probe(
        output,
        representation="A",
        appraisals=appraisal,
        appraisal_names=("a", "b", "c"),
        y=y,
        item_ids=item_ids,
        class_names=names,
        outer_folds=outer,
        inner_folds=inner,
        C_grid=(0.1,),
        selection_metric="log_loss",
    )

    bad_grid = tmp_path / "bad-grid"
    shutil.copytree(output, bad_grid)
    metadata_path = bad_grid / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["C_grid"] = [1.0]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="selected C is outside"):
        validate_crowd_representation_probe(bad_grid)

    bad_counts = tmp_path / "bad-counts"
    shutil.copytree(output, bad_counts)
    selections = pd.read_parquet(bad_counts / "selections.parquet")
    selections.loc[0, "n_train"] += 1
    selections.to_parquet(bad_counts / "selections.parquet", index=False)
    _rewrite_file_record(bad_counts, "selections.parquet")
    with pytest.raises(ValueError, match="train/test counts disagree"):
        validate_crowd_representation_probe(bad_counts)

    bad_gauge = tmp_path / "bad-gauge"
    shutil.copytree(output, bad_gauge)
    geometry_path = bad_gauge / "geometry.npz"
    with np.load(geometry_path, allow_pickle=False) as archive:
        geometry = {name: np.asarray(archive[name]).copy() for name in archive.files}
    geometry["coef"][0, 0, 0] += 0.1
    geometry["sites"] = geometry["coef"] / 2.0
    geometry["power_weights"] = geometry["intercept"] + np.sum(
        geometry["coef"] ** 2, axis=2
    ) / 4.0
    np.savez_compressed(geometry_path, **geometry)
    _rewrite_file_record(bad_gauge, "geometry.npz")
    with pytest.raises(ValueError, match="sum-zero gauge"):
        validate_crowd_representation_probe(bad_gauge)
