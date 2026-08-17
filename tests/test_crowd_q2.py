"""Focused synthetic tests for the crowd-enVENT Q2 suite runner.

All tests use small synthetic data to demonstrate:
- Paired A/H/AH artifact alignment for conditional H-minus-AH contrasts
- PCA and every other transform fitted on outer-train / sealed-train only
- Writer-appraisal coordinates and reader targets stay explicit, never conflated
- Sealed external-test behavior (never OOF, excluded writers preserved)
- Resumable skip of valid children and refusal of corruption/overwrite

No formatters, linters, or corpus runs are executed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from frozen_emotion_spaces.config import embedding_artifact_directory, get_model_spec
from frozen_emotion_spaces.crowd_q2 import (
    EXTERNAL_FORMAT,
    PCA_DIMENSIONS,
    SUITE_FORMAT,
    Q2SuiteArtifact,
    build_q2_suite,
    external_role_partition,
    run_q2_batch,
    run_q2_external_probe,
    run_q2_reader_probe,
    run_q2_representation_triplet,
    summarize_q2_suite,
    validate_q2_external_probe,
    validate_q2_suite,
)
from frozen_emotion_spaces.embeddings import (
    ARTIFACT_FORMAT,
    validate_embedding_artifact,
)
from frozen_emotion_spaces.metrics import multiclass_itemwise_log_loss_bits

# Small grids keep the synthetic fits fast; selection semantics are unchanged.
TINY_C_GRID = (0.1, 1.0)
TINY_MULTIPLIER_GRID = (1.0,)


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _synthetic_crowd_problem(
    n_items: int = 60,
    n_classes: int = 4,
    n_appraisals: int = 21,
    n_folds: int = 3,
    n_inner: int = 2,
    groups_per_fold: int = 5,
    seed: int = 71,
) -> dict:
    """Return a synthetic crowd problem with group-disjoint nested splits.

    Groups never straddle outer folds, inner validation folds are unions of
    whole groups, and every group contains every class so that each inner or
    outer training partition covers the full class axis.
    """

    rng = np.random.default_rng(seed)
    class_names = tuple(f"emo-{i}" for i in range(n_classes))
    appraisal_names = tuple(f"appr-{i}" for i in range(n_appraisals))
    per_fold = n_items // n_folds

    item_ids = [str(i) for i in range(n_items)]
    fold_of = [i % n_folds for i in range(n_items)]
    local_k = [i // n_folds for i in range(n_items)]
    group_of = [
        f"grp-f{fold_of[i]}-g{local_k[i] % groups_per_fold}" for i in range(n_items)
    ]
    # Within a group, local position p cycles 0..per_group-1, so (g + p)
    # covers every class exactly once per group.
    targets = [
        class_names[(local_k[i] % groups_per_fold + local_k[i] // groups_per_fold) % n_classes]
        for i in range(n_items)
    ]
    appraisals = rng.standard_normal((n_items, n_appraisals)).astype(np.float64)

    outer_folds = pd.DataFrame({
        "item_id": item_ids,
        "group_id": group_of,
        "test_fold": fold_of,
    })
    inner_rows = []
    for fold in range(n_folds):
        for i in range(n_items):
            if fold_of[i] == fold:
                continue
            inner_rows.append({
                "outer_fold": fold,
                "item_id": item_ids[i],
                "group_id": group_of[i],
                "validation_fold": (local_k[i] % groups_per_fold) % n_inner,
            })
    inner_folds = pd.DataFrame(inner_rows)
    return {
        "item_ids": item_ids,
        "groups": group_of,
        "targets": targets,
        "class_names": class_names,
        "appraisal_names": appraisal_names,
        "appraisals": appraisals,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
    }


def _external_problem(
    n_train: int = 40,
    n_test: int = 20,
    n_classes: int = 4,
    n_appraisals: int = 21,
    items_per_group: int = 4,
    n_inner: int = 3,
    seed: int = 42,
) -> dict:
    """Return a sealed external-style problem with train/test/excluded roles.

    Inner validation folds are unions of whole groups; every group contains
    every class.  Excluded items share writers with the test partition and
    must be preserved untouched.
    """

    rng = np.random.default_rng(seed)
    class_names = tuple(f"emo-{i}" for i in range(n_classes))
    appraisal_names = tuple(f"appr-{i}" for i in range(n_appraisals))

    train_ids = [f"tr-{i}" for i in range(n_train)]
    test_ids = [f"te-{i}" for i in range(n_test)]
    excluded_ids = [f"ex-{i}" for i in range(6)]
    train_targets = [class_names[i % n_classes] for i in range(n_train)]
    test_targets = [class_names[i % n_classes] for i in range(n_test)]
    excluded_targets = [class_names[i % n_classes] for i in range(len(excluded_ids))]
    train_appraisals = rng.standard_normal((n_train, n_appraisals))
    test_appraisals = rng.standard_normal((n_test, n_appraisals))

    train_groups = [f"grp-{i // items_per_group}" for i in range(n_train)]
    inner_folds = pd.DataFrame({
        "item_id": train_ids,
        "group_id": train_groups,
        "writer_id": [f"wr-{i % 5}" for i in range(n_train)],
        "validation_fold": [
            (i // items_per_group) % n_inner for i in range(n_train)
        ],
    })
    external = pd.DataFrame({
        "item_id": train_ids + test_ids + excluded_ids,
        "group_id": (
            train_groups
            + [f"tg-{i // items_per_group}" for i in range(n_test)]
            + [f"tg-{i // 3}" for i in range(len(excluded_ids))]
        ),
        "writer_id": (
            [f"wr-{i % 5}" for i in range(n_train)]
            + [f"wt-{i % 3}" for i in range(n_test)]
            + [f"wt-{i % 3}" for i in range(len(excluded_ids))]
        ),
        "role": (
            ["train"] * n_train
            + ["test"] * n_test
            + ["excluded_test_writer"] * len(excluded_ids)
        ),
    })
    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "excluded_ids": excluded_ids,
        "train_targets": train_targets,
        "test_targets": test_targets,
        "excluded_targets": excluded_targets,
        "train_appraisals": train_appraisals,
        "test_appraisals": test_appraisals,
        "class_names": class_names,
        "appraisal_names": appraisal_names,
        "inner_folds": inner_folds,
        "external": external,
    }


def _synthetic_embedding_artifact(
    directory: Path,
    item_ids: list[str],
    *,
    model_key: str = "roberta-base",
    seed: int = 123,
    mean: np.ndarray | None = None,
    first: np.ndarray | None = None,
) -> Path:
    """Write a validated synthetic embedding artifact (no encoder weights)."""

    spec = get_model_spec(model_key)
    rng = np.random.default_rng(seed)
    n_layers = spec.emitted_layers
    ids = np.asarray([str(i) for i in item_ids], dtype=np.str_)
    if mean is None:
        mean = rng.standard_normal((n_layers, len(ids), spec.hidden_size)).astype(np.float32)
    if first is None:
        first = rng.standard_normal((n_layers, len(ids), spec.hidden_size)).astype(np.float32)
    expected_shape = (n_layers, len(ids), spec.hidden_size)
    if mean.shape != expected_shape or first.shape != expected_shape:
        raise ValueError("provided arrays must have shape " f"{expected_shape}")
    mean = np.ascontiguousarray(mean, dtype=np.float32)
    first = np.ascontiguousarray(first, dtype=np.float32)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "mean.npy", mean)
    np.save(directory / "first.npy", first)
    np.save(directory / "item_ids.npy", ids)

    def file_record(name: str, array: np.ndarray) -> dict:
        record = {
            "sha256": _sha256_file(directory / name),
            "bytes": (directory / name).stat().st_size,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if name in {"mean.npy", "first.npy"}:
            record["layer_sha256"] = [
                _sha256_array(array[layer]) for layer in range(array.shape[0])
            ]
        return record

    texts = [f"synthetic text for {item}" for item in ids]
    metadata = {
        "artifact_format": ARTIFACT_FORMAT,
        "dataset": "crowd",
        "model_key": spec.key,
        "repository": spec.repository,
        "revision": spec.revision,
        "mode": "pretrained",
        "text_variant": "masked",
        "n_layers": n_layers,
        "n_items": len(ids),
        "hidden_size": spec.hidden_size,
        "max_length": 256,
        "batch_size": 4,
        "seed": 0,
        "truncated_items": 0,
        "truncation_rate": 0.0,
        "maximum_tokenized_length": 64,
        "tokenizer_padding_side": "right",
        "tokenizer_truncation_side": "right",
        "storage_dtype": "float32",
        "model_content_sha256": hashlib.sha256(b"synthetic-model").hexdigest(),
        "tokenizer_fingerprint_sha256": hashlib.sha256(
            b"synthetic-tokenizer"
        ).hexdigest(),
        "ordered_texts_sha256": hashlib.sha256(
            "\n".join(texts).encode("utf-8")
        ).hexdigest(),
        "ordered_item_text_pairs_sha256": _pair_digest(ids.tolist(), texts),
        "files": {
            "mean.npy": file_record("mean.npy", mean),
            "first.npy": file_record("first.npy", first),
            "item_ids.npy": file_record("item_ids.npy", ids),
        },
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_embedding_artifact(directory, expected_item_ids=item_ids)
    return directory


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_digest(left, right) -> str:
    digest = hashlib.sha256()
    for first, second in zip(left, right, strict=True):
        for value in (str(first), str(second)):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _triplet_kwargs(problem: dict) -> dict:
    return dict(
        appraisals=problem["appraisals"],
        y=problem["targets"],
        item_ids=problem["item_ids"],
        outer_folds=problem["outer_folds"],
        inner_folds=problem["inner_folds"],
        class_names=problem["class_names"],
        appraisal_names=problem["appraisal_names"],
        C_grid=TINY_C_GRID,
        block_multiplier_grid=TINY_MULTIPLIER_GRID,
        selection_metric="log_loss",
    )


def _run_A(tmp: Path, problem: dict, **overrides) -> "object":
    kwargs = dict(
        representation="A",
        target_scale="full_writer",
        rater_role="writer_appraisal_to_writer_target",
        **_triplet_kwargs(problem),
    )
    kwargs.update(overrides)
    return run_q2_representation_triplet(tmp / "A-run", **kwargs)


def _run_H(tmp: Path, problem: dict, embedding: Path, layer: int = 5, **overrides):
    kwargs = dict(
        representation="H",
        target_scale="full_writer",
        rater_role="writer_appraisal_to_writer_target",
        embedding_directory=embedding,
        layer=layer,
        **_triplet_kwargs(problem),
    )
    kwargs.update(overrides)
    return run_q2_representation_triplet(tmp / "H-run", **kwargs)


def _run_AH(tmp: Path, problem: dict, embedding: Path, layer: int = 5, **overrides):
    kwargs = dict(
        representation="AH",
        target_scale="full_writer",
        rater_role="writer_appraisal_to_writer_target",
        embedding_directory=embedding,
        layer=layer,
        **_triplet_kwargs(problem),
    )
    kwargs.update(overrides)
    return run_q2_representation_triplet(tmp / "AH-run", **kwargs)


def _run_external(tmp: Path, problem: dict, **overrides):
    kwargs = dict(
        representation="A",
        appraisals=problem["train_appraisals"],
        y_train=problem["train_targets"],
        item_ids_train=problem["train_ids"],
        appraisals_test=problem["test_appraisals"],
        y_test=problem["test_targets"],
        item_ids_test=problem["test_ids"],
        inner_folds=problem["inner_folds"],
        class_names=problem["class_names"],
        appraisal_names=problem["appraisal_names"],
        C_grid=TINY_C_GRID,
        selection_metric="log_loss",
    )
    kwargs.update(overrides)
    return run_q2_external_probe(tmp / "ext", **kwargs)


def _label_encoding_arrays(
    ids: list[str],
    targets_by_id: dict[str, str],
    class_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Layer-0 rows encode the item's class as a near one-hot direction on an
    otherwise constant (zero-variance) feature space, so per-column
    standardization keeps only the four signal columns active.  Any probe
    that selects artifact rows by item id reaches a near-perfect fit;
    positional (misaligned) selection is punished by the shuffled extras.
    """

    spec = get_model_spec("roberta-base")
    shape = (spec.emitted_layers, len(ids), spec.hidden_size)
    mean = np.zeros(shape, dtype=np.float32)
    first = np.zeros(shape, dtype=np.float32)
    class_index = {name: index for index, name in enumerate(class_names)}
    for row, item in enumerate(ids):
        target = targets_by_id.get(item)
        if target is not None:
            mean[0, row, class_index[target]] = 25.0
            first[0, row, class_index[target]] = 25.0
    return mean, first


def _assert_true_class_probs(frame: pd.DataFrame, targets_by_id: dict[str, str]) -> None:
    records = frame.to_dict("records")
    assert records
    for item, row in zip(frame["item_id"].astype(str), records, strict=True):
        assert row[f"prob__{targets_by_id[item]}"] > 0.9


def test_external_probe_aligns_superset_artifact_by_item_id(tmp_path: Path) -> None:
    """A full-corpus artifact (extra items, shuffled order) is selected by id."""
    problem = _external_problem()
    wanted = problem["train_ids"] + problem["test_ids"]
    targets_by_id = dict(
        zip(wanted, problem["train_targets"] + problem["test_targets"], strict=True)
    )
    ids = wanted[:4] + [f"extra-{i}" for i in range(4)] + wanted[4:]
    mean, first = _label_encoding_arrays(ids, targets_by_id, problem["class_names"])
    embedding = _synthetic_embedding_artifact(
        tmp_path / "emb-superset", ids, mean=mean, first=first
    )
    result = _run_external(
        tmp_path,
        problem,
        representation="H",
        embedding_directory=embedding,
        layer=0,
        pooling="mean",
    )
    _assert_true_class_probs(result.test_predictions, targets_by_id)
    validate_q2_external_probe(result.directory)


def test_triplet_aligns_subset_artifact_by_item_id(tmp_path: Path) -> None:
    """A superset artifact serves a probe over any item subset, by id."""
    problem = _synthetic_crowd_problem()
    wanted = problem["item_ids"]
    targets_by_id = dict(zip(wanted, problem["targets"], strict=True))
    ids = wanted[:2] + [f"extra-{i}" for i in range(3)] + wanted[2:]
    mean, first = _label_encoding_arrays(ids, targets_by_id, problem["class_names"])
    embedding = _synthetic_embedding_artifact(
        tmp_path / "emb-superset", ids, mean=mean, first=first
    )
    artifact = _run_H(tmp_path, problem, embedding, layer=0)
    _assert_true_class_probs(artifact.oof, targets_by_id)


# ---------------------------------------------------------------------------
# Paired alignment
# ---------------------------------------------------------------------------

def test_paired_A_H_AH_runs_share_item_ordering(tmp_path: Path) -> None:
    """A, H, and AH OOF tables align row-by-row after item_id sorting, which
    is what paired conditional H-minus-AH contrasts require."""
    problem = _synthetic_crowd_problem()
    embedding = _synthetic_embedding_artifact(
        tmp_path / "emb", problem["item_ids"]
    )
    a = _run_A(tmp_path / "a", problem)
    h = _run_H(tmp_path / "h", problem, embedding)
    ah = _run_AH(tmp_path / "ah", problem, embedding)

    frames = [
        artifact.oof.sort_values("item_id", kind="stable").reset_index(drop=True)
        for artifact in (a, h, ah)
    ]
    for other in frames[1:]:
        assert frames[0]["item_id"].equals(other["item_id"])
        assert frames[0]["outer_fold"].equals(other["outer_fold"])
        assert frames[0]["group_id"].equals(other["group_id"])
        assert frames[0]["y_true"].equals(other["y_true"])


# ---------------------------------------------------------------------------
# Train-only PCA / transforms
# ---------------------------------------------------------------------------

def test_pca_fitted_inside_each_outer_fold(tmp_path: Path) -> None:
    """The stored feature digest must equal the foldwise leak-free reduction:
    every test row transformed by the PCA fitted on its own fold's train."""
    problem = _synthetic_crowd_problem(n_items=60, n_appraisals=21, n_folds=3)
    d = 5
    artifact = _run_A(tmp_path, problem, pca_dimension=d)

    fold_by_id = dict(
        zip(problem["outer_folds"]["item_id"], problem["outer_folds"]["test_fold"])
    )
    folds = np.asarray([fold_by_id[i] for i in problem["item_ids"]])
    expected = np.empty((len(problem["item_ids"]), d), dtype=np.float64)
    for fold in pd.unique(folds):
        mask = folds == fold
        pca = PCA(n_components=d, random_state=0)
        pca.fit(problem["appraisals"][~mask])
        expected[mask] = pca.transform(problem["appraisals"][mask])
    assert artifact.metadata["feature_matrix_sha256"] == _sha256_array(expected)

    assert artifact.metadata["n_features"] == d
    assert artifact.metadata["pca_dimension"] == d
    assert (
        artifact.metadata["feature_matrix_sha256"]
        != artifact.metadata["appraisal_matrix_sha256"]
    )
    assert len(artifact.oof) == len(problem["item_ids"])
    prob_cols = [c for c in artifact.oof.columns if c.startswith("prob__")]
    np.testing.assert_allclose(artifact.oof[prob_cols].sum(axis=1), 1.0, atol=1e-6)


def test_pca_block_dims_record_reduced_width(tmp_path: Path) -> None:
    """PCA runs declare block_dims (d,), never the raw appraisal width."""
    problem = _synthetic_crowd_problem()
    artifact = _run_A(tmp_path, problem, pca_dimension=3)
    assert tuple(artifact.metadata["block_dims"]) == (3,)


def test_external_pca_fitted_on_sealed_train_only(tmp_path: Path) -> None:
    """External PCA features must equal a PCA fitted on the train partition
    alone; fitting on train+test would change the stored digest."""
    problem = _external_problem(n_classes=4)
    d = 5
    result = _run_external(tmp_path, problem, pca_dimension=d)
    pca = PCA(n_components=d, random_state=0)
    expected_train = pca.fit_transform(problem["train_appraisals"])
    assert result.metadata["feature_matrix_sha256"] == _sha256_array(expected_train)
    assert result.metadata["pca_dimension"] == d
    assert result.metadata["n_features"] == d
    assert result.metadata["block_dims"] == [d]


# ---------------------------------------------------------------------------
# Writer coordinates vs reader targets (cross-rater, never conflated)
# ---------------------------------------------------------------------------

def test_reader_probe_separates_writer_coordinates_from_reader_targets(
    tmp_path: Path,
) -> None:
    """The reader run must record writer-side coordinates, reader-side
    targets, the reader split tables, and reader-majority target identity."""
    problem = _synthetic_crowd_problem()
    names = problem["class_names"]
    index = {name: i for i, name in enumerate(names)}
    reader_targets = [names[(index[t] + 1) % len(names)] for t in problem["targets"]]

    artifact = run_q2_reader_probe(
        tmp_path / "reader-A",
        representation="A",
        writer_appraisals=problem["appraisals"],
        reader_targets=reader_targets,
        item_ids=problem["item_ids"],
        reader_outer_folds=problem["outer_folds"],
        reader_inner_folds=problem["inner_folds"],
        class_names=names,
        appraisal_names=problem["appraisal_names"],
        C_grid=TINY_C_GRID,
        block_multiplier_grid=TINY_MULTIPLIER_GRID,
        selection_metric="log_loss",
    )
    meta = artifact.metadata
    assert meta["target"] == "y_reader_majority"
    assert meta["target_scale"] == "reader"
    assert meta["coordinate_rater"] == "writer"
    assert meta["target_rater"] == "reader"
    assert meta["rater_role"] == "writer_appraisal_to_reader_target"
    assert meta["split_outer_table"] == "crowd_reader_outer.csv"
    assert meta["split_inner_table"] == "crowd_reader_inner.csv"
    # Targets actually used are the reader targets, in item order.
    aligned = artifact.oof.sort_values("item_id", kind="stable").reset_index(drop=True)
    assert aligned["y_true"].tolist() == [
        reader_targets[int(i)] for i in aligned["item_id"]
    ]
    assert meta["ordered_item_target_sha256"] == _pair_digest(
        problem["item_ids"], reader_targets
    )


def test_writer_and_reader_runs_are_distinct_in_metadata(tmp_path: Path) -> None:
    """Same coordinates, different target raters → distinct metadata identity."""
    problem = _synthetic_crowd_problem()
    names = problem["class_names"]
    index = {name: i for i, name in enumerate(names)}
    reader_targets = [names[(index[t] + 1) % len(names)] for t in problem["targets"]]

    writer = _run_A(tmp_path / "writer", problem)
    reader = run_q2_reader_probe(
        tmp_path / "reader",
        representation="A",
        writer_appraisals=problem["appraisals"],
        reader_targets=reader_targets,
        item_ids=problem["item_ids"],
        reader_outer_folds=problem["outer_folds"],
        reader_inner_folds=problem["inner_folds"],
        class_names=names,
        appraisal_names=problem["appraisal_names"],
        C_grid=TINY_C_GRID,
        block_multiplier_grid=TINY_MULTIPLIER_GRID,
        selection_metric="log_loss",
    )
    assert writer.metadata["target_scale"] == "full_writer"
    assert writer.metadata["target"] == "y_writer"
    assert writer.metadata["coordinate_rater"] == "writer"
    assert writer.metadata["target_rater"] == "writer"
    assert writer.metadata["split_outer_table"] == "crowd_full_outer.csv"
    assert reader.metadata["target_scale"] != writer.metadata["target_scale"]
    assert reader.metadata["target"] != writer.metadata["target"]
    assert reader.metadata["target_rater"] != writer.metadata["target_rater"]
    assert (
        reader.metadata["ordered_item_target_sha256"]
        != writer.metadata["ordered_item_target_sha256"]
    )


# ---------------------------------------------------------------------------
# Refusal: overwrite, incompatibility, corruption
# ---------------------------------------------------------------------------

def test_refuses_to_overwrite_with_different_data(tmp_path: Path) -> None:
    """Re-running into a completed directory with different targets refuses."""
    problem = _synthetic_crowd_problem()
    _run_A(tmp_path, problem)
    shifted = problem["targets"][1:] + problem["targets"][:1]
    with pytest.raises(FileExistsError, match="ordered_item_target_sha256"):
        _run_A(tmp_path, problem, y=shifted)


def test_refuses_incompatible_target_scale(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    _run_A(tmp_path, problem)
    with pytest.raises(FileExistsError, match="target_scale"):
        _run_A(tmp_path, problem, target_scale="reader", rater_role="reader_target")


def test_refuses_incompatible_rater_role(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    _run_A(tmp_path, problem)
    with pytest.raises(FileExistsError, match="rater_role"):
        _run_A(tmp_path, problem, rater_role="different_role")


def test_refuses_incompatible_pca_dimension(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    _run_A(tmp_path, problem, pca_dimension=5)
    with pytest.raises(FileExistsError, match="pca_dimension"):
        _run_A(tmp_path, problem, pca_dimension=3)


def test_refuses_incompatible_representation(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    embedding = _synthetic_embedding_artifact(
        tmp_path / "emb", problem["item_ids"]
    )
    _run_A(tmp_path, problem)
    with pytest.raises(FileExistsError, match="representation"):
        run_q2_representation_triplet(
            tmp_path / "A-run",
            representation="H",
            embedding_directory=embedding,
            layer=5,
            target_scale="full_writer",
            rater_role="writer_appraisal_to_writer_target",
            **_triplet_kwargs(problem),
        )


def test_corrupted_artifact_refused_on_resume(tmp_path: Path) -> None:
    """A tampered child is never resumed and never silently overwritten."""
    problem = _synthetic_crowd_problem()
    artifact = _run_A(tmp_path, problem)
    oof_path = artifact.directory / "oof.parquet"
    oof_path.write_bytes(oof_path.read_bytes() + b"tampered")
    with pytest.raises(FileExistsError, match="incompatible or corrupt"):
        _run_A(tmp_path, problem)


def test_resumable_skip_valid_artifact(tmp_path: Path) -> None:
    """An identical request reuses the validated artifact without re-fitting."""
    problem = _synthetic_crowd_problem()
    first = _run_A(tmp_path, problem)
    second = _run_A(tmp_path, problem)
    assert first.directory == second.directory
    assert first.oof.equals(second.oof)
    assert first.metadata["ordered_item_target_sha256"] == (
        second.metadata["ordered_item_target_sha256"]
    )


def test_metadata_labels_results_as_reconstructions(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    artifact = _run_A(tmp_path, problem)
    assert artifact.metadata["status"] == "new_replication_not_historical_recovery"


def test_pca_dimensions_match_paper_spec() -> None:
    assert PCA_DIMENSIONS == (3, 5, 7, 10, 21)


def test_invalid_target_scale_rejected(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    with pytest.raises(ValueError, match="target_scale"):
        _run_A(tmp_path, problem, target_scale="invalid_scale")


def test_triplet_rejects_external_writer_scale(tmp_path: Path) -> None:
    """The sealed external configuration must never run as an OOF triplet."""
    problem = _synthetic_crowd_problem()
    with pytest.raises(ValueError, match="external"):
        _run_A(tmp_path, problem, target_scale="external_writer")


def test_pca_rejected_for_hidden_representations(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    with pytest.raises(ValueError, match="pca_dimension"):
        _run_A(tmp_path, problem, representation="H", pca_dimension=5)


# ---------------------------------------------------------------------------
# Sealed external test
# ---------------------------------------------------------------------------

def test_external_probe_sealed_train_test(tmp_path: Path) -> None:
    """The external probe fits on train, predicts test only, and is not OOF."""
    problem = _external_problem()
    result = _run_external(
        tmp_path, problem, excluded_item_ids=problem["excluded_ids"]
    )
    assert result.metadata["run_format"] == EXTERNAL_FORMAT
    assert result.metadata["status"] == "new_replication_not_historical_recovery"
    assert result.metadata["target_scale"] == "external_writer"
    assert result.metadata["n_train"] == len(problem["train_ids"])
    assert result.metadata["n_test"] == len(problem["test_ids"])
    assert result.metadata["n_excluded_items"] == len(problem["excluded_ids"])
    assert len(result.test_predictions) == len(problem["test_ids"])
    assert set(result.test_predictions["item_id"]) == set(problem["test_ids"])
    # Never an OOF fold: no oof.parquet, and train ids never predicted.
    assert not (result.directory / "oof.parquet").exists()
    assert not set(result.test_predictions["item_id"]) & set(problem["train_ids"])
    # Real digests, not placeholders.
    assert len(result.metadata["inner_split_sha256"]) == 64
    # The written artifact validates cleanly.
    validate_q2_external_probe(result.directory)


def test_external_probe_rejects_overlapping_partitions(tmp_path: Path) -> None:
    problem = _external_problem()
    with pytest.raises(ValueError, match="disjoint"):
        _run_external(
            tmp_path / "overlap",
            problem,
            item_ids_test=problem["test_ids"][:-1] + [problem["train_ids"][0]],
            appraisals_test=problem["test_appraisals"],
            y_test=problem["test_targets"],
        )
    with pytest.raises(ValueError, match="excluded"):
        _run_external(
            tmp_path / "excluded",
            problem,
            excluded_item_ids=[problem["train_ids"][0]],
        )


def test_external_probe_rejects_inner_split_beyond_train(tmp_path: Path) -> None:
    """crowd_external_inner.csv covers train only; extra items are rejected."""
    problem = _external_problem()
    inner = pd.concat([
        problem["inner_folds"],
        pd.DataFrame([{
            "item_id": problem["test_ids"][0],
            "group_id": "grp-x",
            "writer_id": "wt-0",
            "validation_fold": 0,
        }]),
    ])
    with pytest.raises(ValueError, match="exactly the training items"):
        _run_external(tmp_path, problem, inner_folds=inner)


def test_external_probe_resumable_and_incompatible_refused(tmp_path: Path) -> None:
    problem = _external_problem()
    first = _run_external(tmp_path, problem)
    second = _run_external(tmp_path, problem)
    assert first.directory == second.directory
    assert first.test_predictions.equals(second.test_predictions)
    with pytest.raises(FileExistsError, match="n_test"):
        _run_external(
            tmp_path,
            problem,
            item_ids_test=problem["test_ids"][:-1],
            appraisals_test=problem["test_appraisals"][:-1],
            y_test=problem["test_targets"][:-1],
        )


def test_external_corruption_refused_on_resume(tmp_path: Path) -> None:
    problem = _external_problem()
    result = _run_external(tmp_path, problem)
    path = result.directory / "test_predictions.parquet"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(FileExistsError, match="incompatible or corrupt"):
        _run_external(tmp_path, problem)


def test_external_role_partition_preserves_excluded() -> None:
    problem = _external_problem()
    partition = external_role_partition(problem["external"])
    assert set(partition.train_ids) == set(problem["train_ids"])
    assert set(partition.test_ids) == set(problem["test_ids"])
    assert set(partition.excluded_test_writer_ids) == set(problem["excluded_ids"])
    assert partition.excluded_test_duplicate_ids == ()
    all_ids = (
        set(partition.train_ids)
        | set(partition.test_ids)
        | set(partition.excluded_test_writer_ids)
        | set(partition.excluded_test_duplicate_ids)
    )
    assert len(all_ids) == (
        len(problem["train_ids"]) + len(problem["test_ids"]) + len(problem["excluded_ids"])
    )
    bad = problem["external"].copy()
    bad.loc[0, "role"] = "mystery"
    with pytest.raises(ValueError, match="unknown roles"):
        external_role_partition(bad)


# ---------------------------------------------------------------------------
# Suite builder, summary, paired contrasts
# ---------------------------------------------------------------------------

def _build_runs_root(tmp_path: Path) -> tuple[Path, dict, Path]:
    problem = _synthetic_crowd_problem()
    embedding = _synthetic_embedding_artifact(
        tmp_path / "emb", problem["item_ids"]
    )
    runs_root = tmp_path / "runs"
    _run_A(runs_root / "a", problem)
    _run_H(runs_root / "h", problem, embedding)
    _run_AH(runs_root / "ah", problem, embedding)
    return runs_root, problem, embedding


def test_suite_builder_collects_metrics_and_paired_contrasts(tmp_path: Path) -> None:
    """Summary carries log loss, macro-F1, Brier, ECE; contrasts are keyed
    paired H-minus-AH deltas on identically ordered OOF tables."""
    runs_root, problem, _ = _build_runs_root(tmp_path)
    suite = build_q2_suite(
        tmp_path / "suite",
        runs_root=runs_root,
        labels=problem["class_names"],
        n_bootstrap=500,
        seed=11,
    )
    assert isinstance(suite, Q2SuiteArtifact)
    assert suite.metadata["suite_format"] == SUITE_FORMAT
    assert suite.metadata["status"] == "new_replication_not_historical_recovery"
    assert suite.metadata["n_runs"] == 3
    assert len(suite.summary) == 3
    for column in ("log_loss_bits", "macro_f1", "brier", "ece"):
        assert column in suite.summary.columns
        assert suite.summary[column].notna().all()

    assert suite.metadata["n_contrasts"] == 1
    row = suite.contrasts.iloc[0]
    assert row["layer"] == 5
    assert row["model_key"] == "roberta-base"
    assert row["target_scale"] == "full_writer"
    assert row["n_items"] == len(problem["item_ids"])

    # Recompute the paired delta independently from the stored OOF tables.
    from frozen_emotion_spaces.experiment_c import (
        validate_crowd_representation_probe,
    )

    h_oof = validate_crowd_representation_probe(
        runs_root / "h" / "H-run"
    ).oof.sort_values("item_id", kind="stable")
    ah_oof = validate_crowd_representation_probe(
        runs_root / "ah" / "AH-run"
    ).oof.sort_values("item_id", kind="stable")
    expected = float(
        multiclass_itemwise_log_loss_bits(h_oof, labels=problem["class_names"]).mean()
        - multiclass_itemwise_log_loss_bits(
            ah_oof, labels=problem["class_names"]
        ).mean()
    )
    assert row["delta_H_minus_AH"] == pytest.approx(expected, abs=1e-12)
    assert row["H_log_loss_bits"] - row["AH_log_loss_bits"] == pytest.approx(
        row["delta_H_minus_AH"], abs=1e-12
    )
    assert row["ci_low"] <= row["delta_H_minus_AH"] <= row["ci_high"]
    assert row["standard_error"] > 0


def test_suite_summary_merges_contrast_columns(tmp_path: Path) -> None:
    runs_root, problem, _ = _build_runs_root(tmp_path)
    build_q2_suite(
        tmp_path / "suite",
        runs_root=runs_root,
        labels=problem["class_names"],
        n_bootstrap=100,
        seed=3,
    )
    table = summarize_q2_suite(tmp_path / "suite")
    assert "log_loss_bits" in table.columns
    assert "macro_f1" in table.columns
    assert "delta_H_minus_AH" in table.columns
    hidden = table.loc[table["representation"] == "H"].iloc[0]
    assert np.isfinite(hidden["delta_H_minus_AH"])


def test_suite_refuses_to_overwrite(tmp_path: Path) -> None:
    runs_root, problem, _ = _build_runs_root(tmp_path)
    build_q2_suite(
        tmp_path / "suite",
        runs_root=runs_root,
        labels=problem["class_names"],
        n_bootstrap=50,
        seed=1,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_q2_suite(
            tmp_path / "suite",
            runs_root=runs_root,
            labels=problem["class_names"],
            n_bootstrap=50,
            seed=1,
        )


def test_validate_rejects_incomplete_suite(tmp_path: Path) -> None:
    incomplete = tmp_path / "bad-suite"
    incomplete.mkdir()
    (incomplete / "suite_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing files"):
        validate_q2_suite(incomplete)


def test_validate_rejects_corrupted_suite(tmp_path: Path) -> None:
    runs_root, problem, _ = _build_runs_root(tmp_path)
    suite = build_q2_suite(
        tmp_path / "suite",
        runs_root=runs_root,
        labels=problem["class_names"],
        n_bootstrap=50,
        seed=1,
    )
    summary_path = suite.directory / "summary.parquet"
    summary_path.write_bytes(summary_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        validate_q2_suite(suite.directory)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def test_batch_runner_appraisal_only_with_pca_variants(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    output_root = tmp_path / "batch"
    results = run_q2_batch(
        output_root,
        model_keys=("roberta-base",),
        layers=(5,),
        pca_dimensions=(3, 5),
        require_embeddings=False,
        target_scale="full_writer",
        rater_role="writer_appraisal_to_writer_target",
        **_triplet_kwargs(problem),
    )
    assert len(results) == 3  # A, A-pca3, A-pca5
    assert {a.metadata["representation"] for a in results} == {"A"}
    pca_dims = {a.metadata["pca_dimension"] for a in results}
    assert pca_dims == {None, 3, 5}
    for artifact in results:
        assert artifact.metadata["status"] == "new_replication_not_historical_recovery"
    # Batch resume: a second identical call reuses the same children.
    again = run_q2_batch(
        output_root,
        model_keys=("roberta-base",),
        layers=(5,),
        pca_dimensions=(3, 5),
        require_embeddings=False,
        target_scale="full_writer",
        rater_role="writer_appraisal_to_writer_target",
        **_triplet_kwargs(problem),
    )
    assert [a.directory for a in again] == [a.directory for a in results]


def test_batch_runner_full_triplet_with_embeddings(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    embedding_root = tmp_path / "cache"
    artifact_dir = embedding_artifact_directory(
        embedding_root,
        dataset="crowd",
        model="roberta-base",
        mode="pretrained",
        text_variant="masked",
    )
    _synthetic_embedding_artifact(artifact_dir, problem["item_ids"])
    results = run_q2_batch(
        tmp_path / "batch",
        model_keys=("roberta-base",),
        layers=(5,),
        pca_dimensions=(),
        embedding_root=embedding_root,
        target_scale="full_writer",
        rater_role="writer_appraisal_to_writer_target",
        **_triplet_kwargs(problem),
    )
    assert {a.metadata["representation"] for a in results} == {"A", "H", "AH"}
    hidden = [a for a in results if a.metadata["representation"] != "A"]
    for artifact in hidden:
        assert artifact.metadata["embedding_model_key"] == "roberta-base"
        assert artifact.metadata["layer"] == 5
        assert artifact.metadata["pooling"] == "mean"


def test_batch_runner_rejects_invalid_configuration(tmp_path: Path) -> None:
    problem = _synthetic_crowd_problem()
    with pytest.raises(ValueError, match="unknown frozen encoder key"):
        run_q2_batch(
            tmp_path / "b1",
            model_keys=("nonexistent-model",),
            layers=(0,),
            require_embeddings=False,
            target_scale="full_writer",
            rater_role="writer_appraisal_to_writer_target",
            **_triplet_kwargs(problem),
        )
    with pytest.raises(ValueError, match="layer"):
        run_q2_batch(
            tmp_path / "b2",
            model_keys=("roberta-base",),
            layers=(99,),
            require_embeddings=False,
            target_scale="full_writer",
            rater_role="writer_appraisal_to_writer_target",
            **_triplet_kwargs(problem),
        )
    with pytest.raises(ValueError, match="embedding_root"):
        run_q2_batch(
            tmp_path / "b3",
            model_keys=("roberta-base",),
            layers=(5,),
            target_scale="full_writer",
            rater_role="writer_appraisal_to_writer_target",
            **_triplet_kwargs(problem),
        )
