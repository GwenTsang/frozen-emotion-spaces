from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from frozen_emotion_spaces.config import get_model_spec
from frozen_emotion_spaces.embeddings import ARTIFACT_FORMAT
from frozen_emotion_spaces.experiment_a import (
    _dataframe_digest,
    _ordered_pair_digest,
    run_crowd_layer_probe,
    validate_crowd_layer_probe,
)
from frozen_emotion_spaces.metrics import reconstruct_multiclass_metrics
from frozen_emotion_spaces.probes import DEFAULT_C_GRID


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layerwise = _load_script("run_crowd_layerwise")
plotscript = _load_script("plot_crowd_layerwise")


class SyntheticProblem(NamedTuple):
    names: tuple[str, ...]
    item_ids: np.ndarray
    y: np.ndarray
    outer: pd.DataFrame
    inner: pd.DataFrame


def _synthetic_problem() -> SyntheticProblem:
    names = ("zeta", "alpha", "middle")
    y = np.tile(np.asarray(names), 9)
    item_ids = np.asarray([f"item-{index:02d}" for index in range(27)])
    groups = np.asarray([f"group-{index:02d}" for index in range(27)])
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
    return SyntheticProblem(names, item_ids, y, outer, pd.DataFrame(rows))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_embedding_artifact(
    directory: Path,
    *,
    item_ids: np.ndarray,
    seed: int,
    model_key: str = "roberta-base",
) -> Path:
    """Write a self-consistent synthetic artifact that passes full validation."""

    spec = get_model_spec(model_key)
    rng = np.random.default_rng(seed)
    shape = (spec.emitted_layers, len(item_ids), spec.hidden_size)
    directory.mkdir(parents=True)
    np.save(
        directory / "mean.npy",
        rng.normal(size=shape).astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        directory / "first.npy",
        rng.normal(size=shape).astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        directory / "item_ids.npy",
        np.asarray([str(item_id) for item_id in item_ids]),
        allow_pickle=False,
    )
    files = {}
    for filename in ("mean.npy", "first.npy", "item_ids.npy"):
        path = directory / filename
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if filename != "item_ids.npy":
            record["layer_sha256"] = [
                hashlib.sha256(
                    memoryview(np.ascontiguousarray(array[layer])).cast("B")
                ).hexdigest()
                for layer in range(array.shape[0])
            ]
        files[filename] = record
    metadata = {
        "artifact_format": ARTIFACT_FORMAT,
        "dataset": "crowd",
        "model_key": spec.key,
        "repository": spec.repository,
        "revision": spec.revision,
        "mode": "pretrained",
        "text_variant": "masked",
        "max_length": 32,
        "batch_size": 8,
        "seed": 0,
        "storage_dtype": "float32",
        "n_items": int(len(item_ids)),
        "n_layers": spec.emitted_layers,
        "hidden_size": spec.hidden_size,
        "truncated_items": 0,
        "truncation_rate": 0.0,
        "maximum_tokenized_length": 8,
        "model_content_sha256": "0" * 64,
        "tokenizer_fingerprint_sha256": "1" * 64,
        "ordered_texts_sha256": "2" * 64,
        "ordered_item_text_pairs_sha256": "3" * 64,
        "tokenizer_padding_side": "right",
        "tokenizer_truncation_side": "right",
        "files": files,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def _run_batch(output_root: Path, embedding: Path, problem: SyntheticProblem, **overrides):
    arguments = {
        "embedding_series": {"synthetic": embedding},
        "layers": (0,),
        "y": [str(value) for value in problem.y],
        "item_ids": [str(value) for value in problem.item_ids],
        "outer_folds": problem.outer,
        "inner_folds": problem.inner,
        "class_names": problem.names,
        "pooling": "mean",
        "C_grid": (0.01, 1.0),
        "selection_metric": "log_loss",
    }
    arguments.update(overrides)
    return layerwise.run_layerwise_batch(output_root, **arguments)


@pytest.fixture(scope="module")
def two_series_batch(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("layerwise")
    problem = _synthetic_problem()
    embeddings = {
        "alpha": _write_embedding_artifact(
            root / "emb-alpha", item_ids=problem.item_ids, seed=11
        ),
        "beta": _write_embedding_artifact(
            root / "emb-beta", item_ids=problem.item_ids, seed=29
        ),
    }
    batch = layerwise.run_layerwise_batch(
        root / "runs",
        embedding_series=embeddings,
        layers=(0, 12),
        y=[str(value) for value in problem.y],
        item_ids=[str(value) for value in problem.item_ids],
        outer_folds=problem.outer,
        inner_folds=problem.inner,
        class_names=problem.names,
        pooling="mean",
        C_grid=(0.01, 1.0),
        selection_metric="log_loss",
    )
    return batch, problem


def test_layerwise_batch_resume_reuses_completed_children(tmp_path: Path) -> None:
    problem = _synthetic_problem()
    embedding = _write_embedding_artifact(
        tmp_path / "embedding", item_ids=problem.item_ids, seed=11
    )
    output_root = tmp_path / "runs"
    # Simulate an earlier interrupted invocation: only layer 0 was completed.
    first_child = output_root / "synthetic" / "mean" / "layer-0"
    run_crowd_layer_probe(
        first_child,
        embedding_directory=embedding,
        layer=0,
        y=problem.y,
        item_ids=problem.item_ids,
        outer_folds=problem.outer,
        inner_folds=problem.inner,
        class_names=problem.names,
        pooling="mean",
        C_grid=(0.01, 1.0),
        selection_metric="log_loss",
    )
    child_metadata_before = (first_child / "metadata.json").read_bytes()

    batch = _run_batch(output_root, embedding, problem, layers=(0, 1))
    assert [record.status for record in batch.records] == ["validated", "completed"]
    # The resumed child was never rewritten.
    assert (first_child / "metadata.json").read_bytes() == child_metadata_before
    manifest_bytes = batch.manifest_path.read_bytes()
    summary_bytes = batch.summary_path.read_bytes()

    resumed = _run_batch(output_root, embedding, problem, layers=(0, 1))
    assert [record.status for record in resumed.records] == ["validated", "validated"]
    assert resumed.manifest_path.read_bytes() == manifest_bytes
    assert resumed.summary_path.read_bytes() == summary_bytes
    assert resumed.manifest == batch.manifest

    # A divergent requested batch must not replace the published aggregate.
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_batch(output_root, embedding, problem, layers=(0, 1, 2))

    # An unreadable aggregate is refused as well.
    batch.manifest_path.write_text("not json{\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_batch(output_root, embedding, problem, layers=(0, 1))


def test_layerwise_batch_rejects_corrupt_child_without_overwriting(
    tmp_path: Path,
) -> None:
    problem = _synthetic_problem()
    embedding = _write_embedding_artifact(
        tmp_path / "embedding", item_ids=problem.item_ids, seed=11
    )
    batch = _run_batch(tmp_path / "runs", embedding, problem)
    child = batch.records[0].directory
    manifest_bytes = batch.manifest_path.read_bytes()

    oof_path = child / "oof.parquet"
    with oof_path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    corrupted_bytes = oof_path.read_bytes()

    with pytest.raises(ValueError, match="hash mismatch"):
        _run_batch(tmp_path / "runs", embedding, problem)
    # The corrupt artifact is rejected, never repaired or overwritten.
    assert oof_path.read_bytes() == corrupted_bytes
    assert batch.manifest_path.read_bytes() == manifest_bytes


def test_layerwise_batch_rejects_partial_child_and_ignores_stale_temp(
    tmp_path: Path,
) -> None:
    problem = _synthetic_problem()
    embedding = _write_embedding_artifact(
        tmp_path / "embedding", item_ids=problem.item_ids, seed=11
    )
    output_root = tmp_path / "runs"
    child = output_root / "synthetic" / "mean" / "layer-0"
    child.mkdir(parents=True)
    (child / "metadata.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="partial crowd layer run"):
        _run_batch(output_root, embedding, problem)
    assert not (child / "oof.parquet").exists()

    # A stale temporary directory from a killed run must not block progress.

    shutil.rmtree(child)
    stale = child.parent / f".{child.name}.tmp-stale"
    stale.mkdir()
    (stale / "junk").write_text("interrupted\n", encoding="utf-8")
    batch = _run_batch(output_root, embedding, problem)
    assert [record.status for record in batch.records] == ["completed"]
    assert validate_crowd_layer_probe(child).metadata["layer"] == 0
    assert stale.is_dir()


def test_layerwise_batch_rejects_incompatible_children(tmp_path: Path) -> None:
    problem = _synthetic_problem()
    embedding = _write_embedding_artifact(
        tmp_path / "embedding", item_ids=problem.item_ids, seed=11
    )

    metric_root = tmp_path / "runs-metric"
    run_crowd_layer_probe(
        metric_root / "synthetic" / "mean" / "layer-0",
        embedding_directory=embedding,
        layer=0,
        y=problem.y,
        item_ids=problem.item_ids,
        outer_folds=problem.outer,
        inner_folds=problem.inner,
        class_names=problem.names,
        pooling="mean",
        C_grid=(0.01, 1.0),
        selection_metric="macro_f1",
    )
    with pytest.raises(ValueError, match="incompatible"):
        _run_batch(metric_root, embedding, problem, selection_metric="log_loss")

    grid_root = tmp_path / "runs-grid"
    run_crowd_layer_probe(
        grid_root / "synthetic" / "mean" / "layer-0",
        embedding_directory=embedding,
        layer=0,
        y=problem.y,
        item_ids=problem.item_ids,
        outer_folds=problem.outer,
        inner_folds=problem.inner,
        class_names=problem.names,
        pooling="mean",
        C_grid=(0.01, 1.0),
        selection_metric="log_loss",
    )
    with pytest.raises(ValueError, match="incompatible"):
        _run_batch(grid_root, embedding, problem, C_grid=(0.1,))


def test_layerwise_batch_summary_integrity(two_series_batch) -> None:
    batch, problem = two_series_batch
    spec = get_model_spec("roberta-base")
    summary = json.loads(batch.summary_path.read_text(encoding="utf-8"))
    assert summary == batch.summary
    assert summary["summary_format"] == layerwise.SUMMARY_FORMAT
    assert summary["summary_format"].endswith("reconstruction-v1")
    assert summary["provenance"]["historical_source_recovered"] is False
    assert summary["series"] == ["alpha", "beta"]
    assert summary["layers"] == [0, 12]
    assert len(summary["rows"]) == 4

    for row in summary["rows"]:
        child = validate_crowd_layer_probe(batch.output_root / row["run_path"])
        overall = reconstruct_multiclass_metrics(child.oof, labels=problem.names).overall
        assert row["oof_macro_f1"] == pytest.approx(float(overall["macro_f1"]))
        assert row["oof_log_loss_bits"] == pytest.approx(
            float(overall["log_loss_bits"])
        )
        assert row["layer"] == child.metadata["layer"]
        assert row["run_metadata_sha256"] == _sha256_file(
            child.directory / "metadata.json"
        )
        assert row["embedding_metadata_sha256"] == child.metadata[
            "embedding_metadata_sha256"
        ]
        assert row["embedding_layer_sha256"] == child.metadata[
            "embedding_layer_sha256"
        ]
        assert row["ordered_item_target_sha256"] == child.metadata[
            "ordered_item_target_sha256"
        ]
        selections = child.selections.sort_values("outer_fold", kind="stable")
        assert [fold["outer_fold"] for fold in row["fold_selections"]] == [
            int(value) for value in selections["outer_fold"]
        ]
        assert [fold["C"] for fold in row["fold_selections"]] == [
            float(value) for value in selections["C"]
        ]

    manifest = json.loads(batch.manifest_path.read_text(encoding="utf-8"))
    assert manifest == batch.manifest
    assert manifest["batch_format"] == layerwise.BATCH_FORMAT
    assert manifest["provenance"]["historical_source_recovered"] is False
    assert manifest["ordered_item_target_sha256"] == _ordered_pair_digest(
        [str(value) for value in problem.item_ids],
        [str(value) for value in problem.y],
    )
    assert manifest["outer_split_sha256"] == _dataframe_digest(problem.outer)
    assert manifest["inner_split_sha256"] == _dataframe_digest(problem.inner)
    assert manifest["implementation_sha256"]["run_crowd_layerwise.py"] == (
        _sha256_file(SCRIPTS / "run_crowd_layerwise.py")
    )
    assert [entry["name"] for entry in manifest["series"]] == ["alpha", "beta"]
    for entry in manifest["series"]:
        assert entry["model_key"] == "roberta-base"
        assert entry["revision"] == spec.revision
        assert entry["mode"] == "pretrained"
        assert entry["text_variant"] == "masked"
    assert len(manifest["runs"]) == 4
    for entry in manifest["runs"]:
        child_directory = batch.output_root / entry["run_path"]
        assert entry["run_metadata_sha256"] == _sha256_file(
            child_directory / "metadata.json"
        )


def test_layerwise_batch_selection_ignores_test_labels(tmp_path: Path) -> None:
    problem = _synthetic_problem()
    embedding = _write_embedding_artifact(
        tmp_path / "embedding", item_ids=problem.item_ids, seed=11
    )
    names = np.asarray(problem.names)
    index = {name: position for position, name in enumerate(problem.names)}
    in_fold_zero_test = problem.outer["test_fold"].to_numpy() == 0
    rotated = problem.y.copy()
    rotated[in_fold_zero_test] = names[
        (np.asarray([index[value] for value in problem.y[in_fold_zero_test]]) + 1)
        % len(problem.names)
    ]

    observed = _run_batch(tmp_path / "observed", embedding, problem)
    permuted = _run_batch(
        tmp_path / "permuted",
        embedding,
        problem,
        y=[str(value) for value in rotated],
    )
    observed_child = validate_crowd_layer_probe(observed.records[0].directory)
    permuted_child = validate_crowd_layer_probe(permuted.records[0].directory)

    # Sanity: the fold-0 test labels really changed.
    assert observed_child.metadata["ordered_item_target_sha256"] != (
        permuted_child.metadata["ordered_item_target_sha256"]
    )
    left = observed_child.oof.sort_values("item_id", kind="stable")
    right = permuted_child.oof.sort_values("item_id", kind="stable")
    test_mask = left["outer_fold"].to_numpy() == 0
    assert not np.array_equal(
        left["y_true"].to_numpy()[test_mask],
        right["y_true"].to_numpy()[test_mask],
    )
    # Fold-0 selection and fold-0 out-of-fold probabilities are bit-identical:
    # hyperparameter selection and fitting consumed outer-train data only.
    probability_columns = [
        column for column in left.columns if column.startswith("prob__")
    ]
    assert np.array_equal(
        left[probability_columns].to_numpy()[test_mask],
        right[probability_columns].to_numpy()[test_mask],
    )
    observed_fold = observed_child.selections.set_index("outer_fold").loc[0]
    permuted_fold = permuted_child.selections.set_index("outer_fold").loc[0]
    assert float(observed_fold["C"]) == float(permuted_fold["C"])
    assert float(observed_fold["inner_log_loss_bits"]) == float(
        permuted_fold["inner_log_loss_bits"]
    )
    # The reported metrics do respond to the labels they are evaluated against.
    assert observed.summary["rows"][0]["oof_log_loss_bits"] != pytest.approx(
        permuted.summary["rows"][0]["oof_log_loss_bits"]
    )


def test_plot_crowd_layerwise_renders_summary_only(
    two_series_batch, tmp_path: Path
) -> None:
    batch, _ = two_series_batch
    summary = layerwise.load_layerwise_summary(batch.summary_path)
    figure = plotscript.build_figure(summary, metric="oof_macro_f1")
    axis = figure.axes[0]
    assert len(axis.lines) == 2
    for line, name in zip(axis.lines, ["alpha", "beta"]):
        rows = sorted(
            (row for row in summary["rows"] if row["series"] == name),
            key=lambda row: row["layer"],
        )
        assert list(line.get_xdata()) == [0, 12]
        assert list(line.get_ydata()) == pytest.approx(
            [row["oof_macro_f1"] for row in rows]
        )

    output = tmp_path / "trajectory.png"
    assert (
        plotscript.main(
            ["--summary", str(batch.summary_path), "--output", str(output)]
        )
        == 0
    )
    assert output.stat().st_size > 0

    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps({"summary_format": "unexpected", "rows": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown crowd layerwise summary format"):
        plotscript.main(
            ["--summary", str(malformed), "--output", str(tmp_path / "x.png")]
        )


def test_layerwise_cli_parsers() -> None:
    assert layerwise._parse_layers("0-2,5") == (0, 1, 2, 5)
    assert layerwise._parse_layers("12") == (12,)
    for invalid in ("", "3-1", "a", "1,1", "-1", "0-1-2"):
        with pytest.raises(argparse.ArgumentTypeError):
            layerwise._parse_layers(invalid)

    spec = layerwise._parse_series_spec("roberta-base")
    assert spec.name == "roberta-base/pretrained/masked/maxlen-256"
    spec = layerwise._parse_series_spec("deberta-v3-base/random/original/512")
    assert spec.name == "deberta-v3-base/random/original/maxlen-512"
    with pytest.raises(argparse.ArgumentTypeError):
        layerwise._parse_series_spec("unknown-model")
    with pytest.raises(argparse.ArgumentTypeError):
        layerwise._parse_series_spec("roberta-base/bogus-mode")
    with pytest.raises(argparse.ArgumentTypeError):
        layerwise._parse_float_grid("0,1")

    parser = layerwise.build_parser()
    arguments = parser.parse_args(
        [
            "--archive", "crowd.zip",
            "--splits", "splits",
            "--cache-root", "cache",
            "--series", "roberta-base",
            "--output-root", "runs",
            "--layers", "0-12",
            "--selection-metric", "log_loss",
        ]
    )
    assert arguments.pooling == "mean"
    assert arguments.C_grid == tuple(DEFAULT_C_GRID)
    assert arguments.layers == tuple(range(13))
    assert arguments.series[0].name == "roberta-base/pretrained/masked/maxlen-256"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--archive", "crowd.zip",
                "--splits", "splits",
                "--cache-root", "cache",
                "--output-root", "runs",
                "--layers", "0-12",
                "--selection-metric", "log_loss",
            ]
        )
