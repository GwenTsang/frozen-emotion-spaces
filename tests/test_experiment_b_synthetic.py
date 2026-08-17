"""Snapshot-free synthetic tests for the EmoTwiCS Q1 all-layer rerun pipeline.

Unlike ``test_experiment_b.py`` (gated on a pinned local RoBERTa snapshot),
these tests write a small synthetic embedding artifact to disk directly, so
the full ``run_emotwics_layer_probe`` path — artifact validation, nested
conversation-disjoint selection, atomic publication, and run validation — is
exercised without any Transformer checkpoint.

Covered contracts:
- conversation (group) disjointness of outer and inner folds
- item-level probability/truth/prediction alignment in the OOF artifact
- train-only fitting of every transformation (no fit ever sees outer-test rows)
- multilabel metric reconstruction against hand-computed values
- rejection of corrupt, partial, tampered, or misaligned artifacts
- layer-bounds enforcement
- resumable-run identity checks
- all-layer aggregate provenance guards and the aggregate-consuming plot path
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import frozen_emotion_spaces.experiment_b as experiment_b
from frozen_emotion_spaces import cli
from frozen_emotion_spaces.config import EMBEDDING_DTYPE, get_model_spec
from frozen_emotion_spaces.embeddings import (
    ARTIFACT_FORMAT,
    validate_embedding_artifact,
)
from frozen_emotion_spaces.experiment_b import (
    RUN_FORMAT,
    build_all_layer_summary,
    resumable_run_emotwics_layer_probe,
    run_emotwics_layer_probe,
    validate_emotwics_layer_probe,
)
from frozen_emotion_spaces.metrics import reconstruct_multilabel_metrics
from frozen_emotion_spaces.probes import make_dense_multilabel_factory


N_ITEMS = 27
LABEL_NAMES = ("Anger", "Joy", "Neutral")
N_LABELS = len(LABEL_NAMES)
C_GRID = (0.1, 1.0)
THRESHOLD_GRID = (0.3, 0.5, 0.7)
PLOT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_emotwics_layer_trajectory.py"
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_synthetic_embedding(
    directory: Path,
    item_ids: np.ndarray,
    *,
    seed: int = 20240804,
) -> Path:
    """Write a spec-conformant synthetic embedding artifact for roberta-base."""

    spec = get_model_spec("roberta-base")
    n_layers, hidden = spec.emitted_layers, spec.hidden_size
    rng = np.random.default_rng(seed)
    mean = rng.standard_normal(
        (n_layers, len(item_ids), hidden), dtype=np.float32
    )
    first = rng.standard_normal(
        (n_layers, len(item_ids), hidden), dtype=np.float32
    )
    for layer in range(n_layers):  # keep layers distinct for trajectories
        mean[layer] *= np.float32(1.0 + 0.05 * layer)
        first[layer] *= np.float32(1.0 + 0.03 * layer)

    directory.mkdir(parents=True)
    np.save(directory / "mean.npy", mean, allow_pickle=False)
    np.save(directory / "first.npy", first, allow_pickle=False)
    np.save(directory / "item_ids.npy", item_ids.astype(str), allow_pickle=False)

    files: dict[str, dict[str, object]] = {}
    for name in ("mean.npy", "first.npy", "item_ids.npy"):
        path = directory / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record: dict[str, object] = {
            "sha256": _sha256_bytes(path.read_bytes()),
            "bytes": path.stat().st_size,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if name in {"mean.npy", "first.npy"}:
            record["layer_sha256"] = [
                _sha256_bytes(np.ascontiguousarray(array[l]).tobytes())
                for l in range(array.shape[0])
            ]
        files[name] = record

    metadata = {
        "artifact_format": ARTIFACT_FORMAT,
        "dataset": "emotwics",
        "model_key": spec.key,
        "repository": spec.repository,
        "revision": spec.revision,
        "mode": "pretrained",
        "text_variant": "original",
        "max_length": 32,
        "batch_size": 8,
        "seed": 0,
        "initialization_seed": None,
        "torch_version": "synthetic",
        "transformers_version": "synthetic",
        "device_type": "cpu",
        "device_name": "cpu",
        "model_content_sha256": "0" * 64,
        "tokenizer_class": "SyntheticTokenizer",
        "tokenizer_is_fast": True,
        "tokenizer_padding_side": "right",
        "tokenizer_truncation_side": "right",
        "tokenizer_fingerprint_sha256": "1" * 64,
        "tokenizers_version": "synthetic",
        "sentencepiece_version": "synthetic",
        "storage_dtype": EMBEDDING_DTYPE,
        "n_items": int(len(item_ids)),
        "n_layers": int(n_layers),
        "hidden_size": int(hidden),
        "truncated_items": 0,
        "truncation_rate": 0.0,
        "maximum_tokenized_length": 16,
        "ordered_texts_sha256": "2" * 64,
        "ordered_item_text_pairs_sha256": "3" * 64,
        "mean_pooling": "synthetic masked mean",
        "first_pooling": "synthetic position zero",
        "files": files,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def _make_item_ids() -> np.ndarray:
    return np.asarray([f"tweet-{index:03d}" for index in range(N_ITEMS)])


def _make_labels() -> np.ndarray:
    """Deterministic non-degenerate binary label matrix."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=(N_ITEMS, N_LABELS), dtype=np.int64)
    for column in range(N_LABELS):
        y[0, column] = 1
        y[-1, column] = 0
    return y


def _nested_tables(item_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conversation-disjoint 3-outer / 2-inner splits (whole conversations)."""
    conversation = np.asarray([index // 3 for index in range(len(item_ids))])
    groups = np.asarray([f"conv-{value:02d}" for value in conversation])
    outer_assignment = conversation % 3
    outer = pd.DataFrame(
        {
            "item_id": item_ids,
            "group_id": groups,
            "conversation_id": groups,
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
                    "conversation_id": groups[index],
                    "validation_fold": int(conversation[index] % 2),
                }
            )
    return outer, pd.DataFrame(rows)


def _run_kwargs(embedding: Path) -> dict[str, object]:
    item_ids = _make_item_ids()
    outer, inner = _nested_tables(item_ids)
    return {
        "embedding_directory": embedding,
        "y": _make_labels(),
        "item_ids": item_ids,
        "label_names": LABEL_NAMES,
        "outer_folds": outer,
        "inner_folds": inner,
        "C_grid": C_GRID,
        "threshold_grid": THRESHOLD_GRID,
        "selection_metric": "log_loss",
    }


@pytest.fixture(scope="module")
def embedding(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic-embedding") / "artifact"
    return _write_synthetic_embedding(root, _make_item_ids())


@pytest.fixture(scope="module")
def completed_run(
    tmp_path_factory: pytest.TempPathFactory, embedding: Path
) -> Path:
    """One completed layer-0 run shared by read-only tests."""
    output = tmp_path_factory.mktemp("runs") / "layer-0"
    run_emotwics_layer_probe(output, layer=0, **_run_kwargs(embedding))
    return output


# ---------------------------------------------------------------------------
# Synthetic artifact sanity
# ---------------------------------------------------------------------------

def test_synthetic_embedding_artifact_passes_validation(embedding: Path) -> None:
    artifact = validate_embedding_artifact(
        embedding, expected_item_ids=[str(v) for v in _make_item_ids()]
    )
    assert artifact.metadata["n_layers"] == 13
    assert artifact.metadata["hidden_size"] == 768
    assert len(artifact.item_ids) == N_ITEMS


# ---------------------------------------------------------------------------
# Probability alignment
# ---------------------------------------------------------------------------

def test_oof_probabilities_align_with_truth_and_predictions(
    completed_run: Path,
) -> None:
    artifact = validate_emotwics_layer_probe(completed_run)
    oof = artifact.oof
    assert len(oof) == N_ITEMS
    assert not oof["item_id"].duplicated().any()

    y = _make_labels()
    truth_by_id = {str(item_id): y[index] for index, item_id in enumerate(_make_item_ids())}
    for label_index, label in enumerate(LABEL_NAMES):
        probabilities = oof[f"prob__{label}"].to_numpy(dtype=float)
        truth = oof[f"y_true__{label}"].to_numpy(dtype=int)
        predictions = oof[f"pred__{label}"].to_numpy(dtype=int)
        assert ((probabilities > 0) & (probabilities < 1)).all()
        # Truth column reproduces the input label matrix row-for-row.
        expected = np.asarray(
            [truth_by_id[str(item_id)][label_index] for item_id in oof["item_id"]]
        )
        assert np.array_equal(truth, expected)
        # Predictions equal the row-wise thresholded probabilities.
        thresholds = oof["threshold"].to_numpy(dtype=float)
        assert np.array_equal(predictions, (probabilities >= thresholds).astype(int))

    # The serialized artifact alone reconstructs a consistent metric bundle.
    metrics = reconstruct_multilabel_metrics(oof, labels=LABEL_NAMES)
    assert 0.0 <= float(metrics.overall["macro_f1"]) <= 1.0
    assert 0.0 <= float(metrics.overall["macro_ap"]) <= 1.0
    assert float(metrics.overall["log_loss_bits"]) >= 0.0


def test_misaligned_probability_columns_rejected(completed_run: Path, tmp_path: Path) -> None:
    """Swapping two prob__ columns must fail the pred/probability contract."""
    corrupted = tmp_path / "run"
    shutil.copytree(completed_run, corrupted)
    oof = pd.read_parquet(corrupted / "oof.parquet", engine="pyarrow")
    oof[[f"prob__{LABEL_NAMES[0]}", f"prob__{LABEL_NAMES[1]}"]] = oof[
        [f"prob__{LABEL_NAMES[1]}", f"prob__{LABEL_NAMES[0]}"]
    ]
    oof.to_parquet(
        corrupted / "oof.parquet", index=False, engine="pyarrow", compression="zstd"
    )
    _refresh_file_record(corrupted, "oof.parquet")
    with pytest.raises(ValueError, match="disagrees with the applicable threshold"):
        validate_emotwics_layer_probe(corrupted)


# ---------------------------------------------------------------------------
# Conversation disjointness
# ---------------------------------------------------------------------------

def test_conversation_disjoint_outer_and_inner_folds(completed_run: Path) -> None:
    artifact = validate_emotwics_layer_probe(completed_run)
    assert artifact.oof.groupby("group_id")["outer_fold"].nunique().max() == 1
    # Each outer fold holds out exactly the conversations assigned to it.
    counts = artifact.oof.groupby("outer_fold")["group_id"].nunique()
    assert sorted(counts.tolist()) == [3, 3, 3]


def test_group_leakage_across_outer_folds_rejected(
    embedding: Path, tmp_path: Path
) -> None:
    item_ids = _make_item_ids()
    outer, inner = _nested_tables(item_ids)
    outer.loc[0, "test_fold"] = 1  # conv-00 now spans folds 0 and 1
    with pytest.raises(ValueError, match="group leakage"):
        run_emotwics_layer_probe(
            tmp_path / "run", layer=0, **{**_run_kwargs(embedding), "outer_folds": outer, "inner_folds": inner}
        )


def test_group_leakage_across_inner_folds_rejected(
    embedding: Path, tmp_path: Path
) -> None:
    item_ids = _make_item_ids()
    outer, inner = _nested_tables(item_ids)
    # Split conv-04 (outer-train for fold 0) across both inner folds.
    mask = (inner["outer_fold"] == 0) & (inner["conversation_id"] == "conv-04")
    inner.loc[mask, "validation_fold"] = [0, 1, 1]
    with pytest.raises(ValueError, match="group leakage in inner fold"):
        run_emotwics_layer_probe(
            tmp_path / "run", layer=0, **{**_run_kwargs(embedding), "outer_folds": outer, "inner_folds": inner}
        )


# ---------------------------------------------------------------------------
# Train-only transformations
# ---------------------------------------------------------------------------

def test_every_transformation_is_fitted_on_outer_train_only(
    embedding: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No estimator fit may ever see an item from the held-out outer fold."""
    item_ids = _make_item_ids()
    outer, inner = _nested_tables(item_ids)
    features = np.load(embedding / "mean.npy", mmap_mode="r")[0]
    # The probe machinery upcasts features to float64 before any fit call.
    features64 = np.asarray(features, dtype=np.float64)
    index_by_row = {features64[i].tobytes(): i for i in range(N_ITEMS)}
    fold_of_item = dict(zip(range(N_ITEMS), outer["test_fold"].tolist()))

    fit_calls: list[np.ndarray] = []
    real_factory = make_dense_multilabel_factory()

    def recording_factory() -> object:
        def factory(C: float) -> object:
            estimator = real_factory(C)
            original_fit = estimator.fit

            def fit(X, y):
                fit_calls.append(np.asarray(X).copy())
                return original_fit(X, y)

            estimator.fit = fit
            return estimator

        return factory

    monkeypatch.setattr(
        experiment_b, "make_dense_multilabel_factory", recording_factory
    )
    run_emotwics_layer_probe(
        tmp_path / "run",
        layer=0,
        **{**_run_kwargs(embedding), "outer_folds": outer, "inner_folds": inner},
    )

    assert fit_calls, "expected the spy to observe estimator fits"
    outer_refit_exclusions: list[int] = []
    for matrix in fit_calls:
        seen = {index_by_row[matrix[row].tobytes()] for row in range(matrix.shape[0])}
        folds_seen = {fold_of_item[index] for index in seen}
        # No fit touches items from every outer fold — that would be leakage.
        assert len(folds_seen) < 3
        if len(seen) == 18:  # outer-train refit: exactly one fold excluded
            excluded = {0, 1, 2} - folds_seen
            assert len(excluded) == 1
            missing_items = set(range(N_ITEMS)) - seen
            assert {fold_of_item[index] for index in missing_items} == excluded
            outer_refit_exclusions.append(excluded.pop())
    # Three outer refits, each leaving out a different fold.
    assert sorted(outer_refit_exclusions) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Multilabel metric calculation
# ---------------------------------------------------------------------------

def _hand_built_oof(thresholds: list[float] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "item_id": ["i0", "i1", "i2", "i3"],
            "y_true__a": [1, 0, 1, 0],
            "y_true__b": [0, 1, 0, 1],
            "prob__a": [0.9, 0.1, 0.4, 0.2],
            "prob__b": [0.2, 0.8, 0.3, 0.7],
        }
    )
    if thresholds is not None:
        frame["threshold"] = thresholds
        frame["pred__a"] = (frame["prob__a"] >= frame["threshold"]).astype(int)
        frame["pred__b"] = (frame["prob__b"] >= frame["threshold"]).astype(int)
    return frame


def test_multilabel_metrics_match_hand_computed_values() -> None:
    metrics = reconstruct_multilabel_metrics(
        _hand_built_oof(), labels=("a", "b"), threshold=0.5
    )
    overall = metrics.overall
    # label a: pred [1,0,0,0] -> P=1, R=1/2, F1=2/3; label b: perfect -> F1=1
    assert float(overall["macro_f1"]) == pytest.approx((2 / 3 + 1.0) / 2)
    # Both labels rank the positives first.
    assert float(overall["macro_ap"]) == pytest.approx(1.0)
    # Exact match on 3 of 4 items; 7 of 8 cells correct.
    assert float(overall["accuracy"]) == pytest.approx(0.75)
    assert float(overall["hamming_accuracy"]) == pytest.approx(0.875)
    # Mean binary cross-entropy in bits over all 8 item-label cells.
    cells = [
        (1, 0.9), (0, 0.1), (1, 0.4), (0, 0.2),
        (0, 0.2), (1, 0.8), (0, 0.3), (1, 0.7),
    ]
    expected_loss = float(
        np.mean([-np.log2(p if t else 1.0 - p) for t, p in cells])
    )
    assert float(overall["log_loss_bits"]) == pytest.approx(expected_loss)
    assert overall["threshold_mode"] == "global"
    assert int(overall["n_thresholds"]) == 1


def test_multilabel_metrics_row_wise_thresholds() -> None:
    metrics = reconstruct_multilabel_metrics(
        _hand_built_oof(thresholds=[0.5, 0.5, 0.6, 0.5]), labels=("a", "b")
    )
    assert metrics.overall["threshold_mode"] == "row_wise"
    assert float(metrics.overall["macro_f1"]) == pytest.approx((2 / 3 + 1.0) / 2)


def test_multilabel_metrics_reject_inconsistent_stored_predictions() -> None:
    frame = _hand_built_oof(thresholds=[0.5] * 4)
    frame.loc[0, "pred__a"] = 0  # contradicts prob 0.9 >= 0.5
    with pytest.raises(ValueError, match="disagrees with the applicable threshold"):
        reconstruct_multilabel_metrics(frame, labels=("a", "b"))


# ---------------------------------------------------------------------------
# Corruption / tampering rejection
# ---------------------------------------------------------------------------

def _refresh_file_record(run_dir: Path, name: str) -> None:
    """Update the metadata file record after a deliberate byte-level change."""
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target = run_dir / name
    metadata["files"][name] = {
        "sha256": _sha256_bytes(target.read_bytes()),
        "bytes": target.stat().st_size,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_oof_parquet_bit_corruption_rejected(completed_run: Path, tmp_path: Path) -> None:
    corrupted = tmp_path / "run"
    shutil.copytree(completed_run, corrupted)
    path = corrupted / "oof.parquet"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_emotwics_layer_probe(corrupted)


def test_partial_artifact_rejected(completed_run: Path, tmp_path: Path) -> None:
    corrupted = tmp_path / "run"
    shutil.copytree(completed_run, corrupted)
    (corrupted / "selections.parquet").unlink()
    with pytest.raises(ValueError, match="missing files"):
        validate_emotwics_layer_probe(corrupted)


def test_tampered_metadata_rejected(completed_run: Path, tmp_path: Path) -> None:
    corrupted = tmp_path / "run"
    shutil.copytree(completed_run, corrupted)
    metadata_path = corrupted / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["selection_metric"] = "accuracy"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid selection metric"):
        validate_emotwics_layer_probe(corrupted)


def test_oof_threshold_misaligned_with_selection_rejected(
    completed_run: Path, tmp_path: Path
) -> None:
    """An OOF threshold inconsistent with the inner-fold selection is caught."""
    corrupted = tmp_path / "run"
    shutil.copytree(completed_run, corrupted)
    oof = pd.read_parquet(corrupted / "oof.parquet", engine="pyarrow")
    selected = set(oof["threshold"].unique())
    replacement = next(v for v in THRESHOLD_GRID if v not in selected)
    oof["threshold"] = replacement
    for label in LABEL_NAMES:  # keep pred consistent so only the cross-check fires
        oof[f"pred__{label}"] = (oof[f"prob__{label}"] >= replacement).astype(int)
    oof.to_parquet(
        corrupted / "oof.parquet", index=False, engine="pyarrow", compression="zstd"
    )
    _refresh_file_record(corrupted, "oof.parquet")
    with pytest.raises(ValueError, match="OOF threshold disagrees"):
        validate_emotwics_layer_probe(corrupted)


# ---------------------------------------------------------------------------
# Layer bounds
# ---------------------------------------------------------------------------

def test_layer_bounds_enforced(embedding: Path, tmp_path: Path) -> None:
    for bad_layer in (-1, 13):
        with pytest.raises(ValueError, match="layer is outside"):
            run_emotwics_layer_probe(
                tmp_path / f"run-{bad_layer}", layer=bad_layer, **_run_kwargs(embedding)
            )
    artifact = run_emotwics_layer_probe(
        tmp_path / "run-12", layer=12, **_run_kwargs(embedding)
    )
    assert artifact.metadata["layer"] == 12


# ---------------------------------------------------------------------------
# Resumable identity
# ---------------------------------------------------------------------------

def test_resumable_run_returns_matching_artifact(
    embedding: Path, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    kwargs = _run_kwargs(embedding)
    first = run_emotwics_layer_probe(output, layer=0, **kwargs)
    second = resumable_run_emotwics_layer_probe(output, layer=0, **kwargs)
    assert second.metadata == first.metadata


def test_resumable_run_rejects_mismatched_request(
    embedding: Path, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    kwargs = _run_kwargs(embedding)
    run_emotwics_layer_probe(output, layer=0, **kwargs)
    with pytest.raises(ValueError, match="does not match the run request"):
        resumable_run_emotwics_layer_probe(output, layer=1, **kwargs)
    with pytest.raises(ValueError, match="does not match the run request"):
        resumable_run_emotwics_layer_probe(
            output, layer=0, **{**kwargs, "selection_metric": "macro_f1"}
        )
    item_ids = _make_item_ids()
    outer, inner = _nested_tables(item_ids)
    shifted = outer.copy()
    shifted["test_fold"] = (shifted["test_fold"] + 1) % 3
    with pytest.raises(ValueError, match="does not match the run request"):
        resumable_run_emotwics_layer_probe(
            output, layer=0, **{**kwargs, "outer_folds": shifted}
        )


# ---------------------------------------------------------------------------
# All-layer aggregate and plotting path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_layer_runs(
    tmp_path_factory: pytest.TempPathFactory, embedding: Path
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("summary-runs")
    for layer in (0, 1):
        run_emotwics_layer_probe(
            root / f"layer-{layer}", layer=layer, **_run_kwargs(embedding)
        )
    return root / "layer-0", root / "layer-1"


def test_all_layer_summary_carries_reconstruction_provenance(
    two_layer_runs: tuple[Path, Path],
) -> None:
    artifacts = [
        validate_emotwics_layer_probe(path) for path in reversed(two_layer_runs)
    ]
    summary = build_all_layer_summary(artifacts)
    frame = summary.layers
    assert frame["layer"].tolist() == [0, 1]
    assert frame["run_format"].str.contains("reconstruction").all()
    assert (frame["run_format"] == RUN_FORMAT).all()
    assert frame["dataset"].eq("emotwics").all()
    metrics_by_layer = {
        int(artifact.metadata["layer"]): reconstruct_multilabel_metrics(
            artifact.oof, labels=LABEL_NAMES
        )
        for artifact in artifacts
    }
    for row in frame.itertuples():
        metrics = metrics_by_layer[int(row.layer)]
        assert row.macro_f1 == pytest.approx(float(metrics.overall["macro_f1"]))
        assert row.macro_ap == pytest.approx(float(metrics.overall["macro_ap"]))


def test_all_layer_summary_rejects_duplicate_layers(
    two_layer_runs: tuple[Path, Path],
) -> None:
    artifact = validate_emotwics_layer_probe(two_layer_runs[0])
    with pytest.raises(ValueError, match="duplicate layer"):
        build_all_layer_summary([artifact, artifact])


def test_all_layer_summary_rejects_mixed_provenance(
    embedding: Path, tmp_path: Path, two_layer_runs: tuple[Path, Path]
) -> None:
    other = run_emotwics_layer_probe(
        tmp_path / "run",
        layer=1,
        **{**_run_kwargs(embedding), "selection_metric": "macro_f1"},
    )
    baseline = validate_emotwics_layer_probe(two_layer_runs[0])
    with pytest.raises(ValueError, match="mix incompatible provenance"):
        build_all_layer_summary([baseline, other])


def _load_plot_module():
    spec = importlib.util.spec_from_file_location(
        "plot_emotwics_layer_trajectory", PLOT_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_consumes_aggregate_values(
    two_layer_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    plot = _load_plot_module()
    artifacts = [validate_emotwics_layer_probe(path) for path in two_layer_runs]
    summary = build_all_layer_summary(artifacts)
    csv_path = tmp_path / "summary.csv"
    summary.layers.to_csv(csv_path, index=False, lineterminator="\n")

    frame = plot.load_summary(csv_path)
    figure_path = tmp_path / "figures" / "trajectory.pdf"
    plotted = plot.render(frame, figure_path)
    assert np.array_equal(plotted["layer"], frame["layer"].to_numpy())
    assert np.allclose(plotted["macro_f1"], frame["macro_f1"].to_numpy())
    assert np.allclose(plotted["macro_ap"], frame["macro_ap"].to_numpy())
    assert plotted["reference_value"] is None
    assert figure_path.read_bytes().startswith(b"%PDF")

    referenced = plot.render(
        frame,
        tmp_path / "figures" / "trajectory-ref.pdf",
        reference_value=0.4,
        reference_label="external reference",
    )
    assert referenced["reference_value"] == 0.4


def test_plot_rejects_invalid_aggregate(tmp_path: Path) -> None:
    plot = _load_plot_module()
    bad = pd.DataFrame({"layer": [0, 0], "macro_f1": [0.1, 0.2], "macro_ap": [0.3, 0.4]})
    csv_path = tmp_path / "bad.csv"
    bad.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="duplicate layers"):
        plot.load_summary(csv_path)
    frame = pd.DataFrame(
        {"layer": [0], "macro_f1": [0.5], "macro_ap": [0.5]}
    )
    with pytest.raises(ValueError, match="supplied together"):
        plot.render(frame, tmp_path / "out.pdf", reference_value=0.4)


def test_cli_summarize_emotwics_layers(
    two_layer_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "trajectory.csv"
    exit_code = cli.main(
        [
            "summarize-emotwics-layers",
            "--run",
            str(two_layer_runs[0]),
            "--run",
            str(two_layer_runs[1]),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    frame = pd.read_csv(output)
    assert frame["layer"].tolist() == [0, 1]
    assert frame["run_format"].str.contains("reconstruction").all()
    assert {"macro_f1", "macro_ap", "embedding_layer_sha256"} <= set(frame.columns)
