from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import frozen_emotion_spaces.cli as cli
from frozen_emotion_spaces.cli import (
    _lock_cache_tokenizer_backend,
    _parse_C_grid,
    _validate_matched_null_arguments,
    build_parser,
)


def test_cli_exposes_only_bounded_reconstruction_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "extract-crowd" in help_text
    assert "extract-emotwics" in help_text
    assert "index-embeddings" in help_text
    assert "index-crowd-runs" in help_text
    assert "index-representation-runs" in help_text
    assert "probe-crowd-layer" in help_text
    assert "probe-crowd-representation" in help_text
    assert "pilot-contrast-representation" in help_text
    assert "index-counterfactual-pilots" in help_text
    assert "analyze-observed-counterfactual" in help_text
    assert "compute-matched-nulls" in help_text
    assert "write-conditional-analysis" in help_text
    assert "analyze-observed-geometry" in help_text


def test_representation_cli_requires_an_explicit_objective() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "probe-crowd-representation",
                "--archive", "crowd.zip",
                "--splits", "splits",
                "--output", "run",
                "--representation", "A",
            ]
        )


def test_C_grid_parser_is_explicit_and_positive() -> None:
    assert _parse_C_grid("0.0001,0.1,100") == (0.0001, 0.1, 100.0)
    with pytest.raises(Exception, match="strictly positive"):
        _parse_C_grid("0,1")


def test_matched_null_cli_rejects_incompatible_space_arguments() -> None:
    with pytest.raises(ValueError, match="must not declare PCA"):
        _validate_matched_null_arguments(
            argparse.Namespace(
                space="A_STANDARDIZED",
                embedding_directory=None,
                pca_components=3,
            )
        )
    with pytest.raises(ValueError, match="embedding directory"):
        _validate_matched_null_arguments(
            argparse.Namespace(
                space="H_PCA",
                embedding_directory=None,
                pca_components=3,
            )
        )
    with pytest.raises(ValueError, match="PCA components"):
        _validate_matched_null_arguments(
            argparse.Namespace(
                space="H_PCA",
                embedding_directory="embeddings",
                pca_components=None,
            )
        )


def test_matched_null_cli_requires_positive_draw_count() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "compute-matched-nulls",
                "--archive", "crowd.zip",
                "--splits", "splits",
                "--source-run", "source",
                "--output", "nulls",
                "--space", "A_STANDARDIZED",
                "--n-draws-per-fold", "0",
            ]
        )


def test_cache_root_cannot_mix_fast_and_slow_tokenizers(tmp_path) -> None:
    _lock_cache_tokenizer_backend(tmp_path, is_fast=True)
    _lock_cache_tokenizer_backend(tmp_path, is_fast=True)

    with pytest.raises(ValueError, match="separate cache root"):
        _lock_cache_tokenizer_backend(tmp_path, is_fast=False)


def _conditional_analysis_argv(output: str = "analysis") -> list[str]:
    return [
        "write-conditional-analysis",
        "--A-run", "runs/A",
        "--H-run", "runs/H",
        "--AH-run", "runs/AH",
        "--output", output,
    ]


def test_conditional_analysis_cli_parses_runs_and_defaults() -> None:
    arguments = build_parser().parse_args(_conditional_analysis_argv())
    assert arguments.A_run == Path("runs/A")
    assert arguments.H_run == Path("runs/H")
    assert arguments.AH_run == Path("runs/AH")
    assert arguments.output == Path("analysis")
    assert arguments.n_bootstrap == 2000
    assert arguments.seed == 20240804


def test_conditional_analysis_cli_requires_every_run() -> None:
    parser = build_parser()
    for dropped in ("--A-run", "--H-run", "--AH-run", "--output"):
        argv = _conditional_analysis_argv()
        index = argv.index(dropped)
        del argv[index : index + 2]
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_conditional_analysis_cli_requires_positive_bootstrap() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [*_conditional_analysis_argv(), "--n-bootstrap", "0"]
        )


def test_conditional_analysis_handler_forwards_arguments(
    monkeypatch, tmp_path, capsys
) -> None:
    recorded: dict[str, object] = {}

    def fake_write(output_directory, **kwargs):
        recorded["output_directory"] = output_directory
        recorded.update(kwargs)
        return SimpleNamespace(
            directory=Path(output_directory),
            metadata={"analysis_format": "fake-conditional-format"},
        )

    monkeypatch.setattr(cli, "write_conditional_analysis", fake_write)
    output = tmp_path / "conditional"
    status = cli.main(
        [
            *_conditional_analysis_argv(output=str(output)),
            "--n-bootstrap", "7",
            "--seed", "13",
        ]
    )
    assert status == 0
    assert recorded == {
        "output_directory": output,
        "A_run": Path("runs/A"),
        "H_run": Path("runs/H"),
        "AH_run": Path("runs/AH"),
        "n_bootstrap": 7,
        "seed": 13,
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "directory": str(output),
        "format": "fake-conditional-format",
    }


def test_conditional_analysis_refuses_to_overwrite(tmp_path) -> None:
    output = tmp_path / "conditional"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.main(_conditional_analysis_argv(output=str(output)))
    assert sorted(path.name for path in output.iterdir()) == []


def _geometry_crowd(appraisal_names: tuple[str, ...]) -> SimpleNamespace:
    generation = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3"],
            **{
                name: [float(index + offset + 1) for index in range(3)]
                for offset, name in enumerate(appraisal_names)
            },
        }
    )
    return SimpleNamespace(generation=generation)


def _geometry_splits() -> SimpleNamespace:
    return SimpleNamespace(
        crowd_full_outer=pd.DataFrame(
            {"item_id": ["i1", "i2", "i3"], "outer_fold": [0, 0, 1]}
        )
    )


def _patch_observed_geometry_io(
    monkeypatch,
    *,
    representation: str,
    metadata_extra: dict | None = None,
) -> dict[str, object]:
    recorded: dict[str, object] = {}
    appraisal_names = ("suddenness", "pleasantness")
    metadata = {"representation": representation}
    if representation != "A":
        metadata.update({"layer": 3, "pooling": "first"})
    if metadata_extra:
        metadata.update(metadata_extra)

    monkeypatch.setattr(cli, "APPRAISAL_NAMES", appraisal_names)
    monkeypatch.setattr(
        cli,
        "build_crowd_manifests",
        lambda archive: _geometry_crowd(appraisal_names),
    )
    monkeypatch.setattr(
        cli, "read_split_bundle", lambda splits: _geometry_splits()
    )
    monkeypatch.setattr(
        cli,
        "validate_crowd_representation_probe",
        lambda source_run: SimpleNamespace(
            directory=Path(source_run), metadata=dict(metadata)
        ),
    )

    def fake_load_embedding_layer(directory, *, layer, pooling, expected_item_ids):
        recorded["embedding_directory"] = directory
        recorded["layer"] = layer
        recorded["pooling"] = pooling
        recorded["expected_item_ids"] = list(expected_item_ids)
        hidden = np.full((3, 2), 0.5, dtype=np.float32)
        return hidden, np.array(["i1", "i2", "i3"])

    monkeypatch.setattr(cli, "load_embedding_layer", fake_load_embedding_layer)

    def fake_write(output_directory, **kwargs):
        recorded["output_directory"] = output_directory
        recorded.update(kwargs)
        return SimpleNamespace(
            directory=Path(output_directory),
            metadata={"analysis_format": "fake-geometry-format"},
        )

    monkeypatch.setattr(cli, "write_observed_geometry_analysis", fake_write)
    return recorded


def _observed_geometry_argv(output: Path, *, embedding: bool) -> list[str]:
    argv = [
        "analyze-observed-geometry",
        "--archive", "crowd.zip",
        "--splits", "splits",
        "--source-run", "runs/source",
        "--output", str(output),
    ]
    if embedding:
        argv += ["--embedding-directory", "embeddings"]
    return argv


def test_observed_geometry_cli_requires_source_run_and_output() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze-observed-geometry",
                "--archive", "crowd.zip",
                "--splits", "splits",
                "--output", "geometry",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze-observed-geometry",
                "--archive", "crowd.zip",
                "--splits", "splits",
                "--source-run", "runs/source",
            ]
        )


def test_observed_geometry_A_handler_uses_appraisals(
    monkeypatch, tmp_path, capsys
) -> None:
    recorded = _patch_observed_geometry_io(monkeypatch, representation="A")
    output = tmp_path / "geometry"
    status = cli.main(_observed_geometry_argv(output, embedding=False))
    assert status == 0
    assert "embedding_directory" not in recorded
    expected = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    np.testing.assert_array_equal(recorded["features"], expected)
    assert recorded["item_ids"] == ["i1", "i2", "i3"]
    assert recorded["run_directory"] == Path("runs/source")
    assert recorded["output_directory"] == output
    pd.testing.assert_frame_equal(
        recorded["outer_folds"], _geometry_splits().crowd_full_outer
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "directory": str(output),
        "format": "fake-geometry-format",
    }


def test_observed_geometry_A_handler_rejects_embedding_directory(
    monkeypatch, tmp_path
) -> None:
    recorded = _patch_observed_geometry_io(monkeypatch, representation="A")
    with pytest.raises(ValueError, match="must not declare an embedding"):
        cli.main(_observed_geometry_argv(tmp_path / "geometry", embedding=True))
    assert "output_directory" not in recorded


def test_observed_geometry_H_handler_loads_source_layer(
    monkeypatch, tmp_path
) -> None:
    recorded = _patch_observed_geometry_io(monkeypatch, representation="H")
    status = cli.main(
        _observed_geometry_argv(tmp_path / "geometry", embedding=True)
    )
    assert status == 0
    assert recorded["embedding_directory"] == Path("embeddings")
    assert recorded["layer"] == 3
    assert recorded["pooling"] == "first"
    assert recorded["expected_item_ids"] == ["i1", "i2", "i3"]
    np.testing.assert_array_equal(
        recorded["features"], np.full((3, 2), 0.5, dtype=np.float32)
    )


def test_observed_geometry_H_handler_requires_embedding_directory(
    monkeypatch, tmp_path
) -> None:
    recorded = _patch_observed_geometry_io(monkeypatch, representation="H")
    with pytest.raises(ValueError, match="require an embedding directory"):
        cli.main(_observed_geometry_argv(tmp_path / "geometry", embedding=False))
    assert "output_directory" not in recorded


def test_observed_geometry_AH_handler_concatenates_blocks(
    monkeypatch, tmp_path
) -> None:
    recorded = _patch_observed_geometry_io(monkeypatch, representation="AH")
    status = cli.main(
        _observed_geometry_argv(tmp_path / "geometry", embedding=True)
    )
    assert status == 0
    expected = np.array(
        [
            [1.0, 2.0, 0.5, 0.5],
            [2.0, 3.0, 0.5, 0.5],
            [3.0, 4.0, 0.5, 0.5],
        ]
    )
    np.testing.assert_allclose(recorded["features"], expected)
