"""Narrow command-line entry points for clean-room replication artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import (
    MODEL_SPECS,
    PRIMARY_MAX_LENGTH,
    embedding_artifact_directory,
)
from .crowd_data import APPRAISAL_NAMES, build_crowd_manifests
from .embedding_index import write_embedding_index
from .embeddings import extract_to_artifact, load_embedding_layer
from .emotwics_data import CLUSTER_COLUMNS, EMOTION_CLUSTERS, build_emotwics_manifest
from .experiment_a import run_crowd_layer_probe
from .experiment_b import (
    build_all_layer_summary,
    run_emotwics_layer_probe,
    validate_emotwics_layer_probe,
)
from .experiment_c import (
    run_crowd_representation_probe,
    validate_crowd_representation_probe,
)
from .counterfactual import run_counterfactual_pilot
from .counterfactual_index import write_counterfactual_index
from .counterfactual import validate_counterfactual_pilot
from .counterfactual_observed import write_observed_counterfactual_analysis
from .counterfactual_nulls import run_observed_matched_nulls
from .category_rank import write_category_rank_analysis
from .conditional_analysis import write_conditional_analysis
from .decoder_ladder import run_decoder_ladder
from .observed_geometry import write_observed_geometry_analysis
from .probes import DEFAULT_BLOCK_MULTIPLIER_GRID, DEFAULT_C_GRID, DEFAULT_THRESHOLD_GRID
from .representation_index import write_representation_run_index
from .run_index import write_crowd_run_index
from .splits import read_split_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frozen-emotion-spaces",
        description="Provenance-explicit frozen emotion-space replication",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    crowd = commands.add_parser(
        "extract-crowd",
        help="extract one frozen crowd encoder/text-variant artifact",
    )
    _add_extraction_arguments(crowd)
    crowd.add_argument("--archive", type=Path, required=True)
    crowd.add_argument(
        "--text-variant",
        choices=("masked", "original"),
        default="masked",
    )
    crowd.set_defaults(handler=_extract_crowd)

    emotwics = commands.add_parser(
        "extract-emotwics",
        help="extract one frozen EmoTwiCS tweet artifact",
    )
    _add_extraction_arguments(emotwics)
    emotwics.add_argument("--archive", type=Path, required=True)
    emotwics.set_defaults(handler=_extract_emotwics)

    index = commands.add_parser(
        "index-embeddings",
        help="validate cache artifacts and atomically publish their index",
    )
    index.add_argument("--cache-root", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.set_defaults(handler=_index_embeddings)

    run_index = commands.add_parser(
        "index-crowd-runs",
        help="validate crowd run artifacts and bind their metadata externally",
    )
    run_index.add_argument("--runs-root", type=Path, required=True)
    run_index.add_argument("--output", type=Path, required=True)
    run_index.set_defaults(handler=_index_crowd_runs)

    representation_index = commands.add_parser(
        "index-representation-runs",
        help="validate A/H/AH run artifacts and bind their metadata externally",
    )
    representation_index.add_argument("--runs-root", type=Path, required=True)
    representation_index.add_argument("--output", type=Path, required=True)
    representation_index.set_defaults(handler=_index_representation_runs)

    counterfactual_index = commands.add_parser(
        "index-counterfactual-pilots",
        help="validate prospective pilot artifacts and bind metadata externally",
    )
    counterfactual_index.add_argument("--runs-root", type=Path, required=True)
    counterfactual_index.add_argument("--output", type=Path, required=True)
    counterfactual_index.set_defaults(handler=_index_counterfactual_pilots)

    probe = commands.add_parser(
        "probe-crowd-layer",
        help="run one nested full-OOF crowd layer probe",
    )
    probe.add_argument("--archive", type=Path, required=True)
    probe.add_argument("--splits", type=Path, required=True)
    probe.add_argument("--embedding-directory", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--layer", type=int, required=True)
    probe.add_argument("--pooling", choices=("mean", "first"), default="mean")
    probe.add_argument(
        "--selection-metric",
        choices=("log_loss", "macro_f1"),
        required=True,
    )
    probe.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
    )
    probe.add_argument(
        "--C-grid",
        type=_parse_C_grid,
        default=DEFAULT_C_GRID,
        metavar="C1,C2,...",
    )
    probe.set_defaults(handler=_probe_crowd_layer)

    conditional = commands.add_parser(
        "probe-crowd-representation",
        help="run one nested crowd A/H/AH conditional representation probe",
    )
    conditional.add_argument("--archive", type=Path, required=True)
    conditional.add_argument("--splits", type=Path, required=True)
    conditional.add_argument("--output", type=Path, required=True)
    conditional.add_argument(
        "--representation", choices=("A", "H", "AH"), required=True
    )
    conditional.add_argument("--embedding-directory", type=Path)
    conditional.add_argument("--layer", type=int)
    conditional.add_argument("--pooling", choices=("mean", "first"), default=None)
    conditional.add_argument(
        "--selection-metric",
        choices=("log_loss", "macro_f1"),
        required=True,
    )
    conditional.add_argument(
        "--class-weight", choices=("none", "balanced"), default="none"
    )
    conditional.add_argument(
        "--C-grid",
        type=_parse_C_grid,
        default=DEFAULT_C_GRID,
        metavar="C1,C2,...",
    )
    conditional.add_argument(
        "--block-multiplier-grid",
        type=_parse_positive_grid,
        default=None,
        metavar="M1,M2,...",
    )
    conditional.set_defaults(handler=_probe_crowd_representation)

    counterfactual = commands.add_parser(
        "pilot-contrast-representation",
        help="run a prospective train-defined counterfactual learnability pilot",
    )
    counterfactual.add_argument("--archive", type=Path, required=True)
    counterfactual.add_argument("--splits", type=Path, required=True)
    counterfactual.add_argument("--source-run", type=Path, required=True)
    counterfactual.add_argument("--output", type=Path, required=True)
    counterfactual.add_argument(
        "--space", choices=("A_STANDARDIZED", "H_PCA"), required=True
    )
    counterfactual.add_argument("--embedding-directory", type=Path)
    counterfactual.add_argument("--pca-components", type=_positive_int)
    counterfactual.add_argument("--n-sites", type=_positive_int, default=13)
    counterfactual.add_argument(
        "--n-constellations-per-fold", type=_positive_int, default=20
    )
    counterfactual.add_argument("--n-repetitions", type=_positive_int, default=5)
    counterfactual.add_argument(
        "--max-samples-per-cell", type=_positive_int, default=25
    )
    counterfactual.add_argument(
        "--sampling-scheme",
        choices=("per_cell_capped_items", "fixed_group_budget"),
        default="per_cell_capped_items",
    )
    counterfactual.add_argument("--sample-group-budget", type=_positive_int)
    counterfactual.add_argument("--seed", type=int, default=20240804)
    counterfactual.set_defaults(handler=_pilot_contrast_representation)

    observed_counterfactual = commands.add_parser(
        "analyze-observed-counterfactual",
        help="locate observed class-centroid sites inside one pilot distribution",
    )
    observed_counterfactual.add_argument("--archive", type=Path, required=True)
    observed_counterfactual.add_argument("--splits", type=Path, required=True)
    observed_counterfactual.add_argument("--source-run", type=Path, required=True)
    observed_counterfactual.add_argument("--pilot", type=Path, required=True)
    observed_counterfactual.add_argument("--embedding-directory", type=Path)
    observed_counterfactual.add_argument("--output", type=Path, required=True)
    observed_counterfactual.set_defaults(handler=_analyze_observed_counterfactual)

    matched_nulls = commands.add_parser(
        "compute-matched-nulls",
        help="draw H-CR4 mechanism-matched nulls on outer-training folds only",
    )
    matched_nulls.add_argument("--archive", type=Path, required=True)
    matched_nulls.add_argument("--splits", type=Path, required=True)
    matched_nulls.add_argument("--source-run", type=Path, required=True)
    matched_nulls.add_argument("--output", type=Path, required=True)
    matched_nulls.add_argument(
        "--space", choices=("A_STANDARDIZED", "H_PCA"), required=True
    )
    matched_nulls.add_argument("--embedding-directory", type=Path)
    matched_nulls.add_argument("--pca-components", type=_positive_int)
    matched_nulls.add_argument("--n-draws-per-fold", type=_positive_int, default=1000)
    matched_nulls.add_argument(
        "--max-attempts-per-draw", type=_positive_int, default=100
    )
    matched_nulls.add_argument("--seed", type=int, default=20240804)
    matched_nulls.set_defaults(handler=_compute_matched_nulls)

    conditional_analysis = commands.add_parser(
        "write-conditional-analysis",
        help="summarize reconstructed A/H/AH runs and bootstrap the H-minus-AH gain",
    )
    conditional_analysis.add_argument("--A-run", type=Path, required=True)
    conditional_analysis.add_argument("--H-run", type=Path, required=True)
    conditional_analysis.add_argument("--AH-run", type=Path, required=True)
    conditional_analysis.add_argument("--output", type=Path, required=True)
    conditional_analysis.add_argument(
        "--n-bootstrap", type=_positive_int, default=2000
    )
    conditional_analysis.add_argument("--seed", type=int, default=20240804)
    conditional_analysis.set_defaults(handler=_write_conditional_analysis)

    category_rank = commands.add_parser(
        "write-category-rank-analysis",
        help="bootstrap category rank stability and A-vs-H rank association (Experiment C secondary)",
    )
    category_rank.add_argument("--A-run", type=Path, required=True)
    category_rank.add_argument(
        "--H-run",
        action="append",
        required=True,
        metavar="MODEL=PATH",
        help="one H run per model: MODEL=PATH (repeatable)",
    )
    category_rank.add_argument("--AH-run", type=Path, required=False, default=None)
    category_rank.add_argument("--output", type=Path, required=True)
    category_rank.add_argument("--n-bootstrap", type=_positive_int, default=2000)
    category_rank.add_argument("--seed", type=int, default=20240804)
    category_rank.set_defaults(handler=_write_category_rank_analysis)

    observed_geometry = commands.add_parser(
        "analyze-observed-geometry",
        help="score stored probe geometry on outer-training folds only",
    )
    observed_geometry.add_argument("--archive", type=Path, required=True)
    observed_geometry.add_argument("--splits", type=Path, required=True)
    observed_geometry.add_argument("--source-run", type=Path, required=True)
    observed_geometry.add_argument("--embedding-directory", type=Path)
    observed_geometry.add_argument("--output", type=Path, required=True)
    observed_geometry.set_defaults(handler=_analyze_observed_geometry)

    ladder = commands.add_parser(
        "run-decoder-ladder",
        help="run the D0-D4 geometric decoder ladder and multi-prototype diagnostics",
    )
    ladder.add_argument("--archive", type=Path, required=True)
    ladder.add_argument("--splits", type=Path, required=True)
    ladder.add_argument("--space", choices=("A", "H"), required=True)
    ladder.add_argument("--embedding-directory", type=Path)
    ladder.add_argument("--layer", type=int, default=12)
    ladder.add_argument("--pooling", default="mean")
    ladder.add_argument("--pca-dim", type=int)
    ladder.add_argument(
        "--decoders",
        nargs="+",
        default=None,
        help="subset of decoder names to run (default: all registered decoders)",
    )
    ladder.add_argument("--output", type=Path, required=True)
    ladder.set_defaults(handler=_run_decoder_ladder)

    emotwics_layer = commands.add_parser(
        "probe-emotwics-layer",
        help="run one nested full-OOF EmoTwiCS multilabel layer probe",
    )
    emotwics_layer.add_argument("--archive", type=Path, required=True)
    emotwics_layer.add_argument("--splits", type=Path, required=True)
    emotwics_layer.add_argument("--embedding-directory", type=Path, required=True)
    emotwics_layer.add_argument("--output", type=Path, required=True)
    emotwics_layer.add_argument("--layer", type=int, required=True)
    emotwics_layer.add_argument("--pooling", choices=("mean", "first"), default="mean")
    emotwics_layer.add_argument(
        "--selection-metric",
        choices=("log_loss", "macro_f1"),
        required=True,
    )
    emotwics_layer.add_argument(
        "--C-grid",
        type=_parse_C_grid,
        default=DEFAULT_C_GRID,
        metavar="C1,C2,...",
    )
    emotwics_layer.add_argument(
        "--threshold-grid",
        type=_parse_positive_grid,
        default=DEFAULT_THRESHOLD_GRID,
        metavar="T1,T2,...",
    )
    emotwics_layer.set_defaults(handler=_probe_emotwics_layer)

    emotwics_summary = commands.add_parser(
        "summarize-emotwics-layers",
        help="build machine-readable macro-F1/macro-AP layer trajectory summary",
    )
    emotwics_summary.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        metavar="PATH",
        help="completed EmoTwiCS layer probe directory (repeatable)",
    )
    emotwics_summary.add_argument("--output", type=Path, required=True)
    emotwics_summary.set_defaults(handler=_summarize_emotwics_layers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    arguments.handler(arguments)
    return 0


def _add_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--mode", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--max-length", type=int, default=PRIMARY_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--slow-tokenizer",
        action="store_true",
        help="declared fast/slow replication sensitivity",
    )


def _extract_crowd(arguments: argparse.Namespace) -> None:
    _lock_cache_tokenizer_backend(
        arguments.cache_root,
        is_fast=not arguments.slow_tokenizer,
    )
    manifests = build_crowd_manifests(arguments.archive)
    frame = manifests.generation
    column = "hidden_emo_text" if arguments.text_variant == "masked" else "generated_text"
    output = embedding_artifact_directory(
        arguments.cache_root,
        dataset="crowd",
        model=arguments.model,
        mode=arguments.mode,
        text_variant=arguments.text_variant,
        max_length=arguments.max_length,
    )
    artifact = extract_to_artifact(
        output,
        item_ids=frame["item_id"].tolist(),
        texts=frame[column].tolist(),
        model=arguments.model,
        dataset="crowd",
        text_variant=arguments.text_variant,
        mode=arguments.mode,
        max_length=arguments.max_length,
        batch_size=arguments.batch_size,
        device=arguments.device,
        local_files_only=arguments.local_files_only,
        use_fast_tokenizer=not arguments.slow_tokenizer,
    )
    _print_artifact(artifact.directory, artifact.metadata)


def _extract_emotwics(arguments: argparse.Namespace) -> None:
    _lock_cache_tokenizer_backend(
        arguments.cache_root,
        is_fast=not arguments.slow_tokenizer,
    )
    manifest = build_emotwics_manifest(arguments.archive)
    frame = manifest.tweets
    output = embedding_artifact_directory(
        arguments.cache_root,
        dataset="emotwics",
        model=arguments.model,
        mode=arguments.mode,
        text_variant="tweet",
        max_length=arguments.max_length,
    )
    artifact = extract_to_artifact(
        output,
        item_ids=frame["item_id"].tolist(),
        texts=frame["text"].tolist(),
        model=arguments.model,
        dataset="emotwics",
        text_variant="tweet",
        mode=arguments.mode,
        max_length=arguments.max_length,
        batch_size=arguments.batch_size,
        device=arguments.device,
        local_files_only=arguments.local_files_only,
        use_fast_tokenizer=not arguments.slow_tokenizer,
    )
    _print_artifact(artifact.directory, artifact.metadata)


def _index_embeddings(arguments: argparse.Namespace) -> None:
    rows = write_embedding_index(arguments.output, cache_root=arguments.cache_root)
    print(json.dumps({"index": str(arguments.output), "artifacts": len(rows)}))


def _index_crowd_runs(arguments: argparse.Namespace) -> None:
    rows = write_crowd_run_index(arguments.output, runs_root=arguments.runs_root)
    print(json.dumps({"index": str(arguments.output), "runs": len(rows)}))


def _index_representation_runs(arguments: argparse.Namespace) -> None:
    rows = write_representation_run_index(
        arguments.output, runs_root=arguments.runs_root
    )
    print(json.dumps({"index": str(arguments.output), "runs": len(rows)}))


def _index_counterfactual_pilots(arguments: argparse.Namespace) -> None:
    rows = write_counterfactual_index(
        arguments.output, runs_root=arguments.runs_root
    )
    print(json.dumps({"index": str(arguments.output), "pilots": len(rows)}))


def _probe_crowd_layer(arguments: argparse.Namespace) -> None:
    crowd = build_crowd_manifests(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    generation = crowd.generation
    run = run_crowd_layer_probe(
        arguments.output,
        embedding_directory=arguments.embedding_directory,
        layer=arguments.layer,
        pooling=arguments.pooling,
        y=generation["y_writer"].tolist(),
        item_ids=generation["item_id"].tolist(),
        outer_folds=splits.crowd_full_outer,
        inner_folds=splits.crowd_full_inner,
        C_grid=arguments.C_grid,
        selection_metric=arguments.selection_metric,
        class_weight=None if arguments.class_weight == "none" else "balanced",
    )
    _print_artifact(run.directory, run.metadata)


def _probe_crowd_representation(arguments: argparse.Namespace) -> None:
    if arguments.representation == "A" and arguments.pooling is not None:
        raise ValueError("--pooling is not applicable to representation A")
    if (
        arguments.representation != "AH"
        and arguments.block_multiplier_grid is not None
    ):
        raise ValueError(
            "--block-multiplier-grid is applicable only to representation AH"
        )
    pooling = arguments.pooling or "mean"
    multiplier_grid = (
        arguments.block_multiplier_grid or DEFAULT_BLOCK_MULTIPLIER_GRID
    )
    crowd = build_crowd_manifests(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    generation = crowd.generation
    run = run_crowd_representation_probe(
        arguments.output,
        representation=arguments.representation,
        appraisals=generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float),
        y=generation["y_writer"].tolist(),
        item_ids=generation["item_id"].tolist(),
        outer_folds=splits.crowd_full_outer,
        inner_folds=splits.crowd_full_inner,
        embedding_directory=arguments.embedding_directory,
        layer=arguments.layer,
        pooling=pooling,
        C_grid=arguments.C_grid,
        block_multiplier_grid=multiplier_grid,
        selection_metric=arguments.selection_metric,
        class_weight=None if arguments.class_weight == "none" else "balanced",
    )
    _print_artifact(run.directory, run.metadata)


def _pilot_contrast_representation(arguments: argparse.Namespace) -> None:
    crowd = build_crowd_manifests(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    generation = crowd.generation
    source = validate_crowd_representation_probe(arguments.source_run)
    item_ids = generation["item_id"].astype(str).tolist()
    if arguments.space == "A_STANDARDIZED":
        if source.metadata["representation"] != "A":
            raise ValueError("A_STANDARDIZED requires an A source run")
        if arguments.embedding_directory is not None:
            raise ValueError("A_STANDARDIZED must not declare an embedding directory")
        if arguments.pca_components is not None:
            raise ValueError("A_STANDARDIZED must not declare PCA components")
        features = generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float)
    else:
        if source.metadata["representation"] != "H":
            raise ValueError("H_PCA requires an H source run")
        if arguments.embedding_directory is None or arguments.pca_components is None:
            raise ValueError("H_PCA requires an embedding directory and PCA components")
        features, loaded_ids = load_embedding_layer(
            arguments.embedding_directory,
            layer=int(source.metadata["layer"]),
            pooling=str(source.metadata["pooling"]),
            expected_item_ids=item_ids,
        )
        if loaded_ids.astype(str).tolist() != item_ids:  # pragma: no cover - loader checks
            raise RuntimeError("embedding item order disagrees with Crowd-enVENT")
    pilot = run_counterfactual_pilot(
        arguments.output,
        source_run=source.directory,
        features=features,
        item_ids=item_ids,
        outer_folds=splits.crowd_full_outer,
        space=arguments.space,
        n_sites=arguments.n_sites,
        n_constellations_per_fold=arguments.n_constellations_per_fold,
        n_repetitions=arguments.n_repetitions,
        pca_components=arguments.pca_components,
        max_samples_per_cell=arguments.max_samples_per_cell,
        sampling_scheme=arguments.sampling_scheme,
        sample_group_budget=arguments.sample_group_budget,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {"directory": str(pilot.directory), "format": pilot.metadata["pilot_format"]},
            sort_keys=True,
        )
    )


def _analyze_observed_counterfactual(arguments: argparse.Namespace) -> None:
    crowd = build_crowd_manifests(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    generation = crowd.generation
    source = validate_crowd_representation_probe(arguments.source_run)
    pilot = validate_counterfactual_pilot(arguments.pilot)
    item_ids = generation["item_id"].astype(str).tolist()
    if pilot.metadata["space"] == "A_STANDARDIZED":
        if arguments.embedding_directory is not None:
            raise ValueError("A observed analysis must not declare an embedding")
        features = generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float)
    else:
        if arguments.embedding_directory is None:
            raise ValueError("H observed analysis requires an embedding directory")
        features, _ = load_embedding_layer(
            arguments.embedding_directory,
            layer=int(source.metadata["layer"]),
            pooling=str(source.metadata["pooling"]),
            expected_item_ids=item_ids,
        )
    analysis = write_observed_counterfactual_analysis(
        arguments.output,
        pilot_directory=pilot.directory,
        source_run=source.directory,
        features=features,
        item_ids=item_ids,
        outer_folds=splits.crowd_full_outer,
    )
    print(
        json.dumps(
            {
                "directory": str(analysis.directory),
                "format": analysis.metadata["analysis_format"],
            },
            sort_keys=True,
        )
    )


def _compute_matched_nulls(arguments: argparse.Namespace) -> None:
    _validate_matched_null_arguments(arguments)
    crowd = build_crowd_manifests(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    generation = crowd.generation
    source = validate_crowd_representation_probe(arguments.source_run)
    item_ids = generation["item_id"].astype(str).tolist()
    if arguments.space == "A_STANDARDIZED":
        if source.metadata["representation"] != "A":
            raise ValueError("A_STANDARDIZED requires an A source run")
        features = generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float)
    else:
        if source.metadata["representation"] != "H":
            raise ValueError("H_PCA requires an H source run")
        features, loaded_ids = load_embedding_layer(
            arguments.embedding_directory,
            layer=int(source.metadata["layer"]),
            pooling=str(source.metadata["pooling"]),
            expected_item_ids=item_ids,
        )
        if loaded_ids.astype(str).tolist() != item_ids:  # pragma: no cover - loader checks
            raise RuntimeError("embedding item order disagrees with Crowd-enVENT")
    nulls = run_observed_matched_nulls(
        arguments.output,
        source_run=source.directory,
        features=features,
        item_ids=item_ids,
        outer_folds=splits.crowd_full_outer,
        space=arguments.space,
        pca_components=arguments.pca_components,
        n_draws_per_fold=arguments.n_draws_per_fold,
        seed=arguments.seed,
        max_attempts_per_draw=arguments.max_attempts_per_draw,
    )
    print(
        json.dumps(
            {
                "directory": str(nulls.directory),
                "format": nulls.metadata["null_format"],
            },
            sort_keys=True,
        )
    )


def _write_conditional_analysis(arguments: argparse.Namespace) -> None:
    analysis = write_conditional_analysis(
        arguments.output,
        A_run=arguments.A_run,
        H_run=arguments.H_run,
        AH_run=arguments.AH_run,
        n_bootstrap=arguments.n_bootstrap,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "directory": str(analysis.directory),
                "format": analysis.metadata["analysis_format"],
            },
            sort_keys=True,
        )
    )


def _write_category_rank_analysis(arguments: argparse.Namespace) -> None:
    # parse H-run specs MODEL=PATH
    H_runs: dict[str, Path] = {}
    for spec in arguments.H_run:
        if "=" not in spec:
            raise ValueError(f"--H-run must be MODEL=PATH, got {spec!r}")
        model, path = spec.split("=", 1)
        model = model.strip()
        path = path.strip()
        if not model or not path:
            raise ValueError(f"--H-run must be MODEL=PATH, got {spec!r}")
        if model in H_runs:
            raise ValueError(f"duplicate --H-run model key: {model!r}")
        H_runs[model] = Path(path)
    artifact = write_category_rank_analysis(
        arguments.output,
        A_run=arguments.A_run,
        H_runs=H_runs,
        AH_run=arguments.AH_run,
        n_bootstrap=arguments.n_bootstrap,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "directory": str(artifact.directory),
                "format": artifact.metadata["analysis_format"],
            },
            sort_keys=True,
        )
    )


def _analyze_observed_geometry(arguments: argparse.Namespace) -> None:
    crowd = build_crowd_manifests(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    generation = crowd.generation
    source = validate_crowd_representation_probe(arguments.source_run)
    item_ids = generation["item_id"].astype(str).tolist()
    appraisals = generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float)
    representation = str(source.metadata["representation"])
    if representation == "A":
        if arguments.embedding_directory is not None:
            raise ValueError(
                "A observed geometry must not declare an embedding directory"
            )
        features = appraisals
    else:
        if arguments.embedding_directory is None:
            raise ValueError(
                "H and AH observed geometry require an embedding directory"
            )
        hidden, loaded_ids = load_embedding_layer(
            arguments.embedding_directory,
            layer=int(source.metadata["layer"]),
            pooling=str(source.metadata["pooling"]),
            expected_item_ids=item_ids,
        )
        if loaded_ids.astype(str).tolist() != item_ids:  # pragma: no cover - loader checks
            raise RuntimeError("embedding item order disagrees with Crowd-enVENT")
        if representation == "H":
            features = hidden
        else:
            features = np.concatenate((appraisals, hidden), axis=1)
    analysis = write_observed_geometry_analysis(
        arguments.output,
        run_directory=source.directory,
        features=features,
        item_ids=item_ids,
        outer_folds=splits.crowd_full_outer,
    )
    print(
        json.dumps(
            {
                "directory": str(analysis.directory),
                "format": analysis.metadata["analysis_format"],
            },
            sort_keys=True,
        )
    )


def _run_decoder_ladder(arguments: argparse.Namespace) -> None:
    crowd = build_crowd_manifests(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    generation = crowd.generation
    item_ids = generation["item_id"].astype(str).tolist()
    y = generation["y_writer"].astype(str).tolist()
    outer = splits.crowd_full_outer
    group_lookup = dict(
        zip(
            outer["item_id"].astype(str),
            outer["group_id"].astype(str),
        )
    )
    group_ids = [group_lookup[i] for i in item_ids]
    if arguments.space == "A":
        features = generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float)
    else:
        if arguments.embedding_directory is None:
            raise ValueError("the H ladder requires --embedding-directory")
        hidden, _ = load_embedding_layer(
            arguments.embedding_directory,
            layer=arguments.layer,
            pooling=arguments.pooling,
            expected_item_ids=item_ids,
        )
        features = np.asarray(hidden, dtype=np.float64)
    from .decoder_ladder import DECODERS

    decoders = tuple(arguments.decoders) if arguments.decoders else DECODERS
    artifact = run_decoder_ladder(
        arguments.output,
        space=arguments.space,
        features=features,
        item_ids=item_ids,
        y=y,
        group_ids=group_ids,
        outer_folds=outer,
        inner_folds=splits.crowd_full_inner,
        pca_dim=arguments.pca_dim,
        decoders=decoders,
    )
    _print_artifact(artifact.directory, artifact.metadata)


def _probe_emotwics_layer(arguments: argparse.Namespace) -> None:
    manifest = build_emotwics_manifest(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    tweets = manifest.tweets
    item_ids = tweets["item_id"].astype(str).tolist()
    label_names = list(EMOTION_CLUSTERS)
    y = tweets[list(CLUSTER_COLUMNS)].to_numpy(dtype=np.int64)
    run = run_emotwics_layer_probe(
        arguments.output,
        embedding_directory=arguments.embedding_directory,
        layer=arguments.layer,
        pooling=arguments.pooling,
        y=y,
        item_ids=item_ids,
        label_names=label_names,
        outer_folds=splits.emotwics_outer,
        inner_folds=splits.emotwics_inner,
        C_grid=arguments.C_grid,
        threshold_grid=arguments.threshold_grid,
        selection_metric=arguments.selection_metric,
    )
    _print_artifact(run.directory, run.metadata)


def _summarize_emotwics_layers(arguments: argparse.Namespace) -> None:
    artifacts = [validate_emotwics_layer_probe(path) for path in arguments.run]
    summary = build_all_layer_summary(artifacts)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    summary.layers.to_csv(arguments.output, index=False, lineterminator="\n")
    print(
        json.dumps(
            {"output": str(arguments.output), "layers": len(summary.layers)},
            sort_keys=True,
        )
    )


def _validate_matched_null_arguments(arguments: argparse.Namespace) -> None:
    """Reject incompatible space arguments before reading any corpus artifact."""

    if arguments.space == "A_STANDARDIZED":
        if arguments.embedding_directory is not None:
            raise ValueError("A_STANDARDIZED must not declare an embedding directory")
        if arguments.pca_components is not None:
            raise ValueError("A_STANDARDIZED must not declare PCA components")
    else:
        if arguments.embedding_directory is None:
            raise ValueError("H_PCA requires an embedding directory")
        if arguments.pca_components is None:
            raise ValueError("H_PCA requires PCA components")


def _parse_C_grid(value: str) -> tuple[float, ...]:
    return _parse_positive_grid(value, name="C")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parse_positive_grid(
    value: str,
    *,
    name: str = "grid",
) -> tuple[float, ...]:
    try:
        grid = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} grid must be comma-separated numbers"
        ) from error
    if not grid or any(number <= 0 for number in grid):
        raise argparse.ArgumentTypeError(f"{name} values must be strictly positive")
    return grid


def _print_artifact(directory: Path, metadata: dict) -> None:
    print(
        json.dumps(
            {
                "directory": str(directory),
                "format": metadata.get("artifact_format", metadata.get("run_format")),
            },
            sort_keys=True,
        )
    )


def _lock_cache_tokenizer_backend(cache_root: Path, *, is_fast: bool) -> None:
    """Prevent fast/slow artifacts from sharing an ambiguous cache namespace."""

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "tokenizer_backend.json"
    expected = {"tokenizer_is_fast": bool(is_fast)}
    if marker.exists():
        try:
            observed = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"tokenizer backend marker is unreadable: {marker}") from error
        if observed != expected:
            raise ValueError(
                "cache root is locked to the other tokenizer backend; "
                "use a separate cache root"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tokenizer_backend.tmp-",
        dir=root,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(expected, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, marker)
        except FileExistsError:
            observed = json.loads(marker.read_text(encoding="utf-8"))
            if observed != expected:
                raise ValueError(
                    "cache root was concurrently locked to the other tokenizer backend"
                )
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
