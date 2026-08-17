#!/usr/bin/env python
"""Run one or all EmoTwiCS Q1 multilabel layer probes from a cached embedding artifact.

This script wraps :func:`run_emotwics_layer_probe` and
:func:`resumable_run_emotwics_layer_probe` so that XLM-R EmoTwiCS Figure-1
reruns are possible from a single cache artifact without modifying the
Makefile or paper.  It is a clean-room reconstruction convenience — it does
not fabricate TF-IDF or historical values.

Usage examples
--------------
Single layer::

    python scripts/run_emotwics_layer_probes.py \\
        --archive datasets/EmoTwiCS_v1.zip \\
        --splits splits \\
        --embedding-directory cache/xlm-roberta-base/original/emotwics \\
        --output runs/emotwics/xlm-roberta-base/mean/layer-0 \\
        --layer 0 --selection-metric log_loss

All layers (resumable — skips completed artifacts)::

    for layer in $(seq 0 12); do
        python scripts/run_emotwics_layer_probes.py \\
            --archive datasets/EmoTwiCS_v1.zip \\
            --splits splits \\
            --embedding-directory cache/xlm-roberta-base/original/emotwics \\
            --output runs/emotwics/xlm-roberta-base/mean/layer-${layer} \\
            --layer ${layer} --selection-metric log_loss
    done

Build trajectory summary from completed layers::

    frozen-emotion-spaces summarize-emotwics-layers \\
        --run runs/emotwics/xlm-roberta-base/mean/layer-0 \\
        --run runs/emotwics/xlm-roberta-base/mean/layer-1 \\
        ... \\
        --output trajectories/xlm-roberta-base_mean_log-loss.csv

Render the Figure-1 trajectory from that aggregate (no hard-coded values)::

    python scripts/plot_emotwics_layer_trajectory.py \\
        --summary trajectories/xlm-roberta-base_mean_log-loss.csv \\
        --output latex_paper/results/figures/layerwise_emotwics.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from frozen_emotion_spaces.emotwics_data import (
    CLUSTER_COLUMNS,
    EMOTION_CLUSTERS,
    build_emotwics_manifest,
)
from frozen_emotion_spaces.experiment_b import (
    resumable_run_emotwics_layer_probe,
    run_emotwics_layer_probe,
)
from frozen_emotion_spaces.probes import DEFAULT_C_GRID, DEFAULT_THRESHOLD_GRID
from frozen_emotion_spaces.splits import read_split_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_emotwics_layer_probes",
        description=(
            "Run one EmoTwiCS Q1 multilabel layer probe from a cached "
            "frozen embedding artifact (clean-room reconstruction)."
        ),
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--embedding-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--pooling", choices=("mean", "first"), default="mean",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("log_loss", "macro_f1"),
        required=True,
    )
    parser.add_argument(
        "--C-grid",
        type=_parse_float_grid,
        default=DEFAULT_C_GRID,
        metavar="C1,C2,...",
        help="comma-separated L2 regularisation strengths (default: built-in grid)",
    )
    parser.add_argument(
        "--threshold-grid",
        type=_parse_float_grid,
        default=DEFAULT_THRESHOLD_GRID,
        metavar="T1,T2,...",
        help="comma-separated thresholds in (0,1) (default: built-in grid)",
    )
    parser.add_argument(
        "--resumable",
        action="store_true",
        default=False,
        help="validate an existing completed artifact instead of re-running",
    )
    return parser


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "grid must be comma-separated numbers"
        ) from error
    if not grid:
        raise argparse.ArgumentTypeError("grid must not be empty")
    return grid


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = build_emotwics_manifest(arguments.archive)
    splits = read_split_bundle(arguments.splits)
    tweets = manifest.tweets
    item_ids = tweets["item_id"].astype(str).tolist()
    label_names = list(EMOTION_CLUSTERS)
    y = tweets[list(CLUSTER_COLUMNS)].to_numpy(dtype=np.int64)
    run_fn = resumable_run_emotwics_layer_probe if arguments.resumable else run_emotwics_layer_probe
    artifact = run_fn(
        arguments.output,
        embedding_directory=arguments.embedding_directory,
        layer=arguments.layer,
        y=y,
        item_ids=item_ids,
        label_names=label_names,
        outer_folds=splits.emotwics_outer,
        inner_folds=splits.emotwics_inner,
        pooling=arguments.pooling,
        C_grid=arguments.C_grid,
        threshold_grid=arguments.threshold_grid,
        selection_metric=arguments.selection_metric,
    )
    print(
        f"EmoTwiCS layer probe complete: "
        f"layer={artifact.metadata['layer']} "
        f"directory={artifact.directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
