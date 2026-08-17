#!/usr/bin/env python
"""Render the multi-series crowd Q1 layer trajectory from an aggregate summary.

The figure is drawn exclusively from the JSON summary published by
``run_crowd_layerwise.py``; the run tree, embeddings, and labels are never
consulted, and no research result values are hard-coded here.

Usage example::

    python scripts/plot_crowd_layerwise.py \\
        --summary runs/crowd-layerwise/summary.json \\
        --output runs/crowd-layerwise/trajectory.pdf \\
        --metric oof_macro_f1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_crowd_layerwise import load_layerwise_summary


METRIC_LABELS = {
    "oof_macro_f1": "OOF macro-F1",
    "oof_log_loss_bits": "OOF log loss (bits/item)",
}


def build_figure(summary: Mapping[str, Any], *, metric: str = "oof_macro_f1"):
    """Build the layer-trajectory figure from a validated summary mapping."""

    if metric not in METRIC_LABELS:
        raise ValueError(f"metric must be one of {sorted(METRIC_LABELS)}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["rows"]
    declared = summary.get("series")
    series_order = (
        [str(name) for name in declared]
        if isinstance(declared, list) and declared
        else sorted({str(row["series"]) for row in rows})
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for name in series_order:
        points = sorted(
            (int(row["layer"]), float(row[metric]))
            for row in rows
            if row["series"] == name
        )
        if not points:
            raise ValueError(f"summary contains no rows for series {name!r}")
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=4,
            linewidth=1.25,
            label=name,
        )
    ax.set_xticks(sorted({int(row["layer"]) for row in rows}))
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(
        "crowd Q1 layer trajectory "
        f"({summary['pooling']} pooling, {summary['selection_metric']} selection)"
    )
    ax.legend(fontsize=8)
    ax.grid(color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_crowd_layerwise",
        description=(
            "Render the multi-series crowd Q1 layer trajectory from a "
            "run_crowd_layerwise aggregate summary (clean-room reconstruction)."
        ),
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_LABELS),
        default="oof_macro_f1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = load_layerwise_summary(arguments.summary)
    figure = build_figure(summary, metric=arguments.metric)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
