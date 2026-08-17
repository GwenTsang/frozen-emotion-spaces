#!/usr/bin/env python
"""Render the EmoTwiCS macro-F1 / macro-AP layer trajectory from a summary CSV.

The only data source is the aggregate produced by::

    frozen-emotion-spaces summarize-emotwics-layers \\
        --run runs/emotwics/<model>/<pooling>/layer-0 ... --output summary.csv

No metric values are hard-coded in this script; the figure is a pure function
of the validated per-layer artifacts behind the summary.  A horizontal
reference line (for example an independently computed TF-IDF macro-F1) is
drawn only when the caller explicitly passes ``--reference-value``; nothing is
fabricated here.  All rendered output is labelled as a clean-room
reconstruction.

Usage::

    python scripts/plot_emotwics_layer_trajectory.py \\
        --summary trajectories/xlm-roberta-base_mean_log-loss.csv \\
        --output latex_paper/results/figures/layerwise_emotwics.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("layer", "macro_f1", "macro_ap")
PROVENANCE_COLUMNS = ("run_format", "dataset", "model_key", "selection_metric")


def load_summary(summary_path: str | Path) -> pd.DataFrame:
    """Read and validate one ``summarize-emotwics-layers`` aggregate CSV."""

    path = Path(summary_path)
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"EmoTwiCS layer summary is unreadable: {path}") from error
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"EmoTwiCS layer summary is missing columns: {missing}")
    if frame.empty:
        raise ValueError("EmoTwiCS layer summary contains no layers")
    if frame["layer"].duplicated().any():
        raise ValueError("EmoTwiCS layer summary contains duplicate layers")
    for column in ("macro_f1", "macro_ap"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"EmoTwiCS layer summary has invalid {column} values")
        frame[column] = values
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    if "run_format" in frame.columns and not frame["run_format"].astype(
        str
    ).str.contains("reconstruction").all():
        raise ValueError(
            "EmoTwiCS layer summary rows are not labelled as reconstructions"
        )
    return frame.sort_values("layer", kind="stable").reset_index(drop=True)


def render(
    frame: pd.DataFrame,
    output: str | Path,
    *,
    reference_value: float | None = None,
    reference_label: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Draw the trajectory figure and return the plotted data for inspection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (reference_value is None) != (reference_label is None):
        raise ValueError(
            "reference_value and reference_label must be supplied together"
        )
    if reference_value is not None and not 0.0 <= reference_value <= 1.0:
        raise ValueError("reference_value must lie in [0, 1]")

    layers = frame["layer"].to_numpy(dtype=int)
    macro_f1 = frame["macro_f1"].to_numpy(dtype=float)
    macro_ap = frame["macro_ap"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        layers,
        macro_f1,
        marker="o",
        markersize=4,
        linewidth=1.25,
        color="#1f77b4",
        label="macro-F1",
    )
    ax.plot(
        layers,
        macro_ap,
        marker="s",
        markersize=4,
        linewidth=1.25,
        color="#d62728",
        label="macro-AP",
    )
    if reference_value is not None:
        ax.axhline(
            reference_value,
            color="black",
            linestyle="--",
            linewidth=0.9,
            label=str(reference_label),
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(layers)
    ax.grid(color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8)
    model = frame["model_key"].iloc[0] if "model_key" in frame.columns else "unknown"
    metric = (
        frame["selection_metric"].iloc[0]
        if "selection_metric" in frame.columns
        else "unknown"
    )
    ax.set_title(
        title
        or f"EmoTwiCS layerwise probes ({model}, inner selection: {metric})"
    )
    fig.text(
        0.01,
        0.01,
        "clean-room reconstruction; values read from the validated run aggregate",
        fontsize=7,
        color="0.45",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return {
        "layer": layers,
        "macro_f1": macro_f1,
        "macro_ap": macro_ap,
        "reference_value": reference_value,
        "output": destination,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_emotwics_layer_trajectory",
        description=(
            "Render the EmoTwiCS macro-F1/macro-AP layer trajectory from the "
            "summarize-emotwics-layers aggregate (clean-room reconstruction)."
        ),
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-value",
        type=float,
        default=None,
        help="optional horizontal reference level (e.g. an independently "
        "computed TF-IDF macro-F1); never assumed by default",
    )
    parser.add_argument("--reference-label", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    frame = load_summary(arguments.summary)
    plotted = render(
        frame,
        arguments.output,
        reference_value=arguments.reference_value,
        reference_label=arguments.reference_label,
        title=arguments.title,
    )
    print(
        f"EmoTwiCS layer trajectory written: layers={plotted['layer'].tolist()} "
        f"output={plotted['output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
