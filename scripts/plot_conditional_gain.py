"""Render Figure 2 from the six paired estimates reported in Table 3.

The historical item-level OOF probabilities behind Table 3 are not recovered in
this reconstruction.  This script therefore uses only the published point
estimates and 95% group-bootstrap intervals transcribed below; it is a
reproducible rendering of that table, not a new analysis run.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConditionalGain:
    """One Table-3 conditional log-loss estimate and its 95% interval."""

    label: str
    estimate: float
    lower: float
    upper: float


# Table 3, ordered to group full-OOF and external designs in the plot.
TABLE_3_GAINS = (
    ConditionalGain("full OOF: XLM-R-base", 0.46, 0.41, 0.48),
    ConditionalGain("full OOF: RoBERTa-base", 0.28, 0.25, 0.30),
    ConditionalGain("full OOF: DeBERTa-v3-base", 0.34, 0.31, 0.36),
    ConditionalGain("external: XLM-R-base", 0.47, 0.40, 0.55),
    ConditionalGain("external: RoBERTa-base", 0.32, 0.27, 0.38),
    ConditionalGain("external: DeBERTa-v3-base", 0.37, 0.32, 0.44),
)


def render(output: Path) -> None:
    """Write the Table-3 forest plot to ``output``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    estimates = np.array([row.estimate for row in TABLE_3_GAINS])
    lower = np.array([row.lower for row in TABLE_3_GAINS])
    upper = np.array([row.upper for row in TABLE_3_GAINS])
    positions = np.arange(len(TABLE_3_GAINS))

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.25,
        capsize=3,
        markersize=5,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(positions, [row.label for row in TABLE_3_GAINS])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 0.58)
    ax.set_xlabel("Conditional log-loss gain (bits/item)")
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
