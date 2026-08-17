"""Paired group-bootstrap contrasts for decoder-ladder OOF artifacts.

Computes itemwise log-loss (bits) deltas between decoder pairs on the same
items, with 2,000-resample writer/duplicate-component group bootstrap 95%
intervals, matching the paper's conditional-gain methodology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_emotion_spaces.crowd_data import CROWD_EMOTIONS
from frozen_emotion_spaces.metrics import (
    multiclass_itemwise_log_loss_bits,
    paired_group_bootstrap_delta,
)

CONTRASTS = (
    ("D4", "D3"),
    ("D4", "D1"),
    ("D4c", "D4"),
    ("D4d", "D4"),
    ("D4d", "D3"),
    ("D4c", "D1"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", type=Path, required=True, action="append")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    outer = pd.read_csv(Path(args.splits) / "crowd_full_outer.csv", dtype=str)
    group_by_item = dict(zip(outer["item_id"], outer["group_id"]))

    results: dict[str, dict[str, object]] = {}
    for ladder_dir in args.ladder:
        metadata = json.loads((ladder_dir / "metadata.json").read_text())
        space = str(metadata["space"])
        oof = pd.read_parquet(ladder_dir / "oof.parquet")
        per_decoder: dict[str, pd.DataFrame] = {}
        for decoder, frame in oof.groupby("decoder"):
            frame = frame.sort_values("item_id", kind="stable").reset_index(drop=True)
            per_decoder[str(decoder)] = frame
        itemwise = {
            decoder: multiclass_itemwise_log_loss_bits(frame, labels=CROWD_EMOTIONS)
            for decoder, frame in per_decoder.items()
        }
        reference = next(iter(per_decoder.values()))
        item_ids = reference["item_id"].astype(str).to_numpy()
        groups = np.array([group_by_item[i] for i in item_ids])
        space_results: dict[str, object] = {}
        for high, low in CONTRASTS:
            if high not in itemwise or low not in itemwise:
                continue
            delta = paired_group_bootstrap_delta(
                itemwise[high],
                itemwise[low],
                groups,
                item_ids_a=item_ids,
                item_ids_b=item_ids,
                n_bootstrap=args.n_bootstrap,
            )
            space_results[f"{high}_minus_{low}"] = {
                "delta_bits": delta.observed_delta,
                "ci95": [delta.ci_low, delta.ci_high],
            }
        results[space] = space_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
