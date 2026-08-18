"""Merge partial decoder-ladder artifacts (split by folds) into one.

Usage:
    python scripts/merge_ladder_folds.py --parts A-part0 A-part1 --output A
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from frozen_emotion_spaces.decoder_ladder import _summarize, RUN_FORMAT, _sha256_file


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge partial ladder artifacts")
    parser.add_argument("--parts", nargs="+", required=True, help="partial artifact dirs")
    parser.add_argument("--output", required=True, help="merged output dir")
    args = parser.parse_args()

    parts = [Path(p) for p in args.parts]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load one reference for metadata template
    ref_meta = json.loads((parts[0] / "metadata.json").read_text())
    ref_sum = json.loads((parts[0] / "summary.json").read_text())

    # Concatenate OOF frames; derive class names from prob__ columns
    oof_frames = []
    for p in parts:
        oof_frames.append(pd.read_parquet(p / "oof.parquet"))
    oof = pd.concat(oof_frames, ignore_index=True).sort_values(
        ["decoder", "item_id"], kind="stable"
    ).reset_index(drop=True)

    class_names = [
        c[len("prob__"):] for c in oof.columns if c.startswith("prob__")
    ]

    # Concatenate selections
    sel_frames = []
    for p in parts:
        sel_frames.append(pd.read_parquet(p / "selections.parquet"))
    selections = pd.concat(sel_frames, ignore_index=True).sort_values(
        ["decoder", "outer_fold"], kind="stable"
    ).reset_index(drop=True)

    # Recompute summary over merged data
    summary = _summarize(oof, selections, class_names=class_names)
    summary["n_groups"] = ref_sum.get("n_groups")

    # Write merged artifact
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        oof_path = staging / "oof.parquet"
        sel_path = staging / "selections.parquet"
        sum_path = staging / "summary.json"
        oof.to_parquet(oof_path, index=False)
        selections.to_parquet(sel_path, index=False)
        sum_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        metadata = {
            "analysis_format": RUN_FORMAT,
            "status": "merged_fold_split",
            "space": ref_meta["space"],
            "pca_dim": ref_meta["pca_dim"],
            "seed": ref_meta["seed"],
            "n_items": int(oof.shape[0]),
            "n_features": ref_meta["n_features"],
            "feature_matrix_sha256": ref_meta["feature_matrix_sha256"],
            "outer_split_sha256": ref_meta["outer_split_sha256"],
            "decoders": sorted(summary["decoders"]),
            "grids": ref_meta["grids"],
            "source_parts": [str(p.name) for p in parts],
            "files": {
                path.name: {
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in (oof_path, sel_path, sum_path)
            },
            "implementation_sha256": ref_meta["implementation_sha256"],
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"merged {len(parts)} parts -> {output}")
    for d, rec in summary["decoders"].items():
        print(f"  {d}: {rec['log_loss_bits']:.3f} bits (n={rec['n_items']})")


if __name__ == "__main__":
    main()
