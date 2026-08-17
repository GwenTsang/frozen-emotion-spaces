"""Driver for the crowd-enVENT Q2 rerun suite (clean-room reconstruction).

Wires the preserved archive and split lock into the ``crowd_q2`` library API:

- ``oof``: full-OOF writer-target A/H/AH triplets per encoder/layer, including
  train-fold appraisal PCA variants (``run_q2_batch``);
- ``external``: sealed external-test probes (``run_q2_external_probe``), with
  excluded test writers/duplicates bound into artifact metadata;
- ``reader``: writer-appraisal to reader-majority-target cross-rater probes
  (``run_q2_reader_probe``);
- ``suite``: build the immutable Q2 suite manifest with paired H-minus-AH
  contrasts (``build_q2_suite``).

Every child run is resumable: completed compatible artifacts are validated and
skipped, never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frozen_emotion_spaces.config import MODEL_SPECS, embedding_artifact_directory  # noqa: E402
from frozen_emotion_spaces.crowd_data import APPRAISAL_NAMES, build_crowd_manifests  # noqa: E402
from frozen_emotion_spaces.crowd_q2 import (  # noqa: E402
    build_q2_suite,
    external_role_partition,
    run_q2_batch,
    run_q2_external_probe,
    run_q2_reader_probe,
    summarize_q2_suite,
)
from frozen_emotion_spaces.splits import read_split_bundle  # noqa: E402


def _parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            layers.extend(range(lo, hi + 1))
        else:
            layers.append(int(part))
    if not layers or any(v < 0 for v in layers) or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must be unique non-negative integers")
    return layers


def _parse_dims(value: str) -> list[int]:
    dims = [int(v) for v in value.split(",")]
    if not dims or any(v < 1 for v in dims) or len(set(dims)) != len(dims):
        raise argparse.ArgumentTypeError("pca dimensions must be unique positive integers")
    return dims


def _model_keys(value: str) -> list[str]:
    keys = [v.strip() for v in value.split(",")]
    for key in keys:
        if key not in MODEL_SPECS:
            raise argparse.ArgumentTypeError(f"unknown frozen encoder key: {key!r}")
    return keys


def _generation_index(crowd):
    generation = crowd.generation
    item_ids = generation["item_id"].astype(str).tolist()
    lookup = {v: i for i, v in enumerate(item_ids)}
    return generation, item_ids, lookup


def _embeddings(cache_root: Path, model_key: str) -> Path:
    return embedding_artifact_directory(
        cache_root,
        dataset="crowd",
        model=model_key,
        mode="pretrained",
        text_variant="masked",
    )


def _cmd_oof(args: argparse.Namespace) -> None:
    crowd = build_crowd_manifests(args.archive)
    splits = read_split_bundle(args.splits)
    generation = crowd.generation
    results = run_q2_batch(
        args.output_root,
        appraisals=generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float),
        y=generation["y_writer"].astype(str).tolist(),
        item_ids=generation["item_id"].astype(str).tolist(),
        outer_folds=splits.crowd_full_outer,
        inner_folds=splits.crowd_full_inner,
        target_scale="full_writer",
        rater_role="writer_appraisal_to_writer_target",
        model_keys=args.models,
        layers=args.layers,
        pca_dimensions=args.pca_dimensions,
        embedding_root=args.cache_root,
    )
    print(json.dumps({"runs": len(results), "output_root": str(args.output_root)}))


def _cmd_external(args: argparse.Namespace) -> None:
    crowd = build_crowd_manifests(args.archive)
    splits = read_split_bundle(args.splits)
    generation, _, lookup = _generation_index(crowd)
    partition = external_role_partition(splits.crowd_external)

    X = generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float)
    y_writer = generation["y_writer"].astype(str).tolist()

    def rows(ids):
        idx = np.array([lookup[i] for i in ids])
        return idx

    train_idx = rows(partition.train_ids)
    test_idx = rows(partition.test_ids)
    excluded = partition.excluded_test_writer_ids + partition.excluded_test_duplicate_ids

    ran = 0
    for model_key in args.models:
        emb_dir = _embeddings(args.cache_root, model_key)
        for layer in args.layers:
            for rep, dims in (("A", args.pca_dimensions), ("H", [None]), ("AH", [None])):
                for d in dims:
                    suffix = rep if d is None else f"{rep}-pca{d}"
                    run_q2_external_probe(
                        args.output_root / model_key / f"L{layer}" / suffix,
                        representation=rep,
                        appraisals=X[train_idx],
                        y_train=[y_writer[i] for i in train_idx],
                        item_ids_train=list(partition.train_ids),
                        appraisals_test=X[test_idx],
                        y_test=[y_writer[i] for i in test_idx],
                        item_ids_test=list(partition.test_ids),
                        inner_folds=splits.crowd_external_inner,
                        embedding_directory=None if rep == "A" else emb_dir,
                        layer=None if rep == "A" else layer,
                        pooling=args.pooling,
                        pca_dimension=d,
                        excluded_item_ids=excluded,
                    )
                    ran += 1
    print(json.dumps({"runs": ran, "output_root": str(args.output_root)}))


def _cmd_reader(args: argparse.Namespace) -> None:
    crowd = build_crowd_manifests(args.archive)
    splits = read_split_bundle(args.splits)
    generation, _, lookup = _generation_index(crowd)

    validation = crowd.validation
    reader_ids = validation["item_id"].astype(str).tolist()
    reader_targets = validation["y_reader_majority"].astype(str).tolist()
    split_item_ids = splits.crowd_reader_outer["item_id"].astype(str).tolist()
    if set(reader_ids) != set(split_item_ids):
        raise ValueError("reader split lock and validation items disagree")
    order = np.array([lookup[i] for i in reader_ids])
    writer_appraisals = generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float)[order]

    ran = 0
    for model_key in args.models:
        emb_dir = _embeddings(args.cache_root, model_key)
        for layer in args.layers:
            for rep, dims in (("A", args.pca_dimensions), ("H", [None]), ("AH", [None])):
                for d in dims:
                    suffix = rep if d is None else f"{rep}-pca{d}"
                    run_q2_reader_probe(
                        args.output_root / model_key / f"L{layer}" / suffix,
                        representation=rep,
                        writer_appraisals=writer_appraisals,
                        reader_targets=reader_targets,
                        item_ids=reader_ids,
                        reader_outer_folds=splits.crowd_reader_outer,
                        reader_inner_folds=splits.crowd_reader_inner,
                        embedding_directory=None if rep == "A" else emb_dir,
                        layer=None if rep == "A" else layer,
                        pooling=args.pooling,
                        pca_dimension=d,
                    )
                    ran += 1
    print(json.dumps({"runs": ran, "output_root": str(args.output_root)}))


def _cmd_suite(args: argparse.Namespace) -> None:
    artifact = build_q2_suite(
        args.output,
        runs_root=args.runs_root,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    summary = summarize_q2_suite(artifact.directory)
    print(
        json.dumps(
            {
                "suite": str(artifact.directory),
                "runs": len(artifact.summary),
                "summary_rows": len(summary),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--archive", type=Path, required=True)
        cmd.add_argument("--splits", type=Path, required=True)
        cmd.add_argument("--cache-root", type=Path, required=True)
        cmd.add_argument("--output-root", type=Path, required=True)
        cmd.add_argument("--models", type=_model_keys, default=["roberta-base"], metavar="K1,K2,...")
        cmd.add_argument("--layers", type=_parse_layers, default=[12], metavar="SPEC")
        cmd.add_argument("--pca-dimensions", type=_parse_dims, default=[3, 5, 7, 10, 21])
        cmd.add_argument("--pooling", choices=("mean", "first"), default="mean")

    oof = commands.add_parser("oof", help="full-OOF writer-target A/H/AH triplets")
    add_common(oof)
    oof.set_defaults(handler=_cmd_oof)

    external = commands.add_parser("external", help="sealed external-test probes")
    add_common(external)
    external.set_defaults(handler=_cmd_external)

    reader = commands.add_parser("reader", help="writer-appraisal to reader-target probes")
    add_common(reader)
    reader.set_defaults(handler=_cmd_reader)

    suite = commands.add_parser("suite", help="build immutable suite manifest and contrasts")
    suite.add_argument("--runs-root", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--n-bootstrap", type=int, default=2000)
    suite.add_argument("--seed", type=int, default=20240804)
    suite.set_defaults(handler=_cmd_suite)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
