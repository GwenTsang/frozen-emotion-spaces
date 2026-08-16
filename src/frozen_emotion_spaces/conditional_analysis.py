"""Atomic summary of reconstructed A/H/AH conditional-information runs."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .crowd_data import CROWD_EMOTIONS
from .experiment_a import _sha256_file
from .experiment_c import validate_crowd_representation_probe
from .metrics import (
    multiclass_itemwise_log_loss_bits,
    paired_group_bootstrap_delta,
    reconstruct_multiclass_metrics,
)


ANALYSIS_FORMAT = "frozen-emotion-spaces-conditional-analysis-reconstruction-v1"
ANALYSIS_FILES = (
    "metrics.parquet",
    "bootstrap_H_minus_AH.json",
    "bootstrap_H_minus_AH.npy",
    "metadata.json",
)


@dataclass(frozen=True)
class ConditionalAnalysisArtifact:
    directory: Path
    metadata: dict[str, Any]
    metrics: pd.DataFrame
    bootstrap: dict[str, Any]


def write_conditional_analysis(
    output_directory: str | Path,
    *,
    A_run: str | Path,
    H_run: str | Path,
    AH_run: str | Path,
    labels: Sequence[str] = CROWD_EMOTIONS,
    n_bootstrap: int = 2000,
    seed: int = 20240804,
) -> ConditionalAnalysisArtifact:
    """Summarize OOF metrics and bootstrap H-minus-AH log-loss gain."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite conditional analysis: {output}")
    names = tuple(str(value) for value in labels)
    artifacts = {
        "A": validate_crowd_representation_probe(A_run),
        "H": validate_crowd_representation_probe(H_run),
        "AH": validate_crowd_representation_probe(AH_run),
    }
    for expected, artifact in artifacts.items():
        if artifact.metadata["representation"] != expected:
            raise ValueError(f"{expected}_run has the wrong representation identity")
        if tuple(artifact.metadata["class_names"]) != names:
            raise ValueError("conditional runs disagree with the requested class axis")
    reference = artifacts["H"]
    for name, artifact in artifacts.items():
        for field in (
            "dataset", "target", "n_items", "class_names",
            "ordered_item_target_sha256", "outer_split_sha256",
            "inner_split_sha256", "selection_metric", "C_grid", "class_weight",
        ):
            if artifact.metadata[field] != reference.metadata[field]:
                raise ValueError(f"conditional run {name} disagrees on {field}")
    for field in ("appraisal_matrix_sha256", "appraisal_names"):
        if artifacts["A"].metadata[field] != artifacts["AH"].metadata[field]:
            raise ValueError(f"conditional A/AH runs disagree on {field}")
    for field in (
        "embedding_artifact_format", "embedding_model_key", "embedding_revision",
        "embedding_mode", "embedding_text_variant", "embedding_metadata_sha256",
        "embedding_item_text_pairs_sha256", "embedding_layer_sha256", "layer",
        "pooling",
    ):
        if artifacts["H"].metadata[field] != artifacts["AH"].metadata[field]:
            raise ValueError(f"conditional H/AH runs disagree on {field}")

    aligned: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for name, artifact in artifacts.items():
        frame = artifact.oof.sort_values("item_id", kind="stable").reset_index(drop=True)
        aligned[name] = frame
        overall = reconstruct_multiclass_metrics(frame, labels=names).overall.to_dict()
        rows.append({"representation": name, **overall})
    for name in ("A", "AH"):
        for column in ("item_id", "group_id", "outer_fold", "y_true"):
            if not aligned[name][column].equals(aligned["H"][column]):
                raise ValueError(f"conditional OOF tables disagree on {column}")

    H_loss = multiclass_itemwise_log_loss_bits(aligned["H"], labels=names)
    AH_loss = multiclass_itemwise_log_loss_bits(aligned["AH"], labels=names)
    result = paired_group_bootstrap_delta(
        H_loss,
        AH_loss,
        aligned["H"]["group_id"],
        item_ids_a=aligned["H"]["item_id"],
        item_ids_b=aligned["AH"]["item_id"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    bootstrap = asdict(result)
    samples = np.asarray(bootstrap.pop("samples"), dtype=np.float64)
    metrics = pd.DataFrame(rows).sort_values("representation", kind="stable")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        metric_path = temporary / "metrics.parquet"
        bootstrap_path = temporary / "bootstrap_H_minus_AH.json"
        sample_path = temporary / "bootstrap_H_minus_AH.npy"
        metrics.to_parquet(metric_path, index=False, engine="pyarrow", compression="zstd")
        bootstrap_path.write_text(
            json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.save(sample_path, samples, allow_pickle=False)
        file_records = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (metric_path, bootstrap_path, sample_path)
        }
        metadata = {
            "analysis_format": ANALYSIS_FORMAT,
            "status": "new_replication_not_historical_recovery",
            "delta_direction": "mean_item_log_loss_bits(H)-mean_item_log_loss_bits(AH)",
            "class_names": list(names),
            "n_items": int(reference.metadata["n_items"]),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "input_run_metadata_sha256": {
                name: _sha256_file(artifact.directory / "metadata.json")
                for name, artifact in artifacts.items()
            },
            "implementation_sha256": {
                "conditional_analysis.py": _sha256_file(Path(__file__)),
                "metrics.py": _sha256_file(Path(__file__).with_name("metrics.py")),
            },
            "files": file_records,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_conditional_analysis(output)


def validate_conditional_analysis(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> ConditionalAnalysisArtifact:
    root = Path(directory)
    missing = [name for name in ANALYSIS_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"partial conditional analysis; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        bootstrap = json.loads(
            (root / "bootstrap_H_minus_AH.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("conditional analysis metadata is unreadable") from error
    if metadata.get("analysis_format") != ANALYSIS_FORMAT:
        raise ValueError("unknown conditional analysis format")
    if (
        metadata.get("status") != "new_replication_not_historical_recovery"
        or metadata.get("delta_direction")
        != "mean_item_log_loss_bits(H)-mean_item_log_loss_bits(AH)"
    ):
        raise ValueError("conditional analysis status or delta direction is invalid")
    class_names = metadata.get("class_names")
    run_hashes = metadata.get("input_run_metadata_sha256")
    if (
        not isinstance(class_names, list)
        or not class_names
        or len(set(class_names)) != len(class_names)
        or not isinstance(run_hashes, Mapping)
        or set(run_hashes) != {"A", "H", "AH"}
        or any(not _is_sha256(value) for value in run_hashes.values())
    ):
        raise ValueError("conditional analysis identity metadata is invalid")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("conditional analysis lacks file records")
    for filename in ANALYSIS_FILES[:-1]:
        record = files.get(filename)
        path = root / filename
        if not isinstance(record, Mapping) or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"conditional analysis file size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"conditional analysis file hash mismatch: {filename}")
    try:
        metrics = pd.read_parquet(root / "metrics.parquet", engine="pyarrow")
        samples = np.load(root / "bootstrap_H_minus_AH.npy", allow_pickle=False)
    except Exception as error:
        raise ValueError("conditional analysis numeric files are unreadable") from error
    required_metrics = {
        "representation", "n_items", "n_labels", "accuracy", "macro_f1",
        "macro_ap", "log_loss_bits", "brier", "ece",
    }
    if (
        not required_metrics.issubset(metrics)
        or len(metrics) != 3
        or metrics["representation"].duplicated().any()
        or set(metrics["representation"]) != {"A", "H", "AH"}
    ):
        raise ValueError("conditional metrics representation axis is invalid")
    metric_numeric = metrics[list(required_metrics - {"representation"})].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(metric_numeric.to_numpy()).all():
        raise ValueError("conditional metrics contain non-finite values")
    n_items = int(metadata.get("n_items", 0))
    if (
        n_items <= 0
        or int(metadata.get("n_bootstrap", 0)) <= 0
        or not (metric_numeric["n_items"].astype(int) == n_items).all()
        or not (metric_numeric["n_labels"].astype(int) == len(class_names)).all()
    ):
        raise ValueError("conditional metrics counts disagree with metadata")
    for column in ("accuracy", "macro_f1", "macro_ap", "ece"):
        if (
            (metric_numeric[column] < -1e-12)
            | (metric_numeric[column] > 1.0 + 1e-12)
        ).any():
            raise ValueError(f"conditional metric {column} is outside [0, 1]")
    if (metric_numeric[["log_loss_bits", "brier"]] < 0).any().any():
        raise ValueError("conditional loss metrics must be non-negative")
    required_bootstrap = {
        "observed_a", "observed_b", "observed_delta", "ci_low", "ci_high",
        "standard_error", "confidence_level", "n_bootstrap", "n_groups", "seed",
    }
    if set(bootstrap) != required_bootstrap:
        raise ValueError("conditional bootstrap schema is invalid")
    if (
        int(bootstrap["n_bootstrap"]) != int(metadata["n_bootstrap"])
        or int(bootstrap["seed"]) != int(metadata["seed"])
        or int(bootstrap["n_bootstrap"]) <= 0
        or int(bootstrap["n_groups"]) <= 0
        or int(bootstrap["n_groups"]) > n_items
        or not 0 < float(bootstrap["confidence_level"]) < 1
    ):
        raise ValueError("conditional bootstrap metadata is inconsistent")
    bootstrap_numeric = np.asarray(
        [
            bootstrap["observed_a"], bootstrap["observed_b"],
            bootstrap["observed_delta"], bootstrap["ci_low"],
            bootstrap["ci_high"], bootstrap["standard_error"],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(bootstrap_numeric).all() or bootstrap["standard_error"] < 0:
        raise ValueError("conditional bootstrap summary is invalid")
    if not np.isclose(
        bootstrap["observed_delta"],
        bootstrap["observed_a"] - bootstrap["observed_b"],
    ):
        raise ValueError("conditional bootstrap delta is arithmetically inconsistent")
    metric_by_representation = metrics.set_index("representation")
    if not np.allclose(
        [bootstrap["observed_a"], bootstrap["observed_b"]],
        [
            metric_by_representation.at["H", "log_loss_bits"],
            metric_by_representation.at["AH", "log_loss_bits"],
        ],
    ):
        raise ValueError("conditional bootstrap observations disagree with metrics")
    if samples.shape != (int(bootstrap["n_bootstrap"]),) or not np.isfinite(samples).all():
        raise ValueError("conditional bootstrap samples are invalid")
    alpha = (1.0 - float(bootstrap["confidence_level"])) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    if not np.allclose([low, high], [bootstrap["ci_low"], bootstrap["ci_high"]]):
        raise ValueError("conditional bootstrap interval disagrees with samples")
    if not np.isclose(np.std(samples, ddof=1), bootstrap["standard_error"]):
        raise ValueError("conditional bootstrap standard error disagrees with samples")
    return ConditionalAnalysisArtifact(root, dict(metadata), metrics, dict(bootstrap))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "ANALYSIS_FILES",
    "ANALYSIS_FORMAT",
    "ConditionalAnalysisArtifact",
    "validate_conditional_analysis",
    "write_conditional_analysis",
]
