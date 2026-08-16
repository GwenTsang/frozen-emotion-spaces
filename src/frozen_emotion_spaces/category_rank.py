"""Category-level rank stability and association for Experiment C.

The primary conditional gain ``L(H)-L([A;H])`` is an aggregate over
items.  This module tests whether the *ordering* of the 13 emotion
categories by difficulty is preserved between appraisal space ``A``
(21D) and each frozen hidden space ``H``.  For each diagnostic

- ``f1``          — one-vs-rest F1
- ``ap``          — one-vs-rest average precision
- ``bce_bits``    — one-vs-rest binary cross-entropy (bits)
- ``ece``         — one-vs-rest expected calibration error

it reports

1. **Rank stability** per representation, per label, per diagnostic:
   mean / variance / bootstrap CI of the rank (1 = best) and a
   normalised stability ``1 - Var / Var_max`` where ``Var_max=(K²-1)/12``
   is the variance of a uniform rank.

2. **Rank association** between ``A`` and each ``H`` (Spearman ρ and
   Kendall τ with percentile bootstrap CIs).

Uncertainty is a paired *writer-group* (crowd) / *conversation-group*
(EmoTwiCS) bootstrap: groups are resampled with replacement, every
item of a selected group is kept with multiplicity, and the identical
index set is applied to all representations.  This preserves
within-writer dependence and pairing — the same resample is scored by
every decoder.

File layout mirrors :mod:`conditional_analysis` (atomic publish,
``metadata.json`` with SHA-256, ``verify_hashes``).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from .config import SEED
from .crowd_data import CROWD_EMOTIONS
from .experiment_a import _sha256_file
from .experiment_c import validate_crowd_representation_probe
from .metrics import PROBABILITY_CLIP, expected_calibration_error

CATEGORY_RANK_FORMAT = "frozen-emotion-spaces-category-rank-reconstruction-v1"
CATEGORY_RANK_FILES = (
    "category_rank_stability.parquet",
    "category_rank_association.parquet",
    "metadata.json",
)

DIAGNOSTICS: tuple[str, ...] = ("f1", "ap", "bce_bits", "ece")
DIAGNOSTIC_MAP: dict[str, str] = {
    "f1": "classwise_f1",
    "ap": "classwise_ap",
    "bce_bits": "classwise_bce",
    "ece": "classwise_ovr_ece",
}
HIGHER_IS_BETTER: dict[str, bool] = {
    "f1": True,
    "ap": True,
    "bce_bits": False,
    "ece": False,
}


@dataclass(frozen=True)
class CategoryRankArtifact:
    directory: Path
    metadata: dict[str, Any]
    stability: pd.DataFrame
    association: pd.DataFrame


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def write_category_rank_analysis(
    output_directory: str | Path,
    *,
    A_run: str | Path,
    H_runs: Mapping[str, str | Path],
    AH_run: str | Path | None = None,
    labels: Sequence[str] = CROWD_EMOTIONS,
    n_bootstrap: int = 2000,
    seed: int = SEED,
) -> CategoryRankArtifact:
    """Compute bootstrap category-rank stability and A-vs-H association.

    Parameters
    ----------
    output_directory:
        Destination directory (must not exist; published atomically).
    A_run:
        Validated ``A`` representation probe directory.
    H_runs:
        Mapping ``model_key -> H probe directory``.  At least one entry
        is required.  Keys are stored verbatim in ``association.model``.
    AH_run:
        Optional ``[A;H]`` directory.  When supplied its rank stability
        is also reported (no association is computed for it).
    labels:
        Canonical class axis.  Defaults to the 13 crowd-enVENT writer
        labels in source order.
    n_bootstrap:
        Number of writer-group resamples (≥2).
    seed:
        PRNG seed (default ``20240804`` — the locked protocol seed).
    """

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite category-rank analysis: {output}")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    names = tuple(str(v) for v in labels)
    if not names or len(set(names)) != len(names):
        raise ValueError("labels must be non-empty and unique")
    if not H_runs:
        raise ValueError("H_runs must contain at least one model")

    # --- validate probe artifacts ---------------------------------------
    a_art = validate_crowd_representation_probe(A_run)
    if a_art.metadata["representation"] != "A":
        raise ValueError("A_run is not an A representation")
    h_arts: dict[str, Any] = {}
    for key, path in H_runs.items():
        art = validate_crowd_representation_probe(path)
        if art.metadata["representation"] != "H":
            raise ValueError(f"H_runs[{key!r}] is not an H representation")
        h_arts[str(key)] = art
    ah_art = None
    if AH_run is not None:
        ah_art = validate_crowd_representation_probe(AH_run)
        if ah_art.metadata["representation"] != "AH":
            raise ValueError("AH_run is not an AH representation")

    # class axis must agree
    for art in (a_art, *h_arts.values(), *([ah_art] if ah_art else [])):
        if tuple(art.metadata["class_names"]) != names:
            raise ValueError("category-rank runs disagree on class axis")
    # dataset / target / splits must agree (A/H/AH share the same outer/inner)
    ref_meta = a_art.metadata
    for art in (*h_arts.values(), *([ah_art] if ah_art else [])):
        for field in (
            "dataset",
            "target",
            "n_items",
            "ordered_item_target_sha256",
            "outer_split_sha256",
            "inner_split_sha256",
        ):
            if art.metadata.get(field) != ref_meta.get(field):
                raise ValueError(f"category-rank runs disagree on {field}")

    # --- load and align OOF tables -------------------------------------
    def _load_oof(art: Any) -> pd.DataFrame:
        df = art.oof.copy()
        if "item_id" not in df.columns or "group_id" not in df.columns:
            raise ValueError("OOF table lacks item_id/group_id")
        return df.sort_values("item_id", kind="stable").reset_index(drop=True)

    a_oof = _load_oof(a_art)
    h_oofs: dict[str, pd.DataFrame] = {k: _load_oof(v) for k, v in h_arts.items()}
    ah_oof = _load_oof(ah_art) if ah_art is not None else None

    # paired alignment: item_id and group_id must be identical in sorted order
    for key, h_oof in h_oofs.items():
        if not a_oof["item_id"].equals(h_oof["item_id"]):
            raise ValueError(f"H_runs[{key!r}] item_id ordering disagrees with A")
        if not a_oof["group_id"].astype(str).equals(h_oof["group_id"].astype(str)):
            raise ValueError(f"H_runs[{key!r}] group_id disagrees with A")
        if not a_oof["y_true"].astype(str).equals(h_oof["y_true"].astype(str)):
            raise ValueError(f"H_runs[{key!r}] y_true disagrees with A")
    if ah_oof is not None:
        if not a_oof["item_id"].equals(ah_oof["item_id"]):
            raise ValueError("AH_run item_id ordering disagrees with A")
        if not a_oof["group_id"].astype(str).equals(ah_oof["group_id"].astype(str)):
            raise ValueError("AH_run group_id disagrees with A")

    # --- observed classwise vectors ------------------------------------
    def _observed_vectors(oof: pd.DataFrame) -> dict[str, NDArray[np.float64]]:
        return _classwise_vectors_from_oof(oof, labels=names)

    a_obs = _observed_vectors(a_oof)
    h_obs: dict[str, dict[str, NDArray[np.float64]]] = {
        k: _observed_vectors(v) for k, v in h_oofs.items()
    }
    ah_obs = _observed_vectors(ah_oof) if ah_oof is not None else None

    # --- bootstrap preparation -----------------------------------------
    uniq_groups = np.array(sorted(pd.unique(a_oof["group_id"].astype(str))), dtype=object)
    n_groups = int(uniq_groups.size)
    if n_groups < 2:
        raise ValueError("paired group bootstrap requires at least two groups")
    n_labels = len(names)

    def _group_indices(oof: pd.DataFrame) -> dict[str, NDArray[np.int64]]:
        g = oof["group_id"].astype(str).to_numpy()
        return {ug: np.flatnonzero(g == ug) for ug in uniq_groups}

    group_map: dict[str, dict[str, NDArray[np.int64]]] = {"a_writer": _group_indices(a_oof)}
    for k, oof in h_oofs.items():
        group_map[k] = _group_indices(oof)
    if ah_oof is not None:
        group_map["AH"] = _group_indices(ah_oof)

    stability_reps: list[str] = ["a_writer", *h_oofs.keys()]
    if ah_oof is not None:
        stability_reps.append("AH")
    oof_by_rep: dict[str, pd.DataFrame] = {"a_writer": a_oof, **h_oofs}
    if ah_oof is not None:
        oof_by_rep["AH"] = ah_oof
    obs_by_rep: dict[str, dict[str, NDArray[np.float64]]] = {"a_writer": a_obs, **h_obs}
    if ah_obs is not None:
        obs_by_rep["AH"] = ah_obs

    rng = np.random.default_rng(int(seed))
    picks = rng.integers(0, n_groups, size=(n_bootstrap, n_groups))

    rank_boot: dict[str, dict[str, NDArray[np.float64]]] = {
        rep: {d: np.empty((n_bootstrap, n_labels), dtype=np.float64) for d in DIAGNOSTICS}
        for rep in stability_reps
    }
    assoc_boot: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {
        m: {
            d: {
                "spearman": np.empty(n_bootstrap, dtype=np.float64),
                "kendall": np.empty(n_bootstrap, dtype=np.float64),
            }
            for d in DIAGNOSTICS
        }
        for m in h_oofs
    }

    # --- bootstrap loop ------------------------------------------------
    for b in range(n_bootstrap):
        sampled_groups = uniq_groups[picks[b]]
        vecs: dict[str, dict[str, NDArray[np.float64]]] = {}
        for rep in stability_reps:
            dmap = group_map[rep]
            idx = np.concatenate([dmap[g] for g in sampled_groups])
            slice_df = oof_by_rep[rep].iloc[idx]
            vd = _classwise_vectors_from_oof(slice_df, labels=names, allow_duplicates=True)
            vecs[rep] = vd
            for d in DIAGNOSTICS:
                vals = vd[d]
                hib = HIGHER_IS_BETTER[d]
                r = _rank_values(vals, higher_is_better=hib)
                rank_boot[rep][d][b] = r
        for m in h_oofs:
            for d in DIAGNOSTICS:
                av = vecs["a_writer"][d]
                hv = vecs[m][d]
                try:
                    s = float(stats.spearmanr(av, hv).statistic)
                except Exception:
                    s = float("nan")
                try:
                    k = float(stats.kendalltau(av, hv).statistic)
                except Exception:
                    k = float("nan")
                assoc_boot[m][d]["spearman"][b] = s
                assoc_boot[m][d]["kendall"][b] = k

    # --- rank stability summary -----------------------------------------
    stability_rows: list[dict[str, Any]] = []
    for rep in stability_reps:
        for d in DIAGNOSTICS:
            mat = rank_boot[rep][d]
            obs_vec = obs_by_rep[rep][d]
            obs_rank = _rank_values(obs_vec, higher_is_better=HIGHER_IS_BETTER[d])
            max_var = (n_labels**2 - 1) / 12.0 if n_labels > 1 else 1.0
            for li, lab in enumerate(names):
                boot_ranks = mat[:, li]
                mean_rank = float(boot_ranks.mean())
                var_rank = float(boot_ranks.var(ddof=1) if n_bootstrap > 1 else 0.0)
                std_rank = float(boot_ranks.std(ddof=1) if n_bootstrap > 1 else 0.0)
                lo, hi = float(np.quantile(boot_ranks, 0.025)), float(np.quantile(boot_ranks, 0.975))
                stability = 1.0 - var_rank / max_var if max_var > 0 else 1.0
                stability_rows.append(
                    {
                        "representation": rep,
                        "diagnostic": DIAGNOSTIC_MAP.get(d, d),
                        "diagnostic_raw": d,
                        "label": lab,
                        "n_bootstrap": int(n_bootstrap),
                        "n_groups": int(n_groups),
                        "observed_rank": float(obs_rank[li]),
                        "mean_bootstrap_rank": float(mean_rank),
                        "rank_variance": float(var_rank),
                        "rank_std": float(std_rank),
                        "rank_ci_low": float(lo),
                        "rank_ci_high": float(hi),
                        "stability": float(stability),
                    }
                )

    # --- association summary -------------------------------------------
    assoc_rows: list[dict[str, Any]] = []
    for m in h_oofs:
        for d in DIAGNOSTICS:
            av = a_obs[d]
            hv = h_obs[m][d]
            try:
                spear = float(stats.spearmanr(av, hv).statistic)
            except Exception:
                spear = float("nan")
            try:
                kend = float(stats.kendalltau(av, hv).statistic)
            except Exception:
                kend = float("nan")
            boot_s = assoc_boot[m][d]["spearman"]
            boot_k = assoc_boot[m][d]["kendall"]
            finite_s = boot_s[np.isfinite(boot_s)]
            finite_k = boot_k[np.isfinite(boot_k)]
            if finite_s.size:
                s_lo, s_hi = float(np.quantile(finite_s, 0.025)), float(np.quantile(finite_s, 0.975))
            else:
                s_lo, s_hi = float("nan"), float("nan")
            if finite_k.size:
                k_lo, k_hi = float(np.quantile(finite_k, 0.025)), float(np.quantile(finite_k, 0.975))
            else:
                k_lo, k_hi = float("nan"), float("nan")
            assoc_rows.append(
                {
                    "model": m,
                    "diagnostic": DIAGNOSTIC_MAP.get(d, d),
                    "diagnostic_raw": d,
                    "n_categories": int(n_labels),
                    "n_bootstrap": int(n_bootstrap),
                    "n_groups": int(n_groups),
                    "spearman": float(spear),
                    "spearman_ci_low": float(s_lo),
                    "spearman_ci_high": float(s_hi),
                    "kendall": float(kend),
                    "kendall_ci_low": float(k_lo),
                    "kendall_ci_high": float(k_hi),
                }
            )

    df_stability = pd.DataFrame.from_records(stability_rows)
    df_assoc = pd.DataFrame.from_records(assoc_rows)

    if not df_stability.empty:
        df_stability = df_stability.sort_values(
            ["representation", "diagnostic_raw", "label"], kind="stable"
        ).reset_index(drop=True)
    if not df_assoc.empty:
        df_assoc = df_assoc.sort_values(["model", "diagnostic_raw"], kind="stable").reset_index(drop=True)

    # --- atomic publish ------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent)))
    try:
        stab_path = tmp / "category_rank_stability.parquet"
        assoc_path = tmp / "category_rank_association.parquet"
        df_stability.to_parquet(stab_path, index=False, engine="pyarrow", compression="zstd")
        df_assoc.to_parquet(assoc_path, index=False, engine="pyarrow", compression="zstd")

        files: dict[str, Any] = {
            p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
            for p in (stab_path, assoc_path)
        }
        metadata: dict[str, Any] = {
            "analysis_format": CATEGORY_RANK_FORMAT,
            "status": "new_replication_not_historical_recovery",
            "class_names": list(names),
            "n_items": int(len(a_oof)),
            "n_groups": int(n_groups),
            "n_categories": int(n_labels),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "diagnostics": list(DIAGNOSTICS),
            "diagnostic_map": dict(DIAGNOSTIC_MAP),
            "representations": list(stability_reps),
            "models": sorted(h_oofs.keys()),
            "input_run_metadata_sha256": {
                "A": _sha256_file(a_art.directory / "metadata.json"),
                **{f"H:{k}": _sha256_file(v.directory / "metadata.json") for k, v in h_arts.items()},
                **({"AH": _sha256_file(ah_art.directory / "metadata.json")} if ah_art else {}),
            },
            "implementation_sha256": {
                "category_rank.py": _sha256_file(Path(__file__)),
                "metrics.py": _sha256_file(Path(__file__).with_name("metrics.py")),
                "experiment_c.py": _sha256_file(Path(__file__).with_name("experiment_c.py")),
            },
            "files": files,
        }
        # metadata.json is written once; like conditional_analysis it carries
        # hashes of the data files but not of itself (self-hash is impossible).
        (tmp / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        tmp.rename(output)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise

    return validate_category_rank_analysis(output)


def validate_category_rank_analysis(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> CategoryRankArtifact:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"category-rank analysis not found: {root}")
    for fname in CATEGORY_RANK_FILES:
        if not (root / fname).is_file():
            raise ValueError(f"category-rank analysis missing file: {fname}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError("category-rank metadata is unreadable") from e
    if metadata.get("analysis_format") != CATEGORY_RANK_FORMAT:
        raise ValueError("unknown category-rank analysis format")
    # metadata.json is present and parsed above but, as in conditional_analysis,
    # excluded from hash verification: a file cannot carry its own hash.
    for fname in CATEGORY_RANK_FILES[:-1]:
        rec = metadata.get("files", {}).get(fname)
        p = root / fname
        if not isinstance(rec, Mapping) or p.stat().st_size != rec.get("bytes"):
            raise ValueError(f"category-rank file size mismatch: {fname}")
        if verify_hashes and _sha256_file(p) != rec.get("sha256"):
            raise ValueError(f"category-rank file hash mismatch: {fname}")
    try:
        stability = pd.read_parquet(root / "category_rank_stability.parquet", engine="pyarrow")
        association = pd.read_parquet(root / "category_rank_association.parquet", engine="pyarrow")
    except Exception as e:
        raise ValueError("category-rank parquet is unreadable") from e
    if not {"representation", "diagnostic", "label", "rank_variance", "stability"}.issubset(stability.columns):
        raise ValueError("category-rank stability schema is incomplete")
    if not {"model", "diagnostic", "spearman", "kendall"}.issubset(association.columns):
        raise ValueError("category-rank association schema is incomplete")
    return CategoryRankArtifact(root, dict(metadata), stability, association)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _rank_values(vals: NDArray[np.float64], *, higher_is_better: bool = True) -> NDArray[np.float64]:
    v = np.asarray(vals, dtype=np.float64)
    if higher_is_better:
        v = -v
    return stats.rankdata(v, method="average").astype(np.float64)


def _classwise_vectors_from_oof(
    oof: pd.DataFrame,
    *,
    labels: Sequence[str],
    allow_duplicates: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Return diagnostic vectors (f1/ap/bce/ece) for an OOF slice."""

    names = tuple(str(v) for v in labels)
    if oof.empty:
        raise ValueError("OOF slice is empty")
    prob_cols = [f"prob__{n}" for n in names]
    missing = [c for c in prob_cols if c not in oof.columns]
    if missing:
        raise ValueError(f"OOF slice missing probability columns: {missing}")
    if "y_true" not in oof.columns:
        raise ValueError("OOF slice must contain y_true")

    prob = oof[prob_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    row_sums = prob.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("probability rows must sum to one")

    y_true = oof["y_true"].astype(str).to_numpy()
    label_to_index = {lab: i for i, lab in enumerate(names)}
    unknown = sorted(set(y_true) - set(names))
    if unknown:
        raise ValueError(f"y_true contains unknown labels: {unknown[:3]}")

    if "y_pred" in oof.columns and not oof["y_pred"].isna().any():
        y_pred = oof["y_pred"].astype(str).to_numpy()
    else:
        pred_idx = prob.argmax(axis=1)
        y_pred = np.array([names[i] for i in pred_idx], dtype=str)

    n = len(y_true)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(names),
        zero_division=0,
    )

    out: dict[str, NDArray[np.float64]] = {}
    out["f1"] = np.array([float(v) for v in f1], dtype=np.float64)

    true_idx = np.array([label_to_index[v] for v in y_true], dtype=int)
    ap_vec = np.empty(len(names), dtype=np.float64)
    bce_vec = np.empty(len(names), dtype=np.float64)
    ece_vec = np.empty(len(names), dtype=np.float64)
    for li in range(len(names)):
        bin_true = (true_idx == li).astype(int)
        bin_prob = prob[:, li]
        if int(bin_true.sum()) == 0:
            ap_vec[li] = 0.0
        else:
            ap_vec[li] = float(average_precision_score(bin_true, bin_prob))
        p = np.clip(bin_prob, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
        bce_vec[li] = float(-np.mean(bin_true * np.log2(p) + (1 - bin_true) * np.log2(1 - p)))
        ece_vec[li] = float(expected_calibration_error(bin_prob, bin_true.astype(float)))

    out["ap"] = ap_vec
    out["bce_bits"] = bce_vec
    out["ece"] = ece_vec
    return out


__all__ = [
    "CATEGORY_RANK_FILES",
    "CATEGORY_RANK_FORMAT",
    "CategoryRankArtifact",
    "validate_category_rank_analysis",
    "write_category_rank_analysis",
]
