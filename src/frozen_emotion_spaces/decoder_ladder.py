"""Geometric decoder ladder D0--D4 and multi-prototype diagnostics.

This module implements the decoder ladder described in the paper's geometric
decoder tests on the preserved crowd-enVENT splits, plus two targeted
multi-prototype diagnostics:

- ``d4_constrained``: one site per class is frozen at the class centroid (the
  D1 solution) and within-class k-means refines only the remaining ``m-1``
  sites.  This separates unsupervised k-means initialization from the number
  of prototypes per se.
- ``d4_discriminative``: sites and per-site power weights are initialized from
  per-class k-means and then refined by L-BFGS directly on the multiclass
  softmax loss, with a log-sum-exp aggregation over the sites of each class.
  For ``m = 1`` the aggregation reduces exactly to a single power-diagram
  score, so D3 is a fitted special case.

None of this code is recovered historical source; every artifact it writes is
labeled as a new replication diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .crowd_data import CROWD_EMOTIONS
from .experiment_a import _dataframe_digest, _sha256_array, _sha256_file


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SEED = 20240804
GAMMA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
M_GRID = (2, 3)
KMEANS_RESTARTS = 5
LBFGS_MAXITER = 500
EPS = 1e-12

RUN_FORMAT = "frozen-emotion-spaces-decoder-ladder-diagnostic-v1"
RUN_FILES = ("oof.parquet", "selections.parquet", "summary.json", "metadata.json")
DECODERS = ("D0", "D1", "D1o", "D2", "D3", "D4", "D4c", "D4d")


# ---------------------------------------------------------------------------
# core decoders
# ---------------------------------------------------------------------------


def encode_labels(y: Sequence[str], class_names: Sequence[str]) -> IntArray:
    """Encode string labels as class indices under a fixed vocabulary."""

    index = {name: i for i, name in enumerate(class_names)}
    try:
        return np.array([index[str(value)] for value in y], dtype=np.int64)
    except KeyError as error:  # pragma: no cover - defensive
        raise ValueError(f"unknown label: {error.args[0]!r}") from error


def _squared_distances(points: FloatArray, sites: FloatArray) -> FloatArray:
    point_norm = np.einsum("ij,ij->i", points, points)[:, None]
    site_norm = np.einsum("ij,ij->i", sites, sites)[None, :]
    squared = point_norm + site_norm - 2.0 * (points @ sites.T)
    return np.maximum(squared, 0.0)


def _softmax(scores: FloatArray) -> FloatArray:
    return np.exp(log_softmax(scores, axis=1))


def _log_loss_bits(proba: FloatArray, y: IntArray) -> float:
    p_true = np.clip(proba[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.mean(np.log2(p_true)))


def d0_proba(n_rows: int, prior: FloatArray) -> FloatArray:
    """Tile the training prior over ``n_rows`` items."""

    return np.tile(prior[None, :], (n_rows, 1))


def fit_prior(y: IntArray, n_classes: int) -> FloatArray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    prior = np.clip(counts / counts.sum(), EPS, 1.0)
    return prior / prior.sum()


def class_centroids(Z: FloatArray, y: IntArray, n_classes: int) -> FloatArray:
    return np.stack([Z[y == k].mean(axis=0) for k in range(n_classes)])


def d1_proba(Z: FloatArray, centroids: FloatArray, gamma: float) -> FloatArray:
    """Nearest-centroid probabilities: softmax over ``-gamma * ||z - mu||^2``."""

    return _softmax(-float(gamma) * _squared_distances(Z, centroids))


def fit_d1o(
    Z: FloatArray, y: IntArray, gamma: float, C: float, *, n_classes: int
) -> tuple[FloatArray, FloatArray]:
    """Fit free power offsets on frozen class-centroid sites (D1o).

    Scores are ``-gamma * ||z - c_k||^2 + omega_k`` with the sites ``c_k``
    fixed at the training class centroids; only the offsets are learned.  For
    fixed ``gamma`` the softmax loss is convex in ``omega``, and the ridge
    penalty pins the otherwise free common-shift gauge of the offsets.
    Returns ``(centroids, omega)``.
    """

    centroids = class_centroids(Z, y, n_classes)
    d2 = _squared_distances(Z, centroids)
    n = Z.shape[0]

    def loss_grad(omega: FloatArray) -> tuple[float, FloatArray]:
        log_p = log_softmax(-float(gamma) * d2 + omega[None, :], axis=1)
        loss = -float(log_p[np.arange(n), y].mean())
        P = np.exp(log_p)
        P[np.arange(n), y] -= 1.0
        reg = 1.0 / (float(C) * n)
        loss += 0.5 * reg * float(omega @ omega)
        return loss, P.mean(axis=0) + reg * omega

    result = minimize(
        loss_grad,
        np.zeros(n_classes, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER},
    )
    return centroids, result.x


def d1o_proba(
    Z: FloatArray, centroids: FloatArray, omega: FloatArray, gamma: float
) -> FloatArray:
    """Fixed-centroid offset probabilities: softmax over ``-gamma * d^2 + omega``."""

    return _softmax(-float(gamma) * _squared_distances(Z, centroids) + omega[None, :])


def _d2_loss_grad(
    flat: FloatArray, Z: FloatArray, y: IntArray, n_classes: int, C: float
) -> tuple[float, FloatArray]:
    """Softmax loss over tied-weight Voronoi scores ``-||z - mu_k||^2``."""

    n, dim = Z.shape
    mu = flat.reshape(n_classes, dim)
    diff = Z[:, None, :] - mu[None, :, :]  # (n, K, d)
    sq = np.einsum("nkd,nkd->nk", diff, diff)
    log_p = log_softmax(-sq, axis=1)
    loss = -float(log_p[np.arange(n), y].mean())
    P = np.exp(log_p)
    P[np.arange(n), y] -= 1.0
    coeff = P / n  # (n, K)
    grad = np.einsum("nk,nkd->kd", coeff, 2.0 * diff)
    reg = 1.0 / (C * n)
    loss += 0.5 * reg * float(np.einsum("kd,kd->", mu, mu))
    grad += reg * mu
    return loss, grad.ravel()


def fit_d2(
    Z: FloatArray, y: IntArray, C: float, *, n_classes: int
) -> FloatArray:
    """Fit a learned ordinary-Voronoi diagram (single site/class, tied weights)."""

    init = class_centroids(Z, y, n_classes).ravel()
    result = minimize(
        _d2_loss_grad,
        init,
        args=(Z, y, n_classes, float(C)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER},
    )
    return result.x.reshape(n_classes, Z.shape[1])


def d2_proba(Z: FloatArray, sites: FloatArray) -> FloatArray:
    return _softmax(-_squared_distances(Z, sites))


def fit_d3(
    Z: FloatArray, y: IntArray, C: float, *, n_classes: int
) -> tuple[FloatArray, FloatArray]:
    """Fit a power diagram as multinomial linear scores via L-BFGS.

    Returns ``(weights, biases)`` of the equivalent linear probe; sites and
    power weights follow from ``p_k = w_k / 2`` and
    ``omega_k = b_k + ||w_k||^2 / 4``.
    """

    n, dim = Z.shape
    init = np.zeros(n_classes * (dim + 1), dtype=np.float64)

    def loss_grad(flat: FloatArray) -> tuple[float, FloatArray]:
        W = flat[: n_classes * dim].reshape(n_classes, dim)
        b = flat[n_classes * dim :]
        S = Z @ W.T + b  # (n, K)
        log_p = log_softmax(S, axis=1)
        loss = -float(log_p[np.arange(n), y].mean())
        P = np.exp(log_p)
        P[np.arange(n), y] -= 1.0
        coeff = P / n
        reg = 1.0 / (C * n)
        loss += 0.5 * reg * (float(np.einsum("kd,kd->", W, W)) + float(b @ b))
        grad_W = coeff.T @ Z + reg * W
        grad_b = coeff.sum(axis=0) + reg * b
        return loss, np.concatenate([grad_W.ravel(), grad_b])

    result = minimize(
        loss_grad, init, jac=True, method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER},
    )
    W = result.x[: n_classes * dim].reshape(n_classes, dim)
    b = result.x[n_classes * dim :]
    return W, b


def d3_proba(Z: FloatArray, W: FloatArray, b: FloatArray) -> FloatArray:
    return _softmax(Z @ W.T + b)


def d3_sites_weights(
    W: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Convert linear scores to power-diagram sites and weights."""

    sites = W / 2.0
    weights = b + np.einsum("kd,kd->k", W, W) / 4.0
    return sites, weights


def fit_class_kmeans(
    Z: FloatArray,
    y: IntArray,
    m: int,
    *,
    n_classes: int,
    seed: int,
    n_restarts: int = KMEANS_RESTARTS,
) -> tuple[FloatArray, FloatArray]:
    """Unsupervised per-class k-means sites, shape ``(K, m, d)`` (D4).

    Returns ``(sites, omega)`` with zero offsets, matching the D4d signature.
    """

    dim = Z.shape[1]
    sites = np.zeros((n_classes, m, dim), dtype=np.float64)
    for k in range(n_classes):
        Zk = Z[y == k]
        km = KMeans(
            n_clusters=m, n_init=n_restarts, random_state=seed + k,
        ).fit(Zk)
        sites[k] = km.cluster_centers_
    omega = np.zeros((n_classes, m), dtype=np.float64)
    return sites, omega


def fit_class_kmeans_constrained(
    Z: FloatArray,
    y: IntArray,
    m: int,
    *,
    n_classes: int,
    seed: int,
    n_restarts: int = KMEANS_RESTARTS,
    max_iter: int = 100,
) -> tuple[FloatArray, FloatArray]:
    """Per-class k-means with site 0 frozen at the class centroid (D4c).

    The frozen centroid is always one of the ``m`` sites; Lloyd iterations
    update only the remaining ``m - 1`` centers.  The best of ``n_restarts``
    random (k-means++) initializations by within-class inertia is kept.

    Returns ``(sites, omega)`` with zero offsets, matching the D4d signature.
    """

    dim = Z.shape[1]
    sites = np.zeros((n_classes, m, dim), dtype=np.float64)
    for k in range(n_classes):
        Zk = Z[y == k]
        centroid = Zk.mean(axis=0)
        best_inertia = np.inf
        best: FloatArray | None = None
        for restart in range(n_restarts):
            rng = np.random.default_rng(seed + 1000 * restart + k)
            centers = np.empty((m, dim), dtype=np.float64)
            centers[0] = centroid
            # k-means++ seeding for the free centers
            chosen = _kmeanspp_free(Zk, m - 1, rng)
            centers[1:] = chosen
            for _ in range(max_iter):
                assign = np.argmin(_squared_distances(Zk, centers), axis=1)
                updated = centers.copy()
                for j in range(1, m):
                    members = Zk[assign == j]
                    updated[j] = (
                        members.mean(axis=0) if len(members) else centers[j]
                    )
                if np.allclose(updated, centers, atol=1e-10):
                    centers = updated
                    break
                centers = updated
            assign = np.argmin(_squared_distances(Zk, centers), axis=1)
            inertia = float(
                _squared_distances(Zk, centers)[np.arange(len(Zk)), assign].sum()
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best = centers
        sites[k] = best if best is not None else np.tile(centroid, (m, 1))
    omega = np.zeros((n_classes, m), dtype=np.float64)
    return sites, omega


def _kmeanspp_free(
    Zk: FloatArray, n_free: int, rng: np.random.Generator
) -> FloatArray:
    """k-means++ seeding of the free centers (center 0 is the centroid)."""

    n = len(Zk)
    if n_free == 0:
        return np.empty((0, Zk.shape[1]))
    first = int(rng.integers(n))
    chosen = [Zk[first]]
    dist2 = _squared_distances(Zk, Zk[first][None, :])[:, 0]
    for _ in range(n_free - 1):
        total = float(dist2.sum())
        probs = dist2 / total if total > 0 else np.full(n, 1.0 / n)
        nxt = int(rng.choice(n, p=probs))
        chosen.append(Zk[nxt])
        dist2 = np.minimum(
            dist2, _squared_distances(Zk, Zk[nxt][None, :])[:, 0]
        )
    return np.asarray(chosen, dtype=np.float64)


def multiprot_proba(
    Z: FloatArray, sites: FloatArray, gamma: float, omega: FloatArray | None = None
) -> FloatArray:
    """Multi-prototype probabilities: softmax over logsumexp_j(-gamma*d^2 + omega).

    Uses ``logsumexp`` over sites (matching D4d) instead of ``min_j`` so that
    unsupervised and discriminative multi-prototype decoders share the same
    aggregation.  Per-site offsets ``omega`` default to zero.
    """

    n = Z.shape[0]
    n_classes, m, _ = sites.shape
    flat = sites.reshape(n_classes * m, -1)
    d2 = _squared_distances(Z, flat).reshape(n, n_classes, m)
    if omega is None:
        omega = np.zeros((n_classes, m), dtype=np.float64)
    A = omega[None, :, :] - float(gamma) * d2
    return _softmax(logsumexp(A, axis=2))


def fit_d4_discriminative(
    Z: FloatArray,
    y: IntArray,
    C: float,
    *,
    init_sites: FloatArray,
    init_weights: FloatArray,
    maxiter: int = LBFGS_MAXITER,
) -> tuple[FloatArray, FloatArray]:
    """Refine multi-prototype sites on the classification loss (D4d).

    Class score: ``s_k(z) = LSE_j( omega_kj - ||z - p_kj||^2 )``.  For
    ``m = 1`` this is exactly a single power-diagram score, so D3 is a fitted
    special case.

    The L2 penalty mirrors D3 exactly: D3 penalises ``||W||^2 + ||b||^2``
    where ``W_k = 2 * p_k`` and ``b_k = omega_k - ||p_k||^2`` (the effective
    linear-probe parameters).  With ``m > 1`` sites per class, the same
    penalty is applied to every effective parameter pair:

        ``0.5 * reg * sum_{k,j}( 4 * ||p_kj||^2 + (omega_kj - ||p_kj||^2)^2 )``

    This eliminates the gauge-dependent over-regularisation that previously
    penalised ``||omega||^2`` instead of ``||b||^2``.
    """

    n, dim = Z.shape
    n_classes, m, _ = init_sites.shape
    n_site_params = n_classes * m * dim
    x0 = np.concatenate([init_sites.ravel(), init_weights.ravel()])
    reg = 1.0 / (float(C) * n)

    def loss_grad(flat: FloatArray) -> tuple[float, FloatArray]:
        P_sites = flat[:n_site_params].reshape(n_classes, m, dim)
        omega = flat[n_site_params:].reshape(n_classes, m)
        diff = Z[:, None, None, :] - P_sites[None, :, :, :]  # (n,K,m,d)
        d2 = np.einsum("nkmd,nkmd->nkm", diff, diff)
        A = omega[None, :, :] - d2  # (n,K,m)
        s_k = logsumexp(A, axis=2)  # (n,K)
        log_p = log_softmax(s_k, axis=1)
        loss = -float(log_p[np.arange(n), y].mean())
        q = np.exp(A - s_k[:, :, None])  # site responsibilities (n,K,m)
        Pc = np.exp(log_p)
        Pc[np.arange(n), y] -= 1.0
        coeff = (Pc / n)[:, :, None] * q  # dLoss/dA (n,K,m)
        grad_sites = np.einsum("nkm,nkmd->kmd", coeff, 2.0 * diff)
        grad_omega = coeff.sum(axis=0)

        # Effective-parameter penalty matching D3: ||W||^2 + ||b||^2
        # where W_kj = 2*p_kj, b_kj = omega_kj - ||p_kj||^2.
        p_sq = np.einsum("kmd,kmd->km", P_sites, P_sites)  # ||p_kj||^2
        omega_tilde = omega - p_sq
        loss += 0.5 * reg * (
            4.0 * float(p_sq.sum())
            + float(np.einsum("km,km->", omega_tilde, omega_tilde))
        )
        grad_sites += reg * (4.0 * P_sites - 2.0 * omega_tilde[:, :, None] * P_sites)
        grad_omega += reg * omega_tilde
        return loss, np.concatenate([grad_sites.ravel(), grad_omega.ravel()])

    result = minimize(
        loss_grad, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    sites = result.x[:n_site_params].reshape(n_classes, m, dim)
    omega = result.x[n_site_params:].reshape(n_classes, m)
    return sites, omega


def d4d_proba(Z: FloatArray, sites: FloatArray, omega: FloatArray) -> FloatArray:
    """Discriminative multi-prototype probabilities (gamma=1, logsumexp)."""
    return multiprot_proba(Z, sites, 1.0, omega)


# ---------------------------------------------------------------------------
# train-only fold transforms
# ---------------------------------------------------------------------------


def fit_fold_transform(
    X_train: FloatArray, *, pca_dim: int | None
) -> StandardScaler | tuple[StandardScaler, PCA]:
    """Fit the train-only standardization (and optional PCA) transform."""

    scaler = StandardScaler().fit(X_train)
    if pca_dim is None:
        return scaler
    pca = PCA(n_components=pca_dim, random_state=SEED).fit(scaler.transform(X_train))
    return scaler, pca


def apply_fold_transform(
    transform: StandardScaler | tuple[StandardScaler, PCA], X: FloatArray
) -> FloatArray:
    if isinstance(transform, tuple):
        scaler, pca = transform
        return np.asarray(pca.transform(scaler.transform(X)), dtype=np.float64)
    return np.asarray(transform.transform(X), dtype=np.float64)


# ---------------------------------------------------------------------------
# nested runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderLadderArtifact:
    directory: Path
    metadata: dict[str, Any]
    oof: pd.DataFrame
    selections: pd.DataFrame
    summary: dict[str, Any]


def _fit_predict_decoder(
    decoder: str,
    Z_tr: FloatArray,
    y_tr: IntArray,
    Z_ev: FloatArray,
    params: Mapping[str, float],
    *,
    n_classes: int,
    seed: int,
    d3_cache: dict[tuple[float, int], tuple[FloatArray, FloatArray]] | None = None,
    cache_token: int = 0,
) -> FloatArray:
    """Fit one decoder on a training matrix and score an evaluation matrix."""

    def _d3(C: float) -> tuple[FloatArray, FloatArray]:
        key = (C, cache_token)
        if d3_cache is not None and key in d3_cache:
            return d3_cache[key]
        fitted = fit_d3(Z_tr, y_tr, C, n_classes=n_classes)
        if d3_cache is not None:
            d3_cache[key] = fitted
        return fitted

    if decoder == "D0":
        return d0_proba(len(Z_ev), fit_prior(y_tr, n_classes))
    if decoder == "D1":
        return d1_proba(
            Z_ev, class_centroids(Z_tr, y_tr, n_classes), float(params["gamma"])
        )
    if decoder == "D1o":
        centroids, omega = fit_d1o(
            Z_tr, y_tr, float(params["gamma"]), float(params["C"]),
            n_classes=n_classes,
        )
        return d1o_proba(Z_ev, centroids, omega, float(params["gamma"]))
    if decoder == "D2":
        return d2_proba(Z_ev, fit_d2(Z_tr, y_tr, float(params["C"]), n_classes=n_classes))
    if decoder == "D3":
        W, b = _d3(float(params["C"]))
        return d3_proba(Z_ev, W, b)
    m = int(params["m"])
    if decoder == "D4":
        sites, omega = fit_class_kmeans(Z_tr, y_tr, m, n_classes=n_classes, seed=seed)
        return multiprot_proba(Z_ev, sites, float(params["gamma"]), omega)
    if decoder == "D4c":
        sites, omega = fit_class_kmeans_constrained(
            Z_tr, y_tr, m, n_classes=n_classes, seed=seed
        )
        return multiprot_proba(Z_ev, sites, float(params["gamma"]), omega)
    if decoder == "D4d":
        init_sites, _ = fit_class_kmeans(Z_tr, y_tr, m, n_classes=n_classes, seed=seed)
        W, b = _d3(float(params["C"]))
        _, omega_d3 = d3_sites_weights(W, b)
        init_weights = np.repeat(omega_d3[:, None], m, axis=1)
        sites, omega = fit_d4_discriminative(
            Z_tr, y_tr, float(params["C"]),
            init_sites=init_sites, init_weights=init_weights,
        )
        return multiprot_proba(Z_ev, sites, 1.0, omega)
    raise ValueError(f"unknown decoder: {decoder!r}")


def _param_grid(decoder: str) -> list[dict[str, float]]:
    if decoder == "D0":
        return [{}]
    if decoder == "D1":
        return [{"gamma": g} for g in GAMMA_GRID]
    if decoder == "D1o":
        return [{"gamma": g, "C": c} for g in GAMMA_GRID for c in C_GRID]
    if decoder in ("D2", "D3"):
        return [{"C": c} for c in C_GRID]
    if decoder in ("D4", "D4c"):
        return [{"gamma": g, "m": m} for g in GAMMA_GRID for m in M_GRID]
    if decoder == "D4d":
        return [{"C": c, "m": m} for c in C_GRID for m in M_GRID]
    raise ValueError(decoder)


def run_decoder_ladder(
    output_directory: str | Path,
    *,
    space: str,
    features: FloatArray,
    item_ids: Sequence[str],
    y: Sequence[str],
    group_ids: Sequence[str],
    outer_folds: pd.DataFrame,
    inner_folds: pd.DataFrame,
    pca_dim: int | None = None,
    class_names: Sequence[str] = CROWD_EMOTIONS,
    decoders: Sequence[str] = DECODERS,
    seed: int = SEED,
    folds: Sequence[int] | None = None,
) -> DecoderLadderArtifact:
    """Run the nested decoder ladder and publish an immutable artifact.

    ``outer_folds`` maps every item to one outer test fold; ``inner_folds``
    maps outer-training items to inner validation folds.  Every transform and
    every hyperparameter is fitted/selected inside the corresponding training
    fold only.
    """

    for decoder in decoders:
        if decoder not in DECODERS:
            raise ValueError(f"unknown decoder: {decoder!r}")
    X = np.asarray(features, dtype=np.float64)
    ids = [str(v) for v in item_ids]
    id_to_row = {value: i for i, value in enumerate(ids)}
    y_enc = encode_labels(y, class_names)
    n_classes = len(class_names)
    n_groups = len({str(v) for v in group_ids})

    outer = outer_folds.copy()
    outer["test_fold"] = outer["test_fold"].astype(int)
    inner = inner_folds.copy()
    inner["outer_fold"] = inner["outer_fold"].astype(int)
    inner["validation_fold"] = inner["validation_fold"].astype(int)

    oof_frames: dict[str, list[pd.DataFrame]] = {d: [] for d in decoders}
    selection_rows: list[dict[str, Any]] = []
    all_folds = sorted(outer["test_fold"].unique())
    n_outer_folds = len(all_folds)
    total_start = time.monotonic()
    for outer_fold in all_folds:
        if folds is not None and outer_fold not in folds:
            continue
        fold_start = time.monotonic()
        print(
            f"[decoder-ladder] space={space} "
            f"fold {outer_fold + 1}/{n_outer_folds} "
            f"({time.monotonic() - total_start:.0f}s elapsed)",
            file=sys.stderr, flush=True,
        )
        test_ids = outer.loc[outer["test_fold"] == outer_fold, "item_id"].astype(str)
        train_ids = outer.loc[outer["test_fold"] != outer_fold, "item_id"].astype(str)
        te_idx = np.array([id_to_row[v] for v in test_ids], dtype=np.int64)
        tr_idx = np.array([id_to_row[v] for v in train_ids], dtype=np.int64)
        transform = fit_fold_transform(X[tr_idx], pca_dim=pca_dim)
        Z_tr = apply_fold_transform(transform, X[tr_idx])
        Z_te = apply_fold_transform(transform, X[te_idx])
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

        inner_sub = inner[inner["outer_fold"] == outer_fold]
        validation_folds = sorted(inner_sub["validation_fold"].unique())
        inner_masks = {
            v: inner_sub.loc[
                inner_sub["validation_fold"] == v, "item_id"
            ].astype(str).tolist()
            for v in validation_folds
        }
        inner_cache: dict[int, tuple[FloatArray, IntArray, FloatArray, IntArray]] = {}

        def _inner_split(v: int) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
            if v not in inner_cache:
                valid = set(inner_masks[v])
                keep = np.array(
                    [str(train_ids.iloc[i]) not in valid for i in range(len(train_ids))]
                )
                sub_transform = fit_fold_transform(
                    X[tr_idx][keep], pca_dim=pca_dim
                )
                Z_tr_i = apply_fold_transform(sub_transform, X[tr_idx][keep])
                Z_va_i = apply_fold_transform(sub_transform, X[tr_idx][~keep])
                inner_cache[v] = (Z_tr_i, y_tr[keep], Z_va_i, y_tr[~keep])
            return inner_cache[v]

        for decoder in decoders:
            grid = _param_grid(decoder)
            n_configs = len(grid)
            print(
                f"[decoder-ladder]   {decoder}: grid {n_configs} configs "
                f"× {len(validation_folds)} inner folds",
                file=sys.stderr, flush=True,
            )
            best_params: dict[str, float] | None = None
            best_loss = np.inf
            d3_cache: dict[tuple[float, int], tuple[FloatArray, FloatArray]] = {}
            dec_start = time.monotonic()
            for ci, params in enumerate(grid):
                losses = []
                for v in validation_folds:
                    Z_tr_i, y_tr_i, Z_va_i, y_va_i = _inner_split(v)
                    if len(np.unique(y_tr_i)) < 2 or len(y_va_i) == 0:
                        continue
                    proba = _fit_predict_decoder(
                        decoder, Z_tr_i, y_tr_i, Z_va_i, params,
                        n_classes=n_classes, seed=seed + 97 * outer_fold + v,
                        d3_cache=d3_cache, cache_token=int(v),
                    )
                    losses.append(_log_loss_bits(proba, y_va_i))
                mean_loss = float(np.mean(losses)) if losses else np.inf
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_params = dict(params)
                if (ci + 1) % 3 == 0 or ci + 1 == n_configs:
                    print(
                        f"[decoder-ladder]   {decoder}: config {ci + 1}/{n_configs} "
                        f"({time.monotonic() - dec_start:.0f}s)",
                        file=sys.stderr, flush=True,
                    )
            assert best_params is not None
            selection_rows.append(
                {"outer_fold": int(outer_fold), "decoder": decoder, **best_params}
            )
            proba = _fit_predict_decoder(
                decoder, Z_tr, y_tr, Z_te, best_params,
                n_classes=n_classes, seed=seed + 97 * outer_fold,
                d3_cache=d3_cache, cache_token=-1,
            )
            elapsed = time.monotonic() - fold_start
            print(
                f"[decoder-ladder]   {decoder} done "
                f"({time.monotonic() - dec_start:.0f}s, fold total {elapsed:.0f}s)",
                file=sys.stderr, flush=True,
            )
            oof_frames[decoder].append(
                _oof_frame(
                    test_ids.tolist(), y_te, proba, class_names, int(outer_fold)
                )
            )
        print(
            f"[decoder-ladder] fold {outer_fold + 1}/{n_outer_folds} done "
            f"({time.monotonic() - fold_start:.0f}s, "
            f"total {time.monotonic() - total_start:.0f}s)",
            file=sys.stderr, flush=True,
        )

    oof_parts = []
    for decoder in decoders:
        frame = pd.concat(oof_frames[decoder], ignore_index=True)
        frame.insert(1, "decoder", decoder)
        oof_parts.append(frame)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["decoder", "item_id"], kind="stable"
    ).reset_index(drop=True)
    selections = pd.DataFrame(selection_rows)

    summary = _summarize(oof, selections, class_names=class_names)
    summary["n_groups"] = n_groups
    return _write_artifact(
        Path(output_directory),
        space=space,
        pca_dim=pca_dim,
        features=X,
        outer_folds=outer_folds,
        oof=oof,
        selections=selections,
        summary=summary,
        seed=seed,
    )


def _oof_frame(
    item_ids: list[str],
    y_true: IntArray,
    proba: FloatArray,
    class_names: Sequence[str],
    outer_fold: int,
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "item_id": item_ids,
        "outer_fold": int(outer_fold),
        "y_true": [class_names[i] for i in y_true],
        "y_pred": [class_names[i] for i in proba.argmax(axis=1)],
    }
    for j, name in enumerate(class_names):
        data[f"prob__{name}"] = proba[:, j]
    return pd.DataFrame(data)


def _summarize(
    oof: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Per-decoder log loss in bits, fold dispersion, and delta vs prior."""

    prob_cols = [f"prob__{c}" for c in class_names]
    class_index = {c: i for i, c in enumerate(class_names)}
    rows: dict[str, Any] = {}
    for decoder, frame in oof.groupby("decoder"):
        proba = frame[prob_cols].to_numpy(dtype=np.float64)
        y_idx = np.array(
            [class_index[v] for v in frame["y_true"].astype(str)], dtype=np.int64
        )
        p_true = np.clip(proba[np.arange(len(y_idx)), y_idx], EPS, 1.0)
        item_loss = -np.log2(p_true)
        fold_losses = [
            float(item_loss[(frame["outer_fold"] == f).to_numpy()].mean())
            for f in sorted(frame["outer_fold"].unique())
        ]
        rows[str(decoder)] = {
            "log_loss_bits": float(item_loss.mean()),
            "fold_sd": float(np.std(fold_losses, ddof=1)),
            "n_items": int(len(frame)),
        }
    prior_bits = rows.get("D0", {}).get("log_loss_bits")
    if prior_bits is not None:
        for decoder, record in rows.items():
            record["delta_vs_prior_bits"] = record["log_loss_bits"] - prior_bits
    selection_summary: dict[str, dict[str, Any]] = {}
    for decoder, frame in selections.groupby("decoder"):
        record: dict[str, Any] = {}
        for column in frame.columns:
            if column in ("outer_fold", "decoder"):
                continue
            values = frame[column].astype(float)
            if values.isna().all():
                continue
            record[column] = {
                "selected": [float(v) for v in values],
                "median": float(values.median()),
            }
        selection_summary[str(decoder)] = record
    return {"decoders": rows, "selections": selection_summary}


def _write_artifact(
    directory: Path,
    *,
    space: str,
    pca_dim: int | None,
    features: FloatArray,
    outer_folds: pd.DataFrame,
    oof: pd.DataFrame,
    selections: pd.DataFrame,
    summary: dict[str, Any],
    seed: int,
) -> DecoderLadderArtifact:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        oof_path = staging / "oof.parquet"
        sel_path = staging / "selections.parquet"
        sum_path = staging / "summary.json"
        oof.to_parquet(oof_path, index=False)
        selections.to_parquet(sel_path, index=False)
        sum_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        metadata = {
            "analysis_format": RUN_FORMAT,
            "status": "new_replication_diagnostic_not_historical_recovery",
            "space": space,
            "pca_dim": pca_dim,
            "seed": seed,
            "n_items": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "feature_matrix_sha256": _sha256_array(np.ascontiguousarray(features)),
            "outer_split_sha256": _dataframe_digest(outer_folds),
            "decoders": sorted(summary["decoders"]),
            "grids": {
                "gamma": list(GAMMA_GRID),
                "C": list(C_GRID),
                "m": list(M_GRID),
                "kmeans_restarts": KMEANS_RESTARTS,
                "lbfgs_maxiter": LBFGS_MAXITER,
            },
            "files": {
                path.name: {
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in (oof_path, sel_path, sum_path)
            },
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
        os_replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DecoderLadderArtifact(directory, metadata, oof, selections, summary)


def os_replace(staging: Path, directory: Path) -> None:
    staging.rename(directory)


__all__ = [
    "C_GRID",
    "DECODERS",
    "GAMMA_GRID",
    "M_GRID",
    "RUN_FORMAT",
    "DecoderLadderArtifact",
    "apply_fold_transform",
    "class_centroids",
    "d0_proba",
    "d1_proba",
    "d2_proba",
    "d3_proba",
    "d3_sites_weights",
    "d4d_proba",
    "encode_labels",
    "fit_class_kmeans",
    "fit_class_kmeans_constrained",
    "fit_d2",
    "fit_d3",
    "fit_d4_discriminative",
    "fit_fold_transform",
    "fit_prior",
    "multiprot_proba",
    "run_decoder_ladder",
]
