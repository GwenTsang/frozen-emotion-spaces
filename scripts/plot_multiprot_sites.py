"""PCA-2D visualization and boundary diagnostics for multi-prototype sites.

Produces the figure behind the paper's multi-prototype diagnostic paragraph:
D3 power-diagram boundaries (evaluated in the full standardized space and
sliced along the first two train-fitted principal components), per-class
k-means sites (D4), centroid-constrained sites (D4c), and discriminatively
refined sites (D4d), for one outer fold of crowd-enVENT appraisals.

Also writes a small JSON of quantitative site-placement diagnostics computed
in the full standardized space (not in the 2D projection):

- ``margin_to_d3_boundary``: signed distance of each site to the nearest D3
  pairwise boundary, in units of the boundary normal; positive means the
  site lies on its own class's side of every boundary.
- ``distance_to_own_centroid``: Euclidean distance of each site to its class
  centroid (D1 site), in standard-deviation units.
- ``cross_class_assignment_share``: fraction of outer-training items of
  other classes whose nearest site belongs to class k (Voronoi intrusion).

This is a new replication diagnostic, not recovered historical code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frozen_emotion_spaces.crowd_data import (  # noqa: E402
    APPRAISAL_NAMES,
    CROWD_EMOTIONS,
    build_crowd_manifests,
)
from frozen_emotion_spaces.decoder_ladder import (  # noqa: E402
    apply_fold_transform,
    class_centroids,
    encode_labels,
    fit_class_kmeans,
    fit_class_kmeans_constrained,
    fit_d3,
    fit_d4_discriminative,
    fit_fold_transform,
    d3_sites_weights,
)
from frozen_emotion_spaces.splits import read_split_bundle  # noqa: E402


def _select(selections: pd.DataFrame, decoder: str, fold: int) -> dict[str, float]:
    row = selections[
        (selections["decoder"] == decoder) & (selections["outer_fold"] == fold)
    ].iloc[0]
    return {c: float(row[c]) for c in selections.columns if c not in ("outer_fold", "decoder")}


def site_diagnostics(
    Z: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    centroids: np.ndarray,
) -> dict[str, list[dict[str, float]]]:
    """Quantitative site-placement diagnostics in the full space."""

    n_classes, m, _ = sites.shape
    boundaries = {}
    for k in range(n_classes):
        for j in range(k + 1, n_classes):
            normal = W[k] - W[j]
            norm = float(np.linalg.norm(normal))
            if norm > 0:
                boundaries[(k, j)] = (normal / norm, float(b[k] - b[j]) / norm)
    rows: list[dict[str, float]] = []
    for k in range(n_classes):
        for j in range(m):
            p = sites[k, j]
            margins = []
            for (a, c), (unit, offset) in boundaries.items():
                if k not in (a, c):
                    continue
                signed = float(unit @ p + offset)
                margins.append(signed if k == a else -signed)
            rows.append(
                {
                    "class": k,
                    "site": j,
                    "margin_to_d3_boundary": min(margins),
                    "distance_to_own_centroid": float(
                        np.linalg.norm(p - centroids[k])
                    ),
                }
            )
    # cross-class Voronoi intrusion on the training domain
    flat = sites.reshape(n_classes * m, -1)
    d2 = (
        np.linalg.norm(Z[:, None, :] - flat[None, :, :], axis=2) ** 2
    )
    nearest = d2.argmin(axis=1)
    site_class = np.repeat(np.arange(n_classes), m)
    intrusion = np.zeros(n_classes)
    totals = np.zeros(n_classes)
    for i in range(len(Z)):
        owner = site_class[nearest[i]]
        if owner != y[i]:
            intrusion[owner] += 1
        totals[y[i]] += 1
    shares = intrusion / max(float(len(Z)), 1.0)
    for k in range(n_classes):
        for j in range(m):
            rows[k * m + j]["cross_class_assignment_share"] = float(shares[k])
    return {"sites": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()

    crowd = build_crowd_manifests(args.archive)
    splits = read_split_bundle(args.splits)
    generation = crowd.generation
    item_ids = generation["item_id"].astype(str).tolist()
    id_to_row = {v: i for i, v in enumerate(item_ids)}
    X = generation[list(APPRAISAL_NAMES)].to_numpy(dtype=float)
    y = encode_labels(generation["y_writer"].astype(str).tolist(), CROWD_EMOTIONS)

    outer = splits.crowd_full_outer
    test_ids = outer.loc[
        outer["test_fold"].astype(int) == args.fold, "item_id"
    ].astype(str)
    train_ids = outer.loc[
        outer["test_fold"].astype(int) != args.fold, "item_id"
    ].astype(str)
    tr = np.array([id_to_row[v] for v in train_ids])
    te = np.array([id_to_row[v] for v in test_ids])

    transform = fit_fold_transform(X[tr], pca_dim=None)
    Z_tr = apply_fold_transform(transform, X[tr])
    y_tr = y[tr]

    selections = pd.read_parquet(args.ladder / "selections.parquet")
    C_d3 = _select(selections, "D3", args.fold)["C"]
    m_d4 = int(_select(selections, "D4", args.fold)["m"])
    C_d4d = _select(selections, "D4d", args.fold)["C"]

    n_classes = len(CROWD_EMOTIONS)
    W, b = fit_d3(Z_tr, y_tr, C_d3, n_classes=n_classes)
    cents = class_centroids(Z_tr, y_tr, n_classes)
    sites_d4 = fit_class_kmeans(Z_tr, y_tr, m_d4, n_classes=n_classes, seed=20240804 + 97 * args.fold)
    sites_d4c = fit_class_kmeans_constrained(Z_tr, y_tr, m_d4, n_classes=n_classes, seed=20240804 + 97 * args.fold)
    _, omega_d3 = d3_sites_weights(W, b)
    sites_d4d, _ = fit_d4_discriminative(
        Z_tr, y_tr, C_d4d,
        init_sites=sites_d4,
        init_weights=np.repeat(omega_d3[:, None], m_d4, axis=1),
    )

    diagnostics = {
        "fold": args.fold,
        "m": m_d4,
        "D4_kmeans": site_diagnostics(Z_tr, y_tr, sites_d4, W, b, cents),
        "D4c_constrained": site_diagnostics(Z_tr, y_tr, sites_d4c, W, b, cents),
        "D4d_discriminative": site_diagnostics(Z_tr, y_tr, sites_d4d, W, b, cents),
        "D3_sites": site_diagnostics(
            Z_tr, y_tr, d3_sites_weights(W, b)[0][:, None, :], W, b, cents
        ),
    }
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.write_text(json.dumps(diagnostics, indent=2))

    # ---- figure: slice the full-space D3 boundaries along PC1-PC2 ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    scaler = transform  # StandardScaler fitted on outer train
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(X[tr]))
    P_tr = pca.transform(scaler.transform(X[tr]))

    grid_n = 240
    pad = 1.0
    xs = np.linspace(P_tr[:, 0].min() - pad, P_tr[:, 0].max() + pad, grid_n)
    ys = np.linspace(P_tr[:, 1].min() - pad, P_tr[:, 1].max() + pad, grid_n)
    GX, GY = np.meshgrid(xs, ys)
    grid2 = np.column_stack([GX.ravel(), GY.ravel()])
    lifted = pca.inverse_transform(grid2)  # points on the PC1-PC2 plane in 21D
    scores = lifted @ W.T + b
    assign = scores.argmax(axis=1).reshape(GX.shape)

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.contourf(GX, GY, assign, levels=np.arange(n_classes + 1) - 0.5,
                cmap="tab20", alpha=0.12)
    ax.contour(GX, GY, assign, levels=np.arange(n_classes + 1) - 0.5,
               colors="0.35", linewidths=0.7)
    for k in range(n_classes):
        members = P_tr[y_tr == k]
        ax.scatter(members[:, 0], members[:, 1], s=3, alpha=0.10,
                   color=plt.cm.tab20(k), rasterized=True)
    proj = lambda S: pca.transform(S.reshape(-1, S.shape[-1])).reshape(S.shape[:-1] + (2,))
    P_d3 = proj(d3_sites_weights(W, b)[0])
    P_d4 = proj(sites_d4)
    P_d4d = proj(sites_d4d)
    ax.scatter(P_d3[:, 0], P_d3[:, 1], marker="*", s=220, c="black",
               label="D3 power-diagram site", zorder=5)
    ax.scatter(P_d4[..., 0].ravel(), P_d4[..., 1].ravel(), marker="^", s=70,
               facecolors="none", edgecolors="red", linewidths=1.4,
               label=f"D4 k-means site (m={m_d4})", zorder=4)
    ax.scatter(P_d4d[..., 0].ravel(), P_d4d[..., 1].ravel(), marker="D", s=55,
               facecolors="none", edgecolors="blue", linewidths=1.4,
               label="D4 discriminative refinement", zorder=4)
    ax.set_xlabel("PC1 (train-only PCA of standardized appraisals)")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"Outer fold {args.fold}: D3 boundaries (PC1-PC2 slice) and multi-prototype sites"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)
    print(json.dumps({"figure": str(args.figure), "diagnostics": str(args.diagnostics)}))


if __name__ == "__main__":
    main()
