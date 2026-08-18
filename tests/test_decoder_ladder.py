from __future__ import annotations

import numpy as np
import pytest

from frozen_emotion_spaces.decoder_ladder import (
    apply_fold_transform,
    class_centroids,
    d1_proba,
    d2_proba,
    d3_proba,
    d3_sites_weights,
    d4d_proba,
    encode_labels,
    fit_class_kmeans,
    fit_class_kmeans_constrained,
    fit_d2,
    fit_d3,
    fit_d4_discriminative,
    fit_fold_transform,
    fit_prior,
    multiprot_proba,
    run_decoder_ladder,
    _log_loss_bits,
)


CLASSES = ("a", "b", "c")


def _gaussian_classes(
    *, n_per_class: int = 60, dim: int = 4, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=3.0, size=(len(CLASSES), dim))
    blocks = [
        rng.normal(loc=means[k], scale=1.0, size=(n_per_class, dim))
        for k in range(len(CLASSES))
    ]
    Z = np.vstack(blocks)
    y = np.repeat(np.arange(len(CLASSES)), n_per_class)
    return Z, y


def test_prior_is_normalized_and_uniform_for_balanced_classes() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    prior = fit_prior(y, 3)
    np.testing.assert_allclose(prior, [1 / 3, 1 / 3, 1 / 3])


def test_d1_prefers_the_nearest_centroid() -> None:
    Z, y = _gaussian_classes()
    cents = class_centroids(Z, y, 3)
    proba = d1_proba(Z, cents, gamma=1.0)
    assert (proba.argmax(axis=1) == y).mean() > 0.95


def test_d1_gamma_scales_confidence_not_argmax() -> None:
    Z, y = _gaussian_classes()
    cents = class_centroids(Z, y, 3)
    sharp = d1_proba(Z, cents, gamma=10.0)
    soft = d1_proba(Z, cents, gamma=0.01)
    assert sharp.max(axis=1).mean() > soft.max(axis=1).mean()
    assert (sharp.argmax(axis=1) == soft.argmax(axis=1)).all()


def test_d2_beats_or_matches_centroid_baseline_on_train() -> None:
    Z, y = _gaussian_classes()
    cents = class_centroids(Z, y, 3)
    sites = fit_d2(Z, y, C=1.0, n_classes=3)
    loss_init = _log_loss_bits(d2_proba(Z, cents), y)
    loss_fit = _log_loss_bits(d2_proba(Z, sites), y)
    assert loss_fit <= loss_init + 1e-6


def test_d3_equivalent_power_sites_give_same_predictions() -> None:
    Z, y = _gaussian_classes()
    W, b = fit_d3(Z, y, C=10.0, n_classes=3)
    sites, omega = d3_sites_weights(W, b)
    linear = d3_proba(Z, W, b)
    power = d4d_proba(Z, sites[:, None, :], omega[:, None])
    np.testing.assert_allclose(linear, power, atol=1e-8)


def test_d3_learns_separable_classes() -> None:
    Z, y = _gaussian_classes()
    W, b = fit_d3(Z, y, C=10.0, n_classes=3)
    assert (d3_proba(Z, W, b).argmax(axis=1) == y).mean() > 0.95


def test_class_kmeans_places_m_sites_per_class() -> None:
    Z, y = _gaussian_classes()
    sites, omega = fit_class_kmeans(Z, y, 2, n_classes=3, seed=0)
    assert sites.shape == (3, 2, 4)
    assert omega.shape == (3, 2)
    # sites of class k are closer to the class centroid than to other classes
    cents = class_centroids(Z, y, 3)
    for k in range(3):
        own = np.linalg.norm(sites[k] - cents[k], axis=1)
        other = np.linalg.norm(sites[k] - cents[(k + 1) % 3], axis=1)
        assert (own < other).all()


def test_constrained_kmeans_keeps_first_site_at_class_centroid() -> None:
    Z, y = _gaussian_classes()
    sites, omega = fit_class_kmeans_constrained(Z, y, 3, n_classes=3, seed=0)
    cents = class_centroids(Z, y, 3)
    np.testing.assert_allclose(sites[:, 0, :], cents, atol=1e-10)


def test_constrained_kmeans_inertia_beats_untuned_centroid_only() -> None:
    Z, y = _gaussian_classes()
    m = 2
    sites, omega = fit_class_kmeans_constrained(Z, y, m, n_classes=3, seed=0)
    cents = class_centroids(Z, y, 3)
    inertia_sites = 0.0
    inertia_cents = 0.0
    for k in range(3):
        Zk = Z[y == k]
        d_sites = np.linalg.norm(Zk[:, None, :] - sites[k][None, :, :], axis=2) ** 2
        inertia_sites += float(d_sites.min(axis=1).sum())
        inertia_cents += float(((Zk - cents[k]) ** 2).sum())
    assert inertia_sites < inertia_cents


def test_multiprot_proba_argmax_is_nearest_site_class() -> None:
    Z, y = _gaussian_classes()
    sites, omega = fit_class_kmeans(Z, y, 2, n_classes=3, seed=0)
    proba = multiprot_proba(Z, sites, gamma=1.0, omega=omega)
    # With logsumexp and zero omega, argmax should still match min distance for well-separated classes
    d2 = (
        np.linalg.norm(Z[:, None, None, :] - sites[None, :, :, :], axis=3) ** 2
    ).min(axis=2)
    # Allow small fraction of mismatches due to logsumexp vs min
    assert (proba.argmax(axis=1) == d2.argmin(axis=1)).mean() > 0.95


def test_d4d_m1_recovers_power_diagram_scores() -> None:
    Z, y = _gaussian_classes()
    W, b = fit_d3(Z, y, C=10.0, n_classes=3)
    sites, omega = d3_sites_weights(W, b)
    refined_sites, refined_omega = fit_d4_discriminative(
        Z, y, C=10.0, init_sites=sites[:, None, :], init_weights=omega[:, None]
    )
    # initialized at the D3 solution, refinement must not increase train loss
    base = _log_loss_bits(d3_proba(Z, W, b), y)
    refined = _log_loss_bits(d4d_proba(Z, refined_sites, refined_omega), y)
    assert refined <= base + 1e-6


def test_d4d_reduces_train_loss_from_kmeans_initialization() -> None:
    rng = np.random.default_rng(3)
    means = rng.normal(scale=1.0, size=(3, 4))
    Z = np.vstack(
        [rng.normal(loc=means[k], scale=1.5, size=(80, 4)) for k in range(3)]
    )
    y = np.repeat(np.arange(3), 80)
    init, _ = fit_class_kmeans(Z, y, 2, n_classes=3, seed=0)
    init_w = np.zeros((3, 2))
    base = _log_loss_bits(d4d_proba(Z, init, init_w), y)
    sites, omega = fit_d4_discriminative(
        Z, y, C=10.0, init_sites=init, init_weights=init_w
    )
    refined = _log_loss_bits(d4d_proba(Z, sites, omega), y)
    assert refined < base - 1e-4


def test_fold_transform_is_train_fitted_and_shapes_consistent() -> None:
    rng = np.random.default_rng(0)
    X_tr = rng.normal(loc=5.0, size=(40, 6))
    X_te = rng.normal(loc=-5.0, size=(10, 6))
    transform = fit_fold_transform(X_tr, pca_dim=3)
    Z_tr = apply_fold_transform(transform, X_tr)
    Z_te = apply_fold_transform(transform, X_te)
    assert Z_tr.shape == (40, 3)
    assert Z_te.shape == (10, 3)
    np.testing.assert_allclose(Z_tr.mean(axis=0), np.zeros(3), atol=1e-8)
    # test rows are not re-centered: their mean stays away from zero
    assert np.abs(Z_te.mean(axis=0)).max() > 1.0


def test_encode_labels_uses_fixed_vocabulary() -> None:
    encoded = encode_labels(["b", "a", "c", "b"], CLASSES)
    np.testing.assert_array_equal(encoded, [1, 0, 2, 1])
    with pytest.raises(ValueError, match="unknown label"):
        encode_labels(["zzz"], CLASSES)


def _synthetic_folds(ids: list[str]) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    import pandas as pd

    outer = pd.DataFrame(
        {
            "item_id": ids,
            "group_id": [f"g{i}" for i in range(len(ids))],
            "writer_id": [f"w{i}" for i in range(len(ids))],
            "test_fold": [i % 5 for i in range(len(ids))],
        }
    )
    rows = []
    for fold in range(5):
        train = outer[outer["test_fold"] != fold].reset_index(drop=True)
        for i, row in train.iterrows():
            rows.append(
                {
                    "outer_fold": fold,
                    "item_id": row["item_id"],
                    "group_id": row["group_id"],
                    "writer_id": row["writer_id"],
                    "validation_fold": i % 3,
                }
            )
    return outer, pd.DataFrame(rows)


def test_run_decoder_ladder_writes_immutable_artifact(tmp_path) -> None:
    Z, y = _gaussian_classes(n_per_class=50, dim=5, seed=11)
    ids = [f"item-{i}" for i in range(len(Z))]
    y_str = [CLASSES[i] for i in y]
    outer, inner = _synthetic_folds(ids)
    artifact = run_decoder_ladder(
        tmp_path / "ladder",
        space="synthetic",
        features=Z,
        item_ids=ids,
        y=y_str,
        group_ids=[f"g{i}" for i in range(len(Z))],
        outer_folds=outer,
        inner_folds=inner,
        decoders=("D0", "D1", "D3"),
        class_names=CLASSES,
    )
    assert (artifact.directory / "metadata.json").is_file()
    assert (artifact.directory / "oof.parquet").is_file()
    summary = artifact.summary["decoders"]
    assert set(summary) == {"D0", "D1", "D3"}
    # separable synthetic classes: D3 must save many bits over the prior
    assert summary["D3"]["log_loss_bits"] < summary["D0"]["log_loss_bits"] - 0.5
    assert artifact.metadata["status"] == (
        "new_replication_diagnostic_not_historical_recovery"
    )
    with pytest.raises(FileExistsError):
        run_decoder_ladder(
            tmp_path / "ladder",
            space="synthetic",
            features=Z,
            item_ids=ids,
            y=y_str,
            group_ids=[f"g{i}" for i in range(len(Z))],
            outer_folds=outer,
            inner_folds=inner,
            decoders=("D0",),
            class_names=CLASSES,
        )
