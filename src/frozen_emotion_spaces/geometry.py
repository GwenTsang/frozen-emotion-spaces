"""Contrast, Representation, and linear-probe power geometry.

The raw Contrast and Representation definitions follow calc() in
IgorDouven/Concept_Learning at commit
2325717f68f9eecbc85cfa7d7e5ada0dc7e95679. This is a clean-room Python
implementation, not recovered project source. All domain points are explicit
inputs: no labels or evaluation data are discovered from ambient experiment
state.

The power-diagram conversion is an algebraic utility for multinomial linear
scores. It is not part of Douven's calc() and is not presented as a Douven
Contrast/Representation ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ContrastRepresentationScores:
    """Raw and explicitly normalized constellation scores.

    contrast_sum is the sum over all unordered pairs of sites.
    representation_sum is the sum, over sites, of the distance from the site
    to the arithmetic centroid of its induced ordinary Voronoi cell. The two
    mean fields are only denominator-normalized versions of those respective
    sums; they are not a combined index or ratio.
    """

    contrast_sum: float
    representation_sum: float
    mean_pairwise_site_distance: float
    mean_site_to_cell_centroid_distance: float
    assignments: IntArray
    centroids: FloatArray


@dataclass(frozen=True)
class MultinomialPowerDiagram:
    """Power-diagram form of sum-zero-gauged multinomial linear scores.

    With class score w_k @ x + b_k, sites[k] = w_k / 2 and
    power_weights[k] = b_k + ||w_k||^2 / 4. Minimizing
    ||x - sites[k]||^2 - power_weights[k] therefore gives the same class as
    maximizing the linear score.
    """

    gauged_coefficients: FloatArray
    gauged_intercepts: FloatArray
    sites: FloatArray
    power_weights: FloatArray


def contrast_sum_pairwise_distances(
    sites: ArrayLike,
    *,
    domain_points: ArrayLike,
) -> float:
    """Return raw Contrast: the Euclidean sum over unordered site pairs.

    domain_points is required and dimension-checked even though the raw
    Contrast formula depends only on the sites. This makes the domain of any
    reported geometry explicit at the API boundary.
    """

    site_array, _ = _validate_geometry_inputs(sites, domain_points)
    distances = _pairwise_site_distances(site_array)
    return float(distances.sum(dtype=np.float64))


def mean_pairwise_site_distance(
    sites: ArrayLike,
    *,
    domain_points: ArrayLike,
) -> float:
    """Return Contrast divided by the number of unordered site pairs."""

    site_array, point_array = _validate_geometry_inputs(sites, domain_points)
    pair_count = site_array.shape[0] * (site_array.shape[0] - 1) // 2
    return contrast_sum_pairwise_distances(
        site_array,
        domain_points=point_array,
    ) / pair_count


def assign_domain_points_to_nearest_sites(
    sites: ArrayLike,
    *,
    domain_points: ArrayLike,
    require_nonempty_cells: bool = True,
) -> IntArray:
    """Assign every domain point to its nearest ordinary Euclidean site.

    Ties are resolved by the first site in input order, following NumPy's
    deterministic argmin convention and Julia's first-minimum behaviour.
    """

    site_array, point_array = _validate_geometry_inputs(sites, domain_points)
    squared_distances = _squared_euclidean_distances(point_array, site_array)
    assignments = np.argmin(squared_distances, axis=1).astype(np.int64, copy=False)
    if require_nonempty_cells:
        _require_all_cells(assignments, n_sites=site_array.shape[0])
    return assignments


def representation_sum_site_to_cell_centroid_distances(
    sites: ArrayLike,
    *,
    domain_points: ArrayLike,
) -> float:
    """Return raw Representation for the ordinary nearest-site partition."""

    site_array, point_array = _validate_geometry_inputs(sites, domain_points)
    assignments = assign_domain_points_to_nearest_sites(
        site_array,
        domain_points=point_array,
        require_nonempty_cells=True,
    )
    centroids = _cell_centroids(
        point_array,
        assignments,
        n_sites=site_array.shape[0],
    )
    return _representation_sum(site_array, centroids)


def mean_site_to_cell_centroid_distance(
    sites: ArrayLike,
    *,
    domain_points: ArrayLike,
) -> float:
    """Return Representation divided by the number of sites."""

    site_array, point_array = _validate_geometry_inputs(sites, domain_points)
    return representation_sum_site_to_cell_centroid_distances(
        site_array,
        domain_points=point_array,
    ) / site_array.shape[0]


def contrast_representation_scores(
    sites: ArrayLike,
    *,
    domain_points: ArrayLike,
) -> ContrastRepresentationScores:
    """Compute both raw source-faithful scores and explicit mean variants."""

    site_array, point_array = _validate_geometry_inputs(sites, domain_points)
    assignments = assign_domain_points_to_nearest_sites(
        site_array,
        domain_points=point_array,
        require_nonempty_cells=True,
    )
    centroids = _cell_centroids(
        point_array,
        assignments,
        n_sites=site_array.shape[0],
    )
    pair_distances = _pairwise_site_distances(site_array)
    contrast = float(pair_distances.sum(dtype=np.float64))
    representation = _representation_sum(site_array, centroids)
    return ContrastRepresentationScores(
        contrast_sum=contrast,
        representation_sum=representation,
        mean_pairwise_site_distance=contrast / len(pair_distances),
        mean_site_to_cell_centroid_distance=representation / site_array.shape[0],
        assignments=assignments,
        centroids=centroids,
    )


def multinomial_coefficients_to_power_diagram(
    coefficients: ArrayLike,
    intercepts: ArrayLike,
) -> MultinomialPowerDiagram:
    """Convert class-by-feature linear coefficients under a sum-zero gauge.

    A common linear function and common constant do not affect argmax. They
    are removed by centering each feature column and the intercept vector
    across classes. At least two explicit class rows are required; a one-row
    binary estimator convention is intentionally not guessed.
    """

    coefficient_array = _as_finite_float_matrix(
        coefficients,
        name="coefficients",
        minimum_rows=2,
    )
    intercept_array = _as_finite_float_vector(intercepts, name="intercepts")
    if intercept_array.shape[0] != coefficient_array.shape[0]:
        raise ValueError("intercepts must contain one value per class row")

    gauged_coefficients = coefficient_array - coefficient_array.mean(
        axis=0,
        keepdims=True,
    )
    gauged_intercepts = intercept_array - intercept_array.mean()
    sites = gauged_coefficients / 2.0
    power_weights = gauged_intercepts + np.einsum("ij,ij->i", sites, sites)
    if not np.isfinite(power_weights).all():
        raise ValueError("power-diagram conversion produced non-finite weights")
    return MultinomialPowerDiagram(
        gauged_coefficients=gauged_coefficients,
        gauged_intercepts=gauged_intercepts,
        sites=sites,
        power_weights=power_weights,
    )


def assign_domain_points_by_power_distance(
    sites: ArrayLike,
    power_weights: ArrayLike,
    *,
    domain_points: ArrayLike,
) -> IntArray:
    """Assign explicit domain points by minimum squared power distance.

    Empty cells are permitted here: this function expresses a classifier
    decision rule, whereas the Representation score requires every ordinary
    nearest-site cell to have a centroid. Ties select the first site.
    """

    site_array, point_array = _validate_geometry_inputs(sites, domain_points)
    weight_array = _as_finite_float_vector(power_weights, name="power_weights")
    if weight_array.shape[0] != site_array.shape[0]:
        raise ValueError("power_weights must contain one value per site")
    power_distance = _squared_euclidean_distances(point_array, site_array)
    power_distance -= weight_array[None, :]
    if not np.isfinite(power_distance).all():
        raise ValueError("power-distance computation produced non-finite values")
    return np.argmin(power_distance, axis=1).astype(np.int64, copy=False)


def _validate_geometry_inputs(
    sites: ArrayLike,
    domain_points: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    site_array = _as_finite_float_matrix(sites, name="sites", minimum_rows=2)
    point_array = _as_finite_float_matrix(
        domain_points,
        name="domain_points",
        minimum_rows=1,
    )
    if point_array.shape[1] != site_array.shape[1]:
        raise ValueError("sites and domain_points must have the same dimension")
    return site_array, point_array


def _as_finite_float_matrix(
    values: ArrayLike,
    *,
    name: str,
    minimum_rows: int,
) -> FloatArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric two-dimensional array") from error
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric two-dimensional array") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] < minimum_rows or array.shape[1] == 0:
        raise ValueError(
            f"{name} must have at least {minimum_rows} row(s) and one column"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_finite_float_vector(values: ArrayLike, *, name: str) -> FloatArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from error
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _squared_euclidean_distances(
    domain_points: FloatArray,
    sites: FloatArray,
) -> FloatArray:
    point_norm = np.einsum("ij,ij->i", domain_points, domain_points)[:, None]
    site_norm = np.einsum("ij,ij->i", sites, sites)[None, :]
    squared = point_norm + site_norm - 2.0 * (domain_points @ sites.T)
    if not np.isfinite(squared).all():
        raise ValueError("Euclidean-distance computation produced non-finite values")
    # Roundoff can make an exact squared distance minutely negative.
    np.maximum(squared, 0.0, out=squared)
    return squared


def _pairwise_site_distances(sites: FloatArray) -> FloatArray:
    distances = pdist(sites, metric="euclidean")
    if not np.isfinite(distances).all():
        raise ValueError("pairwise site-distance computation produced non-finite values")
    return distances


def _representation_sum(sites: FloatArray, centroids: FloatArray) -> float:
    distances = np.linalg.norm(sites - centroids, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("Representation-distance computation produced non-finite values")
    return float(distances.sum(dtype=np.float64))


def _require_all_cells(assignments: IntArray, *, n_sites: int) -> None:
    counts = np.bincount(assignments, minlength=n_sites)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise ValueError(
            "ordinary nearest-site partition contains empty cells for site "
            f"indices {empty.tolist()}"
        )


def _cell_centroids(
    domain_points: FloatArray,
    assignments: IntArray,
    *,
    n_sites: int,
) -> FloatArray:
    _require_all_cells(assignments, n_sites=n_sites)
    centroids = np.vstack(
        [domain_points[assignments == index].mean(axis=0) for index in range(n_sites)]
    )
    if not np.isfinite(centroids).all():
        raise ValueError("cell-centroid computation produced non-finite values")
    return centroids


__all__ = [
    "ContrastRepresentationScores",
    "MultinomialPowerDiagram",
    "assign_domain_points_by_power_distance",
    "assign_domain_points_to_nearest_sites",
    "contrast_representation_scores",
    "contrast_sum_pairwise_distances",
    "mean_pairwise_site_distance",
    "mean_site_to_cell_centroid_distance",
    "multinomial_coefficients_to_power_diagram",
    "representation_sum_site_to_cell_centroid_distances",
]
