from __future__ import annotations

import numpy as np
import pytest

from frozen_emotion_spaces.geometry import (
    assign_domain_points_by_power_distance,
    assign_domain_points_to_nearest_sites,
    contrast_representation_scores,
    contrast_sum_pairwise_distances,
    mean_pairwise_site_distance,
    mean_site_to_cell_centroid_distance,
    multinomial_coefficients_to_power_diagram,
    representation_sum_site_to_cell_centroid_distances,
)


def test_raw_contrast_sums_every_unordered_euclidean_pair() -> None:
    sites = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    domain = sites.copy()

    assert contrast_sum_pairwise_distances(
        sites,
        domain_points=domain,
    ) == pytest.approx(3.0 + 4.0 + 5.0)
    assert mean_pairwise_site_distance(
        sites,
        domain_points=domain,
    ) == pytest.approx(4.0)


def test_representation_uses_centroids_of_induced_ordinary_cells() -> None:
    sites = np.array([[0.0], [10.0]])
    domain = np.array([[0.0], [2.0], [8.0], [10.0]])

    # Induced cell centroids are 1 and 9; each is one unit from its site.
    assert representation_sum_site_to_cell_centroid_distances(
        sites,
        domain_points=domain,
    ) == pytest.approx(2.0)
    assert mean_site_to_cell_centroid_distance(
        sites,
        domain_points=domain,
    ) == pytest.approx(1.0)

    scores = contrast_representation_scores(sites, domain_points=domain)
    assert scores.contrast_sum == pytest.approx(10.0)
    assert scores.representation_sum == pytest.approx(2.0)
    assert scores.mean_pairwise_site_distance == pytest.approx(10.0)
    assert scores.mean_site_to_cell_centroid_distance == pytest.approx(1.0)
    np.testing.assert_array_equal(scores.assignments, [0, 0, 1, 1])
    np.testing.assert_allclose(scores.centroids, [[1.0], [9.0]])


def test_nearest_site_tie_breaks_by_first_input_site() -> None:
    sites = np.array([[0.0], [2.0]])
    domain = np.array([[1.0], [0.0], [2.0]])

    assignments = assign_domain_points_to_nearest_sites(
        sites,
        domain_points=domain,
    )

    np.testing.assert_array_equal(assignments, [0, 0, 1])


def test_representation_rejects_empty_induced_cells() -> None:
    sites = np.array([[0.0], [10.0]])
    domain = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match=r"empty cells.*\[1\]"):
        representation_sum_site_to_cell_centroid_distances(
            sites,
            domain_points=domain,
        )
    with pytest.raises(ValueError, match="empty cells"):
        contrast_representation_scores(sites, domain_points=domain)


@pytest.mark.parametrize(
    ("sites", "domain", "message"),
    [
        ([0.0, 1.0], [[0.0], [1.0]], "sites must be a two-dimensional"),
        ([[0.0], [1.0]], [0.0, 1.0], "domain_points must be a two-dimensional"),
        ([[0.0], [1.0]], [[0.0, 1.0]], "same dimension"),
        ([[0.0]], [[0.0]], "sites must have at least 2"),
        ([[0.0], [np.nan]], [[0.0]], "sites must contain only finite"),
        ([[0.0], [1.0]], [[np.inf]], "domain_points must contain only finite"),
        ([[0.0], [1.0]], [], "domain_points must be a two-dimensional"),
        ([[0.0], [1.0, 2.0]], [[0.0]], "numeric two-dimensional"),
        ([[0.0j], [1.0j]], [[0.0]], "sites must be real-valued"),
    ],
)
def test_geometry_rejects_bad_shapes_and_nonfinite_values(
    sites,
    domain,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        contrast_representation_scores(sites, domain_points=domain)


def test_domain_points_are_explicit_not_ambient_labels() -> None:
    sites = np.array([[0.0], [1.0]])

    with pytest.raises(TypeError, match="domain_points"):
        contrast_representation_scores(sites)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="domain_points"):
        contrast_sum_pairwise_distances(sites)  # type: ignore[call-arg]


def test_multinomial_sum_zero_gauge_and_power_identity() -> None:
    coefficients = np.array(
        [
            [3.0, 1.0],
            [1.0, 4.0],
            [-1.0, -2.0],
        ]
    )
    intercepts = np.array([1.2, -0.7, 0.4])
    domain = np.array(
        [
            [-2.0, -1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [-1.0, 3.0],
        ]
    )

    diagram = multinomial_coefficients_to_power_diagram(
        coefficients,
        intercepts,
    )

    np.testing.assert_allclose(diagram.gauged_coefficients.sum(axis=0), 0.0)
    assert diagram.gauged_intercepts.sum() == pytest.approx(0.0)
    np.testing.assert_allclose(diagram.sites, diagram.gauged_coefficients / 2.0)
    np.testing.assert_allclose(
        diagram.power_weights,
        diagram.gauged_intercepts
        + np.einsum("ij,ij->i", diagram.sites, diagram.sites),
    )
    gauged_scores = (
        domain @ diagram.gauged_coefficients.T
        + diagram.gauged_intercepts[None, :]
    )
    power_distances = (
        np.sum((domain[:, None, :] - diagram.sites[None, :, :]) ** 2, axis=2)
        - diagram.power_weights[None, :]
    )
    np.testing.assert_allclose(
        power_distances + gauged_scores,
        np.repeat(
            np.einsum("ij,ij->i", domain, domain)[:, None],
            coefficients.shape[0],
            axis=1,
        ),
    )

    linear_assignment = np.argmax(
        domain @ coefficients.T + intercepts[None, :],
        axis=1,
    )
    power_assignment = assign_domain_points_by_power_distance(
        diagram.sites,
        diagram.power_weights,
        domain_points=domain,
    )
    np.testing.assert_array_equal(power_assignment, linear_assignment)


def test_power_distance_ties_choose_first_site() -> None:
    assignments = assign_domain_points_by_power_distance(
        np.array([[0.0], [2.0]]),
        np.array([0.0, 0.0]),
        domain_points=np.array([[1.0]]),
    )

    np.testing.assert_array_equal(assignments, [0])


@pytest.mark.parametrize(
    ("coefficients", "intercepts", "message"),
    [
        ([[1.0, 2.0]], [0.0], "at least 2"),
        ([[1.0], [2.0]], [0.0], "one value per class"),
        ([[1.0], [np.inf]], [0.0, 0.0], "finite"),
        ([[1.0], [2.0]], [0.0, np.nan], "finite"),
    ],
)
def test_power_conversion_rejects_malformed_coefficients(
    coefficients,
    intercepts,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        multinomial_coefficients_to_power_diagram(coefficients, intercepts)


def test_power_assignment_rejects_bad_weight_count() -> None:
    with pytest.raises(ValueError, match="one value per site"):
        assign_domain_points_by_power_distance(
            np.array([[0.0], [1.0]]),
            np.array([0.0]),
            domain_points=np.array([[0.5]]),
        )


def test_finite_inputs_whose_distances_overflow_are_rejected() -> None:
    with pytest.raises(ValueError, match="produced non-finite"):
        contrast_sum_pairwise_distances(
            np.array([[-1.0e308], [1.0e308]]),
            domain_points=np.array([[0.0]]),
        )
