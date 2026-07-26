"""Tests for pynerve/algorithms/__init__.py -- TDA algorithmic primitives."""

from __future__ import annotations

import numpy as np
import pytest

from pynerve.algorithms import (
    gaussian_kernel_matrix,
    knn,
    pairwise_distances,
    persistence_heat_vector,
    persistence_image,
    persistence_landscape,
    persistence_silhouette,
)
from pynerve.exceptions import InvalidArgumentError, ValidationError


class TestPairwiseDistances:
    def test_euclidean(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = pairwise_distances(points, metric="euclidean")
        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result), 0.0)
        assert result[0, 1] > 0

    def test_manhattan(self):
        points = np.array([[0.0, 0.0], [3.0, 4.0]])
        result = pairwise_distances(points, metric="cityblock")
        assert result.shape == (2, 2)
        assert result[0, 1] == pytest.approx(7.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValidationError, match="2-D"):
            pairwise_distances(np.array([1.0, 2.0]))

    def test_non_numpy_raises(self):
        with pytest.raises(ValidationError, match="2-D"):
            pairwise_distances([1.0, 2.0])


class TestKNN:
    def test_basic(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        dist, idx = knn(points, k=2)
        assert dist.shape == (3, 2)
        assert idx.shape == (3, 2)

    def test_brute_force(self):
        points = np.random.rand(20, 5)
        dist, idx = knn(points, k=3, algorithm="brute_force")
        assert dist.shape == (20, 3)

    def test_kd_tree(self):
        points = np.random.rand(20, 5)
        dist, idx = knn(points, k=3, algorithm="kd_tree")
        assert dist.shape == (20, 3)

    def test_k_less_than_1_raises(self):
        with pytest.raises(InvalidArgumentError, match="k must be >= 1"):
            knn(np.random.rand(10, 3), k=0)

    def test_invalid_algorithm_raises(self):
        with pytest.raises(InvalidArgumentError, match="algorithm"):
            knn(np.random.rand(10, 3), algorithm="invalid")

    def test_non_2d_raises(self):
        with pytest.raises(ValidationError, match="2-D"):
            knn(np.array([1.0]))


class TestPersistenceLandscape:
    def test_basic(self):
        diagram = np.array([[0.0, 1.0], [2.0, 5.0], [0.0, 3.0]])
        result = persistence_landscape(diagram)
        assert result.shape == (5, 100)

    def test_custom_params(self):
        diagram = np.array([[0.0, 2.0]])
        result = persistence_landscape(diagram, num_levels=3, resolution=50)
        assert result.shape == (3, 50)

    def test_empty_after_filter(self):
        diagram = np.array([[0.0, np.inf], [0.0, np.inf]])
        result = persistence_landscape(diagram)
        assert result.shape == (5, 100)
        assert np.all(result == 0.0)


class TestPersistenceImage:
    def test_basic(self):
        diagram = np.array([[0.0, 1.0, 0], [0.5, 2.0, 0]])
        result = persistence_image(diagram)
        assert result.shape == (64, 64)

    def test_custom_resolution(self):
        diagram = np.array([[0.0, 1.0]])
        result = persistence_image(diagram, resolution=32)
        assert result.shape == (32, 32)


class TestPersistenceSilhouette:
    def test_basic(self):
        diagram = np.array([[0.0, 1.0], [2.0, 5.0]])
        result = persistence_silhouette(diagram)
        assert result.shape == (100,)

    def test_empty_after_filter(self):
        diagram = np.array([[0.0, np.inf]])
        result = persistence_silhouette(diagram)
        assert result.shape == (100,)
        assert np.all(result == 0.0)

    def test_custom_resolution(self):
        diagram = np.array([[0.0, 2.0]])
        result = persistence_silhouette(diagram, resolution=50, weight_power=2.0)
        assert result.shape == (50,)


class TestPersistenceHeatVector:
    def test_basic(self):
        diagram = np.array([[0.0, 1.0], [2.0, 5.0]])
        result = persistence_heat_vector(diagram)
        assert result.shape == (100,)

    def test_empty_after_filter(self):
        diagram = np.array([[0.0, np.inf]])
        result = persistence_heat_vector(diagram)
        assert np.all(result == 0.0)

    def test_custom_params(self):
        diagram = np.array([[0.0, 2.0]])
        result = persistence_heat_vector(diagram, resolution=50, sigma=2.0, t=0.5)
        assert result.shape == (50,)


class TestGaussianKernelMatrix:
    def test_same_diagram(self):
        d1 = np.array([[0.0, 1.0], [2.0, 5.0]])
        result = gaussian_kernel_matrix(d1)
        assert result.shape == (2, 2)
        assert np.all(np.diag(result) > 0)

    def test_two_diagrams(self):
        d1 = np.array([[0.0, 1.0]])
        d2 = np.array([[0.5, 2.0], [0.0, 3.0]])
        result = gaussian_kernel_matrix(d1, d2)
        assert result.shape == (1, 2)

    def test_empty_after_filter(self):
        d1 = np.array([[0.0, np.inf]])
        d2 = np.array([[0.0, np.inf]])
        result = gaussian_kernel_matrix(d1, d2)
        assert result.shape == (0, 0)
