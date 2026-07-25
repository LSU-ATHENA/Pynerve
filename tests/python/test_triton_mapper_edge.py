"""Edge-case tests for triton/_mapper.py CPU fallback paths.

Covers boundary conditions: single-point, k=1, resolution=1, overlap=0,
max_cover_size edge cases, no-overlap nerves, empty nodes.
"""

from __future__ import annotations

import warnings

import pytest
import torch

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestDensityFilterEdge:
    def test_single_point(self):
        from pynerve.triton._mapper import density_filter

        pts = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        result = density_filter(pts, k_neighbors=10)
        assert result.shape == (1,)
        assert result[0] >= 0

    def test_all_identical_points(self):
        from pynerve.triton._mapper import density_filter

        pts = torch.ones(5, 3, dtype=torch.float32)
        result = density_filter(pts, k_neighbors=2)
        assert result.shape == (5,)
        assert torch.isfinite(result).all()

    def test_large_k_neighbors(self):
        from pynerve.triton._mapper import density_filter

        pts = torch.randn(5, 2, dtype=torch.float32)
        # k > n_points should still work (cpu fallback uses n_points-1)
        result = density_filter(pts, k_neighbors=100)
        assert (result >= 0).all()


class TestEccentricityFilterEdge:
    def test_single_point(self):
        from pynerve.triton._mapper import eccentricity_filter

        pts = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        result = eccentricity_filter(pts)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_two_points(self):
        from pynerve.triton._mapper import eccentricity_filter

        pts = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32)
        result = eccentricity_filter(pts)
        assert result[0] == pytest.approx(5.0, abs=1e-5)
        assert result[1] == pytest.approx(5.0, abs=1e-5)


class TestKmeansPlusPlusEdge:
    def test_k_equals_one(self):
        from pynerve.triton._mapper import kmeans_plusplus_init

        pts = torch.randn(20, 3, dtype=torch.float32)
        centroids = kmeans_plusplus_init(pts, k=1, seed=42)
        assert centroids.shape == (1, 3)

    def test_k_equals_n(self):
        from pynerve.triton._mapper import kmeans_plusplus_init

        pts = torch.randn(5, 2, dtype=torch.float32)
        centroids = kmeans_plusplus_init(pts, k=5, seed=42)
        assert centroids.shape == (5, 2)

    def test_different_seeds(self):
        from pynerve.triton._mapper import kmeans_plusplus_init

        pts = torch.randn(50, 3, dtype=torch.float32)
        c1 = kmeans_plusplus_init(pts, k=3, seed=1)
        c2 = kmeans_plusplus_init(pts, k=3, seed=999)
        assert c1.shape == c2.shape


class TestKmeansAssignEdge:
    def test_many_centroids(self):
        from pynerve.triton._mapper import kmeans_assign

        pts = torch.randn(10, 2, dtype=torch.float32)
        centroids = torch.randn(5, 2, dtype=torch.float32)
        labels = kmeans_assign(pts, centroids)
        assert labels.shape == (10,)
        assert labels.dtype == torch.int32

    def test_single_centroid(self):
        from pynerve.triton._mapper import kmeans_assign

        pts = torch.randn(10, 3, dtype=torch.float32)
        centroids = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        labels = kmeans_assign(pts, centroids)
        assert labels.shape == (10,)
        assert (labels == 0).all()


class TestKmeansClusterEdge:
    def test_max_iter_zero(self):
        from pynerve.triton._mapper import kmeans_cluster

        pts = torch.randn(20, 3, dtype=torch.float32)
        labels = kmeans_cluster(pts, k=3, max_iter=0, seed=42)
        assert labels.shape == (20,)

    def test_k_equals_one(self):
        from pynerve.triton._mapper import kmeans_cluster

        pts = torch.randn(15, 2, dtype=torch.float32)
        labels = kmeans_cluster(pts, k=1, max_iter=5, seed=42)
        assert labels.shape == (15,)
        assert (labels == 0).all()


class TestBuildCoverEdge:
    def test_resolution_one(self):
        from pynerve.triton._mapper import build_cover

        filter_values = torch.tensor([[0.0], [0.1], [0.9]], dtype=torch.float32)
        sizes, indices = build_cover(filter_values, resolution=1, overlap=0.5)
        assert sizes.shape == (3,)
        assert (sizes <= 1).all()  # at most 1 bin

    def test_overlap_zero(self):
        from pynerve.triton._mapper import build_cover

        filter_values = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)
        sizes, indices = build_cover(filter_values, resolution=5, overlap=0.0)
        assert sizes.shape == (3,)

    def test_2d_filter(self):
        from pynerve.triton._mapper import build_cover

        filter_values = torch.tensor(
            [[0.1, 0.2], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float32
        )
        sizes, indices = build_cover(filter_values, resolution=3, overlap=0.3)
        assert sizes.shape == (3,)

    def test_falls_back_with_warning(self):
        from pynerve.triton._mapper import build_cover

        filter_values = torch.tensor([[0.3]], dtype=torch.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sizes, indices = build_cover(filter_values, resolution=3, overlap=0.2)
            assert len(w) >= 1
            assert "build_cover" in str(w[0].message)


class TestComputeNerveEdgesEdge:
    def test_empty_nodes(self):
        from pynerve.triton._mapper import compute_nerve_edges

        # 0 nodes
        cover_sets = torch.tensor([], dtype=torch.int32)
        starts = torch.tensor([], dtype=torch.int32)
        sizes = torch.tensor([], dtype=torch.int32)
        edges = compute_nerve_edges(cover_sets, starts, sizes, max_edges=10)
        assert edges.shape[0] == 0

    def test_fully_connected(self):
        from pynerve.triton._mapper import compute_nerve_edges

        # All 3 nodes share the same cover index
        cover_sets = torch.tensor([0, 0, 0], dtype=torch.int32)
        starts = torch.tensor([0, 1, 2], dtype=torch.int32)
        sizes = torch.tensor([1, 1, 1], dtype=torch.int32)
        edges = compute_nerve_edges(cover_sets, starts, sizes, max_edges=20)
        assert edges.shape[0] == 3  # 3 pairs: (0,1), (0,2), (1,2)

    def test_falls_back_with_warning(self):
        from pynerve.triton._mapper import compute_nerve_edges

        cover_sets = torch.tensor([0, 1], dtype=torch.int32)
        starts = torch.tensor([0, 1], dtype=torch.int32)
        sizes = torch.tensor([1, 1], dtype=torch.int32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            edges = compute_nerve_edges(cover_sets, starts, sizes, max_edges=5)
            assert len(w) >= 1
            assert "compute_nerve_edges" in str(w[0].message)
