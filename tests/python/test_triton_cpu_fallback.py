"""Tests for triton module CPU fallback paths.

All triton kernels have CPU fallback paths that activate when tensors
are on CPU (via _use_triton() returning False). These tests exercise
those fallback paths without requiring a GPU.
"""

from __future__ import annotations

import warnings

import pytest
import torch

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestTritonInit:
    def test_check_triton(self):
        from pynerve.triton import _check_triton

        result = _check_triton()
        assert isinstance(result, bool)

    def test_use_triton_cpu_tensor(self):
        from pynerve.triton import _use_triton

        t = torch.randn(10, 3)
        assert _use_triton(t) is False

    def test_warn_cpu_fallback(self):
        from pynerve.triton import _warn_cpu_fallback

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_cpu_fallback("test_op")
            assert len(w) == 1
            assert "test_op" in str(w[0].message)
            assert "fallback" in str(w[0].message).lower()


class TestTritonDistanceFallback:
    def test_pairwise_euclidean_self(self):
        from pynerve.triton._distance import pairwise_euclidean

        a = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32)
        result = pairwise_euclidean(a)
        assert result.shape == (2, 2)
        assert torch.isclose(result[0, 1], torch.tensor(5.0), rtol=1e-5)

    def test_pairwise_euclidean_two_arrays(self):
        from pynerve.triton._distance import pairwise_euclidean

        a = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        b = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        result = pairwise_euclidean(a, b)
        assert result.shape == (1, 1)
        assert torch.isclose(result[0, 0], torch.tensor(5.0), rtol=1e-5)

    def test_pairwise_euclidean_dim_mismatch(self):
        from pynerve.triton._distance import pairwise_euclidean

        a = torch.randn(5, 3)
        b = torch.randn(5, 4)
        with pytest.raises(ValueError, match="dimension"):
            pairwise_euclidean(a, b)

    def test_compute_norms(self):
        from pynerve.triton._distance import compute_norms

        points = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32)
        norms = compute_norms(points)
        assert norms.shape == (2,)
        assert torch.isclose(norms[0], torch.tensor(25.0))
        assert norms[1] == 0.0


class TestTritonPersistenceFallback:
    def test_persistence_image_basic(self):
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        deaths = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
        img = persistence_image_from_diagram(births, deaths, resolution=16, sigma=1.0)
        assert img.shape == (16, 16)
        assert img.sum() > 0

    def test_persistence_image_empty(self):
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([], dtype=torch.float32)
        deaths = torch.tensor([], dtype=torch.float32)
        img = persistence_image_from_diagram(births, deaths, resolution=8)
        assert img.shape == (8, 8)
        assert img.sum() == 0.0

    def test_persistence_image_single_pair(self):
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([0.0], dtype=torch.float32)
        deaths = torch.tensor([5.0], dtype=torch.float32)
        img = persistence_image_from_diagram(births, deaths, resolution=8, sigma=2.0)
        assert img.shape == (8, 8)
        assert img.sum() > 0

    def test_select_strategy_pixel(self):
        from pynerve.triton._persistence import _select_strategy

        assert _select_strategy(10000, 64) == "pixel"

    def test_select_strategy_pair(self):
        from pynerve.triton._persistence import _select_strategy

        assert _select_strategy(10, 64) == "pair"

    def test_bounds_and_valid(self):
        from pynerve.triton._persistence import _bounds_and_valid

        births = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        deaths = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
        b_valid, d_valid, x_min, x_max, y_min, y_max, xr, yr, bm, pm = _bounds_and_valid(
            births, deaths
        )
        assert b_valid.numel() == 3
        assert x_min < x_max
        assert y_min < y_max

    def test_bounds_filters_invalid(self):
        from pynerve.triton._persistence import _bounds_and_valid

        births = torch.tensor([0.0, 1.0], dtype=torch.float32)
        deaths = torch.tensor([3.0, 1.0], dtype=torch.float32)  # death <= birth for 2nd
        b_valid, d_valid, *_ = _bounds_and_valid(births, deaths)
        assert b_valid.numel() == 1  # only first pair is valid

    def test_persistence_image_cpu(self):
        from pynerve.triton._persistence import _persistence_image_cpu

        b = torch.tensor([0.0, 2.0], dtype=torch.float32)
        d = torch.tensor([3.0, 5.0], dtype=torch.float32)
        img = _persistence_image_cpu(b, d, 16, 1.0, -0.2, 2.2, -0.3, 3.3)
        assert img.shape == (16, 16)
        assert img.sum() > 0


class TestTritonWassersteinFallback:
    def test_linf_distance(self):
        from pynerve.triton._wasserstein import _linf

        a_x = torch.tensor([0.0, 3.0])
        a_y = torch.tensor([0.0, 4.0])
        b_x = torch.tensor([0.0])
        b_y = torch.tensor([0.0])
        result = _linf(a_x, a_y, b_x, b_y)
        assert result.shape == (2, 1)
        assert result[0, 0] == 0.0
        assert result[1, 0] == 4.0  # max(|3-0|, |4-0|) = 4

    def test_build_cost_matrix(self):
        from pynerve.triton._wasserstein import build_cost_matrix

        d1_x = torch.tensor([0.0, 1.0], dtype=torch.float32)
        d1_y = torch.tensor([0.0, 2.0], dtype=torch.float32)
        d2_x = torch.tensor([0.0], dtype=torch.float32)
        d2_y = torch.tensor([0.0], dtype=torch.float32)
        cost = build_cost_matrix(d1_x, d1_y, d2_x, d2_y, p=2.0)
        assert cost.shape == (2, 1)

    def test_build_cost_matrix_empty(self):
        from pynerve.triton._wasserstein import build_cost_matrix

        d1_x = torch.tensor([], dtype=torch.float32)
        d1_y = torch.tensor([], dtype=torch.float32)
        d2_x = torch.tensor([0.0], dtype=torch.float32)
        d2_y = torch.tensor([0.0], dtype=torch.float32)
        cost = build_cost_matrix(d1_x, d1_y, d2_x, d2_y, p=1.0)
        assert cost.shape == (0, 1)

    def test_sinkhorn_kernel_matrix(self):
        from pynerve.triton._wasserstein import sinkhorn_kernel_matrix

        d1_x = torch.tensor([0.0, 1.0], dtype=torch.float32)
        d1_y = torch.tensor([0.0, 2.0], dtype=torch.float32)
        d2_x = torch.tensor([0.0, 0.5], dtype=torch.float32)
        d2_y = torch.tensor([0.0, 1.0], dtype=torch.float32)
        kernel = sinkhorn_kernel_matrix(d1_x, d1_y, d2_x, d2_y, reg=0.1)
        assert kernel.shape == (2, 2)
        assert (kernel >= 0).all()

    def test_sinkhorn_distance(self):
        from pynerve.triton._wasserstein import sinkhorn_distance

        d1_x = torch.tensor([0.0, 1.0], dtype=torch.float32)
        d1_y = torch.tensor([0.0, 2.0], dtype=torch.float32)
        d2_x = torch.tensor([0.0, 0.5], dtype=torch.float32)
        d2_y = torch.tensor([0.0, 1.0], dtype=torch.float32)
        dist = sinkhorn_distance(d1_x, d1_y, d2_x, d2_y, p=2.0, reg=0.1, max_iter=10)
        assert isinstance(dist, float)
        assert dist >= 0.0

    def test_sinkhorn_distance_empty(self):
        from pynerve.triton._wasserstein import sinkhorn_distance

        d1_x = torch.tensor([], dtype=torch.float32)
        d1_y = torch.tensor([], dtype=torch.float32)
        d2_x = torch.tensor([0.0], dtype=torch.float32)
        d2_y = torch.tensor([0.0], dtype=torch.float32)
        dist = sinkhorn_distance(d1_x, d1_y, d2_x, d2_y)
        assert dist == 0.0

    def test_sinkhorn_row_normalise(self):
        from pynerve.triton._wasserstein import _sinkhorn_row_normalise

        kernel = torch.ones(3, 3, dtype=torch.float32)
        u = torch.ones(3, dtype=torch.float32)
        v = torch.ones(3, dtype=torch.float32)
        result = _sinkhorn_row_normalise(kernel, u, v)
        assert result.shape == (3, 3)

    def test_sinkhorn_col_normalise(self):
        from pynerve.triton._wasserstein import _sinkhorn_col_normalise

        kernel = torch.ones(3, 3, dtype=torch.float32)
        u = torch.ones(3, dtype=torch.float32)
        v = torch.ones(3, dtype=torch.float32)
        result = _sinkhorn_col_normalise(kernel, u, v)
        assert result.shape == (3, 3)


class TestTritonLaplacianFallback:
    def test_csr_spmv(self):
        from pynerve.triton._laplacian import csr_spmv

        # 3x3 matrix: [[1,0,0],[0,2,0],[0,0,3]]
        row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        col_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        x = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        y = csr_spmv(row_offsets, col_indices, values, x)
        assert y.shape == (3,)
        assert torch.allclose(y, torch.tensor([1.0, 2.0, 3.0]))

    def test_axpy(self):
        from pynerve.triton._laplacian import axpy

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([10.0, 20.0, 30.0])
        result = axpy(2.0, x, y)
        assert torch.allclose(result, torch.tensor([12.0, 24.0, 36.0]))

    def test_scale(self):
        from pynerve.triton._laplacian import scale

        x = torch.tensor([1.0, 2.0, 3.0])
        result = scale(3.0, x)
        assert torch.allclose(result, torch.tensor([3.0, 6.0, 9.0]))

    def test_orthogonalize(self):
        from pynerve.triton._laplacian import orthogonalize

        v = torch.tensor([1.0, 0.0])
        w = torch.tensor([1.0, 1.0])
        result = orthogonalize(v, w, dot=1.0)
        assert torch.allclose(result, torch.tensor([0.0, 1.0]))


class TestTritonMapperFallback:
    def test_density_filter(self):
        from pynerve.triton._mapper import density_filter

        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        result = density_filter(points, k_neighbors=2)
        assert result.shape == (3,)
        assert (result >= 0).all()

    def test_eccentricity_filter(self):
        from pynerve.triton._mapper import eccentricity_filter

        points = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32)
        result = eccentricity_filter(points)
        assert result.shape == (2,)
        assert result[0] > 0
        assert result[1] > 0

    def test_kmeans_plusplus_init(self):
        from pynerve.triton._mapper import kmeans_plusplus_init

        points = torch.randn(20, 3, dtype=torch.float32)
        centroids = kmeans_plusplus_init(points, k=3, seed=42)
        assert centroids.shape == (3, 3)

    def test_kmeans_assign(self):
        from pynerve.triton._mapper import kmeans_assign

        points = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32)
        centroids = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32)
        labels = kmeans_assign(points, centroids)
        assert labels.shape == (2,)
        assert labels[0] == 0
        assert labels[1] == 1

    def test_kmeans_cluster(self):
        from pynerve.triton._mapper import kmeans_cluster

        points = torch.cat([
            torch.randn(15, 2, dtype=torch.float32) + torch.tensor([0.0, 0.0]),
            torch.randn(15, 2, dtype=torch.float32) + torch.tensor([10.0, 10.0]),
        ])
        labels = kmeans_cluster(points, k=2, max_iter=5, seed=42)
        assert labels.shape == (30,)
        assert len(torch.unique(labels)) <= 2

    def test_build_cover(self):
        from pynerve.triton._mapper import build_cover

        filter_values = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)
        sizes, indices = build_cover(filter_values, resolution=5, overlap=0.3)
        assert sizes.shape == (3,)
        assert indices.shape == (3, 128)
        assert (sizes >= 0).all()

    def test_compute_nerve_edges(self):
        from pynerve.triton._mapper import compute_nerve_edges

        # 3 nodes with overlapping covers
        cover_sets = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
        starts = torch.tensor([0, 1, 2], dtype=torch.int32)
        sizes = torch.tensor([1, 1, 2], dtype=torch.int32)
        edges = compute_nerve_edges(cover_sets, starts, sizes, max_edges=10)
        # Nodes 1 and 2 share cover index 1
        assert edges.shape[1] == 2 or edges.shape[0] == 0

    def test_compute_nerve_edges_no_overlap(self):
        from pynerve.triton._mapper import compute_nerve_edges

        cover_sets = torch.tensor([0, 1, 2], dtype=torch.int32)
        starts = torch.tensor([0, 1, 2], dtype=torch.int32)
        sizes = torch.tensor([1, 1, 1], dtype=torch.int32)
        edges = compute_nerve_edges(cover_sets, starts, sizes, max_edges=10)
        assert edges.shape[0] == 0 or edges.shape == (0,)


class TestTritonNnOpsFallback:
    def test_diagram_conv1d_none(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 10, dtype=torch.float32)
        kernel = torch.randn(4, 3, 3, dtype=torch.float32)
        bias = torch.zeros(4, dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias, activation="none")
        assert out.shape == (2, 4, 8)

    def test_diagram_conv1d_relu(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 10, dtype=torch.float32)
        kernel = torch.randn(4, 3, 3, dtype=torch.float32)
        bias = torch.zeros(4, dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias, activation="relu")
        assert out.shape == (2, 4, 8)
        assert (out >= 0).all()

    def test_diagram_conv1d_sigmoid(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 10, dtype=torch.float32)
        kernel = torch.randn(4, 3, 3, dtype=torch.float32)
        bias = torch.zeros(4, dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias, activation="sigmoid")
        assert out.shape == (2, 4, 8)
        assert (out >= 0).all() and (out <= 1).all()

    def test_diagram_conv1d_cpu(self):
        from pynerve.triton._nn_ops import _diagram_conv1d_cpu

        features = torch.randn(2, 3, 10, dtype=torch.float32)
        kernel = torch.randn(4, 3, 3, dtype=torch.float32)
        bias = torch.zeros(4, dtype=torch.float32)
        out = _diagram_conv1d_cpu(features, kernel, bias, "relu")
        assert out.shape == (2, 4, 8)

    def test_apply_activation_relu(self):
        from pynerve.triton._nn_ops import apply_activation

        x = torch.tensor([-1.0, 0.0, 1.0])
        result = apply_activation(x, "relu")
        assert torch.allclose(result, torch.tensor([0.0, 0.0, 1.0]))

    def test_apply_activation_sigmoid(self):
        from pynerve.triton._nn_ops import apply_activation

        x = torch.tensor([0.0])
        result = apply_activation(x, "sigmoid")
        assert torch.isclose(result, torch.tensor(0.5), rtol=1e-5)

    def test_apply_activation_tanh(self):
        from pynerve.triton._nn_ops import apply_activation

        x = torch.tensor([0.0])
        result = apply_activation(x, "tanh")
        assert torch.isclose(result, torch.tensor(0.0))

    def test_apply_activation_unknown(self):
        from pynerve.triton._nn_ops import apply_activation

        x = torch.tensor([1.0, 2.0])
        result = apply_activation(x, "unknown")
        assert torch.equal(result, x)
