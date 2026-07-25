"""Tests for jit/_cpu_kernels.py — JIT-compiled CPU topology kernels."""

from __future__ import annotations

import numpy as np
import pytest

# Do NOT use mock_gpu_deps — numba must be real for JIT compilation


class TestJitPairwiseDistances:
    def test_basic_2d(self):
        from pynerve.jit._cpu_kernels import _jit_pairwise_distances_impl

        points = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        dists = _jit_pairwise_distances_impl(points)
        assert dists.shape == (3, 3)
        assert np.allclose(np.diag(dists), 0.0)
        # Upper triangle populated, lower triangle is 0 (implementation fills [i,j] not [j,i] back)
        assert np.isclose(dists[0, 1], 5.0, rtol=1e-5)
        assert np.isclose(dists[0, 2], 10.0, rtol=1e-5)

    def test_symmetry(self):
        from pynerve.jit._cpu_kernels import _jit_pairwise_distances_impl

        rng = np.random.default_rng(42)
        points = rng.uniform(-5, 5, (10, 3)).astype(np.float32)
        dists = _jit_pairwise_distances_impl(points)
        np.testing.assert_allclose(dists, dists.T, rtol=1e-5)

    def test_single_point(self):
        from pynerve.jit._cpu_kernels import _jit_pairwise_distances_impl

        points = np.array([[1.0, 2.0]], dtype=np.float32)
        dists = _jit_pairwise_distances_impl(points)
        assert dists.shape == (1, 1)
        assert dists[0, 0] == 0.0

    def test_two_points(self):
        from pynerve.jit._cpu_kernels import _jit_pairwise_distances_impl

        points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        dists = _jit_pairwise_distances_impl(points)
        assert dists.shape == (2, 2)
        assert np.isclose(dists[0, 1], 1.0, rtol=1e-5)


class TestJitFilterPairs:
    def test_basic_threshold(self):
        from pynerve.jit._cpu_kernels import _jit_filter_pairs_impl

        pairs = np.array([[0.0, 1.0], [0.0, 5.0], [0.0, 0.5]], dtype=np.float32)
        mask = _jit_filter_pairs_impl(pairs, threshold=0.9)
        assert mask[0] == True  # persistence = 1.0 > 0.9
        assert mask[1] == True  # persistence = 5.0 > 0.9
        assert mask[2] == False  # persistence = 0.5 < 0.9

    def test_all_above(self):
        from pynerve.jit._cpu_kernels import _jit_filter_pairs_impl

        pairs = np.array([[0.0, 10.0], [0.0, 20.0]], dtype=np.float32)
        mask = _jit_filter_pairs_impl(pairs, threshold=0.5)
        assert mask.all()

    def test_all_below(self):
        from pynerve.jit._cpu_kernels import _jit_filter_pairs_impl

        pairs = np.array([[0.0, 0.5], [0.0, 0.1]], dtype=np.float32)
        mask = _jit_filter_pairs_impl(pairs, threshold=1.0)
        assert not mask.any()


class TestJitBettiCurve:
    def test_basic_curve(self):
        from pynerve.jit._cpu_kernels import _jit_betti_curve_impl

        pairs = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=1, resolution=10)
        assert betti.shape == (2, 10)
        assert betti.dtype == np.int32

    def test_empty_pairs(self):
        from pynerve.jit._cpu_kernels import _jit_betti_curve_impl

        pairs = np.array([[0.0, 1e9, 0.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=2, resolution=5)
        assert betti.shape == (3, 5)

    def test_single_dim(self):
        from pynerve.jit._cpu_kernels import _jit_betti_curve_impl

        pairs = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
        betti = _jit_betti_curve_impl(pairs, max_dim=0, resolution=10)
        assert betti.shape == (1, 10)
        assert betti[0, 0] == 1


class TestJitPersistenceImage:
    def test_basic_image(self):
        from pynerve.jit._cpu_kernels import _jit_persistence_image_impl

        pairs = np.array([[0.0, 5.0], [2.0, 8.0]], dtype=np.float32)
        img = _jit_persistence_image_impl(pairs, resolution=16, sigma=2.0)
        assert img.shape == (16, 16)
        assert img.dtype == np.float32
        assert img.sum() > 0

    def test_single_pair(self):
        from pynerve.jit._cpu_kernels import _jit_persistence_image_impl

        pairs = np.array([[0.0, 5.0]], dtype=np.float32)
        img = _jit_persistence_image_impl(pairs, resolution=8, sigma=2.0)
        assert img.shape == (8, 8)
        assert img.sum() > 0


class TestJitVietorisRipsEdges:
    def test_basic_edges(self):
        from pynerve.jit._cpu_kernels import _jit_vietoris_rips_edges_impl

        points = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
        edges = _jit_vietoris_rips_edges_impl(points, max_dist=1.5)
        assert edges.shape[1] == 2
        # Only (0,1) with dist=1.0 <= 1.5; (1,2) has dist=2.0 > 1.5; (0,2) has dist=3.0 > 1.5
        assert len(edges) == 1

    def test_no_edges(self):
        from pynerve.jit._cpu_kernels import _jit_vietoris_rips_edges_impl

        points = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        edges = _jit_vietoris_rips_edges_impl(points, max_dist=1.0)
        assert len(edges) == 0

    def test_all_edges(self):
        from pynerve.jit._cpu_kernels import _jit_vietoris_rips_edges_impl

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        edges = _jit_vietoris_rips_edges_impl(points, max_dist=2.0)
        assert len(edges) == 3


class TestJitBatchBettiCurves:
    def test_batch(self):
        from pynerve.jit._cpu_kernels import _jit_batch_betti_curves_impl

        diagrams = np.array([
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        ], dtype=np.float32)
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=1, resolution=10)
        assert curves.shape == (2, 2, 10)
        assert curves.dtype == np.int32

    def test_single_batch(self):
        from pynerve.jit._cpu_kernels import _jit_batch_betti_curves_impl

        diagrams = np.array([[[0.0, 5.0, 0.0]]], dtype=np.float32)
        curves = _jit_batch_betti_curves_impl(diagrams, max_dim=0, resolution=5)
        assert curves.shape == (1, 1, 5)
