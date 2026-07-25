"""Edge-case tests for triton/_nn_ops.py and triton/_persistence.py CPU fallback paths.

Covers: diagram_conv1d kernel_size edge cases, persistence_image edge cases,
_bounds_and_valid all-invalid, _persistence_image_cpu single-point and resolution=1.
"""

from __future__ import annotations

import warnings

import pytest
import torch

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestDiagramConv1dEdge:
    def test_kernel_size_equals_n_pairs(self):
        """kernel_size = n_pairs produces output_len=1 for 'valid' conv."""
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 5, dtype=torch.float32)
        kernel = torch.randn(4, 3, 5, dtype=torch.float32)
        bias = torch.zeros(4, dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias, activation="none")
        assert out.shape == (2, 4, 1)

    def test_kernel_size_one(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 10, dtype=torch.float32)
        kernel = torch.randn(4, 3, 1, dtype=torch.float32)
        bias = torch.zeros(4, dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias)
        assert out.shape == (2, 4, 10)

    def test_single_batch(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(1, 3, 8, dtype=torch.float32)
        kernel = torch.randn(2, 3, 3, dtype=torch.float32)
        bias = torch.zeros(2, dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias)
        assert out.shape == (1, 2, 6)

    def test_single_channel(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 1, 10, dtype=torch.float32)
        kernel = torch.randn(3, 1, 3, dtype=torch.float32)
        bias = torch.zeros(3, dtype=torch.float32)
        out = diagram_conv1d(features, kernel, bias)
        assert out.shape == (2, 3, 8)

    def test_falls_back_with_warning(self):
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 2, 5, dtype=torch.float32)
        kernel = torch.randn(1, 2, 2, dtype=torch.float32)
        bias = torch.zeros(1, dtype=torch.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = diagram_conv1d(features, kernel, bias, activation="relu")
            assert len(w) >= 1
            assert "diagram_conv1d" in str(w[0].message)

    def test_none_activation_falls_back(self):
        """Activation='none' produces output via CPU fallback."""
        from pynerve.triton._nn_ops import diagram_conv1d

        features = torch.randn(2, 3, 8, dtype=torch.float32)
        kernel = torch.randn(2, 3, 3, dtype=torch.float32)
        bias = torch.zeros(2, dtype=torch.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = diagram_conv1d(features, kernel, bias, activation="none")
            assert out.shape == (2, 2, 6)


class TestPersistenceImageEdge:
    def test_single_point(self):
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([0.0], dtype=torch.float32)
        deaths = torch.tensor([5.0], dtype=torch.float32)
        img = persistence_image_from_diagram(births, deaths, resolution=32, sigma=1.0)
        assert img.shape == (32, 32)
        assert img.sum() > 0

    def test_resolution_one(self):
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([0.0, 1.0], dtype=torch.float32)
        deaths = torch.tensor([3.0, 4.0], dtype=torch.float32)
        img = persistence_image_from_diagram(births, deaths, resolution=1, sigma=0.5)
        assert img.shape == (1, 1)
        assert img.sum() >= 0

    def test_all_same_birth_death(self):
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        deaths = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
        img = persistence_image_from_diagram(births, deaths, resolution=16, sigma=1.0)
        assert img.shape == (16, 16)

    def test_bounds_all_invalid(self):
        """When all pairs have death <= birth, _bounds_and_valid returns empty."""
        from pynerve.triton._persistence import _bounds_and_valid

        births = torch.tensor([1.0, 2.0], dtype=torch.float32)
        deaths = torch.tensor([1.0, 1.0], dtype=torch.float32)  # death <= birth
        b_valid, d_valid, x_min, x_max, y_min, y_max, *_ = _bounds_and_valid(births, deaths)
        assert b_valid.numel() == 0
        assert x_min < x_max
        assert y_min < y_max

    def test_bounds_identical_pairs(self):
        """All pairs identical => b_max == b_min needs EPS guard."""
        from pynerve.triton._persistence import _bounds_and_valid

        births = torch.tensor([1.0, 1.0], dtype=torch.float32)
        deaths = torch.tensor([3.0, 3.0], dtype=torch.float32)
        b_valid, d_valid, x_min, x_max, y_min, y_max, *_ = _bounds_and_valid(
            births, deaths
        )
        assert b_valid.numel() == 2
        assert x_min < x_max
        assert y_min < y_max

    def test_all_invalid_returns_empty(self):
        """persistence_image_from_diagram returns zeros when no valid pairs."""
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([1.0], dtype=torch.float32)
        deaths = torch.tensor([0.5], dtype=torch.float32)  # death < birth → invalid
        img = persistence_image_from_diagram(births, deaths, resolution=8)
        assert img.shape == (8, 8)
        assert img.sum() == 0.0

    def test_persistence_image_cpu_single(self):
        from pynerve.triton._persistence import _persistence_image_cpu

        b = torch.tensor([0.0], dtype=torch.float32)
        d = torch.tensor([5.0], dtype=torch.float32)
        img = _persistence_image_cpu(b, d, 32, 1.0, -0.5, 5.5, -0.5, 5.5)
        assert img.shape == (32, 32)
        assert img.sum() > 0

    def test_persistence_image_cpu_resolution_one(self):
        from pynerve.triton._persistence import _persistence_image_cpu

        b = torch.tensor([1.0, 2.0], dtype=torch.float32)
        d = torch.tensor([3.0, 4.0], dtype=torch.float32)
        img = _persistence_image_cpu(b, d, 1, 0.5, 0.0, 5.0, 0.0, 5.0)
        assert img.shape == (1, 1)

    def test_select_strategy_boundary(self):
        from pynerve.triton._persistence import _select_strategy

        # n_pairs == resolution^2 → pair strategy
        assert _select_strategy(4096, 64) == "pair"  # 64*64 = 4096
        # n_pairs > resolution^2 → pixel strategy
        assert _select_strategy(4097, 64) == "pixel"

    def test_falls_back_with_warning(self):
        from pynerve.triton._persistence import persistence_image_from_diagram

        births = torch.tensor([0.0, 1.0], dtype=torch.float32)
        deaths = torch.tensor([3.0, 4.0], dtype=torch.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            img = persistence_image_from_diagram(births, deaths, resolution=8)
            assert len(w) >= 1
            assert "persistence_image_from_diagram" in str(w[0].message)
