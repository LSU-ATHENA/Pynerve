"""Tests for torch/_vectorization_basis.py — persistence images, landscapes, silhouettes."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch._vectorization_basis import (
    _finite_birth_death,
    _validate_diagram,
    _validate_positive_finite,
    _validate_range,
    adaptive_persistence_image,
    persistence_image,
    persistence_landscape,
    persistence_silhouette,
)


# ── _validate_diagram ──────────────────────────────────────────────────────


class TestValidateDiagram:
    def test_valid(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = _validate_diagram(d)
        assert result is d

    def test_not_tensor_raises(self):
        import numpy as np
        with pytest.raises(Exception, match="torch.Tensor"):
            _validate_diagram(np.array([[0.0, 1.0, 0]]))  # type: ignore[arg-type]

    def test_1d_raises(self):
        with pytest.raises(Exception, match="2D"):
            _validate_diagram(torch.tensor([0.0, 1.0], dtype=torch.float32))

    def test_wrong_dtype_raises(self):
        with pytest.raises(Exception, match="floating-point"):
            _validate_diagram(torch.tensor([[0, 1, 0]]))

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = _validate_diagram(d)
        assert result.numel() == 0


# ── _validate_positive_finite ──────────────────────────────────────────────


class TestValidatePositiveFinite:
    def test_valid(self):
        assert _validate_positive_finite(1.0, "x") == 1.0

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_positive_finite(0.0, "x")

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_positive_finite(float("nan"), "x")


# ── _validate_range ────────────────────────────────────────────────────────


class TestValidateRange:
    def test_valid(self):
        assert _validate_range((0.0, 10.0), "x") == (0.0, 10.0)

    def test_none(self):
        assert _validate_range(None, "x") is None

    def test_lower_above_upper_raises(self):
        with pytest.raises(ValueError, match="minimum must not exceed"):
            _validate_range((10.0, 0.0), "x")

    def test_single_element_raises(self):
        with pytest.raises(ValueError, match="exactly two"):
            _validate_range((1.0,), "x")  # type: ignore[arg-type]

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_range((float("inf"), 10.0), "x")


# ── _finite_birth_death ────────────────────────────────────────────────────


class TestFiniteBirthDeath:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, 3.0, 0]], dtype=torch.float32)
        births, deaths = _finite_birth_death(d)
        assert births.shape[0] == 2
        assert deaths.shape[0] == 2

    def test_infinite_death_removed(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, float("inf"), 0]], dtype=torch.float32)
        births, deaths = _finite_birth_death(d)
        assert births.shape[0] == 1

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        births, deaths = _finite_birth_death(d)
        assert births.numel() == 0


# ── persistence_image ──────────────────────────────────────────────────────


class TestPersistenceImage:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = persistence_image(d, resolution=(10, 10))
        assert result.shape == (10, 10)
        assert result.dtype == torch.float32

    def test_custom_resolution(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = persistence_image(d, resolution=(5, 8))
        assert result.shape == (5, 8)

    def test_empty_diagram_returns_zeros(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = persistence_image(d, resolution=(5, 5))
        assert torch.all(result == 0)

    def test_weight_constant(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, 4.0, 0]], dtype=torch.float32)
        result = persistence_image(d, weight_fn="constant")
        assert result.shape == (20, 20)

    def test_weight_linear(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, 4.0, 0]], dtype=torch.float32)
        result = persistence_image(d, weight_fn="linear")
        assert torch.any(result > 0)

    def test_normalize_false(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = persistence_image(d, normalize=False)
        assert result.sum() > 1.0  # not normalized

    def test_custom_sigma(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = persistence_image(d, sigma=0.5)
        assert result.shape == (20, 20)

    def test_invalid_weight_fn_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="weight_fn"):
            persistence_image(d, weight_fn="bad")  # type: ignore[arg-type]

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 3.0, 0]]], dtype=torch.float32
        )
        result = persistence_image(d, resolution=(8, 8))
        assert result.dim() == 3
        assert result.shape == (2, 8, 8)


# ── adaptive_persistence_image ─────────────────────────────────────────────


class TestAdaptivePersistenceImage:
    def test_basic(self):
        d = torch.tensor([[0.0, 1.0, 0], [2.0, 5.0, 0]], dtype=torch.float32)
        result = adaptive_persistence_image(d, target_resolution=10)
        assert result.dim() == 2
        assert result.shape[0] > 0

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = adaptive_persistence_image(d)
        assert torch.all(result == 0)

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 3.0, 0]]], dtype=torch.float32
        )
        result = adaptive_persistence_image(d)
        assert result.dim() == 3

    def test_custom_sigma_range(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = adaptive_persistence_image(d, min_sigma=0.5, max_sigma=5.0)
        assert result.dim() == 2

    def test_invalid_sigma_range_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="min_sigma"):
            adaptive_persistence_image(d, min_sigma=5.0, max_sigma=0.5)


# ── persistence_landscape ──────────────────────────────────────────────────


class TestPersistenceLandscape:
    def test_basic(self):
        d = torch.tensor([[0.0, 2.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = persistence_landscape(d, k=3, num_samples=50)
        assert result.shape == (3, 50)

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = persistence_landscape(d)
        assert torch.all(result == 0)

    def test_fewer_features_than_k(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = persistence_landscape(d, k=5, num_samples=10)
        assert result.shape == (5, 10)

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[1.0, 2.0, 0]]], dtype=torch.float32
        )
        result = persistence_landscape(d, k=2, num_samples=10)
        assert result.dim() == 3
        assert result.shape == (2, 2, 10)

    def test_custom_x_range(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = persistence_landscape(d, x_range=(0.0, 5.0))
        assert result.shape[1] == 100


# ── persistence_silhouette ─────────────────────────────────────────────────


class TestPersistenceSilhouette:
    def test_basic(self):
        d = torch.tensor([[0.0, 2.0, 0], [1.0, 3.0, 0]], dtype=torch.float32)
        result = persistence_silhouette(d, num_samples=50)
        assert result.shape == (50,)
        assert torch.any(result > 0)

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = persistence_silhouette(d)
        assert torch.all(result == 0)

    def test_weight_constant(self):
        d = torch.tensor([[0.0, 2.0, 0]], dtype=torch.float32)
        result = persistence_silhouette(d, weight_fn="constant")
        assert result.shape == (100,)

    def test_batched_3d(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[1.0, 2.0, 0]]], dtype=torch.float32
        )
        result = persistence_silhouette(d, num_samples=20)
        assert result.dim() == 2
        assert result.shape == (2, 20)

    def test_invalid_weight_fn_raises(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="weight_fn"):
            persistence_silhouette(d, weight_fn="bad")  # type: ignore[arg-type]
