"""Tests for torch/_persistence_image.py -- persistence_image function and helpers.

Covers weight_fn variants (constant, linear, persistence), batched/empty/inf-death
diagrams, validation errors, and the PersistenceDiagram wrapper path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch


@pytest.fixture(autouse=True)
def _force_python_backend():
    """Patch _torch_backend and _core_backend to return None so the Python fallback is used."""
    with patch("pynerve.torch._persistence_image._torch_backend", return_value=None), \
         patch("pynerve.torch._persistence_image._core_backend", return_value=None):
        yield


class TestPersistenceImageWeightFunctions:
    def test_constant_weight(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(10, 10), sigma=1.0, weight_fn="constant")
        assert img.shape == (10, 10)
        assert img.sum() > 0

    def test_linear_weight(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 2.0], [1.0, 4.0]], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(10, 10), sigma=1.0, weight_fn="linear")
        assert img.shape == (10, 10)
        assert img.sum() > 0

    def test_persistence_weight(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 2.0], [1.0, 4.0]], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(10, 10), sigma=1.0, weight_fn="persistence")
        assert img.shape == (10, 10)
        assert img.sum() > 0

    def test_invalid_weight_fn_raises(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="Unsupported weight_fn"):
            persistence_image(diagram, weight_fn="bogus")


class TestPersistenceImageEdgeCases:
    def test_empty_diagram(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.empty((0, 2), dtype=torch.float32)
        img = persistence_image(diagram, resolution=(8, 8), sigma=1.0)
        assert img.shape == (8, 8)
        assert img.sum() == 0.0

    def test_inf_death_filtered(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 1.0], [0.0, float("inf")]], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(8, 8), sigma=1.0)
        assert img.shape == (8, 8)
        assert img.sum() > 0  # at least the finite pair contributes

    def test_all_inf_deaths(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, float("inf")], [1.0, float("inf")]], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(8, 8), sigma=1.0)
        assert img.shape == (8, 8)
        assert img.sum() == 0.0

    def test_single_pair(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.5, 2.5]], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(10, 10), sigma=0.8)
        assert img.shape == (10, 10)
        assert img.sum() > 0


class TestPersistenceImageBatched:
    def test_batched_tensor(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([
            [[0.0, 1.0], [1.0, 2.0]],
            [[0.0, 3.0], [2.0, 5.0]],
        ], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(8, 8), sigma=1.0)
        assert img.shape == (2, 8, 8)
        assert img.sum() > 0

    def test_batched_single_diagram(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[[0.0, 2.0], [1.0, 3.0]]], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(6, 6), sigma=1.0)
        assert img.shape == (1, 6, 6)


class TestPersistenceImageValidation:
    def test_invalid_resolution_length(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="exactly two"):
            persistence_image(diagram, resolution=(10,), sigma=1.0)

    def test_negative_resolution(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="positive"):
            persistence_image(diagram, resolution=(-1, 10), sigma=1.0)

    def test_zero_sigma_raises(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite and positive"):
            persistence_image(diagram, sigma=0.0)

    def test_negative_sigma_raises(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite and positive"):
            persistence_image(diagram, sigma=-1.0)

    def test_nan_sigma_raises(self):
        from pynerve.torch._persistence_image import persistence_image

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite and positive"):
            persistence_image(diagram, sigma=float("nan"))


class TestPersistenceImageWithPersistenceDiagram:
    def test_persistence_diagram_2d(self):
        from pynerve.torch._diagram import PersistenceDiagram
        from pynerve.torch._persistence_image import persistence_image

        # PersistenceDiagram requires 3 columns (birth, death, dim) and adds batch dim
        tensor = torch.tensor([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=torch.float32)
        pd = PersistenceDiagram(tensor)
        # pd.diagrams is 3D (1, 2, 3) -> batched path returns (1, 8, 8)
        img = persistence_image(pd, resolution=(8, 8), sigma=1.0)
        assert img.shape[0] >= 1
        assert img.shape[-1] == 8 and img.shape[-2] == 8
        assert img.sum() > 0

    def test_persistence_diagram_3d_batched(self):
        from pynerve.torch._diagram import PersistenceDiagram
        from pynerve.torch._persistence_image import persistence_image

        tensor = torch.tensor([
            [[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]],
            [[0.0, 3.0, 0.0], [1.0, 4.0, 1.0]],
        ], dtype=torch.float32)
        pd = PersistenceDiagram(tensor)
        img = persistence_image(pd, resolution=(6, 6), sigma=1.0)
        assert img.shape == (2, 6, 6)
        assert img.sum() > 0


class TestComputeSinglePersistenceImage:
    def test_validates_2d_only(self):
        from pynerve.torch._persistence_image import _compute_single_persistence_image

        diagram = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        with pytest.raises(ValueError, match="2D"):
            _compute_single_persistence_image(diagram, (10, 10), 1.0, "persistence")

    def test_non_float_raises(self):
        from pynerve.torch._persistence_image import _compute_single_persistence_image

        diagram = torch.tensor([[0, 1]], dtype=torch.int32)
        with pytest.raises(TypeError, match="floating-point"):
            _compute_single_persistence_image(diagram, (10, 10), 1.0, "persistence")

    def test_birth_must_be_finite(self):
        from pynerve.torch._persistence_image import _compute_single_persistence_image

        diagram = torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite"):
            _compute_single_persistence_image(diagram, (10, 10), 1.0, "persistence")

    def test_death_nan_raises(self):
        from pynerve.torch._persistence_image import _compute_single_persistence_image

        diagram = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
        with pytest.raises(ValueError, match="NaN"):
            _compute_single_persistence_image(diagram, (10, 10), 1.0, "persistence")

    def test_death_before_birth_raises(self):
        from pynerve.torch._persistence_image import _compute_single_persistence_image

        diagram = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="greater than or equal"):
            _compute_single_persistence_image(diagram, (10, 10), 1.0, "persistence")

    def test_empty_returns_zeros(self):
        from pynerve.torch._persistence_image import _compute_single_persistence_image

        diagram = torch.empty((0, 2), dtype=torch.float32)
        img = _compute_single_persistence_image(diagram, (8, 8), 1.0, "persistence")
        assert img.shape == (8, 8)
        assert img.sum() == 0.0

    def test_constant_weight_assignment(self):
        from pynerve.torch._persistence_image import _compute_single_persistence_image

        diagram = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
        img_const = _compute_single_persistence_image(diagram, (10, 10), 1.0, "constant")
        img_pers = _compute_single_persistence_image(diagram, (10, 10), 1.0, "persistence")
        # Both pairs have same persistence so constant and persistence should match
        assert torch.allclose(img_const, img_pers, atol=1e-6)
