"""Tests for pynerve/torch/_persistence_vr.py -- VR persistence validation and autograd function."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor

from pynerve.torch._persistence_vr import vr_persistence
from pynerve.torch._diagram import PersistenceDiagram


class TestVRPersistenceValidation:
    def test_basic_cpu(self):
        points = torch.rand(10, 3, dtype=torch.float32)
        result = vr_persistence(points, max_dim=1)
        assert isinstance(result, PersistenceDiagram)

    def test_float64(self):
        points = torch.rand(5, 2, dtype=torch.float64)
        result = vr_persistence(points, max_dim=0)
        assert isinstance(result, PersistenceDiagram)

    def test_2d_input_unsqueezed(self):
        points = torch.rand(10, 3)
        result = vr_persistence(points, max_dim=0)
        assert isinstance(result, PersistenceDiagram)

    def test_return_simplices(self):
        points = torch.rand(10, 3)
        diagram, simplices = vr_persistence(points, max_dim=0, return_simplices=True)
        assert isinstance(diagram, PersistenceDiagram)
        assert isinstance(simplices, Tensor)

    def test_return_simplices_non_bool_raises(self):
        with pytest.raises(TypeError, match="boolean"):
            vr_persistence(torch.rand(5, 2), return_simplices=1)  # type: ignore[arg-type]

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            vr_persistence(torch.rand(10))

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError, match="at least one batch"):
            vr_persistence(torch.empty(0, 5, 3))

    def test_empty_points_raises(self):
        with pytest.raises(ValueError, match="at least one point"):
            vr_persistence(torch.empty(1, 0, 3))

    def test_empty_coords_raises(self):
        with pytest.raises(ValueError, match="coordinate dimension"):
            vr_persistence(torch.empty(1, 5, 0))

    def test_int_dtype_raises(self):
        points = torch.randint(0, 10, (5, 3))
        with pytest.raises(TypeError, match="dtype"):
            vr_persistence(points)

    def test_nan_coords_raises(self):
        points = torch.tensor([[0.0, 1.0], [float("nan"), 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="finite"):
            vr_persistence(points)

    def test_max_dim_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            vr_persistence(torch.rand(5, 2), max_dim=-1)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            vr_persistence(torch.rand(5, 2), metric="invalid")

    def test_negative_max_radius_raises(self):
        with pytest.raises(ValueError, match="radius"):
            vr_persistence(torch.rand(5, 2), max_radius=-1.0)
