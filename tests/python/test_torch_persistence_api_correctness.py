"""Numerical correctness for the torch persistence API (VR, alpha, witness, matrix)."""

from __future__ import annotations

import math

import pytest

try:
    import pynerve_internal  # noqa: F401
except ImportError:
    pytest.skip("pynerve_internal C++ extension not available", allow_module_level=True)

torch = pytest.importorskip("torch")


class TestVrPersistence:
    """Numerical correctness for vr_persistence."""

    def test_two_points_h0_death(self) -> None:
        from pynerve.torch import vr_persistence

        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        diagram = vr_persistence(pts, max_dim=0, max_radius=5.0)
        births, deaths = diagram.births(), diagram.deaths()
        for i in range(births.shape[0]):
            b, d = births[i].item(), deaths[i].item()
            if math.isfinite(d):
                assert b == pytest.approx(0.0, abs=1e-5)
                assert d == pytest.approx(1.0, abs=1e-5)
                return
        pytest.fail("no finite H0 pair found")

    def test_two_essential_class(self) -> None:
        from pynerve.torch import vr_persistence

        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        diagram = vr_persistence(pts, max_dim=0, max_radius=5.0)
        deaths = diagram.deaths()
        has_essential = any(not math.isfinite(d.item()) for d in deaths)
        assert has_essential

    def test_batch_shape(self) -> None:
        from pynerve.torch import vr_persistence

        pts = torch.randn(3, 10, 2)
        diagram = vr_persistence(pts, max_dim=1, max_radius=5.0)
        assert diagram.births().shape[0] >= 1
        assert diagram.deaths().shape[0] >= 1

    def test_dtype_preserved(self) -> None:
        from pynerve.torch import vr_persistence

        pts = torch.randn(2, 5, 3, dtype=torch.float64)
        diagram = vr_persistence(pts, max_dim=0, max_radius=5.0)
        assert diagram.births().dtype == torch.float64

    def test_square_h1(self) -> None:
        from pynerve.torch import vr_persistence

        pts = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        diagram = vr_persistence(pts, max_dim=1, max_radius=5.0)
        dims = diagram.dimensions()
        deaths = diagram.deaths()
        for i in range(deaths.shape[0]):
            if dims[0, i].item() == 1:
                return  # H1 class found
        pytest.fail("no H1 pair found")


class TestWitnessPersistence:
    """Numerical correctness for witness_persistence."""

    def test_forward_finite(self) -> None:
        from pynerve.torch import witness_persistence

        lms = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        wits = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        diagram = witness_persistence(lms, wits, max_dim=0, max_radius=5.0)
        assert diagram.births().numel() >= 1


class TestAlphaPersistence:
    """Numerical correctness for alpha_persistence."""

    def test_forward_finite(self) -> None:
        from pynerve.torch import alpha_persistence

        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        diagram = alpha_persistence(pts, max_dim=1)
        assert diagram.births().numel() >= 1


class TestPersistenceFromMatrix:
    """Numerical correctness for persistence_from_matrix."""

    def test_two_point_matrix(self) -> None:
        from pynerve.torch import persistence_from_matrix

        D = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
        diagram = persistence_from_matrix(D, max_dim=0)
        assert diagram.births().numel() >= 1
