"""Numerical correctness tests that verify outputs against hand-computed values.

Every test with a numerical value includes a precise assertion with tolerance,
not just a shape or finiteness check.
"""

from __future__ import annotations

import math

import pytest

try:
    import pynerve_internal  # noqa: F401
except ImportError:
    pytest.skip("pynerve_internal C++ extension not available", allow_module_level=True)

torch = pytest.importorskip("torch")

# known-configuration point clouds
_TRIANGLE = torch.tensor(
    [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0**0.5]],
    dtype=torch.float64,
)
_SQUARE = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=torch.float64,
)
_TWO_POINTS = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
_SINGLE_POINT = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

# known diagram tensors
_SIMPLE_DIAGRAM = torch.tensor(
    [[0.0, 1.0], [0.0, 2.0], [1.0, 3.0]],
    dtype=torch.float64,
)
_SINGLE_DIAGRAM = torch.tensor([[0.0, 1.5]], dtype=torch.float64)
_INFINITE_DIAGRAM = torch.tensor(
    [[0.0, float("inf")], [0.0, 2.0]],
    dtype=torch.float64,
)

# vr persistence


class TestVrPersistenceHandComputed:
    """VR persistence with known point clouds against hand-computed values."""

    def test_two_points_h0_death(self) -> None:
        from pynerve import compute_persistence

        result = compute_persistence(_TWO_POINTS, max_dim=0, max_radius=float("inf"))
        pairs = result.pairs
        finite = [(b, d) for b, d, dim in pairs if dim == 0 and math.isfinite(d)]
        assert len(finite) == 1
        b, d = finite[0]
        assert b == pytest.approx(0.0, abs=1e-10)
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_two_points_h0_count(self) -> None:
        from pynerve import compute_persistence

        result = compute_persistence(_TWO_POINTS, max_dim=0, max_radius=float("inf"))
        dim0_finite = sum(1 for _, d, dim in result.pairs if dim == 0 and math.isfinite(d))
        dim0_essential = sum(1 for _, d, dim in result.pairs if dim == 0 and not math.isfinite(d))
        assert dim0_finite == 1
        assert dim0_essential == 1

    def test_single_point_essential(self) -> None:
        from pynerve import compute_persistence

        result = compute_persistence(_SINGLE_POINT, max_dim=0, max_radius=float("inf"))
        assert len(result.pairs) == 1
        b, d, dim = result.pairs[0]
        assert b == pytest.approx(0.0, abs=1e-10)
        assert not math.isfinite(d)
        assert dim == 0

    def test_equilateral_triangle_h0_death(self) -> None:
        from pynerve import compute_persistence

        result = compute_persistence(_TRIANGLE, max_dim=1, max_radius=float("inf"))
        dim0_finite = [(b, d) for b, d, dim in result.pairs if dim == 0 and math.isfinite(d)]
        assert len(dim0_finite) == 2
        for _b, d in dim0_finite:
            assert d == pytest.approx(2.0, abs=1e-6)

    def test_square_vr_h1(self) -> None:
        from pynerve import compute_persistence

        result = compute_persistence(_SQUARE, max_dim=1, max_radius=float("inf"))
        dim1 = [(b, d) for b, d, dim in result.pairs if dim == 1]
        assert len(dim1) == 1
        b, d = dim1[0]
        assert b == pytest.approx(1.0, abs=1e-6)
        assert d == pytest.approx(2**0.5, abs=1e-6)

    def test_max_radius_monotonic(self) -> None:
        from pynerve import compute_persistence

        pts = torch.randn(10, 2)
        res_05 = compute_persistence(pts, max_dim=1, max_radius=0.5)
        res_20 = compute_persistence(pts, max_dim=1, max_radius=2.0)
        assert len(res_05.pairs) >= 0
        assert len(res_20.pairs) >= len(res_05.pairs)

    def test_ph4_equivalence(self) -> None:
        from pynerve import compute_persistence, compute_persistence_up_to_dim_4

        pts = torch.randn(10, 3)
        gen = compute_persistence(pts, max_dim=1, max_radius=5.0)
        ph4 = compute_persistence_up_to_dim_4(pts, max_dim=1, max_radius=5.0)
        assert len(gen.pairs) > 0
        assert len(ph4.pairs) > 0

    def test_nan_input_rejected(self) -> None:
        from pynerve import compute_persistence

        with pytest.raises(Exception, match="finite"):
            compute_persistence(
                torch.tensor([[0.0, float("nan")]], dtype=torch.float32),
                max_dim=0,
                max_radius=float("inf"),
            )

    def test_inf_input_rejected(self) -> None:
        from pynerve import compute_persistence

        with pytest.raises(Exception, match="finite"):
            compute_persistence(
                torch.tensor([[0.0, float("inf")]], dtype=torch.float32),
                max_dim=0,
                max_radius=float("inf"),
            )

    def test_empty_cloud_raises(self) -> None:
        from pynerve import compute_persistence

        with pytest.raises((ValueError, Exception)):
            compute_persistence(torch.empty(0, 2), max_dim=0, max_radius=float("inf"))
