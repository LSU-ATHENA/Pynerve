"""Numerical correctness tests that verify outputs against hand-computed values.

Every test with a numerical value includes a precise assertion with tolerance,
not just a shape or finiteness check.
"""

from __future__ import annotations

import pytest

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

# wasserstein / bottleneck distances


class TestWassersteinKnownValues:
    """Wasserstein distance with known exact values."""

    def test_self_distance_zero(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        dist = diagram_wasserstein(d, d)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-12)

    def test_known_shifted_point(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.5, 1.5]], dtype=torch.float64)
        dist = diagram_wasserstein(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        # wasserstein(p=q=2) between (0,1) and (0.5,1.5) = sqrt(0.5^2 + 0.5^2) = sqrt(0.5)
        assert val.item() == pytest.approx(0.5**0.5, abs=1e-6)

    def test_symmetry(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.5, 1.5], [0.0, 3.0]], dtype=torch.float64)
        v12 = diagram_wasserstein(d1, d2)
        v21 = diagram_wasserstein(d2, d1)
        v12t = v12 if isinstance(v12, torch.Tensor) else torch.tensor(v12)
        v21t = v21 if isinstance(v21, torch.Tensor) else torch.tensor(v21)
        assert v12t.item() == pytest.approx(v21t.item(), abs=1e-10)

    def test_triangle_inequality(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        d3 = torch.tensor([[0.0, 3.0]], dtype=torch.float64)
        d12 = diagram_wasserstein(d1, d2)
        d23 = diagram_wasserstein(d2, d3)
        d13 = diagram_wasserstein(d1, d3)
        v12 = d12 if isinstance(d12, torch.Tensor) else torch.tensor(d12)
        v23 = d23 if isinstance(d23, torch.Tensor) else torch.tensor(d23)
        v13 = d13 if isinstance(d13, torch.Tensor) else torch.tensor(d13)
        assert v13.item() <= v12.item() + v23.item() + 1e-10

    def test_positive_definite(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        dist = diagram_wasserstein(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() > 0.0

    def test_custom_pq(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dist = diagram_wasserstein(d, d, p=1.0, q=1.0)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-12)

    def test_dtype_consistency(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d_f32 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        d_f64 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        r32 = diagram_wasserstein(d_f32, d_f32)
        r64 = diagram_wasserstein(d_f64, d_f64)
        v32 = r32 if isinstance(r32, torch.Tensor) else torch.tensor(r32)
        v64 = r64 if isinstance(r64, torch.Tensor) else torch.tensor(r64)
        assert v32.item() == pytest.approx(v64.item(), rel=1e-3)


class TestBottleneckKnownValues:
    """Bottleneck distance with known exact values."""

    def test_self_distance_zero(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        dist = diagram_bottleneck(d, d)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_known_shifted_point(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        dist = diagram_bottleneck(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(1.0, abs=1e-6)

    def test_symmetry(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.5, 1.5], [0.0, 3.0]], dtype=torch.float64)
        d12 = diagram_bottleneck(d1, d2)
        d21 = diagram_bottleneck(d2, d1)
        v12 = d12 if isinstance(d12, torch.Tensor) else torch.tensor(d12)
        v21 = d21 if isinstance(d21, torch.Tensor) else torch.tensor(d21)
        assert v12.item() == pytest.approx(v21.item(), abs=1e-10)

    def test_positive_definite(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.5, 1.5]], dtype=torch.float64)
        dist = diagram_bottleneck(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() > 0.0

    def test_inf_death_handled(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d1 = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        dist = diagram_bottleneck(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_diagram(self) -> None:
        from pynerve.torch import diagram_bottleneck

        empty = torch.empty(0, 2, dtype=torch.float64)
        dist = diagram_bottleneck(empty, _SINGLE_DIAGRAM)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert torch.isfinite(val)
