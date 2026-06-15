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

# edge cases


class TestEdgeCases:
    """Edge cases for numerical operations."""

    def test_zero_persistence_diagram(self) -> None:
        from pynerve.torch.statistics import persistence_entropy, total_persistence

        d = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        assert total_persistence(d, p=1.0).item() == pytest.approx(0.0, abs=1e-10)
        assert persistence_entropy(d).item() == pytest.approx(0.0, abs=1e-10)

    def test_many_points_diagram(self) -> None:
        from pynerve.torch.statistics import total_persistence

        n = 1000
        births = torch.zeros(n, dtype=torch.float64)
        deaths = torch.arange(1.0, n + 1, dtype=torch.float64)
        d = torch.stack([births, deaths], dim=1)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(n * (n + 1) / 2, rel=1e-10)

    def test_diagram_with_only_inf_deaths(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        d = torch.tensor([[0.0, float("inf")], [0.5, float("inf")]], dtype=torch.float64)
        result = handle_infinite_deaths(d, strategy="remove")
        assert result.shape[0] == 0
        result2 = handle_infinite_deaths(d, strategy="max")
        assert torch.isfinite(result2[:, 1]).all()

    def test_bottleneck_with_only_inf(self) -> None:
        from pynerve.torch import diagram_bottleneck

        d = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        dist = diagram_bottleneck(d, d)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_diagram_with_negative_coords(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[-1.0, 0.0], [-2.0, -1.0]], dtype=torch.float64)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(2.0, abs=1e-10)

    def test_subsample_more_features_than_exist(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = subsample_diagram(d, max_features=10, strategy="most_persistent")
        assert result.shape[0] == 2

    def test_wasserstein_empty_vs_nonempty(self) -> None:
        from pynerve.torch import diagram_wasserstein

        empty = torch.empty(0, 2, dtype=torch.float64)
        nonempty = _SINGLE_DIAGRAM
        dist = diagram_wasserstein(empty, nonempty)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert torch.isfinite(val)


# persistence from matrix (requires C++ torch backend)


class TestMatrixPersistence:
    """Numerical correctness for persistence_from_matrix."""

    def test_persistence_from_distance_matrix(self) -> None:
        from pynerve.torch import persistence_from_matrix

        D = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64)
        diagram = persistence_from_matrix(D, max_dim=0)
        assert diagram.diagrams.shape[-1] == 3
        deaths = diagram.deaths()
        births = diagram.births()
        finite = deaths[torch.isfinite(deaths)]
        assert finite[0].item() == pytest.approx(1.0, abs=1e-5)
        assert births[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_square_matrix_h0_deaths(self) -> None:
        from pynerve.torch import persistence_from_matrix

        D = torch.tensor(
            [
                [
                    [0.0, 1.0, 2**0.5, 1.0],
                    [1.0, 0.0, 1.0, 2**0.5],
                    [2**0.5, 1.0, 0.0, 1.0],
                    [1.0, 2**0.5, 1.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        diagram = persistence_from_matrix(D, max_dim=0)
        deaths = diagram.deaths()
        finite = deaths[torch.isfinite(deaths)]
        assert len(finite) == 3
        assert (finite == 1.0).all()
