"""Tests for helper functions in torch/_distance_core_impl.py.

Covers _point_distance, _diagonal_distance, _assignment_cost_matrix,
_greedy_wasserstein, _greedy_bottleneck, _finite_points,
_fallback_compute warning paths, and DistanceMetric.__call__ validation.
"""

from __future__ import annotations

import math
import warnings
from unittest.mock import patch

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _force_python_backend():
    """Force Python backend by setting _torch_c and _core_c to None and preventing reload."""
    from pynerve.torch._backend import backend

    with patch.object(backend, '_torch_c', None), \
         patch.object(backend, '_core_c', None), \
         patch.object(backend, '_ensure_backends', lambda: None):
        yield


class TestPointDistance:
    def test_l2_norm(self):
        from pynerve.torch._distance_core_impl import _point_distance

        d1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        d2 = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        result = _point_distance(d1, d2, 2.0)
        assert result.shape == (2, 2)
        # dist([0,0],[0,1]) = 1.0
        assert torch.isclose(result[0, 0], torch.tensor(1.0), rtol=1e-5)

    def test_l1_norm(self):
        from pynerve.torch._distance_core_impl import _point_distance

        d1 = torch.tensor([[0.0, 0.0]])
        d2 = torch.tensor([[3.0, 4.0]])
        result = _point_distance(d1, d2, 1.0)
        assert torch.isclose(result[0, 0], torch.tensor(7.0), rtol=1e-5)

    def test_inf_norm(self):
        from pynerve.torch._distance_core_impl import _point_distance

        d1 = torch.tensor([[0.0, 0.0]])
        d2 = torch.tensor([[3.0, 4.0]])
        result = _point_distance(d1, d2, math.inf)
        assert torch.isclose(result[0, 0], torch.tensor(4.0), rtol=1e-5)

    def test_general_norm(self):
        from pynerve.torch._distance_core_impl import _point_distance

        d1 = torch.tensor([[0.0, 0.0]])
        d2 = torch.tensor([[3.0, 4.0]])
        result = _point_distance(d1, d2, 3.0)
        expected = (27.0 + 64.0) ** (1.0 / 3.0)
        assert torch.isclose(result[0, 0], torch.tensor(expected), rtol=1e-4)


class TestDiagonalDistance:
    def test_l2_norm(self):
        from pynerve.torch._distance_core_impl import _diagonal_distance

        points = torch.tensor([[0.0, 4.0]])
        result = _diagonal_distance(points, 2.0)
        assert torch.isclose(result, torch.tensor(4.0 * math.pow(2.0, -0.5)), rtol=1e-5)

    def test_inf_norm(self):
        from pynerve.torch._distance_core_impl import _diagonal_distance

        points = torch.tensor([[0.0, 6.0]])
        result = _diagonal_distance(points, math.inf)
        assert torch.isclose(result, torch.tensor(3.0), rtol=1e-5)

    def test_l1_norm(self):
        from pynerve.torch._distance_core_impl import _diagonal_distance

        points = torch.tensor([[0.0, 4.0]])
        result = _diagonal_distance(points, 1.0)
        assert torch.isclose(result, torch.tensor(4.0), rtol=1e-5)


class TestAssignmentCostMatrix:
    def test_square_cost(self):
        from pynerve.torch._distance_core_impl import _assignment_cost_matrix

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        d2 = torch.tensor([[0.0, 1.5], [0.0, 3.0]])
        cost = _assignment_cost_matrix(d1, d2, 2.0)
        assert cost.shape == (4, 4)
        assert cost[0, 0] >= 0.0
        assert cost[1, 1] >= 0.0

    def test_one_empty(self):
        from pynerve.torch._distance_core_impl import _assignment_cost_matrix

        d1 = torch.empty((0, 2))
        d2 = torch.tensor([[0.0, 1.0]])
        cost = _assignment_cost_matrix(d1, d2, 2.0)
        assert cost.shape == (1, 1)

    def test_both_empty(self):
        from pynerve.torch._distance_core_impl import _assignment_cost_matrix

        d1 = torch.empty((0, 2))
        d2 = torch.empty((0, 2))
        cost = _assignment_cost_matrix(d1, d2, 2.0)
        assert cost.shape == (0, 0)


class TestGreedyWasserstein:
    def test_basic_greedy(self):
        from pynerve.torch._distance_core_impl import _greedy_wasserstein

        n, m = 2, 2
        cost = np.array([
            [0.5, 1.0, 0.5, 0.0],
            [1.0, 0.3, 0.0, 0.3],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0, 0.0],
        ], dtype=np.float64)
        result = _greedy_wasserstein(cost, p=2.0, n=n, m=m)
        assert result > 0.0

    def test_all_matched(self):
        from pynerve.torch._distance_core_impl import _greedy_wasserstein

        n, m = 1, 1
        cost = np.array([
            [1.0, 0.5],
            [0.5, 0.0],
        ], dtype=np.float64)
        result = _greedy_wasserstein(cost, p=1.0, n=n, m=m)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_unmatched_rows(self):
        from pynerve.torch._distance_core_impl import _greedy_wasserstein

        # n=2, m=0: cost matrix is (2,2). Diagonal distances at cost[row, m=0].
        n, m = 2, 0
        cost = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
        ], dtype=np.float64)
        result = _greedy_wasserstein(cost, p=2.0, n=n, m=m)
        # All unmatched, total = 1^2 + 2^2 = 5, then 5^(1/2)
        assert result == pytest.approx(math.sqrt(5.0), abs=1e-5)

    def test_unmatched_cols(self):
        from pynerve.torch._distance_core_impl import _greedy_wasserstein

        # n=0, m=2: cost matrix is (2,2). Diagonal distances at cost[n=0, col].
        n, m = 0, 2
        cost = np.array([
            [0.5, 2.0],
            [1.0, 0.5],
        ], dtype=np.float64)
        result = _greedy_wasserstein(cost, p=2.0, n=n, m=m)
        # total = 0.5^2 + 2.0^2 = 4.25, then sqrt(4.25)
        assert result == pytest.approx(math.sqrt(4.25), abs=1e-5)


class TestGreedyBottleneck:
    def test_basic_greedy(self):
        from pynerve.torch._distance_core_impl import _greedy_bottleneck

        n, m = 2, 2
        cost = np.array([
            [0.5, 1.0, 0.5, 0.0],
            [1.0, 0.3, 0.0, 0.3],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0, 0.0],
        ], dtype=np.float64)
        result = _greedy_bottleneck(cost, n, m)
        assert result >= 0.0

    def test_single_match(self):
        from pynerve.torch._distance_core_impl import _greedy_bottleneck

        n, m = 1, 1
        cost = np.array([
            [2.0, 1.0],
            [1.0, 0.0],
        ], dtype=np.float64)
        result = _greedy_bottleneck(cost, n, m)
        assert result == pytest.approx(2.0, abs=1e-6)

    def test_all_unmatched_rows(self):
        from pynerve.torch._distance_core_impl import _greedy_bottleneck

        # n=2, m=0: cost matrix is (2,2). Diagonal distances at cost[row, m=0].
        n, m = 2, 0
        cost = np.array([
            [3.0, 0.0],
            [5.0, 0.0],
        ], dtype=np.float64)
        result = _greedy_bottleneck(cost, n, m)
        assert result == pytest.approx(5.0, abs=1e-6)


class TestFinitePoints:
    def test_all_finite(self):
        from pynerve.torch._distance_core_impl import _finite_points

        d = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        result = _finite_points(d)
        assert result.shape == (2, 2)

    def test_filters_inf(self):
        from pynerve.torch._distance_core_impl import _finite_points

        d = torch.tensor([[0.0, 1.0], [0.0, float("inf")]])
        result = _finite_points(d)
        assert result.shape == (1, 2)

    def test_all_inf(self):
        from pynerve.torch._distance_core_impl import _finite_points

        d = torch.tensor([[0.0, float("inf")], [1.0, float("inf")]])
        result = _finite_points(d)
        assert result.shape == (0, 2)


class TestFallbackComputeWarnings:
    def test_warns_on_python_fallback(self):
        from pynerve.torch._distance_core_impl import BottleneckDistance, _bottleneck_python

        metric = BottleneckDistance()
        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.tensor([[0.0, 2.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metric._fallback_compute(d1, d2, "test_bottleneck", try_torch_c=False)
            assert any("Python implementation" in str(warning.message) for warning in w)

    def test_warned_once(self):
        from pynerve.torch._distance_core_impl import BottleneckDistance

        metric = BottleneckDistance()
        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.tensor([[0.0, 2.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metric._fallback_compute(d1, d2, "test_once", try_torch_c=False)
            metric._fallback_compute(d1, d2, "test_once", try_torch_c=False)
            python_warnings = [x for x in w if "Python implementation" in str(x.message)]
            assert len(python_warnings) == 1


class TestWassersteinPythonDirect:
    def test_empty_cost_returns_zero(self):
        from pynerve.torch._distance_core_impl import _wasserstein_python

        d1 = torch.empty((0, 2))
        d2 = torch.empty((0, 2))
        result = _wasserstein_python(d1, d2, p=2.0, q=2.0)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_nonempty_result(self):
        from pynerve.torch._distance_core_impl import _wasserstein_python

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        d2 = torch.tensor([[0.0, 3.0], [0.0, 4.0]])
        result = _wasserstein_python(d1, d2, p=2.0, q=2.0)
        assert result.item() > 0.0

    def test_inf_death_filtered(self):
        from pynerve.torch._distance_core_impl import _wasserstein_python

        d1 = torch.tensor([[0.0, 1.0], [0.0, float("inf")]])
        d2 = torch.tensor([[0.0, 1.0]])
        result = _wasserstein_python(d1, d2, p=2.0, q=2.0)
        assert torch.isfinite(result).item()


class TestBottleneckPythonDirect:
    def test_empty_cost_returns_zero(self):
        from pynerve.torch._distance_core_impl import _bottleneck_python

        d1 = torch.empty((0, 2))
        d2 = torch.empty((0, 2))
        result = _bottleneck_python(d1, d2)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_inf_death_filtered(self):
        from pynerve.torch._distance_core_impl import _bottleneck_python

        d1 = torch.tensor([[0.0, 1.0], [0.0, float("inf")]])
        d2 = torch.tensor([[0.0, 1.0]])
        result = _bottleneck_python(d1, d2)
        assert torch.isfinite(result).item()

    def test_identical_diagrams(self):
        from pynerve.torch._distance_core_impl import _bottleneck_python

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        result = _bottleneck_python(d, d)
        assert result.item() == pytest.approx(0.0, abs=1e-6)


class TestValidateFiniteDeaths:
    def test_filters_inf_deaths(self):
        from pynerve.torch._distance_core_impl import _validate_finite_deaths

        diagram = torch.tensor([[0.0, 1.0], [0.0, float("inf")]])
        result = _validate_finite_deaths(diagram, "test")
        assert result.shape == (1, 2)

    def test_all_finite_unchanged(self):
        from pynerve.torch._distance_core_impl import _validate_finite_deaths

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        result = _validate_finite_deaths(diagram, "test")
        assert result.shape == (2, 2)

    def test_empty_diagram(self):
        from pynerve.torch._distance_core_impl import _validate_finite_deaths

        diagram = torch.empty((0, 2))
        result = _validate_finite_deaths(diagram, "test")
        assert result.shape == (0, 2)


class TestDiagramWassersteinWrapper:
    def test_default_p_q(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.tensor([[0.0, 1.0]])
        result = diagram_wasserstein(d1, d2)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_custom_p_q_creates_new_metric(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.tensor([[0.0, 2.0]])
        result = diagram_wasserstein(d1, d2, p=1.0, q=1.0)
        assert result.item() > 0.0


class TestDiagramBottleneckWrapper:
    def test_identical(self):
        from pynerve.torch._distance_core_impl import _bottleneck_python, _validate_distance_diagram

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        t1 = _validate_distance_diagram(d1)
        result = _bottleneck_python(t1, t1)
        assert result.item() == pytest.approx(0.0, abs=1e-6)


class TestDistanceMetricCallValidation:
    def test_4d_diagram_rejected(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance

        metric = WassersteinDistance()
        d = torch.zeros((1, 1, 1, 2))
        with pytest.raises(Exception, match="1D or 2D"):
            metric(d, d)

    def test_0d_diagram_rejected(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance

        metric = WassersteinDistance()
        d = torch.tensor(0.0)
        with pytest.raises(Exception, match="1D or 2D"):
            metric(d, d)
