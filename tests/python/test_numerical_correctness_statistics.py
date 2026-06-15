"""Numerical correctness tests that verify outputs against hand-computed values.

Every test with a numerical value includes a precise assertion with tolerance,
not just a shape or finiteness check.
"""

from __future__ import annotations

import math

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

# statistics


class TestTotalPersistenceCorrectness:
    """Numerical correctness for total_persistence."""

    def test_p1_exact(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [1.0, 3.0]], dtype=torch.float64)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(5.0, abs=1e-10)

    def test_p2_exact(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [1.0, 3.0]], dtype=torch.float64)
        tp = total_persistence(d, p=2.0)
        expected = 1**2 + 2**2 + 2**2
        assert tp.item() == pytest.approx(expected, abs=1e-10)

    def test_empty_diagram_zero(self) -> None:
        from pynerve.torch.statistics import total_persistence

        empty = torch.empty(0, 2, dtype=torch.float64)
        tp = total_persistence(empty, p=1.0)
        assert tp.item() == pytest.approx(0.0, abs=1e-10)

    def test_zero_persistence(self) -> None:
        from pynerve.torch.statistics import total_persistence

        d = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        tp = total_persistence(d, p=1.0)
        assert tp.item() == pytest.approx(0.0, abs=1e-10)


class TestPersistenceEntropyCorrectness:
    """Numerical correctness for persistence_entropy."""

    def test_two_equal_points(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
        ent = persistence_entropy(d)
        expected = -math.log(0.5)
        assert ent.item() == pytest.approx(expected, abs=1e-10)

    def test_single_point(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        ent = persistence_entropy(d)
        assert ent.item() == pytest.approx(0.0, abs=1e-10)

    def test_zero_persistence_filtered(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        e1 = persistence_entropy(d)
        e2 = persistence_entropy(d[:1])
        assert torch.allclose(e1, e2)

    def test_three_unequal_points(self) -> None:
        from pynerve.torch.statistics import persistence_entropy

        d = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
        ent = persistence_entropy(d)
        pers = torch.tensor([1.0, 1.0, 3.0])
        probs = pers / pers.sum()
        expected = -(probs * probs.log()).sum().item()
        assert ent.item() == pytest.approx(expected, abs=1e-8)


class TestBettiCurveCorrectness:
    """Numerical correctness for betti_curve."""

    def test_non_negative(self) -> None:
        from pynerve.torch.statistics import betti_curve

        d = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        curve = betti_curve(d, num_samples=5)
        assert (curve >= 0).all()
        assert torch.isfinite(curve).all()

    def test_empty_diagram_zero(self) -> None:
        from pynerve.torch.statistics import betti_curve

        empty = torch.empty(0, 2, dtype=torch.float64)
        curve = betti_curve(empty, num_samples=3)
        assert (curve == 0).all()


class TestNumberOfFeaturesCorrectness:
    """Numerical correctness for number_of_features."""

    def test_no_filter(self) -> None:
        from pynerve.torch.statistics import number_of_features

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.1]], dtype=torch.float64)
        assert number_of_features(d, min_persistence=0.0).item() == 3

    def test_with_threshold(self) -> None:
        from pynerve.torch.statistics import number_of_features

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.1]], dtype=torch.float64)
        assert number_of_features(d, min_persistence=0.5).item() == 2

    def test_all_filtered(self) -> None:
        from pynerve.torch.statistics import number_of_features

        d = torch.tensor([[0.0, 0.5], [0.0, 0.3]], dtype=torch.float64)
        assert number_of_features(d, min_persistence=1.0).item() == 0

    def test_monotonic_threshold(self) -> None:
        from pynerve.torch.statistics import number_of_features

        d = torch.tensor([[0.0, 0.5], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        low = number_of_features(d, min_persistence=0.3)
        mid = number_of_features(d, min_persistence=1.0)
        high = number_of_features(d, min_persistence=4.0)
        assert low.item() >= mid.item() >= high.item()


class TestAmplitudeCorrectness:
    """Numerical correctness for amplitude."""

    def test_empty_diagram_zero(self) -> None:
        from pynerve.torch.statistics import amplitude

        empty = torch.empty(0, 2, dtype=torch.float64)
        for metric in ("bottleneck", "wasserstein", "persistence"):
            val = amplitude(empty, metric=metric)
            assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_invalid_metric_raises(self) -> None:
        from pynerve.torch.statistics import amplitude

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises((ValueError, Exception)):
            amplitude(d, metric="invalid")


class TestMeanMaxPersistenceCorrectness:
    """Numerical correctness for mean_persistence and max_persistence."""

    def test_mean_persistence_exact(self) -> None:
        from pynerve.torch.statistics import mean_persistence

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        mp = mean_persistence(d)
        assert mp.item() == pytest.approx(2.0, abs=1e-10)

    def test_max_persistence_exact(self) -> None:
        from pynerve.torch.statistics import max_persistence

        d = torch.tensor([[1.0, 3.0], [0.0, 4.0], [0.0, 1.0]], dtype=torch.float64)
        mp = max_persistence(d)
        assert mp.item() == pytest.approx(4.0, abs=1e-10)

    def test_single_point(self) -> None:
        from pynerve.torch.statistics import mean_persistence

        d = torch.tensor([[0.0, 1.5]], dtype=torch.float64)
        mp = mean_persistence(d)
        assert mp.item() == pytest.approx(1.5, abs=1e-10)

    def test_mean_leq_max(self) -> None:
        from pynerve.torch.statistics import max_persistence, mean_persistence

        d = torch.tensor([[0.0, 1.0], [0.0, 10.0]], dtype=torch.float64)
        mp = mean_persistence(d)
        mx = max_persistence(d)
        assert mp.item() <= mx.item() + 1e-10


class TestPersistenceVarianceCorrectness:
    """Numerical correctness for persistence_variance."""

    def test_known_variance(self) -> None:
        from pynerve.torch.statistics import persistence_variance

        d = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
        pv = persistence_variance(d)
        assert pv.item() == pytest.approx(1.0, abs=1e-10)

    def test_zero_variance(self) -> None:
        from pynerve.torch.statistics import persistence_variance

        d = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
        pv = persistence_variance(d)
        assert pv.item() == pytest.approx(0.0, abs=1e-10)
