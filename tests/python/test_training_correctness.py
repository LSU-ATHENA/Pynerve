"""Numerical correctness tests for training utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestTopologicalComplexityCalculator:
    """Numerical correctness for TopologicalComplexityCalculator."""

    def test_total_persistence(self) -> None:
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator(persistence_threshold=0.0)
        d = torch.tensor([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
        val = calc.compute_complexity(d, ComplexityMeasure.TOTAL_PERSISTENCE)
        assert val == pytest.approx(4.0, abs=1e-10)

    def test_num_features(self) -> None:
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator(persistence_threshold=0.5)
        d = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.3, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
        val = calc.compute_complexity(d, ComplexityMeasure.NUM_FEATURES)
        assert val == 2

    def test_max_persistence(self) -> None:
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator(persistence_threshold=0.0)
        d = torch.tensor([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
        val = calc.compute_complexity(d, ComplexityMeasure.MAX_PERSISTENCE)
        assert val == pytest.approx(3.0, abs=1e-10)

    def test_batch_complexity(self) -> None:
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator(persistence_threshold=0.0)
        diagrams = [
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float64),
        ]
        vals = calc.compute_batch_complexity(diagrams, ComplexityMeasure.TOTAL_PERSISTENCE)
        assert vals == [4.0, 2.0]

    def test_empty_diagram(self) -> None:
        from pynerve.training.curriculum import ComplexityMeasure, TopologicalComplexityCalculator

        calc = TopologicalComplexityCalculator(persistence_threshold=0.0)
        d = torch.empty(0, 3, dtype=torch.float64)
        val = calc.compute_complexity(d, ComplexityMeasure.TOTAL_PERSISTENCE)
        assert val == pytest.approx(0.0, abs=1e-10)


class TestBettiCurriculum:
    """Numerical correctness for BettiCurriculum."""

    def test_active_dimensions_initial(self) -> None:
        from pynerve.training.curriculum import BettiCurriculum

        bc = BettiCurriculum(max_dim=3, epochs_per_dim=10)
        dims = bc.get_active_dimensions(0)
        assert dims == [0]

    def test_active_dimensions_after_epochs(self) -> None:
        from pynerve.training.curriculum import BettiCurriculum

        bc = BettiCurriculum(max_dim=3, epochs_per_dim=10)
        dims = bc.get_active_dimensions(10)
        assert dims == [0, 1]

    def test_active_dimensions_all(self) -> None:
        from pynerve.training.curriculum import BettiCurriculum

        bc = BettiCurriculum(max_dim=3, epochs_per_dim=10)
        dims = bc.get_active_dimensions(30)
        assert dims == [0, 1, 2, 3]

    def test_filter_diagram(self) -> None:
        from pynerve.training.curriculum import BettiCurriculum

        bc = BettiCurriculum(max_dim=3, epochs_per_dim=10)
        d = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 3.0, 2.0]],
            dtype=torch.float64,
        )
        filtered = bc.filter_diagram_by_dim(d, 0)
        assert filtered.shape[0] >= 1
        assert filtered.shape[1] == 3


class TestFiltrationCurriculum:
    """Numerical correctness for FiltrationCurriculum."""

    def test_stage_radius_increasing(self) -> None:
        from pynerve.training.curriculum import FiltrationCurriculum

        fc = FiltrationCurriculum(max_radius=10.0, num_stages=5)
        radii = [fc.get_stage_radius(i) for i in range(5)]
        assert all(radii[i] < radii[i + 1] for i in range(4))

    def test_stage_radius_range(self) -> None:
        from pynerve.training.curriculum import FiltrationCurriculum

        fc = FiltrationCurriculum(max_radius=10.0, num_stages=5)
        r0 = fc.get_stage_radius(0)
        r4 = fc.get_stage_radius(4)
        assert r0 > 0
        assert r4 == pytest.approx(10.0, abs=1e-10)

    def test_stage_threshold(self) -> None:
        from pynerve.training.curriculum import FiltrationCurriculum

        fc = FiltrationCurriculum(max_radius=10.0, num_stages=5)
        for i in range(5):
            t = fc.get_stage_threshold(i)
            assert 0 <= t <= 1.0


class TestStabilityRegularizer:
    """Numerical correctness for StabilityRegularizer."""

    def test_identical_diagrams_zero(self) -> None:
        from pynerve.training.stability_reg import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.0, num_perturbations=1)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dist = reg.wasserstein_distance([d], [d])
        assert dist.item() == pytest.approx(0.0, abs=1e-10)

    def test_different_diagrams_positive(self) -> None:
        from pynerve.training.stability_reg import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.0, num_perturbations=1)
        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        dist = reg.wasserstein_distance([d1], [d2])
        assert dist.item() > 0

    def test_l2_distance(self) -> None:
        from pynerve.training.stability_reg import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.0, num_perturbations=1)
        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dist = reg.l2_diagram_distance([d1], [d2])
        assert dist.item() == pytest.approx(0.0, abs=1e-10)

    def test_forward_finite(self) -> None:
        from pynerve.training.stability_reg import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.0, num_perturbations=1)
        x = torch.randn(2, 3, dtype=torch.float64)
        loss = reg(x, lambda p: [torch.tensor([[0.0, 1.0]], dtype=torch.float64)])
        assert torch.isfinite(loss)

    def test_compute_theoretical_bound(self) -> None:
        from pynerve.training.stability_reg import StabilityRegularizer

        reg = StabilityRegularizer(epsilon=0.1, num_perturbations=1)
        bound = reg.compute_theoretical_bound(0.1, 100)
        assert bound > 0
