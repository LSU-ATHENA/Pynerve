"""Numerical correctness tests for regularization modules."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestMorseRegularizer:
    """Numerical correctness for MorseRegularizer."""

    def test_constant_function_zero(self) -> None:
        from pynerve.regularization.topology_constraints import MorseRegularizer

        reg = MorseRegularizer(lambda_critical=0.1, lambda_morse=0.0)
        f = torch.ones(10, dtype=torch.float64)
        loss = reg(f)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_monotonic_function_zero(self) -> None:
        from pynerve.regularization.topology_constraints import MorseRegularizer

        reg = MorseRegularizer(lambda_critical=0.0, lambda_morse=0.0)
        f = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
        loss = reg(f)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_forward_finite(self) -> None:
        from pynerve.regularization.topology_constraints import MorseRegularizer

        reg = MorseRegularizer(lambda_critical=0.1, lambda_morse=0.05)
        f = torch.sin(torch.linspace(0, 3, 20, dtype=torch.float64))
        loss = reg(f)
        assert torch.isfinite(loss)


class TestBettiConstraintLayer:
    """Numerical correctness for BettiConstraintLayer."""

    def test_matching_betti_zero_loss(self) -> None:
        from pynerve.regularization.topology_constraints import BettiConstraintLayer

        def persistence_fn(x):
            return torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)

        layer = BettiConstraintLayer([1, 0], persistence_fn, lambda_constraint=1.0)
        x = torch.randn(1, 3, dtype=torch.float64)
        _, loss = layer(x)
        assert loss.item() >= 0

    def test_non_matching_betti_positive(self) -> None:
        from pynerve.regularization.topology_constraints import BettiConstraintLayer

        def persistence_fn(x):
            return torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)

        layer = BettiConstraintLayer([0, 0], persistence_fn, lambda_constraint=1.0)
        x = torch.randn(1, 3, dtype=torch.float64)
        _, loss = layer(x)
        assert loss.item() >= 0


class TestTopologicalSmoothness:
    """Numerical correctness for TopologicalSmoothness."""

    def test_identical_features_zero(self) -> None:
        from pynerve.regularization.topology_constraints import TopologicalSmoothness

        smooth = TopologicalSmoothness(lambda_smooth=1.0, neighborhood_size=3)
        features = torch.ones(5, 3, dtype=torch.float64)
        diagrams = [torch.tensor([[0.0, 1.0]], dtype=torch.float64) for _ in range(5)]
        loss = smooth(features, diagrams)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_gradient_flow(self) -> None:
        from pynerve.regularization.topology_constraints import TopologicalSmoothness

        smooth = TopologicalSmoothness(lambda_smooth=0.1, neighborhood_size=3)
        features = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        diagrams = [torch.tensor([[0.0, 1.0]], dtype=torch.float64) for _ in range(5)]
        loss = smooth(features, diagrams)
        loss.backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()


class TestHomotopyRegularizer:
    """Numerical correctness for HomotopyRegularizer."""

    def test_identical_outputs_zero(self) -> None:
        from pynerve.regularization.topology_constraints import HomotopyRegularizer

        reg = HomotopyRegularizer(lambda_homotopy=1.0)
        x = torch.randn(5, 3, dtype=torch.float64)
        loss = reg(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_different_outputs_positive(self) -> None:
        from pynerve.regularization.topology_constraints import HomotopyRegularizer

        reg = HomotopyRegularizer(lambda_homotopy=1.0)
        x = torch.randn(5, 3, dtype=torch.float64)
        y = torch.randn(5, 3, dtype=torch.float64)
        loss = reg(x, y)
        assert loss.item() > 0

    def test_gradient_flow(self) -> None:
        from pynerve.regularization.topology_constraints import HomotopyRegularizer

        reg = HomotopyRegularizer(lambda_homotopy=1.0)
        x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        y = torch.randn(5, 3, dtype=torch.float64)
        loss = reg(x, y)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestAdaptivePersistentDropout:
    """Numerical correctness for AdaptivePersistentDropout."""

    def test_forward_no_dropout(self) -> None:
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout

        dropout = AdaptivePersistentDropout(4, adaptation_epochs=1, min_persistence_to_keep=0.0)
        x = torch.ones(2, 4, dtype=torch.float64)
        result = dropout(x, training=False)
        assert torch.allclose(result, x)

    def test_gradient_flow(self) -> None:
        from pynerve.regularization.persistent_dropout import AdaptivePersistentDropout

        dropout = AdaptivePersistentDropout(4, adaptation_epochs=1, min_persistence_to_keep=0.0)
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        result = dropout(x, training=True)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestFeaturePersistenceTracker:
    """Numerical correctness for FeaturePersistenceTracker."""

    def test_update_and_ranking(self) -> None:
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker

        tracker = FeaturePersistenceTracker(4, momentum=0.0)
        importance = torch.tensor([0.1, 0.5, 0.3, 0.9], dtype=torch.float64)
        tracker.update(importance)
        ranking = tracker.get_persistence_ranking()
        # Most persistent feature should be index 3 (value 0.9)
        assert ranking[0].item() == 3

    def test_momentum_update(self) -> None:
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker

        tracker = FeaturePersistenceTracker(2, momentum=0.5)
        tracker.update(torch.tensor([1.0, 0.0], dtype=torch.float64))
        tracker.update(torch.tensor([0.0, 1.0], dtype=torch.float64))
        ranking = tracker.get_persistence_ranking()
        assert ranking[0].item() in (0, 1)

    def test_top_k(self) -> None:
        from pynerve.regularization.persistent_dropout import FeaturePersistenceTracker

        tracker = FeaturePersistenceTracker(5, momentum=0.0)
        tracker.update(torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7], dtype=torch.float64))
        top2 = tracker.get_top_k_persistent(2)
        assert top2.shape == (2,)
        assert top2[0].item() == 2
        assert top2[1].item() == 4


class TestCurricularPersistentDropout:
    """Numerical correctness for CurricularPersistentDropout."""

    def test_forward_eval_no_dropout(self) -> None:
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout

        dropout = CurricularPersistentDropout(4, warmup_epochs=1, full_epochs=2)
        x = torch.ones(2, 4, dtype=torch.float64)
        result = dropout(x, training=False)
        assert torch.allclose(result, x)

    def test_gradient_flow(self) -> None:
        from pynerve.regularization.persistent_dropout import CurricularPersistentDropout

        dropout = CurricularPersistentDropout(4, warmup_epochs=1, full_epochs=2)
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        result = dropout(x, training=True)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
