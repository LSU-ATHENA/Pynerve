"""Numerical correctness tests for the differentiable topology module."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


class TestPersistenceLoss:
    """Numerical correctness for PersistenceLoss static methods."""

    def test_softmin_two_values(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = PersistenceLoss.softmin(x, temperature=1.0)
        expected = -torch.logsumexp(-x / 1.0, dim=0) * 1.0
        assert result.item() == pytest.approx(expected.item(), abs=1e-10)

    def test_softmax_sum_one(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = PersistenceLoss.softmax(x, temperature=1.0)
        assert result.sum().item() == pytest.approx(1.0, abs=1e-10)

    def test_softmax_identity_at_zero_temp(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        x = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = PersistenceLoss.softmax(x, temperature=0.5)
        assert result[0].item() > 0.5

    def test_diagram_wasserstein_identical(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        dist = PersistenceLoss.diagram_wasserstein(d, d, temperature=0.01)
        assert dist.item() >= 0
        assert torch.isfinite(dist)

    def test_diagram_wasserstein_positive(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        dist = PersistenceLoss.diagram_wasserstein(d1, d2, temperature=0.01)
        assert dist.item() > 0

    def test_diagram_wasserstein_empty_both(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        empty = torch.empty(0, 2, dtype=torch.float64)
        dist = PersistenceLoss.diagram_wasserstein(empty, empty, temperature=0.01)
        assert dist.item() == pytest.approx(0.0, abs=1e-10)

    def test_diagram_bottleneck_identical(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        dist = PersistenceLoss.diagram_bottleneck(d, d, temperature=0.001)
        assert dist.item() >= 0

    def test_diagram_bottleneck_positive(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        dist = PersistenceLoss.diagram_bottleneck(d1, d2, temperature=0.001)
        assert dist.item() > 0

    def test_diagram_bottleneck_empty_both(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        empty = torch.empty(0, 2, dtype=torch.float64)
        dist = PersistenceLoss.diagram_bottleneck(empty, empty, temperature=0.001)
        assert dist.item() == pytest.approx(0.0, abs=1e-10)

    def test_persistence_kernel_self_similarity(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        k = PersistenceLoss.persistence_kernel(d, d, sigma=1.0)
        assert k.item() > 0
        assert torch.isfinite(k)

    def test_persistence_kernel_empty(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        empty = torch.empty(0, 2, dtype=torch.float64)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        k = PersistenceLoss.persistence_kernel(empty, d, sigma=1.0)
        assert k.item() == pytest.approx(0.0, abs=1e-10)

    def test_persistence_kernel_symmetric(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        k12 = PersistenceLoss.persistence_kernel(d1, d2, sigma=1.0)
        k21 = PersistenceLoss.persistence_kernel(d2, d1, sigma=1.0)
        assert k12.item() == pytest.approx(k21.item(), abs=1e-10)

    def test_persistence_kernel_sigma_effect(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[5.0, 6.0]], dtype=torch.float64)
        k_small = PersistenceLoss.persistence_kernel(d1, d2, sigma=0.1)
        k_large = PersistenceLoss.persistence_kernel(d1, d2, sigma=10.0)
        assert k_small.item() < k_large.item()

    def test_gradient_through_wasserstein(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64, requires_grad=True)
        dist = PersistenceLoss.diagram_wasserstein(d, d.detach(), temperature=0.1)
        dist.backward()
        assert d.grad is not None

    def test_diagram_wasserstein_empty_first(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        empty = torch.empty(0, 2, dtype=torch.float64)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dist = PersistenceLoss.diagram_wasserstein(empty, d, temperature=0.01)
        assert torch.isfinite(dist)

    def test_diagram_bottleneck_empty_first(self) -> None:
        from pynerve.diff.topology_loss import PersistenceLoss

        empty = torch.empty(0, 2, dtype=torch.float64)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dist = PersistenceLoss.diagram_bottleneck(empty, d, temperature=0.001)
        assert torch.isfinite(dist)


class TestBettiNumberLoss:
    """Numerical correctness for BettiNumberLoss."""

    def test_soft_step_at_threshold(self) -> None:
        from pynerve.diff.topology_loss import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.5, temperature=0.01)
        x = torch.tensor([0.5], dtype=torch.float64)
        step = loss.soft_step(x)
        assert step.item() == pytest.approx(0.5, abs=1e-2)

    def test_soft_step_above_threshold(self) -> None:
        from pynerve.diff.topology_loss import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.5, temperature=0.01)
        x = torch.tensor([1.0], dtype=torch.float64)
        step = loss.soft_step(x)
        assert step.item() > 0.5

    def test_soft_step_below_threshold(self) -> None:
        from pynerve.diff.topology_loss import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.5, temperature=0.01)
        x = torch.tensor([0.0], dtype=torch.float64)
        step = loss.soft_step(x)
        assert step.item() < 0.5

    def test_forward_identical_betti(self) -> None:
        from pynerve.diff.topology_loss import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.1, temperature=0.01)
        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
        target = torch.tensor([2.0, 0.0], dtype=torch.float64)
        l = loss(diagram, target)
        assert l.item() == pytest.approx(0.0, abs=1e-1)

    def test_forward_different_betti(self) -> None:
        from pynerve.diff.topology_loss import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.1, temperature=0.01)
        diagram = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        target = torch.tensor([2.0, 0.0], dtype=torch.float64)
        l = loss(diagram, target)
        assert l.item() > 0

    def test_betti_gradient_flow(self) -> None:
        from pynerve.diff.topology_loss import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.1, temperature=0.1)
        diagram = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor([2.0, 0.0], dtype=torch.float64)
        l = loss(diagram, target)
        l.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()


class TestDiagramComplexityLoss:
    """Numerical correctness for DiagramComplexityLoss."""

    def test_total_persistence(self) -> None:
        from pynerve.diff.topology_loss import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="total_persistence")
        d = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
        val = loss(d)
        assert val.item() == pytest.approx(4.0, abs=1e-10)

    def test_max_persistence(self) -> None:
        from pynerve.diff.topology_loss import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="max_persistence")
        d = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
        val = loss(d)
        assert val.item() == pytest.approx(3.0, abs=1e-10)

    def test_num_features(self) -> None:
        from pynerve.diff.topology_loss import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="num_features")
        d = torch.tensor([[0.0, 1.0], [0.0, 0.05], [0.0, 3.0]], dtype=torch.float64)
        val = loss(d)
        assert val.item() == 2

    def test_persistence_entropy(self) -> None:
        from pynerve.diff.topology_loss import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="persistence_entropy")
        d = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
        val = loss(d)
        expected = -math.log(0.5)
        assert val.item() == pytest.approx(expected, abs=1e-6)

    def test_empty_diagram_zero(self) -> None:
        from pynerve.diff.topology_loss import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="total_persistence")
        d = torch.empty(0, 2, dtype=torch.float64)
        val = loss(d)
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_gradient_flow(self) -> None:
        from pynerve.diff.topology_loss import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="total_persistence")
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64, requires_grad=True)
        val = loss(d)
        val.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()


class TestLandscapeLoss:
    """Numerical correctness for LandscapeLoss."""

    def test_identical_diagrams_zero(self) -> None:
        from pynerve.diff.topology_loss import LandscapeLoss

        loss = LandscapeLoss(n_layers=2, resolution=10)
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        val = loss(d, d)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_diagrams_positive(self) -> None:
        from pynerve.diff.topology_loss import LandscapeLoss

        loss = LandscapeLoss(n_layers=2, resolution=10)
        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        val = loss(d1, d2)
        assert val.item() > 0

    def test_gradient_flow(self) -> None:
        from pynerve.diff.topology_loss import LandscapeLoss

        loss = LandscapeLoss(n_layers=2, resolution=10)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64, requires_grad=True)
        val = loss(d, d)
        val.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()


class TestTopologyLoss:
    """Numerical correctness for combined TopologyLoss."""

    def test_wasserstein_only(self) -> None:
        from pynerve.diff.topology_loss import TopologyLoss

        loss = TopologyLoss(wasserstein_weight=1.0, betti_weight=0.0, complexity_weight=0.0)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        result = loss(d, d)
        assert "wasserstein" in result
        assert result["wasserstein"].item() >= 0
        assert result["total"].item() >= 0

    def test_betti_only(self) -> None:
        from pynerve.diff.topology_loss import TopologyLoss

        loss = TopologyLoss(wasserstein_weight=0.0, betti_weight=1.0, complexity_weight=0.0)
        d = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        target_betti = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = loss(d, d, target_betti=target_betti)
        assert "betti" in result
        assert result["total"].item() >= 0

    def test_complexity_only(self) -> None:
        from pynerve.diff.topology_loss import TopologyLoss

        loss = TopologyLoss(wasserstein_weight=0.0, betti_weight=0.0, complexity_weight=1.0)
        d = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
        result = loss(d, d)
        assert "complexity" in result
        assert result["complexity"].item() == pytest.approx(4.0, abs=1e-10)

    def test_combined_output_keys(self) -> None:
        from pynerve.diff.topology_loss import TopologyLoss

        loss = TopologyLoss()
        d = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        target_betti = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = loss(d, d, target_betti=target_betti)
        assert "total" in result
        assert "wasserstein" in result
        assert "betti" in result
        assert "complexity" in result


class TestTopologyRegularizer:
    """Numerical correctness for topology_regularizer in ph_layer_module."""

    def test_matching_betti_zero(self) -> None:
        from pynerve.diff.ph_layer_module import topology_regularizer

        diagrams = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        loss = topology_regularizer(diagrams, target_betti=[1], weight=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_mismatched_betti(self) -> None:
        from pynerve.diff.ph_layer_module import topology_regularizer

        diagrams = [torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)]
        loss = topology_regularizer(diagrams, target_betti=[1], weight=1.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-10)

    def test_weight_scaling(self) -> None:
        from pynerve.diff.ph_layer_module import topology_regularizer

        diagrams = [torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)]
        loss_w1 = topology_regularizer(diagrams, target_betti=[1], weight=1.0)
        loss_w2 = topology_regularizer(diagrams, target_betti=[1], weight=2.0)
        assert loss_w2.item() == pytest.approx(2.0 * loss_w1.item(), abs=1e-10)

    def test_empty_diagram(self) -> None:
        from pynerve.diff.ph_layer_module import topology_regularizer

        diagrams = [torch.empty(0, 2, dtype=torch.float64)]
        loss = topology_regularizer(diagrams, target_betti=[0], weight=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)


class TestPersistencePenalty:
    """Numerical correctness for persistence_penalty in ph_layer_module."""

    def test_all_above_threshold(self) -> None:
        from pynerve.diff.ph_layer_module import persistence_penalty

        diagrams = [torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)]
        loss = persistence_penalty(diagrams, min_persistence=0.5, weight=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_some_below_threshold(self) -> None:
        from pynerve.diff.ph_layer_module import persistence_penalty

        # persistence = [0.3, 2.0]; min_persistence=0.5
        # penalty = relu(0.5-0.3) + relu(0.5-2.0) = 0.2 + 0 = 0.2
        diagrams = [torch.tensor([[0.0, 0.3], [0.0, 2.0]], dtype=torch.float64)]
        loss = persistence_penalty(diagrams, min_persistence=0.5, weight=1.0)
        assert loss.item() == pytest.approx(0.2, abs=1e-6)

    def test_weight_scaling(self) -> None:
        from pynerve.diff.ph_layer_module import persistence_penalty

        diagrams = [torch.tensor([[0.0, 0.3]], dtype=torch.float64)]
        loss_w1 = persistence_penalty(diagrams, min_persistence=0.5, weight=1.0)
        loss_w2 = persistence_penalty(diagrams, min_persistence=0.5, weight=2.0)
        assert loss_w2.item() == pytest.approx(2.0 * loss_w1.item(), abs=1e-10)

    def test_empty_diagram(self) -> None:
        from pynerve.diff.ph_layer_module import persistence_penalty

        diagrams = [torch.empty(0, 2, dtype=torch.float64)]
        loss = persistence_penalty(diagrams, min_persistence=0.5, weight=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)
