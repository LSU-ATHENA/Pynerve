"""Numerical correctness tests for nn topological regularization losses."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestDiagramMatchingLoss:
    """Numerical correctness for DiagramMatchingLoss with known values."""

    def test_identical_diagrams_zero(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        d = [torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)]
        val = loss([d], [d])
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_bottleneck_identical_zero(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="bottleneck")
        d = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        val = loss([d], [d])
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_single_point_distance(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        d1 = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        d2 = [torch.tensor([[0.0, 2.0]], dtype=torch.float64)]
        val = loss([d1], [d2])
        # wasserstein(p=2): (mean(pred_to_target^2) + mean(target_to_pred^2))^(1/2)
        # = ((1^2) + (1^2))^(1/2) = sqrt(2)
        assert val.item() == pytest.approx(2**0.5, abs=1e-6)

    def test_bottleneck_single_point(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="bottleneck")
        d1 = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        d2 = [torch.tensor([[0.0, 2.0]], dtype=torch.float64)]
        val = loss([d1], [d2])
        assert val.item() == pytest.approx(1.0, abs=1e-6)

    def test_shifted_point_known_value(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        d1 = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        d2 = [torch.tensor([[0.5, 1.5]], dtype=torch.float64)]
        val = loss([d1], [d2])
        # pairwise dist = sqrt(0.5^2 + 0.5^2) = sqrt(0.5)
        # (dist^2 * 2)^(1/2) = sqrt(2 * 0.5) = sqrt(1) = 1.0
        assert val.item() == pytest.approx(1.0, abs=1e-6)

    def test_empty_one_side(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        empty = [torch.empty(0, 2, dtype=torch.float64)]
        d = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        val = loss([empty], [d])
        # diagonal distance = |1-0| / sqrt(2) = 1/sqrt(2) ~ 0.707
        assert val.item() == pytest.approx(1.0 / 2**0.5, abs=1e-6)

    def test_empty_both_zero(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss()
        empty = [torch.empty(0, 2, dtype=torch.float64)]
        val = loss([empty], [empty])
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_single_point_bottleneck_shifted(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="bottleneck")
        d1 = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        d2 = [torch.tensor([[0.5, 1.5]], dtype=torch.float64)]
        val = loss([d1], [d2])
        assert val.item() == pytest.approx(0.5**0.5, abs=1e-6)

    def test_multi_dimension(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        d1 = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        ]
        d2 = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        ]
        val = loss([d1], [d2])
        assert val.item() == pytest.approx(0.0, abs=1e-10)

    def test_gradient_flow(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        val = loss([[d]], [[target]])
        val.backward()
        assert d.grad is not None

    def test_diagonal_distance_empty_target(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0)
        d = [torch.tensor([[0.0, 2.0]], dtype=torch.float64)]
        empty = [torch.empty(0, 2, dtype=torch.float64)]
        val = loss([d], [empty])
        assert val.item() > 0

    def test_weight_scaling(self) -> None:
        from pynerve.nn.topo_regularization import DiagramMatchingLoss

        loss_w2 = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0, weight=2.0)
        loss_w1 = DiagramMatchingLoss(distance_metric="wasserstein", p=2.0, weight=1.0)
        d1 = [torch.tensor([[0.0, 1.0]], dtype=torch.float64)]
        d2 = [torch.tensor([[0.0, 2.0]], dtype=torch.float64)]
        v2 = loss_w2([d1], [d2])
        v1 = loss_w1([d1], [d2])
        assert v2.item() == pytest.approx(2.0 * v1.item(), abs=1e-10)


class TestPersistenceEntropyLoss:
    """Numerical correctness for PersistenceEntropyLoss."""

    def test_forward_finite(self) -> None:
        from pynerve.nn.topo_regularization import PersistenceEntropyLoss

        loss = PersistenceEntropyLoss(target_entropy=2.0, max_dim=0)
        x = torch.randn(2, 10, 3)
        val = loss(x)
        assert torch.isfinite(val)


class TestTopologicalRegularizationLoss:
    """Numerical correctness for TopologicalRegularizationLoss."""

    def test_forward_finite(self) -> None:
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        loss = TopologicalRegularizationLoss(target_betti=[10], max_dim=0)
        x = torch.randn(2, 20, 3)
        val = loss(x)
        assert torch.isfinite(val)

    def test_with_default_target(self) -> None:
        from pynerve.nn.topo_regularization import TopologicalRegularizationLoss

        loss = TopologicalRegularizationLoss(target_betti=None, max_dim=0)
        x = torch.randn(2, 5, 3)
        val = loss(x)
        assert torch.isfinite(val)


class TestTopologicalComplexityLoss:
    """Numerical correctness for TopologicalComplexityLoss."""

    def test_forward_finite(self) -> None:
        from pynerve.nn.topo_regularization import TopologicalComplexityLoss

        loss = TopologicalComplexityLoss(min_features=1, max_features=100, max_dim=0)
        x = torch.randn(2, 10, 3)
        val = loss(x)
        assert torch.isfinite(val)
