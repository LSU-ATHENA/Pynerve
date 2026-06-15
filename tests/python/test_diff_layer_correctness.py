"""Numerical correctness tests for differentiable PH layers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestFiltrationLearningLayer:
    """Numerical correctness for FiltrationLearningLayer."""

    def test_forward_shape(self) -> None:
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3, hidden_dims=[8, 8])
        pts = torch.randn(2, 10, 3)
        out = layer(pts)
        assert out.shape == (2, 10)

    def test_output_finite(self) -> None:
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3, hidden_dims=[8, 8])
        pts = torch.randn(2, 10, 3)
        out = layer(pts)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3, hidden_dims=[8, 8])
        pts = torch.randn(2, 10, 3, requires_grad=True)
        out = layer(pts)
        loss = out.sum()
        loss.backward()
        assert pts.grad is not None
        assert pts.grad.abs().sum().item() > 0


class TestLearnableFiltrationPersistence:
    """Numerical correctness for LearnableFiltrationPersistence."""

    def test_forward_tuple_shapes(self) -> None:
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        model = LearnableFiltrationPersistence(input_dim=2, max_dim=0, hidden_dims=[4, 4])
        pts = torch.randn(2, 5, 2)
        result, filt = model(pts)
        assert isinstance(result, tuple)
        assert isinstance(filt, torch.Tensor)
        assert filt.shape == (2, 5)

    def test_forward_finite(self) -> None:
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        model = LearnableFiltrationPersistence(input_dim=2, max_dim=0, hidden_dims=[4, 4])
        pts = torch.randn(2, 5, 2)
        result, filt = model(pts)
        assert isinstance(result, tuple)
        assert isinstance(filt, torch.Tensor)
        assert torch.isfinite(filt).all()


class TestTopologyLoss:
    """Numerical correctness for TopologyLoss from ph_layer_module."""

    def test_identical_diagrams_zero(self) -> None:
        from pynerve.diff.ph_layer_module import TopologyLoss

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        loss_fn = TopologyLoss(target_diagrams=[d], wasserstein_p=2.0)
        val = loss_fn([d])
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_diagrams_positive(self) -> None:
        from pynerve.diff.ph_layer_module import TopologyLoss

        d1 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        loss_fn = TopologyLoss(target_diagrams=[d2], wasserstein_p=2.0)
        val = loss_fn([d1])
        assert val.item() > 0

    def test_gradient_flow(self) -> None:
        from pynerve.diff.ph_layer_module import TopologyLoss

        d = torch.tensor([[0.0, 1.0]], dtype=torch.float64, requires_grad=True)
        loss_fn = TopologyLoss(target_diagrams=[d.detach()], wasserstein_p=2.0)
        val = loss_fn([d])
        val.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()


class TestDifferentiablePersistentHomology:
    """Numerical correctness for DifferentiablePersistentHomology."""

    def test_forward_two_points(self) -> None:
        from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology

        layer = DifferentiablePersistentHomology(max_dim=0, max_radius=5.0)
        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        diagrams = layer(pts)
        assert len(diagrams) == 1
        d0 = diagrams[0]
        assert d0.shape[1] == 2

    def test_forward_batched(self) -> None:
        from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology

        layer = DifferentiablePersistentHomology(max_dim=0, max_radius=5.0)
        pts = torch.randn(3, 10, 2)
        result = layer(pts)
        assert isinstance(result, tuple)
        assert len(result) == 1
