"""Numerical correctness tests for SSL topology modules."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn


class _RowEncoder(nn.Module):
    """Encoder that maps (1, n, 2) -> (1, n, out_dim)."""

    def __init__(self, out_dim: int = 8) -> None:
        super().__init__()
        self.output_dim = out_dim
        self.fc = nn.Linear(2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _RowDecoder(nn.Module):
    """Decoder that maps (n, in_dim) -> (n, 2)."""

    def __init__(self, in_dim: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestTopologyAugmentation:
    """Numerical correctness for TopologyAugmentation."""

    def test_augment_preserves_shape(self) -> None:
        from pynerve.ssl.contrastive_learning import TopologyAugmentation

        aug = TopologyAugmentation()
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        out = aug(d)
        assert out.shape == d.shape

    def test_augment_preserves_birth_death_order(self) -> None:
        from pynerve.ssl.contrastive_learning import TopologyAugmentation

        aug = TopologyAugmentation(min_death_gap=0.01)
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        out = aug(d)
        assert (out[:, 1] >= out[:, 0]).all()


class TestSimCLRTopology:
    """Numerical correctness for SimCLRTopology."""

    def test_forward_finite(self) -> None:
        from pynerve.ssl.contrastive_learning import SimCLRTopology

        enc = _RowEncoder(out_dim=32)
        model = SimCLRTopology(encoder=enc, projection_dim=4, temperature=0.5)
        diagrams = [
            torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.5], [0.0, 2.5]], dtype=torch.float32),
        ]
        loss = model(diagrams)
        assert torch.isfinite(loss)

    def test_temperature_effect(self) -> None:
        from pynerve.ssl.contrastive_learning import SimCLRTopology

        enc = _RowEncoder(out_dim=32)
        m1 = SimCLRTopology(encoder=enc, projection_dim=4, temperature=0.1)
        m2 = SimCLRTopology(encoder=enc, projection_dim=4, temperature=5.0)
        diagrams = [
            torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.5], [0.0, 2.5]], dtype=torch.float32),
        ]
        l1 = m1(diagrams)
        l2 = m2(diagrams)
        assert l1.item() != pytest.approx(l2.item(), abs=1e-6)


class TestBYOLTopology:
    """Numerical correctness for BYOLTopology."""

    def test_forward_finite(self) -> None:
        from pynerve.ssl.contrastive_learning import BYOLTopology

        enc = _RowEncoder(out_dim=64)
        model = BYOLTopology(encoder=enc, projection_dim=4, hidden_dim=8)
        diagrams = [
            torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.5], [0.0, 2.5]], dtype=torch.float32),
        ]
        loss = model(diagrams)
        assert torch.isfinite(loss)


class TestTopologyCompletionModel:
    """Numerical correctness for TopologyCompletionModel."""

    def test_forward_shape(self) -> None:
        from pynerve.ssl.topology_completion import TopologyCompletionModel

        enc = _RowEncoder(out_dim=8)
        dec = _RowDecoder(in_dim=8)
        model = TopologyCompletionModel(encoder=enc, decoder=dec, completion_threshold=0.1)
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        mask = torch.tensor([True, False], dtype=torch.bool)
        completed, encoding = model(d, mask)
        assert completed.shape == d.shape
        assert encoding.shape == (2, 8)

    def test_compute_loss_scalar(self) -> None:
        from pynerve.ssl.topology_completion import TopologyCompletionModel

        enc = _RowEncoder(out_dim=8)
        dec = _RowDecoder(in_dim=8)
        model = TopologyCompletionModel(encoder=enc, decoder=dec, completion_threshold=0.1)
        partial = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        mask = torch.tensor([True, False], dtype=torch.bool)
        target = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        loss = model.compute_loss(partial, mask, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_compute_loss_zero_when_no_mask(self) -> None:
        from pynerve.ssl.topology_completion import TopologyCompletionModel

        enc = _RowEncoder(out_dim=8)
        dec = _RowDecoder(in_dim=8)
        model = TopologyCompletionModel(encoder=enc, decoder=dec, completion_threshold=0.1)
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        mask = torch.tensor([True, True], dtype=torch.bool)
        loss = model.compute_loss(d, mask, d)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)


class TestTopologyDenoising:
    """Numerical correctness for TopologyDenoising."""

    def test_forward_shape(self) -> None:
        from pynerve.ssl.topology_completion import TopologyDenoising

        enc = _RowEncoder(out_dim=8)
        model = TopologyDenoising(encoder=enc, noise_threshold=0.1)
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        denoised, noise_scores = model(d)
        assert denoised.shape == d.shape
        assert noise_scores.shape == (2,)

    def test_noise_scores_are_probabilities(self) -> None:
        from pynerve.ssl.topology_completion import TopologyDenoising

        enc = _RowEncoder(out_dim=8)
        model = TopologyDenoising(encoder=enc, noise_threshold=0.1)
        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        _, noise_scores = model(d)
        assert (noise_scores >= 0).all()
        assert (noise_scores <= 1).all()

    def test_compute_loss_scalar(self) -> None:
        from pynerve.ssl.topology_completion import TopologyDenoising

        enc = _RowEncoder(out_dim=8)
        model = TopologyDenoising(encoder=enc, noise_threshold=0.1)
        noisy = torch.tensor([[0.1, 1.1], [0.0, 2.0]], dtype=torch.float32)
        clean = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        loss = model.compute_loss(noisy, clean)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_compute_loss_with_labels(self) -> None:
        from pynerve.ssl.topology_completion import TopologyDenoising

        enc = _RowEncoder(out_dim=8)
        model = TopologyDenoising(encoder=enc, noise_threshold=0.1)
        noisy = torch.tensor([[0.1, 1.1], [0.0, 2.0]], dtype=torch.float32)
        clean = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        labels = torch.tensor([True, False], dtype=torch.float32)
        loss = model.compute_loss(noisy, clean, labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
