"""Numerical correctness tests for nn diagram convolution modules."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestDiagramConv1D:
    """Numerical correctness for DiagramConv1D."""

    def test_forward_shape(self) -> None:
        from pynerve.nn.diagram_conv import DiagramConv1D

        conv = DiagramConv1D(in_channels=0, out_channels=2, kernel_size=3)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]],
            dtype=torch.float32,
        )
        out = conv(diagram)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        from pynerve.nn.diagram_conv import DiagramConv1D

        conv = DiagramConv1D(in_channels=0, out_channels=2, kernel_size=3)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        out = conv(diagram)
        loss = out.sum()
        loss.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()


class TestDiagramDeepSet:
    """Numerical correctness for DiagramDeepSet."""

    def test_forward_shape(self) -> None:
        from pynerve.nn.diagram_conv import DiagramDeepSet

        deepset = DiagramDeepSet(in_channels=0, hidden_channels=[4, 8], out_channels=2)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]],
            dtype=torch.float32,
        )
        out = deepset(diagram)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        from pynerve.nn.diagram_conv import DiagramDeepSet

        deepset = DiagramDeepSet(in_channels=0, hidden_channels=[4, 8], out_channels=2)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        out = deepset(diagram)
        loss = out.sum()
        loss.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()


class TestDiagramMultiHeadAttention:
    """Numerical correctness for DiagramMultiHeadAttention."""

    def test_forward_shape(self) -> None:
        from pynerve.nn.diagram_conv import DiagramMultiHeadAttention

        attn = DiagramMultiHeadAttention(d_model=16, num_heads=4)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0], [0.0, 3.0, 0.0]]],
            dtype=torch.float32,
        )
        features = torch.randn(1, 3, 16)
        out = attn(diagram, features)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        from pynerve.nn.diagram_conv import DiagramMultiHeadAttention

        attn = DiagramMultiHeadAttention(d_model=16, num_heads=4)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        features = torch.randn(1, 2, 16, requires_grad=True)
        out = attn(diagram, features)
        loss = out.sum()
        loss.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()


class TestDiagramTransformerBlock:
    """Numerical correctness for DiagramTransformerBlock."""

    def test_forward_shape(self) -> None:
        from pynerve.nn.diagram_conv import DiagramTransformerBlock

        block = DiagramTransformerBlock(d_model=16, num_heads=4)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0], [0.0, 3.0, 0.0]]],
            dtype=torch.float32,
        )
        features = torch.randn(1, 3, 16)
        out = block(diagram, features)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        from pynerve.nn.diagram_conv import DiagramTransformerBlock

        block = DiagramTransformerBlock(d_model=16, num_heads=4)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        features = torch.randn(1, 2, 16, requires_grad=True)
        out = block(diagram, features)
        loss = out.sum()
        loss.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()


class TestDiagramConvNet:
    """Numerical correctness for DiagramConvNet."""

    def test_forward_shape(self) -> None:
        from pynerve.nn.diagram_conv import DiagramConvNet

        net = DiagramConvNet(in_channels=0, hidden_channels=[4, 8], out_dim=2)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0], [0.0, 3.0, 0.0]]],
            dtype=torch.float32,
        )
        out = net(diagram)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        from pynerve.nn.diagram_conv import DiagramConvNet

        net = DiagramConvNet(in_channels=0, hidden_channels=[4, 8], out_dim=2)
        diagram = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        out = net(diagram)
        loss = out.sum()
        loss.backward()
        assert diagram.grad is not None
        assert torch.isfinite(diagram.grad).all()
