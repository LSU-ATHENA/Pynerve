"""Comprehensive tests for pynerve.torch subpackage.

Covers the public API surface: lazy imports, PersistenceDiagram, diagram
distance functions, data utilities, nn layers, preprocessing, statistics,
vectorization, kernels, mapper, sklearn transformers, tensorboard logging,
and visualisation helpers.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _torch_backend_available() -> bool:
    """Check if the PyTorch C++ extension (pynerve_torch_internal) is present."""
    try:
        import nerve_torch_internal  # noqa: F401

        return True
    except ImportError:
        try:
            import pynerve_torch_internal  # noqa: F401

            return True
        except ImportError:
            return False


_torch_backend = pytest.mark.skipif(
    not _torch_backend_available(),
    reason="pynerve_torch_internal not available",
)


def _core_backend_available() -> bool:
    """Check if the core C++ extension (pynerve_internal) is present."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


_core_backend = pytest.mark.skipif(
    not _core_backend_available(),
    reason="pynerve_internal not available",
)


def _networkx_available() -> bool:
    try:
        import networkx  # noqa: F401

        return True
    except ImportError:
        return False


def _make_diagram_tensor(
    batch: int = 2,
    pairs: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """Create a valid persistence diagram tensor (birth, death, dim) with birth < death."""
    torch.manual_seed(seed)
    births = torch.rand(batch, pairs, 1)
    deaths = births + torch.rand(batch, pairs, 1) + 0.1
    dims = torch.randint(0, 2, (batch, pairs, 1)).float()
    return torch.cat([births, deaths, dims], dim=-1)


def _make_2d_diagram(pairs: int = 3, seed: int = 42) -> torch.Tensor:
    """Create a 2D (unbatched) diagram tensor with (birth, death) columns."""
    torch.manual_seed(seed)
    births = torch.rand(pairs, 1)
    deaths = births + torch.rand(pairs, 1) + 0.1
    return torch.cat([births, deaths], dim=-1)


def _make_point_cloud(
    n_points: int = 8,
    dim: int = 3,
    batch: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if batch == 1:
        return torch.rand(n_points, dim)
    return torch.rand(batch, n_points, dim)


# Lazy import mechanism


# helpers


# tensorboard


class TestTensorboardLogging:
    """tensorboard logging functions -- smoke tests with mock writer."""

    @pytest.fixture
    def mock_writer(self):
        """Create a minimal mock SummaryWriter-compatible object."""

        class MockWriter:
            def __init__(self):
                self.calls: list[tuple[str, Any]] = []

            def add_image(self, tag, img_tensor, step):
                self.calls.append(("add_image", (tag, img_tensor, step)))

            def add_scalar(self, tag, value, step):
                self.calls.append(("add_scalar", (tag, value, step)))

            def add_histogram(self, tag, values, step):
                self.calls.append(("add_histogram", (tag, values, step)))

            def close(self):
                self.calls.append(("close", ()))

        return MockWriter()

    def test_log_diagram_image(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_diagram

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        log_diagram(mock_writer, diagram, step=0, tag="test", method="image")
        assert any(c[0] == "add_image" for c in mock_writer.calls)

    def test_log_diagram_heatmap(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_diagram

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        log_diagram(mock_writer, diagram, step=0, tag="test", method="heatmap")
        assert any(c[0] == "add_image" for c in mock_writer.calls)

    def test_log_diagram_scatter(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_diagram

        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        log_diagram(mock_writer, diagram, step=0, tag="test", method="scatter")
        assert any(c[0] == "add_histogram" for c in mock_writer.calls)

    def test_log_diagram_invalid_method_raises(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_diagram

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError):
            log_diagram(mock_writer, diagram, step=0, method="bogus")

    def test_log_landscape(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_landscape

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        log_landscape(mock_writer, diagram, step=0, k=2)
        assert any(c[0] == "add_scalar" for c in mock_writer.calls)

    def test_log_landscape_invalid_k_raises(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_landscape

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError):
            log_landscape(mock_writer, diagram, step=0, k=0)

    def test_log_betti_curve(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_betti_curve

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        log_betti_curve(mock_writer, diagram, step=0, num_samples=5)
        assert any(c[0] == "add_scalar" for c in mock_writer.calls)

    def test_log_betti_curve_invalid_samples_raises(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_betti_curve

        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError):
            log_betti_curve(mock_writer, diagram, step=0, num_samples=0)

    def test_log_statistics(self, mock_writer) -> None:
        from pynerve.torch.tensorboard import log_statistics

        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        log_statistics(mock_writer, diagram, step=0, dims=[0, 1])
        assert any(c[0] == "add_scalar" for c in mock_writer.calls)


class TestDiagramSummaryWriter:
    """DiagramSummaryWriter construction and delegation."""

    def test_construct(self) -> None:
        pytest.importorskip("tensorboard")
        from pynerve.torch.tensorboard import DiagramSummaryWriter

        writer = DiagramSummaryWriter(log_dir="/tmp/_test_tb_unused")
        assert writer is not None
        writer.close()

    def test_is_context_manager(self) -> None:
        from pynerve.torch.tensorboard import DiagramSummaryWriter

        assert hasattr(DiagramSummaryWriter, "__enter__")
        assert hasattr(DiagramSummaryWriter, "__exit__")

    def test_methods_exist(self) -> None:
        from pynerve.torch.tensorboard import DiagramSummaryWriter

        assert hasattr(DiagramSummaryWriter, "add_diagram")
        assert hasattr(DiagramSummaryWriter, "add_landscape")
        assert hasattr(DiagramSummaryWriter, "add_betti_curve")
        assert hasattr(DiagramSummaryWriter, "add_diagram_stats")
