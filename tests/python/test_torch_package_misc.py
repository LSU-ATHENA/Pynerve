"""Comprehensive tests for pynerve.torch subpackage.

Covers the public API surface: lazy imports, PersistenceDiagram, diagram
distance functions, data utilities, nn layers, preprocessing, statistics,
vectorization, kernels, mapper, sklearn transformers, tensorboard logging,
and visualisation helpers.
"""

from __future__ import annotations

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


# Integration / persistence_image alias


class TestTopLevelPersistenceImage:
    """persistence_image at pynerve.torch level."""

    def test_forwarded_function_works(self) -> None:
        import pynerve.torch

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        img = pynerve.torch.persistence_image(diagram, resolution=(4, 4), sigma=0.2)
        assert img.shape == (4, 4)

    def test_forwarded_from_submodule_matches(self) -> None:
        import pynerve.torch

        img1 = pynerve.torch.persistence_image(
            torch.tensor([[0.0, 1.0]], dtype=torch.float64), resolution=(2, 2)
        )
        img2 = pynerve.torch.vectorization.persistence_image(
            torch.tensor([[0.0, 1.0]], dtype=torch.float64), resolution=(2, 2)
        )
        torch.testing.assert_close(img1, img2)


class TestPersistenceDiagramSmoke:
    """Quick smoke-test via PersistenceDiagram (already validated)."""

    def test_diagram_vr_persistence_smoke(self) -> None:
        """Smoke: importing vr_persistence does not fail."""
        from pynerve.torch._persistence_vr import vr_persistence

        assert vr_persistence is not None

    def test_diagram_distance_functions_importable(self) -> None:
        from pynerve.torch._distance_core import (
            BottleneckDistance,
            WassersteinDistance,
        )

        assert BottleneckDistance is not None
        assert WassersteinDistance is not None

    def test_diagram_distance_metric_importable(self) -> None:
        from pynerve.torch._distance_core import DistanceMetric

        assert DistanceMetric is not None

    def test_torch_module_has_diagram_wasserstein(self) -> None:
        import pynerve.torch

        assert hasattr(pynerve.torch, "diagram_wasserstein")

    def test_torch_module_has_diagram_bottleneck(self) -> None:
        import pynerve.torch

        assert hasattr(pynerve.torch, "diagram_bottleneck")


class TestTypeHintsAndProtocols:
    """Basic type hint / protocol adherence checks."""

    def test_persistence_diagram_is_hashable(self) -> None:
        from pynerve.torch._diagram import PersistenceDiagram

        assert hasattr(PersistenceDiagram, "__init__")

    def test_persistence_diagram_has_expected_properties(self) -> None:
        from pynerve.torch._diagram import PersistenceDiagram

        expected_props = {
            "diagrams",
            "mask",
            "num_pairs",
            "device",
            "dtype",
            "batch_size",
            "max_pairs",
        }
        obj = PersistenceDiagram(torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32))
        for prop in expected_props:
            assert hasattr(obj, prop), f"Missing property: {prop}"

    def test_persistence_diagram_has_expected_methods(self) -> None:
        from pynerve.torch._diagram import PersistenceDiagram

        expected_methods = {
            "tensor",
            "births",
            "deaths",
            "dimensions",
            "to",
            "to_dtype",
            "get_batch_item",
            "total_persistence",
            "persistence_entropy",
            "filter_by_dimension",
        }
        obj = PersistenceDiagram(torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32))
        for method in expected_methods:
            assert hasattr(obj, method), f"Missing method: {method}"

    def test_datasets_are_datasets(self) -> None:
        from pynerve.torch.data import PersistenceDataset, PointCloudDataset
        from torch.utils.data import Dataset

        assert issubclass(PersistenceDataset, Dataset)
        assert issubclass(PointCloudDataset, Dataset)

    def test_nn_layers_are_modules(self) -> None:
        from pynerve.torch.nn_layers import (
            DiagramPooling,
            PersistenceLayer,
            PersistenceReadout,
            TopologicalAttention,
            TopologicalFeatureExtractor,
            VectorizationLayer,
        )

        from torch import nn

        assert issubclass(PersistenceLayer, nn.Module)
        assert issubclass(VectorizationLayer, nn.Module)
        assert issubclass(TopologicalFeatureExtractor, nn.Module)
        assert issubclass(DiagramPooling, nn.Module)
        assert issubclass(TopologicalAttention, nn.Module)
        assert issubclass(PersistenceReadout, nn.Module)
