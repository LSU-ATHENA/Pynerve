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


class TestLazyImports:
    """__getattr__ lazy import of submodules and forwarded attributes."""

    def test_submodule_data_accessible(self) -> None:
        import pynerve.torch

        data_mod = pynerve.torch.data
        assert data_mod is not None
        assert hasattr(data_mod, "PersistenceDataset")

    def test_submodule_preprocessing_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.preprocessing
        assert mod is not None
        assert hasattr(mod, "handle_infinite_deaths")

    def test_submodule_vectorization_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.vectorization
        assert mod is not None
        assert hasattr(mod, "persistence_image")

    def test_submodule_statistics_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.statistics
        assert mod is not None
        assert hasattr(mod, "total_persistence")

    def test_submodule_kernels_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.kernels
        assert mod is not None
        assert hasattr(mod, "gaussian_kernel")

    def test_submodule_sklearn_transformers_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.sklearn_transformers
        assert mod is not None
        assert hasattr(mod, "PersistenceTransformer")

    def test_submodule_nn_layers_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.nn_layers
        assert mod is not None
        assert hasattr(mod, "PersistenceLayer")

    def test_submodule_training_utils_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.training_utils
        assert mod is not None
        assert hasattr(mod, "DiagramDistanceLoss")

    def test_submodule_viz_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.viz
        assert mod is not None
        assert hasattr(mod, "get_plot_limits")

    def test_submodule_tensorboard_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.tensorboard
        assert mod is not None
        assert hasattr(mod, "DiagramSummaryWriter")

    def test_submodule_mapper_accessible(self) -> None:
        import pynerve.torch

        mod = pynerve.torch.mapper
        assert mod is not None
        assert hasattr(mod, "mapper")

    def test_forwarded_attr_batch_diagrams(self) -> None:
        import pynerve.torch

        assert pynerve.torch.batch_diagrams is not None

    def test_forwarded_attr_unbatch_diagrams(self) -> None:
        import pynerve.torch

        assert pynerve.torch.unbatch_diagrams is not None

    def test_forwarded_attr_persistence_dataset(self) -> None:
        import pynerve.torch

        assert pynerve.torch.PersistenceDataset is not None

    def test_forwarded_attr_point_cloud_dataset(self) -> None:
        import pynerve.torch

        assert pynerve.torch.PointCloudDataset is not None

    def test_forwarded_attr_collate_diagrams(self) -> None:
        import pynerve.torch

        assert pynerve.torch.collate_diagrams is not None

    def test_forwarded_attr_collate_point_clouds(self) -> None:
        import pynerve.torch

        assert pynerve.torch.collate_point_clouds is not None

    def test_forwarded_attr_mapper_transformer(self) -> None:
        import pynerve.torch

        assert pynerve.torch.MapperTransformer is not None

    def test_forwarded_attr_visualize_mapper_graph(self) -> None:
        import pynerve.torch

        assert pynerve.torch.visualize_mapper_graph is not None

    def test_forwarded_attr_alpha_persistence(self) -> None:
        import pynerve.torch

        assert pynerve.torch.alpha_persistence is not None

    def test_forwarded_attr_persistence_from_matrix(self) -> None:
        import pynerve.torch

        assert pynerve.torch.persistence_from_matrix is not None

    def test_forwarded_attr_persistence_image(self) -> None:
        import pynerve.torch

        assert pynerve.torch.persistence_image is not None

    def test_forwarded_attr_vr_persistence(self) -> None:
        import pynerve.torch

        assert pynerve.torch.vr_persistence is not None

    def test_forwarded_attr_witness_persistence(self) -> None:
        import pynerve.torch

        assert pynerve.torch.witness_persistence is not None

    def test_missing_attr_raises(self) -> None:
        import pynerve.torch

        with pytest.raises(AttributeError):
            _ = pynerve.torch.nonexistent_attribute


# Public API accessibility (__all__)


class TestPublicAPIAccessibility:
    """Every name in __all__ must be reachable via pynerve.torch."""

    def test_all_names_accessible(self) -> None:
        import pynerve.torch

        names = [
            "PersistenceDiagram",
            "PersistenceDataset",
            "PointCloudDataset",
            "vr_persistence",
            "witness_persistence",
            "alpha_persistence",
            "persistence_from_matrix",
            "mapper",
            "MapperTransformer",
            "visualize_mapper_graph",
            "diagram_wasserstein",
            "diagram_bottleneck",
            "persistence_image",
            "batch_diagrams",
            "unbatch_diagrams",
            "collate_diagrams",
            "collate_point_clouds",
            "preprocessing",
            "vectorization",
            "statistics",
            "kernels",
            "sklearn_transformers",
            "nn_layers",
            "training_utils",
            "data",
            "viz",
            "tensorboard",
        ]
        for name in names:
            assert hasattr(pynerve.torch, name), f"Missing: {name}"
            obj = getattr(pynerve.torch, name)
            assert obj is not None, f"None: {name}"
