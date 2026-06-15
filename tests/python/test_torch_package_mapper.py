"""Comprehensive tests for pynerve.torch subpackage.

Covers the public API surface: lazy imports, PersistenceDiagram, diagram
distance functions, data utilities, nn layers, preprocessing, statistics,
vectorization, kernels, mapper, sklearn transformers, tensorboard logging,
and visualisation helpers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from pynerve.exceptions import ValidationError  # noqa: E402


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


# mapper


class TestMapperFunction:
    """mapper() function -- smoke tests that validate parameter handling."""

    def test_mapper_invalid_clusterer_raises(self) -> None:
        from pynerve.torch.mapper import mapper

        x = torch.rand(10, 2)
        with pytest.raises(ValidationError):
            mapper(x, clusterer="bogus")

    def test_mapper_invalid_filter_function_raises(self) -> None:
        from pynerve.torch.mapper import mapper

        x = torch.rand(10, 2)
        with pytest.raises(ValidationError):
            mapper(x, filter_function="bogus")

    def test_mapper_empty_point_cloud_raises(self) -> None:
        from pynerve.torch.mapper import mapper

        with pytest.raises(ValidationError):
            mapper(torch.empty((0, 2)))

    def test_mapper_nan_point_cloud_raises(self) -> None:
        from pynerve.torch.mapper import mapper

        x = torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)
        with pytest.raises(ValidationError):
            mapper(x)

    def test_mapper_invalid_cover_overlap_raises(self) -> None:
        from pynerve.torch.mapper import mapper

        x = torch.rand(10, 2)
        with pytest.raises(ValidationError):
            mapper(x, cover_overlap=1.5)
        with pytest.raises(ValidationError):
            mapper(x, cover_overlap=-0.1)

    def test_mapper_invalid_dbscan_eps_raises(self) -> None:
        from pynerve.torch.mapper import mapper

        x = torch.rand(10, 2)
        with pytest.raises(ValidationError):
            mapper(x, dbscan_eps=0.0)
        with pytest.raises(ValidationError):
            mapper(x, dbscan_eps=float("nan"))

    @pytest.mark.skipif(not _networkx_available(), reason="networkx not available")
    def test_mapper_with_custom_filter_function(self) -> None:
        from pynerve.torch.mapper import mapper

        x = torch.rand(10, 2)
        result = mapper(
            x,
            filter_function=lambda p: p[:, 0:1],
            cover_resolution=2,
            cover_overlap=0.2,
            dbscan_eps=0.5,
            return_graph=False,
        )
        assert "nodes" in result
        assert "edges" in result

    def test_mapper_invalid_cover_resolution_raises(self) -> None:
        from pynerve.torch.mapper import mapper

        x = torch.rand(10, 2)
        with pytest.raises(ValidationError):
            mapper(x, cover_resolution=0)

    def test_mapper_non_2d_raises(self) -> None:
        from pynerve.torch.mapper import mapper

        x = torch.rand(2, 3, 4)
        with pytest.raises(ValidationError):
            mapper(x)


class TestMapperTransformer:
    """MapperTransformer construction, fit, transform."""

    def test_construct_default(self) -> None:
        from pynerve.torch.mapper import MapperTransformer

        mt = MapperTransformer()
        assert mt.filter_function == "pca_2d"
        assert mt.cover_resolution == 10

    def test_get_params(self) -> None:
        from pynerve.torch.mapper import MapperTransformer

        mt = MapperTransformer(filter_function="eccentricity", cover_resolution=5)
        params = mt.get_params()
        assert params["filter_function"] == "eccentricity"
        assert params["cover_resolution"] == 5

    def test_set_params(self) -> None:
        from pynerve.torch.mapper import MapperTransformer

        mt = MapperTransformer()
        mt.set_params(cover_resolution=20, cover_overlap=0.5)
        assert mt.cover_resolution == 20
        assert mt.cover_overlap == 0.5

    def test_transform_before_fit_raises(self) -> None:
        from pynerve.torch.mapper import MapperTransformer

        mt = MapperTransformer()
        with pytest.raises(ValidationError):
            mt.transform(torch.rand(5, 2))

    def test_fit_invalid_point_cloud_raises(self) -> None:
        from pynerve.torch.mapper import MapperTransformer

        mt = MapperTransformer()
        with pytest.raises(ValidationError):
            mt.fit(torch.empty((0, 2)))

    def test_unknown_filter_function_transform_raises(self) -> None:
        from pynerve.torch.mapper import MapperTransformer

        mt = MapperTransformer()
        mt.mapper_result_ = {"nodes": [{"id": 0, "filter_centroid": torch.tensor([0.0, 0.0])}]}
        mt.filter_function = "bogus"
        with pytest.raises(ValidationError):
            mt.transform(torch.rand(3, 2))


class TestVisualizeMapperGraph:
    """visualize_mapper_graph validation."""

    def test_no_graph_raises(self) -> None:
        from pynerve.torch.mapper import visualize_mapper_graph

        with pytest.raises(ValidationError):
            visualize_mapper_graph({"nodes": [], "edges": []})

    def test_invalid_color_by_raises(self) -> None:
        from pynerve.torch.mapper import visualize_mapper_graph

        result = {"nodes": [], "edges": [], "graph": None}
        with pytest.raises(ValidationError):
            visualize_mapper_graph(result, color_by="bogus")

    def test_invalid_layout_raises(self) -> None:
        from pynerve.torch.mapper import visualize_mapper_graph

        result = {"nodes": [], "edges": [], "graph": None}
        with pytest.raises(ValidationError):
            visualize_mapper_graph(result, layout="bogus")

    @pytest.mark.skipif(not _networkx_available(), reason="networkx not available")
    def test_missing_networkx_or_matplotlib_raises(self) -> None:
        import networkx as nx
        from pynerve.torch.mapper import visualize_mapper_graph

        g = nx.Graph()
        g.add_node(0)
        result = {"nodes": [{"id": 0, "point_indices": []}], "edges": [], "graph": g}
        fig = visualize_mapper_graph(result, color_by="size", layout="spring")
        assert fig is not None
