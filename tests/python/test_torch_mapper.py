"""Tests for torch/mapper.py -- mapper, MapperTransformer, validators."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


def _make_points(n=20, d=3):
    return torch.randn(n, d)


class TestMapperValidation:
    def test_invalid_point_cloud_not_2d(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="2D"):
            mapper(torch.randn(5))

    def test_empty_point_cloud(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="non-empty"):
            mapper(torch.empty((0, 3)))

    def test_invalid_cover_resolution(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="positive"):
            mapper(_make_points(), cover_resolution=0)

    def test_invalid_cover_overlap_negative(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="cover_overlap"):
            mapper(_make_points(), cover_overlap=-0.1)

    def test_invalid_cover_overlap_ge_one(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="cover_overlap"):
            mapper(_make_points(), cover_overlap=1.0)

    def test_invalid_dbscan_eps(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="dbscan_eps"):
            mapper(_make_points(), dbscan_eps=0.0)

    def test_invalid_clusterer(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="clusterer"):
            mapper(_make_points(), clusterer="bad")

    def test_invalid_filter_function(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="filter_function"):
            mapper(_make_points(), filter_function="bad")

    def test_nan_point_cloud(self):
        from pynerve.torch.mapper import mapper
        from pynerve.exceptions import ValidationError

        pts = torch.tensor([[0.0, 0.0], [float("nan"), 1.0]])
        with pytest.raises(ValidationError):
            mapper(pts)


class TestMapperFunction:
    def test_pca_2d_fallback(self):
        from pynerve.torch.mapper import mapper

        pts = _make_points(30, 3)
        result = mapper(pts, filter_function="pca_2d", cover_resolution=5, cover_overlap=0.3)
        assert "nodes" in result
        assert "edges" in result
        assert "filter_values" in result

    def test_pca_1d_fallback(self):
        from pynerve.torch.mapper import mapper

        pts = _make_points(30, 3)
        result = mapper(pts, filter_function="pca_1d", cover_resolution=5)
        assert "nodes" in result

    def test_eccentricity_fallback(self):
        from pynerve.torch.mapper import mapper

        pts = _make_points(25, 2)
        result = mapper(pts, filter_function="eccentricity", cover_resolution=5)
        assert "nodes" in result

    def test_identity_filter_fallback(self):
        from pynerve.torch.mapper import mapper

        pts = _make_points(20, 2)
        result = mapper(pts, filter_function="identity", cover_resolution=5)
        assert "nodes" in result

    def test_custom_filter_callable(self):
        from pynerve.torch.mapper import mapper

        pts = _make_points(20, 3)

        def my_filter(points):
            return torch.randn(points.shape[0], 2)

        result = mapper(pts, filter_function=my_filter, cover_resolution=5)
        assert "nodes" in result
        assert "filter_values" in result

    def test_return_graph_false(self):
        from pynerve.torch.mapper import mapper

        pts = _make_points(20, 2)
        result = mapper(pts, cover_resolution=5, return_graph=False)
        assert "graph" not in result or result.get("graph") is None

    def test_connected_clusterer(self):
        from pynerve.torch.mapper import mapper

        pts = _make_points(20, 2)
        result = mapper(pts, clusterer="connected", cover_resolution=5)
        assert "nodes" in result


class TestMapperTransformer:
    def test_init_defaults(self):
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer()
        assert t.filter_function == "pca_2d"
        assert t.cover_resolution == 10
        assert t.cover_overlap == 0.25
        assert t.mapper_result_ is None

    def test_fit(self):
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer(cover_resolution=5)
        pts = _make_points(20, 2)
        result = t.fit(pts)
        assert result is t
        assert t.mapper_result_ is not None
        assert t.training_filter_values_ is not None

    def test_fit_transform(self):
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer(cover_resolution=5)
        pts = _make_points(20, 2)
        result = t.fit_transform(pts)
        assert "nodes" in result

    def test_transform_not_fitted(self):
        from pynerve.torch.mapper import MapperTransformer
        from pynerve.exceptions import ValidationError

        t = MapperTransformer()
        with pytest.raises(ValidationError, match="not fitted"):
            t.transform(_make_points(5))

    def test_transform_after_fit(self):
        """Test transform() assignment logic with manually constructed nodes.

        Uses identity filter so filter values are exactly X[:, :2],
        making nearest-node assignments predictable.
        """
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer(filter_function="identity", cover_resolution=5)
        # Manually construct mapper_result_ with 3 nodes at known filter positions
        t.mapper_result_ = {
            "nodes": [
                {"id": 0, "point_indices": [0], "filter_centroid": torch.tensor([0.0, 0.0])},
                {"id": 1, "point_indices": [1], "filter_centroid": torch.tensor([5.0, 5.0])},
                {"id": 2, "point_indices": [2], "filter_centroid": torch.tensor([10.0, 10.0])},
            ],
            "edges": [],
            "filter_values": torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]),
        }
        t.training_filter_values_ = t.mapper_result_["filter_values"]

        # With identity filter, filter_vals = X[:, :2], so assignments are
        # determined by Euclidean distance to each node's filter_centroid.
        new_pts = torch.tensor([[0.1, 0.1], [4.9, 5.1], [10.2, 9.8]])
        assignments = t.transform(new_pts)
        assert assignments.shape == (3,)
        assert assignments.dtype == torch.long
        assert assignments[0] == 0  # [0.1,0.1] closest to node 0 at [0,0]
        assert assignments[1] == 1  # [4.9,5.1] closest to node 1 at [5,5]
        assert assignments[2] == 2  # [10.2,9.8] closest to node 2 at [10,10]

    def test_transform_identity_filter(self):
        """Test transform() with identity filter -- filter values are X[:, :2] directly."""
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer(filter_function="identity", cover_resolution=5)
        t.mapper_result_ = {
            "nodes": [
                {"id": 10, "point_indices": [0], "filter_centroid": torch.tensor([0.0, 0.0])},
                {"id": 20, "point_indices": [1], "filter_centroid": torch.tensor([1.0, 1.0])},
            ],
            "edges": [],
            "filter_values": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        }
        t.training_filter_values_ = t.mapper_result_["filter_values"]

        # With identity filter, filter_vals = X[:, :2], so [0.1, 0.1] is closest
        # to node 10 at [0,0], and [0.9, 0.9] is closest to node 20 at [1,1]
        new_pts = torch.tensor([[0.1, 0.1, 5.0], [0.9, 0.9, 3.0]])
        assignments = t.transform(new_pts)
        assert assignments.shape == (2,)
        assert assignments[0] == 10
        assert assignments[1] == 20

    def test_transform_pca_1d_filter(self):
        """Test transform() with pca_1d filter -- 1D filter values."""
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer(filter_function="pca_1d", cover_resolution=5)
        t.mapper_result_ = {
            "nodes": [
                {"id": 0, "point_indices": [0], "filter_centroid": torch.tensor([0.0, 0.0])},
                {"id": 1, "point_indices": [1], "filter_centroid": torch.tensor([3.0, 0.0])},
            ],
            "edges": [],
            "filter_values": torch.tensor([[0.0, 0.0], [3.0, 0.0]]),
        }
        t.training_filter_values_ = t.mapper_result_["filter_values"]

        new_pts = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
        assignments = t.transform(new_pts)
        assert assignments.shape == (2,)

    def test_transform_eccentricity_filter(self):
        """Test transform() with eccentricity filter."""
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer(filter_function="eccentricity", cover_resolution=5)
        t.mapper_result_ = {
            "nodes": [
                {"id": 0, "point_indices": [0], "filter_centroid": torch.tensor([0.0, 0.0])},
                {"id": 1, "point_indices": [1], "filter_centroid": torch.tensor([2.0, 0.0])},
            ],
            "edges": [],
            "filter_values": torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        }
        t.training_filter_values_ = t.mapper_result_["filter_values"]

        new_pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        assignments = t.transform(new_pts)
        assert assignments.shape == (2,)

    def test_transform_empty_nodes_raises(self):
        """transform() raises ValidationError when mapper_result_ has no nodes."""
        from pynerve.torch.mapper import MapperTransformer
        from pynerve.exceptions import ValidationError

        t = MapperTransformer(filter_function="pca_2d")
        t.mapper_result_ = {"nodes": [], "edges": [], "filter_values": torch.empty(0, 2)}
        with pytest.raises(ValidationError, match="no nodes"):
            t.transform(_make_points(5, 2))

    def test_transform_unknown_filter(self):
        from pynerve.torch.mapper import MapperTransformer
        from pynerve.exceptions import ValidationError

        t = MapperTransformer(filter_function="pca_2d", cover_resolution=5)
        t.fit(_make_points(20, 2))
        t.filter_function = "bad"
        with pytest.raises(ValidationError, match="filter_function"):
            t.transform(_make_points(5, 2))

    def test_get_params(self):
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer(filter_function="eccentricity", cover_resolution=7)
        params = t.get_params()
        assert params["filter_function"] == "eccentricity"
        assert params["cover_resolution"] == 7

    def test_set_params(self):
        from pynerve.torch.mapper import MapperTransformer

        t = MapperTransformer()
        t.set_params(cover_resolution=15, dbscan_eps=1.0)
        assert t.cover_resolution == 15
        assert t.dbscan_eps == 1.0


class TestVisualizeMapperGraph:
    def test_no_graph_raises(self):
        from pynerve.torch.mapper import visualize_mapper_graph
        from pynerve.exceptions import ValidationError

        result = {"nodes": [], "edges": []}
        with pytest.raises(ValidationError, match="no graph"):
            visualize_mapper_graph(result)

    def test_invalid_color_by(self):
        from pynerve.torch.mapper import visualize_mapper_graph
        from pynerve.exceptions import ValidationError

        result = {
            "nodes": [{"id": 0, "point_indices": [0]}],
            "edges": [],
            "graph": "dummy",
        }
        with pytest.raises(ValidationError, match="color_by"):
            visualize_mapper_graph(result, color_by="bad")

    def test_invalid_layout(self):
        from pynerve.torch.mapper import visualize_mapper_graph
        from pynerve.exceptions import ValidationError

        result = {
            "nodes": [{"id": 0, "point_indices": [0]}],
            "edges": [],
            "graph": "dummy",
        }
        with pytest.raises(ValidationError, match="layout"):
            visualize_mapper_graph(result, layout="bad")
