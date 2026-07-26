"""Tests for pynerve/torch/mapper.py -- Mapper algorithm API and transformer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pynerve.torch.mapper import (
    MapperTransformer,
    _validate_mapper_params,
    _validate_public_point_cloud,
)
from pynerve.exceptions import ValidationError


class TestValidatePointCloud:
    def test_valid_2d(self):
        pc = torch.rand(10, 3)
        _validate_public_point_cloud(pc)

    def test_non_2d_raises(self):
        with pytest.raises(ValidationError, match="2D"):
            _validate_public_point_cloud(torch.rand(10))

    def test_empty_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            _validate_public_point_cloud(torch.empty(0, 3))


class TestValidateMapperParams:
    def test_valid(self):
        pc = torch.rand(10, 3)
        cr, co, eps, ms = _validate_mapper_params(
            pc, 10, 0.25, 0.5, 5, "dbscan", "pca_2d"
        )
        assert cr == 10

    def test_invalid_overlap(self):
        pc = torch.rand(10, 3)
        with pytest.raises(ValidationError, match="overlap"):
            _validate_mapper_params(pc, 10, 1.5, 0.5, 5, "dbscan", "pca_2d")

    def test_invalid_eps(self):
        pc = torch.rand(10, 3)
        with pytest.raises(ValidationError, match="eps"):
            _validate_mapper_params(pc, 10, 0.25, -0.1, 5, "dbscan", "pca_2d")

    def test_invalid_clusterer(self):
        pc = torch.rand(10, 3)
        with pytest.raises(ValidationError, match="clusterer"):
            _validate_mapper_params(pc, 10, 0.25, 0.5, 5, "kmeans", "pca_2d")

    def test_invalid_filter(self):
        pc = torch.rand(10, 3)
        with pytest.raises(ValidationError, match="filter"):
            _validate_mapper_params(pc, 10, 0.25, 0.5, 5, "dbscan", "nonexistent")


class TestMapperTransformer:
    def test_construction(self):
        mt = MapperTransformer()
        assert mt.cover_resolution == 10
        assert mt.cover_overlap == 0.25
        assert mt.clusterer == "dbscan"

    def test_construction_custom(self):
        mt = MapperTransformer(
            filter_function="eccentricity",
            cover_resolution=15,
            cover_overlap=0.3,
            clusterer="single_linkage",
        )
        assert mt.cover_resolution == 15

    def test_get_params(self):
        mt = MapperTransformer(cover_resolution=20)
        params = mt.get_params()
        assert params["cover_resolution"] == 20
        assert "dbscan_eps" in params

    def test_set_params(self):
        mt = MapperTransformer()
        mt.set_params(cover_resolution=30)
        assert mt.cover_resolution == 30

    def test_fit_python_fallback(self):
        mt = MapperTransformer()
        pc = torch.rand(50, 3)
        result = mt.fit(pc)
        assert result is mt
        assert mt.mapper_result_ is not None
        assert "nodes" in mt.mapper_result_
        assert "edges" in mt.mapper_result_

    def test_transform_not_fitted_raises(self):
        mt = MapperTransformer()
        with pytest.raises(ValidationError, match="not fitted"):
            mt.transform(torch.rand(5, 3))

    def test_transform_basic(self):
        mt = MapperTransformer()
        pc = torch.rand(50, 3)
        mt.fit(pc)
        if mt.mapper_result_ is not None and mt.mapper_result_["nodes"]:
            result = mt.transform(pc)
            assert result.shape == (50,)

    def test_transform_unknown_filter(self):
        mt = MapperTransformer(filter_function="pca_2d")
        pc = torch.rand(50, 3)
        mt.fit(pc)
        mt.filter_function = "unknown"
        with pytest.raises(ValidationError, match="filter"):
            mt.transform(pc)
