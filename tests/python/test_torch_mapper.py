"""Tests for torch/mapper.py — MapperTransformer and public validators."""

from __future__ import annotations

import pytest
import torch

from pynerve.exceptions._validation import ValidationError
from pynerve.torch.mapper import (
    MapperTransformer,
    _validate_mapper_params,
    _validate_public_point_cloud,
)


# _validate_public_point_cloud 


class TestValidatePublicPointCloud:
    def test_valid(self):
        pc = torch.randn(5, 3, dtype=torch.float32)
        _validate_public_point_cloud(pc)

    def test_1d_raises(self):
        pc = torch.randn(5, dtype=torch.float32)
        with pytest.raises(ValidationError, match="2D"):
            _validate_public_point_cloud(pc)

    def test_empty_raises(self):
        pc = torch.empty((0, 3), dtype=torch.float32)
        with pytest.raises(ValidationError, match="non-empty"):
            _validate_public_point_cloud(pc)

    def test_nan_raises(self):
        pc = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="point_cloud"):
            _validate_public_point_cloud(pc)


# _validate_mapper_params 


class TestValidateMapperParams:
    def test_valid(self):
        pc = torch.randn(10, 3, dtype=torch.float32)
        result = _validate_mapper_params(pc, 10, 0.25, 0.5, 5, "dbscan", "pca_2d")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_invalid_cover_overlap_raises(self):
        pc = torch.randn(10, 3, dtype=torch.float32)
        with pytest.raises(ValidationError, match="cover_overlap"):
            _validate_mapper_params(pc, 10, 1.5, 0.5, 5, "dbscan", "pca_2d")

    def test_invalid_dbscan_eps_raises(self):
        pc = torch.randn(10, 3, dtype=torch.float32)
        with pytest.raises(ValidationError, match="dbscan_eps"):
            _validate_mapper_params(pc, 10, 0.25, -0.5, 5, "dbscan", "pca_2d")

    def test_unknown_clusterer_raises(self):
        pc = torch.randn(10, 3, dtype=torch.float32)
        with pytest.raises(ValidationError, match="clusterer"):
            _validate_mapper_params(pc, 10, 0.25, 0.5, 5, "kmeans", "pca_2d")

    def test_unknown_filter_raises(self):
        pc = torch.randn(10, 3, dtype=torch.float32)
        with pytest.raises(ValidationError, match="filter_function"):
            _validate_mapper_params(pc, 10, 0.25, 0.5, 5, "dbscan", "unknown_filter")

    def test_callable_filter_accepted(self):
        pc = torch.randn(10, 3, dtype=torch.float32)

        def my_filter(x):
            return x[:, 0]

        result = _validate_mapper_params(pc, 10, 0.25, 0.5, 5, "dbscan", my_filter)
        assert len(result) == 4

    def test_non_callable_non_string_raises(self):
        pc = torch.randn(10, 3, dtype=torch.float32)
        with pytest.raises(ValidationError, match="filter_function"):
            _validate_mapper_params(pc, 10, 0.25, 0.5, 5, "dbscan", 123)  # type: ignore[arg-type]


# MapperTransformer 


class TestMapperTransformer:
    def test_construction_defaults(self):
        mt = MapperTransformer()
        assert mt.filter_function == "pca_2d"
        assert mt.cover_resolution == 10
        assert mt.cover_overlap == 0.25

    def test_construction_custom(self):
        mt = MapperTransformer(
            filter_function="eccentricity",
            cover_resolution=20,
            cover_overlap=0.5,
            clusterer="single_linkage",
            dbscan_eps=1.0,
            dbscan_min_samples=3,
        )
        assert mt.cover_resolution == 20
        assert mt.clusterer == "single_linkage"

    def test_get_params(self):
        mt = MapperTransformer()
        params = mt.get_params()
        assert "filter_function" in params
        assert "cover_resolution" in params
        assert params["filter_function"] == "pca_2d"

    def test_set_params(self):
        mt = MapperTransformer()
        mt.set_params(cover_resolution=5, cover_overlap=0.1)
        assert mt.cover_resolution == 5
        assert mt.cover_overlap == 0.1

    def test_transform_not_fitted_raises(self):
        mt = MapperTransformer()
        pc = torch.randn(5, 3, dtype=torch.float32)
        with pytest.raises(ValidationError, match="not fitted"):
            mt.transform(pc)

    def test_initial_state(self):
        mt = MapperTransformer()
        assert mt.mapper_result_ is None
        assert mt.training_filter_values_ is None

    def test_custom_params_persist(self):
        mt = MapperTransformer(cover_resolution=42)
        params = mt.get_params()
        assert params["cover_resolution"] == 42
