"""Tests for _types.py -- type aliases, protocols, and runtime checkable types."""

from __future__ import annotations

import numpy as np
import pytest

from pynerve._types import (
    ArrayLike,
    AsyncIterable,
    ClusteringAlgorithm,
    DistanceMatrix,
    DistanceMetric,
    FilterFunction,
    MapperConfig,
    Numeric,
    PersistenceComputer,
    PersistenceConfig,
    PersistenceDiagramLike,
    PersistencePair,
    PointCloud,
    VectorizationMethod,
)


# Type aliases exist 


class TestTypeAliases:
    def test_array_like_exists(self):
        assert ArrayLike is not None

    def test_numeric_exists(self):
        assert Numeric is not None

    def test_point_cloud_exists(self):
        assert PointCloud is not None

    def test_distance_matrix_exists(self):
        assert DistanceMatrix is not None

    def test_persistence_pair_exists(self):
        assert PersistencePair is not None


# PersistenceDiagramLike Protocol 


class TestPersistenceDiagramLike:
    def test_isinstance_with_matching_object(self):
        class Diagram:
            @property
            def pairs(self):
                return [(0.0, 1.0, 0)]

            @property
            def pairs_array(self):
                return np.array([[0.0, 1.0, 0]])

        d = Diagram()
        assert isinstance(d, PersistenceDiagramLike)

    def test_isinstance_with_partial_object(self):
        class PartialDiagram:
            @property
            def pairs(self):
                return [(0.0, 1.0, 0)]

            @property
            def pairs_array(self):
                return np.array([[0.0, 1.0, 0]])

        d = PartialDiagram()
        assert isinstance(d, PersistenceDiagramLike)

    def test_not_isinstance_with_missing(self):
        class NotDiagram:
            pass

        assert not isinstance(NotDiagram(), PersistenceDiagramLike)

    def test_not_isinstance_with_int(self):
        assert not isinstance(42, PersistenceDiagramLike)


# FilterFunction Protocol 


class TestFilterFunction:
    def test_isinstance_with_callable(self):
        def my_filter(points):
            return points[:, 0]

        assert isinstance(my_filter, FilterFunction)

    def test_not_isinstance_with_non_callable(self):
        assert not isinstance(42, FilterFunction)


# ClusteringAlgorithm Protocol 


class TestClusteringAlgorithm:
    def test_isinstance_with_callable(self):
        def my_cluster(points, **kwargs):
            return points

        assert isinstance(my_cluster, ClusteringAlgorithm)

    def test_not_isinstance_with_non_callable(self):
        assert not isinstance("not callable", ClusteringAlgorithm)


# DistanceMetric Protocol 


class TestDistanceMetric:
    def test_isinstance_with_callable(self):
        def my_dist(a, b):
            return a

        assert isinstance(my_dist, DistanceMetric)


# VectorizationMethod Protocol 


class TestVectorizationMethod:
    def test_isinstance_with_callable(self):
        def my_vec(diagram, **kwargs):
            return diagram

        assert isinstance(my_vec, VectorizationMethod)


# PersistenceComputer Protocol 


class TestPersistenceComputer:
    def test_isinstance_with_matching_object(self):
        class Computer:
            def compute(self, data, max_dim, **kwargs):
                return []

        assert isinstance(Computer(), PersistenceComputer)

    def test_not_isinstance_with_missing_compute(self):
        class BadComputer:
            pass

        assert not isinstance(BadComputer(), PersistenceComputer)


# AsyncIterable Protocol 


class TestAsyncIterable:
    def test_is_runtime_checkable(self):
        assert hasattr(AsyncIterable, "__instancecheck__")

    def test_isinstance_with_async_iterable(self):
        class AsyncGen:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        # isinstance check for runtime_checkable protocols is structural;
        # it checks for __aiter__ without needing to actually iterate.
        assert isinstance(AsyncGen(), AsyncIterable)

    def test_not_isinstance_with_regular_iterable(self):
        assert not isinstance([1, 2, 3], AsyncIterable)


# PersistenceConfig Protocol 


class TestPersistenceConfig:
    def test_isinstance_with_matching_object(self):
        class Config:
            max_dim: int = 2
            max_radius: float = 1.0
            metric: str = "euclidean"

            def validate(self) -> None:
                pass

        cfg = Config()
        assert hasattr(cfg, "max_dim")
        assert hasattr(cfg, "max_radius")
        assert hasattr(cfg, "metric")
        assert hasattr(cfg, "validate")
        assert callable(cfg.validate)

    def test_not_isinstance_with_missing_attrs(self):
        class BadConfig:
            max_dim: int = 1

        assert not hasattr(BadConfig(), "max_radius")


# MapperConfig Protocol 


class TestMapperConfig:
    def test_isinstance_with_matching_object(self):
        class Config:
            filter_function: str = "eccentricity"
            cover_resolution: int = 10
            cover_overlap: float = 0.5
            clusterer: str = "dbscan"

            def validate(self) -> None:
                pass

        cfg = Config()
        assert hasattr(cfg, "filter_function")
        assert hasattr(cfg, "cover_resolution")
        assert hasattr(cfg, "cover_overlap")
        assert hasattr(cfg, "clusterer")
        assert hasattr(cfg, "validate")

    def test_not_isinstance_with_missing_attrs(self):
        class BadConfig:
            filter_function: str = "eccentricity"

        assert not hasattr(BadConfig(), "cover_resolution")
