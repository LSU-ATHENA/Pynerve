"""Tests for torch/_mapper_impl.py — pure-Python Mapper filters, clustering, cover, edges."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch._mapper_impl import (
    _CLUSTERERS,
    _FILTER_FUNCTIONS,
    _build_1d_cover,
    _build_edges_from_nodes,
    _create_grid_cover,
    _dbscan_python,
    _filter_eccentricity_python,
    _filter_pca_python,
    _mapper_from_filter_values,
    _single_linkage_python,
    _validate_filter_vals,
    _validate_mapper_inputs,
)


# Constants 


class TestConstants:
    def test_filter_functions_set(self):
        assert "pca_2d" in _FILTER_FUNCTIONS
        assert "eccentricity" in _FILTER_FUNCTIONS

    def test_clusterers_set(self):
        assert "dbscan" in _CLUSTERERS
        assert "single_linkage" in _CLUSTERERS
        assert "connected" in _CLUSTERERS


# _validate_mapper_inputs 


class TestValidateMapperInputs:
    def test_valid(self):
        pc = torch.randn(10, 3, dtype=torch.float32)
        r = _validate_mapper_inputs(pc, 10, 0.25, 0.5, 5)
        assert r == (10, 0.25, 0.5, 5)

    def test_non_2d_raises(self):
        pc = torch.randn(10, dtype=torch.float32)
        with pytest.raises(ValueError, match="2D"):
            _validate_mapper_inputs(pc, 10, 0.25, 0.5, 5)

    def test_empty_raises(self):
        pc = torch.empty((0, 3), dtype=torch.float32)
        with pytest.raises(ValueError, match="non-empty"):
            _validate_mapper_inputs(pc, 10, 0.25, 0.5, 5)

    def test_nan_raises(self):
        pc = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite"):
            _validate_mapper_inputs(pc, 10, 0.25, 0.5, 5)

    def test_cover_resolution_zero_raises(self):
        pc = torch.randn(5, 2, dtype=torch.float32)
        with pytest.raises(ValueError, match="positive"):
            _validate_mapper_inputs(pc, 0, 0.25, 0.5, 5)

    def test_cover_overlap_negative_raises(self):
        pc = torch.randn(5, 2, dtype=torch.float32)
        with pytest.raises(ValueError, match="cover_overlap"):
            _validate_mapper_inputs(pc, 10, -0.1, 0.5, 5)

    def test_cover_overlap_one_raises(self):
        pc = torch.randn(5, 2, dtype=torch.float32)
        with pytest.raises(ValueError, match="cover_overlap"):
            _validate_mapper_inputs(pc, 10, 1.0, 0.5, 5)

    def test_dbscan_eps_zero_raises(self):
        pc = torch.randn(5, 2, dtype=torch.float32)
        with pytest.raises(ValueError, match="dbscan_eps"):
            _validate_mapper_inputs(pc, 10, 0.25, 0.0, 5)

    def test_dbscan_min_samples_zero_raises(self):
        pc = torch.randn(5, 2, dtype=torch.float32)
        with pytest.raises(ValueError, match="positive"):
            _validate_mapper_inputs(pc, 10, 0.25, 0.5, 0)


# _filter_pca_python 


class TestFilterPcaPython:
    def test_2d_output(self):
        pc = torch.randn(20, 5, dtype=torch.float32)
        result = _filter_pca_python(pc, 2)
        assert result.shape == (20, 2)

    def test_1d_output(self):
        pc = torch.randn(20, 5, dtype=torch.float32)
        result = _filter_pca_python(pc, 1)
        assert result.shape == (20, 1)

    def test_single_point(self):
        pc = torch.randn(1, 3, dtype=torch.float32)
        result = _filter_pca_python(pc, 2)
        assert result.shape == (1, min(2, 3))

    def test_nan_input_raises(self):
        pc = torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite"):
            _filter_pca_python(pc, 1)


# _filter_eccentricity_python 


class TestFilterEccentricityPython:
    def test_basic(self):
        pc = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        result = _filter_eccentricity_python(pc)
        assert result.shape == (3,)

    def test_single_point(self):
        pc = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
        result = _filter_eccentricity_python(pc)
        assert result.item() == 0.0


_build_1d_cover 


class TestBuild1dCover:
    def test_basic(self):
        fv = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]], dtype=torch.float32)
        cover = _build_1d_cover(fv, cover_resolution=2, cover_overlap=0.0)
        assert len(cover) >= 1

    def test_overlap_creates_multiple_hits(self):
        fv = torch.tensor([[0.5]], dtype=torch.float32)
        cover = _build_1d_cover(fv, cover_resolution=3, cover_overlap=0.5)
        assert len(cover) > 1

    def test_cover_has_expected_keys(self):
        fv = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        cover = _build_1d_cover(fv, cover_resolution=2, cover_overlap=0.1)
        if len(cover) > 0:
            assert "indices" in cover[0]
            assert "range" in cover[0]
            assert "center" in cover[0]


# _dbscan_python 


class TestDbscanPython:
    def test_single_cluster(self):
        points = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0]], dtype=torch.float32)
        labels = _dbscan_python(points, eps=1.0, min_samples=2)
        unique = torch.unique(labels)
        # All points should be in same cluster
        assert 0 in unique

    def test_noise_points(self):
        points = torch.tensor([[0.0, 0.0], [5.0, 5.0], [5.1, 5.1]], dtype=torch.float32)
        labels = _dbscan_python(points, eps=0.5, min_samples=2)
        # First point is noise (-1), others clustered
        assert (labels == -1).any()

    def test_empty(self):
        points = torch.empty((0, 2), dtype=torch.float32)
        labels = _dbscan_python(points, eps=1.0, min_samples=2)
        assert labels.numel() == 0


# _single_linkage_python 


class TestSingleLinkagePython:
    def test_basic(self):
        points = torch.tensor([[0.0, 0.0], [0.5, 0.0], [5.0, 5.0]], dtype=torch.float32)
        labels = _single_linkage_python(points, threshold=1.0)
        # First two should be in the same cluster
        assert labels[0] == labels[1]

    def test_all_separate(self):
        points = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32)
        labels = _single_linkage_python(points, threshold=1.0)
        assert labels[0] != labels[1]

    def test_empty(self):
        points = torch.empty((0, 2), dtype=torch.float32)
        labels = _single_linkage_python(points, threshold=1.0)
        assert labels.numel() == 0


# _build_edges_from_nodes 


class TestBuildEdgesFromNodes:
    def test_overlapping_nodes_create_edge(self):
        nodes = [
            {"id": 0, "point_indices": [0, 1], "centroid": None, "filter_centroid": None, "cover_index": 0},
            {"id": 1, "point_indices": [1, 2], "centroid": None, "filter_centroid": None, "cover_index": 1},
        ]
        edges = _build_edges_from_nodes(nodes)
        assert len(edges) == 1
        assert edges[0]["source"] == 0
        assert edges[0]["target"] == 1

    def test_disjoint_nodes_no_edge(self):
        nodes = [
            {"id": 0, "point_indices": [0, 1], "centroid": None, "filter_centroid": None, "cover_index": 0},
            {"id": 1, "point_indices": [2, 3], "centroid": None, "filter_centroid": None, "cover_index": 1},
        ]
        edges = _build_edges_from_nodes(nodes)
        assert len(edges) == 0

    def test_edge_weight(self):
        nodes = [
            {"id": 0, "point_indices": [0, 1], "centroid": None, "filter_centroid": None, "cover_index": 0},
            {"id": 1, "point_indices": [0, 1], "centroid": None, "filter_centroid": None, "cover_index": 1},
        ]
        edges = _build_edges_from_nodes(nodes)
        assert edges[0]["weight"] == 1.0


# _create_grid_cover 


class TestCreateGridCover:
    def test_1d_grid(self):
        fv = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        cover = _create_grid_cover(fv[:, :1], resolution=2, overlap=0.1)
        assert len(cover) >= 1

    def test_2d_grid(self):
        fv = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
        cover = _create_grid_cover(fv, resolution=2, overlap=0.25)
        assert len(cover) >= 1

    def test_empty_dim_raises(self):
        fv = torch.empty((5, 0), dtype=torch.float32)
        with pytest.raises(ValueError, match="dimension"):
            _create_grid_cover(fv, resolution=2, overlap=0.1)


# _validate_filter_vals 


class TestValidateFilterVals:
    def test_valid_1d(self):
        fv = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        pc = torch.randn(3, 2, dtype=torch.float32)
        result = _validate_filter_vals(fv, pc)
        assert result.shape == (3, 1)

    def test_row_mismatch_raises(self):
        fv = torch.tensor([0.0, 1.0], dtype=torch.float32)
        pc = torch.randn(3, 2, dtype=torch.float32)
        with pytest.raises(ValueError, match="one row per point"):
            _validate_filter_vals(fv, pc)

    def test_nan_raises(self):
        fv = torch.tensor([[float("nan")]], dtype=torch.float32)
        pc = torch.randn(1, 2, dtype=torch.float32)
        with pytest.raises(ValueError, match="finite"):
            _validate_filter_vals(fv, pc)


# _mapper_from_filter_values 


class TestMapperFromFilterValues:
    def test_basic(self):
        pc = torch.randn(10, 2, dtype=torch.float32)
        fv = torch.randn(10, 1, dtype=torch.float32)
        result = _mapper_from_filter_values(pc, fv, 3, 0.5, "dbscan", 1.0, 2, True)
        assert "nodes" in result
        assert "edges" in result
        assert "filter_values" in result

    def test_with_graph(self):
        pc = torch.randn(10, 2, dtype=torch.float32)
        fv = torch.randn(10, 1, dtype=torch.float32)
        result = _mapper_from_filter_values(pc, fv, 3, 0.5, "dbscan", 1.0, 2, True)
        # graph is included only if networkx is installed
        from importlib.util import find_spec
        if find_spec("networkx"):
            assert "graph" in result
        else:
            assert "graph" not in result

    def test_connected_clusterer(self):
        pc = torch.randn(10, 2, dtype=torch.float32)
        fv = torch.randn(10, 1, dtype=torch.float32)
        result = _mapper_from_filter_values(pc, fv, 3, 0.5, "connected", 1.0, 2, False)
        assert "nodes" in result
