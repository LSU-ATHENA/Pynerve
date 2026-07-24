"""Tests for torch/_persistence_helpers.py — edge sorting and diagram assembly."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch._diagram import PersistenceDiagram
from pynerve.torch._persistence_helpers import (
    _build_sorted_edges,
    _diagram_from_backend_parts,
)


# _build_sorted_edges 


class TestBuildSortedEdges:
    def test_basic(self):
        dist = torch.tensor(
            [[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]], dtype=torch.float32
        )
        edges = _build_sorted_edges(dist, n_points=3)
        # Should return sorted edges (excluding self and infinite)
        assert len(edges) == 3  # 3 choose 2 = 3 non-self edges
        assert edges[0][0] < edges[1][0] < edges[2][0]

    def test_sorted_by_weight(self):
        dist = torch.tensor([[0.0, 10.0, 1.0], [10.0, 0.0, 5.0], [1.0, 5.0, 0.0]], dtype=torch.float32)
        edges = _build_sorted_edges(dist, n_points=3)
        assert edges[0][0] == 1.0
        assert edges[1][0] == 5.0
        assert edges[2][0] == 10.0

    def test_inf_skipped(self):
        dist = torch.tensor(
            [[0.0, float("inf"), 2.0], [float("inf"), 0.0, 3.0], [2.0, 3.0, 0.0]],
            dtype=torch.float32,
        )
        edges = _build_sorted_edges(dist, n_points=3)
        # Only 2 non-inf edges
        assert len(edges) == 2

    def test_single_point(self):
        dist = torch.tensor([[0.0]], dtype=torch.float32)
        edges = _build_sorted_edges(dist, n_points=1)
        assert len(edges) == 0

    def test_two_points(self):
        dist = torch.tensor([[0.0, 3.0], [3.0, 0.0]], dtype=torch.float32)
        edges = _build_sorted_edges(dist, n_points=2)
        assert len(edges) == 1
        assert edges[0][0] == 3.0

    def test_edge_tuple_format(self):
        dist = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float32)
        edges = _build_sorted_edges(dist, n_points=2)
        assert len(edges) == 1
        weight, i, j = edges[0]
        assert weight == 0.5
        assert i == 0
        assert j == 1


# _diagram_from_backend_parts 


class TestDiagramFromBackendParts:
    def test_single_diagram_not_batched(self):
        diagrams = torch.tensor([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=torch.float32)
        masks = torch.tensor([True, True], dtype=torch.bool)
        num_pairs = torch.tensor([2, 0], dtype=torch.long)

        result = _diagram_from_backend_parts(
            [diagrams], [masks], [num_pairs], batched=False
        )
        assert isinstance(result, PersistenceDiagram)
        assert result.batch_size > 0

    def test_multiple_diagrams_batched(self):
        d1 = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        d2 = torch.tensor([[2.0, 3.0, 0], [4.0, 5.0, 0]], dtype=torch.float32)
        m1 = torch.tensor([True], dtype=torch.bool)
        m2 = torch.tensor([True, True], dtype=torch.bool)
        n1 = torch.tensor([1], dtype=torch.long)
        n2 = torch.tensor([2], dtype=torch.long)

        result = _diagram_from_backend_parts(
            [d1, d2], [m1, m2], [n1, n2], batched=True
        )
        assert isinstance(result, PersistenceDiagram)
        # Should pad to max_pairs=2
        assert result.max_pairs == 2

    def test_single_batched_returns_direct(self):
        diagrams = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        masks = torch.tensor([True], dtype=torch.bool)
        num_pairs = torch.tensor([1], dtype=torch.long)

        result = _diagram_from_backend_parts(
            [diagrams], [masks], [num_pairs], batched=False
        )
        assert isinstance(result, PersistenceDiagram)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="no diagrams"):
            _diagram_from_backend_parts([], [], [], batched=True)

    def test_uneven_col_counts(self):
        """Test diagrams with different row counts are padded properly."""
        d1 = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        d2 = torch.tensor([[2.0, 3.0, 0], [4.0, 5.0, 0]], dtype=torch.float32)
        m1 = torch.tensor([True], dtype=torch.bool)
        m2 = torch.tensor([True, True], dtype=torch.bool)

        result = _diagram_from_backend_parts(
            [d1, d2], [m1, m2], [torch.tensor([1]), torch.tensor([2])], batched=True
        )
        assert result.max_pairs == 2

    def test_varying_num_pairs_shapes(self):
        d1 = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        d2 = torch.tensor([[2.0, 3.0, 0]], dtype=torch.float32)
        m1 = torch.tensor([True], dtype=torch.bool)
        m2 = torch.tensor([True], dtype=torch.bool)
        n1 = torch.tensor([1, 0, 0], dtype=torch.long)  # 3 dims
        n2 = torch.tensor([1], dtype=torch.long)  # 1 dim

        result = _diagram_from_backend_parts(
            [d1, d2], [m1, m2], [n1, n2], batched=True
        )
        assert isinstance(result, PersistenceDiagram)
