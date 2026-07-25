"""Tests for pynerve/torch/_persistence_helpers.py — sorted edges, distance matrix diagrams, backend assembly."""

from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch")
from torch import Tensor

from pynerve.torch._persistence_helpers import (
    _build_sorted_edges,
    _diagram_from_backend_parts,
    _diagram_from_distance_matrix_python,
)
from pynerve.torch._diagram import PersistenceDiagram


class TestBuildSortedEdges:
    def test_basic(self):
        dist = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        edges = _build_sorted_edges(dist, 3)
        assert len(edges) == 3
        assert edges[0][0] <= edges[-1][0]

    def test_inf_weights_filtered(self):
        dist = torch.tensor([[0.0, float("inf")], [float("inf"), 0.0]])
        edges = _build_sorted_edges(dist, 2)
        assert len(edges) == 0

    def test_all_finite(self):
        dist = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
        edges = _build_sorted_edges(dist, 2)
        assert len(edges) == 1


class TestDiagramFromDistanceMatrix:
    def test_basic(self):
        dist = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        result = _diagram_from_distance_matrix_python(dist, max_dim=0, single_input=False)
        assert isinstance(result, PersistenceDiagram)
        assert result.diagrams.shape[0] == 1  # batch_size

    def test_single_input(self):
        dist = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        result = _diagram_from_distance_matrix_python(dist, max_dim=0, single_input=True)
        assert isinstance(result, PersistenceDiagram)
        # single_input squeezes batch dim, PersistenceDiagram may wrap back
        assert result.diagrams.shape[-1] == 3


class TestDiagramFromBackendParts:
    def test_single_diagram(self):
        d = torch.zeros(5, 3)
        d[0, 0] = 0.0
        d[0, 1] = 1.0
        m = torch.zeros(5, dtype=torch.bool)
        m[0] = True
        n = torch.tensor([1, 0])
        result = _diagram_from_backend_parts([d], [m], [n], batched=False)
        assert isinstance(result, PersistenceDiagram)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="no diagrams"):
            _diagram_from_backend_parts([], [], [], batched=False)

    def test_batched_multiple(self):
        d1 = torch.zeros(5, 3)
        m1 = torch.zeros(5, dtype=torch.bool)
        n1 = torch.tensor([2])
        d1[0, :] = torch.tensor([0.0, 1.0, 0.0])
        m1[0] = True
        d1[1, :] = torch.tensor([0.0, 2.0, 0.0])
        m1[1] = True

        result = _diagram_from_backend_parts([d1, d1], [m1, m1], [n1, n1], batched=True)
        assert isinstance(result, PersistenceDiagram)
        assert result.diagrams.shape[0] == 2
