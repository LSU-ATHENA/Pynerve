"""Tests for merge.py — persistence diagram merging and matching."""

from __future__ import annotations

import numpy as np
import pytest
from pynerve.merge import match_persistence_diagrams, _bottleneck_match, _filter_by_dim


class TestFilterByDim:
    def test_filters_by_dimension_column(self):
        diagram = np.array([
            [0.0, 1.0, 0],
            [1.0, 2.0, 1],
            [2.0, 3.0, 0],
            [3.0, 4.0, 1],
        ], dtype=np.float64)
        result = _filter_by_dim(diagram, 0)
        assert result.shape[0] == 2
        assert np.all(result[:, 2] == 0)

    def test_filters_dim_1(self):
        diagram = np.array([
            [0.0, 1.0, 0],
            [1.0, 2.0, 1],
            [2.0, 3.0, 2],
        ], dtype=np.float64)
        result = _filter_by_dim(diagram, 1)
        assert result.shape[0] == 1
        assert result[0, 2] == 1

    def test_no_dim_column_returns_original(self):
        diagram = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        result = _filter_by_dim(diagram, 0)
        np.testing.assert_array_equal(result, diagram)

    def test_no_matching_dim_returns_empty(self):
        diagram = np.array([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=np.float64)
        result = _filter_by_dim(diagram, 1)
        assert result.shape[0] == 0

    def test_dim_as_int_column(self):
        diagram = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float64)
        result = _filter_by_dim(diagram, 1)
        assert result.shape[0] == 1


class TestBottleneckMatch:
    def test_empty_ref_returns_target(self):
        ref = np.empty((0, 3), dtype=np.float64)
        target = np.array([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=np.float64)
        result = _bottleneck_match(ref, target, float("inf"))
        np.testing.assert_array_equal(result, target[:, :result.shape[1]])

    def test_empty_target_returns_ref(self):
        ref = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        target = np.empty((0, 3), dtype=np.float64)
        result = _bottleneck_match(ref, target, float("inf"))
        np.testing.assert_array_equal(result, ref[:, :result.shape[1]])

    def test_exact_match_single_pair(self):
        ref = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        target = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        result = _bottleneck_match(ref, target, float("inf"))
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result[0], [0.0, 1.0, 0.0])

    def test_match_within_threshold(self):
        ref = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        target = np.array([[0.1, 0.9, 0]], dtype=np.float64)
        result = _bottleneck_match(ref, target, 0.5)
        assert len(result) == 1

    def test_no_match_beyond_threshold(self):
        ref = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        target = np.array([[5.0, 6.0, 0]], dtype=np.float64)
        result = _bottleneck_match(ref, target, 1.0)
        assert len(result) == 0

    def test_multiple_pairs_best_match(self):
        ref = np.array([[0.0, 1.0, 0], [2.0, 3.0, 0]], dtype=np.float64)
        target = np.array([[0.1, 1.1, 0], [2.1, 2.9, 0]], dtype=np.float64)
        result = _bottleneck_match(ref, target, float("inf"))
        assert len(result) == 2

    def test_inf_deaths_filtered_out(self):
        ref = np.array([[0.0, float("inf"), 0]], dtype=np.float64)
        target = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        result = _bottleneck_match(ref, target, float("inf"))
        assert len(result) == 1  # returns target's finite pair

    def test_death_less_than_birth_filtered_out(self):
        ref = np.array([[2.0, 1.0, 0]], dtype=np.float64)
        target = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        result = _bottleneck_match(ref, target, float("inf"))
        assert len(result) == 1  # returns target's valid pair

    def test_zero_threshold_no_match(self):
        ref = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        target = np.array([[0.0, 1.1, 0]], dtype=np.float64)
        result = _bottleneck_match(ref, target, 0.0)
        assert len(result) == 0


class TestMatchPersistenceDiagrams:
    def test_single_diagram_returns_itself(self):
        d = np.array([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=np.float64)
        result = match_persistence_diagrams([d])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], d)

    def test_empty_list(self):
        result = match_persistence_diagrams([])
        assert result == []

    def test_two_identical_diagrams(self):
        d = np.array([[0.0, 1.0, 0], [1.0, 2.0, 1]], dtype=np.float64)
        result = match_persistence_diagrams([d, d.copy()])
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], d)
        assert len(result[1]) == 2

    def test_two_diagrams_matching(self):
        d1 = np.array([[0.0, 1.0, 0], [2.0, 3.0, 0]], dtype=np.float64)
        d2 = np.array([[0.1, 0.9, 0], [2.1, 3.1, 0]], dtype=np.float64)
        result = match_persistence_diagrams([d1, d2], threshold=1.0)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) >= 1

    def test_three_diagrams(self):
        d1 = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        d2 = np.array([[0.1, 1.1, 0]], dtype=np.float64)
        d3 = np.array([[0.2, 0.8, 0]], dtype=np.float64)
        result = match_persistence_diagrams([d1, d2, d3], threshold=2.0)
        assert len(result) == 3

    def test_dim_filter(self):
        d1 = np.array([[0.0, 1.0, 0], [1.0, 2.0, 1]], dtype=np.float64)
        d2 = np.array([[0.1, 0.9, 0], [1.1, 2.1, 1]], dtype=np.float64)
        result = match_persistence_diagrams([d1, d2], dim=0)
        assert len(result) == 2
        assert result[0].shape[0] == 1

    def test_dim_filter_no_match(self):
        d1 = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        d2 = np.array([[0.1, 0.9, 0]], dtype=np.float64)
        result = match_persistence_diagrams([d1, d2], dim=99)
        assert len(result) == 2
        assert result[0].shape[0] == 0

    def test_infinite_threshold_default(self):
        d1 = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        d2 = np.array([[100.0, 101.0, 0]], dtype=np.float64)
        result = match_persistence_diagrams([d1, d2])
        assert len(result) == 2

    def test_strict_threshold_excludes(self):
        d1 = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        d2 = np.array([[100.0, 101.0, 0]], dtype=np.float64)
        result = match_persistence_diagrams([d1, d2], threshold=1.0)
        assert len(result) == 2
        assert len(result[1]) == 0

    def test_diagrams_with_2_columns(self):
        d1 = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        d2 = np.array([[0.1, 0.9], [2.1, 3.1]], dtype=np.float64)
        result = match_persistence_diagrams([d1, d2], threshold=1.0)
        assert len(result) == 2

    def test_diagrams_with_different_sizes(self):
        d1 = np.array([[0.0, 1.0, 0], [2.0, 3.0, 0], [4.0, 5.0, 0]], dtype=np.float64)
        d2 = np.array([[0.1, 0.9, 0]], dtype=np.float64)
        result = match_persistence_diagrams([d1, d2], threshold=2.0)
        assert len(result) == 2
        assert len(result[0]) == 3
