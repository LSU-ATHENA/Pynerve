"""Tests for pynerve/_streaming_persistence.py — streaming validation, compute kwargs, format results."""

from __future__ import annotations

import numpy as np
import pytest

from pynerve._streaming_persistence import (
    StreamingPersistence,
    _validate_streaming_array,
)
from pynerve._fallback_classes import PersistenceBackend
from pynerve.exceptions import InvalidArgumentError


class TestValidateStreamingArray:
    def test_valid_2d(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_streaming_array(arr, "test")
        assert result.shape == (2, 2)

    def test_non_2d_raises(self):
        with pytest.raises(InvalidArgumentError, match="2D"):
            _validate_streaming_array(np.array([1.0, 2.0]), "test")

    def test_empty_raises(self):
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            _validate_streaming_array(np.array([[]]), "test")

    def test_non_numeric_raises(self):
        with pytest.raises(InvalidArgumentError, match="numeric"):
            _validate_streaming_array(np.array([["a", "b"]]), "test")

    def test_non_finite_raises(self):
        with pytest.raises(InvalidArgumentError, match="finite"):
            _validate_streaming_array(np.array([[np.nan, 1.0]]), "test")


class TestStreamingPersistenceInit:
    def test_default_construction(self):
        sp = StreamingPersistence()
        assert sp.chunk_size == 1000
        assert sp.max_buffered == 3
        assert sp.use_gpu is True

    def test_custom_params(self):
        sp = StreamingPersistence(chunk_size=500, max_buffered_chunks=5, use_gpu=False)
        assert sp.chunk_size == 500
        assert sp.max_buffered == 5
        assert sp.use_gpu is False

    def test_invalid_chunk_size(self):
        with pytest.raises(Exception, match="positive"):
            StreamingPersistence(chunk_size=0)

    def test_invalid_max_buffered(self):
        with pytest.raises(Exception, match="positive"):
            StreamingPersistence(max_buffered_chunks=0)

    def test_passes_extra_kwargs(self):
        sp = StreamingPersistence(max_dim=2, metric="euclidean")
        assert sp.persistence_kwargs["max_dim"] == 2
        assert sp.persistence_kwargs["metric"] == "euclidean"


class TestComputeKwargs:
    def test_with_gpu(self):
        sp = StreamingPersistence(use_gpu=True)
        kwargs = sp._compute_kwargs({})
        assert "backend" not in kwargs

    def test_without_gpu_adds_cpu_backend(self):
        sp = StreamingPersistence(use_gpu=False)
        kwargs = sp._compute_kwargs({})
        assert kwargs["backend"] == PersistenceBackend.CPU_EXACT

    def test_override_kwargs(self):
        sp = StreamingPersistence(use_gpu=True, max_dim=2)
        kwargs = sp._compute_kwargs({"max_dim": 3})
        assert kwargs["max_dim"] == 3


class TestFormatResult:
    def test_format_diagrams(self):
        sp = StreamingPersistence()
        result = sp._format_result({"data": "test"}, "diagrams")
        assert result == {"data": "test"}

    def test_format_betti(self):
        sp = StreamingPersistence()
        # betti format calls _diagrams_to_betti which requires pairs
        result = sp._format_result({"betti_numbers": [1, 0]}, "betti")
        assert "betti_0" in result

    def test_format_stats(self):
        sp = StreamingPersistence()
        pairs = np.array([[0.0, 1.0, 0], [2.0, 5.0, 0]])
        result = sp._format_result(pairs, "stats")
        assert "num_features" in result
        assert result["num_features"] == 2


class TestPairArray:
    def test_dict_with_pairs(self):
        sp = StreamingPersistence()
        result = sp._pair_array({"pairs": np.array([[0.0, 1.0, 0]])})
        assert result.shape == (1, 3)

    def test_raw_array(self):
        sp = StreamingPersistence()
        result = sp._pair_array(np.array([[0.0, 1.0, 0]]))
        assert result.shape == (1, 3)

    def test_dict_missing_pairs_raises(self):
        sp = StreamingPersistence()
        with pytest.raises(InvalidArgumentError, match="pairs"):
            sp._pair_array({"other": "data"})

    def test_empty_pairs(self):
        sp = StreamingPersistence()
        result = sp._pair_array(np.empty((0, 3)))
        assert result.shape == (0, 3)


class TestDiagramsToBetti:
    def test_dict_with_betti_numbers(self):
        sp = StreamingPersistence()
        result = sp._diagrams_to_betti({"betti_numbers": [2, 1, 0]})
        assert result == {"betti_0": 2, "betti_1": 1, "betti_2": 0}

    def test_from_pairs(self):
        sp = StreamingPersistence()
        pairs = np.array([[0.0, float("inf"), 0], [0.0, float("inf"), 0], [1.0, 2.0, 0]])
        result = sp._diagrams_to_betti(pairs)
        assert result["betti_0"] >= 2


class TestDiagramsToStats:
    def test_basic(self):
        sp = StreamingPersistence()
        pairs = np.array([[0.0, 1.0, 0], [2.0, 5.0, 0]])
        result = sp._diagrams_to_stats(pairs)
        assert "num_features" in result
        assert result["num_features"] == 2
        assert "avg_persistence" in result
        assert "max_persistence" in result
