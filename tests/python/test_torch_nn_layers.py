"""Tests for torch/nn_layers_impl.py -- pure-CPU nn.Module classes.

Tests modules that don't require C++ extensions: DiagramPooling,
PersistenceReadout, and their validation helpers.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pynerve.torch.nn_layers_impl import (
    DiagramPooling,
    PersistenceReadout,
    _validate_positive_dims,
    _validate_probability,
)


# _validate_probability 


class TestValidateProbability:
    def test_valid(self):
        assert _validate_probability("dropout", 0.0) == 0.0
        assert _validate_probability("dropout", 0.5) == 0.5
        assert _validate_probability("dropout", 0.999) == 0.999

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            _validate_probability("dropout", -0.1)

    def test_one_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            _validate_probability("dropout", 1.0)

    def test_over_one_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            _validate_probability("dropout", 1.5)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            _validate_probability("dropout", float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            _validate_probability("dropout", float("inf"))


# _validate_positive_dims 


class TestValidatePositiveDims:
    def test_valid(self):
        result = _validate_positive_dims("hidden_dims", (128, 64))
        assert result == (128, 64)

    def test_single_dim(self):
        result = _validate_positive_dims("hidden_dims", (256,))
        assert result == (256,)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="positive dimensions"):
            _validate_positive_dims("hidden_dims", ())

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_positive_dims("hidden_dims", (0,))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_positive_dims("hidden_dims", (-1, 128))


# DiagramPooling 


class TestDiagramPooling:
    def test_construction_mean(self):
        pool = DiagramPooling(method="mean")
        assert isinstance(pool, nn.Module)

    def test_construction_max(self):
        pool = DiagramPooling(method="max")
        assert pool.method == "max"

    def test_construction_sum(self):
        pool = DiagramPooling(method="sum")
        assert pool.method == "sum"

    def test_construction_attention(self):
        pool = DiagramPooling(method="attention")
        assert pool.method == "attention"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            DiagramPooling(method="bad")  # type: ignore[arg-type]

    def test_custom_dim(self):
        pool = DiagramPooling(method="mean", dim=0)
        assert pool.dim == 0

    def test_forward_mean(self):
        pool = DiagramPooling(method="mean", dim=0)
        x = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=torch.float32)
        result = pool(x)
        assert result.shape == (2, 2)
        expected = x.mean(dim=0)
        assert torch.allclose(result, expected)

    def test_forward_max(self):
        pool = DiagramPooling(method="max", dim=0)
        x = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=torch.float32)
        result = pool(x)
        assert result.shape == (2, 2)
        assert torch.allclose(result, x.max(dim=0).values)

    def test_forward_sum(self):
        pool = DiagramPooling(method="sum", dim=0)
        x = torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]], dtype=torch.float32)
        result = pool(x)
        assert result.shape == (1, 2)
        assert torch.allclose(result, x.sum(dim=0))

    def test_forward_attention(self):
        pool = DiagramPooling(method="attention", dim=0)
        x = torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]], dtype=torch.float32)
        result = pool(x)
        assert result.shape == (1, 2)

    def test_forward_mean_dim1(self):
        pool = DiagramPooling(method="mean", dim=1)
        x = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32)
        result = pool(x)
        assert result.shape == (1, 2)
        assert torch.allclose(result[0], torch.tensor([1.0, 1.0]))

    def test_empty_raises(self):
        pool = DiagramPooling(method="mean")
        x = torch.empty((0, 2, 3), dtype=torch.float32)
        with pytest.raises(ValueError, match="non-empty"):
            pool(x)


# PersistenceReadout 


class TestPersistenceReadout:
    def test_construction_defaults(self):
        readout = PersistenceReadout(in_features=10, out_features=2)
        assert isinstance(readout, nn.Module)

    def test_construction_custom_hidden(self):
        readout = PersistenceReadout(in_features=10, out_features=2, hidden_dims=(32, 16))
        assert isinstance(readout.mlp, nn.Sequential)

    def test_construction_with_dropout(self):
        readout = PersistenceReadout(in_features=10, out_features=2, hidden_dims=(64,), dropout=0.3)
        assert isinstance(readout, nn.Module)

    def test_construction_activation_relu(self):
        """Test default relu activation explicitly."""
        readout = PersistenceReadout(in_features=10, out_features=2, activation="relu")
        assert isinstance(readout, nn.Module)

    def test_construction_activation_gelu(self):
        readout = PersistenceReadout(in_features=10, out_features=2, activation="gelu")
        assert isinstance(readout, nn.Module)

    def test_construction_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            PersistenceReadout(in_features=10, out_features=2, activation="sigmoid")

    def test_construction_negative_in_features_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PersistenceReadout(in_features=0, out_features=2)

    def test_construction_negative_out_features_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PersistenceReadout(in_features=10, out_features=0)

    def test_construction_zero_hidden_dims_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PersistenceReadout(in_features=10, out_features=2, hidden_dims=(0,))

    def test_construction_dropout_one_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            PersistenceReadout(in_features=10, out_features=2, dropout=1.0)

    def test_forward_basic(self):
        readout = PersistenceReadout(in_features=4, out_features=2, hidden_dims=(8,))
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        result = readout(x)
        assert result.shape == (1, 2)

    def test_forward_with_hidden(self):
        readout = PersistenceReadout(in_features=4, out_features=2, hidden_dims=(8,))
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        result = readout(x)
        assert result.shape == (1, 2)

    def test_forward_batched(self):
        readout = PersistenceReadout(in_features=4, out_features=3, hidden_dims=(8,))
        x = torch.randn(16, 4, dtype=torch.float32)
        result = readout(x)
        assert result.shape == (16, 3)

    def test_forward_no_dropout(self):
        readout = PersistenceReadout(in_features=4, out_features=2, dropout=0.0)
        x = torch.randn(8, 4, dtype=torch.float32)
        result = readout(x)
        assert result.shape == (8, 2)
