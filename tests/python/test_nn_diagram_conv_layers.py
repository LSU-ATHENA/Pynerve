"""Tests for nn/_diagram_conv_layers.py -- DiagramConv1D, DiagramDeepSet, validators."""

from __future__ import annotations


import pytest
import torch
import torch.nn as nn
from _test_helpers import make_diag_batched

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")

torch = pytest.importorskip("torch")


class TestValidateDiagram:
    def test_valid_2d(self):
        from pynerve.nn._diagram_conv_layers import _validate_diagram
        d = make_diag_batched(5).squeeze(0)
        _validate_diagram(d)  # should not raise

    def test_valid_3d(self):
        from pynerve.nn._diagram_conv_layers import _validate_diagram
        _validate_diagram(make_diag_batched(5, batch=2))

    def test_death_before_birth(self):
        from pynerve.nn._diagram_conv_layers import _validate_diagram
        d = torch.tensor([[[1.0, 0.5]]])
        with pytest.raises(ValueError, match="deaths"):
            _validate_diagram(d)

    def test_empty(self):
        from pynerve.nn._diagram_conv_layers import _validate_diagram
        _validate_diagram(torch.empty(1, 0, 2))  # should not raise


class TestValidateNonNegativeInt:
    def test_valid(self):
        from pynerve.nn._diagram_conv_layers import _validate_non_negative_int
        assert _validate_non_negative_int("x", 5) == 5
        assert _validate_non_negative_int("x", 0) == 0

    def test_negative(self):
        from pynerve.nn._diagram_conv_layers import _validate_non_negative_int
        with pytest.raises(ValueError, match="non-negative"):
            _validate_non_negative_int("x", -1)

    def test_bool(self):
        from pynerve.nn._diagram_conv_layers import _validate_non_negative_int
        with pytest.raises(ValueError, match="non-negative"):
            _validate_non_negative_int("x", True)


class TestDiagramConv1D:
    def test_construct(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        layer = DiagramConv1D(in_channels=0, out_channels=8, kernel_size=3)
        assert layer.in_channels == 0
        assert layer.out_channels == 8

    def test_construct_with_weighting(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        layer = DiagramConv1D(in_channels=4, out_channels=16, use_persistence_weighting=True)
        assert layer.use_weighting is True

    def test_construct_without_weighting(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        layer = DiagramConv1D(in_channels=0, out_channels=8, use_persistence_weighting=False)
        assert layer.use_weighting is False

    def test_construct_invalid_out_channels(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        with pytest.raises(ValueError, match="positive"):
            DiagramConv1D(in_channels=0, out_channels=0)

    def test_construct_invalid_kernel(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        with pytest.raises(ValueError, match="positive"):
            DiagramConv1D(in_channels=0, out_channels=8, kernel_size=0)

    def test_forward_no_features(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        layer = DiagramConv1D(in_channels=0, out_channels=8, kernel_size=3)
        d = make_diag_batched(10, batch=2)
        result = layer(d)
        assert result.dim() == 3
        assert result.shape[0] == 2
        assert result.shape[1] == 8

    def test_forward_with_features(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        layer = DiagramConv1D(in_channels=3, out_channels=8, kernel_size=3)
        d = make_diag_batched(10, batch=2)
        feats = torch.rand(2, 10, 3)
        result = layer(d, features=feats)
        assert result.dim() == 3
        assert result.shape[0] == 2

    def test_forward_features_required(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        layer = DiagramConv1D(in_channels=3, out_channels=8, kernel_size=3)
        d = make_diag_batched(10, batch=2)
        with pytest.raises(ValueError, match="features are required"):
            layer(d)

    def test_forward_features_wrong_shape(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        layer = DiagramConv1D(in_channels=3, out_channels=8, kernel_size=3)
        d = make_diag_batched(10, batch=2)
        with pytest.raises(ValueError, match="features"):
            layer(d, features=torch.rand(2, 5, 3))

    def test_forward_empty_diagram(self):
        from pynerve.nn._diagram_conv_layers import DiagramConv1D
        layer = DiagramConv1D(in_channels=0, out_channels=8, kernel_size=3)
        result = layer(torch.empty(2, 0, 2))
        assert result.shape == (2, 8, 0)


class TestDiagramDeepSet:
    def test_construct(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=0, hidden_channels=[16, 8], out_channels=4)
        assert ds.pooling == "persistence_weighted"

    def test_construct_sum(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=0, hidden_channels=[16], out_channels=4, pooling="sum")
        assert ds.pooling == "sum"

    def test_construct_mean(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=0, hidden_channels=[16], out_channels=4, pooling="mean")

    def test_construct_max(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=0, hidden_channels=[16], out_channels=4, pooling="max")

    def test_construct_invalid_pooling(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        with pytest.raises(ValueError, match="pooling"):
            DiagramDeepSet(in_channels=0, hidden_channels=[16], out_channels=4, pooling="bad")

    def test_construct_empty_hidden(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        with pytest.raises(ValueError, match="non-empty"):
            DiagramDeepSet(in_channels=0, hidden_channels=[], out_channels=4)

    def test_forward_no_features(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=0, hidden_channels=[16], out_channels=4, pooling="sum")
        d = make_diag_batched(10, batch=2)
        result = ds(d)
        assert result.shape == (2, 4)

    def test_forward_with_features(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=3, hidden_channels=[16], out_channels=4, pooling="mean")
        d = make_diag_batched(10, batch=2)
        feats = torch.rand(2, 10, 3)
        result = ds(d, features=feats)
        assert result.shape == (2, 4)

    def test_forward_features_required(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=3, hidden_channels=[16], out_channels=4)
        d = make_diag_batched(10, batch=2)
        with pytest.raises(ValueError, match="features are required"):
            ds(d)

    def test_forward_empty(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=0, hidden_channels=[16], out_channels=4, pooling="sum")
        result = ds(torch.empty(2, 0, 2))
        assert result.shape == (2, 4)

    def test_forward_persistence_weighted(self):
        from pynerve.nn._diagram_conv_layers import DiagramDeepSet
        ds = DiagramDeepSet(in_channels=0, hidden_channels=[16], out_channels=4, pooling="persistence_weighted")
        d = make_diag_batched(10, batch=2)
        result = ds(d)
        assert result.shape == (2, 4)
