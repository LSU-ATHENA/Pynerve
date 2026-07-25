"""Tests for nn/_diagram_attention.py — DiagramMultiHeadAttention, DiagramTransformerBlock."""

from __future__ import annotations


import pytest
import torch
from _test_helpers import make_diag_batched

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")

torch = pytest.importorskip("torch")


class TestValidateProbability:
    def test_valid(self):
        from pynerve.nn._diagram_attention import _validate_probability
        assert _validate_probability("x", 0.0) == 0.0
        assert _validate_probability("x", 0.5) == 0.5

    def test_one(self):
        from pynerve.nn._diagram_attention import _validate_probability
        with pytest.raises(ValueError, match="0 <="):
            _validate_probability("x", 1.0)

    def test_negative(self):
        from pynerve.nn._diagram_attention import _validate_probability
        with pytest.raises(ValueError, match="0 <="):
            _validate_probability("x", -0.1)

    def test_nan(self):
        from pynerve.nn._diagram_attention import _validate_probability
        with pytest.raises(ValueError, match="0 <="):
            _validate_probability("x", float("nan"))


class TestValidateDiagram:
    def test_valid_2d(self):
        from pynerve.nn._diagram_attention import _validate_diagram
        _validate_diagram(make_diag_batched(5).squeeze(0))

    def test_valid_3d(self):
        from pynerve.nn._diagram_attention import _validate_diagram
        _validate_diagram(make_diag_batched(5, batch=2))

    def test_death_before_birth(self):
        from pynerve.nn._diagram_attention import _validate_diagram
        with pytest.raises(ValueError, match="deaths"):
            _validate_diagram(torch.tensor([[[1.0, 0.5]]]))

    def test_empty(self):
        from pynerve.nn._diagram_attention import _validate_diagram
        _validate_diagram(torch.empty(1, 0, 2))


class TestDiagramMultiHeadAttention:
    def test_construct(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        assert mha.d_model == 16
        assert mha.num_heads == 4
        assert mha.head_dim == 4

    def test_construct_with_positional(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, use_birth_death_positional=True)
        assert mha.use_positional is True

    def test_construct_without_positional(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, use_birth_death_positional=False)
        assert mha.use_positional is False

    def test_construct_not_divisible(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        with pytest.raises(ValueError, match="divisible"):
            DiagramMultiHeadAttention(d_model=10, num_heads=4)

    def test_construct_invalid_d_model(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        with pytest.raises(ValueError, match="positive"):
            DiagramMultiHeadAttention(d_model=0, num_heads=4)

    def test_construct_invalid_heads(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        with pytest.raises(ValueError, match="positive"):
            DiagramMultiHeadAttention(d_model=16, num_heads=0)

    def test_construct_invalid_dropout(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        with pytest.raises(ValueError, match="0 <="):
            DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=1.0)

    def test_forward(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        d = make_diag_batched(5, batch=2)
        feats = torch.rand(2, 5, 16)
        result = mha(d, feats)
        assert result.shape == (2, 5, 16)

    def test_forward_without_positional(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0, use_birth_death_positional=False)
        d = make_diag_batched(5, batch=2)
        feats = torch.rand(2, 5, 16)
        result = mha(d, feats)
        assert result.shape == (2, 5, 16)

    def test_forward_empty(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        result = mha(torch.empty(2, 0, 2), torch.rand(2, 0, 16))
        assert result.shape == (2, 0, 16)

    def test_forward_wrong_features_shape(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        d = make_diag_batched(5, batch=2)
        with pytest.raises(ValueError, match="features"):
            mha(d, torch.rand(2, 5, 8))

    def test_forward_same_device(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        d = make_diag_batched(5, batch=2)
        feats = torch.rand(2, 5, 16)
        # Both on CPU — should work, not raise
        result = mha(d, feats)
        assert result.shape == (2, 5, 16)

    def test_forward_with_2d_mask(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        d = make_diag_batched(5, batch=2)
        feats = torch.rand(2, 5, 16)
        mask = torch.ones(2, 5)
        result = mha(d, feats, mask=mask)
        assert result.shape == (2, 5, 16)

    def test_forward_with_3d_mask(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        d = make_diag_batched(5, batch=2)
        feats = torch.rand(2, 5, 16)
        mask = torch.ones(2, 5, 5)
        result = mha(d, feats, mask=mask)
        assert result.shape == (2, 5, 16)

    def test_forward_with_bad_mask(self):
        from pynerve.nn._diagram_attention import DiagramMultiHeadAttention
        mha = DiagramMultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        d = make_diag_batched(5, batch=2)
        feats = torch.rand(2, 5, 16)
        # All-zero mask — no attention target per row
        mask = torch.zeros(2, 5)
        with pytest.raises(ValueError, match="at least one"):
            mha(d, feats, mask=mask)


class TestDiagramTransformerBlock:
    def test_construct(self):
        from pynerve.nn._diagram_attention import DiagramTransformerBlock
        block = DiagramTransformerBlock(d_model=16, num_heads=4, d_ff=32, dropout=0.0)
        assert block is not None

    def test_construct_invalid_d_ff(self):
        from pynerve.nn._diagram_attention import DiagramTransformerBlock
        with pytest.raises(ValueError, match="positive"):
            DiagramTransformerBlock(d_model=16, num_heads=4, d_ff=0)

    def test_construct_invalid_dropout(self):
        from pynerve.nn._diagram_attention import DiagramTransformerBlock
        with pytest.raises(ValueError, match="0 <="):
            DiagramTransformerBlock(d_model=16, num_heads=4, dropout=1.0)

    def test_forward(self):
        from pynerve.nn._diagram_attention import DiagramTransformerBlock
        block = DiagramTransformerBlock(d_model=16, num_heads=4, d_ff=32, dropout=0.0)
        d = make_diag_batched(5, batch=2)
        feats = torch.rand(2, 5, 16)
        result = block(d, feats)
        assert result.shape == (2, 5, 16)

    def test_forward_with_mask(self):
        from pynerve.nn._diagram_attention import DiagramTransformerBlock
        block = DiagramTransformerBlock(d_model=16, num_heads=4, d_ff=32, dropout=0.0)
        d = make_diag_batched(5, batch=2)
        feats = torch.rand(2, 5, 16)
        mask = torch.ones(2, 5)
        result = block(d, feats, mask=mask)
        assert result.shape == (2, 5, 16)
