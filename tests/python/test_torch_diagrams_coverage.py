from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pynerve._torch_diagrams import (
    birth_death,
    encode_diagram_embedding,
    encode_diagram_rows,
    encoder_output_dim,
    persistence,
    validate_diagram,
)


class _EncModule(nn.Module):
    def __init__(self, output_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self._captured_input = None

    def forward(self, x):
        self._captured_input = x
        raise NotImplementedError


class _FixedRank2(nn.Module):
    def forward(self, x):
        return x.squeeze(0)


class _FixedRank3(nn.Module):
    def forward(self, x):
        return x


class _FixedRank1(nn.Module):
    def forward(self, x):
        return x.squeeze(0).mean(dim=0)


class _FixedRank3WrongBatch(nn.Module):
    def forward(self, x):
        return x.repeat(2, 1, 1)


class _FixedRank2WrongRows(nn.Module):
    def forward(self, x):
        n = x.shape[1]
        return x.squeeze(0)[: n - 1]


class _NonTensorEncoder(nn.Module):
    def forward(self, x):
        return x.squeeze(0).tolist()


def _make_diagram(n_pairs=5, n_cols=3):
    return torch.randn(n_pairs, n_cols, dtype=torch.float32)


class TestEncoderOutputDim:
    def test_returns_output_dim_attribute(self):
        encoder = nn.Linear(10, 5)
        encoder.output_dim = 42
        assert encoder_output_dim(encoder) == 42

    def test_falls_back_to_default_when_missing(self):
        encoder = nn.Linear(10, 5)
        assert encoder_output_dim(encoder) == 64

    def test_respects_custom_default(self):
        encoder = nn.Linear(10, 5)
        assert encoder_output_dim(encoder, default=128) == 128

    def test_casts_output_dim_to_int(self):
        encoder = nn.Linear(10, 5)
        encoder.output_dim = 99.7
        assert encoder_output_dim(encoder) == 99


class TestValidateDiagram:
    def test_accepts_valid_2d_tensor(self):
        validate_diagram(_make_diagram())

    def test_accepts_exact_min_cols(self):
        validate_diagram(_make_diagram(n_pairs=3, n_cols=2), min_cols=2)

    def test_accepts_custom_min_cols(self):
        validate_diagram(_make_diagram(n_pairs=3, n_cols=4), min_cols=3)

    def test_rejects_list(self):
        with pytest.raises(TypeError, match="must be a tensor"):
            validate_diagram([1.0, 2.0, 3.0])

    def test_rejects_numpy_array(self):
        with pytest.raises(TypeError, match="must be a tensor"):
            validate_diagram(__import__("numpy").array([[1.0, 2.0]]))

    def test_rejects_1d_tensor(self):
        with pytest.raises(ValueError, match="must have shape"):
            validate_diagram(torch.tensor([1.0, 2.0]))

    def test_rejects_3d_tensor(self):
        with pytest.raises(ValueError, match="must have shape"):
            validate_diagram(torch.randn(2, 3, 4))

    def test_rejects_too_few_columns(self):
        with pytest.raises(ValueError, match="must have shape"):
            validate_diagram(torch.randn(5, 1), min_cols=2)

    def test_error_message_includes_custom_name(self):
        with pytest.raises(TypeError, match="my_tensor must be a tensor"):
            validate_diagram(42, name="my_tensor")

    def test_error_message_includes_custom_name_in_value_error(self):
        with pytest.raises(ValueError, match="batch must have shape"):
            validate_diagram(torch.randn(5, 1), min_cols=3, name="batch")

    def test_accepts_single_row(self):
        validate_diagram(torch.randn(1, 2))


class TestBirthDeath:
    def test_returns_first_two_columns(self):
        diagram = torch.tensor([[1.0, 2.0, 0.5], [3.0, 5.0, 0.8]])
        result = birth_death(diagram)
        assert result.shape == (2, 2)
        assert torch.equal(result, diagram[:, :2])

    def test_works_with_exactly_two_columns(self):
        diagram = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        result = birth_death(diagram)
        assert torch.equal(result, diagram)

    def test_raises_on_invalid_diagram(self):
        with pytest.raises((TypeError, ValueError)):
            birth_death(torch.tensor([1.0, 2.0]))

    def test_preserves_dtype(self):
        diagram = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        result = birth_death(diagram)
        assert result.dtype == torch.float64


class TestPersistence:
    def test_computes_death_minus_birth(self):
        diagram = torch.tensor([[1.0, 3.0], [2.0, 5.0]])
        result = persistence(diagram)
        expected = torch.tensor([2.0, 3.0])
        assert torch.allclose(result, expected)

    def test_returns_zero_when_birth_equals_death(self):
        diagram = torch.tensor([[4.0, 4.0], [7.0, 7.0]])
        result = persistence(diagram)
        assert torch.allclose(result, torch.zeros(2))

    def test_handles_extra_columns(self):
        diagram = torch.tensor([[1.0, 2.0, 9.9], [3.0, 4.0, 8.8]])
        result = persistence(diagram)
        expected = torch.tensor([1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_raises_on_invalid_diagram(self):
        with pytest.raises((TypeError, ValueError)):
            persistence(torch.tensor([1.0]))

    def test_preserves_dtype(self):
        diagram = torch.tensor([[1.0, 3.0], [2.0, 5.0]], dtype=torch.float64)
        result = persistence(diagram)
        assert result.dtype == torch.float64


class TestEncodeDiagramRows:
    def test_squeezes_rank3_encoder_output(self):
        encoder = _FixedRank3()
        diagram = _make_diagram(n_pairs=4, n_cols=3)
        result = encode_diagram_rows(encoder, diagram)
        assert result.shape == (4, 3)

    def test_expands_rank2_single_row_to_n_pairs(self):
        encoder = _FixedRank2()
        diagram = _make_diagram(n_pairs=7, n_cols=3)
        result = encode_diagram_rows(encoder, diagram)
        assert result.shape == (7, 3)

    def test_expands_rank1_to_n_pairs(self):
        encoder = _FixedRank1()
        diagram = _make_diagram(n_pairs=6, n_cols=3)
        result = encode_diagram_rows(encoder, diagram)
        assert result.shape == (6, 3)

    def test_accepts_custom_min_cols(self):
        encoder = _FixedRank3()
        diagram = _make_diagram(n_pairs=2, n_cols=5)
        result = encode_diagram_rows(encoder, diagram, min_cols=4)
        assert result.shape == (2, 5)

    def test_raises_on_non_tensor_encoder_output(self):
        encoder = _NonTensorEncoder()
        diagram = _make_diagram(n_pairs=2, n_cols=2)
        with pytest.raises(TypeError, match="encoder must return a tensor"):
            encode_diagram_rows(encoder, diagram)

    def test_raises_when_batched_output_has_wrong_batch_size(self):
        encoder = _FixedRank3WrongBatch()
        diagram = _make_diagram(n_pairs=3, n_cols=2)
        with pytest.raises(ValueError, match="preserve one input diagram"):
            encode_diagram_rows(encoder, diagram)

    def test_raises_when_output_rows_dont_match_n_pairs(self):
        encoder = _FixedRank2WrongRows()
        diagram = _make_diagram(n_pairs=5, n_cols=2)
        with pytest.raises(ValueError, match="encoder output must be graph-level"):
            encode_diagram_rows(encoder, diagram)

    def test_raises_on_invalid_diagram(self):
        encoder = _FixedRank3()
        with pytest.raises((TypeError, ValueError)):
            encode_diagram_rows(encoder, torch.tensor([1.0]))

    def test_preserves_existing_2d_with_correct_rows(self):
        class _Passthrough(nn.Module):
            def forward(self, x):
                return x.squeeze(0)

        encoder = _Passthrough()
        diagram = _make_diagram(n_pairs=3, n_cols=4)
        result = encode_diagram_rows(encoder, diagram)
        assert result.shape == (3, 4)
        assert torch.equal(result, diagram)


class TestEncodeDiagramEmbedding:
    def test_returns_rowwise_mean(self):
        encoder = _FixedRank2()
        diagram = _make_diagram(n_pairs=4, n_cols=3)
        result = encode_diagram_embedding(encoder, diagram)
        rows = encode_diagram_rows(encoder, diagram)
        expected = rows.mean(dim=0)
        assert torch.equal(result, expected)

    def test_returns_zeros_for_empty_diagram_with_default_dim(self):
        encoder = _FixedRank2()
        diagram = torch.empty(0, 2)
        result = encode_diagram_embedding(encoder, diagram)
        assert result.shape == (64,)
        assert torch.allclose(result, torch.zeros(64))

    def test_returns_zeros_for_empty_diagram_with_explicit_output_dim(self):
        encoder = nn.Linear(3, 7)
        encoder.output_dim = 7
        diagram = torch.empty(0, 3)
        result = encode_diagram_embedding(encoder, diagram)
        assert result.shape == (7,)
        assert torch.allclose(result, torch.zeros(7))

    def test_empty_diagram_zeros_are_same_device_and_dtype(self):
        encoder = _FixedRank2()
        diagram = torch.empty(0, 2, dtype=torch.float64)
        result = encode_diagram_embedding(encoder, diagram)
        assert result.dtype == torch.float64

    def test_accepts_custom_min_cols(self):
        encoder = _FixedRank2()
        diagram = _make_diagram(n_pairs=3, n_cols=4)
        result = encode_diagram_embedding(encoder, diagram, min_cols=3)
        assert result.shape == (4,)

    def test_raises_on_invalid_diagram(self):
        encoder = _FixedRank2()
        with pytest.raises((TypeError, ValueError)):
            encode_diagram_embedding(encoder, torch.tensor([1.0]))

    def test_output_is_1d(self):
        encoder = _FixedRank2()
        diagram = _make_diagram(n_pairs=3, n_cols=4)
        result = encode_diagram_embedding(encoder, diagram)
        assert result.dim() == 1
