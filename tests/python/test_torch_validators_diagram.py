"""Tests for torch/_persistence_validators.py and torch/_diagram.py."""

from __future__ import annotations

import pytest
import torch

from pynerve.exceptions._validation import ShapeError, ValidationError
from pynerve.torch._diagram import PersistenceDiagram, batch_diagrams, unbatch_diagrams
from pynerve.torch._persistence_validators import (
    _validate_image_resolution,
    _validate_max_dim,
    _validate_max_radius,
    _validate_metric,
    _validate_persistence_image_diagram,
)


# ── _validate_max_dim ──────────────────────────────────────────────────────


class TestValidateMaxDim:
    def test_valid(self):
        assert _validate_max_dim(0) == 0
        assert _validate_max_dim(5) == 5

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_max_dim(-1)

    def test_bool_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_max_dim(True)  # type: ignore[arg-type]

    def test_not_int_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_max_dim("abc")  # type: ignore[arg-type]


# ── _validate_max_radius ───────────────────────────────────────────────────


class TestValidateMaxRadius:
    def test_valid(self):
        assert _validate_max_radius(1.0) == 1.0
        assert _validate_max_radius(100.0) == 100.0

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_max_radius(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_max_radius(-1.0)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_max_radius(float("nan"))


# ── _validate_metric ───────────────────────────────────────────────────────


class TestValidateMetric:
    def test_valid_metrics(self):
        for m in ("euclidean", "manhattan", "chebyshev", "cosine"):
            assert _validate_metric(m) == m

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _validate_metric("minkowski")


# ── _validate_image_resolution ─────────────────────────────────────────────


class TestValidateImageResolution:
    def test_valid(self):
        assert _validate_image_resolution((20, 30)) == (20, 30)

    def test_single_element_raises(self):
        with pytest.raises(ValueError, match="exactly two"):
            _validate_image_resolution((20,))

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_image_resolution((0, 20))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_image_resolution((-1, 20))


# ── _validate_persistence_image_diagram ────────────────────────────────────


class TestValidatePersistenceImageDiagram:
    def test_valid(self):
        d = torch.tensor([[0.0, 1.0, 0]], dtype=torch.float32)
        result = _validate_persistence_image_diagram(d)
        assert result is d

    def test_empty(self):
        d = torch.empty((0, 3), dtype=torch.float32)
        result = _validate_persistence_image_diagram(d)
        assert result.numel() == 0

    def test_1d_raises(self):
        d = torch.tensor([0.0, 1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="2D"):
            _validate_persistence_image_diagram(d)

    def test_int_dtype_raises(self):
        d = torch.tensor([[0, 1, 0]], dtype=torch.int64)
        with pytest.raises(TypeError, match="floating-point"):
            _validate_persistence_image_diagram(d)

    def test_nan_birth_raises(self):
        d = torch.tensor([[float("nan"), 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="births"):
            _validate_persistence_image_diagram(d)

    def test_nan_death_raises(self):
        d = torch.tensor([[0.0, float("nan"), 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="NaN"):
            _validate_persistence_image_diagram(d)

    def test_death_less_than_birth_raises(self):
        d = torch.tensor([[5.0, 1.0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="greater than"):
            _validate_persistence_image_diagram(d)

    def test_infinite_death_allowed(self):
        d = torch.tensor([[0.0, float("inf"), 0]], dtype=torch.float32)
        result = _validate_persistence_image_diagram(d)
        assert result is d


# ── PersistenceDiagram ─────────────────────────────────────────────────────


class TestPersistenceDiagram:
    def test_construction_2d(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        assert pd.batch_size == 1
        assert pd.max_pairs == 1

    def test_construction_3d(self):
        d = torch.tensor([[[0.0, 1.0, 0], [1.0, 2.0, 1]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        assert pd.batch_size == 1
        assert pd.max_pairs == 2

    def test_construction_wrong_last_dim_raises(self):
        d = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        with pytest.raises(ShapeError, match="last dimension"):
            PersistenceDiagram(d)

    def test_construction_wrong_dim_raises(self):
        d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with pytest.raises(ShapeError, match="2D or 3D"):
            PersistenceDiagram(d)

    def test_construction_with_mask(self):
        d = torch.tensor([[[0.0, 1.0, 0], [1.0, 2.0, 1]]], dtype=torch.float32)
        mask = torch.tensor([[True, False]])
        pd = PersistenceDiagram(d, mask=mask)
        assert pd.mask.shape == (1, 2)

    def test_births(self):
        d = torch.tensor([[[0.0, 1.0, 0], [2.0, 3.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        births = pd.births()
        assert births.shape[0] == 2
        assert births[0] == 0.0
        assert births[1] == 2.0

    def test_deaths(self):
        d = torch.tensor([[[0.0, 1.0, 0], [2.0, 3.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        deaths = pd.deaths()
        assert deaths[0] == 1.0
        assert deaths[1] == 3.0

    def test_dimensions(self):
        d = torch.tensor([[[0.0, 1.0, 0], [2.0, 3.0, 1]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        dims = pd.dimensions()
        assert dims[0, 0] == 0
        assert dims[0, 1] == 1

    def test_tensor(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        assert torch.allclose(pd.tensor(), d)

    def test_device(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        assert pd.device == torch.device("cpu")

    def test_dtype(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float64)
        pd = PersistenceDiagram(d)
        assert pd.dtype == torch.float64

    def test_to_device(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        pd_cpu = pd.to("cpu")
        assert pd_cpu.device == torch.device("cpu")

    def test_to_dtype(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        pd_64 = pd.to_dtype(torch.float64)
        assert pd_64.dtype == torch.float64

    def test_to_dtype_non_float_raises(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        with pytest.raises(ValidationError, match="floating-point"):
            pd.to_dtype(torch.int64)

    def test_get_batch_item(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 3.0, 0]]], dtype=torch.float32
        )
        pd = PersistenceDiagram(d)
        item = pd.get_batch_item(0)
        assert item.batch_size == 1
        assert item.tensor()[0, 0, 0] == 0.0

    def test_get_batch_item_out_of_range(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        with pytest.raises(IndexError):
            pd.get_batch_item(5)

    def test_filter_by_dimension(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0], [2.0, 3.0, 1], [4.0, 5.0, 0]]], dtype=torch.float32
        )
        pd = PersistenceDiagram(d)
        filtered = pd.filter_by_dimension(0)
        births = filtered.births()
        # Should have 2 pairs in dim 0
        assert births.shape[0] == 2

    def test_filter_by_dimension_negative_raises(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        with pytest.raises(ValidationError, match="non-negative"):
            pd.filter_by_dimension(-1)

    def test_repr(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        r = repr(pd)
        assert "PersistenceDiagram" in r
        assert "batch=1" in r

    def test_batch_size_property(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        assert pd.batch_size == 1

    def test_diagrams_property(self):
        d = torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        pd = PersistenceDiagram(d)
        assert pd.diagrams.shape == (1, 1, 3)

    def test_invalid_dim_finite(self):
        d = torch.tensor([[[0.0, 1.0, float("nan")]]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="finite"):
            PersistenceDiagram(d)

    def test_negative_dim(self):
        d = torch.tensor([[[0.0, 1.0, -1.0]]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="non-negative"):
            PersistenceDiagram(d)

    def test_non_integer_dim(self):
        d = torch.tensor([[[0.0, 1.0, 0.5]]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="integers"):
            PersistenceDiagram(d)


# ── batch_diagrams ─────────────────────────────────────────────────────────


class TestBatchDiagrams:
    def test_basic(self):
        d1 = PersistenceDiagram(
            torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        )
        d2 = PersistenceDiagram(
            torch.tensor([[[2.0, 3.0, 0]]], dtype=torch.float32)
        )
        result = batch_diagrams([d1, d2])
        assert result.batch_size == 2
        assert result.max_pairs == 1

    def test_uneven_sizes(self):
        d1 = PersistenceDiagram(
            torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        )
        d2 = PersistenceDiagram(
            torch.tensor([[[2.0, 3.0, 0], [4.0, 5.0, 0]]], dtype=torch.float32)
        )
        result = batch_diagrams([d1, d2])
        assert result.max_pairs == 2

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            batch_diagrams([])

    def test_different_devices_raises(self):
        d1 = PersistenceDiagram(
            torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        )
        if torch.cuda.is_available():
            d2 = PersistenceDiagram(
                torch.tensor([[[2.0, 3.0, 0]]], dtype=torch.float32, device="cuda")
            )
            with pytest.raises(ValidationError, match="same device"):
                batch_diagrams([d1, d2])  # type: ignore[list-item]

    def test_with_num_pairs(self):
        d1 = PersistenceDiagram(
            torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32),
            num_pairs=torch.tensor([[1]]),
        )
        d2 = PersistenceDiagram(
            torch.tensor([[[2.0, 3.0, 0]]], dtype=torch.float32),
            num_pairs=torch.tensor([[1]]),
        )
        result = batch_diagrams([d1, d2])
        assert result.num_pairs is not None


# ── unbatch_diagrams ───────────────────────────────────────────────────────


class TestUnbatchDiagrams:
    def test_basic(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 3.0, 0]]], dtype=torch.float32
        )
        pd = PersistenceDiagram(d)
        items = unbatch_diagrams(pd)
        assert len(items) == 2
        assert all(isinstance(item, PersistenceDiagram) for item in items)
        assert items[0].batch_size == 1

    def test_roundtrip(self):
        d = torch.tensor(
            [[[0.0, 1.0, 0]], [[2.0, 3.0, 0]]], dtype=torch.float32
        )
        pd = PersistenceDiagram(d)
        batched = batch_diagrams(unbatch_diagrams(pd))
        assert batched.batch_size == 2
