"""Tests for torch/data.py -- datasets, collation, padding, validation."""

from __future__ import annotations

import pytest
import torch

from pynerve.torch.data import (
    PointCloudDataset,
    _pad_point_clouds,
    _validate_point_cloud_batch,
    _validate_single_point_cloud,
    collate_diagrams,
    collate_point_clouds,
    create_dataloader,
)
from pynerve.torch._diagram import PersistenceDiagram


# _validate_single_point_cloud 


class TestValidateSinglePointCloud:
    def test_valid(self):
        pc = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
        _validate_single_point_cloud(pc, dim=2, device=pc.device, dtype=pc.dtype)

    def test_not_tensor_raises(self):
        with pytest.raises(TypeError, match="tensors"):
            _validate_single_point_cloud(
                [[0.0, 1.0]], dim=2, device=torch.device("cpu"), dtype=torch.float32
            )  # type: ignore[arg-type]

    def test_1d_raises(self):
        pc = torch.tensor([0.0, 1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="2D"):
            _validate_single_point_cloud(pc, dim=2, device=pc.device, dtype=pc.dtype)

    def test_empty_rows_raises(self):
        pc = torch.empty((0, 2), dtype=torch.float32)
        with pytest.raises(ValueError, match="at least one point"):
            _validate_single_point_cloud(pc, dim=2, device=pc.device, dtype=pc.dtype)

    def test_empty_cols_raises(self):
        pc = torch.empty((3, 0), dtype=torch.float32)
        with pytest.raises(ValueError, match="at least one coordinate"):
            _validate_single_point_cloud(pc, dim=0, device=pc.device, dtype=pc.dtype)

    def test_wrong_dim_raises(self):
        pc = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="coordinate dimension"):
            _validate_single_point_cloud(pc, dim=2, device=pc.device, dtype=pc.dtype)

    def test_wrong_device_raises(self):
        pc = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        pc_cuda = pc.to("cuda")
        with pytest.raises(ValueError, match="same device"):
            _validate_single_point_cloud(pc_cuda, dim=2, device=torch.device("cpu"), dtype=torch.float32)

    def test_non_float_raises(self):
        pc = torch.tensor([[0, 1]], dtype=torch.int64)
        with pytest.raises(TypeError, match="floating-point"):
            _validate_single_point_cloud(pc, dim=2, device=pc.device, dtype=pc.dtype)

    def test_non_finite_raises(self):
        pc = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite"):
            _validate_single_point_cloud(pc, dim=2, device=pc.device, dtype=pc.dtype)


# _validate_point_cloud_batch 


class TestValidatePointCloudBatch:
    def test_valid(self):
        batch = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        ]
        dim, device, dtype = _validate_point_cloud_batch(batch)
        assert dim == 2
        assert device == torch.device("cpu")
        assert dtype == torch.float32

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_point_cloud_batch([])

    def test_different_dims_raises(self):
        batch = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
        ]
        with pytest.raises(ValueError):
            _validate_point_cloud_batch(batch)

    def test_different_dtypes_raises(self):
        batch = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        ]
        with pytest.raises(ValueError, match="dtype"):
            _validate_point_cloud_batch(batch)


# _pad_point_clouds 


class TestPadPointClouds:
    def test_uniform_no_padding(self):
        batch = [
            torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32),
            torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float32),
        ]
        result = _pad_point_clouds(batch)
        assert result.shape == (2, 2, 2)

    def test_uneven_sizes(self):
        batch = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=torch.float32),
        ]
        result = _pad_point_clouds(batch)
        assert result.shape == (2, 3, 2)
        # First item should have padding zeros
        assert torch.all(result[0, 1:] == 0)

    def test_custom_pad_value(self):
        batch = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        ]
        result = _pad_point_clouds(batch, pad_value=-1.0)
        assert result[0, 1, 0] == -1.0

    def test_empty_batch(self):
        result = _pad_point_clouds([])
        assert result.numel() == 0

    def test_non_finite_pad_value_raises(self):
        batch = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]
        with pytest.raises(ValueError, match="finite"):
            _pad_point_clouds(batch, pad_value=float("nan"))

    def test_single_item(self):
        batch = [torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)]
        result = _pad_point_clouds(batch)
        assert result.shape == (1, 1, 3)


# collate_point_clouds 


class TestCollatePointClouds:
    def test_basic(self):
        batch = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        ]
        result = collate_point_clouds(batch)
        assert result.shape == (2, 1, 2)

    def test_with_labels(self):
        batch = [
            (torch.tensor([[0.0, 1.0]], dtype=torch.float32), torch.tensor(0)),
            (torch.tensor([[2.0, 3.0]], dtype=torch.float32), torch.tensor(1)),
        ]
        result = collate_point_clouds(batch)
        assert isinstance(result, tuple)
        pc_batch, labels = result
        assert pc_batch.shape == (2, 1, 2)
        assert labels.shape == (2,)

    def test_empty_batch(self):
        result = collate_point_clouds([])
        assert result.numel() == 0

    def test_mixed_tuple_non_tuple_raises(self):
        batch = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            (torch.tensor([[0.0, 1.0]], dtype=torch.float32), torch.tensor(0)),
        ]
        with pytest.raises(TypeError, match="tensors"):
            collate_point_clouds(batch)  # type: ignore[list-item]


# collate_diagrams 


class TestCollateDiagrams:
    def test_basic(self):
        d1 = PersistenceDiagram(
            torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        )
        d2 = PersistenceDiagram(
            torch.tensor([[[2.0, 3.0, 0]]], dtype=torch.float32)
        )
        result = collate_diagrams([d1, d2])
        assert isinstance(result, PersistenceDiagram)
        assert result.batch_size == 2

    def test_with_labels(self):
        d1 = PersistenceDiagram(
            torch.tensor([[[0.0, 1.0, 0]]], dtype=torch.float32)
        )
        d2 = PersistenceDiagram(
            torch.tensor([[[2.0, 3.0, 0]]], dtype=torch.float32)
        )
        result = collate_diagrams([(d1, torch.tensor(0)), (d2, torch.tensor(1))])
        assert isinstance(result, tuple)
        batched, labels = result
        assert batched.batch_size == 2
        assert labels.shape == (2,)

    def test_empty_batch(self):
        result = collate_diagrams([])
        assert isinstance(result, PersistenceDiagram)
        assert result.batch_size == 0


# PointCloudDataset 


class TestPointCloudDataset:
    def test_basic(self):
        pcs = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        ]
        ds = PointCloudDataset(pcs)
        assert len(ds) == 2
        item = ds[0]
        assert isinstance(item, torch.Tensor)

    def test_with_labels(self):
        pcs = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]
        labels = torch.tensor([0])
        ds = PointCloudDataset(pcs, labels=labels)
        item = ds[0]
        assert isinstance(item, tuple)
        pc, label = item
        assert label.item() == 0

    def test_label_length_mismatch_raises(self):
        pcs = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]
        labels = torch.tensor([0, 1])
        with pytest.raises(ValueError, match="labels"):
            PointCloudDataset(pcs, labels=labels)

    def test_empty_pcs_allowed(self):
        ds = PointCloudDataset([])
        assert len(ds) == 0

    def test_invalid_point_cloud_raises(self):
        pcs = [torch.tensor([0.0, 1.0], dtype=torch.float32)]  # 1D
        with pytest.raises(ValueError, match="2D"):
            PointCloudDataset(pcs)


# create_dataloader 


class TestCreateDataloader:
    def test_basic(self):
        pcs = [
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 3.0]], dtype=torch.float32),
            torch.tensor([[4.0, 5.0]], dtype=torch.float32),
            torch.tensor([[6.0, 7.0]], dtype=torch.float32),
            torch.tensor([[8.0, 9.0]], dtype=torch.float32),
        ]
        ds = PointCloudDataset(pcs)
        loader = create_dataloader(ds, batch_size=2)
        batches = list(loader)
        assert len(batches) >= 2  # at least ceil(5/2) = 3 batches

    def test_custom_batch_size(self):
        pcs = [torch.tensor([[0.0, 1.0]], dtype=torch.float32) for _ in range(10)]
        ds = PointCloudDataset(pcs)
        loader = create_dataloader(ds, batch_size=5, shuffle=False)
        batches = list(loader)
        assert len(batches) == 2

    def test_invalid_batch_size_raises(self):
        pcs = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]
        ds = PointCloudDataset(pcs)
        with pytest.raises(ValueError, match="batch_size"):
            create_dataloader(ds, batch_size=0)

    def test_negative_num_workers_raises(self):
        pcs = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]
        ds = PointCloudDataset(pcs)
        with pytest.raises(ValueError, match="non-negative"):
            create_dataloader(ds, num_workers=-1)

    def test_bool_num_workers_raises(self):
        pcs = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]
        ds = PointCloudDataset(pcs)
        with pytest.raises(TypeError, match="integer"):
            create_dataloader(ds, num_workers=True)  # type: ignore[arg-type]
