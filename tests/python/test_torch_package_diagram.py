"""Comprehensive tests for pynerve.torch subpackage.

Covers the public API surface: lazy imports, PersistenceDiagram, diagram
distance functions, data utilities, nn layers, preprocessing, statistics,
vectorization, kernels, mapper, sklearn transformers, tensorboard logging,
and visualisation helpers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from pynerve.exceptions import ValidationError  # noqa: E402


def _torch_backend_available() -> bool:
    """Check if the PyTorch C++ extension (pynerve_torch_internal) is present."""
    try:
        import nerve_torch_internal  # noqa: F401

        return True
    except ImportError:
        try:
            import pynerve_torch_internal  # noqa: F401

            return True
        except ImportError:
            return False


_torch_backend = pytest.mark.skipif(
    not _torch_backend_available(),
    reason="pynerve_torch_internal not available",
)


def _core_backend_available() -> bool:
    """Check if the core C++ extension (pynerve_internal) is present."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


_core_backend = pytest.mark.skipif(
    not _core_backend_available(),
    reason="pynerve_internal not available",
)


def _networkx_available() -> bool:
    try:
        import networkx  # noqa: F401

        return True
    except ImportError:
        return False


def _make_diagram_tensor(
    batch: int = 2,
    pairs: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """Create a valid persistence diagram tensor (birth, death, dim) with birth < death."""
    torch.manual_seed(seed)
    births = torch.rand(batch, pairs, 1)
    deaths = births + torch.rand(batch, pairs, 1) + 0.1
    dims = torch.randint(0, 2, (batch, pairs, 1)).float()
    return torch.cat([births, deaths, dims], dim=-1)


def _make_2d_diagram(pairs: int = 3, seed: int = 42) -> torch.Tensor:
    """Create a 2D (unbatched) diagram tensor with (birth, death) columns."""
    torch.manual_seed(seed)
    births = torch.rand(pairs, 1)
    deaths = births + torch.rand(pairs, 1) + 0.1
    return torch.cat([births, deaths], dim=-1)


def _make_point_cloud(
    n_points: int = 8,
    dim: int = 3,
    batch: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if batch == 1:
        return torch.rand(n_points, dim)
    return torch.rand(batch, n_points, dim)


# Lazy import mechanism


# helpers


# PersistenceDiagram


class TestPersistenceDiagram:
    """PersistenceDiagram construction, properties, and methods."""

    def test_construct_3d(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=4)
        pd = PersistenceDiagram(tensor)
        assert pd.batch_size == 2
        assert pd.max_pairs == 4
        assert pd.dtype == tensor.dtype
        assert pd.device == tensor.device
        assert pd.diagrams.shape == (2, 4, 3)

    def test_construct_2d_auto_unsqueeze(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=1, pairs=3).squeeze(0)
        assert tensor.dim() == 2
        pd = PersistenceDiagram(tensor)
        assert pd.batch_size == 1
        assert pd.diagrams.dim() == 3

    def test_construct_with_mask(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=4)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        pd = PersistenceDiagram(tensor, mask=mask)
        assert pd.mask.shape == (2, 4)
        assert pd.mask.dtype == torch.bool

    def test_construct_with_1d_mask_unsqueezes(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=1, pairs=4).squeeze(0)
        mask = torch.tensor([True, True, False, False])
        pd = PersistenceDiagram(tensor, mask=mask)
        assert pd.mask.shape == (1, 4)

    def test_construct_with_num_pairs(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=4)
        num_pairs = torch.tensor([[2, 2], [2, 2]])
        pd = PersistenceDiagram(tensor, num_pairs=num_pairs)
        assert pd.num_pairs is not None
        assert pd.num_pairs.shape == (2, 2)

    def test_mask_shape_mismatch_raises(self) -> None:
        from pynerve.exceptions import ShapeError
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=4)
        bad_mask = torch.ones(3, 4, dtype=torch.bool)
        with pytest.raises(ShapeError):
            PersistenceDiagram(tensor, mask=bad_mask)

    def test_construct_wrong_last_dim_raises(self) -> None:
        from pynerve.exceptions import ShapeError
        from pynerve.torch import PersistenceDiagram

        tensor = torch.rand(2, 4, 2)
        with pytest.raises(ShapeError):
            PersistenceDiagram(tensor)

    def test_construct_1d_raises(self) -> None:
        from pynerve.exceptions import ShapeError
        from pynerve.torch import PersistenceDiagram

        with pytest.raises(ShapeError):
            PersistenceDiagram(torch.randn(10))

    def test_construct_4d_raises(self) -> None:
        from pynerve.exceptions import ShapeError
        from pynerve.torch import PersistenceDiagram

        with pytest.raises(ShapeError):
            PersistenceDiagram(torch.rand(2, 3, 4, 5))

    def test_construct_invalid_dimensions_raises(self) -> None:
        from pynerve.exceptions import ValidationError
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, -1.0]]], dtype=torch.float32)
        with pytest.raises(ValidationError):
            PersistenceDiagram(tensor)

    def test_construct_non_integer_dimensions_raises(self) -> None:
        from pynerve.exceptions import ValidationError
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.5]]], dtype=torch.float32)
        with pytest.raises(ValidationError):
            PersistenceDiagram(tensor)

    def test_births_property(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        births = pd.births(apply_mask=False)
        assert births.shape == (2, 3)
        torch.testing.assert_close(births, tensor[..., 0])

    def test_births_masked(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=1, pairs=3)
        mask = torch.tensor([[True, False, True]])
        pd = PersistenceDiagram(tensor, mask=mask)
        births = pd.births(apply_mask=True)
        assert births.shape == (2,)

    def test_deaths_property(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        deaths = pd.deaths(apply_mask=False)
        torch.testing.assert_close(deaths, tensor[..., 1])

    def test_deaths_masked(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=1, pairs=3)
        mask = torch.tensor([[True, True, False]])
        pd = PersistenceDiagram(tensor, mask=mask)
        deaths = pd.deaths(apply_mask=True)
        assert deaths.shape == (2,)

    def test_dimensions_property(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        dims = pd.dimensions()
        assert dims.dtype == torch.long
        torch.testing.assert_close(dims.float(), tensor[..., 2])

    def test_tensor_method(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        t = pd.tensor()
        assert t is pd.diagrams

    def test_to_device(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        pd_cpu = pd.to("cpu")
        assert pd_cpu.device == torch.device("cpu")
        assert pd_cpu.dtype == pd.dtype

    def test_to_dtype(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        pd_f64 = pd.to_dtype(torch.float64)
        assert pd_f64.dtype == torch.float64
        assert pd_f64.batch_size == pd.batch_size

    def test_to_dtype_non_float_raises(self) -> None:
        from pynerve.exceptions import ValidationError
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        with pytest.raises(ValidationError):
            pd.to_dtype(torch.long)

    def test_get_batch_item(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=3, pairs=4)
        pd = PersistenceDiagram(tensor)
        item = pd.get_batch_item(1)
        assert item.batch_size == 1
        assert item.max_pairs == 4
        torch.testing.assert_close(item.diagrams[0], tensor[1])

    def test_get_batch_item_out_of_range_raises(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=3)
        pd = PersistenceDiagram(tensor)
        with pytest.raises(IndexError):
            pd.get_batch_item(5)
        with pytest.raises(IndexError):
            pd.get_batch_item(-1)

    def test_total_persistence_default_p(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        tp = pd.total_persistence()
        assert tp.dtype == torch.float64
        assert tp.shape == (1,)

    def test_total_persistence_p_one(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        tp = pd.total_persistence(p=1.0)
        assert tp.shape == (1,)
        assert tp[0].item() == pytest.approx(3.0)

    def test_total_persistence_batched(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=3, pairs=5)
        pd = PersistenceDiagram(tensor)
        tp = pd.total_persistence()
        assert tp.shape == (3,)

    def test_total_persistence_invalid_p_raises(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=1, pairs=2)
        pd = PersistenceDiagram(tensor)
        with pytest.raises((ValueError, ValidationError)):
            pd.total_persistence(p=float("nan"))

    def test_persistence_entropy(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        ent = pd.persistence_entropy()
        assert ent.dtype == torch.float64
        assert ent.shape == (1,)
        assert torch.isfinite(ent).all()

    def test_persistence_entropy_batched(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=3, pairs=5)
        pd = PersistenceDiagram(tensor)
        ent = pd.persistence_entropy()
        assert ent.shape == (3,)

    def test_persistence_entropy_zero_persistence(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.zeros(1, 3, 3, dtype=torch.float32)
        tensor[..., 2] = 0.0
        pd = PersistenceDiagram(tensor)
        ent = pd.persistence_entropy()
        assert torch.isfinite(ent).all()

    def test_filter_by_dimension(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=4)
        pd = PersistenceDiagram(tensor)
        filtered = pd.filter_by_dimension(0)
        assert filtered.batch_size == 2
        assert filtered.max_pairs == 4

    def test_filter_by_dimension_negative_raises(self) -> None:
        from pynerve.exceptions import ValidationError
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=1, pairs=2)
        pd = PersistenceDiagram(tensor)
        with pytest.raises(ValidationError):
            pd.filter_by_dimension(-1)

    def test_repr(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = _make_diagram_tensor(batch=2, pairs=4)
        pd = PersistenceDiagram(tensor)
        r = repr(pd)
        assert "PersistenceDiagram" in r
        assert "batch=2" in r
        assert "max_pairs=4" in r

    def test_births_raises_with_invalid(self) -> None:
        from pynerve.torch import PersistenceDiagram

        bad = torch.tensor([[[float("nan"), 1.0, 0.0]]], dtype=torch.float32)
        with pytest.raises(ValueError):
            PersistenceDiagram(bad)

    def test_deaths_raises_with_invalid(self) -> None:
        from pynerve.torch import PersistenceDiagram

        bad = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
        with pytest.raises(ValueError):
            PersistenceDiagram(bad)
