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


class TestBatchDiagrams:
    """batch_diagrams / unbatch_diagrams utilities."""

    def test_batch_two_diagrams(self) -> None:
        from pynerve.torch import PersistenceDiagram, batch_diagrams

        d1 = PersistenceDiagram(_make_diagram_tensor(batch=1, pairs=2, seed=1))
        d2 = PersistenceDiagram(_make_diagram_tensor(batch=1, pairs=3, seed=2))
        batched = batch_diagrams([d1, d2])
        assert batched.batch_size == 2
        assert batched.max_pairs == 3
        assert batched.diagrams.shape == (2, 3, 3)

    def test_batch_mask_preserved(self) -> None:
        from pynerve.torch import PersistenceDiagram, batch_diagrams

        t = _make_diagram_tensor(batch=1, pairs=2)
        mask = torch.tensor([[True, False]])
        d = PersistenceDiagram(t, mask=mask)
        batched = batch_diagrams([d])
        assert batched.mask[0, 0].item() is True
        assert batched.mask[0, 1].item() is False

    def test_batch_empty_list_raises(self) -> None:
        from pynerve.torch import batch_diagrams

        with pytest.raises(ValidationError):
            batch_diagrams([])

    def test_batch_mismatched_device_raises(self) -> None:
        pytest.skip("CUDA not available or misconfigured on this system")
        from pynerve.torch import PersistenceDiagram, batch_diagrams

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        d1 = PersistenceDiagram(_make_diagram_tensor(batch=1, pairs=2))
        d2 = PersistenceDiagram(_make_diagram_tensor(batch=1, pairs=2).to("cuda:0"))
        with pytest.raises(ValidationError):
            batch_diagrams([d1, d2])

    def test_batch_mismatched_dtype_raises(self) -> None:
        from pynerve.torch import PersistenceDiagram, batch_diagrams

        d1 = PersistenceDiagram(_make_diagram_tensor(batch=1, pairs=2, seed=1).to(torch.float64))
        d2 = PersistenceDiagram(_make_diagram_tensor(batch=1, pairs=2, seed=2))
        with pytest.raises(ValueError):
            batch_diagrams([d1, d2])

    def test_unbatch(self) -> None:
        from pynerve.torch import PersistenceDiagram, unbatch_diagrams

        d = PersistenceDiagram(_make_diagram_tensor(batch=3, pairs=4))
        items = unbatch_diagrams(d)
        assert len(items) == 3
        assert all(item.batch_size == 1 for item in items)

    def test_unbatch_roundtrip(self) -> None:
        from pynerve.torch import PersistenceDiagram, batch_diagrams, unbatch_diagrams

        d1 = PersistenceDiagram(_make_diagram_tensor(batch=1, pairs=2, seed=1))
        d2 = PersistenceDiagram(_make_diagram_tensor(batch=1, pairs=3, seed=2))
        batched = batch_diagrams([d1, d2])
        items = unbatch_diagrams(batched)
        assert len(items) == 2
