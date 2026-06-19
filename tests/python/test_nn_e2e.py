"""Correctness tests for pynerve.nn subpackage."""

from __future__ import annotations

import pytest

try:
    import pynerve_internal  # noqa: F401
except ImportError:
    pytest.skip("pynerve_internal C++ extension not available", allow_module_level=True)

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

# Helpers


def _has_torch_backend() -> bool:
    try:
        from pynerve.torch._persistence_validators import _torch_backend  # noqa: PLC0415

        return _torch_backend() is not None
    except ImportError:
        return False


_torch_skip = pytest.mark.skipif(
    not _has_torch_backend(), reason="torch C++ backend (pynerve_torch_internal) required"
)


def _has_core_backend() -> bool:
    """Check if the C++ persistence backend (pynerve_internal) is importable."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


def _has_internal_backend() -> bool:
    """Check if pynerve_internal (witness persistence) is importable."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


_core_skip = pytest.mark.skipif(
    not _has_core_backend(), reason="C++ persistence backend not available"
)
_internal_skip = pytest.mark.skipif(
    not _has_internal_backend(), reason="pynerve_internal C++ extension not available"
)

BATCH = 2
N_POINTS = 8
DIM = 3
SEED = 42


@pytest.fixture(scope="module")
def rng():
    torch.manual_seed(SEED)
    return torch.Generator().manual_seed(SEED)


@pytest.fixture(scope="module")
def point_cloud_2d() -> Tensor:
    """Single point cloud (N, D)."""
    torch.manual_seed(SEED)
    return torch.rand(N_POINTS, DIM)


@pytest.fixture(scope="module")
def point_cloud_batch() -> Tensor:
    """Batch of point clouds (B, N, D)."""
    torch.manual_seed(SEED)
    return torch.rand(BATCH, N_POINTS, DIM)


@pytest.fixture(scope="module")
def diagram_batch() -> Tensor:
    """Fake persistence diagram batch (B, pairs, 3) with birth<death."""
    torch.manual_seed(SEED)
    births = torch.rand(BATCH, 4, 1)
    deaths = births + torch.rand(BATCH, 4, 1) + 0.1
    dims = torch.randint(0, 2, (BATCH, 4, 1)).float()
    return torch.cat([births, deaths, dims], dim=-1)


@pytest.fixture(scope="module")
def simple_diagram() -> Tensor:
    """A simple (batch, 2, 3) diagram for attention / pooling tests."""
    return torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.2, 0.8, 1.0]],
            [[0.1, 0.9, 0.0], [0.3, 0.7, 1.0]],
        ],
        dtype=torch.float32,
    )


# Integration / End-to-End tests


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_full_pipeline_ph_to_convnet(self, point_cloud_batch):
        from pynerve.nn.diagram_conv import DiagramConvNet
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=1, max_radius=float("inf"))
        diagrams = ph(point_cloud_batch)

        # Concatenate per-dimension diagrams into a single tensor
        # Each dimension produces (batch, max_pairs, 2).
        # We stack into (batch, pairs, 3) with dim as third channel.
        diag_tensors = []
        for dim_idx, d in enumerate(diagrams):
            if d.shape[1] == 0:
                continue
            dim_col = torch.full(
                (d.shape[0], d.shape[1], 1), float(dim_idx), device=d.device, dtype=d.dtype
            )
            diag_tensors.append(torch.cat([d, dim_col], dim=-1))

        if not diag_tensors:
            combined = torch.zeros(BATCH, 0, 3, device=point_cloud_batch.device)
        elif len(diag_tensors) == 1:
            combined = diag_tensors[0]
        else:
            max_pairs = max(t.shape[1] for t in diag_tensors)
            padded = []
            for t in diag_tensors:
                if t.shape[1] < max_pairs:
                    pad = torch.zeros(t.shape[0], max_pairs - t.shape[1], 3, device=t.device)
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            combined = torch.cat(padded, dim=1)

        net = DiagramConvNet(in_channels=3, hidden_channels=[32], out_dim=5)
        out = net(combined)
        assert out.shape == (BATCH, 5)

    def test_ph_gradient_flow_to_conv(self, point_cloud_batch):
        from pynerve.nn.diagram_conv import DiagramDeepSet
        from pynerve.nn.persistent_homology import PersistentHomology

        ph = PersistentHomology(max_dim=0, max_radius=float("inf"))
        pts = point_cloud_batch.clone().requires_grad_(True)
        diagrams = ph(pts)

        d0 = diagrams[0]
        if d0.shape[1] == 0:
            pytest.skip("No 0D features; gradient test skipped")
        d0_diag = torch.cat([d0, torch.zeros(BATCH, d0.shape[1], 1, device=d0.device)], dim=-1)

        ds = DiagramDeepSet(in_channels=0, hidden_channels=[16], out_channels=5, pooling="max")
        out = ds(d0_diag)
        loss = out.sum()
        loss.backward()

        assert pts.grad is not None
