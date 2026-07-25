"""Shared test helpers: GPU dependency mocking and persistence diagram factories.

This module is imported by test files that need to mock out GPU/CuPy/triton/numba/C++
dependencies so they can run on CPU-only machines.  It also provides factory
functions for creating valid persistence-diagram tensors.

Usage in test files::

    from _test_helpers import make_diag_2d, make_diag_3d

    pytestmark = pytest.mark.usefixtures("mock_gpu_deps")
    _GPU_MOCK_CUDA_AVAILABLE = False   # set True if tests need cuda.is_available() == True
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Modules that are replaced with MagicMock when GPU deps are not installed.
_MOCK_MODULES = [
    "cupy",
    "cupy.cuda",
    "cupyx",
    "cupyx.scipy",
    "cupyx.scipy.sparse",
    "numba",
    "numba.cuda",
    "triton",
    "triton.language",
    "pynerve_torch_internal",
    "pynerve_internal",
    "nerve_torch_internal",
    "h5py",
]


def install_gpu_mocks(*, cuda_available: bool = False) -> dict[str, object]:
    """Inject MagicMock replacements for all GPU/CuPy/triton/numba/C++ deps.

    Returns a saved-state dict to pass to :func:`restore_gpu_mocks`.
    Call :func:`restore_gpu_mocks` in a ``finally`` block or fixture teardown
    to undo the injection.
    """
    saved: dict[str, object] = {}
    for mod in _MOCK_MODULES:
        saved[mod] = sys.modules.get(mod)
        sys.modules[mod] = MagicMock()

    # ── CuPy ────────────────────────────────────────────────────────────
    import torch

    sys.modules["cupy"].cuda = MagicMock()
    sys.modules["cupy"].cuda.is_available = MagicMock(return_value=cuda_available)
    sys.modules["cupy"].ndarray = torch.Tensor
    sys.modules["cupy"].asarray = lambda x, **kw: torch.as_tensor(x)
    sys.modules["cupyx"].scipy = MagicMock()
    sys.modules["cupyx"].scipy.sparse = MagicMock()

    # ── Numba ───────────────────────────────────────────────────────────
    sys.modules["numba"].cuda = MagicMock()
    sys.modules["numba"].jit = lambda *a, **k: lambda f: f

    # ── Triton ──────────────────────────────────────────────────────────
    sys.modules["triton"].jit = lambda *a, **k: lambda f: f
    sys.modules["triton"].language = MagicMock()
    sys.modules["triton"].autotune = lambda *a, **k: lambda f: f

    # ── h5py ────────────────────────────────────────────────────────────
    sys.modules["h5py"].File = MagicMock()

    # ── C++ extension mocks ─────────────────────────────────────────────
    _pi = sys.modules["pynerve_internal"]
    _pi.PersistenceOptions = MagicMock()
    _pi.PersistenceMode = MagicMock()
    _pi.PersistenceBackend = MagicMock()
    _pi.compute_persistence = MagicMock(return_value={"pairs": []})

    _pt = sys.modules["pynerve_torch_internal"]
    _pt.MapperConfig = MagicMock()
    _pt.Mapper = MagicMock()
    _pt.ClustererType = MagicMock()
    return saved


def restore_gpu_mocks(saved: dict[str, object]) -> None:
    """Restore original ``sys.modules`` state after :func:`install_gpu_mocks`.

    Args:
        saved: The dict returned by :func:`install_gpu_mocks`.
    """
    for mod, orig in saved.items():
        if orig is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = orig


# ── Persistence-diagram factory functions ─────────────────────────────────


def make_diag_2d(n: int = 5):
    """Create a valid 2-column persistence diagram ``(n, 2)`` with birth < death."""
    import torch

    births = torch.rand(n) * 0.5
    deaths = births + torch.rand(n) * 0.5 + 0.01
    return torch.stack([births, deaths], dim=1)


def make_diag_3d(n: int = 5):
    """Create a valid 3-column diagram ``(n, 3)`` with birth, death, dim columns."""
    import torch

    births = torch.rand(n) * 0.5
    deaths = births + torch.rand(n) * 0.5 + 0.01
    dims = torch.randint(0, 2, (n,)).float()
    return torch.stack([births, deaths, dims], dim=1)


def make_diag_batched(n: int = 5, batch: int = 1):
    """Create valid batched persistence diagrams ``(batch, n, 2)``."""
    import torch

    births = torch.rand(batch, n) * 0.5
    deaths = births + torch.rand(batch, n) * 0.5 + 0.01
    return torch.stack([births, deaths], dim=-1)
