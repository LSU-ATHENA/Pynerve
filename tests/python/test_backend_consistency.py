"""Cross-backend consistency tests.

Verifies that different persistence backends produce equivalent results.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _has_torch_c_backend() -> bool:
    try:
        import pynerve_torch_internal  # noqa: F401

        return True
    except ImportError:
        try:
            import nerve_torch_internal  # noqa: F401

            return True
        except ImportError:
            return False


_torch_c = pytest.mark.skipif(
    not _has_torch_c_backend(),
    reason="torch C++ extension not available",
)


def _diagrams_equal(
    pairs1: list[tuple[float, float, int]],
    pairs2: list[tuple[float, float, int]],
) -> bool:
    if len(pairs1) != len(pairs2):
        return False
    for p1, p2 in zip(sorted(pairs1), sorted(pairs2), strict=True):
        b1, d1, dim1 = p1
        b2, d2, dim2 = p2
        if dim1 != dim2:
            return False
        if not abs(b1 - b2) < 1e-6:
            return False
        if (np.isfinite(d1) and np.isfinite(d2)) and abs(d1 - d2) > 1e-6:
            return False
        if np.isfinite(d1) != np.isfinite(d2):
            return False
    return True


# persistence backend parity


class TestPersistenceBackendConsistency:
    """Consistency across different persistence engines."""

    def test_standard_vs_ph4_pair_count(self) -> None:
        from pynerve import compute_persistence

        pts = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
            dtype=torch.float64,
        )
        r1 = compute_persistence(pts, max_dim=1, max_radius=float("inf"), engine="ph0")
        r2 = compute_persistence(pts, max_dim=1, max_radius=float("inf"), engine="ph4")
        assert len(r1.pairs) == len(r2.pairs)

    def test_ph3_vs_ph4_betti(self) -> None:
        from pynerve import compute_persistence

        pts = torch.randn(10, 3)
        r3 = compute_persistence(pts, max_dim=1, max_radius=5.0, engine="ph3")
        r4 = compute_persistence(pts, max_dim=1, max_radius=5.0, engine="ph4")
        assert len(r3.pairs) == len(r4.pairs)
        assert r3.betti_numbers == r4.betti_numbers

    def test_determinism_same_seed(self) -> None:
        from pynerve import compute_persistence

        pts = torch.randn(20, 3)
        r1 = compute_persistence(pts, max_dim=1, max_radius=5.0, engine="ph4", seed=42)
        r2 = compute_persistence(pts, max_dim=1, max_radius=5.0, engine="ph4", seed=42)
        assert _diagrams_equal(r1.pairs, r2.pairs)

    def test_cpu_forward(self) -> None:
        from pynerve import compute_persistence

        pts = torch.randn(10, 2)
        r_cpu = compute_persistence(pts, max_dim=0, max_radius=5.0, device="cpu")
        assert r_cpu.betti_numbers is not None

    def test_python_fallback_h0(self) -> None:
        from pynerve._compute_api import compute_persistence

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        r = compute_persistence(pts, max_dim=0, max_radius=float("inf"))
        finite_h0 = [(b, d) for b, d, dim in r.pairs if dim == 0 and np.isfinite(d)]
        assert len(finite_h0) == 1
        b, d = finite_h0[0]
        assert b == pytest.approx(0.0, abs=1e-10)
        assert d == pytest.approx(1.0, abs=1e-6)


# distance backend parity


class TestDistanceBackendConsistency:
    """Distance calculations consistent across implementations."""

    def test_wasserstein_python_vs_wrapped(self) -> None:
        from pynerve.torch import diagram_wasserstein as wrapped
        from pynerve.torch._distance_core_impl import _wasserstein_python

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.5, 1.5], [0.0, 3.0]], dtype=torch.float64)
        py_dist = _wasserstein_python(d1, d2, 2.0, 2.0)
        w_dist = wrapped(d1, d2)
        wv = w_dist if isinstance(w_dist, torch.Tensor) else torch.tensor(w_dist)
        assert py_dist.item() == pytest.approx(wv.item(), rel=1e-3)

    def test_wasserstein_l1_ge_l2(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        d2 = torch.tensor([[0.5, 1.5], [0.0, 3.0]], dtype=torch.float64)
        w1 = diagram_wasserstein(d1, d2, p=1.0, q=2.0)
        w2 = diagram_wasserstein(d1, d2, p=2.0, q=2.0)
        v1 = w1 if isinstance(w1, torch.Tensor) else torch.tensor(w1)
        v2 = w2 if isinstance(w2, torch.Tensor) else torch.tensor(w2)
        assert v1.item() >= v2.item() - 1e-6

    def test_wasserstein_inf_q_matches_bottleneck(self) -> None:
        from pynerve.torch import diagram_bottleneck, diagram_wasserstein

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        w_inf = diagram_wasserstein(d, d, p=1.0, q=float("inf"))
        b = diagram_bottleneck(d, d)
        wv = w_inf if isinstance(w_inf, torch.Tensor) else torch.tensor(w_inf)
        bv = b if isinstance(b, torch.Tensor) else torch.tensor(b)
        assert wv.item() == pytest.approx(bv.item(), abs=1e-6)


# torch C++ backend forward passes


class TestTorchBackendForward:
    """Forward passes through Torch C++ extension."""

    @_torch_c
    def test_vr_persistence(self) -> None:
        from pynerve.torch import vr_persistence

        points = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        diagram = vr_persistence(points, max_dim=1, max_radius=2.0)
        assert diagram.diagrams.shape[-1] == 3

    @_torch_c
    def test_alpha_persistence(self) -> None:
        from pynerve.torch import alpha_persistence

        points = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        diagram = alpha_persistence(points, max_dim=1)
        assert diagram.diagrams.shape[-1] == 3

    @_torch_c
    def test_witness_persistence(self) -> None:
        from pynerve.torch import witness_persistence

        landmarks = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        witnesses = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        diagram = witness_persistence(landmarks, witnesses, max_dim=1, max_radius=2.0)
        assert diagram.diagrams.shape[-1] == 3

    @_torch_c
    def test_persistence_from_matrix(self) -> None:
        from pynerve.torch import persistence_from_matrix

        D = torch.tensor(
            [[[0.0, 1.0, 1.0], [1.0, 0.0, 1.414], [1.0, 1.414, 0.0]]],
            dtype=torch.float32,
        )
        diagram = persistence_from_matrix(D, max_dim=1)
        assert diagram.diagrams.shape[-1] == 3

    @_torch_c
    def test_persistence_image(self) -> None:
        from pynerve.torch import persistence_image

        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        img = persistence_image(diagram, resolution=(4, 4), sigma=0.5)
        assert img.shape == (4, 4)

    @_torch_c
    def test_wasserstein_self_zero(self) -> None:
        from pynerve.torch import diagram_wasserstein

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        dist = diagram_wasserstein(d, d)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-6)
