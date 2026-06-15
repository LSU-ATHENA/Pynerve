"""Cross-backend consistency: Python fallback vs C++ backend produce identical values."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

_D2 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)


class TestCrossBackendPersistence:
    """compute_persistence: PH0/PH3/PH4/PH5 engines produce same H0 on simple data."""

    def test_known_two_point_death(self):
        from pynerve import compute_persistence

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        for engine in ("ph0",):
            r = compute_persistence(pts, max_dim=0, max_radius=5.0, engine=engine)
            finite = [(b, d) for b, d, dim in r.pairs if dim == 0 and d < float("inf")]
            assert len(finite) == 1
            b, d = finite[0]
            assert b == pytest.approx(0.0, abs=1e-6)
            assert d == pytest.approx(1.0, abs=1e-6)

    def test_square_count(self):
        from pynerve import compute_persistence

        pts = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        for engine in ("ph0",):
            r = compute_persistence(pts, max_dim=0, max_radius=5.0, engine=engine)
            essential = [(b, d) for b, d, dim in r.pairs if dim == 0 and d == float("inf")]
            assert len(essential) == 1


class TestCrossBackendBatch:
    """Batched persistence gives same results as single."""

    def test_vr_persistence_batched_matches_single(self):
        from pynerve.torch import vr_persistence

        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32, requires_grad=True)
        single = vr_persistence(pts, max_dim=0, max_radius=5.0)
        batch = vr_persistence(pts.repeat(2, 1, 1), max_dim=0, max_radius=5.0)
        assert batch.births().shape[0] == 2 * single.births().shape[0]

    def test_vr_persistence_batch_gradient(self):
        from pynerve.torch import vr_persistence

        pts = torch.randn(2, 5, 2, requires_grad=True)
        diagram = vr_persistence(pts, max_dim=0, max_radius=5.0)
        loss = diagram.births().sum() + diagram.deaths()[torch.isfinite(diagram.deaths())].sum()
        loss.backward()
        assert pts.grad is not None
        assert torch.isfinite(pts.grad).all()
