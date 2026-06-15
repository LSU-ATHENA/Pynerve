"""Cross-dtype consistency tests.

Every major operation should produce the same result (within numeric precision)
regardless of whether inputs are float32 or float64.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _to(provider, dtype):
    return provider.clone().detach().to(dtype=dtype)


# shared test data
_F32_DIAGRAM = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float32)
_F64_DIAGRAM = _F32_DIAGRAM.to(dtype=torch.float64)
_F32_POINTS = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
_F64_POINTS = _F32_POINTS.to(dtype=torch.float64)


def _rtol(dtype):
    return 1e-5 if dtype == torch.float32 else 1e-12


class TestDistanceDtypeConsistency:
    """Wasserstein / bottleneck give same results in f32 and f64."""

    def test_wasserstein_self(self):
        from pynerve.torch import diagram_wasserstein as fn

        r32 = fn(_F32_DIAGRAM, _F32_DIAGRAM)
        r64 = fn(_F64_DIAGRAM, _F64_DIAGRAM)
        v32 = r32 if isinstance(r32, torch.Tensor) else torch.tensor(r32)
        v64 = r64 if isinstance(r64, torch.Tensor) else torch.tensor(r64)
        assert v32.item() == pytest.approx(v64.item(), rel=1e-3)

    def test_wasserstein_cross(self):
        from pynerve.torch import diagram_wasserstein as fn

        d2_f32 = torch.tensor([[0.5, 2.5]], dtype=torch.float32)
        d2_f64 = d2_f32.to(dtype=torch.float64)
        r32 = fn(_F32_DIAGRAM, d2_f32)
        r64 = fn(_F64_DIAGRAM, d2_f64)
        v32 = r32 if isinstance(r32, torch.Tensor) else torch.tensor(r32)
        v64 = r64 if isinstance(r64, torch.Tensor) else torch.tensor(r64)
        assert v32.item() == pytest.approx(v64.item(), rel=1e-3)

    def test_bottleneck_self(self):
        from pynerve.torch import diagram_bottleneck as fn

        r32 = fn(_F32_DIAGRAM, _F32_DIAGRAM)
        r64 = fn(_F64_DIAGRAM, _F64_DIAGRAM)
        v32 = r32 if isinstance(r32, torch.Tensor) else torch.tensor(r32)
        v64 = r64 if isinstance(r64, torch.Tensor) else torch.tensor(r64)
        assert v32.item() == pytest.approx(v64.item(), rel=1e-3)


class TestStatisticsDtypeConsistency:
    """Statistics functions produce equivalent f32/f64 results."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "total_persistence",
            "persistence_entropy",
            "mean_persistence",
            "max_persistence",
            "persistence_variance",
            "number_of_features",
        ],
    )
    def test_statistic(self, fn_name):
        import pynerve.torch.statistics as st

        fn = getattr(st, fn_name)

        r32 = fn(_F32_DIAGRAM)
        r64 = fn(_F64_DIAGRAM)
        v32 = r32.item() if hasattr(r32, "item") else r32
        v64 = r64.item() if hasattr(r64, "item") else r64
        assert v32 == pytest.approx(v64, rel=1e-3, abs=1e-4)


class TestVectorizationDtypeConsistency:
    """Vectorization functions produce equivalent f32/f64 results."""

    def test_persistence_image(self):
        from pynerve.torch.vectorization import persistence_image as fn

        img32 = fn(_F32_DIAGRAM, resolution=(4, 4), sigma=0.5, weight_fn="constant")
        img64 = fn(_F64_DIAGRAM, resolution=(4, 4), sigma=0.5, weight_fn="constant")
        assert torch.allclose(img32, img64.to(dtype=torch.float32), atol=1e-4)

    def test_persistence_landscape(self):
        from pynerve.torch.vectorization import persistence_landscape as fn

        xr: tuple[float, float] = (0.0, 4.0)
        l32 = fn(_F32_DIAGRAM, k=2, num_samples=8, x_range=xr)
        l64 = fn(_F64_DIAGRAM, k=2, num_samples=8, x_range=xr)
        assert torch.allclose(l32, l64.to(dtype=torch.float32), atol=1e-4)

    def test_persistence_silhouette(self):
        from pynerve.torch.vectorization import persistence_silhouette as fn

        s32 = fn(_F32_DIAGRAM, num_samples=8)
        s64 = fn(_F64_DIAGRAM, num_samples=8)
        assert torch.allclose(s32, s64.to(dtype=torch.float32), atol=1e-4)

    def test_birth_death_curve(self):
        from pynerve.torch.vectorization import birth_death_curve as fn

        c32 = fn(_F32_DIAGRAM, num_bins=4, statistic="count")
        c64 = fn(_F64_DIAGRAM, num_bins=4, statistic="count")
        assert torch.allclose(c32, c64.to(dtype=torch.float32))


class TestPreprocessingDtypeConsistency:
    """Preprocessing functions produce equivalent f32/f64 results."""

    def test_handle_infinite_deaths(self):
        from pynerve.torch.preprocessing import handle_infinite_deaths as fn

        d32 = torch.tensor([[0.0, float("inf")], [0.0, 2.0]], dtype=torch.float32)
        d64 = d32.to(dtype=torch.float64)
        r32 = fn(d32, strategy="max")
        r64 = fn(d64, strategy="max")
        assert torch.allclose(r32.to(dtype=torch.float64), r64, atol=1e-5)

    def test_normalize_diagram(self):
        from pynerve.torch.preprocessing import normalize_diagram as fn

        r32 = fn(_F32_DIAGRAM, method="minmax")
        r64 = fn(_F64_DIAGRAM, method="minmax")
        assert torch.allclose(r32.to(dtype=torch.float64), r64, atol=1e-5)

    def test_threshold_diagram(self):
        from pynerve.torch.preprocessing import threshold_diagram as fn

        r32 = fn(_F32_DIAGRAM, min_persistence=0.5)
        r64 = fn(_F64_DIAGRAM, min_persistence=0.5)
        assert r32.shape == r64.shape


class TestComputePersistenceDtypeConsistency:
    """Persistence computation produces equivalent results for f32/f64."""

    def test_vr_persistence_two_points(self):
        from pynerve import compute_persistence

        pts32 = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        pts64 = pts32.to(dtype=torch.float64)
        r32 = compute_persistence(pts32, max_dim=0, max_radius=5.0)
        r64 = compute_persistence(pts64, max_dim=0, max_radius=5.0)
        f32 = [(b, d) for b, d, dim in r32.pairs if dim == 0 and d < float("inf")]
        f64 = [(b, d) for b, d, dim in r64.pairs if dim == 0 and d < float("inf")]
        assert len(f32) == len(f64)
        for (b32, d32), (b64, d64) in zip(f32, f64, strict=False):
            assert b32 == pytest.approx(b64, abs=1e-5)
            assert d32 == pytest.approx(d64, abs=1e-5)
