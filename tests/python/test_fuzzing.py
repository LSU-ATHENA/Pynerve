"""Fuzzing tests: random valid diagrams must not crash."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

_NUMS = [0, 1, 2, 5, 10, 50]


def _rd(n):
    births = torch.randn(n) * 10
    deaths = births + torch.rand(n).abs() * 10 + 0.001
    return torch.stack([births, deaths], dim=1)


class TestFuzzStatistics:
    @pytest.mark.parametrize("num", _NUMS)
    def test_total_persistence(self, num):
        from pynerve.torch.statistics import total_persistence

        r = total_persistence(_rd(num))
        assert torch.isfinite(r)

    @pytest.mark.parametrize("num", _NUMS)
    def test_mean_persistence(self, num):
        from pynerve.torch.statistics import mean_persistence

        r = mean_persistence(_rd(num))
        assert torch.isfinite(r)

    @pytest.mark.parametrize("num", _NUMS)
    def test_max_persistence(self, num):
        from pynerve.torch.statistics import max_persistence

        r = max_persistence(_rd(num))
        assert torch.isfinite(r)

    @pytest.mark.parametrize("num", _NUMS)
    def test_entropy(self, num):
        from pynerve.torch.statistics import persistence_entropy

        r = persistence_entropy(_rd(num))
        assert torch.isfinite(r)

    @pytest.mark.parametrize("num", _NUMS)
    def test_count(self, num):
        from pynerve.torch.statistics import number_of_features

        r = number_of_features(_rd(num), min_persistence=0.0)
        assert 0 <= r.item() <= max(1, num)

    @pytest.mark.parametrize("num", _NUMS)
    def test_amplitude(self, num):
        from pynerve.torch.statistics import amplitude

        r = amplitude(_rd(num), metric="persistence")
        assert r.item() >= 0


class TestFuzzVectorization:
    @pytest.mark.parametrize("num", _NUMS)
    def test_image(self, num):
        from pynerve.torch.vectorization import persistence_image

        r = persistence_image(_rd(num), resolution=(4, 4), sigma=1.0)
        assert torch.isfinite(r).all()

    @pytest.mark.parametrize("num", _NUMS)
    def test_landscape(self, num):
        from pynerve.torch.vectorization import persistence_landscape

        r = persistence_landscape(_rd(num), k=2, num_samples=5)
        assert torch.isfinite(r).all()

    @pytest.mark.parametrize("num", _NUMS)
    def test_silhouette(self, num):
        from pynerve.torch.vectorization import persistence_silhouette

        r = persistence_silhouette(_rd(num), num_samples=5)
        assert torch.isfinite(r).all()

    @pytest.mark.parametrize("num", _NUMS)
    def test_curve(self, num):
        from pynerve.torch.vectorization import birth_death_curve

        r = birth_death_curve(_rd(num), num_bins=4, statistic="count")
        assert torch.isfinite(r).all()

    @pytest.mark.parametrize("num", _NUMS)
    def test_heat(self, num):
        from pynerve.torch.vectorization import heat_kernel_signature

        r = heat_kernel_signature(_rd(num), num_samples=5, sigma=0.5)
        assert torch.isfinite(r).all()


class TestFuzzDistances:
    @pytest.mark.parametrize("num", [0, 1, 2, 5, 10])
    def test_w_same(self, num):
        from pynerve.torch import diagram_wasserstein as fn

        d = _rd(num)
        r = fn(d, d)
        v = r if isinstance(r, torch.Tensor) else torch.tensor(r)
        assert v.item() >= 0

    @pytest.mark.parametrize("num", [0, 1, 2, 5, 10])
    def test_w_cross(self, num):
        from pynerve.torch import diagram_wasserstein as fn

        r = fn(_rd(num), _rd(num))
        v = r if isinstance(r, torch.Tensor) else torch.tensor(r)
        assert torch.isfinite(v)

    @pytest.mark.parametrize("num", [0, 1, 2, 5, 10])
    def test_b_same(self, num):
        from pynerve.torch import diagram_bottleneck as fn

        d = _rd(num)
        r = fn(d, d)
        v = r if isinstance(r, torch.Tensor) else torch.tensor(r)
        assert v.item() >= 0

    @pytest.mark.parametrize("num", [0, 1, 2, 5, 10])
    def test_b_cross(self, num):
        from pynerve.torch import diagram_bottleneck as fn

        r = fn(_rd(num), _rd(num))
        v = r if isinstance(r, torch.Tensor) else torch.tensor(r)
        assert torch.isfinite(v)
