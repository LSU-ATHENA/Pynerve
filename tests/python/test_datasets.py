"""Comprehensive correctness tests for pynerve.datasets.

Covers all dataset generators: shape, manifold properties, edge cases,
reproducibility, tensor output, and __all__ completeness.
"""

from __future__ import annotations

import numpy as np
import pytest
from pynerve.datasets import (
    load_circle,
    load_klein_bottle,
    load_mobius_strip,
    load_sphere,
    load_swiss_roll,
    load_torus,
)
from pynerve.exceptions import ValidationError

# Test data: all loaders with their expected output dimensions and kwargs

LOADERS = {
    "swiss_roll": (load_swiss_roll, 3, {}),
    "mobius_strip": (load_mobius_strip, 3, {}),
    "klein_bottle": (load_klein_bottle, 4, {}),
    "torus": (load_torus, 3, {}),
    "sphere": (load_sphere, 3, {"dim": 2}),
    "circle": (load_circle, 2, {}),
}


def _get_loader_info(name):
    return LOADERS[name]


# Basic shape and finiteness


class TestBasicOutput:
    @pytest.mark.parametrize("name", sorted(LOADERS))
    @pytest.mark.parametrize("n_samples", [1, 10, 100])
    def test_output_shape(self, name, n_samples):
        fn, ndim, kwargs = _get_loader_info(name)
        data = fn(n_samples=n_samples, seed=42, **kwargs)
        assert data.shape == (n_samples, ndim)

    @pytest.mark.parametrize("name", sorted(LOADERS))
    @pytest.mark.parametrize("n_samples", [1, 50])
    def test_output_is_finite(self, name, n_samples):
        fn, ndim, kwargs = _get_loader_info(name)
        data = fn(n_samples=n_samples, seed=42, **kwargs)
        assert np.isfinite(data).all(), f"{name}: non-finite values detected"

    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_output_dtype_is_float64(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        data = fn(n_samples=30, seed=7, **kwargs)
        assert data.dtype == np.float64

    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_default_n_samples(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        data = fn(**kwargs)
        assert data.shape == (100, ndim)


# Manifold geometry verification


class TestManifoldGeometry:
    def test_circle_lies_on_circle(self):
        radius = 3.0
        data = load_circle(n_samples=500, radius=radius, noise=0.0, seed=1)
        distances = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        assert distances == pytest.approx(radius, abs=1e-12)

    def test_circle_with_noise_is_near_circle(self):
        radius = 2.5
        noise = 0.1
        data = load_circle(n_samples=500, radius=radius, noise=noise, seed=2)
        distances = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        # With noise=0.1 and 500 points, max deviation ~ 5*noise is safe
        assert (distances >= radius - 5 * noise).all()
        assert (distances <= radius + 5 * noise).all()

    def test_sphere_lies_on_sphere(self):
        radius = 2.0
        data = load_sphere(n_samples=500, dim=3, radius=radius, seed=3)
        distances = np.linalg.norm(data, axis=1)
        assert distances == pytest.approx(radius, abs=1e-10)

    def test_sphere_2d_lives_in_3d(self):
        data = load_sphere(n_samples=100, dim=2, seed=4)
        assert data.shape == (100, 3)
        distances = np.linalg.norm(data, axis=1)
        assert distances == pytest.approx(1.0, abs=1e-10)

    def test_sphere_different_dims(self):
        for dim in (1, 3, 5):
            data = load_sphere(n_samples=50, dim=dim, seed=5)
            assert data.shape == (50, dim + 1)
            distances = np.linalg.norm(data, axis=1)
            assert distances == pytest.approx(1.0, abs=1e-10)

    def test_torus_invariant(self):
        major, minor = 3.0, 1.5
        data = load_torus(n_samples=800, major_radius=major, minor_radius=minor, seed=6)
        xy_dist = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        # (sqrt(x^2+y^2) - R)^2 + z^2 = r^2
        invariant = (xy_dist - major) ** 2 + data[:, 2] ** 2
        assert invariant == pytest.approx(minor**2, abs=1e-9)

    def test_torus_default_radii(self):
        data = load_torus(n_samples=500, seed=7)
        xy_dist = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        invariant = (xy_dist - 2.0) ** 2 + data[:, 2] ** 2
        assert invariant == pytest.approx(1.0**2, abs=1e-9)

    def test_swiss_roll_parametric(self):
        data = load_swiss_roll(n_samples=500, seed=8)
        # x = t * cos(t), z = t * sin(t), so sqrt(x^2+z^2) = |t|
        # t = 1.5*pi * (1 + 2*U) where U in (0,1) => t in [1.5*pi, 4.5*pi)
        t_observed = np.sqrt(data[:, 0] ** 2 + data[:, 2] ** 2)
        t_min = 1.5 * np.pi * 0.9  # lower bound with tolerance
        t_max = 1.5 * np.pi * 3 * 1.1  # upper bound with tolerance
        assert np.all(t_observed >= t_min)
        assert np.all(t_observed <= t_max)

    def test_mobius_strip_parametric(self):
        data = load_mobius_strip(n_samples=1000, seed=9)
        # v small => x ~ cos(u), y ~ sin(u), z ~ 0 => x^2 + y^2 ~ 1
        near_center = np.abs(data[:, 2]) < 0.05
        assert near_center.sum() >= 20, "expected at least 20 points near center strip"
        r_center = np.sqrt(data[near_center, 0] ** 2 + data[near_center, 1] ** 2)
        assert r_center.max() <= 1.5
        assert r_center.min() >= 0.5

    def test_klein_bottle_4d_shape(self):
        data = load_klein_bottle(n_samples=100, seed=10)
        assert data.shape == (100, 4)
        assert np.isfinite(data).all()


# Reproducibility: same seed -> same points


class TestReproducibility:
    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_same_seed_same_output(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        data1 = fn(n_samples=50, seed=42, **kwargs)
        data2 = fn(n_samples=50, seed=42, **kwargs)
        np.testing.assert_array_equal(data1, data2)

    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_different_seeds_different_output(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        data1 = fn(n_samples=100, seed=1, **kwargs)
        data2 = fn(n_samples=100, seed=2, **kwargs)
        assert not np.allclose(data1, data2), f"{name}: identical across seeds"

    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_none_seed_is_nondeterministic(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        data1 = fn(n_samples=100, seed=None, **kwargs)
        data2 = fn(n_samples=100, seed=None, **kwargs)
        assert not np.allclose(data1, data2), f"{name}: None seed produced same output"


# Edge cases


class TestEdgeCases:
    def test_n_samples_zero_raises(self):
        for fn, _ndim, kwargs in LOADERS.values():
            with pytest.raises(ValidationError, match="positive"):
                fn(n_samples=0, seed=0, **kwargs)

    def test_n_samples_one(self):
        for fn, ndim, kwargs in LOADERS.values():
            data = fn(n_samples=1, seed=0, **kwargs)
            assert data.shape == (1, ndim)
            assert np.isfinite(data).all()

    def test_sphere_zero_radius(self):
        data = load_sphere(n_samples=10, dim=2, radius=0.0, seed=0)
        assert data.shape == (10, 3)
        assert np.allclose(data, 0.0)

    def test_sphere_zero_dim(self):
        data = load_sphere(n_samples=20, dim=0, radius=0.0, seed=0)
        assert data.shape == (20, 1)

    def test_torus_zero_minor_radius(self):
        data = load_torus(n_samples=100, major_radius=3.0, minor_radius=0.0, seed=0)
        xy_dist = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        assert xy_dist == pytest.approx(3.0, abs=1e-12)
        assert np.allclose(data[:, 2], 0.0)

    def test_torus_zero_major_radius(self):
        data = load_torus(n_samples=100, major_radius=0.0, minor_radius=2.0, seed=0)
        norm = np.linalg.norm(data, axis=1)
        assert norm == pytest.approx(2.0, abs=1e-10)

    def test_circle_zero_radius(self):
        data = load_circle(n_samples=50, radius=0.0, noise=0.0, seed=0)
        assert np.allclose(data, 0.0, atol=1e-12)

    def test_circle_zero_radius_with_noise(self):
        noise = 0.5
        data = load_circle(n_samples=200, radius=0.0, noise=noise, seed=0)
        distances = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        # With radius=0 and Gaussian noise(0, 0.5), 200 points stay within ~5*noise
        assert distances.max() <= 5 * noise

    def test_large_n_samples(self):
        data = load_circle(n_samples=20000, radius=1.0, noise=0.0, seed=0)
        assert data.shape == (20000, 2)
        assert np.isfinite(data).all()

    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_negative_n_samples_raises(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        with pytest.raises(ValidationError, match="positive"):
            fn(n_samples=-1, **kwargs)

    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_non_integer_n_samples_raises(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        with pytest.raises(ValidationError, match="integer"):
            fn(n_samples=1.5, **kwargs)

    def test_sphere_negative_radius_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            load_sphere(n_samples=10, dim=2, radius=-1.0)

    def test_sphere_negative_dim_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            load_sphere(n_samples=10, dim=-1)

    def test_circle_negative_radius_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            load_circle(n_samples=10, radius=-0.5)

    def test_circle_negative_noise_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            load_circle(n_samples=10, noise=-0.1)

    def test_circle_nan_radius_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            load_circle(n_samples=10, radius=float("nan"))

    def test_circle_inf_radius_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            load_circle(n_samples=10, radius=float("inf"))

    def test_torus_negative_major_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            load_torus(n_samples=10, major_radius=-1.0)

    def test_torus_negative_minor_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            load_torus(n_samples=10, minor_radius=-0.5)

    def test_swiss_roll_zero_samples_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            load_swiss_roll(n_samples=0)


# Torch tensor output


class TestTensorOutput:
    @pytest.mark.cpu
    @pytest.mark.torch
    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_as_tensor_returns_tensor(self, name, torch):
        fn, ndim, kwargs = _get_loader_info(name)
        data = fn(n_samples=30, seed=42, as_tensor=True, **kwargs)
        assert isinstance(data, torch.Tensor)

    @pytest.mark.cpu
    @pytest.mark.torch
    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_tensor_shape_and_dtype(self, name, torch):
        fn, ndim, kwargs = _get_loader_info(name)
        data = fn(n_samples=30, seed=42, as_tensor=True, **kwargs)
        assert data.shape == (30, ndim)
        assert data.dtype == torch.float64

    @pytest.mark.cpu
    @pytest.mark.torch
    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_tensor_is_on_cpu(self, name):
        torch = pytest.importorskip("torch")
        fn, ndim, kwargs = _get_loader_info(name)
        data = fn(n_samples=30, seed=42, as_tensor=True, **kwargs)
        assert isinstance(data, torch.Tensor), f"expected torch.Tensor, got {type(data)}"
        assert data.device.type == "cpu"

    @pytest.mark.cpu
    @pytest.mark.torch
    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_tensor_reproducibility(self, name, torch):
        fn, ndim, kwargs = _get_loader_info(name)
        t1 = fn(n_samples=30, seed=99, as_tensor=True, **kwargs)
        t2 = fn(n_samples=30, seed=99, as_tensor=True, **kwargs)
        assert torch.equal(t1, t2)

    @pytest.mark.cpu
    @pytest.mark.torch
    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_tensor_finite(self, name, torch):
        fn, ndim, kwargs = _get_loader_info(name)
        data = fn(n_samples=30, seed=42, as_tensor=True, **kwargs)
        assert torch.isfinite(data).all()

    @pytest.mark.cpu
    @pytest.mark.torch
    def test_tensor_circle_geometry(self, torch):
        data = load_circle(n_samples=500, radius=2.0, noise=0.0, seed=3, as_tensor=True)
        distances = torch.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        assert torch.allclose(distances, torch.tensor(2.0, dtype=torch.float64))

    def test_as_tensor_false_returns_numpy(self):
        for fn, _ndim, kwargs in LOADERS.values():
            data = fn(n_samples=5, seed=0, as_tensor=False, **kwargs)
            assert isinstance(data, np.ndarray)


# __all__ completeness


class TestAllList:
    def test_all_contains_all_public_loaders(self):
        from pynerve import datasets

        expected = {
            "load_swiss_roll",
            "load_mobius_strip",
            "load_klein_bottle",
            "load_torus",
            "load_sphere",
            "load_circle",
        }
        assert set(datasets.__all__) == expected

    def test_all_matches_importable_symbols(self):
        from pynerve import datasets

        for name in datasets.__all__:
            assert hasattr(datasets, name), f"{name} in __all__ but not importable"
            obj = getattr(datasets, name)
            assert callable(obj), f"{name} is not callable"


# Seed: invalid seeds raise errors from numpy (not validated by datasets)


class TestSeedValidation:
    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_negative_seed_raises(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        with pytest.raises(ValueError, match="non-negative"):
            fn(n_samples=10, seed=-1, **kwargs)

    @pytest.mark.parametrize("name", sorted(LOADERS))
    def test_float_seed_raises(self, name):
        fn, ndim, kwargs = _get_loader_info(name)
        with pytest.raises(TypeError, match="SeedSequence"):
            fn(n_samples=10, seed=3.14, **kwargs)
