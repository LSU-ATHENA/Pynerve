"""Numerical correctness tests for the PRNG module."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class TestPRNGKey:
    """Statistical correctness for PRNGKey."""

    def test_normal_output_shape(self) -> None:
        from pynerve.random import PRNGKey

        key = PRNGKey(42)
        result = key.normal(shape=(100,))
        assert result.shape == (100,)

    def test_normal_deterministic(self) -> None:
        from pynerve.random import PRNGKey

        k1 = PRNGKey(42)
        k2 = PRNGKey(42)
        r1 = k1.normal(shape=(50,))
        r2 = k2.normal(shape=(50,))
        np.testing.assert_array_equal(r1, r2)

    def test_normal_different_seeds_differ(self) -> None:
        from pynerve.random import PRNGKey

        k1 = PRNGKey(42)
        k2 = PRNGKey(99)
        r1 = k1.normal(shape=(50,))
        r2 = k2.normal(shape=(50,))
        assert not np.allclose(r1, r2)

    def test_uniform_bounds(self) -> None:
        from pynerve.random import PRNGKey

        key = PRNGKey(42)
        result = key.uniform(0.0, 1.0, shape=(1000,))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uniform_mean(self) -> None:
        from pynerve.random import PRNGKey

        key = PRNGKey(42)
        result = key.uniform(0.0, 1.0, shape=(10000,))
        assert abs(result.mean() - 0.5) < 0.02

    def test_randint_range(self) -> None:
        from pynerve.random import PRNGKey

        key = PRNGKey(42)
        result = key.randint(0, 10, size=(1000,))
        assert result.min() >= 0
        assert result.max() < 10

    def test_choice_reproducible(self) -> None:
        from pynerve.random import PRNGKey

        k1 = PRNGKey(42)
        k2 = PRNGKey(42)
        population = np.array([10, 20, 30, 40, 50])
        c1 = k1.choice(population, size=5)
        c2 = k2.choice(population, size=5)
        np.testing.assert_array_equal(c1, c2)

    def test_permutation_reproducible(self) -> None:
        from pynerve.random import PRNGKey

        k1 = PRNGKey(42)
        k2 = PRNGKey(42)
        p1 = k1.permutation(100)
        p2 = k2.permutation(100)
        np.testing.assert_array_equal(p1, p2)

    def test_split_deterministic(self) -> None:
        from pynerve.random import PRNGKey

        k1 = PRNGKey(42)
        sub1, sub2 = k1.split(2)
        k2 = PRNGKey(42)
        sub3, sub4 = k2.split(2)
        n1 = sub1.normal(shape=(10,))
        n3 = sub3.normal(shape=(10,))
        np.testing.assert_array_equal(n1, n3)
        n2 = sub2.normal(shape=(10,))
        n4 = sub4.normal(shape=(10,))
        np.testing.assert_array_equal(n2, n4)

    def test_normal_mean_std(self) -> None:
        from pynerve.random import PRNGKey

        key = PRNGKey(42)
        result = key.normal(shape=(10000,))
        assert abs(result.mean()) < 0.03
        assert abs(result.std() - 1.0) < 0.03


class TestManualSeed:
    """Numerical correctness for manual_seed."""

    def test_manual_seed_reproducible(self) -> None:
        from pynerve.random import PRNGKey, manual_seed

        manual_seed(42)
        k1 = PRNGKey(42)
        r1 = k1.normal(shape=(20,))
        manual_seed(42)
        k2 = PRNGKey(42)
        r2 = k2.normal(shape=(20,))
        np.testing.assert_array_equal(r1, r2)


class TestReproducible:
    """Numerical correctness for reproducible context."""

    def test_reproducible_context(self) -> None:
        from pynerve.random import reproducible

        with reproducible(42) as key1:
            r1 = key1.normal(shape=(30,))
        with reproducible(42) as key2:
            r2 = key2.normal(shape=(30,))
        np.testing.assert_array_equal(r1, r2)

    def test_reproducible_different_seeds(self) -> None:
        from pynerve.random import reproducible

        with reproducible(42) as key1:
            r1 = key1.normal(shape=(30,))
        with reproducible(99) as key2:
            r2 = key2.normal(shape=(30,))
        assert not np.allclose(r1, r2)


class TestSeed:
    """Numerical correctness for seed function."""

    def test_seed_deterministic(self) -> None:
        from pynerve.random import seed

        r1 = seed(42).normal(shape=(20,))
        r2 = seed(42).normal(shape=(20,))
        np.testing.assert_array_equal(r1, r2)
