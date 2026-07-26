"""Tests for pynerve/random.py -- PRNGKey, global RNG state, reproducible context, samplers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pynerve.random import (
    PRNGKey,
    ReproducibleContext,
    key,
    manual_seed,
    next_key,
    reproducible,
    seed,
    split,
)
from pynerve.exceptions import InvalidArgumentError


class TestPRNGKeyConstruction:
    def test_default_counter(self):
        k = PRNGKey(42)
        assert k.seed == 42
        assert k.counter == 0

    def test_explicit_counter(self):
        k = PRNGKey(42, 5)
        assert k.seed == 42
        assert k.counter == 5

    def test_repr(self):
        k = PRNGKey(42, 3)
        assert "PRNGKey" in repr(k)
        assert "42" in repr(k)
        assert "3" in repr(k)

    def test_bool_raises(self):
        with pytest.raises(InvalidArgumentError, match="must be an integer"):
            PRNGKey(True)  # type: ignore[arg-type]

    def test_float_raises(self):
        with pytest.raises(InvalidArgumentError, match="must be an integer"):
            PRNGKey(3.14)  # type: ignore[arg-type]

    def test_negative_counter_raises(self):
        with pytest.raises(InvalidArgumentError, match="must be non-negative"):
            PRNGKey(42, -1)

    def test_string_seed_raises(self):
        with pytest.raises(InvalidArgumentError, match="must be an integer"):
            PRNGKey("bad")  # type: ignore[arg-type]


class TestPRNGKeySplit:
    def test_default_n(self):
        k = PRNGKey(42)
        keys = k.split()
        assert len(keys) == 2
        assert all(isinstance(child, PRNGKey) for child in keys)

    def test_custom_n(self):
        k = PRNGKey(42)
        keys = k.split(4)
        assert len(keys) == 4

    def test_deterministic(self):
        k1 = PRNGKey(42)
        k2 = PRNGKey(42)
        assert k1.split(3)[0].seed == k2.split(3)[0].seed

    def test_negative_n_raises(self):
        k = PRNGKey(42)
        with pytest.raises(Exception, match="positive"):
            k.split(0)

    def test_unpack(self):
        k = PRNGKey(42)
        a, b = k.split()
        assert a.seed != b.seed

    def test_zero_counter_in_children(self):
        k = PRNGKey(42, 99)
        children = k.split(3)
        assert all(c.counter == 0 for c in children)


class TestPRNGKeyNormal:
    def test_default_shape(self):
        k = PRNGKey(42)
        result = k.normal()
        assert result.shape == (1,)

    def test_custom_shape(self):
        k = PRNGKey(42)
        result = k.normal((3, 4))
        assert result.shape == (3, 4)

    def test_integer_shape(self):
        k = PRNGKey(42)
        result = k.normal(5)
        assert result.shape == (5,)

    def test_as_tensor(self):
        k = PRNGKey(42)
        result = k.normal(as_tensor=True)
        assert isinstance(result, torch.Tensor)

    def test_deterministic_same_seed(self):
        k1 = PRNGKey(42)
        k2 = PRNGKey(42)
        r1 = k1.normal((10,))
        r2 = k2.normal((10,))
        assert np.allclose(r1, r2)

    def test_different_counter_different_result(self):
        k1 = PRNGKey(42, 0)
        k2 = PRNGKey(42, 1)
        r1 = k1.normal((100,))
        r2 = k2.normal((100,))
        assert not np.allclose(r1, r2)

    def test_dtype_float32(self):
        k = PRNGKey(42)
        result = k.normal((5,), dtype=np.float32)
        assert result.dtype == np.float32


class TestPRNGKeyUniform:
    def test_default_bounds(self):
        k = PRNGKey(42)
        result = k.uniform()
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0)

    def test_custom_bounds(self):
        k = PRNGKey(42)
        result = k.uniform(low=5.0, high=10.0)
        assert np.all(result >= 5.0)
        assert np.all(result < 10.0)

    def test_high_less_than_low_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="greater than or equal"):
            k.uniform(low=10.0, high=5.0)

    def test_as_tensor(self):
        k = PRNGKey(42)
        result = k.uniform(as_tensor=True)
        assert isinstance(result, torch.Tensor)


class TestPRNGKeyRandint:
    def test_single_low_high(self):
        k = PRNGKey(42)
        result = k.randint(0, 10)
        assert isinstance(result, np.ndarray)
        assert 0 <= result.item() < 10

    def test_low_only(self):
        k = PRNGKey(42)
        result = k.randint(5)
        assert 0 <= result < 5

    def test_low_only_zero_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="positive"):
            k.randint(0)

    def test_high_less_than_low_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="greater than low"):
            k.randint(10, 5)

    def test_with_size(self):
        k = PRNGKey(42)
        result = k.randint(0, 10, size=(3, 4))
        assert result.shape == (3, 4)


class TestPRNGKeyChoice:
    def test_from_range(self):
        k = PRNGKey(42)
        result = k.choice(10, size=5)
        assert result.shape == (5,)
        assert np.all(result >= 0)
        assert np.all(result < 10)

    def test_from_sequence(self):
        k = PRNGKey(42)
        result = k.choice([10, 20, 30], size=2)
        assert result.shape == (2,)

    def test_empty_sequence_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            k.choice([])

    def test_non_bool_replace_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="boolean"):
            k.choice(5, replace=1)  # type: ignore[arg-type]

    def test_p_bad_shape_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="shape"):
            k.choice(5, p=np.array([0.2, 0.8]))

    def test_p_negative_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="non-negative"):
            k.choice(3, p=np.array([-0.1, 0.6, 0.5]))

    def test_p_not_sum_one_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="sum to 1"):
            k.choice(3, p=np.array([0.1, 0.2, 0.3]))

    def test_p_non_finite_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="finite"):
            k.choice(3, p=np.array([np.nan, 0.5, 0.5]))


class TestPRNGKeyPermutation:
    def test_from_range(self):
        k = PRNGKey(42)
        result = k.permutation(10)
        assert len(result) == 10
        assert set(result) == set(range(10))

    def test_from_array(self):
        k = PRNGKey(42)
        arr = np.array([1.0, 2.0, 3.0])
        result = k.permutation(arr)
        assert len(result) == 3

    def test_as_tensor(self):
        k = PRNGKey(42)
        result = k.permutation(10, as_tensor=True)
        assert isinstance(result, torch.Tensor)


class TestGlobalSeed:
    def test_seed_with_int(self):
        k = seed(42)
        assert isinstance(k, PRNGKey)
        assert k.seed == 42

    def test_seed_with_none(self):
        k = seed(None)
        assert isinstance(k, PRNGKey)

    def test_manual_seed(self):
        k = manual_seed(123)
        assert isinstance(k, PRNGKey)
        assert k.seed == 123

    def test_key_after_seed(self):
        seed(42)
        k = key()
        assert k.seed == 42

    def test_key_auto_init(self):
        # ensure _global_key starts fresh by calling key() without seed()
        import pynerve.random as rmod
        rmod._global_key = None
        k = key()
        assert isinstance(k, PRNGKey)


class TestGlobalSplit:
    def test_split_default(self):
        seed(42)
        keys = split()
        assert len(keys) == 2

    def test_next_key(self):
        seed(42)
        nk = next_key()
        assert isinstance(nk, PRNGKey)


class TestReproducibleContext:
    def test_enter_exit(self):
        seed(99)
        old = key()
        with ReproducibleContext(42) as ctx_key:
            assert key().seed == 42
            assert isinstance(ctx_key, PRNGKey)
        assert key().seed == old.seed

    def test_repr(self):
        ctx = ReproducibleContext(7)
        assert "7" in repr(ctx)

    def test_alias(self):
        assert reproducible is ReproducibleContext

    def test_exception_restores(self):
        seed(99)
        old = key()
        try:
            with ReproducibleContext(42):
                raise ValueError("boom")
        except ValueError:
            pass
        assert key().seed == old.seed
