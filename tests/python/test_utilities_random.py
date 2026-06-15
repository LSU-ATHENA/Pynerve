"""Tests for random utility module."""

from __future__ import annotations

import numpy as np
import pytest
from pynerve.exceptions import InvalidArgumentError
from pynerve.random import PRNGKey, ReproducibleContext, key, manual_seed, next_key, seed, split

# random.py


class TestPRNGKeyCreation:
    def test_default_counter_is_zero(self):
        k = PRNGKey(42)
        assert k.counter == 0

    def test_explicit_counter(self):
        k = PRNGKey(42, counter=5)
        assert k.counter == 5

    def test_negative_counter_raises(self):
        with pytest.raises(InvalidArgumentError, match="counter"):
            PRNGKey(42, counter=-1)

    def test_seed_must_be_integral(self):
        with pytest.raises(InvalidArgumentError, match="seed"):
            PRNGKey("not_int")  # type: ignore[arg-type]

    def test_counter_must_be_integral(self):
        with pytest.raises(InvalidArgumentError, match="counter"):
            PRNGKey(42, counter=3.14)  # type: ignore[arg-type]

    def test_bool_seed_rejected(self):
        with pytest.raises(InvalidArgumentError, match="seed"):
            PRNGKey(True)  # type: ignore[arg-type]

    def test_repr(self):
        k = PRNGKey(42, counter=7)
        assert "42" in repr(k)
        assert "7" in repr(k)


class TestPRNGKeySplit:
    def test_split_default_n(self):
        k = PRNGKey(42)
        keys = k.split()
        assert len(keys) == 2
        assert all(isinstance(sub, PRNGKey) for sub in keys)

    def test_split_custom_n(self):
        k = PRNGKey(42)
        keys = k.split(n=5)
        assert len(keys) == 5

    def test_split_produces_different_keys(self):
        k = PRNGKey(42)
        k1, k2 = k.split()
        assert k1.seed != k2.seed

    def test_split_produces_keys_with_counter_zero(self):
        k = PRNGKey(42)
        keys = k.split(n=3)
        for sub in keys:
            assert sub.counter == 0

    def test_split_of_split(self):
        k = PRNGKey(42)
        k1, k2 = k.split()
        k1a, k1b = k1.split()
        assert len({k1a.seed, k1b.seed, k2.seed}) == 3
        assert k1a.seed != k1b.seed

    def test_split_deterministic(self):
        k = PRNGKey(42)
        keys_a = k.split(n=3)
        k = PRNGKey(42)
        keys_b = k.split(n=3)
        assert [s.seed for s in keys_a] == [s.seed for s in keys_b]

    def test_split_same_parent_different_children(self):
        k = PRNGKey(42)
        children_a = k.split(n=4)
        k = PRNGKey(42)
        children_b = k.split(n=4)
        assert len(children_a) == 4
        assert len(children_b) == 4
        assert children_a[0].seed == children_b[0].seed

    def test_large_split(self):
        k = PRNGKey(42)
        keys = k.split(n=100)
        assert len(keys) == 100
        assert len({sub.seed for sub in keys}) == 100

    def test_split_n_one(self):
        k = PRNGKey(42)
        keys = k.split(n=1)
        assert len(keys) == 1
        assert keys[0].counter == 0

    def test_split_zero_raises(self):
        from pynerve.exceptions import ValidationError

        k = PRNGKey(42)
        with pytest.raises(ValidationError, match="positive"):
            k.split(n=0)

    def test_iter_unpacks(self):
        k = PRNGKey(42)
        a, b = k
        assert isinstance(a, PRNGKey)
        assert isinstance(b, PRNGKey)
        assert a.seed != b.seed


class TestPRNGKeyUniform:
    def test_default_shape(self):
        k = PRNGKey(42)
        result = k.uniform()
        assert result.shape == (1,)

    def test_custom_shape(self):
        k = PRNGKey(42)
        result = k.uniform(shape=(10, 3))
        assert result.shape == (10, 3)

    def test_bounded_within_range(self):
        k = PRNGKey(42)
        result = k.uniform(low=-5.0, high=5.0, shape=(1000,))
        assert result.min() >= -5.0
        assert result.max() <= 5.0

    def test_default_range_0_to_1(self):
        k = PRNGKey(42)
        result = k.uniform(shape=(500,))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_high_lt_low_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="high"):
            k.uniform(low=5.0, high=1.0)

    def test_deterministic_same_key(self):
        k = PRNGKey(42)
        a = k.uniform(shape=(100,))
        k = PRNGKey(42)
        b = k.uniform(shape=(100,))
        assert np.array_equal(a, b)

    def test_different_keys_different_output(self):
        k1 = PRNGKey(42)
        k2 = PRNGKey(99)
        a = k1.uniform(shape=(50,))
        b = k2.uniform(shape=(50,))
        assert not np.array_equal(a, b)

    def test_different_counters_different_output(self):
        k1 = PRNGKey(42, counter=0)
        k2 = PRNGKey(42, counter=1)
        a = k1.uniform(shape=(30,))
        b = k2.uniform(shape=(30,))
        assert not np.array_equal(a, b)


class TestPRNGKeyNormal:
    def test_default_shape(self):
        k = PRNGKey(42)
        result = k.normal()
        assert result.shape == (1,)

    def test_custom_shape(self):
        k = PRNGKey(42)
        result = k.normal(shape=(5, 4))
        assert result.shape == (5, 4)

    def test_deterministic(self):
        k = PRNGKey(42)
        a = k.normal(shape=(100,))
        k = PRNGKey(42)
        b = k.normal(shape=(100,))
        assert np.array_equal(a, b)

    def test_dtype(self):
        k = PRNGKey(42)
        result = k.normal(shape=(10,), dtype=np.float32)
        assert result.dtype == np.float32


class TestPRNGKeyRandint:
    def test_single_value(self):
        k = PRNGKey(42)
        result = k.randint(0, 10, size=None)
        assert isinstance(result, int)
        assert 0 <= result < 10

    def test_scalar_low_only(self):
        k = PRNGKey(42)
        result = k.randint(5, size=None)
        assert isinstance(result, int)
        assert 0 <= result < 5

    def test_array_shape(self):
        k = PRNGKey(42)
        result = k.randint(0, 100, size=(20,))
        assert result.shape == (20,)
        assert result.min() >= 0
        assert result.max() < 100

    def test_deterministic(self):
        k = PRNGKey(42)
        a = k.randint(0, 50, size=(30,))
        k = PRNGKey(42)
        b = k.randint(0, 50, size=(30,))
        assert np.array_equal(a, b)

    def test_low_zero_omitted_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="positive"):
            k.randint(0)

    def test_high_le_low_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="high"):
            k.randint(5, 3)


class TestPRNGKeyChoice:
    def test_from_int(self):
        k = PRNGKey(42)
        result = k.choice(10, size=5)
        assert result.shape == (5,)
        assert result.min() >= 0
        assert result.max() < 10

    def test_from_sequence(self):
        k = PRNGKey(42)
        items = ["a", "b", "c", "d", "e"]
        result = k.choice(items, size=10)
        assert result.shape == (10,)
        assert all(x in items for x in result)

    def test_without_replacement(self):
        k = PRNGKey(42)
        result = k.choice(10, size=5, replace=False)
        assert result.shape == (5,)
        assert len(set(result.tolist())) == 5

    def test_deterministic(self):
        k = PRNGKey(42)
        a = k.choice(100, size=20)
        k = PRNGKey(42)
        b = k.choice(100, size=20)
        assert np.array_equal(a, b)

    def test_empty_sequence_raises(self):
        k = PRNGKey(42)
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            k.choice([])


class TestPRNGKeyPermutation:
    def test_from_int(self):
        k = PRNGKey(42)
        result = k.permutation(10)
        assert result.shape == (10,)
        assert set(result.tolist()) == set(range(10))

    def test_from_array(self):
        k = PRNGKey(42)
        arr = np.array([10, 20, 30, 40, 50])
        result = k.permutation(arr)
        assert result.shape == (5,)
        assert sorted(result.tolist()) == [10, 20, 30, 40, 50]

    def test_deterministic(self):
        k = PRNGKey(42)
        a = k.permutation(20)
        k = PRNGKey(42)
        b = k.permutation(20)
        assert np.array_equal(a, b)


class TestSeedAndGlobalKey:
    def test_seed_sets_global_key(self):
        k = seed(42)
        assert k.seed == 42
        assert key().seed == 42

    def test_seed_same_value_returns_same_key(self):
        k1 = seed(42)
        k2 = seed(42)
        assert k1.seed == k2.seed

    def test_seed_different_values_different_keys(self):
        k1 = seed(42)
        k2 = seed(99)
        assert k1.seed != k2.seed

    def test_seed_none_generates_random(self):
        k = seed(None)
        assert isinstance(k.seed, int)

    def test_manual_seed_is_alias(self):
        k1 = seed(42)
        # Reset before manual_seed to confirm it sets the same thing
        k2 = manual_seed(42)
        assert k1.seed == k2.seed

    def test_key_creates_default_when_none(self):
        # Force global reset by seeding with None then calling key
        seed(12345)
        k = key()
        assert isinstance(k, PRNGKey)

    def test_split_module_function(self):
        seed(42)
        keys = split(4)
        assert len(keys) == 4
        assert all(isinstance(k, PRNGKey) for k in keys)

    def test_next_key_returns_different_key(self):
        seed(42)
        k0 = key()
        k1 = next_key()
        assert k0.seed != k1.seed


class TestReproducibleContext:
    def test_context_restores_global_key(self):
        seed(42)
        old = key()
        with ReproducibleContext(999):
            assert key().seed == 999
        assert key().seed == old.seed

    def test_context_restores_on_exception(self):
        seed(42)
        old = key()
        try:
            with ReproducibleContext(888):
                assert key().seed == 888
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert key().seed == old.seed

    def test_reproducible_is_alias(self):
        from pynerve.random import reproducible

        seed(42)
        old = key()
        with reproducible(777):
            assert key().seed == 777
        assert key().seed == old.seed

    def test_repr(self):
        ctx = ReproducibleContext(42)
        assert "42" in repr(ctx)
