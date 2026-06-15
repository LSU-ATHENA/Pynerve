"""Tests for caching and memoization."""

from __future__ import annotations

import numpy as np
import pytest
from pynerve.cache import (
    DiagramCache,
    MemoizePersistent,
    _validate_cache_key,
    cached_persistence,
    memoize_persistent,
)
from pynerve.exceptions import ValidationError


class TestValidateCacheKey:
    def test_accepts_valid_key(self):
        assert _validate_cache_key("my_key") == "my_key"

    def test_rejects_empty_key(self):
        with pytest.raises(ValidationError, match="non-empty"):
            _validate_cache_key("")

    def test_rejects_non_string(self):
        with pytest.raises(ValidationError, match="non-empty"):
            _validate_cache_key(123)


class TestDiagramCache:
    def test_init_with_defaults(self):
        cache = DiagramCache()
        assert cache.maxsize == 1024
        assert cache.ttl is None

    def test_init_with_custom_params(self):
        cache = DiagramCache(memory_maxsize=512, ttl=3600)
        assert cache.maxsize == 512
        assert cache.ttl == 3600

    def test_get_miss_with_data(self):
        cache = DiagramCache()
        data = np.array([[1.0, 2.0, 0]])
        assert cache.get(data) is None

    def test_set_and_get(self):
        cache = DiagramCache()
        data = np.array([[0.0, 1.0, 0]])
        result = np.array([0.5])
        cache.set(data, result)
        cached = cache.get(data)
        assert cached is not None
        np.testing.assert_array_equal(cached, result)

    def test_get_by_key(self):
        cache = DiagramCache()
        cache.set_by_key("test_key", 42)
        assert cache.get_by_key("test_key") == 42

    def test_get_by_key_missing(self):
        cache = DiagramCache()
        assert cache.get_by_key("nonexistent") is None
        assert cache.get_by_key("nonexistent", "default") == "default"

    def test_clear(self):
        cache = DiagramCache()
        data = np.array([[0.0, 1.0, 0]])
        cache.set(data, "value")
        assert cache.get(data) is not None
        cache.clear()
        assert cache.get(data) is None

    def test_context_manager(self):
        with DiagramCache() as cache:
            assert isinstance(cache, DiagramCache)

    def test_repr(self):
        cache = DiagramCache()
        r = repr(cache)
        assert "DiagramCache" in r


class TestCachedPersistenceDecorator:
    def test_caches_result(self):
        call_count = 0

        @cached_persistence()
        def compute(data, factor=1):
            nonlocal call_count
            call_count += 1
            return np.array([data.sum() * factor])

        data = np.array([[1.0, 2.0, 0]])
        result1 = compute(data, factor=2)
        result2 = compute(data, factor=2)
        assert call_count == 1
        np.testing.assert_array_equal(result1, result2)

    def test_different_data_different_cache(self):
        call_count = 0

        @cached_persistence()
        def compute(data):
            nonlocal call_count
            call_count += 1
            return np.array([data.sum()])

        compute(np.array([[1.0, 2.0, 0]]))
        compute(np.array([[3.0, 4.0, 0]]))
        assert call_count == 2

    def test_cache_attribute(self):
        @cached_persistence()
        def compute(data):
            return np.array([data.sum()])

        assert hasattr(compute, "cache")
        assert isinstance(compute.cache, DiagramCache)


class TestMemoizePersistent:
    def test_basic_memoization(self, tmp_path):
        call_count = 0

        @memoize_persistent(cache_dir=str(tmp_path / "memo"))
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert call_count == 1
        assert compute(5) == 10
        assert call_count == 1

    def test_different_args(self, tmp_path):
        call_count = 0

        @memoize_persistent(cache_dir=str(tmp_path / "memo2"))
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(10) == 20
        assert call_count == 2

    def test_clear(self, tmp_path):
        call_count = 0

        @memoize_persistent(cache_dir=str(tmp_path / "memo3"))
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)
        compute.clear()
        compute(5)
        assert call_count == 2

    def test_repr(self, tmp_path):
        @memoize_persistent(cache_dir=str(tmp_path / "memo4"))
        def compute(x):
            return x

        r = repr(compute)
        assert "MemoizePersistent" in r
        assert "compute" in r

    def test_memoize_persistent_class_direct(self, tmp_path):
        def add_one(x):
            return x + 1

        mp = MemoizePersistent(add_one, cache_dir=str(tmp_path / "direct"))
        assert mp(5) == 6
        assert mp(5) == 6
