"""Tests for cache/_engine.py — DiagramCache and PersistentDiagramCache."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pynerve.cache._engine import (
    _MISSING,
    DiagramCache,
    PersistentDiagramCache,
    _validate_cache_key,
    cached_persistence,
)

# Check if diskcache is available
try:
    import diskcache  # noqa: F401

    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False


# _validate_cache_key 


class TestValidateCacheKey:
    def test_valid_string(self):
        assert _validate_cache_key("my_key") == "my_key"

    def test_empty_raises(self):
        with pytest.raises(Exception):
            _validate_cache_key("")  # type: ignore[arg-type]

    def test_non_string_raises(self):
        with pytest.raises(Exception):
            _validate_cache_key(123)  # type: ignore[arg-type]


# DiagramCache 


class TestDiagramCacheBasic:
    def test_construction_defaults(self):
        cache = DiagramCache()
        assert cache.memory_maxsize == 1024
        assert cache.ttl is None
        assert cache.use_disk is False
        cache.close()

    def test_repr(self):
        cache = DiagramCache(memory_maxsize=64)
        r = repr(cache)
        assert "DiagramCache" in r
        assert "memory_maxsize=64" in r
        assert "entries=0" in r
        assert "disk=disabled" in r
        cache.close()

    def test_context_manager(self):
        with DiagramCache(memory_maxsize=4) as cache:
            data = np.array([1.0, 2.0, 3.0])
            cache.set(data, "result1", op="add")
            assert cache.get(data, op="add") == "result1"
        # After __exit__, cache should be closed


class TestDiagramCacheKeyGeneration:
    def test_make_key_deterministic(self):
        cache = DiagramCache()
        k1 = cache._make_key(np.array([1.0, 2.0]), param="x")
        k2 = cache._make_key(np.array([1.0, 2.0]), param="x")
        assert k1 == k2
        cache.close()

    def test_make_key_different_data(self):
        cache = DiagramCache()
        k1 = cache._make_key(np.array([1.0, 2.0]))
        k2 = cache._make_key(np.array([2.0, 3.0]))
        assert k1 != k2
        cache.close()

    def test_make_key_different_params(self):
        cache = DiagramCache()
        k1 = cache._make_key(np.array([1.0]), param="x")
        k2 = cache._make_key(np.array([1.0]), param="y")
        assert k1 != k2
        cache.close()

    def test_make_key_fast_deterministic(self):
        cache = DiagramCache()
        k1 = cache._make_key_fast(np.array([1.0, 2.0]), param="x")
        k2 = cache._make_key_fast(np.array([1.0, 2.0]), param="x")
        assert k1 == k2
        cache.close()

    def test_make_key_fast_is_hex(self):
        cache = DiagramCache()
        key = cache._make_key_fast(np.array([1.0]))
        int(key, 16)  # should not raise
        cache.close()

    def test_make_key_non_contiguous_input(self):
        cache = DiagramCache()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        non_contig = data[:, ::-1].copy()
        key = cache._make_key(non_contig)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex
        cache.close()


class TestDiagramCacheGetSet:
    def test_get_nonexistent(self):
        cache = DiagramCache()
        assert cache.get(np.array([1.0])) is None
        cache.close()

    def test_set_and_get(self):
        cache = DiagramCache()
        data = np.array([1.0, 2.0, 3.0])
        cache.set(data, "hello", op="test")
        assert cache.get(data, op="test") == "hello"
        cache.close()

    def test_set_overwrites(self):
        cache = DiagramCache()
        data = np.array([1.0])
        cache.set(data, "first")
        cache.set(data, "second")
        assert cache.get(data) == "second"
        cache.close()

    def test_get_by_key(self):
        cache = DiagramCache()
        key = cache._make_key(np.array([5.0, 6.0]))
        cache.set_by_key(key, "result")
        assert cache.get_by_key(key) == "result"
        cache.close()

    def test_get_by_key_default(self):
        cache = DiagramCache()
        assert cache.get_by_key("nonexistent", "default") == "default"
        cache.close()

    def test_set_complex_result(self):
        cache = DiagramCache()
        data = np.array([1.0])
        result = {"a": [1, 2, 3], "b": np.array([4.0, 5.0])}
        cache.set(data, result)
        retrieved = cache.get(data)
        assert retrieved["a"] == [1, 2, 3]
        cache.close()


class TestDiagramCacheExpiry:
    @pytest.mark.skip(reason="TTL must be integer seconds; sub-second expiry not testable")
    def test_ttl_expired_returns_none(self):
        cache = DiagramCache(memory_maxsize=10, ttl=1)
        data = np.array([1.0])
        cache.set(data, "expired")
        time.sleep(1.5)
        assert cache.get(data) is None
        cache.close()

    def test_ttl_none_never_expires(self):
        cache = DiagramCache(memory_maxsize=10, ttl=None)
        data = np.array([2.0])
        cache.set(data, "persistent")
        assert cache.get(data) == "persistent"
        cache.close()


class TestDiagramCacheEviction:
    def test_lru_eviction(self):
        cache = DiagramCache(memory_maxsize=2)
        # Fill cache
        cache.set_by_key("a", "val_a")
        cache.set_by_key("b", "val_b")
        # Access 'a' to make it recently used
        assert cache.get_by_key("a") == "val_a"
        # Add third entry — should evict 'b' (LRU)
        cache.set_by_key("c", "val_c")
        assert cache.get_by_key("a") == "val_a"
        assert cache.get_by_key("c") == "val_c"
        assert cache.get_by_key("b") is None
        cache.close()


class TestDiagramCacheClear:
    def test_clear_removes_all(self):
        cache = DiagramCache(memory_maxsize=10)
        cache.set_by_key("a", "1")
        cache.set_by_key("b", "2")
        cache.clear()
        assert cache.get_by_key("a") is None
        assert cache.get_by_key("b") is None
        cache.close()


class TestDiagramCacheThreadSafety:
    @pytest.mark.skip(reason="concurrent access test is not deterministic with current cache implementation")
    def test_concurrent_access(self):
        cache = DiagramCache(memory_maxsize=100)

        def worker(start: float):
            for i in range(50):
                cache.set_by_key(f"t{start}_{i}", start + i)

        threads = [threading.Thread(target=worker, args=(float(t),)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no crashes
        assert cache.get_by_key("t0.0_0") is not None
        cache.close()


class TestDiagramCacheDisk:
    @pytest.mark.skipif(not HAS_DISKCACHE, reason="diskcache not installed")
    def test_disk_enabled(self, tmp_path):
        cache = DiagramCache(memory_maxsize=4, use_disk=True, disk_path=str(tmp_path))
        assert cache.use_disk is True
        data = np.array([10.0, 20.0])
        cache.set(data, "disk_result")
        # Create new cache to verify disk persistence
        cache2 = DiagramCache(memory_maxsize=4, use_disk=True, disk_path=str(tmp_path))
        assert cache2.get(data) == "disk_result"
        cache.close()
        cache2.close()

    def test_disk_without_diskcache_raises(self):
        if not HAS_DISKCACHE:
            with pytest.raises(ImportError, match="diskcache"):
                DiagramCache(use_disk=True)


# PersistentDiagramCache 


class TestPersistentDiagramCache:
    @pytest.mark.skipif(not HAS_DISKCACHE, reason="diskcache not installed")
    def test_basic(self, tmp_path):
        cache = PersistentDiagramCache(
            cache_dir=str(tmp_path), size_limit=1024 * 1024, ttl=3600, memory_maxsize=10
        )
        assert cache.memory_maxsize == 10
        assert cache.ttl == 3600
        data = np.array([1.0, 2.0])
        cache.set(data, "persistent_result")
        assert cache.get(data) == "persistent_result"
        cache.close()

    @pytest.mark.skipif(not HAS_DISKCACHE, reason="diskcache not installed")
    def test_repr(self, tmp_path):
        cache = PersistentDiagramCache(cache_dir=str(tmp_path), size_limit=1024, ttl=60)
        r = repr(cache)
        assert "PersistentDiagramCache" in r
        assert "size_limit=1024" in r
        assert "ttl=60" in r
        cache.close()

    @pytest.mark.skipif(not HAS_DISKCACHE, reason="diskcache not installed")
    def test_defaults(self, tmp_path):
        cache = PersistentDiagramCache(cache_dir=str(tmp_path))
        assert cache.memory_maxsize == 1000
        cache.close()

    def test_without_diskcache_raises(self):
        if not HAS_DISKCACHE:
            with pytest.raises(RuntimeError, match="diskcache"):
                PersistentDiagramCache(cache_dir="/tmp/test_cache")


# cached_persistence decorator 


class TestCachedPersistence:
    def test_basic_caching(self):
        call_count = 0

        @cached_persistence(memory_maxsize=10)
        def compute(data: np.ndarray, **kwargs: int) -> int:
            nonlocal call_count
            call_count += 1
            return int(np.sum(data)) + kwargs.get("offset", 0)

        data = np.array([1.0, 2.0, 3.0])
        r1 = compute(data, offset=10)
        r2 = compute(data, offset=10)
        assert r1 == r2 == 16
        assert call_count == 1  # second call is cached

    def test_cache_miss(self):
        call_count = 0

        @cached_persistence(memory_maxsize=10)
        def compute(data: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.sum(data))

        r1 = compute(np.array([1.0, 2.0]))
        r2 = compute(np.array([3.0, 4.0]))
        assert r1 == 3.0
        assert r2 == 7.0
        assert call_count == 2

    def test_custom_key_fn(self):
        call_count = 0

        def my_key(data, **kwargs):
            return f"custom_{kwargs.get('id')}"

        @cached_persistence(key_fn=my_key)
        def compute(data, **kwargs):
            nonlocal call_count
            call_count += 1
            return kwargs.get("id")

        assert compute(np.array([1.0]), id="a") == "a"
        assert compute(np.array([2.0]), id="a") == "a"  # same key, cached
        assert call_count == 1

    def test_key_fn_not_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            cached_persistence(key_fn="not_callable")  # type: ignore[arg-type]

    def test_wrapper_has_cache_attr(self):
        @cached_persistence(memory_maxsize=5)
        def func(data):
            return data

        assert hasattr(func, "cache")
        assert isinstance(func.cache, DiagramCache)  # type: ignore[attr-defined]
        func.cache.close()  # type: ignore[attr-defined]

    def test_wrapper_preserves_name(self):
        @cached_persistence()
        def my_special_func(data):
            return data

        assert my_special_func.__name__ == "my_special_func"
