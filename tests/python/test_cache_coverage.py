from __future__ import annotations

import time

import numpy as np
import pytest
from pynerve.cache import (
    DiagramCache,
    PersistentDiagramCache,
    SmartCache,
    _validate_cache_key,
    cached_persistence,
    get_cache_stats,
)
from pynerve.exceptions import ValidationError

try:
    import diskcache  # noqa: F401

    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False

needs_diskcache = pytest.mark.skipif(not HAS_DISKCACHE, reason="diskcache not installed")


class TestDiagramCacheAdvanced:
    def test_init_with_disk_requires_diskcache(self, monkeypatch):
        import pynerve.cache._engine as _eng

        monkeypatch.setattr(_eng, "HAS_DISKCACHE", False)
        with pytest.raises(ImportError, match="diskcache"):
            DiagramCache(use_disk=True)

    def test_init_with_ttl_zero_validated(self):
        with pytest.raises(ValidationError, match="must be positive"):
            DiagramCache(ttl=0)

    def test_init_with_memory_maxsize_zero(self):
        with pytest.raises(ValidationError, match="must be positive"):
            DiagramCache(memory_maxsize=0)

    def test_init_with_invalid_use_disk(self):
        with pytest.raises(ValidationError, match="must be a boolean"):
            DiagramCache(use_disk=1)

    def test_make_key_deterministic(self):
        cache = DiagramCache()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        key1 = cache._make_key(data, dim=0)
        key2 = cache._make_key(data, dim=0)
        assert key1 == key2

    def test_make_key_different_data(self):
        cache = DiagramCache()
        data1 = np.array([[1.0, 2.0]])
        data2 = np.array([[3.0, 4.0]])
        assert cache._make_key(data1) != cache._make_key(data2)

    def test_make_key_different_params(self):
        cache = DiagramCache()
        data = np.array([[1.0, 2.0]])
        assert cache._make_key(data, dim=0) != cache._make_key(data, dim=1)

    def test_make_key_different_dtype(self):
        cache = DiagramCache()
        data_f32 = np.array([[1.0, 2.0]], dtype=np.float32)
        data_f64 = np.array([[1.0, 2.0]], dtype=np.float64)
        assert cache._make_key(data_f32) != cache._make_key(data_f64)

    def test_make_key_non_contiguous_input(self):
        cache = DiagramCache()
        contiguous = np.array([[1.0, 2.0], [3.0, 4.0]])
        noncontig = contiguous.T
        assert not noncontig.flags.c_contiguous
        key_contig = cache._make_key(contiguous)
        key_noncontig = cache._make_key(noncontig)
        assert key_contig != key_noncontig

    def test_make_key_fast_deterministic(self):
        cache = DiagramCache()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        key1 = cache._make_key_fast(data, dim=0)
        key2 = cache._make_key_fast(data, dim=0)
        assert key1 == key2

    def test_make_key_fast_different_data(self):
        cache = DiagramCache()
        a = np.array([[1.0]])
        b = np.array([[2.0]])
        assert cache._make_key_fast(a) != cache._make_key_fast(b)

    def test_get_with_params(self):
        cache = DiagramCache()
        data = np.array([[0.0, 1.0]])
        cache.set(data, "result", dim=0)
        assert cache.get(data, dim=0) == "result"
        assert cache.get(data, dim=1) is None

    def test_set_with_params(self):
        cache = DiagramCache()
        data = np.array([[0.0, 1.0]])
        cache.set(data, 42, dim=0, homology=1)
        assert cache.get(data, dim=0, homology=1) == 42

    def test_get_by_key_promotes_lru(self):
        cache = DiagramCache(memory_maxsize=2)
        cache.set_by_key("a", 1)
        cache.set_by_key("b", 2)
        cache.get_by_key("a")
        cache.set_by_key("c", 3)
        assert cache.get_by_key("a") == 1
        assert cache.get_by_key("b") is None
        assert cache.get_by_key("c") == 3

    def test_get_by_key_expired_memory(self, monkeypatch):
        cache = DiagramCache(ttl=1)
        cache.set_by_key("a", "val")
        future = time.time() + 10
        monkeypatch.setattr(time, "time", lambda: future)
        assert cache.get_by_key("a") is None

    def test_get_by_key_without_ttl_never_expires(self, monkeypatch):
        cache = DiagramCache()
        cache.set_by_key("a", "val")
        far_future = time.time() + 999999999
        monkeypatch.setattr(time, "time", lambda: far_future)
        assert cache.get_by_key("a") == "val"

    def test_add_to_memory_evicts_lru_when_full(self):
        cache = DiagramCache(memory_maxsize=2)
        cache._add_to_memory("a", 1)
        cache._add_to_memory("b", 2)
        cache._add_to_memory("c", 3)
        assert cache.get_by_key("a") is None
        assert cache.get_by_key("b") == 2
        assert cache.get_by_key("c") == 3

    def test_add_to_memory_updates_existing(self):
        cache = DiagramCache(memory_maxsize=2)
        cache._add_to_memory("a", 1)
        cache._add_to_memory("b", 2)
        cache._add_to_memory("a", 11)
        assert len(cache._cache) == 2
        assert cache.get_by_key("a") == 11

    def test_is_expired_with_ttl(self, monkeypatch):
        cache = DiagramCache(ttl=10)
        cache.set_by_key("a", 1)
        assert not cache._is_expired("a")
        future = time.time() + 20
        monkeypatch.setattr(time, "time", lambda: future)
        assert cache._is_expired("a")

    def test_is_expired_none_ttl(self):
        cache = DiagramCache()
        cache.set_by_key("a", 1)
        assert not cache._is_expired("a")

    def test_remove_key(self):
        cache = DiagramCache()
        cache.set_by_key("a", 1)
        cache.set_by_key("b", 2)
        cache._remove_key("a")
        assert cache.get_by_key("a") is None
        assert cache.get_by_key("b") == 2

    def test_remove_key_nonexistent(self):
        cache = DiagramCache()
        cache._remove_key("nonexistent")

    def test_close_without_disk(self):
        cache = DiagramCache()
        cache.close()

    def test_repr_with_entries(self):
        cache = DiagramCache()
        cache.set_by_key("a", 1)
        r = repr(cache)
        assert "entries=1" in r

    def test_repr_with_disk_disabled(self):
        cache = DiagramCache()
        r = repr(cache)
        assert "disk=disabled" in r

    def test_context_manager_exit(self):
        cache = DiagramCache()
        data = np.array([[1.0]])
        cache.set(data, "val")
        with cache as c:
            assert c.get(data) == "val"

    def test_make_key_empty_array(self):
        cache = DiagramCache()
        data = np.empty((0, 2))
        key = cache._make_key(data)
        assert isinstance(key, str)
        assert len(key) == 64

    def test_make_key_with_zero_dimension(self):
        cache = DiagramCache()
        data = np.empty((3, 0))
        key = cache._make_key(data)
        assert isinstance(key, str)

    def test_get_by_key_memory_hit_never_checks_disk(self):
        cache = DiagramCache()
        cache.set_by_key("a", "val")
        result = cache.get_by_key("a")
        assert result == "val"


@needs_diskcache
class TestDiagramCacheDisk:
    def test_init_with_disk(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        assert cache._disk_cache is not None
        assert cache.use_disk is True

    def test_get_by_key_disk_hit(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        cache.set_by_key("disk_key", "disk_val")
        fresh = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        assert fresh.get_by_key("disk_key") == "disk_val"

    def test_get_by_key_disk_miss(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        assert cache.get_by_key("nonexistent") is None

    def test_get_by_key_disk_hit_promotes_to_memory(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        cache.set_by_key("k", "v")
        fresh = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        fresh.get_by_key("k")
        assert "k" in fresh._cache

    def test_get_by_key_expired_memory_falls_through_to_disk(self, tmp_path, monkeypatch):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"), ttl=1)
        cache.set_by_key("k", "v")
        far_future = time.time() + 10
        monkeypatch.setattr(time, "time", lambda: far_future)
        assert cache.get_by_key("k") is None

    def test_set_by_key_writes_to_disk(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        cache.set_by_key("k", "v")
        assert cache._disk_cache.get("k") == "v"

    def test_clear_with_disk(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        data = np.array([[1.0]])
        cache.set(data, "val")
        cache.clear()
        assert cache.get(data) is None
        assert cache.get_by_key(cache._make_key(data)) is None
        assert len(cache._disk_cache) == 0

    def test_close_with_disk(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        cache.set_by_key("k", "v")
        cache.close()

    def test_context_manager_with_disk(self, tmp_path):
        with DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache")) as cache:
            cache.set_by_key("k", "v")
            assert cache.get_by_key("k") == "v"

    def test_repr_with_disk_enabled(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"))
        cache.set_by_key("k", "v")
        r = repr(cache)
        assert "disk=enabled" in r

    def test_set_by_key_with_ttl_passed_to_disk(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "cache"), ttl=3600)
        cache.set_by_key("k", "v")
        assert cache._disk_cache.get("k") == "v"


class TestPersistentDiagramCacheNoDisk:
    def test_init_missing_diskcache_raises_runtime_error(self, monkeypatch):
        import pynerve.cache._engine as _eng

        monkeypatch.setattr(_eng, "HAS_DISKCACHE", False)
        with pytest.raises(RuntimeError, match="diskcache required"):
            PersistentDiagramCache(cache_dir="/tmp/fake")


@needs_diskcache
class TestPersistentDiagramCache:
    def test_init_with_defaults(self, tmp_path):
        cache = PersistentDiagramCache(cache_dir=str(tmp_path / "persist"))
        assert cache.use_disk is True
        assert cache._disk_cache is not None

    def test_init_with_custom_params(self, tmp_path):
        cache = PersistentDiagramCache(
            cache_dir=str(tmp_path / "persist2"),
            size_limit=5 * 1024 * 1024,
            ttl=86400,
            memory_maxsize=500,
        )
        assert cache.maxsize == 500
        assert cache.ttl == 86400

    def test_init_creates_directory(self, tmp_path):
        path = tmp_path / "new_dir" / "nested"
        PersistentDiagramCache(cache_dir=str(path))
        assert path.exists()

    def test_repr(self, tmp_path):
        cache = PersistentDiagramCache(cache_dir=str(tmp_path / "persist_repr"))
        r = repr(cache)
        assert "PersistentDiagramCache" in r

    def test_set_and_get(self, tmp_path):
        cache = PersistentDiagramCache(cache_dir=str(tmp_path / "persist_sg"))
        data = np.array([[0.0, 1.0]])
        cache.set(data, "result")
        assert cache.get(data) == "result"

    def test_set_and_get_with_params(self, tmp_path):
        cache = PersistentDiagramCache(cache_dir=str(tmp_path / "persist_params"))
        data = np.array([[0.0, 1.0]])
        cache.set(data, 42, dim=0)
        assert cache.get(data, dim=0) == 42

    def test_clear(self, tmp_path):
        cache = PersistentDiagramCache(cache_dir=str(tmp_path / "persist_clear"))
        data = np.array([[1.0]])
        cache.set(data, "val")
        cache.clear()
        assert cache.get(data) is None

    def test_ttl_expiry(self, tmp_path, monkeypatch):
        cache = PersistentDiagramCache(cache_dir=str(tmp_path / "persist_ttl"), ttl=1)
        data = np.array([[1.0]])
        cache.set(data, "val")
        far_future = time.time() + 10
        monkeypatch.setattr(time, "time", lambda: far_future)
        assert cache.get(data) is None


class TestCachedPersistenceAdvanced:
    def test_key_fn_invalid_raises_typeerror(self):
        with pytest.raises(TypeError, match="key_fn must be callable"):
            cached_persistence(key_fn="not_callable")

    def test_key_fn_is_used(self):
        call_count = 0

        def compute(data, **kwargs):
            nonlocal call_count
            call_count += 1
            return np.array([data.sum()])

        decorated = cached_persistence(key_fn=lambda d, **kw: "static_key")(compute)
        data1 = np.array([[1.0, 2.0, 0]])
        data2 = np.array([[5.0, 6.0, 0]])
        r1 = decorated(data1)
        r2 = decorated(data2)
        assert call_count == 1
        np.testing.assert_array_equal(r1, r2)

    def test_cache_attribute_set_by_key(self):
        @cached_persistence()
        def compute(data):
            return np.array([data.sum()])

        data = np.array([[1.0, 2.0]])
        compute(data)
        cache = compute.cache  # type: ignore[attr-defined]
        key = cache._make_key(data)
        assert cache.get_by_key(key) is not None

    def test_wrapper_preserves_name(self):
        @cached_persistence()
        def my_func(data):
            return data

        assert my_func.__name__ == "my_func"

    def test_decorator_with_memory_maxsize(self):
        @cached_persistence(memory_maxsize=64)
        def compute(data):
            return np.array([data.sum()])

        data = np.array([[1.0, 2.0]])
        compute(data)
        c = compute.cache  # type: ignore[attr-defined]
        assert c.memory_maxsize == 64

    def test_decorator_with_ttl(self, monkeypatch):
        call_count = 0

        @cached_persistence(ttl=1)
        def compute(data):
            nonlocal call_count
            call_count += 1
            return np.array([data.sum()])

        data = np.array([[1.0, 2.0]])
        compute(data)
        assert call_count == 1
        far_future = time.time() + 10
        monkeypatch.setattr(time, "time", lambda: far_future)
        compute(data)
        assert call_count == 2


class TestValidateIgnoreArgs:
    def test_none_returns_empty_set(self):
        from pynerve.cache._smart import _validate_ignore_args

        result = _validate_ignore_args(None)
        assert result == set()

    def test_valid_list(self):
        from pynerve.cache._smart import _validate_ignore_args

        result = _validate_ignore_args(["a", "b"])
        assert result == {"a", "b"}

    def test_valid_tuple(self):
        from pynerve.cache._smart import _validate_ignore_args

        result = _validate_ignore_args(("x", "y"))
        assert result == {"x", "y"}

    def test_string_raises_typeerror(self):
        from pynerve.cache._smart import _validate_ignore_args

        with pytest.raises(TypeError, match="ignore_args must be a sequence"):
            _validate_ignore_args("abc")

    def test_bytes_raises_typeerror(self):
        from pynerve.cache._smart import _validate_ignore_args

        with pytest.raises(TypeError, match="ignore_args must be a sequence"):
            _validate_ignore_args(b"abc")

    def test_non_sequence_raises_typeerror(self):
        from pynerve.cache._smart import _validate_ignore_args

        with pytest.raises(TypeError, match="ignore_args must be a sequence"):
            _validate_ignore_args(42)

    def test_empty_strings_raises_valueerror(self):
        from pynerve.cache._smart import _validate_ignore_args

        with pytest.raises(ValueError, match="ignore_args must contain non-empty strings"):
            _validate_ignore_args(["a", ""])

    def test_non_string_items_raises_valueerror(self):
        from pynerve.cache._smart import _validate_ignore_args

        with pytest.raises(ValueError, match="ignore_args must contain non-empty strings"):
            _validate_ignore_args(["a", 1])

    def test_mixed_types_raises_valueerror(self):
        from pynerve.cache._smart import _validate_ignore_args

        with pytest.raises(ValueError, match="ignore_args must contain non-empty strings"):
            _validate_ignore_args([None])


class TestSmartCache:
    def test_init_defaults(self):
        cache = SmartCache()
        assert cache.small_threshold == 1024 * 1024
        assert cache.memory_cache.memory_maxsize == 128

    def test_init_custom_params(self):
        cache = SmartCache(memory_maxsize=64, small_threshold=512)
        assert cache.memory_cache.memory_maxsize == 64
        assert cache.small_threshold == 512

    def test_init_without_diskcache(self, monkeypatch):
        import pynerve.cache._engine as _eng
        from pynerve.cache import _smart

        monkeypatch.setattr(_eng, "HAS_DISKCACHE", False)
        monkeypatch.setattr(_smart, "HAS_DISKCACHE", False)
        cache = SmartCache()
        assert cache.disk_cache is None

    def test_repr_without_disk(self, monkeypatch):
        import pynerve.cache._engine as _eng
        from pynerve.cache import _smart

        monkeypatch.setattr(_eng, "HAS_DISKCACHE", False)
        monkeypatch.setattr(_smart, "HAS_DISKCACHE", False)
        cache = SmartCache()
        r = repr(cache)
        assert "disk=no" in r

    def test_context_manager(self):
        with SmartCache() as cache:
            assert isinstance(cache, SmartCache)

    def test_context_manager_with_set_get(self):
        with SmartCache() as cache:
            cache.set("my_key", "my_val")
            assert cache.get("my_key") == "my_val"

    def test_get_memory_hit(self):
        cache = SmartCache()
        cache.memory_cache.set_by_key("k", "v")
        assert cache.get("k") == "v"

    def test_get_miss(self):
        cache = SmartCache()
        assert cache.get("nonexistent") is None

    def test_get_empty_key_raises(self):
        cache = SmartCache()
        with pytest.raises(ValidationError, match="non-empty"):
            cache.get("")

    def test_get_non_string_key_raises(self):
        cache = SmartCache()
        with pytest.raises(ValidationError, match="non-empty"):
            cache.get(123)

    def test_set_small_result_to_memory(self):
        cache = SmartCache(small_threshold=1024 * 1024)
        cache.set("k", "small_value")
        assert cache.memory_cache.get_by_key("k") == "small_value"

    def test_set_large_result_no_disk_no_error(self, monkeypatch):
        import pynerve.cache._engine as _eng
        from pynerve.cache import _smart

        monkeypatch.setattr(_eng, "HAS_DISKCACHE", False)
        monkeypatch.setattr(_smart, "HAS_DISKCACHE", False)
        cache = SmartCache(small_threshold=10)
        cache.set("k", "a very large value that exceeds threshold")
        assert cache.memory_cache.get_by_key("k") is None

    def test_set_empty_key_raises(self):
        cache = SmartCache()
        with pytest.raises(ValidationError, match="non-empty"):
            cache.set("", "val")

    def test_set_non_string_key_raises(self):
        cache = SmartCache()
        with pytest.raises(ValidationError, match="non-empty"):
            cache.set(None, "val")


@needs_diskcache
class TestSmartCacheDisk:
    def test_init_with_disk(self):
        cache = SmartCache()
        assert cache.disk_cache is not None

    def test_repr_with_disk(self):
        cache = SmartCache()
        r = repr(cache)
        assert "disk=yes" in r

    def test_get_disk_hit_by_internal_set_by_key(self):
        cache = SmartCache()
        dc = cache.disk_cache
        assert dc is not None
        dc.set_by_key("direct_key", "from_disk")
        assert cache.get("direct_key") == "from_disk"

    def test_get_disk_hit_promotes_small_result(self):
        cache = SmartCache(small_threshold=1024 * 1024)
        dc = cache.disk_cache
        assert dc is not None
        dc.set_by_key("k", "small_promotable")
        result = cache.get("k")
        assert result == "small_promotable"
        assert cache.memory_cache.get_by_key("k") == "small_promotable"

    def test_get_disk_hit_large_not_promoted(self):
        cache = SmartCache(small_threshold=10)
        dc = cache.disk_cache
        assert dc is not None
        large_value = "x" * 100
        dc.set_by_key("k", large_value)
        result = cache.get("k")
        assert result == large_value
        assert cache.memory_cache.get_by_key("k") is None

    def test_get_disk_miss(self):
        cache = SmartCache()
        assert cache.get("definitely_not_in_disk") is None

    def test_set_large_stores_in_disk(self):
        cache = SmartCache(small_threshold=10)
        large_value = "x" * 100
        data = np.array([[0.0, 1.0]])
        cache.set("my_key", large_value, data=data)
        assert cache.memory_cache.get_by_key("my_key") is None
        dc = cache.disk_cache
        assert dc is not None
        composite_key = dc._make_key(data, key="my_key")
        assert dc.get_by_key(composite_key) == large_value

    def test_get_with_data_warning(self, caplog):
        import logging

        cache = SmartCache()
        with caplog.at_level(logging.WARNING, logger="pynerve.cache"):
            result = cache.get("nonexistent_key", data=np.array([[1.0]]))
        assert result is None
        assert "SmartCache.get() called with data but disk cache miss" in caplog.text


class TestGetCacheStats:
    def test_memory_only(self):
        cache = DiagramCache()
        cache.set_by_key("a", 1)
        cache.set_by_key("b", 2)
        stats = get_cache_stats(cache)
        assert stats["memory_entries"] == 2
        assert stats["memory_maxsize"] == 1024

    def test_type_error_non_diagramcache(self):
        with pytest.raises(TypeError, match="must be a DiagramCache"):
            get_cache_stats("not_a_cache")

    def test_type_error_smartcache(self):
        with pytest.raises(TypeError, match="must be a DiagramCache"):
            get_cache_stats(SmartCache())


@needs_diskcache
class TestGetCacheStatsDisk:
    def test_memory_and_disk(self, tmp_path):
        cache = DiagramCache(use_disk=True, disk_path=str(tmp_path / "stats"))
        cache.set_by_key("a", 1)
        stats = get_cache_stats(cache)
        assert stats["memory_entries"] == 1
        assert "disk_entries" in stats
        assert stats["disk_entries"] == 1
        assert "disk_size" in stats
        assert stats["disk_size"] >= 0


class TestValidationImportsFromEngine:
    def test_validate_cache_key_present(self):
        assert callable(_validate_cache_key)

    def test_validate_cache_key_none(self):
        with pytest.raises(ValidationError, match="non-empty"):
            _validate_cache_key(None)

    def test_validate_cache_key_bytes(self):
        with pytest.raises(ValidationError, match="non-empty"):
            _validate_cache_key(b"key")

    def test_validate_cache_key_whitespace(self):
        result = _validate_cache_key("  ")
        assert result == "  "
