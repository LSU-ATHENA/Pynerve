from __future__ import annotations

import sys
import threading
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from pynerve._persistence_result import (
    PersistenceResult,
    _estimate_n_points,
    _nerve_state,
    _warn_large_max_radius_cap,
)

torch = pytest.importorskip("torch", reason="PyTorch required for tensor tests")


class TestEstimateNPoints:
    def test_numpy_1d(self):
        pts = np.array([1.0, 2.0, 3.0, 4.0])
        assert _estimate_n_points(pts) == 4

    def test_numpy_2d(self):
        pts = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        assert _estimate_n_points(pts) == 3

    def test_numpy_scalar(self):
        pts = np.array(42.0)
        assert _estimate_n_points(pts) == 0

    def test_list(self):
        pts = [1, 2, 3, 4, 5]
        assert _estimate_n_points(pts) == 5

    def test_empty_list(self):
        assert _estimate_n_points([]) == 0

    def test_object_with_shape_as_int_raises_returns_zero(self):
        class Bad:
            shape = 7

        assert _estimate_n_points(Bad()) == 0

    def test_object_without_shape_or_len(self):
        class Foo:
            pass

        assert _estimate_n_points(Foo()) == 0

    def test_object_without_shape_nor_len(self):
        class NoLen:
            def __len__(self):
                raise TypeError("no len")

        assert _estimate_n_points(NoLen()) == 0

    def test_object_with_shape_raises_attribute_error(self):
        class BadShape:
            @property
            def shape(self):
                raise AttributeError("bad")

        assert _estimate_n_points(BadShape()) == 0

    def test_none_returns_zero(self):
        assert _estimate_n_points(None) == 0


class TestWarnLargeMaxRadiusCap:
    def test_small_cap_no_warning(self, monkeypatch):
        monkeypatch.setattr("pynerve._persistence_result._MAX_RADIUS_CAP", 1e10)
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARNED", [False])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_max_radius_cap()
        assert len(w) == 0

    def test_already_warned_no_repeat(self, monkeypatch):
        monkeypatch.setattr("pynerve._persistence_result._MAX_RADIUS_CAP", 1e16)
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARNED", [True])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_max_radius_cap()
        assert len(w) == 0

    def test_large_cap_first_warning(self, monkeypatch):
        monkeypatch.setattr("pynerve._persistence_result._MAX_RADIUS_CAP", 1e16)
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARNED", [False])
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARN_LOCK", threading.Lock())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_max_radius_cap()
        assert len(w) == 1
        assert "NERVE_MAX_RADIUS_CAP" in str(w[0].message)
        assert issubclass(w[0].category, UserWarning)

    def test_large_cap_warns_only_once(self, monkeypatch):
        monkeypatch.setattr("pynerve._persistence_result._MAX_RADIUS_CAP", 1e16)
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARNED", [False])
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARN_LOCK", threading.Lock())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_max_radius_cap()
            _warn_large_max_radius_cap()
        assert len(w) == 1

    def test_boundary_just_below_1e14_no_warning(self, monkeypatch):
        monkeypatch.setattr("pynerve._persistence_result._MAX_RADIUS_CAP", 1e14 - 1)
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARNED", [False])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_max_radius_cap()
        assert len(w) == 0

    def test_exactly_1e14_warns(self, monkeypatch):
        monkeypatch.setattr("pynerve._persistence_result._MAX_RADIUS_CAP", 1e14)
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARNED", [False])
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARN_LOCK", threading.Lock())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_max_radius_cap()
        assert len(w) == 1

    def test_default_cap_1e15_warns(self, monkeypatch):
        monkeypatch.setattr("pynerve._persistence_result._MAX_RADIUS_CAP", 1e15)
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARNED", [False])
        monkeypatch.setattr("pynerve._persistence_result._CAP_WARN_LOCK", threading.Lock())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_max_radius_cap()
        assert len(w) == 1
        assert "1e+15" in str(w[0].message)


class TestNerveState:
    @pytest.fixture(autouse=True)
    def _reset_globals(self, monkeypatch):
        import pynerve._persistence_result as pr

        with pr._NERVE_STATE_LOCK:
            monkeypatch.setattr(pr, "_CORE", None)
            monkeypatch.setattr(pr, "_CORE_IMPORT_ERROR", None)
            monkeypatch.setattr(pr, "_PYTORCH", None)
        yield
        with pr._NERVE_STATE_LOCK:
            monkeypatch.setattr(pr, "_CORE", None)
            monkeypatch.setattr(pr, "_CORE_IMPORT_ERROR", None)
            monkeypatch.setattr(pr, "_PYTORCH", None)

    def test_no_pynerve_module_in_sys_modules(self):
        pynerve_mod = sys.modules.get("pynerve")
        if pynerve_mod is not None:
            with patch.dict(sys.modules, {"pynerve": None}):
                core, err, pt = _nerve_state()
            sys.modules["pynerve"] = pynerve_mod
        else:
            core, err, pt = _nerve_state()
        assert core is None
        assert err is None
        assert pt is None

    def test_pynerve_module_with_core(self, monkeypatch):
        class FakeCore:
            pass

        fake_module = type("mod", (), {"_core": FakeCore, "_pytorch": torch})()
        import pynerve._persistence_result as pr

        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "pynerve", fake_module)
            mp.setattr(pr, "_CORE", None)
            mp.setattr(pr, "_CORE_IMPORT_ERROR", None)
            mp.setattr(pr, "_PYTORCH", None)
            core, err, pt = _nerve_state()
        assert core is FakeCore
        assert err is None
        assert pt is torch

    def test_pynerve_module_with_import_error(self, monkeypatch):
        exc = ImportError("no core")
        fake_module = type(
            "mod", (), {"_core": None, "_core_import_error": exc, "_pytorch": None}
        )()
        import pynerve._persistence_result as pr

        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "pynerve", fake_module)
            mp.setattr(pr, "_CORE", None)
            mp.setattr(pr, "_CORE_IMPORT_ERROR", None)
            mp.setattr(pr, "_PYTORCH", None)
            core, err, pt = _nerve_state()
        assert core is None
        assert err is exc
        assert pt is None

    def test_cached_core_returns_immediately(self, monkeypatch):
        import pynerve._persistence_result as pr

        fake_core = object()
        fake_pt = object()
        with pr._NERVE_STATE_LOCK:
            monkeypatch.setattr(pr, "_CORE", fake_core)
            monkeypatch.setattr(pr, "_CORE_IMPORT_ERROR", None)
            monkeypatch.setattr(pr, "_PYTORCH", fake_pt)
        core, err, pt = _nerve_state()
        assert core is fake_core
        assert err is None
        assert pt is fake_pt

    def test_cached_error_returns_immediately(self, monkeypatch):
        import pynerve._persistence_result as pr

        fake_error = ImportError("cached")
        with pr._NERVE_STATE_LOCK:
            monkeypatch.setattr(pr, "_CORE", None)
            monkeypatch.setattr(pr, "_CORE_IMPORT_ERROR", fake_error)
            monkeypatch.setattr(pr, "_PYTORCH", None)
        core, err, pt = _nerve_state()
        assert core is None
        assert err is fake_error
        assert pt is None

    def test_pynerve_module_no_core_no_error(self, monkeypatch):
        fake_module = type("mod", (), {})()
        import pynerve._persistence_result as pr

        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "pynerve", fake_module)
            mp.setattr(pr, "_CORE", None)
            mp.setattr(pr, "_CORE_IMPORT_ERROR", None)
            mp.setattr(pr, "_PYTORCH", None)
            core, err, pt = _nerve_state()
        assert core is None
        assert err is None
        assert pt is None


class TestPersistenceResultDefaults:
    def test_default_construction(self):
        result = PersistenceResult()
        assert result.pairs == []
        assert result.betti_numbers == []
        assert result.max_dim == 0
        assert result.max_radius == 0.0
        assert result.diagnostics == {}

    def test_explicit_construction(self):
        pairs = [(0.0, 1.0, 0), (1.0, 2.0, 1)]
        betti = [2, 1]
        result = PersistenceResult(
            pairs=pairs,
            betti_numbers=betti,
            max_dim=1,
            max_radius=2.5,
            diagnostics={"num_simplices": 100},
        )
        assert result.pairs == pairs
        assert result.betti_numbers == betti
        assert result.max_dim == 1
        assert result.max_radius == 2.5
        assert result.diagnostics == {"num_simplices": 100}

    def test_fields_are_independent_instances(self):
        r1 = PersistenceResult(pairs=[(0.0, 1.0, 0)])
        r2 = PersistenceResult()
        r1.pairs.append((1.0, 2.0, 1))
        assert len(r2.pairs) == 0


class TestNumPairs:
    def test_zero_pairs(self):
        result = PersistenceResult()
        assert result.num_pairs == 0

    def test_single_pair(self):
        result = PersistenceResult(pairs=[(0.0, 1.0, 0)])
        assert result.num_pairs == 1

    def test_multiple_pairs(self):
        result = PersistenceResult(pairs=[(0.0, 1.0, 0), (1.0, 2.0, 1), (0.5, 0.8, 0)])
        assert result.num_pairs == 3


class TestPairsArray:
    def test_empty_pairs(self):
        result = PersistenceResult()
        arr = result.pairs_array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (0,)
        assert arr.dtype == np.float64

    def test_single_pair(self):
        result = PersistenceResult(pairs=[(0.0, 1.0, 0)])
        arr = result.pairs_array
        np.testing.assert_array_equal(arr, np.array([[0.0, 1.0, 0.0]], dtype=np.float64))

    def test_multiple_pairs(self):
        pairs = [(0.0, 1.0, 0), (1.5, 3.0, 1), (2.0, float("inf"), 2)]
        result = PersistenceResult(pairs=pairs)
        arr = result.pairs_array
        expected = np.array(
            [[0.0, 1.0, 0.0], [1.5, 3.0, 1.0], [2.0, float("inf"), 2.0]],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(arr, expected)

    def test_caching(self):
        result = PersistenceResult(pairs=[(0.0, 1.0, 0)])
        arr1 = result.pairs_array
        arr2 = result.pairs_array
        assert arr1 is arr2

    def test_cache_stale_after_pairs_change(self):
        result = PersistenceResult(pairs=[(0.0, 1.0, 0)])
        arr1 = result.pairs_array
        result.pairs.append((1.0, 2.0, 1))
        arr2 = result.pairs_array
        assert arr1 is arr2
        assert arr2.shape == (1, 3)

    def test_dtype_is_float64(self):
        result = PersistenceResult(pairs=[(0, 1, 0)])
        assert result.pairs_array.dtype == np.float64

    def test_int_pairs_converted_to_float64(self):
        result = PersistenceResult(pairs=[(0.0, 1.0, 0)])
        arr = result.pairs_array
        assert arr[0, 2] == 0.0


class TestFromDict:
    def test_empty_dict(self):
        result = PersistenceResult.from_dict({})
        assert result.pairs == []
        assert result.betti_numbers == []
        assert result.max_dim == 0
        assert result.max_radius == 0.0
        assert result.diagnostics == {}

    def test_full_dict(self):
        result = PersistenceResult.from_dict(
            {
                "pairs": [(0.0, 1.0, 0), (1.0, 2.0, 1)],
                "betti_numbers": [2, 1],
                "max_dim": 1,
                "max_radius": 3.0,
                "diagnostics": {"num_simplices": 50},
            }
        )
        assert len(result.pairs) == 2
        assert result.betti_numbers == [2, 1]
        assert result.max_dim == 1
        assert result.max_radius == 3.0
        assert result.diagnostics == {"num_simplices": 50}

    def test_unknown_fields_warns(self):
        data = {"pairs": [], "unknown_key": "value", "extra": 42}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = PersistenceResult.from_dict(data)
        assert len(w) == 1
        assert "unknown" in str(w[0].message).lower()
        assert result.pairs == []

    def test_pairs_with_death_below_birth_clamped(self):
        data = {"pairs": [(2.0, 1.0, 0), (3.0, 2.5, 1)]}
        result = PersistenceResult.from_dict(data)
        assert len(result.pairs) == 2
        assert result.pairs[0] == (2.0, 2.0, 0)
        assert result.pairs[1] == (3.0, 3.0, 1)

    def test_pairs_death_equal_birth_no_clamp(self):
        data = {"pairs": [(1.0, 1.0, 0)]}
        result = PersistenceResult.from_dict(data)
        assert result.pairs[0] == (1.0, 1.0, 0)

    def test_pairs_death_above_birth_no_clamp(self):
        data = {"pairs": [(0.0, 1.0, 0)]}
        result = PersistenceResult.from_dict(data)
        assert result.pairs[0] == (0.0, 1.0, 0)

    def test_pairs_with_inf_death_no_clamp(self):
        data = {"pairs": [(0.0, float("inf"), 0), (1.0, float("inf"), 1)]}
        result = PersistenceResult.from_dict(data)
        assert result.pairs[0] == (0.0, float("inf"), 0)
        assert result.pairs[1] == (1.0, float("inf"), 1)

    def test_pairs_with_inf_death_below_birth_no_clamp(self):
        data = {"pairs": [(2.0, float("inf"), 0)]}
        result = PersistenceResult.from_dict(data)
        assert result.pairs[0] == (2.0, float("inf"), 0)

    def test_mixed_clamp_and_no_clamp(self):
        data = {"pairs": [(3.0, 2.0, 0), (4.0, float("inf"), 1), (1.0, 0.5, 0)]}
        result = PersistenceResult.from_dict(data)
        assert result.pairs[0] == (3.0, 3.0, 0)
        assert result.pairs[1] == (4.0, float("inf"), 1)
        assert result.pairs[2] == (1.0, 1.0, 0)

    def test_known_fields_all_present_pairs_as_list_of_lists(self):
        data = {
            "pairs": [[0.0, 1.0, 0], [1.0, 2.0, 1]],
            "betti_numbers": [1, 1],
            "max_dim": 1,
            "max_radius": 2.0,
            "diagnostics": {},
        }
        result = PersistenceResult.from_dict(data)
        assert result.pairs == [(0.0, 1.0, 0), (1.0, 2.0, 1)]

    def test_empty_pairs_in_dict(self):
        result = PersistenceResult.from_dict({"pairs": []})
        assert result.pairs == []

    def test_none_pairs_defaults_to_empty(self):
        result = PersistenceResult.from_dict({"max_dim": 2})
        assert result.pairs == []

    def test_betti_numbers_as_numpy_ints(self):
        data = {"betti_numbers": [np.int64(2), np.int64(1)]}
        result = PersistenceResult.from_dict(data)
        assert result.betti_numbers == [2, 1]

    def test_pairs_array_on_from_dict_result(self):
        data = {"pairs": [(0.0, 1.0, 0), (1.0, 2.0, 1)]}
        result = PersistenceResult.from_dict(data)
        arr = result.pairs_array
        assert arr.shape == (2, 3)


class TestRepr:
    def test_no_diagnostics(self):
        result = PersistenceResult(
            pairs=[(0.0, 1.0, 0)],
            betti_numbers=[1, 0],
            max_dim=1,
            max_radius=2.0,
        )
        r = repr(result)
        assert r.startswith("PersistenceResult(")
        assert r.endswith(")")
        assert "pairs=1" in r
        assert "betti_numbers=[1, 0]" in r
        assert "max_dim=1" in r
        assert "max_radius=2.0" in r
        assert "diagnostics" not in r

    def test_with_diagnostics(self):
        result = PersistenceResult(diagnostics={"num_simplices": 100, "peak_memory_bytes": 8192})
        r = repr(result)
        assert "diagnostics" in r
        assert "num_simplices" in r
        assert "peak_memory_bytes" in r

    def test_with_empty_diagnostics(self):
        result = PersistenceResult(diagnostics={})
        r = repr(result)
        assert "diagnostics" not in r

    def test_zero_pairs(self):
        result = PersistenceResult()
        r = repr(result)
        assert "pairs=0" in r

    def test_multiple_pairs(self):
        result = PersistenceResult(pairs=[(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 2)])
        r = repr(result)
        assert "pairs=3" in r


class TestStr:
    def test_str_same_as_repr(self):
        result = PersistenceResult(
            pairs=[(0.0, 1.0, 0)],
            betti_numbers=[1],
            max_dim=0,
            max_radius=1.5,
        )
        assert str(result) == repr(result)

    def test_str_with_diagnostics(self):
        result = PersistenceResult(diagnostics={"key": "val"})
        assert str(result) == repr(result)
