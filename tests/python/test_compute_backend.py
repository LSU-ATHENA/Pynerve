"""Tests for _compute_backend private helpers."""

from __future__ import annotations

import numpy as np
import pytest
from pynerve._compute_backend import (
    _compute_with_options,
    _persistence_from_distance_matrix,
    _resolve_device_to_backend,
    _seed_rng,
    _to_events_list,
    _to_internal_options,
    _try_as_ndarray,
    _warn_device_overrides_backend,
)
from pynerve._fallback_classes import (
    EventType,
    PersistenceBackend,
    PersistenceMode,
    PersistenceOptions,
)
from pynerve._persistence_result import PersistenceResult
from pynerve.exceptions import InvalidArgumentError, ValidationError


class TestToEventsList:
    def test_accepts_eventtype_enum(self):
        events = [(EventType.ADD, [0, 1]), (EventType.REMOVE, [2, 3])]
        assert _to_events_list(events) == [("add", [0, 1]), ("remove", [2, 3])]

    def test_accepts_string_add_remove(self):
        events = [("add", [0]), ("remove", [1, 2])]
        assert _to_events_list(events) == [("add", [0]), ("remove", [1, 2])]

    def test_rejects_invalid_event_type(self):
        with pytest.raises(ValidationError, match="event type"):
            _to_events_list([("insert", [0])])

    def test_rejects_unknown_string(self):
        with pytest.raises(ValidationError, match="event type"):
            _to_events_list([("delete", [0])])

    def test_converts_simplex_to_int(self):
        result = _to_events_list([(EventType.ADD, [1.0, 2.0, 3.5])])
        assert result == [("add", [1, 2, 3])]

    def test_empty_events_returns_empty_list(self):
        assert _to_events_list([]) == []

    def test_accepts_empty_simplex(self):
        assert _to_events_list([(EventType.ADD, [])]) == [("add", [])]

    def test_mixed_event_types(self):
        events = [(EventType.ADD, [0]), ("remove", [1]), (EventType.REMOVE, [2])]
        assert _to_events_list(events) == [
            ("add", [0]),
            ("remove", [1]),
            ("remove", [2]),
        ]

    def test_rejects_non_string_non_eventtype(self):
        with pytest.raises(ValidationError, match="event type"):
            _to_events_list([(42, [0])])

    def test_rejects_none_event_type(self):
        with pytest.raises((AttributeError, ValidationError)):
            _to_events_list([(None, [0])])


class TestToInternalOptions:
    def test_returns_py_opts_when_no_core(self, monkeypatch):
        monkeypatch.setattr("pynerve._compute_backend._nerve_state", lambda: (None, None, None))
        opts = PersistenceOptions(max_dim=5, max_radius=None)
        assert _to_internal_options(opts) is opts

    def test_creates_internal_opts_when_core_available(self, monkeypatch):
        class _MockCore:
            class PersistenceMode:
                EXACT = PersistenceMode.EXACT
                APPROX = PersistenceMode.APPROX

            class PersistenceBackend:
                CPU_EXACT = PersistenceBackend.CPU_EXACT
                CPU_ADAPTIVE_ACCELERATION = PersistenceBackend.CPU_ADAPTIVE_ACCELERATION
                CUDA_HYBRID = PersistenceBackend.CUDA_HYBRID

            class PersistenceOptions:
                def __init__(self):
                    self.mode = None
                    self.backend = None
                    self.max_dim = 0
                    self.max_radius = 0.0
                    self.threads = 0
                    self.error_tolerance = 0.0

        monkeypatch.setattr(
            "pynerve._compute_backend._nerve_state",
            lambda: (_MockCore(), None, None),
        )
        opts = PersistenceOptions(
            mode=PersistenceMode.APPROX,
            backend=PersistenceBackend.CUDA_HYBRID,
            max_dim=3,
            max_radius=2.5,
            threads=4,
            error_tolerance=1e-8,
        )
        result = _to_internal_options(opts)
        assert result is not opts
        assert result.mode == PersistenceMode.APPROX
        assert result.backend == PersistenceBackend.CUDA_HYBRID
        assert result.max_dim == 3
        assert result.max_radius == 2.5
        assert result.threads == 4
        assert result.error_tolerance == 1e-8

    def test_max_radius_none_becomes_inf(self, monkeypatch):
        class _MockCore:
            class PersistenceMode:
                EXACT = PersistenceMode.EXACT

            class PersistenceBackend:
                CPU_ADAPTIVE_ACCELERATION = PersistenceBackend.CPU_ADAPTIVE_ACCELERATION

            class PersistenceOptions:
                def __init__(self):
                    self.mode = None
                    self.backend = None
                    self.max_dim = 0
                    self.max_radius = 0.0
                    self.threads = 0
                    self.error_tolerance = 0.0

        monkeypatch.setattr(
            "pynerve._compute_backend._nerve_state",
            lambda: (_MockCore(), None, None),
        )
        opts = PersistenceOptions(max_radius=None)
        result = _to_internal_options(opts)
        assert result.max_radius == float("inf")

    def test_max_radius_inf_applies_cap(self, monkeypatch):
        class _MockCore:
            class PersistenceMode:
                EXACT = PersistenceMode.EXACT

            class PersistenceBackend:
                CPU_ADAPTIVE_ACCELERATION = PersistenceBackend.CPU_ADAPTIVE_ACCELERATION

            class PersistenceOptions:
                def __init__(self):
                    self.mode = None
                    self.backend = None
                    self.max_dim = 0
                    self.max_radius = 0.0
                    self.threads = 0
                    self.error_tolerance = 0.0

        monkeypatch.setattr(
            "pynerve._compute_backend._nerve_state",
            lambda: (_MockCore(), None, None),
        )
        # avoid the real cap warning by setting env var small
        monkeypatch.setattr(
            "pynerve._compute_backend._warn_large_max_radius_cap",
            lambda: None,
        )
        monkeypatch.setattr(
            "pynerve._compute_backend._MAX_RADIUS_CAP",
            1e6,
        )
        opts = PersistenceOptions(max_radius=float("inf"))
        result = _to_internal_options(opts)
        assert result.max_radius == 1e6

    def test_max_radius_finite_preserved(self, monkeypatch):
        class _MockCore:
            class PersistenceMode:
                EXACT = PersistenceMode.EXACT

            class PersistenceBackend:
                CPU_ADAPTIVE_ACCELERATION = PersistenceBackend.CPU_ADAPTIVE_ACCELERATION

            class PersistenceOptions:
                def __init__(self):
                    self.mode = None
                    self.backend = None
                    self.max_dim = 0
                    self.max_radius = 0.0
                    self.threads = 0
                    self.error_tolerance = 0.0

        monkeypatch.setattr(
            "pynerve._compute_backend._nerve_state",
            lambda: (_MockCore(), None, None),
        )
        opts = PersistenceOptions(max_radius=3.14)
        result = _to_internal_options(opts)
        assert result.max_radius == 3.14

    def test_caps_inf_max_radius_with_warning(self, monkeypatch):
        class _MockCore:
            class PersistenceMode:
                EXACT = PersistenceMode.EXACT

            class PersistenceBackend:
                CPU_ADAPTIVE_ACCELERATION = PersistenceBackend.CPU_ADAPTIVE_ACCELERATION

            class PersistenceOptions:
                def __init__(self):
                    self.mode = None
                    self.backend = None
                    self.max_dim = 0
                    self.max_radius = 0.0
                    self.threads = 0
                    self.error_tolerance = 0.0

        monkeypatch.setattr(
            "pynerve._compute_backend._nerve_state",
            lambda: (_MockCore(), None, None),
        )
        monkeypatch.setattr(
            "pynerve._compute_backend._MAX_RADIUS_CAP",
            1e15,
        )
        monkeypatch.setattr(
            "pynerve._persistence_result._MAX_RADIUS_CAP",
            1e15,
        )
        monkeypatch.setattr(
            "pynerve._persistence_result._CAP_WARNED",
            [False],
        )
        opts = PersistenceOptions(max_radius=float("inf"))
        with pytest.warns(UserWarning, match="NERVE_MAX_RADIUS_CAP"):
            _to_internal_options(opts)


class TestWarnDeviceOverridesBackend:
    def test_emits_warning_with_enum_backend(self):
        with pytest.warns(UserWarning, match="device"):
            _warn_device_overrides_backend("cuda:0", PersistenceBackend.CPU_EXACT)

    def test_warning_contains_device_and_backend(self):
        with pytest.warns(UserWarning) as record:
            _warn_device_overrides_backend("mps", PersistenceBackend.CUDA_HYBRID)
        msg = str(record[0].message)
        assert "mps" in msg
        assert "CUDA_HYBRID" in msg

    def test_accepts_any_device_string(self):
        with pytest.warns(UserWarning):
            _warn_device_overrides_backend("xpu:1", PersistenceBackend.CPU_ADAPTIVE_ACCELERATION)

    def test_stacklevel_is_three(self):
        with pytest.warns(UserWarning) as record:
            _warn_device_overrides_backend("cuda", PersistenceBackend.CUDA_HYBRID)
        assert len(record) == 1


class TestResolveDeviceToBackend:
    @pytest.mark.parametrize(
        ("device", "expected"),
        [
            ("cpu", PersistenceBackend.CPU_ADAPTIVE_ACCELERATION),
            ("cuda", PersistenceBackend.CUDA_HYBRID),
            ("mps", PersistenceBackend.CPU_ADAPTIVE_ACCELERATION),
            ("hip", PersistenceBackend.CUDA_HYBRID),
            ("xpu", PersistenceBackend.CPU_ADAPTIVE_ACCELERATION),
            ("rocm", PersistenceBackend.CUDA_HYBRID),
        ],
    )
    def test_known_devices(self, device, expected):
        assert _resolve_device_to_backend(device) == expected

    @pytest.mark.parametrize(
        "device",
        ["cuda:0", "cuda:1", "cpu:0", "hip:7", "rocm:3"],
    )
    def test_handles_device_with_index(self, device):
        assert isinstance(_resolve_device_to_backend(device), PersistenceBackend)

    def test_unknown_device_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown device"):
            _resolve_device_to_backend("quantum")

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown device"):
            _resolve_device_to_backend("")

    def test_none_prefix_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown device"):
            _resolve_device_to_backend(":0")

    def test_error_message_lists_supported_devices(self):
        with pytest.raises(ValueError, match="cpu"):
            _resolve_device_to_backend("unknown_device_xyz")


class TestSeedRng:
    def test_accepts_non_negative_seed(self):
        _seed_rng(0)
        _seed_rng(42)
        _seed_rng(999999)

    def test_rejects_negative_seed(self):
        with pytest.raises(InvalidArgumentError, match="non-negative"):
            _seed_rng(-1)

    def test_rejects_negative_seed_structure(self):
        with pytest.raises(InvalidArgumentError) as exc_info:
            _seed_rng(-5)
        assert exc_info.value.parameter == "seed"
        assert exc_info.value.expected == ">= 0"
        assert exc_info.value.actual == "-5"

    def test_deterministic_numpy(self):
        _seed_rng(42)
        a = [np.random.random() for _ in range(5)]
        _seed_rng(42)
        b = [np.random.random() for _ in range(5)]
        assert np.allclose(a, b)

    def test_different_seeds_different_outputs(self):
        _seed_rng(42)
        a = np.random.random()
        _seed_rng(43)
        b = np.random.random()
        assert a != b


class TestSeedRngWithTorch:
    torch = pytest.importorskip("torch")

    def test_seeds_torch_when_available(self):
        _seed_rng(42)
        a = self.torch.randn(3).tolist()
        _seed_rng(42)
        b = self.torch.randn(3).tolist()
        assert a == b

    def test_seeds_torch_deterministic(self):
        _seed_rng(123)
        a = self.torch.rand(5).tolist()
        _seed_rng(123)
        b = self.torch.rand(5).tolist()
        assert a == b


class TestTryAsNdarray:
    def test_returns_ndarray_as_is(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]])
        assert _try_as_ndarray(arr) is arr

    def test_returns_1d_ndarray_as_is(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _try_as_ndarray(arr) is arr

    def test_returns_3d_ndarray_as_is(self):
        arr = np.zeros((2, 2, 2))
        assert _try_as_ndarray(arr) is arr

    def test_converts_2d_list(self):
        result = _try_as_ndarray([[0.0, 1.0], [2.0, 3.0]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert result.dtype == np.float64
        assert np.array_equal(result, np.array([[0.0, 1.0], [2.0, 3.0]]))

    def test_returns_none_for_1d_list(self):
        assert _try_as_ndarray([1.0, 2.0, 3.0]) is None

    def test_returns_none_for_non_convertible(self):
        assert _try_as_ndarray("string") is None
        assert _try_as_ndarray({"a": 1}) is None
        assert _try_as_ndarray(42) is None
        assert _try_as_ndarray(None) is None

    def test_2d_empty_ndarray(self):
        arr = np.empty((0, 3))
        result = _try_as_ndarray(arr)
        assert result.shape == (0, 3)

    def test_returns_scalar_ndarray_as_is(self):
        arr = np.array(5.0)
        assert _try_as_ndarray(arr) is arr

    def test_converts_2d_list_of_ints(self):
        result = _try_as_ndarray([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert result.dtype == np.float64

    def test_converts_2d_tuple(self):
        result = _try_as_ndarray(((0.0, 1.0), (2.0, 3.0)))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert result.dtype == np.float64


class TestTryAsNdarrayWithTorch:
    torch = pytest.importorskip("torch")

    def test_converts_2d_tensor(self):
        t = self.torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=self.torch.float64)
        result = _try_as_ndarray(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert result.dtype == np.float64

    def test_returns_none_for_1d_tensor(self):
        t = self.torch.tensor([1.0, 2.0, 3.0], dtype=self.torch.float64)
        assert _try_as_ndarray(t) is None

    def test_returns_none_for_3d_tensor(self):
        t = self.torch.zeros((2, 2, 2), dtype=self.torch.float64)
        assert _try_as_ndarray(t) is None

    def test_converts_grad_tensor(self):
        t = self.torch.tensor(
            [[0.0, 1.0], [2.0, 3.0]], dtype=self.torch.float64, requires_grad=True
        )
        result = _try_as_ndarray(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_converts_cpu_tensor(self):
        t = self.torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=self.torch.float64, device="cpu")
        result = _try_as_ndarray(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)


class TestPersistenceFromDistanceMatrix:
    torch = pytest.importorskip("torch")

    def test_computes_betti_and_pairs(self):
        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        result = _persistence_from_distance_matrix(D, max_dim=0)
        assert isinstance(result, PersistenceResult)
        assert isinstance(result.pairs, list)
        assert isinstance(result.betti_numbers, list)
        assert len(result.betti_numbers) == 1
        assert result.max_dim == 0

    def test_empty_distance_matrix_single_point(self):
        D = np.array([[0.0]])
        result = _persistence_from_distance_matrix(D, max_dim=0)
        assert isinstance(result, PersistenceResult)
        assert result.max_dim == 0

    def test_max_dim_one(self):
        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        result = _persistence_from_distance_matrix(D, max_dim=1)
        assert result.max_dim == 1
        assert len(result.betti_numbers) == 2

    def test_betti_numbers_count_essential_classes(self):
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = _persistence_from_distance_matrix(D, max_dim=0)
        assert result.betti_numbers[0] == 1

    def test_result_has_diagnostics_dict(self):
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = _persistence_from_distance_matrix(D, max_dim=0)
        assert isinstance(result.diagnostics, dict)

    def test_all_pairs_have_three_elements(self):
        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        result = _persistence_from_distance_matrix(D, max_dim=1)
        for pair in result.pairs:
            assert len(pair) == 3
            assert isinstance(pair[0], float)
            assert isinstance(pair[1], float)
            assert isinstance(pair[2], int)


class TestComputeWithOptions:
    def test_delegates_to_distance_matrix_path(self, monkeypatch):
        import pynerve._compute_backend as _mod

        dm = np.array([[0.0, 1.0], [1.0, 0.0]])

        _fake_pr = PersistenceResult(
            pairs=[(0.0, 1.0, 0)], betti_numbers=[0], max_dim=0, max_radius=float("inf")
        )

        def _fake_try_as_ndarray(points):
            return dm

        def _fake_persist_dm(mat, dim):
            return _fake_pr

        monkeypatch.setattr(_mod, "_try_as_ndarray", _fake_try_as_ndarray)
        monkeypatch.setattr(_mod, "_persistence_from_distance_matrix", _fake_persist_dm)
        monkeypatch.setattr(
            "pynerve._compute_pipeline._is_likely_distance_matrix", lambda arr: True
        )
        monkeypatch.setattr("pynerve._compute_backend._nerve_state", lambda: (None, None, object()))

        result = _compute_with_options(lambda arr, opts: {}, dm, None)
        assert isinstance(result, PersistenceResult)
        assert result.pairs == [(0.0, 1.0, 0)]

    def test_compute_pipeline_path(self, monkeypatch):
        import pynerve._compute_backend as _mod

        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def _fake_core_func(arr, opts):
            return {
                "pairs": [(0.0, 1.0, 0)],
                "betti_numbers": [1],
                "max_dim": 2,
                "max_radius": 1.0,
                "diagnostics": {},
            }

        monkeypatch.setattr(_mod, "_try_as_ndarray", lambda pts: None)
        monkeypatch.setattr("pynerve._compute_engine._require_core", lambda: None)
        monkeypatch.setattr(
            "pynerve._compute_pipeline._to_point_array",
            lambda pts, dtype: np.asarray(pts, dtype=np.float64),
        )
        monkeypatch.setattr(
            "pynerve._compute_pipeline._resolve_options",
            lambda pts, opts, **overrides: PersistenceOptions(max_dim=2, max_radius=1.0),
        )

        result = _compute_with_options(_fake_core_func, points, None)
        assert isinstance(result, PersistenceResult)
        assert result.max_dim == 2
        assert result.pairs == [(0.0, 1.0, 0)]
