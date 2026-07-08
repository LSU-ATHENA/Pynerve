"""Coverage tests for pynerve._compute_api -- no C++ build required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pynerve._fallback_classes import (
    EventType,
    PersistenceBackend,
    PersistenceEngine,
    PersistenceMode,
    PersistenceOptions,
)
from pynerve._persistence_result import PersistenceResult

try:
    import torch as _torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    _torch = MagicMock()


def _torch_tensor(data, **kwargs):
    if _HAS_TORCH:
        return _torch.tensor(data, **kwargs)
    raise NotImplementedError


class TestResolveEngineForPoints:
    def test_non_auto_returns_engine_directly(self):
        from pynerve._compute_api import _resolve_engine_for_points

        pts = np.array([[1.0, 2.0, 3.0]])
        result = _resolve_engine_for_points(pts, PersistenceEngine.PH3, 2, None, None, None, None)
        assert result == PersistenceEngine.PH3

    def test_non_auto_ignores_other_args(self):
        from pynerve._compute_api import _resolve_engine_for_points

        pts = np.array([[1.0, 2.0]])
        result = _resolve_engine_for_points(
            pts, PersistenceEngine.PH5, 3, "cuda", "APPROX", "CPU_EXACT", _torch
        )
        assert result == PersistenceEngine.PH5

    @patch("pynerve._compute_api._auto_select_engine")
    def test_auto_with_numpy_resolves(self, mock_select):
        from pynerve._compute_api import _resolve_engine_for_points

        mock_select.return_value = PersistenceEngine.PH4
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = _resolve_engine_for_points(
            pts, PersistenceEngine.AUTO, 2, "cpu", "EXACT", "CPU_EXACT", None
        )
        assert result == PersistenceEngine.PH4
        mock_select.assert_called_once_with(2, 3, "cpu", "EXACT", "CPU_EXACT")

    @patch("pynerve._compute_api._auto_select_engine")
    def test_auto_with_list_of_lists(self, mock_select):
        from pynerve._compute_api import _resolve_engine_for_points

        mock_select.return_value = PersistenceEngine.PH3
        pts = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = _resolve_engine_for_points(pts, PersistenceEngine.AUTO, 2, None, None, None, None)
        assert result == PersistenceEngine.PH3
        mock_select.assert_called_once_with(3, 3, None, None, None)

    @patch("pynerve._compute_api._auto_select_engine")
    def test_auto_with_torch_tensor(self, mock_select):
        from pynerve._compute_api import _resolve_engine_for_points

        mock_select.return_value = PersistenceEngine.PH5
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        pts = _torch.randn(10, 3)
        result = _resolve_engine_for_points(
            pts, PersistenceEngine.AUTO, 2, "cuda", None, None, _torch
        )
        assert result == PersistenceEngine.PH5
        mock_select.assert_called_once_with(10, 3, "cuda", None, None)

    @patch("pynerve._compute_api._auto_select_engine")
    def test_auto_resolves_without_torch(self, mock_select):
        from pynerve._compute_api import _resolve_engine_for_points

        mock_select.return_value = PersistenceEngine.PH0
        pts = np.array([[1.0, 2.0]])
        result = _resolve_engine_for_points(pts, PersistenceEngine.AUTO, 1, None, None, None, None)
        assert result == PersistenceEngine.PH0
        mock_select.assert_called_once_with(1, 2, None, None, None)

    @patch("pynerve._compute_api._auto_select_engine")
    def test_auto_torch_import_block_hits_generic_path(self, mock_select):
        from pynerve._compute_api import _resolve_engine_for_points

        mock_select.return_value = PersistenceEngine.PH3
        pts = [1.0, 2.0, 3.0]
        result = _resolve_engine_for_points(pts, PersistenceEngine.AUTO, 2, None, None, None, None)
        assert result == PersistenceEngine.PH3
        mock_select.assert_called_once_with(3, 3, None, None, None)

    @patch("pynerve._compute_api._auto_select_engine")
    def test_auto_exception_during_estimation(self, mock_select):
        from pynerve._compute_api import _resolve_engine_for_points

        mock_select.return_value = PersistenceEngine.PH3

        class BadObj:
            def __len__(self):
                raise TypeError("no len")

        result = _resolve_engine_for_points(
            BadObj(), PersistenceEngine.AUTO, 2, None, None, None, None
        )
        assert result == PersistenceEngine.PH3
        mock_select.assert_called_once_with(0, 3, None, None, None)

    @patch("pynerve._compute_api._auto_select_engine")
    def test_auto_tensor_no_dimension(self, mock_select):
        from pynerve._compute_api import _resolve_engine_for_points

        mock_select.return_value = PersistenceEngine.PH3
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        pts = _torch.tensor([1.0, 2.0, 3.0])
        result = _resolve_engine_for_points(
            pts, PersistenceEngine.AUTO, 2, None, None, None, _torch
        )
        assert result == PersistenceEngine.PH3
        mock_select.assert_called_once_with(0, 0, None, None, None)


class TestMakeEngineFunc:
    def test_func_name_matches_engine(self):
        from pynerve._compute_api import _make_engine_func

        func = _make_engine_func(PersistenceEngine.PH0)
        assert func.__name__ == "compute_persistence_ph0"
        assert func.__qualname__ == "compute_persistence_ph0"

    def test_func_name_ph3(self):
        from pynerve._compute_api import _make_engine_func

        func = _make_engine_func(PersistenceEngine.PH3)
        assert func.__name__ == "compute_persistence_ph3"

    def test_func_name_ph4(self):
        from pynerve._compute_api import _make_engine_func

        func = _make_engine_func(PersistenceEngine.PH4)
        assert func.__name__ == "compute_persistence_up_to_dim_4"

    def test_func_name_ph5(self):
        from pynerve._compute_api import _make_engine_func

        func = _make_engine_func(PersistenceEngine.PH5)
        assert func.__name__ == "compute_persistence_up_to_dim_5"

    def test_func_name_ph6(self):
        from pynerve._compute_api import _make_engine_func

        func = _make_engine_func(PersistenceEngine.PH6)
        assert func.__name__ == "compute_persistence_up_to_dim_6"

    def test_unknown_engine_still_has_name(self):
        from pynerve._compute_api import _make_engine_func

        func = _make_engine_func(PersistenceEngine.AUTO)
        assert func.__name__ == "compute_persistence_auto"

    def test_docstring_set_for_known_engine(self):
        from pynerve._compute_api import _make_engine_func

        func = _make_engine_func(PersistenceEngine.PH0)
        assert func.__doc__ is not None
        assert "PH0 standard homology engine" in func.__doc__

    def test_docstring_set_for_unknown_engine(self):
        from pynerve._compute_api import _make_engine_func

        func = _make_engine_func(PersistenceEngine.AUTO)
        assert func.__doc__ is not None
        assert "AUTO engine" in func.__doc__

    @patch("pynerve._compute_api.compute_persistence")
    def test_created_func_delegates_to_compute_persistence(self, mock_compute):
        from pynerve._compute_api import _make_engine_func

        mock_compute.return_value = PersistenceResult(pairs=[(0.0, 1.0, 0)])
        func = _make_engine_func(PersistenceEngine.PH3)
        pts = np.array([[1.0, 2.0, 3.0]])
        result = func(pts, max_dim=1, seed=42)

        assert result.pairs == [(0.0, 1.0, 0)]
        mock_compute.assert_called_once_with(
            pts,
            None,
            engine=PersistenceEngine.PH3,
            max_dim=1,
            max_radius=None,
            mode=PersistenceMode.EXACT,
            backend=None,
            threads=None,
            device=None,
            seed=42,
            error_tolerance=None,
            dtype=None,
            max_radius_cap=None,
        )

    @patch("pynerve._compute_api.compute_persistence")
    def test_created_func_passes_options(self, mock_compute):
        from pynerve._compute_api import _make_engine_func

        mock_compute.return_value = PersistenceResult()
        func = _make_engine_func(PersistenceEngine.PH4)
        opts = PersistenceOptions(max_dim=3, threads=4)
        func(np.array([[1.0, 2.0]]), opts)

        assert mock_compute.call_args[0][1] is opts


class TestModuleLevelEngineFunctions:
    def test_compute_persistence_ph0_exists(self):
        from pynerve._compute_api import compute_persistence_ph0

        assert callable(compute_persistence_ph0)
        assert compute_persistence_ph0.__name__ == "compute_persistence_ph0"

    def test_compute_persistence_ph3_exists(self):
        from pynerve._compute_api import compute_persistence_ph3

        assert callable(compute_persistence_ph3)
        assert compute_persistence_ph3.__name__ == "compute_persistence_ph3"

    def test_compute_persistence_up_to_dim_4_exists(self):
        from pynerve._compute_api import compute_persistence_up_to_dim_4

        assert callable(compute_persistence_up_to_dim_4)
        assert compute_persistence_up_to_dim_4.__name__ == "compute_persistence_up_to_dim_4"

    def test_compute_persistence_up_to_dim_5_exists(self):
        from pynerve._compute_api import compute_persistence_up_to_dim_5

        assert callable(compute_persistence_up_to_dim_5)
        assert compute_persistence_up_to_dim_5.__name__ == "compute_persistence_up_to_dim_5"

    def test_compute_persistence_up_to_dim_6_exists(self):
        from pynerve._compute_api import compute_persistence_up_to_dim_6

        assert callable(compute_persistence_up_to_dim_6)
        assert compute_persistence_up_to_dim_6.__name__ == "compute_persistence_up_to_dim_6"


class TestComputePersistence:
    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_basic_call_delegates(self, mock_est, mock_state, mock_req, mock_res, mock_cwo):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 5
        mock_req.return_value = MagicMock()
        mock_res.return_value = lambda pts, opts: {"pairs": [(0.0, 1.0, 0)]}
        mock_cwo.return_value = PersistenceResult(
            pairs=[(0.0, 1.0, 0)],
            betti_numbers=[1],
            max_dim=1,
            max_radius=2.0,
        )

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 1.0]])
        result = compute_persistence(pts, max_dim=1, max_radius=2.0)

        assert result.pairs == [(0.0, 1.0, 0)]
        assert result.betti_numbers == [1]

    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._resolve_engine_for_points")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_passes_engine_argument(
        self, mock_est, mock_state, mock_req, mock_refp, mock_res, mock_cwo
    ):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 3
        mock_req.return_value = MagicMock()
        mock_refp.return_value = PersistenceEngine.PH4
        mock_res.return_value = lambda pts, opts: {}
        mock_cwo.return_value = PersistenceResult()

        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        compute_persistence(pts, engine=PersistenceEngine.PH4)

        mock_res.assert_called_once()
        called_engine = mock_res.call_args[0][1]
        assert called_engine == PersistenceEngine.PH4

    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._resolve_engine_for_points")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_string_engine_conversion_valid(
        self, mock_est, mock_state, mock_req, mock_refp, mock_res, mock_cwo
    ):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 3
        mock_req.return_value = MagicMock()
        mock_refp.return_value = PersistenceEngine.PH5
        mock_res.return_value = lambda pts, opts: {}
        mock_cwo.return_value = PersistenceResult()

        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        compute_persistence(pts, engine="ph5")
        called_engine = mock_res.call_args[0][1]
        assert called_engine == PersistenceEngine.PH5

        mock_refp.assert_called_once()

    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._resolve_engine_for_points")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_string_engine_conversion_uppercase(
        self, mock_est, mock_state, mock_req, mock_refp, mock_res, mock_cwo
    ):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 3
        mock_req.return_value = MagicMock()
        mock_refp.return_value = PersistenceEngine.PH0
        mock_res.return_value = lambda pts, opts: {}
        mock_cwo.return_value = PersistenceResult()

        pts = np.array([[1.0, 2.0, 3.0]])
        compute_persistence(pts, engine="PH0")
        called_engine = mock_res.call_args[0][1]
        assert called_engine == PersistenceEngine.PH0

    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._resolve_engine_for_points")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_string_engine_conversion_mixed_case(
        self, mock_est, mock_state, mock_req, mock_refp, mock_res, mock_cwo
    ):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 3
        mock_req.return_value = MagicMock()
        mock_refp.return_value = PersistenceEngine.PH3
        mock_res.return_value = lambda pts, opts: {}
        mock_cwo.return_value = PersistenceResult()

        pts = np.array([[1.0, 2.0, 3.0]])
        compute_persistence(pts, engine="Ph3")
        called_engine = mock_res.call_args[0][1]
        assert called_engine == PersistenceEngine.PH3

    def test_string_engine_conversion_invalid(self):
        from pynerve._compute_api import compute_persistence

        pts = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Unknown engine"):
            compute_persistence(pts, engine="nonexistent")

    def test_string_engine_conversion_empty(self):
        from pynerve._compute_api import compute_persistence

        pts = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Unknown engine"):
            compute_persistence(pts, engine="")

    def test_device_and_backend_warns(self):
        from pynerve._compute_api import compute_persistence

        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.warns(UserWarning):
            compute_persistence(
                pts, engine="ph0", device="cuda", backend=PersistenceBackend.CPU_EXACT
            )

    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._resolve_engine_for_points")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_all_parameters_forwarded(
        self, mock_est, mock_state, mock_req, mock_refp, mock_res, mock_cwo
    ):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 10
        mock_req.return_value = MagicMock()
        mock_refp.return_value = PersistenceEngine.PH5
        mock_res.return_value = lambda pts, opts: {}
        mock_cwo.return_value = PersistenceResult()

        pts = np.array([[i, i + 1, i + 2] for i in range(10)])
        with pytest.warns(UserWarning, match="device argument takes precedence"):
            compute_persistence(
                pts,
                max_dim=3,
                max_radius=5.0,
                mode=PersistenceMode.APPROX,
                backend=PersistenceBackend.CUDA_HYBRID,
                threads=8,
                device="cuda:0",
                seed=12345,
                error_tolerance=1e-5,
                dtype="float32",
                max_radius_cap=100.0,
            )

        mock_cwo.assert_called_once()
        overrides = mock_cwo.call_args[1]
        assert overrides["max_dim"] == 3
        assert overrides["max_radius"] == 5.0
        assert overrides["mode"] == PersistenceMode.APPROX
        assert overrides["backend"] == PersistenceBackend.CUDA_HYBRID
        assert overrides["threads"] == 8
        assert overrides["device"] == "cuda:0"
        assert overrides["seed"] == 12345
        assert overrides["error_tolerance"] == 1e-5
        assert overrides["dtype"] == "float32"
        assert overrides["max_radius_cap"] == 100.0

    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._resolve_engine_for_points")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_default_engine_is_auto(
        self, mock_est, mock_state, mock_req, mock_refp, mock_res, mock_cwo
    ):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 3
        mock_req.return_value = MagicMock()
        mock_refp.return_value = PersistenceEngine.PH4
        mock_res.return_value = lambda pts, opts: {}
        mock_cwo.return_value = PersistenceResult()

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        compute_persistence(pts)

        mock_refp.assert_called_once()
        engine_arg = mock_refp.call_args[0][1]
        assert engine_arg == PersistenceEngine.AUTO

    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_with_persistence_options(self, mock_est, mock_state, mock_req, mock_res, mock_cwo):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 3
        mock_req.return_value = MagicMock()
        mock_res.return_value = lambda pts, opts: {}
        mock_cwo.return_value = PersistenceResult()

        opts = PersistenceOptions(max_dim=4, threads=2, max_radius=3.0)
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        compute_persistence(pts, opts)

        mock_cwo.assert_called_once()
        assert mock_cwo.call_args[0][2] is opts

    @patch("pynerve._compute_api._compute_with_options")
    @patch("pynerve._compute_api._resolve_engine_func")
    @patch("pynerve._compute_api._require_core")
    @patch("pynerve._compute_api._nerve_state")
    @patch("pynerve._compute_api._estimate_n_points")
    def test_options_none_explicit(self, mock_est, mock_state, mock_req, mock_res, mock_cwo):
        from pynerve._compute_api import compute_persistence

        mock_state.return_value = (None, None, None)
        mock_est.return_value = 2
        mock_req.return_value = MagicMock()
        mock_res.return_value = lambda pts, opts: {}
        mock_cwo.return_value = PersistenceResult()

        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        compute_persistence(pts, options=None)

        mock_cwo.assert_called_once()
        assert mock_cwo.call_args[0][2] is None


class TestUpdatePersistence:
    @patch("pynerve._compute_api._to_events_list")
    @patch("pynerve._compute_api._require_core")
    def test_delegates_to_core(self, mock_req, mock_tel):
        from pynerve._compute_api import update_persistence

        mock_core = MagicMock()
        mock_core.update_persistence.return_value = {
            "pairs": [(0.0, 1.0, 0)],
            "betti_numbers": [1],
            "max_dim": 1,
            "max_radius": 2.0,
        }
        mock_req.return_value = mock_core
        mock_tel.return_value = [("add", [0, 1])]

        events = [(EventType.ADD, (0, 1))]
        result = update_persistence(events, max_dim=1)

        assert isinstance(result, PersistenceResult)
        assert result.pairs == [(0.0, 1.0, 0)]
        mock_tel.assert_called_once_with(events)
        mock_core.update_persistence.assert_called_once()

    @patch("pynerve._compute_api._to_events_list")
    @patch("pynerve._compute_api._require_core")
    def test_passes_overrides_to_options(self, mock_req, mock_tel):
        from pynerve._compute_api import update_persistence

        mock_core = MagicMock()
        mock_core.update_persistence.return_value = {
            "pairs": [],
            "betti_numbers": [],
            "max_dim": 0,
            "max_radius": 0.0,
        }
        mock_req.return_value = mock_core
        mock_tel.return_value = [("add", [0])]

        events = [(EventType.ADD, (0,))]
        update_persistence(
            events,
            max_dim=3,
            max_radius=5.0,
            mode=PersistenceMode.APPROX,
            backend=PersistenceBackend.CUDA_HYBRID,
            threads=4,
            device="cuda",
            seed=42,
            error_tolerance=0.001,
            max_radius_cap=50.0,
        )

        call_args = mock_core.update_persistence.call_args
        resolved = call_args[0][1]
        assert resolved.max_dim == 3
        assert resolved.max_radius == 5.0
        assert resolved.mode == PersistenceMode.APPROX
        assert resolved.backend == PersistenceBackend.CUDA_HYBRID
        assert resolved.threads == 4

    @patch("pynerve._compute_api._require_core")
    def test_string_engine_valid(self, mock_req):
        from pynerve._compute_api import update_persistence

        mock_core = MagicMock()
        mock_core.update_persistence.return_value = {
            "pairs": [],
            "betti_numbers": [],
            "max_dim": 0,
            "max_radius": 0.0,
        }
        mock_req.return_value = mock_core

        events = [(EventType.ADD, (0,))]
        result = update_persistence(events, engine="ph3")
        assert isinstance(result, PersistenceResult)

    @patch("pynerve._compute_api._require_core")
    def test_string_engine_uppercase_valid(self, mock_req):
        from pynerve._compute_api import update_persistence

        mock_core = MagicMock()
        mock_core.update_persistence.return_value = {
            "pairs": [],
            "betti_numbers": [],
            "max_dim": 0,
            "max_radius": 0.0,
        }
        mock_req.return_value = mock_core

        events = [(EventType.ADD, (0,))]
        result = update_persistence(events, engine="PH4")
        assert isinstance(result, PersistenceResult)

    @patch("pynerve._compute_api._require_core")
    def test_string_engine_invalid(self, mock_req):
        from pynerve._compute_api import update_persistence

        mock_core = MagicMock()
        mock_req.return_value = mock_core

        events = [(EventType.ADD, (0,))]
        with pytest.raises(ValueError, match="Unknown engine"):
            update_persistence(events, engine="bad_engine")

    @patch("pynerve._compute_api._require_core")
    def test_string_engine_empty(self, mock_req):
        from pynerve._compute_api import update_persistence

        mock_core = MagicMock()
        mock_req.return_value = mock_core

        events = [(EventType.ADD, (0,))]
        with pytest.raises(ValueError, match="Unknown engine"):
            update_persistence(events, engine="")

    @patch("pynerve._compute_api._to_events_list")
    @patch("pynerve._compute_api._require_core")
    def test_with_options_instance(self, mock_req, mock_tel):
        from pynerve._compute_api import update_persistence

        mock_core = MagicMock()
        mock_core.update_persistence.return_value = {
            "pairs": [],
            "betti_numbers": [],
            "max_dim": 0,
            "max_radius": 0.0,
        }
        mock_req.return_value = mock_core
        mock_tel.return_value = [("add", [0])]

        opts = PersistenceOptions(max_dim=2, threads=2)
        events = [(EventType.ADD, (0,))]
        update_persistence(events, opts)

        call_args = mock_core.update_persistence.call_args
        resolved = call_args[0][1]
        assert resolved.max_dim == 2
        assert resolved.threads == 2

    @patch("pynerve._compute_api._to_events_list")
    @patch("pynerve._compute_api._require_core")
    def test_multiple_events(self, mock_req, mock_tel):
        from pynerve._compute_api import update_persistence

        mock_core = MagicMock()
        mock_core.update_persistence.return_value = {
            "pairs": [(0.0, 1.0, 0), (0.5, 2.0, 1)],
            "betti_numbers": [1, 0],
            "max_dim": 1,
            "max_radius": 2.0,
        }
        mock_req.return_value = mock_core
        mock_tel.return_value = [
            ("add", [0]),
            ("add", [1]),
            ("add", [0, 1]),
        ]

        events = [
            (EventType.ADD, (0,)),
            (EventType.ADD, (1,)),
            (EventType.ADD, (0, 1)),
        ]
        result = update_persistence(events, max_dim=1)

        assert len(result.pairs) == 2
        assert result.betti_numbers == [1, 0]


class TestExports:
    def test_all_exports(self):
        from pynerve._compute_api import __all__

        assert "compute_persistence" in __all__
        assert "update_persistence" in __all__
        assert "compute_persistence_ph0" in __all__
        assert "compute_persistence_ph3" in __all__
        assert "compute_persistence_up_to_dim_4" in __all__
        assert "compute_persistence_up_to_dim_5" in __all__
        assert "compute_persistence_up_to_dim_6" in __all__


class TestEngineDoc:
    def test_all_engines_have_doc(self):
        from pynerve._compute_api import _ENGINE_DOC

        for engine in [
            PersistenceEngine.PH0,
            PersistenceEngine.PH3,
            PersistenceEngine.PH4,
            PersistenceEngine.PH5,
            PersistenceEngine.PH6,
        ]:
            assert engine in _ENGINE_DOC
            desc, details = _ENGINE_DOC[engine]
            assert isinstance(desc, str)
            assert len(desc) > 0
            assert isinstance(details, str)
            assert len(details) > 0

    def test_auto_not_in_doc(self):
        from pynerve._compute_api import _ENGINE_DOC

        assert PersistenceEngine.AUTO not in _ENGINE_DOC
