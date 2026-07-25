"""Tests for torch/_distance_core_impl.py and torch/_backend.py."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestWassersteinDistance:
    def test_identical_diagrams(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0], [2.0, 5.0]])
        d2 = torch.tensor([[0.0, 1.0], [2.0, 5.0]])
        result = diagram_wasserstein(d1, d2)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_single_pair_different(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.tensor([[0.0, 3.0]])
        result = diagram_wasserstein(d1, d2)
        assert result.item() > 0.0

    def test_empty_diagrams(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d1 = torch.empty((0, 2))
        d2 = torch.empty((0, 2))
        result = diagram_wasserstein(d1, d2)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_p(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance

        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        d2 = torch.tensor([[0.0, 3.0], [0.0, 4.0]])
        metric = WassersteinDistance(p=1.0, q=1.0)
        result = metric(d1, d2)
        assert result.item() > 0.0

    def test_invalid_p_negative(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance

        with pytest.raises(ValueError, match="p must be"):
            WassersteinDistance(p=-1.0)

    def test_invalid_p_nan(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance

        with pytest.raises(ValueError, match="p must be"):
            WassersteinDistance(p=float("nan"))

    def test_invalid_q(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance

        with pytest.raises(ValueError, match="q must be"):
            WassersteinDistance(q=0.0)

    def test_one_empty_one_nonempty(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d1 = torch.empty((0, 2))
        d2 = torch.tensor([[0.0, 1.0]])
        result = diagram_wasserstein(d1, d2)
        assert result.item() >= 0.0

    def test_custom_p_q(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.tensor([[0.0, 2.0]])
        result = diagram_wasserstein(d1, d2, p=1.0, q=1.0)
        assert result.item() > 0.0


class TestBottleneckDistance:
    def _bottleneck(self, d1, d2):
        """Compute bottleneck distance using Python fallback directly."""
        from pynerve.torch._distance_core_impl import _bottleneck_python, _validate_distance_diagram

        t1 = _validate_distance_diagram(d1)
        t2 = _validate_distance_diagram(d2)
        return _bottleneck_python(t1, t2)

    def test_identical_diagrams(self):
        d1 = torch.tensor([[0.0, 1.0], [2.0, 5.0]])
        d2 = torch.tensor([[0.0, 1.0], [2.0, 5.0]])
        result = self._bottleneck(d1, d2)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_diagrams(self):
        d1 = torch.tensor([[0.0, 1.0]])
        d2 = torch.tensor([[0.0, 3.0]])
        result = self._bottleneck(d1, d2)
        assert result.item() > 0.0

    def test_empty_diagrams(self):
        d1 = torch.empty((0, 2))
        d2 = torch.empty((0, 2))
        result = self._bottleneck(d1, d2)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_one_empty_one_nonempty(self):
        d1 = torch.empty((0, 2))
        d2 = torch.tensor([[0.0, 1.0]])
        result = self._bottleneck(d1, d2)
        assert result.item() >= 0.0


class TestDistanceMetricValidation:
    def test_1d_input_unsqueezed(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d = torch.tensor([0.0, 1.0])
        result = diagram_wasserstein(d, d)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_3d_diagram_rejected(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        # 3D tensor with last dim = 1 goes through _valid_rows -> _validate_stat_diagram
        # which raises ValueError for < 2 columns
        d = torch.tensor([[[0.0]]])  # shape (1, 1, 1)
        with pytest.raises(ValueError, match="columns"):
            diagram_wasserstein(d, d)

    def test_single_column_rejected(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d = torch.tensor([[0.0], [1.0]])
        with pytest.raises(Exception, match="2 columns|at least"):
            diagram_wasserstein(d, d)

    def test_inf_death_filtered(self):
        from pynerve.torch._distance_core_impl import diagram_wasserstein

        d1 = torch.tensor([[0.0, 1.0], [0.0, float("inf")]])
        d2 = torch.tensor([[0.0, 1.0]])
        result = diagram_wasserstein(d1, d2)
        assert torch.isfinite(result).item()

    def test_extract_tensor_from_object_with_diagrams(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance

        class FakeDiagram:
            def __init__(self):
                self.diagrams = torch.tensor([[[0.0, 1.0]]])

        metric = WassersteinDistance()
        fake = FakeDiagram()
        result = metric(fake, fake)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_extract_tensor_from_object_with_underscore_diagrams(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance

        class FakeDiagram:
            def __init__(self):
                self._diagrams = torch.tensor([[[0.0, 1.0]]])

        metric = WassersteinDistance()
        fake = FakeDiagram()
        result = metric(fake, fake)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_extract_tensor_unsupported_type(self):
        from pynerve.torch._distance_core_impl import WassersteinDistance
        from pynerve.exceptions import ValidationError

        metric = WassersteinDistance()
        with pytest.raises(ValidationError):
            metric(42, 42)


class TestSortDiagramByPersistence:
    def test_sort(self):
        from pynerve.torch._distance_core_impl import _sort_diagram_by_persistence

        d = torch.tensor([[0.0, 1.0], [0.0, 5.0], [0.0, 2.0]])
        sorted_d = _sort_diagram_by_persistence(d)
        pers = sorted_d[:, 1] - sorted_d[:, 0]
        assert pers[0] >= pers[1] >= pers[2]

    def test_single_row(self):
        from pynerve.torch._distance_core_impl import _sort_diagram_by_persistence

        d = torch.tensor([[0.0, 1.0]])
        result = _sort_diagram_by_persistence(d)
        assert torch.equal(result, d)


class TestBackendDispatcher:
    def test_backend_properties(self):
        from pynerve.torch._backend import backend

        assert isinstance(backend.torch_c_available, bool)
        assert isinstance(backend.core_c_available, bool)
        assert isinstance(backend.any_backend_available, bool)

    def test_get_backend_info(self):
        from pynerve.torch._backend import get_backend_info

        info = get_backend_info()
        assert "torch_c_available" in info
        assert "core_c_available" in info
        assert "python_impl" in info

    def test_require_backend_invalid_name(self):
        from pynerve.torch._backend import backend

        with pytest.raises(ValueError, match="backend must be"):
            backend.require_backend("invalid")

    def test_require_backend_torch_c(self):
        from pynerve.torch._backend import backend

        if not backend.torch_c_available:
            with pytest.raises(Exception, match="Torch-native|not loaded"):
                backend.require_backend("torch_c")

    def test_require_backend_core_c(self):
        from pynerve.torch._backend import backend

        if not backend.core_c_available:
            with pytest.raises(Exception, match="Core C|not loaded"):
                backend.require_backend("core_c")

    def test_dispatch_python_fallback(self):
        from pynerve.torch._backend import backend

        result = backend.dispatch(
            "test_op",
            torch_fn=lambda c: "torch",
            core_fn=lambda c: "core",
            python_fn=lambda: "python",
        )
        assert isinstance(result, str)

    def test_dispatch_warn_on_python(self):
        from pynerve.torch._backend import backend
        from pynerve.torch._backend import BackendContext

        with BackendContext("python"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                backend.dispatch(
                    "test_warn_op",
                    torch_fn=lambda c: "torch",
                    core_fn=lambda c: "core",
                    python_fn=lambda: "python",
                    warn_on_python=True,
                )
                assert any("Python implementation" in str(warning.message) for warning in w)

    def test_get_torch_c_backend(self):
        from pynerve.torch._backend import backend

        result = backend.get_torch_c_backend()
        assert result is None or hasattr(result, "__module__")


class TestBackendContext:
    def test_python_context(self):
        from pynerve.torch._backend import backend, BackendContext

        with BackendContext("python"):
            assert backend._torch_c is None
            assert backend._core_c is None
        # After exit, restored

    def test_invalid_context(self):
        from pynerve.torch._backend import BackendContext

        ctx = BackendContext("invalid")
        with pytest.raises(ValueError, match="preferred_backend"):
            ctx.__enter__()

    def test_context_restores_state(self):
        from pynerve.torch._backend import backend, BackendContext

        original_torch = backend._torch_c
        original_core = backend._core_c
        with BackendContext("python"):
            pass
        assert backend._torch_c == original_torch
        assert backend._core_c == original_core


class TestUseBackendDecorator:
    def test_use_backend_rejects(self):
        from pynerve.torch._backend import backend, use_backend

        @use_backend("torch_c")
        def my_func():
            return "ok"

        if not backend.torch_c_available:
            with pytest.raises(Exception, match="Torch-native|not loaded"):
                my_func()
