"""Tests for torch/_backend.py -- remaining uncovered paths.

Covers: dispatch_simple, dispatch with backend_errors, get_backend_info
with version attributes, BackendContext torch_c/core_c variants, and
require_backend no-raise paths.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest


class TestDispatchSimple:
    def test_dispatch_simple_python_fallback(self):
        """dispatch_simple falls back to python_impl when no backends."""
        from pynerve.torch._backend import backend, BackendContext

        with BackendContext("python"):
            result = backend.dispatch_simple(
                operation="test_simple",
                torch_attr="test_fn",
                core_attr="test_fn",
                python_impl=lambda: 42,
            )
        assert result == 42

    def test_dispatch_simple_passes_args(self):
        """dispatch_simple forwards *args and **kwargs to the dispatched fn."""
        from pynerve.torch._backend import backend, BackendContext

        with BackendContext("python"):
            result = backend.dispatch_simple(
                "test_args", "add", "add", lambda: 7, 3, 4
            )
        assert result == 7


class TestDispatchBackendErrors:
    def test_dispatch_warn_with_errors(self):
        """warn_on_python=True includes last backend failure in warning message."""
        from pynerve.torch._backend import backend

        original_torch = backend._torch_c
        original_core = backend._core_c
        original_initialized = backend._initialized
        try:
            backend._torch_c = MagicMock()
            backend._core_c = None  # _initialized=True prevents _initialize from importing
            backend._initialized = True

            # Make the torch_c dispatch raise
            def _failing_torch_fn(c):
                raise RuntimeError("simulated torch_c failure")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = backend.dispatch(
                    operation="test_err",
                    torch_fn=_failing_torch_fn,
                    core_fn=lambda c: "core",
                    python_fn=lambda: "python",
                    warn_on_python=True,
                )
                assert result == "python"
                assert len(w) >= 1
                msg = str(w[0].message)
                assert "Python implementation" in msg
                assert "torch_c" in msg
        finally:
            backend._torch_c = original_torch
            backend._core_c = original_core
            backend._initialized = original_initialized

    def test_dispatch_core_fallback_when_torch_raises(self):
        """When torch_c raises, core_c is tried next before python."""
        from pynerve.torch._backend import backend

        fake_core = MagicMock()
        fake_core.some_attr = lambda *a, **kw: "core_result"

        original_torch = backend._torch_c
        original_core = backend._core_c
        original_initialized = backend._initialized
        try:
            backend._torch_c = "fake_torch"
            backend._core_c = fake_core
            backend._initialized = True

            def _failing_torch_fn(c):
                raise RuntimeError("torch failed")

            result = backend.dispatch(
                operation="test_chain",
                torch_fn=_failing_torch_fn,
                core_fn=lambda c: c.some_attr(),
                python_fn=lambda: "python",
            )
            assert result == "core_result"
        finally:
            backend._torch_c = original_torch
            backend._core_c = original_core
            backend._initialized = original_initialized


class TestGetBackendInfo:
    def test_backend_info_with_versions(self):
        """get_backend_info extracts version from mock backends with __version__."""
        from pynerve.torch._backend import get_backend_info, backend

        original_torch = backend._torch_c
        original_core = backend._core_c
        original_initialized = backend._initialized
        try:
            mock_torch = MagicMock()
            mock_torch.__version__ = "1.2.3"
            mock_core = MagicMock()
            mock_core.__version__ = "4.5.6"
            backend._torch_c = mock_torch
            backend._core_c = mock_core
            backend._initialized = True

            info = get_backend_info()
            assert info["torch_c_available"] is True
            assert info["core_c_available"] is True
            assert info["torch_c_version"] == "1.2.3"
            assert info["core_c_version"] == "4.5.6"
        finally:
            backend._torch_c = original_torch
            backend._core_c = original_core
            backend._initialized = original_initialized

    def test_backend_info_version_attribute_error(self):
        """get_backend_info handles AttributeError when reading version."""
        from pynerve.torch._backend import get_backend_info, backend

        original_torch = backend._torch_c
        original_initialized = backend._initialized
        try:
            mock_torch = MagicMock()
            del mock_torch.__version__
            backend._torch_c = mock_torch
            backend._core_c = MagicMock()
            backend._initialized = True

            info = get_backend_info()
            assert info["torch_c_version"] == "unknown"
        finally:
            backend._torch_c = original_torch
            backend._initialized = original_initialized

    def test_backend_info_version_runtime_error(self):
        """get_backend_info handles RuntimeError when reading version."""
        from pynerve.torch._backend import get_backend_info, backend

        original_torch = backend._torch_c
        original_initialized = backend._initialized
        try:
            mock_torch = MagicMock()
            type(mock_torch).__version__ = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
            backend._torch_c = mock_torch
            backend._core_c = MagicMock()
            backend._initialized = True

            info = get_backend_info()
            assert info["torch_c_version"] == "unknown"
        finally:
            backend._torch_c = original_torch
            backend._initialized = original_initialized


class TestBackendContextVariants:
    def test_torch_c_context(self):
        """BackendContext('torch_c') nullifies core_c but keeps torch_c."""
        from pynerve.torch._backend import backend, BackendContext

        original_torch = backend._torch_c
        original_core = backend._core_c
        original_initialized = backend._initialized
        try:
            backend._torch_c = "fake_torch"
            backend._core_c = "fake_core"
            backend._initialized = True

            with BackendContext("torch_c"):
                assert backend._torch_c is not None
                assert backend._core_c is None
        finally:
            backend._torch_c = original_torch
            backend._core_c = original_core
            backend._initialized = original_initialized

    def test_core_c_context(self):
        """BackendContext('core_c') nullifies torch_c but keeps core_c."""
        from pynerve.torch._backend import backend, BackendContext

        original_torch = backend._torch_c
        original_core = backend._core_c
        original_initialized = backend._initialized
        try:
            backend._torch_c = "fake_torch"
            backend._core_c = "fake_core"
            backend._initialized = True

            with BackendContext("core_c"):
                assert backend._torch_c is None
                assert backend._core_c is not None
        finally:
            backend._torch_c = original_torch
            backend._core_c = original_core
            backend._initialized = original_initialized

    def test_context_restores_on_exception(self):
        """BackendContext restores state even when an exception is raised inside."""
        from pynerve.torch._backend import backend, BackendContext

        original_torch = backend._torch_c
        original_core = backend._core_c
        original_initialized = backend._initialized
        try:
            backend._torch_c = "fake_torch"
            backend._core_c = "fake_core"
            backend._initialized = True

            with pytest.raises(ValueError, match="test exception"):
                with BackendContext("python"):
                    raise ValueError("test exception")
            assert backend._torch_c == "fake_torch"
            assert backend._core_c == "fake_core"
        finally:
            backend._torch_c = original_torch
            backend._core_c = original_core
            backend._initialized = original_initialized


class TestRequireBackend:
    def test_require_backend_no_raise_when_available(self):
        """require_backend does not raise when the backend is available."""
        from pynerve.torch._backend import backend

        original_torch = backend._torch_c
        original_initialized = backend._initialized
        try:
            backend._torch_c = MagicMock()
            backend._core_c = MagicMock()
            backend._initialized = True
            backend.require_backend("torch_c")
            backend.require_backend("core_c")
        finally:
            backend._torch_c = original_torch
            backend._initialized = original_initialized
