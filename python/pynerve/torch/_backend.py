"""Backend discovery and dispatch helpers for ``pynerve.torch``."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from types import TracebackType
from typing import Any, ParamSpec, TypeVar, cast

import torch

from ..exceptions import BackendRequiredError

P = ParamSpec("P")
T = TypeVar("T")
_BACKEND_RETRY_ERRORS = (
    AttributeError,
    BackendRequiredError,
    RuntimeError,
    TypeError,
)


class BackendDispatcher:
    """Dispatch calls through torch-native, core C++, then Python backends."""

    def __init__(self) -> None:
        self._torch_c: Any = None
        self._core_c: Any = None
        self._initialized: bool = False

    def _initialize(self) -> None:
        if self._initialized:
            return

        self._try_torch_c()
        self._try_core_c()

        self._initialized = True

    def _try_torch_c(self) -> None:
        if self._torch_c is not None:
            return
        try:
            import nerve_torch_internal as _torch_c_module  # noqa: PLC0415

            self._torch_c = _torch_c_module
        except ImportError:
            try:
                import pynerve_torch_internal as _torch_c_module  # noqa: PLC0415

                self._torch_c = _torch_c_module
            except ImportError:
                pass

    def _try_core_c(self) -> None:
        if self._core_c is not None:
            return
        try:
            import pynerve_internal as _core_c_module  # noqa: PLC0415

            self._core_c = _core_c_module
        except ImportError:
            pass

    def _ensure_backends(self) -> None:
        self._try_torch_c()
        self._try_core_c()
        self._initialized = True

    @property
    def torch_c_available(self) -> bool:
        self._ensure_backends()
        return self._torch_c is not None

    @property
    def core_c_available(self) -> bool:
        self._ensure_backends()
        return self._core_c is not None

    @property
    def any_backend_available(self) -> bool:
        return self.torch_c_available or self.core_c_available

    def dispatch(
        self,
        operation: str,
        torch_fn: Callable[[Any], T],
        core_fn: Callable[[Any], T],
        python_fn: Callable[[], T],
        warn_on_python: bool = False,
    ) -> T:
        """Dispatch to available backends in order."""
        self._initialize()
        backend_errors: list[tuple[str, Exception]] = []

        if self._torch_c is not None:
            try:
                return torch_fn(self._torch_c)
            except _BACKEND_RETRY_ERRORS as exc:
                backend_errors.append(("torch_c", exc))

        if self._core_c is not None:
            try:
                return core_fn(self._core_c)
            except _BACKEND_RETRY_ERRORS as exc:
                backend_errors.append(("core_c", exc))

        if warn_on_python:
            reason = ""
            if backend_errors:
                failed_backend, error = backend_errors[-1]
                reason = f" Last backend failure ({failed_backend}): {error}."
            warnings.warn(
                f"Using Python implementation for {operation}. "
                f"Install C++ extensions for better performance.{reason}",
                RuntimeWarning,
                stacklevel=3,
            )

        return python_fn()

    def dispatch_simple(
        self,
        operation: str,
        torch_attr: str,
        core_attr: str,
        python_impl: Callable[[], T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Dispatch by backend attribute names."""
        return cast(
            T,
            self.dispatch(
                operation=operation,
                torch_fn=lambda c: getattr(c, torch_attr)(*args, **kwargs),
                core_fn=lambda _c: getattr(torch.ops.pynerve, core_attr)(*args, **kwargs),
                python_fn=python_impl,
            ),
        )

    def get_torch_c_backend(self) -> Any:
        """Return the torch C++ backend module or None."""
        self._initialize()
        return self._torch_c

    def require_backend(self, backend: str) -> None:
        """Raise unless the requested backend is loaded."""
        self._initialize()

        if backend == "torch_c" and self._torch_c is None:
            raise BackendRequiredError("Torch-native C++ backend is not loaded.")
        if backend == "core_c" and self._core_c is None:
            raise BackendRequiredError("Core C++ backend is not loaded.")
        if backend not in {"torch_c", "core_c"}:
            raise ValueError("backend must be 'torch_c' or 'core_c'")


backend = BackendDispatcher()


def use_backend(backend_name: str) -> Callable[..., Any]:
    """Decorator that rejects calls unless ``backend_name`` is available."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            backend.require_backend(backend_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def with_python_backend(
    operation: str,
    torch_attr: str | None = None,
    core_attr: str | None = None,
    python_impl: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Add backend dispatch around a Python implementation."""
    if python_impl is None:
        raise ValueError("python_impl is required")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return cast(
                T,
                backend.dispatch(
                    operation=operation,
                    torch_fn=lambda c: (
                        getattr(c, torch_attr)(*args, **kwargs)
                        if torch_attr is not None
                        else func(*args, **kwargs)
                    ),
                    core_fn=lambda c: (
                        getattr(c, core_attr)(*args, **kwargs)
                        if core_attr is not None
                        else func(*args, **kwargs)
                    ),
                    python_fn=lambda: python_impl(*args, **kwargs),
                ),
            )

        return wrapper

    return decorator


class BackendContext:
    """Temporarily restrict backend availability."""

    def __init__(self, preferred_backend: str):
        self.preferred = preferred_backend
        self._original_torch_c = None
        self._original_core_c = None

    def __enter__(self) -> BackendContext:
        backend._initialize()
        self._original_torch_c = backend._torch_c
        self._original_core_c = backend._core_c

        if self.preferred == "python":
            backend._torch_c = None
            backend._core_c = None
        elif self.preferred == "torch_c":
            backend._core_c = None
        elif self.preferred == "core_c":
            backend._torch_c = None
        else:
            raise ValueError("preferred_backend must be 'python', 'torch_c', or 'core_c'")

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        backend._torch_c = self._original_torch_c
        backend._core_c = self._original_core_c


def get_backend_info() -> dict[str, Any]:
    """Return backend availability and version metadata."""
    info: dict[str, Any] = {
        "torch_c_available": backend.torch_c_available,
        "core_c_available": backend.core_c_available,
        "python_impl": True,
    }

    if backend._torch_c is not None:
        try:
            info["torch_c_version"] = getattr(backend._torch_c, "__version__", "unknown")
        except (AttributeError, RuntimeError):
            info["torch_c_version"] = "unknown"

    if backend._core_c is not None:
        try:
            info["core_c_version"] = getattr(backend._core_c, "__version__", "unknown")
        except (AttributeError, RuntimeError):
            info["core_c_version"] = "unknown"

    return info


__all__ = [
    "BackendDispatcher",
    "BackendContext",
    "backend",
    "use_backend",
    "with_python_backend",
    "get_backend_info",
]
