"""System profiling and GPU diagnostics."""

from __future__ import annotations

import os
import platform
import sys
import traceback
import warnings
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal, TextIO


def profile_memory(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, dict[str, float]]:
    """Execute a function and measure its host memory impact.

    The function's result is returned alongside a dictionary with
    ``memory_before_mb``, ``memory_after_mb``, and ``memory_delta_mb``.

    :param func: The callable to profile.
    :param args: Positional arguments forwarded to ``func``.
    :param kwargs: Keyword arguments forwarded to ``func``.
    :returns: A ``(result, stats)`` tuple where ``stats`` contains the
        memory measurements in MB.
    :raises TypeError: If ``func`` is not callable.
    :raises ImportError: If ``psutil`` is not installed.
    """
    if not callable(func):
        raise TypeError("func must be callable")
    try:
        import psutil  # noqa: PLC0415

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        stats = {
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "memory_delta_mb": mem_after - mem_before,
        }

        return result, stats

    except ImportError as exc:
        raise ImportError(
            "psutil is required for memory profiling. Install it with: pip install psutil"
        ) from exc


class DebugMode:
    """Context manager that enables debug-friendly settings.

    Activates full warning display and optionally writes debug output
    (including caught exceptions) to a stream.

    :param print_intermediate: If ``True``, indicates intermediate results
        should be printed (consumer-defined).
    :param stream: A text stream for debug output (e.g. ``sys.stderr``);
        ``None`` suppresses output.
    :raises TypeError: If ``print_intermediate`` is not a boolean or if
        ``stream`` does not provide a ``write`` method.
    """

    def __init__(self, print_intermediate: bool = False, stream: TextIO | None = None):
        if not isinstance(print_intermediate, bool):
            raise TypeError("print_intermediate must be a boolean")
        if stream is not None and not callable(getattr(stream, "write", None)):
            raise TypeError("stream must provide a write method")
        self.print_intermediate = print_intermediate
        self.stream = stream
        self._original_warning_filter = list(warnings.filters)

    def __repr__(self) -> str:
        return (
            f"DebugMode(print_intermediate={self.print_intermediate}, "
            f"stream={type(self.stream).__name__ if self.stream else None})"
        )

    def __enter__(self) -> DebugMode:
        warnings.simplefilter("always")
        self._write("Debug mode enabled")
        self._write(f"Print intermediate: {self.print_intermediate}")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        warnings.filters = self._original_warning_filter

        if exc_type:
            self._write(f"Exception caught: {exc_type.__name__}")
            if self.stream is not None:
                traceback.print_exception(exc_type, exc_val, exc_tb, file=self.stream)

        self._write("Debug mode disabled")
        return False

    def _write(self, message: str) -> None:
        if self.stream is not None:
            self.stream.write(f"{message}\n")


_GPU_PROBE_ERRORS = (AttributeError, ImportError, OSError)


def check_gpu_availability() -> dict[str, Any]:
    """Probe GPU hardware via CuPy and return device information.

    :returns: A dictionary with keys ``cuda_available`` (bool),
        ``cuda_version`` (int or ``None``), ``device_count`` (int), and
        ``devices`` (list of dicts with ``id``, ``name``,
        ``total_memory_mb``).  If CuPy is not installed or no GPU is
        available, ``cuda_available`` is ``False`` and all other fields
        are their defaults.
    """
    result: dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }

    with suppress(*_GPU_PROBE_ERRORS):
        import cupy as cp  # noqa: PLC0415

        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        device_count = cp.cuda.runtime.getDeviceCount()
        devices = []

        for i in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            devices.append(
                {
                    "id": i,
                    "name": props["name"].decode()
                    if isinstance(props["name"], bytes)
                    else props["name"],
                    "total_memory_mb": props["totalGlobalMem"] / 1024 / 1024,
                }
            )
        result.update(
            cuda_available=True,
            cuda_version=cuda_version,
            device_count=device_count,
            devices=devices,
        )

    return result


def system_info() -> dict[str, Any]:
    """Gather system, Python, and Nerve version information.

    :returns: A dictionary with keys ``python_version``, ``platform``,
        ``processor``, ``cpu_count``, and ``gpu_info`` (result of
        :func:`check_gpu_availability`).  If Nerve is importable,
        ``pynerve_version`` is also included.
    """
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu_info": check_gpu_availability(),
    }

    with suppress(ImportError):
        from . import __version__ as _nerve_version  # noqa: PLC0415

        info["pynerve_version"] = _nerve_version

    return info
