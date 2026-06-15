"""Async GPU transfer helpers."""

from __future__ import annotations

from contextlib import nullcontext
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from ._cupy_compat import HAS_CUPY, cp

if TYPE_CHECKING:
    import cupy


def _require_cupy_transfer() -> None:
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for async GPU transfer. Install with: pip install cupy")


class AsyncGPUTransfer:
    """CUDA stream-backed host/device transfers.

    :param device_id: CUDA device index.
    :raises TypeError: If *device_id* is not an integer.
    :raises ValueError: If *device_id* is negative.
    """

    def __init__(self, device_id: int = 0) -> None:
        if isinstance(device_id, bool) or not isinstance(device_id, Integral):
            raise TypeError("device_id must be an integer")
        device_id = int(device_id)
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        self.device_id: int = device_id

    async def upload_async(self, host_array: np.ndarray, stream: Any = None) -> cupy.ndarray:
        """Upload a host array to the configured CUDA device.

        :param host_array: Source array on the host.
        :param stream: Optional CUDA stream. Defaults to a new stream.
        :returns: Device-side copy of the array.
        :raises RuntimeError: If CuPy is not installed.
        :raises TypeError: If *host_array* uses object dtype.
        """
        _require_cupy_transfer()
        host_array = np.asarray(host_array)
        if host_array.dtype.hasobject:
            raise TypeError("host_array cannot use object dtype")

        with cp.cuda.Device(self.device_id):
            stream_context = nullcontext(stream) if stream else cp.cuda.Stream()
            with stream_context as active_stream:
                device_array = cp.empty(host_array.shape, dtype=host_array.dtype)
                device_array.set(host_array, stream=active_stream)
                return device_array

    async def download_async(self, device_array: cupy.ndarray, stream: Any = None) -> np.ndarray:
        """Download a device array to host memory.

        :param device_array: Array on the CUDA device.
        :param stream: Optional CUDA stream. Defaults to a new stream.
        :returns: Host-side copy of the array.
        :raises RuntimeError: If CuPy is not installed.
        """
        _require_cupy_transfer()

        with cp.cuda.Device(self.device_id):
            stream_context = nullcontext(stream) if stream else cp.cuda.Stream()
            with stream_context as active_stream:
                host_array = device_array.get(stream=active_stream)
                return host_array  # type: ignore[no-any-return]
