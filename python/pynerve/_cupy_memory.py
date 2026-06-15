"""GPU memory management utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._cupy_compat import HAS_CUPY, cp
from ._validation import validate_nonnegative_int

if TYPE_CHECKING:
    import cupy


def _validate_device_id(device_id: int) -> int:
    return validate_nonnegative_int(device_id, "device_id")


def _validate_dtype(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise TypeError("GPU buffers cannot use object dtype")
    return dtype


class GPUBuffer:
    """Device buffer with explicit host transfers.

    Allocates a contiguous GPU memory region that can be populated from
    a host array and read back synchronously or asynchronously.
    """

    def __init__(self, size: int, dtype: np.dtype = np.float32, device_id: int = 0) -> None:  # type: ignore[assignment]
        """Allocate a GPU buffer.

        :param size: Number of elements.
        :param dtype: NumPy dtype of each element (default ``float32``).
        :param device_id: Target GPU device ID (default 0).
        :raises RuntimeError: If CuPy is not installed.
        :raises TypeError: If *dtype* uses object references.
        :raises ValueError: If *size* is negative.
        """
        size = validate_nonnegative_int(size, "size")
        dtype = _validate_dtype(dtype)
        device_id = _validate_device_id(device_id)
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install with: pip install cupy")

        self.size: int = size
        self.dtype: np.dtype = np.dtype(dtype)
        self.device_id: int = device_id

        with cp.cuda.Device(device_id):
            self._buffer: Any = cp.empty(size, dtype=self.dtype)

    def upload(self, host_array: np.ndarray, stream: Any = None) -> None:
        """Copy data from host to the GPU buffer.

        :param host_array: Host array. Will be cast to the buffer's dtype
            and reshaped to 1D.
        :param stream: Optional CuPy stream for asynchronous transfer.
        :raises ValueError: If *host_array* size does not match the buffer
            size.
        """
        host_array = np.asarray(host_array, dtype=self.dtype)
        if host_array.size != self.size:
            raise ValueError(f"host array has {host_array.size} values, expected {self.size}")
        host_array = host_array.reshape((self.size,))
        if stream:
            self._buffer.set(host_array, stream=stream)
        else:
            self._buffer.set(host_array)

    def download(self, stream: Any = None) -> np.ndarray:
        """Copy data from the GPU buffer back to host.

        :param stream: Optional CuPy stream for synchronous download.
        :returns: A 1D NumPy array with the buffer contents.
        """
        result: np.ndarray = self._buffer.get(stream=stream)
        return result

    def as_cupy(self) -> cupy.ndarray:
        """Return a view of the underlying CuPy array.

        :returns: A CuPy ndarray referencing the GPU buffer memory.
        """
        return self._buffer

    def __array__(self) -> np.ndarray:
        return self.download()


class CudaStream:
    """Context manager for a CUDA stream.

    Creates and enters a CuPy stream on ``__enter__``, synchronising on
    ``__exit__``.
    """

    def __init__(self, non_blocking: bool = True) -> None:
        """Configure the CUDA stream.

        :param non_blocking: Whether to enable non-blocking stream
            behaviour (default ``True``).
        :raises RuntimeError: If CuPy is not installed.
        :raises TypeError: If *non_blocking* is not a boolean.
        """
        if not isinstance(non_blocking, bool):
            raise TypeError("non_blocking must be a boolean")
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install with: pip install cupy")

        self.non_blocking: bool = non_blocking
        self._stream: Any = None

    def __enter__(self) -> Any:
        """Create and enter the CUDA stream.

        :returns: The underlying ``cp.cuda.Stream`` instance.
        """
        self._stream = cp.cuda.Stream(non_blocking=self.non_blocking)
        self._stream.use()
        return self._stream

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        if self._stream:
            self._stream.synchronize()


class UnifiedMemoryBuffer:
    """Pinned (unified) host buffer exposed as NumPy and CuPy views.

    Provides zero-copy access to the same physical memory from both CPU
    and GPU via separate views.
    """

    def __init__(self, size: int, dtype: np.dtype = np.float32) -> None:  # type: ignore[assignment]
        """Allocate pinned host memory accessible from both CPU and GPU.

        :param size: Number of elements.
        :param dtype: NumPy dtype of each element (default ``float32``).
        :raises RuntimeError: If CuPy is not installed.
        :raises TypeError: If *dtype* uses object references.
        :raises ValueError: If *size* is negative.
        """
        size = validate_nonnegative_int(size, "size")
        dtype = _validate_dtype(dtype)
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install with: pip install cupy")

        self.size: int = size
        self.dtype: np.dtype = np.dtype(dtype)

        self._array: Any = cp.cuda.alloc_pinned_memory(size * self.dtype.itemsize)
        self._cupy_array: Any = cp.ndarray(
            (size,),
            dtype=self.dtype,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(self._array.ptr, size * self.dtype.itemsize, self),
                0,
            ),
        )

    def cpu_view(self) -> np.ndarray:
        """Return a NumPy view of the pinned host memory.

        :returns: A 1D NumPy array backed by pinned memory.
        """
        return np.frombuffer(self._array, dtype=self.dtype, count=self.size)

    def gpu_view(self) -> cupy.ndarray:
        """Return a CuPy view of the same memory for GPU access.

        :returns: A CuPy ndarray backed by the same pinned memory.
        """
        return self._cupy_array

    def __del__(self) -> None:
        free = getattr(getattr(self, "_array", None), "free", None)
        if callable(free):
            free()
