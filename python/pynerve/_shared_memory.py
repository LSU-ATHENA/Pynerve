"""Shared-memory array backed by multiprocessing shared memory."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from ._validation import validate_shape as _validate_shape


def _validate_batches(data_batches: list[np.ndarray], name: str) -> list[np.ndarray]:
    if isinstance(data_batches, (str, bytes)) or not isinstance(data_batches, Iterable):
        raise TypeError(f"{name} must be an iterable of arrays")
    result = list(data_batches)
    if result:
        _validate_shape(result[0].shape)
    return result


class SharedMemoryArray:
    """A numpy array backed by multiprocessing shared memory.

    Supports creation, attachment to existing blocks, and context-manager
    lifecycle. Usable across process boundaries via the shared memory name.

    :param shape: Shape of the array.
    :param dtype: Data type of the array elements.
    :param name: If given, attaches to an existing shared memory block instead
        of creating a new one.
    :raises TypeError: If *dtype* contains Python objects.
    :raises ValueError: If *name* is provided but empty.
    """

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype, name: str | None = None) -> None:
        self._closed: bool = True
        self.shape = _validate_shape(shape)
        self.dtype = np.dtype(dtype)
        if self.dtype.hasobject:
            raise TypeError("shared memory arrays cannot use object dtype")
        self.itemsize: int = self.dtype.itemsize
        self.nbytes: int = int(np.prod(self.shape, dtype=np.int64)) * self.itemsize

        self._owner: bool
        if name is None:
            self._shm = SharedMemory(create=True, size=max(1, self.nbytes))
            self.name = self._shm.name
            self._owner = True
        else:
            if not isinstance(name, str) or not name:
                raise ValueError("name must be a non-empty string")
            self._shm = SharedMemory(name=name)
            self.name = name
            self._owner = False

        self._array: np.ndarray | None = np.ndarray(
            self.shape, dtype=self.dtype, buffer=self._shm.buf
        )
        self._closed = False

    @classmethod
    def from_array(cls, array: np.ndarray) -> SharedMemoryArray:
        """Create a shared-memory copy of an existing array.

        :param array: Source array to copy into shared memory.
        :returns: A new :class:`SharedMemoryArray` populated with *array* data.
        :raises ValueError: If *array* is 0-dimensional.
        """
        array = np.asarray(array)
        if array.ndim == 0:
            raise ValueError("array must have at least one dimension")
        shm_array = cls(array.shape, array.dtype)
        shm_array.array[...] = array
        return shm_array

    @classmethod
    def attach(cls, name: str, shape: tuple[int, ...], dtype: np.dtype) -> SharedMemoryArray:
        """Attach to an existing shared memory block.

        :param name: Name of the existing shared memory block.
        :param shape: Expected array shape.
        :param dtype: Expected array data type.
        :returns: A new :class:`SharedMemoryArray` backed by the named block.
        """
        return cls(shape, dtype, name)

    @property
    def array(self) -> np.ndarray:
        """Return the underlying numpy view.

        :returns: The numpy array backed by shared memory.
        :raises RuntimeError: If the array has been closed.
        """
        if self._array is None:
            raise RuntimeError("SharedMemoryArray is closed")
        return self._array

    def __getitem__(self, key: Any) -> np.ndarray:
        """Index into the shared-memory array.

        :param key: Indexing key (slice, integer, mask, ...).
        :returns: The selected sub-array.
        :raises RuntimeError: If the array has been closed.
        """
        arr = self._array
        if arr is None:
            raise RuntimeError("SharedMemoryArray is closed")
        return arr[key]  # type: ignore[no-any-return]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Assign into the shared-memory array.

        :param key: Indexing key (slice, integer, mask, ...).
        :param value: Value to assign.
        :raises RuntimeError: If the array has been closed.
        """
        arr = self._array
        if arr is None:
            raise RuntimeError("SharedMemoryArray is closed")
        arr[key] = value

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Support ``np.asarray()`` on this instance.

        :param dtype: Optional dtype to cast to.
        :returns: A numpy array view (or copy if *dtype* differs).
        :raises RuntimeError: If the array has been closed.
        """
        arr = self._array
        if arr is None:
            raise RuntimeError("SharedMemoryArray is closed")
        return np.asarray(arr, dtype=dtype)

    def close(self) -> None:
        """Release the shared memory block.

        If this instance owns the block it is also unlinked. Safe to call
        multiple times.
        """
        if self._closed or not hasattr(self, "_shm"):
            return
        self._closed = True
        self._array = None
        self._shm.close()
        if self._owner:
            with suppress(FileNotFoundError):
                self._shm.unlink()

    def __repr__(self) -> str:
        return (
            f"SharedMemoryArray(shape={self.shape}, dtype={self.dtype}, "
            f"name={self.name!r}, owner={self._owner})"
        )

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> SharedMemoryArray:
        """Enter context manager.

        :returns: *self*.
        """
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Release resources on context exit."""
        self.close()
