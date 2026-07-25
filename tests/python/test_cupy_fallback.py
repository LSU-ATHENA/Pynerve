"""Tests for cupy modules — validation, no-cupy fallback, and error handling.

When HAS_CUPY is False (no GPU/cupy installed), all cupy modules
raise RuntimeError or fall back to NumPy/host computation.
These tests exercise those paths without requiring a GPU.

NOTE: This file does NOT use the mock_gpu_deps fixture because that
fixture mocks cupy to a MagicMock, making HAS_CUPY=True.  Instead we
patch _cupy_compat.HAS_CUPY to False where needed.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _force_no_cupy():
    """Force HAS_CUPY=False for all tests in this file.

    Patches each consuming module directly because `from ._cupy_compat import
    HAS_CUPY` creates a local binding that patching the source module
    would not affect.
    """
    mods = ("_cupy_compat", "_cupy_persistence", "_cupy_convert", "_cupy_memory", "_cupy_api")
    with ExitStack() as stack:
        stack.enter_context(patch("pynerve._cupy_compat.cp", None))
        for m in mods:
            stack.enter_context(patch(f"pynerve.{m}.HAS_CUPY", False))
        yield


class TestCuPyPersistenceInit:
    def test_init_no_cupy_succeeds(self):
        """CuPyPersistence.__init__ does not raise when cupy is absent —
        it just skips the GPU device setup.  The RuntimeError is raised
        by individual methods, not the constructor."""
        from pynerve._cupy_persistence import CuPyPersistence

        computer = CuPyPersistence(device_id=0)
        assert computer.device_id == 0

    def test_init_negative_device(self):
        from pynerve._cupy_persistence import CuPyPersistence

        with pytest.raises(ValueError, match="non-negative"):
            CuPyPersistence(device_id=-1)


class TestCuPyPersistenceValidation:
    def test_validate_core_points_valid(self):
        from pynerve._cupy_persistence import _validate_core_points

        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_core_points(pts)
        assert result.shape == (2, 2)

    def test_validate_core_points_1d(self):
        from pynerve._cupy_persistence import _validate_core_points

        with pytest.raises(ValueError, match="2D"):
            _validate_core_points(np.array([1.0, 2.0]))

    def test_validate_core_points_empty(self):
        from pynerve._cupy_persistence import _validate_core_points

        with pytest.raises(ValueError, match="non-empty"):
            _validate_core_points(np.empty((0, 2)))

    def test_validate_core_points_complex(self):
        from pynerve._cupy_persistence import _validate_core_points

        with pytest.raises(TypeError, match="real"):
            _validate_core_points(np.array([[1.0 + 2.0j, 3.0]]))

    def test_validate_core_points_non_finite(self):
        from pynerve._cupy_persistence import _validate_core_points

        with pytest.raises(ValueError, match="finite"):
            _validate_core_points(np.array([[1.0, float("inf")]]))

    def test_validate_core_points_non_numeric(self):
        from pynerve._cupy_persistence import _validate_core_points

        with pytest.raises(TypeError, match="real"):
            _validate_core_points(np.array([["a", "b"]]))

    def test_validate_persistence_image_diagrams_not_array(self):
        from pynerve._cupy_persistence import _validate_persistence_image_diagrams

        with pytest.raises(TypeError, match="array"):
            _validate_persistence_image_diagrams(42)

    def test_compute_core_persistence_negative_dim(self):
        from pynerve._cupy_persistence import _compute_core_persistence

        with pytest.raises(ValueError, match="non-negative"):
            _compute_core_persistence(np.array([[0.0, 0.0]]), None, -1)

    def test_pairwise_distances_no_cupy(self):
        from pynerve._cupy_persistence import CuPyPersistence

        computer = CuPyPersistence.__new__(CuPyPersistence)
        computer.device_id = 0
        with pytest.raises(RuntimeError, match="CuPy required"):
            computer.pairwise_distances_cupy(np.array([[0.0, 0.0]]))

    def test_build_vr_complex_no_cupy(self):
        from pynerve._cupy_persistence import CuPyPersistence

        computer = CuPyPersistence.__new__(CuPyPersistence)
        computer.device_id = 0
        with pytest.raises(RuntimeError, match="CuPy required"):
            computer.build_vr_complex_cupy(np.array([[0.0, 0.0]]), 1.0)

    def test_compute_persistence_no_cupy(self):
        from pynerve._cupy_persistence import CuPyPersistence

        computer = CuPyPersistence.__new__(CuPyPersistence)
        computer.device_id = 0
        with pytest.raises(RuntimeError, match="CuPy required"):
            computer.compute_persistence_cupy(np.array([[0.0, 0.0]]))

    def test_persistence_image_no_cupy(self):
        from pynerve._cupy_persistence import CuPyPersistence

        computer = CuPyPersistence.__new__(CuPyPersistence)
        computer.device_id = 0
        with pytest.raises(RuntimeError, match="CuPy required"):
            computer.persistence_image_cupy(np.array([[0.0, 1.0]]))

    def test_batch_diagrams_no_cupy(self):
        from pynerve._cupy_persistence import CuPyPersistence

        computer = CuPyPersistence.__new__(CuPyPersistence)
        computer.device_id = 0
        with pytest.raises(RuntimeError, match="CuPy required"):
            computer.batch_diagrams_cupy([np.array([[0.0, 0.0]])])


class TestCuPyConvert:
    def test_validate_target_type_valid(self):
        from pynerve._cupy_convert import _validate_target_type

        for t in ("numpy", "torch", "tensorflow", "jax"):
            assert _validate_target_type(t) == t

    def test_validate_target_type_invalid(self):
        from pynerve._cupy_convert import _validate_target_type

        with pytest.raises(ValueError, match="Unknown target"):
            _validate_target_type("bad")

    def test_require_cupy_no_cupy(self):
        from pynerve._cupy_convert import _require_cupy

        with pytest.raises(RuntimeError, match="CuPy required"):
            _require_cupy()

    def test_is_cupy_array_not_cupy(self):
        from pynerve._cupy_convert import _is_cupy_array

        assert _is_cupy_array(np.array([1.0])) is False

    def test_to_cupy_object_dtype(self):
        from pynerve._cupy_convert import to_cupy

        with pytest.raises(TypeError, match="object"):
            to_cupy(np.array([object()], dtype=object))

    def test_to_cupy_no_cupy(self):
        from pynerve._cupy_convert import to_cupy

        with pytest.raises(RuntimeError, match="CuPy required"):
            to_cupy(np.array([1.0, 2.0]))

    def test_from_cupy_no_cupy(self):
        from pynerve._cupy_convert import from_cupy

        with pytest.raises(RuntimeError, match="CuPy required"):
            from_cupy(np.array([1.0]))

    def test_from_cupy_invalid_target(self):
        from pynerve._cupy_convert import from_cupy

        with pytest.raises(ValueError, match="Unknown target"):
            from_cupy(np.array([1.0]), target_type="bad")

    def test_dlpack_capsule_no_dlpack(self):
        from pynerve._cupy_convert import _dlpack_capsule

        with pytest.raises(TypeError, match="DLPack"):
            _dlpack_capsule(42)


class TestCuPyMemory:
    def test_validate_dtype_object(self):
        from pynerve._cupy_memory import _validate_dtype

        with pytest.raises(TypeError, match="object"):
            _validate_dtype(np.dtype(object))

    def test_validate_dtype_valid(self):
        from pynerve._cupy_memory import _validate_dtype

        result = _validate_dtype(np.dtype(np.float32))
        assert result == np.float32

    def test_gpu_buffer_no_cupy(self):
        from pynerve._cupy_memory import GPUBuffer

        with pytest.raises(RuntimeError, match="CuPy required"):
            GPUBuffer(size=10)

    def test_gpu_buffer_negative_size(self):
        from pynerve._cupy_memory import GPUBuffer

        with pytest.raises(ValueError, match="non-negative"):
            GPUBuffer(size=-1)

    def test_cuda_stream_no_cupy(self):
        from pynerve._cupy_memory import CudaStream

        with pytest.raises(RuntimeError, match="CuPy required"):
            CudaStream()

    def test_cuda_stream_non_bool(self):
        from pynerve._cupy_memory import CudaStream

        with pytest.raises(TypeError, match="boolean"):
            CudaStream(non_blocking="yes")

    def test_unified_memory_no_cupy(self):
        from pynerve._cupy_memory import UnifiedMemoryBuffer

        with pytest.raises(RuntimeError, match="CuPy required"):
            UnifiedMemoryBuffer(size=10)

    def test_unified_memory_negative_size(self):
        from pynerve._cupy_memory import UnifiedMemoryBuffer

        with pytest.raises(ValueError, match="non-negative"):
            UnifiedMemoryBuffer(size=-1)


class TestCuPyApiValidation:
    def test_validate_max_dim_negative(self):
        from pynerve._cupy_api import _validate_max_dim

        with pytest.raises(ValueError, match="non-negative"):
            _validate_max_dim(-1)

    def test_validate_point_cloud_1d(self):
        from pynerve._cupy_api import _validate_point_cloud
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="2D"):
            _validate_point_cloud(np.array([1.0, 2.0]))

    def test_validate_point_cloud_empty(self):
        from pynerve._cupy_api import _validate_point_cloud
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="non-empty"):
            _validate_point_cloud(np.empty((0, 2)))

    def test_validate_point_cloud_non_numeric(self):
        from pynerve._cupy_api import _validate_point_cloud
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="numeric"):
            _validate_point_cloud(np.array([["a", "b"]]))

    def test_validate_point_cloud_non_finite(self):
        from pynerve._cupy_api import _validate_point_cloud
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="finite"):
            _validate_point_cloud(np.array([[1.0, float("inf")]]))

    def test_validate_point_clouds_not_iterable(self):
        from pynerve._cupy_api import _validate_point_clouds
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="iterable"):
            _validate_point_clouds(42)

    def test_validate_point_clouds_string(self):
        from pynerve._cupy_api import _validate_point_clouds
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="iterable"):
            _validate_point_clouds("bad")

    def test_compute_diagrams_no_cupy(self):
        from pynerve._cupy_api import compute_diagrams_cupy

        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        result = compute_diagrams_cupy(pts, max_radius=2.0, max_dim=1)
        assert isinstance(result, dict)
        assert "pairs" in result

    def test_compute_diagrams_negative_device(self):
        from pynerve._cupy_api import compute_diagrams_cupy

        with pytest.raises(ValueError, match="non-negative"):
            compute_diagrams_cupy(np.array([[0.0, 0.0]]), device_id=-1)

    def test_batch_diagrams_no_cupy(self):
        from pynerve._cupy_api import batch_diagrams_cupy

        clouds = [
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64),
        ]
        results = batch_diagrams_cupy(clouds, max_radius=2.0, max_dim=1)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_batch_diagrams_negative_device(self):
        from pynerve._cupy_api import batch_diagrams_cupy

        with pytest.raises(ValueError, match="non-negative"):
            batch_diagrams_cupy([], device_id=-1)
