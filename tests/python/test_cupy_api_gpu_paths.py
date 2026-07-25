"""Tests for _cupy_api.py — HAS_CUPY=True GPU paths.

Complements test_cupy_fallback.py (which tests HAS_CUPY=False).
Tests the cupy-enabled branches in _validate_point_cloud, compute_diagrams_cupy,
and batch_diagrams_cupy via the mock_gpu_deps fixture.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.usefixtures("mock_gpu_deps")
class TestValidatePointCloudCupy:
    def test_validate_point_cloud_2d_host(self):
        """2D numpy array passes host validation."""
        from pynerve._cupy_api import _validate_point_cloud

        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_point_cloud(pts)
        assert result is pts

    def test_validate_point_cloud_0d_host_empty(self):
        """Empty rows on host rejected."""
        from pynerve._cupy_api import _validate_point_cloud
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="non-empty"):
            _validate_point_cloud(np.empty((0, 2)))

    def test_validate_point_cloud_2d_cupy_mock(self):
        """A cupy-like 2D array passes cupy validation branch."""
        from pynerve._cupy_api import _validate_point_cloud

        cupy_arr = MagicMock()
        cupy_arr.ndim = 2
        cupy_arr.shape = (3, 2)
        cupy_arr.dtype = np.float32
        del cupy_arr.get  # no .get() method → falls through to cp.asarray branch

        # The cupy branch needs HAS_CUPY=True and cp.isfinite to return True
        from pynerve import _cupy_api as api_mod
        from pynerve import _cupy_compat

        with patch.object(_cupy_compat, "HAS_CUPY", True), \
             patch.object(api_mod, "HAS_CUPY", True), \
             patch.object(api_mod, "cp") as mock_cp:
            mock_cp.isfinite.return_value.all.return_value.item.return_value = True
            mock_cp.asarray.return_value = MagicMock()

            result = _validate_point_cloud(cupy_arr)
            # Since cupy_arr has ndim == 2 and no .get(), it uses cp.asarray
            assert result is cupy_arr

    def test_validate_point_cloud_cupy_with_get(self):
        """Cupy array with .get() method uses cp.isfinite directly."""
        from pynerve._cupy_api import _validate_point_cloud
        from pynerve import _cupy_compat

        cupy_arr = MagicMock()
        cupy_arr.ndim = 2
        cupy_arr.shape = (2, 2)
        cupy_arr.dtype = np.float64
        cupy_arr.get.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])

        from pynerve import _cupy_api as api_mod

        with patch.object(_cupy_compat, "HAS_CUPY", True), \
             patch.object(api_mod, "HAS_CUPY", True), \
             patch.object(api_mod, "cp") as mock_cp:
            mock_cp.isfinite.return_value.all.return_value.item.return_value = True

            result = _validate_point_cloud(cupy_arr)
            assert result is cupy_arr

    def test_validate_point_cloud_cupy_non_finite_with_get(self):
        """Cupy array with non-finite values raises ValidationError."""
        from pynerve._cupy_api import _validate_point_cloud
        from pynerve.exceptions import ValidationError
        from pynerve import _cupy_compat

        cupy_arr = MagicMock()
        cupy_arr.ndim = 2
        cupy_arr.shape = (2, 2)
        cupy_arr.dtype = np.float32

        from pynerve import _cupy_api as api_mod

        with patch.object(_cupy_compat, "HAS_CUPY", True), \
             patch.object(api_mod, "HAS_CUPY", True), \
             patch.object(api_mod, "cp") as mock_cp:
            mock_cp.isfinite.return_value.all.return_value.item.return_value = False

            with pytest.raises(ValidationError, match="finite"):
                _validate_point_cloud(cupy_arr)


class TestValidatePointCloudsCupy:
    def test_validate_point_clouds_with_invalid_inner(self):
        """_validate_point_clouds rejects list with invalid inner cloud."""
        from pynerve._cupy_api import _validate_point_clouds
        from pynerve.exceptions import ValidationError

        bad_cloud = np.empty((0, 2))  # empty → rejected
        with pytest.raises(ValidationError, match="non-empty"):
            _validate_point_clouds([bad_cloud])


class TestComputeDiagramsCupy:
    def test_compute_diagrams_cupy_array_path(self):
        """compute_diagrams_cupy with cupy array uses CuPyPersistence."""
        from pynerve._cupy_api import compute_diagrams_cupy
        from pynerve import _cupy_api as api_mod
        from pynerve import _cupy_compat

        cupy_pts = MagicMock()
        cupy_pts.ndim = 2
        cupy_pts.shape = (3, 2)
        cupy_pts.dtype = np.float32
        del cupy_pts.get

        mock_result = {"pairs": [(0.0, 1.0)], "betti_numbers": [1, 0]}

        with patch.object(_cupy_compat, "HAS_CUPY", True), \
             patch.object(api_mod, "cp") as mock_cp, \
             patch.object(api_mod, "to_cupy", return_value=cupy_pts), \
             patch.object(api_mod, "HAS_CUPY", True):
            mock_cp.isfinite.return_value.all.return_value.item.return_value = True
            mock_cp.asarray.return_value = MagicMock()

            mock_computer = MagicMock()
            mock_computer.compute_persistence_cupy.return_value = mock_result

            with patch.object(api_mod, "CuPyPersistence", return_value=mock_computer):
                result = compute_diagrams_cupy(cupy_pts, max_radius=1.0, max_dim=1, device_id=0)

            assert isinstance(result, dict)
            assert "pairs" in result
            mock_computer.compute_persistence_cupy.assert_called_once()

    def test_compute_diagrams_numpy_array_with_cupy_available(self):
        """NumPy array is converted to cupy when HAS_CUPY=True."""
        from pynerve._cupy_api import compute_diagrams_cupy
        from pynerve import _cupy_api as api_mod
        from pynerve import _cupy_compat

        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        cupy_pts = MagicMock()

        mock_result = {"pairs": [], "betti_numbers": [1]}

        with patch.object(_cupy_compat, "HAS_CUPY", True), \
             patch.object(api_mod, "cp") as mock_cp, \
             patch.object(api_mod, "to_cupy", return_value=cupy_pts), \
             patch.object(api_mod, "HAS_CUPY", True):
            mock_cp.isfinite.return_value.all.return_value.item.return_value = True
            mock_cp.asarray.return_value = MagicMock()

            mock_computer = MagicMock()
            mock_computer.compute_persistence_cupy.return_value = mock_result

            with patch.object(api_mod, "CuPyPersistence", return_value=mock_computer):
                result = compute_diagrams_cupy(pts, max_dim=1, max_radius=None, device_id=0)

            assert isinstance(result, dict)
            api_mod.to_cupy.assert_called_once_with(pts, device_id=0)


class TestBatchDiagramsCupy:
    def test_batch_diagrams_with_max_dist_alias(self):
        """max_dist is remapped to max_radius."""
        from pynerve._cupy_api import batch_diagrams_cupy
        from pynerve import _cupy_api as api_mod

        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        mock_result = {"pairs": [], "betti_numbers": [1]}

        with patch.object(api_mod, "HAS_CUPY", False), \
             patch.object(api_mod, "_compute_core_persistence", return_value=mock_result) as mock_core:
            result = batch_diagrams_cupy([pts], max_dist=2.5, max_dim=2, device_id=0)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_core.assert_called_once_with(pts, 2.5, 2)

    def test_batch_diagrams_cupy_array_path(self):
        """batch_diagrams_cupy with cupy arrays uses CuPyPersistence.batch_diagrams_cupy."""
        from pynerve._cupy_api import batch_diagrams_cupy
        from pynerve import _cupy_api as api_mod
        from pynerve import _cupy_compat

        cupy_cloud = MagicMock()
        cupy_cloud.ndim = 2
        cupy_cloud.shape = (3, 2)
        cupy_cloud.dtype = np.float32
        del cupy_cloud.get

        mock_result = {"pairs": [], "betti_numbers": [1]}
        mock_results = [mock_result, mock_result]

        with patch.object(_cupy_compat, "HAS_CUPY", True), \
             patch.object(api_mod, "cp") as mock_cp, \
             patch.object(api_mod, "to_cupy", return_value=cupy_cloud), \
             patch.object(api_mod, "HAS_CUPY", True):
            mock_cp.isfinite.return_value.all.return_value.item.return_value = True
            mock_cp.asarray.return_value = MagicMock()

            mock_computer = MagicMock()
            mock_computer.batch_diagrams_cupy.return_value = mock_results

            with patch.object(api_mod, "CuPyPersistence", return_value=mock_computer):
                result = batch_diagrams_cupy(
                    [cupy_cloud, cupy_cloud], max_radius=1.0, max_dim=1, device_id=0
                )

            assert len(result) == 2
            mock_computer.batch_diagrams_cupy.assert_called_once()

    def test_batch_diagrams_validates_max_dim(self):
        """Negative max_dim raises ValueError."""
        from pynerve._cupy_api import batch_diagrams_cupy

        pts = np.array([[0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="non-negative"):
            batch_diagrams_cupy([pts], max_dim=-1)

    def test_batch_diagrams_validates_max_radius(self):
        """Negative max_radius raises ValueError."""
        from pynerve._cupy_api import batch_diagrams_cupy

        pts = np.array([[0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="non-negative"):
            batch_diagrams_cupy([pts], max_radius=-5.0)


class TestValidateMaxDim:
    def test_validate_max_dim_zero(self):
        """max_dim=0 is valid (non-negative)."""
        from pynerve._cupy_api import _validate_max_dim

        assert _validate_max_dim(0) == 0

    def test_validate_max_dim_positive(self):
        from pynerve._cupy_api import _validate_max_dim

        assert _validate_max_dim(3) == 3
