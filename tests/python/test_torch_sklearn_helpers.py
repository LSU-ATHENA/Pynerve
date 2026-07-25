"""Tests for torch/sklearn_transformers.py helper functions and edge cases.

Covers _as_float_tensor, _validate_point_cloud, _validate_diagram_tensor,
_tensor_to_numpy, _single_diagram_tensor, _diagram_from_tensor,
_unpack_witness_sample, and fit_transform paths.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestAsFloatTensor:
    def test_float_tensor_passthrough(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor

        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = _as_float_tensor(t)
        assert torch.is_floating_point(result)
        assert torch.equal(result, t)

    def test_int_tensor_converted(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor

        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = _as_float_tensor(t)
        assert result.dtype == torch.float32

    def test_numpy_array_converted(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor

        result = _as_float_tensor(np.array([1.0, 2.0], dtype=np.float64))
        assert isinstance(result, torch.Tensor)
        assert torch.is_floating_point(result)

    def test_float_scalar_converted(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor

        result = _as_float_tensor(3.14)
        assert isinstance(result, torch.Tensor)
        assert result.item() == pytest.approx(3.14)

    def test_int_scalar_converted(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor

        result = _as_float_tensor(5)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_non_finite_raises(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor

        with pytest.raises(ValueError, match="finite"):
            _as_float_tensor(torch.tensor([float("inf")], dtype=torch.float32))

    def test_nan_raises(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor

        with pytest.raises(ValueError, match="finite"):
            _as_float_tensor(torch.tensor([float("nan")], dtype=torch.float32))

    def test_empty_tensor_ok(self):
        from pynerve.torch.sklearn_transformers import _as_float_tensor

        result = _as_float_tensor(torch.empty(0, dtype=torch.float32))
        assert result.numel() == 0


class TestValidatePointCloud:
    def test_valid_2d(self):
        from pynerve.torch.sklearn_transformers import _validate_point_cloud

        cloud = torch.rand(5, 3, dtype=torch.float32)
        _validate_point_cloud(cloud)

    def test_1d_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_point_cloud

        cloud = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="shape"):
            _validate_point_cloud(cloud)

    def test_3d_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_point_cloud

        cloud = torch.zeros(2, 3, 4, dtype=torch.float32)
        with pytest.raises(ValueError, match="shape"):
            _validate_point_cloud(cloud)

    def test_empty_points_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_point_cloud

        cloud = torch.empty((0, 3), dtype=torch.float32)
        with pytest.raises(ValueError, match="non-empty"):
            _validate_point_cloud(cloud)

    def test_empty_coords_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_point_cloud

        cloud = torch.empty((5, 0), dtype=torch.float32)
        with pytest.raises(ValueError, match="non-empty"):
            _validate_point_cloud(cloud)

    def test_custom_name_in_message(self):
        from pynerve.torch.sklearn_transformers import _validate_point_cloud

        cloud = torch.tensor([1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="landmarks"):
            _validate_point_cloud(cloud, "landmarks")


class TestValidateDiagramTensor:
    def test_valid_2d(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
        result = _validate_diagram_tensor(t)
        assert result.shape == (2, 2)

    def test_int_converted_to_float(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32)
        result = _validate_diagram_tensor(t)
        assert torch.is_floating_point(result)

    def test_1d_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([0.0, 1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="shape"):
            _validate_diagram_tensor(t)

    def test_single_column_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="at least 2"):
            _validate_diagram_tensor(t)

    def test_nan_birth_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite"):
            _validate_diagram_tensor(t)

    def test_nan_death_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
        with pytest.raises(ValueError, match="NaN"):
            _validate_diagram_tensor(t)

    def test_death_before_birth_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="greater than"):
            _validate_diagram_tensor(t)

    def test_inf_death_ok(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[0.0, float("inf")]], dtype=torch.float32)
        result = _validate_diagram_tensor(t)
        assert result.shape == (1, 2)

    def test_3_columns_with_dim(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=torch.float32)
        result = _validate_diagram_tensor(t)
        assert result.shape == (2, 3)

    def test_negative_dim_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="non-negative"):
            _validate_diagram_tensor(t)

    def test_nan_dim_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.tensor([[0.0, 1.0, float("nan")]], dtype=torch.float32)
        with pytest.raises(ValueError, match="finite"):
            _validate_diagram_tensor(t)

    def test_empty_returns_unchanged(self):
        from pynerve.torch.sklearn_transformers import _validate_diagram_tensor

        t = torch.empty((0, 2), dtype=torch.float32)
        result = _validate_diagram_tensor(t)
        assert result.shape == (0, 2)


class TestTensorToNumpy:
    def test_tensor_to_numpy(self):
        from pynerve.torch.sklearn_transformers import _tensor_to_numpy

        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = _tensor_to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, [1.0, 2.0, 3.0])

    def test_numpy_passthrough(self):
        from pynerve.torch.sklearn_transformers import _tensor_to_numpy

        arr = np.array([1.0, 2.0])
        result = _tensor_to_numpy(arr)
        assert result is arr


class TestSingleDiagramTensor:
    def test_2d_tensor(self):
        from pynerve.torch.sklearn_transformers import _single_diagram_tensor

        t = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
        result = _single_diagram_tensor(t)
        assert result.shape == (2, 2)

    def test_3d_single_batch_squeezed(self):
        from pynerve.torch.sklearn_transformers import _single_diagram_tensor

        t = torch.tensor([[[0.0, 1.0], [0.5, 2.0]]], dtype=torch.float32)
        result = _single_diagram_tensor(t)
        assert result.shape == (2, 2)

    def test_3d_multi_batch_raises(self):
        from pynerve.torch.sklearn_transformers import _single_diagram_tensor

        t = torch.tensor([
            [[0.0, 1.0], [0.5, 2.0]],
            [[0.0, 3.0], [1.0, 4.0]],
        ], dtype=torch.float32)
        with pytest.raises(ValueError, match="batched"):
            _single_diagram_tensor(t)

    def test_persistence_diagram_object(self):
        from pynerve.torch._diagram import PersistenceDiagram
        from pynerve.torch.sklearn_transformers import _single_diagram_tensor

        # PersistenceDiagram requires 3 columns (birth, death, dim)
        tensor = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        pd = PersistenceDiagram(tensor)
        result = _single_diagram_tensor(pd)
        assert result.dim() == 2


class TestUnpackWitnessSample:
    def test_dict_with_landmarks_witnesses(self):
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        landmarks = torch.rand(3, 2)
        witnesses = torch.rand(5, 2)
        result = PersistenceTransformer._unpack_witness_sample(
            {"landmarks": landmarks, "witnesses": witnesses}
        )
        assert torch.equal(result[0], landmarks)
        assert torch.equal(result[1], witnesses)

    def test_dict_missing_landmarks_raises(self):
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        with pytest.raises(ValueError, match="landmarks"):
            PersistenceTransformer._unpack_witness_sample({"witnesses": torch.rand(3, 2)})

    def test_tuple(self):
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        lm = torch.rand(3, 2)
        wt = torch.rand(5, 2)
        result = PersistenceTransformer._unpack_witness_sample((lm, wt))
        assert torch.equal(result[0], lm)
        assert torch.equal(result[1], wt)

    def test_list_of_two(self):
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        lm = torch.rand(3, 2)
        wt = torch.rand(5, 2)
        result = PersistenceTransformer._unpack_witness_sample([lm, wt])
        assert torch.equal(result[0], lm)
        assert torch.equal(result[1], wt)

    def test_invalid_type_raises(self):
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        with pytest.raises(ValueError, match="Witness"):
            PersistenceTransformer._unpack_witness_sample(42)

    def test_wrong_length_tuple_raises(self):
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        with pytest.raises(ValueError, match="Witness"):
            PersistenceTransformer._unpack_witness_sample((torch.rand(3, 2),))


class TestValidateSequence:
    def test_string_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_sequence

        with pytest.raises(TypeError, match="sequence"):
            _validate_sequence("x", "not a sequence")

    def test_bytes_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_sequence

        with pytest.raises(TypeError, match="sequence"):
            _validate_sequence("x", b"bytes")

    def test_empty_raises(self):
        from pynerve.torch.sklearn_transformers import _validate_sequence

        with pytest.raises(ValueError):
            _validate_sequence("x", [])

    def test_valid_list(self):
        from pynerve.torch.sklearn_transformers import _validate_sequence

        _validate_sequence("x", [1, 2, 3])  # should not raise


class TestPersistenceTransformerFitTransform:
    def test_fit_transform_returns_self_then_diagrams(self):
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer(max_dim=0)
        clouds = [torch.rand(5, 2)]
        # fit_transform delegates to fit then transform; transform needs C++ backend
        # so we just test fit
        result = pt.fit(clouds)
        assert result is pt


class TestVectorizationTransformerFitTransform:
    def test_fit_transform_with_silhouette(self):
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer(method="silhouette", num_samples=8)
        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = vt.fit_transform([diagram, diagram])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 8)

    def test_fit_transform_with_landscape(self):
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer(method="landscape", k=3, num_samples=10)
        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = vt.fit_transform([diagram])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1

    def test_transform_with_persistence_image(self):
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer(method="image", resolution=(10, 10), sigma=1.0)
        diagram = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
        result = vt.transform([diagram])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 10, 10)

    def test_transform_with_heat(self):
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer(method="heat", sigma=1.0)
        diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
        result = vt.transform([diagram])
        assert isinstance(result, np.ndarray)

    def test_transform_with_histogram(self):
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer(method="histogram", num_bins=10)
        diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
        result = vt.transform([diagram])
        assert isinstance(result, np.ndarray)

    def test_get_vectorization_fn_unknown_after_init(self):
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer(method="landscape")
        vt.method = "bogus"
        with pytest.raises(ValueError, match="Unknown"):
            vt._get_vectorization_fn()


class TestStatisticsTransformerFitTransform:
    def test_fit_transform_with_tensor(self):
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer(dims=[0])
        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
        result = st.fit_transform([diagram, diagram])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)

    def test_transform_with_persistence_diagram_object(self):
        from pynerve.torch._diagram import PersistenceDiagram
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer(dims=[0])
        # PersistenceDiagram requires 3 columns (birth, death, dim)
        tensor = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
        pd = PersistenceDiagram(tensor)
        result = st.transform([pd])
        assert isinstance(result, np.ndarray)

    def test_get_feature_names_out_with_features(self):
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer(dims=[0], features=["total"])
        names = st.get_feature_names_out()
        assert isinstance(names, np.ndarray)
        assert len(names) > 0

    def test_get_feature_names_out_no_torch(self):
        """When torch is not None (it is installed), the probe path runs."""
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer(dims=[0])
        names = st.get_feature_names_out()
        assert isinstance(names, np.ndarray)
