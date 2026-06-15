"""Comprehensive tests for pynerve.torch subpackage.

Covers the public API surface: lazy imports, PersistenceDiagram, diagram
distance functions, data utilities, nn layers, preprocessing, statistics,
vectorization, kernels, mapper, sklearn transformers, tensorboard logging,
and visualisation helpers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from pynerve.exceptions import ValidationError  # noqa: E402


def _torch_backend_available() -> bool:
    """Check if the PyTorch C++ extension (pynerve_torch_internal) is present."""
    try:
        import nerve_torch_internal  # noqa: F401

        return True
    except ImportError:
        try:
            import pynerve_torch_internal  # noqa: F401

            return True
        except ImportError:
            return False


_torch_backend = pytest.mark.skipif(
    not _torch_backend_available(),
    reason="pynerve_torch_internal not available",
)


def _core_backend_available() -> bool:
    """Check if the core C++ extension (pynerve_internal) is present."""
    try:
        import pynerve_internal  # noqa: F401

        return True
    except ImportError:
        return False


_core_backend = pytest.mark.skipif(
    not _core_backend_available(),
    reason="pynerve_internal not available",
)


def _networkx_available() -> bool:
    try:
        import networkx  # noqa: F401

        return True
    except ImportError:
        return False


def _make_diagram_tensor(
    batch: int = 2,
    pairs: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """Create a valid persistence diagram tensor (birth, death, dim) with birth < death."""
    torch.manual_seed(seed)
    births = torch.rand(batch, pairs, 1)
    deaths = births + torch.rand(batch, pairs, 1) + 0.1
    dims = torch.randint(0, 2, (batch, pairs, 1)).float()
    return torch.cat([births, deaths, dims], dim=-1)


def _make_2d_diagram(pairs: int = 3, seed: int = 42) -> torch.Tensor:
    """Create a 2D (unbatched) diagram tensor with (birth, death) columns."""
    torch.manual_seed(seed)
    births = torch.rand(pairs, 1)
    deaths = births + torch.rand(pairs, 1) + 0.1
    return torch.cat([births, deaths], dim=-1)


def _make_point_cloud(
    n_points: int = 8,
    dim: int = 3,
    batch: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if batch == 1:
        return torch.rand(n_points, dim)
    return torch.rand(batch, n_points, dim)


# Lazy import mechanism


# helpers


# sklearn_transformers


class TestPersistenceTransformer:
    """PersistenceTransformer construction and validation."""

    def test_construct_default(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer()
        assert pt.complex_type == "vr"
        assert pt.max_dim == 1

    def test_construct_invalid_complex_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        with pytest.raises((ValueError, TypeError, ValidationError)):
            PersistenceTransformer(complex_type="bogus")

    def test_fit_validates_sequence(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer()
        with pytest.raises((TypeError, ValueError, ValidationError)):
            pt.fit("not a sequence")

    def test_fit_empty_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer()
        with pytest.raises((TypeError, ValueError)):
            pt.fit([])

    def test_fit_with_valid_data(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer(max_dim=0)
        clouds = [torch.rand(5, 2)]
        result = pt.fit(clouds)
        assert result is pt

    @_torch_backend
    def test_transform_vr(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer(max_dim=0)
        clouds = [torch.rand(5, 2)]
        diagrams = pt.transform(clouds)
        assert len(diagrams) == 1

    def test_transform_witness_invalid_sample_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer(complex_type="witness")
        with pytest.raises(ValueError):
            pt.transform([{"bad_key": torch.rand(5, 2)}])

    def test_transform_unknown_complex_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer()
        pt.complex_type = "bogus"
        with pytest.raises(ValueError):
            pt.transform([torch.rand(3, 2)])

    def test_sklearn_base_class(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistenceTransformer

        pt = PersistenceTransformer()
        params = pt.get_params()
        assert "max_dim" in params


class TestVectorizationTransformer:
    """VectorizationTransformer construction and transform."""

    def test_construct_default(self) -> None:
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer()
        assert vt.method == "landscape"

    def test_invalid_method_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        with pytest.raises((ValueError, ValidationError)):
            VectorizationTransformer(method="bogus")

    def test_transform_empty_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer()
        with pytest.raises((ValueError, ValidationError)):
            vt.transform([])

    def test_transform_with_tensor(self) -> None:
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer(method="silhouette", num_samples=8)
        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        result = vt.transform([diagram, diagram])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 8)

    def test_fit_returns_self(self) -> None:
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer()
        result = vt.fit([torch.rand(3, 2)])
        assert result is vt


class TestStatisticsTransformer:
    """StatisticsTransformer construction and transform."""

    def test_construct_default(self) -> None:
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer()
        assert st.dims == [0, 1]

    def test_transform_empty_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer()
        with pytest.raises((ValueError, ValidationError)):
            st.transform([])

    def test_transform_with_tensor(self) -> None:
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer(dims=[0])
        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
        result = st.transform([diagram, diagram])
        import numpy as np

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)

    def test_get_feature_names_out(self) -> None:
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer(dims=[0])
        names = st.get_feature_names_out()
        import numpy as np

        assert isinstance(names, np.ndarray)
        assert len(names) > 0


class TestPersistencePipeline:
    """PersistencePipeline end-to-end construction."""

    def test_construct_default(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistencePipeline

        pipe = PersistencePipeline()
        assert pipe.complex_type == "vr"

    def test_construct_stats_vectorization(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistencePipeline

        pipe = PersistencePipeline(vectorization="stats")
        assert pipe.vectorization == "stats"

    def test_invalid_complex_type_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistencePipeline

        with pytest.raises((ValueError, ValidationError)):
            PersistencePipeline(complex_type="bogus")

    def test_invalid_vectorization_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import PersistencePipeline

        with pytest.raises((ValueError, ValidationError)):
            PersistencePipeline(vectorization="bogus")


class TestBatchedTransformer:
    """BatchedTransformer construction and validation."""

    def test_construct(self) -> None:
        from pynerve.torch.sklearn_transformers import (
            BatchedTransformer,
            VectorizationTransformer,
        )

        inner = VectorizationTransformer()
        bt = BatchedTransformer(inner, batch_size=16, n_jobs=1)
        assert bt.batch_size == 16

    def test_invalid_batch_size_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import (
            BatchedTransformer,
            VectorizationTransformer,
        )

        inner = VectorizationTransformer()
        with pytest.raises((ValueError, TypeError, ValidationError)):
            BatchedTransformer(inner, batch_size=0)

    def test_fit_empty_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import (
            BatchedTransformer,
            VectorizationTransformer,
        )

        inner = VectorizationTransformer()
        bt = BatchedTransformer(inner)
        with pytest.raises((ValueError, ValidationError)):
            bt.fit([])


class TestMakeTdaPipeline:
    """make_tda_pipeline construction."""

    def test_construct_default(self) -> None:
        from pynerve.torch.sklearn_transformers import make_tda_pipeline
        from sklearn.pipeline import Pipeline

        pipe = make_tda_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_with_classifier(self) -> None:
        from pynerve.torch.sklearn_transformers import make_tda_pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC

        pipe = make_tda_pipeline(classifier=SVC())
        assert isinstance(pipe, Pipeline)
        assert len(pipe.steps) == 3

    def test_invalid_complex_type_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import make_tda_pipeline

        with pytest.raises((ValueError, ValidationError)):
            make_tda_pipeline(complex_type="bogus")

    def test_invalid_vectorization_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import make_tda_pipeline

        with pytest.raises((ValueError, ValidationError)):
            make_tda_pipeline(vectorization="bogus")


class TestPersistenceTrainTestSplit:
    """persistence_train_test_split validation."""

    def test_basic_split(self) -> None:
        from pynerve.torch.sklearn_transformers import persistence_train_test_split

        x = [torch.rand(5, 2) for _ in range(10)]
        y = list(range(10))
        x_train, x_test, y_train, y_test = persistence_train_test_split(
            x, y, test_size=0.3, random_state=42
        )
        assert len(x_train) + len(x_test) == 10
        assert len(y_train) + len(y_test) == 10

    def test_invalid_test_size_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import persistence_train_test_split

        x = [torch.rand(3, 2)]
        y = [0]
        with pytest.raises((ValueError, ValidationError)):
            persistence_train_test_split(x, y, test_size=0.0)
        with pytest.raises((ValueError, ValidationError)):
            persistence_train_test_split(x, y, test_size=1.0)

    def test_mismatched_lengths_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import persistence_train_test_split

        x = [torch.rand(3, 2) for _ in range(5)]
        y = [0, 1]
        with pytest.raises((ValueError, ValidationError)):
            persistence_train_test_split(x, y)

    def test_empty_x_raises(self) -> None:
        from pynerve.torch.sklearn_transformers import persistence_train_test_split

        with pytest.raises((TypeError, ValueError)):
            persistence_train_test_split([], [])
