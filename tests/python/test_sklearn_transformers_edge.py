"""Tests for pynerve/torch/sklearn_transformers.py -- sklearn-compatible TDA transformers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pynerve.torch.sklearn_transformers import (
    PersistenceTransformer,
    StatisticsTransformer,
    VectorizationTransformer,
)
from pynerve.torch._diagram import PersistenceDiagram


def _make_diagram():
    """Create a simple PersistenceDiagram for testing."""
    d = torch.zeros(5, 3, dtype=torch.float32)
    d[0, :] = torch.tensor([0.0, 1.0, 0.0])
    d[1, :] = torch.tensor([0.5, 2.0, 0.0])
    d[2, :] = torch.tensor([1.0, 3.0, 1.0])
    m = torch.zeros(5, dtype=torch.bool)
    m[0] = True
    m[1] = True
    m[2] = True
    n = torch.tensor([2, 1, 0])
    return PersistenceDiagram(d, m, n)


class TestPersistenceTransformer:
    def test_construction_vr(self):
        pt = PersistenceTransformer(complex_type="vr")
        assert pt.complex_type == "vr"
        assert pt.max_dim == 1
        assert pt.max_radius == float("inf")

    def test_construction_witness(self):
        pt = PersistenceTransformer(complex_type="witness", max_dim=2)
        assert pt.complex_type == "witness"
        assert pt.max_dim == 2

    def test_construction_alpha(self):
        pt = PersistenceTransformer(complex_type="alpha")
        assert pt.complex_type == "alpha"

    def test_invalid_complex_type(self):
        with pytest.raises(ValueError, match="complex"):
            PersistenceTransformer(complex_type="invalid")  # type: ignore[arg-type]

    def test_preprocessing_params(self):
        pt = PersistenceTransformer(preprocessing_params={"reduce_infinite": True})
        assert "reduce_infinite" in pt.preprocessing_params

    def test_fit_accepts_sequence(self):
        pt = PersistenceTransformer()
        result = pt.fit([torch.rand(10, 3)])
        assert result is pt

    def test_fit_empty_list_raises(self):
        pt = PersistenceTransformer()
        with pytest.raises((TypeError, ValueError), match="non-empty|sequence"):
            pt.fit([])

    def test_transform_with_preprocessing(self):
        pt = PersistenceTransformer(
            preprocessing_params={"handle_inf": True}
        )
        diagrams = pt.transform([torch.rand(10, 3)])
        assert len(diagrams) == 1
        assert isinstance(diagrams[0], PersistenceDiagram)

    def test_witness_unpack_dict(self):
        pt = PersistenceTransformer(complex_type="witness")
        landmarks = torch.rand(5, 3)
        witnesses = torch.rand(20, 3)
        with pytest.raises(ValueError, match="landmarks"):
            pt._unpack_witness_sample({"bad": "data"})

    def test_witness_unpack_tuple(self):
        pt = PersistenceTransformer(complex_type="witness")
        l, w = torch.rand(5, 3), torch.rand(20, 3)
        result = pt._unpack_witness_sample((l, w))
        assert len(result) == 2

    def test_fit_transform(self):
        pt = PersistenceTransformer()
        diagrams = pt.fit_transform([torch.rand(10, 3)])
        assert len(diagrams) == 1


class TestVectorizationTransformer:
    def test_construction(self):
        vt = VectorizationTransformer(method="landscape")
        assert vt.method == "landscape"

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            VectorizationTransformer(method="invalid")  # type: ignore[arg-type]

    def test_fit_returns_self(self):
        vt = VectorizationTransformer(method="image")
        assert vt.fit([]) is vt

    def test_transform_landscape(self):
        vt = VectorizationTransformer(method="landscape", k=3, num_samples=32)
        diagrams = [_make_diagram().diagrams]
        result = vt.transform(diagrams)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1

    def test_transform_silhouette(self):
        vt = VectorizationTransformer(method="silhouette", num_samples=50)
        diagrams = [_make_diagram().diagrams]
        result = vt.transform(diagrams)
        assert result.shape[0] == 1
        assert result.shape[-1] == 50

    def test_transform_histogram(self):
        vt = VectorizationTransformer(method="histogram", num_bins=32)
        diagrams = [_make_diagram().diagrams]
        result = vt.transform(diagrams)
        assert isinstance(result, np.ndarray)

    def test_transform_heat(self):
        vt = VectorizationTransformer(method="heat", num_samples=32)
        diagrams = [_make_diagram().diagrams]
        result = vt.transform(diagrams)
        assert isinstance(result, np.ndarray)

    def test_transform_empty_raises(self):
        vt = VectorizationTransformer(method="landscape")
        with pytest.raises(ValueError, match="non-empty"):
            vt.transform([])


class TestStatisticsTransformer:
    def test_construction(self):
        st = StatisticsTransformer()
        assert st.dims == [0, 1]
        assert len(st.track_stats) == 4

    def test_construction_custom(self):
        st = StatisticsTransformer(dims=[0, 2], features=["total", "max"])
        assert st.dims == [0, 2]

    def test_fit_returns_self(self):
        st = StatisticsTransformer()
        assert st.fit([]) is st

    def test_transform_basic(self):
        st = StatisticsTransformer(track_stats=["num_features", "total_persistence"])
        diagrams = [_make_diagram()]
        result = st.transform(diagrams)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2

    def test_fit_transform(self):
        st = StatisticsTransformer()
        diagrams = [_make_diagram()]
        result = st.fit_transform(diagrams)
        assert isinstance(result, np.ndarray)

    def test_get_feature_names(self):
        st = StatisticsTransformer(track_stats=["num_features"])
        names = st.get_feature_names_out()
        assert len(names) > 0
