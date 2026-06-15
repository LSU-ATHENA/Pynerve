from __future__ import annotations

import numpy as np
import pytest
from pynerve.exceptions import ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_torch_sklearn_transformers_reject_invalid_numeric_inputs(torch) -> None:
    from pynerve.torch.sklearn_transformers import (
        BatchedTransformer,
        PersistencePipeline,
        PersistenceTransformer,
        StatisticsTransformer,
        VectorizationTransformer,
        persistence_train_test_split,
    )

    with pytest.raises(ValueError, match="max_dim"):
        PersistenceTransformer(max_dim=float("nan"))
    with pytest.raises(ValueError, match="X"):
        PersistenceTransformer().fit([])
    with pytest.raises(ValueError, match="tensor inputs"):
        PersistenceTransformer().transform([torch.tensor([[float("nan"), 0.0]])])
    with pytest.raises(ValueError, match="point cloud"):
        PersistenceTransformer().transform([torch.empty(0, 2)])

    with pytest.raises(ValueError, match="births"):
        VectorizationTransformer().transform([torch.tensor([[float("nan"), 1.0]])])
    with pytest.raises(ValueError, match="deaths"):
        StatisticsTransformer().transform([torch.tensor([[1.0, 0.0]])])
    with pytest.raises(ValueError, match="vectorization"):
        PersistencePipeline(vectorization="not_a_method")

    class _DummyTransformer:
        def fit(self, x, y=None):  # noqa: ARG002
            self.fit_seen = len(x)
            return self

        def transform(self, x):
            return np.ones((len(x), 1), dtype=np.float32)

    batched = BatchedTransformer(_DummyTransformer(), batch_size=2)
    assert batched.fit_transform([1, 2, 3]).shape == (3, 1), (
        f"expected (3, 1), got {batched.fit_transform([1, 2, 3]).shape}"
    )
    with pytest.raises(ValidationError, match="batch_size"):
        BatchedTransformer(_DummyTransformer(), batch_size=float("nan"))
    with pytest.raises(ValueError, match="X"):
        batched.transform([])

    sklearn = pytest.importorskip("sklearn")
    assert sklearn is not None, "sklearn import returned None"
    with pytest.raises(ValidationError, match="test_size"):
        persistence_train_test_split([1, 2], [0, 1], test_size=float("nan"))
    with pytest.raises(ValueError, match="matching lengths"):
        persistence_train_test_split([1, 2], [0], test_size=0.5)
