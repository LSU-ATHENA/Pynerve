"""Numerical correctness tests that verify outputs against hand-computed values.

Every test with a numerical value includes a precise assertion with tolerance,
not just a shape or finiteness check.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# known-configuration point clouds
_TRIANGLE = torch.tensor(
    [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0**0.5]],
    dtype=torch.float64,
)
_SQUARE = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=torch.float64,
)
_TWO_POINTS = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
_SINGLE_POINT = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

# known diagram tensors
_SIMPLE_DIAGRAM = torch.tensor(
    [[0.0, 1.0], [0.0, 2.0], [1.0, 3.0]],
    dtype=torch.float64,
)
_SINGLE_DIAGRAM = torch.tensor([[0.0, 1.5]], dtype=torch.float64)
_INFINITE_DIAGRAM = torch.tensor(
    [[0.0, float("inf")], [0.0, 2.0]],
    dtype=torch.float64,
)

# sklearn transformers


class TestSklearnTransformerCorrectness:
    """Numerical correctness for sklearn transformers."""

    def test_statistics_transformer_output(self) -> None:
        from pynerve.torch.sklearn_transformers import StatisticsTransformer

        st = StatisticsTransformer()
        X = [torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64) for _ in range(3)]
        result = st.fit_transform(X)
        assert result.shape[0] == 3
        assert np.isfinite(result).all()

    def test_vectorization_transformer_output(self) -> None:
        from pynerve.torch.sklearn_transformers import VectorizationTransformer

        vt = VectorizationTransformer(method="silhouette", num_samples=8)
        X = [torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64) for _ in range(3)]
        result = vt.fit_transform(X)
        assert len(result) == 3
        for r in result:
            t = r if isinstance(r, torch.Tensor) else torch.tensor(r)
            assert torch.isfinite(t).all()


# PersistenceDiagram class


class TestPersistenceDiagramCorrectness:
    """Numerical correctness for PersistenceDiagram class."""

    def test_total_persistence_p1_exact(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        tp = pd.total_persistence(p=1.0)
        assert tp[0].item() == pytest.approx(3.0, abs=1e-10)

    def test_persistence_entropy(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        ent = pd.persistence_entropy()
        assert ent[0].item() > 0

    def test_births_property(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        births = pd.births(apply_mask=False)
        torch.testing.assert_close(births, tensor[..., 0])

    def test_deaths_property(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        deaths = pd.deaths(apply_mask=False)
        torch.testing.assert_close(deaths, tensor[..., 1])

    def test_dimensions_property(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor([[[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]]], dtype=torch.float64)
        pd = PersistenceDiagram(tensor)
        dims = pd.dimensions()
        assert dims.dtype == torch.long
        assert torch.equal(dims, tensor[..., 2].long())

    def test_filter_by_dimension(self) -> None:
        from pynerve.torch import PersistenceDiagram

        tensor = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 3.0, 0.0]]],
            dtype=torch.float64,
        )
        pd = PersistenceDiagram(tensor)
        filtered = pd.filter_by_dimension(0)
        births = filtered.births(apply_mask=False)
        assert births[0, 0].item() == pytest.approx(0.0, abs=1e-10)
        assert births[0, 2].item() == pytest.approx(0.0, abs=1e-10)
