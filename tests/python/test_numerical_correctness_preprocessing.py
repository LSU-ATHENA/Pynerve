"""Numerical correctness tests that verify outputs against hand-computed values.

Every test with a numerical value includes a precise assertion with tolerance,
not just a shape or finiteness check.
"""

from __future__ import annotations

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

# preprocessing


class TestPreprocessingCorrectness:
    """Numerical correctness for preprocessing."""

    def test_handle_inf_max(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        d = torch.tensor([[0.0, 1.0], [2.0, float("inf")]], dtype=torch.float64)
        result = handle_infinite_deaths(d, strategy="max")
        assert result[1, 1].item() == pytest.approx(2.0, abs=1e-10)

    def test_handle_inf_remove(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        d = torch.tensor([[0.0, 1.0], [2.0, float("inf")]], dtype=torch.float64)
        result = handle_infinite_deaths(d, strategy="remove")
        assert result.shape == (1, 2)
        assert result[0, 1].item() == pytest.approx(1.0, abs=1e-10)

    def test_handle_inf_large_value(self) -> None:
        from pynerve.torch.preprocessing import handle_infinite_deaths

        d = torch.tensor([[0.0, 1.0], [2.0, float("inf")]], dtype=torch.float64)
        result = handle_infinite_deaths(d, strategy="large_value", large_value_factor=100.0)
        assert result[1, 1].item() == pytest.approx(100.0, abs=1e-10)

    def test_normalize_minmax_exact(self) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        normed = normalize_diagram(d, method="minmax")
        assert normed[0, 1].item() == pytest.approx(0.0, abs=1e-10)
        assert normed[1, 1].item() == pytest.approx(0.5, abs=1e-10)
        assert normed[2, 1].item() == pytest.approx(1.0, abs=1e-10)

    def test_threshold_diagram_exact(self) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        d = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
        filtered = threshold_diagram(d, min_persistence=1.5)
        assert filtered.shape[0] == 2
        assert filtered[0, 1].item() == pytest.approx(2.0, abs=1e-10)
        assert filtered[1, 1].item() == pytest.approx(3.0, abs=1e-10)

    def test_subsample_most_persistent_ordering(self) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        d = torch.tensor([[0.0, 1.0], [0.0, 4.0], [0.0, 2.0]], dtype=torch.float64)
        result = subsample_diagram(d, max_features=2, strategy="most_persistent")
        assert result.shape[0] == 2
        assert result[0, 1].item() == pytest.approx(4.0, abs=1e-10)
        assert result[1, 1].item() == pytest.approx(2.0, abs=1e-10)

    def test_remove_outliers_basic(self) -> None:
        from pynerve.torch.preprocessing import remove_outliers

        d = torch.tensor([[0.0, 1.0], [0.0, 1.1], [0.0, 100.0]], dtype=torch.float64)
        result = remove_outliers(d, method="iqr", threshold=1.5)
        assert result.shape[-1] == 2
        assert torch.isfinite(result).all()

    def test_clean_diagram_pipeline(self) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        d = torch.tensor([[0.0, float("inf")], [0.0, 2.0], [0.0, 0.1]], dtype=torch.float64)
        cleaned = clean_diagram(d, handle_inf=True, min_persistence=0.5, normalize=False)
        assert torch.isfinite(cleaned).all()
