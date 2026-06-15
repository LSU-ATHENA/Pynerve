from __future__ import annotations

import pytest
from pynerve.exceptions import ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_synthetic_datasets_reject_invalid_numeric_inputs() -> None:
    from pynerve import datasets

    assert datasets.load_sphere(n_samples=4, dim=2, seed=1).shape == (4, 3), (
        f"expected (4, 3), got {datasets.load_sphere(n_samples=4, dim=2, seed=1).shape}"
    )

    with pytest.raises(ValidationError, match="n_samples"):
        datasets.load_sphere(n_samples=1.5)
    with pytest.raises(ValidationError, match="radius"):
        datasets.load_sphere(radius=float("inf"))
    with pytest.raises(ValidationError, match="dim"):
        datasets.load_sphere(dim=1.2)
