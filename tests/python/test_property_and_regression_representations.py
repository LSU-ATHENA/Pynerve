from __future__ import annotations

import numpy as np
import pytest

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_numpy_persistence_image_rejects_nonfinite_scalars_and_invalid_diagrams() -> None:
    from pynerve import persistence_image

    diagram = np.asarray([[0.0, 1.0], [0.25, 0.75]], dtype=np.float64)
    image = persistence_image(diagram, resolution=(4, 6), sigma=0.2)

    assert image.shape == (4, 6), f"expected (4, 6), got {image.shape}"
    assert np.isfinite(image).all(), "expected all finite values in persistence image"
    with pytest.raises(Exception, match="sigma"):
        persistence_image(diagram, sigma=float("nan"))
    with pytest.raises(Exception, match="birth"):
        persistence_image(np.asarray([[float("inf"), float("inf")]], dtype=np.float64))
    with pytest.raises(Exception, match="death"):
        persistence_image(np.asarray([[1.0, 0.0]], dtype=np.float64))
    with pytest.raises(Exception, match="death"):
        persistence_image(np.asarray([[0.0, float("nan")]], dtype=np.float64))


def test_fast_representations_reject_invalid_numeric_inputs() -> None:
    from pynerve._fast_representations import (
        betti_curve_fast as betti_curve,
    )
    from pynerve._fast_representations import (
        persistence_image_fast as persistence_image,
    )
    from pynerve._fast_representations import (
        persistence_landscape_fast as persistence_landscape,
    )

    pairs = np.asarray([[0.0, 1.0, 0.0], [0.25, 0.75, 1.0]], dtype=np.float64)

    assert persistence_image(pairs, resolution=4, sigma=0.2).shape == (4, 4), (
        f"expected (4, 4), got {persistence_image(pairs, resolution=4, sigma=0.2).shape}"
    )
    assert persistence_landscape(pairs, n_layers=2, resolution=5).shape == (2, 5), (
        f"expected (2, 5), got {persistence_landscape(pairs, n_layers=2, resolution=5).shape}"
    )
    assert betti_curve(pairs, max_dim=1, resolution=5).shape == (2, 5), (
        f"expected (2, 5), got {betti_curve(pairs, max_dim=1, resolution=5).shape}"
    )
    with pytest.raises(ValueError, match="sigma"):
        persistence_image(pairs, sigma=float("nan"))
    with pytest.raises(ValueError, match="birth"):
        persistence_image(np.asarray([[float("nan"), 1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="death"):
        persistence_landscape(np.asarray([[1.0, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="max_time"):
        betti_curve(pairs, max_time=float("inf"))
