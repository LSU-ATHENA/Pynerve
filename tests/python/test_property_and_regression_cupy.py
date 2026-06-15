from __future__ import annotations

import numpy as np
import pytest

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_cupy_core_rejects_nonfinite_max_dist_before_backend_import() -> None:
    from pynerve._cupy_persistence import _compute_core_persistence

    points = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    with pytest.raises(Exception, match="max_radius"):
        _compute_core_persistence(points, float("nan"), 1)
    with pytest.raises(Exception, match="max_radius"):
        _compute_core_persistence(points, float("inf"), 1)
    with pytest.raises(ValueError, match="finite coordinates"):
        _compute_core_persistence(np.asarray([[0.0, np.nan]], dtype=np.float64), 1.0, 1)
    with pytest.raises(ValueError, match="2D"):
        _compute_core_persistence(np.asarray([0.0, 1.0], dtype=np.float64), 1.0, 1)
    with pytest.raises(TypeError, match="real numeric"):
        _compute_core_persistence(np.asarray([[1.0 + 1.0j]], dtype=np.complex128), 1.0, 1)


def test_cupy_persistence_image_rejects_invalid_diagrams_without_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pynerve._cupy_persistence as cupy_persistence
    from pynerve._cupy_persistence import CuPyPersistence

    monkeypatch.setattr(cupy_persistence, "HAS_CUPY", True)
    monkeypatch.setattr(cupy_persistence, "cp", np)
    computer = object.__new__(CuPyPersistence)
    diagram = np.asarray([[0.0, 1.0], [0.25, 0.75]], dtype=np.float64)

    image = computer.persistence_image_cupy(diagram, resolution=4, sigma=0.2)
    assert image.shape == (4, 4), f"expected (4, 4), got {image.shape}"
    with pytest.raises(ValueError, match="diagram"):
        computer.persistence_image_cupy(np.asarray([[0.0, np.inf]], dtype=np.float64))
    with pytest.raises(ValueError, match="deaths"):
        computer.persistence_image_cupy(np.asarray([[1.0, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        computer.persistence_image_cupy(np.asarray([0.0, 1.0], dtype=np.float64))
    with pytest.raises(TypeError, match="numeric"):
        computer.persistence_image_cupy(np.asarray([["0.0", "1.0"]], dtype=object))
    with pytest.raises(TypeError, match="array"):
        computer.persistence_image_cupy([[0.0, 1.0]])
