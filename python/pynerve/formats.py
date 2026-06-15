"""Format converters and file serializers for Pynerve.

Provides load/save functions for common diagram and point-cloud file formats
(CSV, JSON, OFF, PLY, NPY) and converters to/from third-party TDA libraries
(GUDHI, Dionysus, Giotto, scikit-tda).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._formats_auto import auto_load, auto_save
from ._formats_files import (
    load_csv,
    load_json,
    load_off,
    load_ply,
    save_csv,
    save_json,
    save_off,
    save_ply,
)
from ._formats_interop import (
    from_dionysus,
    from_external,
    from_giotto,
    from_gudhi,
    from_sktda,
    to_dionysus,
    to_external,
    to_gudhi,
)
from .exceptions import ShapeError, ValidationError

Diagram = list[tuple[float, float, int]]
DiagramLike = np.ndarray | list[tuple[float, float, int]]


def load_diagrams(filepath: str | Path) -> DiagramLike:
    """Load a persistence diagram from a file, auto-detecting format by extension.

    :param filepath: Path to the input file.
    :returns: List of ``(birth, death, dimension)`` tuples.
    :raises ValidationError: If the file extension is not recognised.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".json":
        data = load_json(path)
        raw = data.get("diagram") or data.get("diagrams", [])
        return [(float(b), float(d), int(dim)) for b, d, dim in raw]
    if suffix in (".off", ".ply"):
        points = load_off(path) if suffix == ".off" else load_ply(path)
        return [(float(p[0]), float(p[0]), 0) for p in points]
    raise ValidationError(f"Unrecognized file extension: {suffix}. Use .csv, .json, .off, or .ply")


def save_diagrams(diagram: DiagramLike, filepath: str | Path, **kwargs: Any) -> None:
    """Save a persistence diagram to a file, auto-detecting format by extension.

    :param diagram: Diagram data as a list of ``(birth, death, dimension)`` tuples or a 2-D ``ndarray``.
    :param filepath: Output file path.
    Extra keyword arguments are forwarded to the underlying writer
    (e.g. ``header`` for CSV, ``metadata`` for JSON).
    :returns: None.
    :raises ShapeError: If the diagram array does not have at least 2 columns.
    :raises ValidationError: If the file extension is not recognised.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    if isinstance(diagram, np.ndarray):
        if diagram.ndim != 2:
            raise ShapeError(
                f"diagram must be a 2D array, got shape {diagram.shape}",
                expected_ndim=2,
                actual_ndim=diagram.ndim,
            )
        ncols = diagram.shape[1]
        if ncols == 2:
            diagram = [(float(b), float(d), 0) for b, d in diagram[:, :2]]
        elif ncols >= 3:
            diagram = [(float(b), float(d), int(dim)) for b, d, dim in diagram[:, :3]]
        else:
            raise ShapeError(
                f"diagram must have at least 2 columns, got shape {diagram.shape}",
                parameter="diagram",
            )
    if suffix == ".csv":
        save_csv(diagram, path, **kwargs)
    elif suffix == ".json":
        save_json(diagram, path, **kwargs)
    elif suffix in (".off", ".ply"):
        points = np.array(
            [[b, d if np.isfinite(d) else 0, dim] for b, d, dim in diagram] or [[0, 0, 0]],
            dtype=float,
        )
        if suffix == ".off":
            save_off(points, path)
        else:
            save_ply(points, path)
    else:
        raise ValidationError(
            f"Unrecognized file extension: {suffix}. Use .csv, .json, .off, or .ply"
        )


__all__ = [
    "Diagram",
    "DiagramLike",
    "from_gudhi",
    "from_external",
    "from_dionysus",
    "from_giotto",
    "from_sktda",
    "to_gudhi",
    "to_external",
    "to_dionysus",
    "load_csv",
    "save_csv",
    "load_json",
    "save_json",
    "load_off",
    "save_off",
    "load_ply",
    "save_ply",
    "auto_load",
    "auto_save",
    "load_diagrams",
    "save_diagrams",
]
