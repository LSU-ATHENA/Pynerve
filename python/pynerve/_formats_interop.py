"""Converters for third-party TDA library formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._validation import parse_nonnegative_int, validate_diagram_array
from .exceptions import InvalidArgumentError, ValidationError

Diagram = list[tuple[float, float, int]]


def _validate_diagram_entry(birth: float, death: float, dim: int) -> tuple[float, float, int]:
    birth = float(birth)
    death = float(death)
    dim = parse_nonnegative_int(dim, "dimension")
    if not np.isfinite(birth):
        raise ValidationError("diagram births must be finite")
    if np.isnan(death) or np.isneginf(death):
        raise ValidationError("diagram deaths must be finite or +inf")
    if np.isfinite(death) and death < birth:
        raise ValidationError("diagram finite deaths must be greater than or equal to births")
    return birth, death, dim


def _validate_diagram(diagram: Diagram) -> Diagram:
    return [_validate_diagram_entry(birth, death, dim) for birth, death, dim in diagram]


def _as_diagram_array(diagram: Any) -> np.ndarray:
    array = np.asarray(diagram, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    return validate_diagram_array(array, require_dims=True)


def _from_gudhi_diagram(gudhi_object: Any) -> Diagram:
    result: Diagram = []
    for item in gudhi_object:
        if len(item) == 2:
            dim, (birth, death) = item
            result.append(_validate_diagram_entry(birth, death, dim))
        elif len(item) == 3:
            birth, death, dim = item
            result.append(_validate_diagram_entry(birth, death, dim))
        else:
            raise ValueError("GUDHI diagram entries must have length 2 or 3")
    return result


def _resolve_simplex_iter(gudhi_object: Any) -> Any:
    if hasattr(gudhi_object, "get_filtration"):
        return gudhi_object.get_filtration()
    if hasattr(gudhi_object, "get_simplices"):
        return gudhi_object.get_simplices()
    if isinstance(gudhi_object, dict) and "simplices" in gudhi_object:
        return gudhi_object["simplices"]
    if isinstance(gudhi_object, (list, tuple)):
        return gudhi_object
    raise ValidationError(
        "Unsupported simplex_tree source: expected GUDHI simplex tree "
        "or iterable of (simplex, filtration) pairs."
    )


def _from_gudhi_simplex_tree(gudhi_object: Any) -> list[tuple[tuple[int, ...], float]]:
    result: list[tuple[tuple[int, ...], float]] = []
    for item in _resolve_simplex_iter(gudhi_object):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValidationError("simplex entries must be (simplex, filtration) pairs")
        simplex, filtration = item
        vertices = tuple(parse_nonnegative_int(v, "simplex vertex") for v in simplex)
        filtration = float(filtration)
        if not np.isfinite(filtration):
            raise ValidationError("simplex filtration values must be finite")
        result.append((vertices, filtration))
    return result


def from_gudhi(
    gudhi_object: Any, format_type: str = "diagram"
) -> Diagram | list[tuple[tuple[int, ...], float]]:
    """Convert a GUDHI object to a Nerve-native diagram or simplex list.

    :param gudhi_object: GUDHI simplex tree or diagram object, or an iterable of
        ``(simplex, filtration)`` pairs.
    :param format_type: ``"diagram"`` (default) extracts persistence pairs;
        ``"simplex_tree"`` extracts ``(vertices, filtration)`` pairs.
    :returns: If ``format_type="diagram"``, a list of ``(birth, death, dimension)`` tuples.
        If ``"simplex_tree"``, a list of ``((vertex, ...), filtration)`` pairs.
    :raises InvalidArgumentError: If ``format_type`` is unknown.
    :raises ValidationError: If simplex entries are malformed.
    """
    if format_type == "diagram":
        return _from_gudhi_diagram(gudhi_object)
    if format_type == "simplex_tree":
        return _from_gudhi_simplex_tree(gudhi_object)
    raise InvalidArgumentError(f"Unknown format_type: {format_type}")


def from_external(external_output: dict[str, Any]) -> list[tuple[float, float, int]]:
    """Convert External output to a Nerve-native diagram list.

    :param external_output: Dictionary produced by external library containing a
        ``"dgms"`` key (list of diagrams per dimension).
    :returns: List of ``(birth, death, dimension)`` tuples.
    :raises ValidationError: If ``external_output`` lacks a ``"dgms"`` key.
    """
    if "dgms" not in external_output:
        raise ValidationError("external output must contain a 'dgms' entry")
    result: Diagram = []
    for dim, diagram in enumerate(external_output["dgms"]):
        for birth, death in diagram:
            result.append(_validate_diagram_entry(birth, death, dim))

    return result


def from_dionysus(dionysus_diagrams: list[Any]) -> Diagram:
    """Convert Dionysus diagram objects to a Nerve-native diagram list.

    :param dionysus_diagrams: List of Dionysus diagrams, each containing points with
        ``.birth`` / ``.death`` attributes or indexable as 2-tuples.
    :returns: List of ``(birth, death, dimension)`` tuples.
    :raises ValidationError: If a point does not expose ``birth``/``death`` values.
    """
    result: Diagram = []

    for dim, dgm in enumerate(dionysus_diagrams):
        for point in dgm:
            if hasattr(point, "birth") and hasattr(point, "death"):
                result.append(_validate_diagram_entry(point.birth, point.death, dim))
            elif len(point) >= 2:
                result.append(_validate_diagram_entry(point[0], point[1], dim))
            else:
                raise ValidationError("Dionysus points must expose birth/death values")

    return result


def from_giotto(giotto_diagrams: np.ndarray) -> Diagram:
    """Convert a Giotto-tda diagram array to a Nerve-native diagram list.

    :param giotto_diagrams: ``(N, 3)`` array with ``(birth, death, dimension)`` columns.
    :returns: List of ``(birth, death, dimension)`` tuples.
    """
    result: Diagram = []
    for row in _as_diagram_array(giotto_diagrams):
        result.append(_validate_diagram_entry(row[0], row[1], int(row[2])))
    return result


def from_sktda(sktda_diagram: Any) -> Diagram:
    """Convert a scikit-tda diagram object to a Nerve-native diagram list.

    :param sktda_diagram: scikit-tda diagram object exposing a ``dgms`` attribute
        (list of diagrams per dimension).
    :returns: List of ``(birth, death, dimension)`` tuples.
    :raises ValidationError: If the object does not have a ``dgms`` attribute.
    """
    if not hasattr(sktda_diagram, "dgms"):
        raise ValidationError("scikit-tda diagram object must expose a 'dgms' attribute")
    result: Diagram = []
    for dim, diagram in enumerate(sktda_diagram.dgms):
        for birth, death in diagram:
            result.append(_validate_diagram_entry(birth, death, dim))

    return result


def to_gudhi(
    diagram: Diagram,
) -> list[tuple[int, tuple[float, float]]]:
    """Convert a Nerve-native diagram to GUDHI format.

    :param diagram: List of ``(birth, death, dimension)`` tuples.
    :returns: List of ``(dimension, (birth, death))`` pairs compatible with GUDHI.
    :raises ValidationError: If any diagram entry is invalid.
    """
    diagram = _validate_diagram(diagram)
    return [(int(dim), (float(birth), float(death))) for birth, death, dim in diagram]


def to_external(diagram: Diagram, filepath: str | Path | None = None) -> str:
    """Convert a Nerve-native diagram to External-compatible JSON.

    :param diagram: List of ``(birth, death, dimension)`` tuples.
    :param filepath: Optional output path. If provided, writes JSON to disk and returns the path.
    :returns: JSON string representation of the diagram, or ``str(filepath)`` if written to disk.
    :raises ValidationError: If any diagram entry is invalid.
    """
    diagram = _validate_diagram(diagram)
    data = {
        "format": "nerve_diagram",
        "diagrams": [
            {
                "birth": float(birth),
                "death": float(death) if not np.isinf(death) else None,
                "dimension": int(dim),
            }
            for birth, death, dim in diagram
        ],
    }

    if filepath:
        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(data, f, indent=2)
        return str(filepath)

    return json.dumps(data)


def to_dionysus(
    diagram: Diagram,
) -> dict[int, list[tuple[float, float]]]:
    """Convert a Nerve-native diagram to Dionysus-compatible dict.

    :param diagram: List of ``(birth, death, dimension)`` tuples.
    :returns: Dictionary mapping each dimension ``int`` to a list of ``(birth, death)`` pairs.
    :raises ValidationError: If any diagram entry is invalid.
    """
    diagram = _validate_diagram(diagram)
    result: dict[int, list[tuple[float, float]]] = {}
    for birth, death, dim in diagram:
        result.setdefault(int(dim), []).append((float(birth), float(death)))
    return result


__all__ = [
    "from_gudhi",
    "from_external",
    "from_dionysus",
    "from_giotto",
    "from_sktda",
    "to_gudhi",
    "to_external",
    "to_dionysus",
]
