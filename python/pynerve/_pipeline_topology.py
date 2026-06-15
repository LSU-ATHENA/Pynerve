"""Topology-specific pipeline builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._compute_api import compute_persistence
from ._pipeline_core import Pipeline, _validate_representations
from ._validation import validate_nonnegative_finite, validate_nonnegative_int
from .exceptions import InvalidArgumentError


def _diagram_pairs(diagram: Any) -> Any:
    if isinstance(diagram, dict):
        if "pairs" not in diagram:
            raise InvalidArgumentError(
                "persistence result must include a 'pairs' entry", parameter="diagram"
            )
        return diagram["pairs"]
    return diagram


def _diagram_pair_array(diagram: Any) -> np.ndarray:
    from ._validation import validate_diagram_array  # noqa: PLC0415

    pairs = np.asarray(_diagram_pairs(diagram), dtype=float)
    if pairs.size == 0:
        return np.empty((0, 3), dtype=float)
    return validate_diagram_array(pairs, name="diagram")


def _filter_persistence_pairs(diagram: Any, min_persistence: float) -> Any:
    pairs = list(_diagram_pairs(diagram))
    pair_array = _diagram_pair_array(diagram)
    filtered = [
        pairs[i]
        for i, (birth, death) in enumerate(pair_array[:, :2])
        if float(death - birth) > min_persistence
    ]
    if isinstance(diagram, dict):
        result = dict(diagram)
        result["pairs"] = filtered
        return result
    return filtered


def _persistence_vector(diagram: Any) -> list[float]:
    pairs = _diagram_pair_array(diagram)
    return [float(death - birth) for birth, death in pairs[:, :2]]


def vr_pipeline(
    max_dim: int = 2, max_radius: float | None = None, min_persistence: float = 0.0
) -> Pipeline:
    """Build a Vietoris-Rips persistence pipeline.

    :param max_dim: Maximum homology dimension to compute (default 2).
    :param max_radius: Maximum radius for the Vietoris-Rips complex.
        ``None`` means no limit.
    :param min_persistence: Minimum persistence ``(death - birth)`` required
        to retain a pair (default 0.0).
    :returns: A configured :class:`Pipeline`.
    :raises ValidationError: If any parameter is out of range.
    """
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    min_persistence = validate_nonnegative_finite(min_persistence, "min_persistence")
    if max_radius is not None:
        max_radius = validate_nonnegative_finite(max_radius, "max_radius")
    steps = [
        (
            "compute_vr",
            lambda data: compute_persistence(data, max_dim=max_dim, max_radius=max_radius),
        ),
    ]

    if min_persistence > 0:
        steps.append(
            (
                "filter",
                lambda dgm: _filter_persistence_pairs(dgm, min_persistence),  # pyright: ignore[reportArgumentType]
            )
        )

    return Pipeline(*steps)


def _representation_fn(rep: str) -> Callable[[Any], Any]:
    if not isinstance(rep, str) or not rep:
        raise InvalidArgumentError(
            "representation names must be non-empty strings", parameter="rep"
        )
    if rep == "image":
        from .fast_ops import _persistence_image_fast  # noqa: PLC0415

        return lambda diagram: _persistence_image_fast(_diagram_pair_array(diagram))
    if rep == "landscape":
        from .fast_ops import _persistence_landscape_fast  # noqa: PLC0415

        return lambda diagram: _persistence_landscape_fast(_diagram_pair_array(diagram))
    if rep == "vector":
        return _persistence_vector
    raise InvalidArgumentError(f"unknown representation: {rep}", parameter="rep")


def analysis_pipeline(
    compute_fn: Callable[..., Any], representations: list[str] | None = None
) -> Pipeline:
    """Build a persistence analysis pipeline with configurable output representations.

    :param compute_fn: Callable that receives data and returns a persistence
        diagram (as a dict with a ``"pairs"`` key or as a raw array).
    :param representations: List of output representations to include.
        Supported values: ``"diagram"``, ``"image"``, ``"landscape"``,
        ``"vector"``. Defaults to ``["diagram"]``.
    :returns: A configured :class:`Pipeline`.
    :raises TypeError: If *compute_fn* is not callable.
    :raises InvalidArgumentError: If any representation is unknown or
        invalid.
    """
    if not callable(compute_fn):
        raise TypeError("compute_fn must be callable")
    if representations is None:
        representations = ["diagram"]
    representations = _validate_representations(representations)
    supported = {"diagram", "image", "landscape", "vector"}
    unknown = set(representations) - supported
    if unknown:
        raise InvalidArgumentError(
            f"unknown representations: {sorted(unknown)}", parameter="representations"
        )

    steps = [("compute", compute_fn)]
    transforms = [rep for rep in representations if rep != "diagram"]

    if not transforms:
        return Pipeline(*steps)
    if len(representations) == 1:
        rep = transforms[0]
        steps.append((f"to_{rep}", _representation_fn(rep)))
        return Pipeline(*steps)

    rep_fns = {rep: _representation_fn(rep) for rep in transforms}

    def represent(diagram: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for rep in representations:
            if rep == "diagram":
                result[rep] = diagram
            else:
                result[rep] = rep_fns[rep](diagram)
        return result

    steps.append(("represent", represent))
    return Pipeline(*steps)
