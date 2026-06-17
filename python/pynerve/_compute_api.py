"""Public persistence computation API."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ._compute_core import (
    PersistenceEngine,
    PersistenceResult,
    _apply_option_overrides,
    _auto_select_engine,
    _clone_options,
    _compute_with_options,
    _estimate_n_points,
    _nerve_state,
    _require_core,
    _resolve_engine_func,
    _to_events_list,
    _warn_device_overrides_backend,
)
from ._fallback_classes import EventType, PersistenceBackend, PersistenceMode, PersistenceOptions
from ._types import PointCloud

__all__ = [
    "PersistenceEngine",
    "compute_persistence",
    "compute_persistence_ph0",
    "compute_persistence_ph3",
    "compute_persistence_ph4",
    "compute_persistence_ph5",
    "compute_persistence_ph6",
    "update_persistence",
]


_ENGINE_DOC = {
    PersistenceEngine.PH0: (
        "PH0 standard homology engine",
        "Standard (homology) matrix reduction. Zomorodian-Carlsson 2005 algorithm.\n    Best for tiny point clouds (<1K points) where cohomology overhead dominates.\n    Use for reproducing older results or when exact historical compatibility is needed.",
    ),
    PersistenceEngine.PH3: (
        "PH3 cohomology engine",
        "Cohomology-based persistence. The default algorithm for small-to-medium datasets.\n    Uses coboundary matrix and reverse-order processing.\n    Typically 30-70% fewer column operations than homology.\n    Recommended for general persistent homology workloads.",
    ),
    PersistenceEngine.PH4: (
        "PH4 cohomology + aggressive clearing engine",
        "Cohomology with dimension-cascading clearing and improved memory handling.\n    Recursively clears columns up the chain after each pairing.\n    Up to 50% additional column elimination over basic cohomology.\n    Recommended for higher-dimensional homology, up to dimension 4.",
    ),
    PersistenceEngine.PH5: (
        "PH5 unified adaptive engine",
        "Adaptive engine combining cohomology, approximate mode, and iterative refinement.\n    Automatically selects algorithms based on data characteristics.\n    Use for large datasets (10K-1M points).",
    ),
    PersistenceEngine.PH6: (
        "PH6 block-sparse speculative engine",
        "High-performance engine with cache-blocked reduction and speculative execution.\n    Uses L2-blocked matrix layout and adaptive pivoting.\n    Recommended for million-scale datasets and sparse filtrations.",
    ),
}


def _make_engine_func(engine: PersistenceEngine) -> Callable[..., PersistenceResult]:
    def func(
        points: PointCloud,
        options: PersistenceOptions | None = None,
        *,
        max_dim: int = 2,
        max_radius: float | None = None,
        mode: PersistenceMode = PersistenceMode.EXACT,
        backend: PersistenceBackend | None = None,
        threads: int | None = None,
        device: str | None = None,
        seed: int | None = None,
        error_tolerance: float | None = None,
        dtype: str | None = None,
        max_radius_cap: float | None = None,
    ) -> PersistenceResult:
        return compute_persistence(
            points,
            options,
            engine=engine,
            max_dim=max_dim,
            max_radius=max_radius,
            mode=mode,
            backend=backend,
            threads=threads,
            device=device,
            seed=seed,
            error_tolerance=error_tolerance,
            dtype=dtype,
            max_radius_cap=max_radius_cap,
        )

    func.__name__ = f"compute_persistence_{engine.name.lower()}"
    func.__qualname__ = func.__name__
    doc_info = _ENGINE_DOC.get(engine, (f"{engine.name} engine", ""))
    desc, details = doc_info
    func.__doc__ = f"""Compute persistence using the {desc}.

{details}

    Args:
        points: Point cloud as an (N, D) array (NumPy or PyTorch tensor).
        options: PersistenceOptions instance with preset configuration.
            When both ``options`` and individual keyword arguments are provided,
            the keyword arguments take precedence.
        max_dim: Maximum homology dimension to compute (default: 2).
        max_radius: Filtration radius cutoff (default: auto-selected).
        mode: PersistenceMode override (default: EXACT).
        backend: PersistenceBackend override (default: auto-selected).
        threads: Number of threads (0 = auto).
        device: Target device string ("cpu", "cuda", "cuda:N").
        seed: RNG seed for deterministic computation.
        error_tolerance: Approximation error tolerance (>0 enables APPROX mode).
        dtype: NumPy dtype name for point array conversion (e.g. ``"float32"``).
            By default inputs are converted to ``float64``.
        max_radius_cap: Override the upper bound for ``max_radius=inf``.
            By default uses the ``NERVE_MAX_RADIUS_CAP`` env var (1e15).

    Returns:
        A :class:`PersistenceResult` with fields ``pairs``, ``betti_numbers``,
        ``max_dim``, ``max_radius``, and ``diagnostics``.

    See :func:`compute_persistence` for full parameter documentation and examples.
    """
    return func


compute_persistence_ph0 = _make_engine_func(PersistenceEngine.PH0)
compute_persistence_ph3 = _make_engine_func(PersistenceEngine.PH3)
compute_persistence_ph4 = _make_engine_func(PersistenceEngine.PH4)
compute_persistence_ph5 = _make_engine_func(PersistenceEngine.PH5)
compute_persistence_ph6 = _make_engine_func(PersistenceEngine.PH6)


def _resolve_engine_for_points(
    points: Any,
    engine: PersistenceEngine,
    _max_dim: int,
    device: str | None,
    mode: Any,
    backend: Any,
    _pytorch: Any,
) -> PersistenceEngine:
    """Resolve AUTO or user-specified engine to a concrete engine."""
    if engine != PersistenceEngine.AUTO:
        return engine
    n_points: int = 0
    point_dim: int = 3
    try:
        if _pytorch is not None and isinstance(points, _pytorch.Tensor):
            shape = points.shape
            n_points = shape[-2] if len(shape) >= 2 else 0
            point_dim = shape[-1] if len(shape) >= 2 else 0
        else:
            n_points = len(points) if hasattr(points, "__len__") else 0
            if hasattr(points, "shape") and len(points.shape) >= 2:
                point_dim = points.shape[-1]
    except (TypeError, IndexError, AttributeError):
        n_points = 0
        point_dim = 3
    return _auto_select_engine(n_points, point_dim, device, mode, backend)


def compute_persistence(
    points: PointCloud,
    options: PersistenceOptions | None = None,
    *,
    engine: PersistenceEngine | str = PersistenceEngine.AUTO,
    max_dim: int = 2,
    max_radius: float | None = None,
    mode: PersistenceMode = PersistenceMode.EXACT,
    backend: PersistenceBackend | None = None,
    threads: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    error_tolerance: float | None = None,
    dtype: str | None = None,
    max_radius_cap: float | None = None,
) -> PersistenceResult:
    """Compute persistent homology for a point cloud.

    By default uses cohomology-based computation (PH3--PH6 range)
    and auto-selects the best engine based on input size.
    Results are bitwise deterministic given the same input.

    Args:
        points: Point cloud as an (N, D) array (NumPy or PyTorch tensor).
        options: PersistenceOptions instance with preset configuration.
            When both ``options`` and individual keyword arguments (``max_dim``,
            ``max_radius``, etc.) are provided, the keyword arguments take
            precedence and override the corresponding fields in ``options``.
        engine: Persistence engine to use. Accepts a ``PersistenceEngine`` enum
            or its lowercase string name. One of:

            - ``PersistenceEngine.AUTO`` or ``"auto"`` (default) -- selects from
              PH0, PH3--PH6 based on point count, dimension, device, and mode.
            - ``PersistenceEngine.PH0`` or ``"ph0"`` -- standard homology.
              Best for <1K points or when exact historical compatibility is needed.
            - ``PersistenceEngine.PH3`` or ``"ph3"`` -- cohomology reduction.
              General persistent homology workloads.
            - ``PersistenceEngine.PH4`` or ``"ph4"`` -- cohomology + aggressive
              clearing. Higher-dimensional homology up to dimension 4.
            - ``PersistenceEngine.PH5`` or ``"ph5"`` -- unified adaptive engine.
              Recommended for 10K--1M points.
            - ``PersistenceEngine.PH6`` or ``"ph6"`` -- block-sparse speculative
              engine. Million-scale datasets and sparse filtrations.

        max_dim: Maximum homology dimension to compute (default: 2).
        max_radius: Filtration radius cutoff (default: auto-selected).
            Use a finite value matching your data scale (e.g. max pairwise distance).
        mode: PersistenceMode override (default: EXACT).
        backend: PersistenceBackend override (default: auto-selected).
        threads: Number of threads (0 = auto).
        device: Target device string ("cpu", "cuda", "cuda:N").
        seed: RNG seed for deterministic computation.
        error_tolerance: Approximation error tolerance (>0 enables APPROX mode).
        dtype: NumPy dtype name for point array conversion (e.g. ``"float32"``).
            By default inputs are converted to ``float64``.
        max_radius_cap: Override the upper bound for ``max_radius=inf``.
            By default uses the ``NERVE_MAX_RADIUS_CAP`` env var (1e15).

    Returns:
        A :class:`PersistenceResult` with the following fields:

        - **pairs** (list[tuple[float, float, int]]):
            Persistence pairs ``(birth, death, dimension)``. Infinite death
            indicates an essential class (never dies within the filtration).
        - **betti_numbers** (list[int]):
            Betti numbers ``[b0, b1, ..., b_{max_dim}]``.
        - **max_dim** (int):
            Maximum homology dimension computed.
        - **max_radius** (float):
            Filtration radius cutoff used.
        - **diagnostics** (dict):
            Optional backend-specific diagnostic information.

    Tip:
        To collect timing and memory diagnostics across multiple calls,
        wrap with :class:`pynerve.diagnostics.DiagnosticsCollector`::

            from pynerve.diagnostics import DiagnosticsCollector

            dc = DiagnosticsCollector()
            with dc.track("first_run"):
                r1 = pynerve.compute_persistence(points, max_dim=1)
            with dc.track("second_run"):
                r2 = pynerve.compute_persistence(points, max_dim=2)
            print(dc.report())

    Note:
        Common parameter combinations for typical use cases:

        **Quick exploration** (small data, <10K points):
        ``result = pynerve.compute_persistence(points, max_dim=2, max_radius=2.0)``

        **Large sparse data** (50K+ points):
        ``result = pynerve.compute_persistence(points, engine="ph5")``

        **GPU acceleration** (requires CUDA):
        ``result = pynerve.compute_persistence(points, max_dim=2, device="cuda")``

        **Approximate mode** (faster, lower memory):
        ``result = pynerve.compute_persistence(points, error_tolerance=0.01)``

        **Reproducible results**:
        ``result = pynerve.compute_persistence(points, seed=42)``

    Example:
        >>> import pynerve
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> points = rng.random((100, 3))
        >>> result = pynerve.compute_persistence(points, max_dim=2, max_radius=1.0)
        >>> result.betti_numbers  # doctest: +SKIP
        [1, 0, 0]
    """
    if isinstance(engine, str):
        engine_lower = engine.lower()
        try:
            engine = PersistenceEngine(engine_lower)
        except ValueError:
            valid = sorted(e.value for e in PersistenceEngine)
            raise ValueError(
                f"Unknown engine: {engine!r}. Valid engines (case-insensitive): {', '.join(valid)}"
            ) from None
    if backend is not None and device is not None:
        _warn_device_overrides_backend(device, backend)

    _, _, _PYTORCH = _nerve_state()  # noqa: N806
    resolved_engine = _resolve_engine_for_points(
        points, engine, max_dim, device, mode, backend, _PYTORCH
    )

    core = _require_core()
    core_func = _resolve_engine_func(
        core,
        resolved_engine,
        n_points=_estimate_n_points(points),
        dim=max_dim,
        device=device,
        mode=mode,
        backend=backend,
    )
    return _compute_with_options(
        core_func,
        points,
        options,
        max_dim=max_dim,
        max_radius=max_radius,
        mode=mode,
        backend=backend,
        threads=threads,
        device=device,
        seed=seed,
        error_tolerance=error_tolerance,
        dtype=dtype,
        max_radius_cap=max_radius_cap,
    )


def update_persistence(
    events: Iterable[tuple[EventType | str, Sequence[int]]],
    options: PersistenceOptions | None = None,
    *,
    engine: PersistenceEngine | str = PersistenceEngine.AUTO,
    max_dim: int = 2,
    max_radius: float | None = None,
    mode: PersistenceMode = PersistenceMode.EXACT,
    backend: PersistenceBackend | None = None,
    threads: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    error_tolerance: float | None = None,
    max_radius_cap: float | None = None,
) -> PersistenceResult:
    """Update persistence with incremental add/remove events.

    Args:
        events: Iterable of (event_type, simplex) pairs where event_type
            is ``EventType.ADD`` or ``EventType.REMOVE`` (or their string
            equivalents ``"add"`` / ``"remove"``).
        options: PersistenceOptions override.
        engine: Persistence engine to use (default: AUTO).
        max_dim: Maximum homology dimension to compute (default: 2).
        max_radius: Filtration radius cutoff (default: auto-selected).
        mode: PersistenceMode override (default: EXACT).
        backend: PersistenceBackend override (default: auto-selected).
        threads: Number of threads (0 = auto).
        device: Target device string ("cpu", "cuda", "cuda:N").
        seed: RNG seed for deterministic computation.
        error_tolerance: Approximation error tolerance.

    Returns:
        Updated PersistenceResult.

    Note:
        Each event is a ``(event_type, simplex)`` pair where:
        - ``event_type`` is ``EventType.ADD`` or ``"add"`` (insertion)
          or ``EventType.REMOVE`` or ``"remove"`` (deletion).
        - ``simplex`` is a list of vertex indices, e.g. ``(0, 1)`` for an edge
          or ``(0, 1, 2)`` for a triangle.

    Example:
        >>> import pynerve
        >>> from pynerve import EventType
        >>> events = [
        ...     (EventType.ADD, (0, 1)),     # add edge between vertices 0 and 1
        ...     (EventType.ADD, (1, 2)),     # add edge between vertices 1 and 2
        ...     (EventType.ADD, (0, 2)),     # add edge between vertices 0 and 2
        ...     (EventType.ADD, (0, 1, 2)),  # add triangle
        ... ]
        >>> result = pynerve.update_persistence(events, max_dim=2)
    """
    if isinstance(engine, str):
        engine_lower = engine.lower()
        try:
            engine = PersistenceEngine(engine_lower)
        except ValueError:
            valid = sorted(e.value for e in PersistenceEngine)
            raise ValueError(
                f"Unknown engine: {engine!r}. Valid engines (case-insensitive): {', '.join(valid)}"
            ) from None
    core = _require_core()
    resolved = _apply_option_overrides(
        _clone_options(options),
        max_dim=max_dim,
        max_radius=max_radius,
        mode=mode,
        backend=backend,
        threads=threads,
        device=device,
        seed=seed,
        error_tolerance=error_tolerance,
        max_radius_cap=max_radius_cap,
    )
    raw = core.update_persistence(_to_events_list(events), resolved)
    return PersistenceResult.from_dict(raw)
