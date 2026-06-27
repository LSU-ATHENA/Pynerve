"""Persistent homology nn.Module with computation utilities."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from .._compute_api import PersistenceEngine, compute_persistence
from .._compute_core import PersistenceResult
from ..exceptions import InvalidArgumentError, ValidationError
from ._ph_autograd import _MEMORY_MODES, _REDUCTIONS, PersistentHomologyFunction

Device = str | torch.device | int


def _effective_max_radius(points: Tensor, max_radius: float) -> float:
    if math.isfinite(max_radius):
        return max_radius
    if points.shape[0] < 2:
        return 0.0
    radius_points = points.detach().to(dtype=torch.float64)
    distances = torch.cdist(radius_points, radius_points)
    finite = distances[torch.isfinite(distances)]
    return float(finite.max().item()) if finite.numel() else 0.0


def _core_compute_result(
    points: Tensor,
    max_dim: int,
    max_radius: float,
    reduction: str,
) -> PersistenceResult:
    pts = points.detach().cpu().numpy()
    radius = _effective_max_radius(points, max_radius)
    if reduction == "cohomology":
        return compute_persistence(
            pts, max_dim=max_dim, max_radius=radius, engine=PersistenceEngine.PH4
        )
    return compute_persistence(pts, max_dim=max_dim, max_radius=radius)


def _result_to_diagram_tensors(
    result: PersistenceResult,
    max_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[Tensor]:
    dim_pairs: list[list[list[float]]] = [[] for _ in range(max_dim + 1)]
    for birth, death, dim in result.pairs:
        dim_index = int(dim)
        if 0 <= dim_index <= max_dim:
            dim_pairs[dim_index].append([float(birth), float(death)])

    diagrams: list[Tensor] = []
    for pairs in dim_pairs:
        if pairs:
            diagrams.append(torch.tensor(pairs, dtype=dtype, device=device))
        else:
            diagrams.append(torch.empty((0, 2), dtype=dtype, device=device))
    return diagrams


class PersistentHomology(nn.Module):
    """Persistent homology module backed by the compiled C++/Rust core.

    Accepts batched point clouds ``(batch, N, D)`` and returns a list of
    persistence-diagram tensors (one per homology dimension). Supports
    multiple reduction strategies and memory modes.
    """

    def __init__(
        self,
        max_dim: int = 1,
        max_radius: float = float("inf"),
        metric: str = "euclidean",
        reduction: str = "clearing",
        memory_mode: Literal[
            "standard", "memory_mapped", "streaming", "extreme"
        ] = "standard",
        max_memory_gb: float | None = None,
        device: Device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialise the persistent homology module.

        :param max_dim: Maximum homology dimension (default: ``1``).
            Must be non-negative.
        :param max_radius: Filtration radius cutoff (default: ``inf``).
            Must be positive or infinite.
        :param metric: Distance metric (default: ``"euclidean"``).
            Currently only ``"euclidean"`` is supported.
        :param reduction: Reduction strategy (default: ``"clearing"``).
            One of ``"standard"``, ``"clearing"``, ``"cohomology"``.
        :param memory_mode: Memory management mode
            (default: ``"standard"``). One of ``"standard"``,
            ``"memory_mapped"``, ``"streaming"``, ``"extreme"``.
        :param max_memory_gb: Memory budget in gigabytes for the extreme
            memory mode (default: ``None``).
        :param device: Target device (default: ``None``, uses CPU).
        :param dtype: Target floating-point dtype
            (default: ``None``, uses ``float32``).
        :raises InvalidArgumentError: If *max_dim* is negative, *max_radius*
            is non-positive/nan, *metric* is not ``"euclidean"``,
            *reduction* or *memory_mode* is unrecognised, or
            *max_memory_gb* is non-positive.
        """
        super().__init__()
        max_radius = float(max_radius)
        if max_dim < 0:
            raise InvalidArgumentError(
                "max_dim must be non-negative",
                parameter="max_dim",
                expected=">= 0",
                actual=str(max_dim),
            )
        if math.isnan(max_radius) or max_radius <= 0:
            raise InvalidArgumentError(
                "max_radius must be positive or infinite",
                parameter="max_radius",
                expected="positive or infinite",
                actual=str(max_radius),
            )
        if metric != "euclidean":
            raise InvalidArgumentError(
                "this build only supports euclidean metric",
                parameter="metric",
                expected="euclidean",
                actual=metric,
            )
        if reduction not in _REDUCTIONS:
            raise InvalidArgumentError(
                f"reduction must be one of {sorted(_REDUCTIONS)}",
                parameter="reduction",
                actual=reduction,
            )
        if memory_mode not in _MEMORY_MODES:
            raise InvalidArgumentError(
                f"memory_mode must be one of {sorted(_MEMORY_MODES)}",
                parameter="memory_mode",
                actual=memory_mode,
            )
        if max_memory_gb is not None and max_memory_gb <= 0:
            raise InvalidArgumentError(
                "max_memory_gb must be positive",
                parameter="max_memory_gb",
                expected="> 0",
                actual=str(max_memory_gb),
            )

        self.max_dim = max_dim
        self.max_radius = max_radius
        self.metric = metric
        self.reduction = reduction
        self.memory_mode = memory_mode
        self.max_memory_gb = max_memory_gb

        self._device = torch.device("cpu")
        self._dtype = torch.float32

        if device is not None:
            self.to(device)
        if dtype is not None:
            self._dtype = dtype

        self._memory_optimizer: Any | None = None

    def forward(self, points: Tensor) -> list[Tensor]:
        """Compute persistence diagrams for a batched point cloud."""
        points = points.to(device=self._device, dtype=self._dtype)

        if points.dim() != 3:
            raise ValidationError(
                f"Expected 3D input (batch, n_points, dim), got {points.dim()}D",
                parameter="points",
                expected="3",
                actual=str(points.dim()),
            )
        if points.shape[0] == 0 or points.shape[1] == 0 or points.shape[2] == 0:
            raise ValidationError(
                "points must have non-empty batch, point, and feature dimensions",
                parameter="points",
                expected="(batch>0, n_points>0, dim>0)",
                actual=str(tuple(points.shape)),
            )
        if not torch.isfinite(points).all():
            raise ValidationError(
                "points must contain only finite values",
                parameter="points",
            )

        if self.memory_mode == "extreme":
            return self._forward_extreme_memory(points)

        use_gpu = self._device.type == "cuda"

        return cast(
            list[Tensor],
            list(
                PersistentHomologyFunction.apply(  # pyright: ignore[reportArgumentType]
                    points,
                    self.max_dim,
                    self.max_radius,
                    self.metric,
                    self.reduction,
                    use_gpu,
                    self.memory_mode,
                )
            ),
        )

    def _forward_extreme_memory(self, points: Tensor) -> list[Tensor]:
        _, n_points, dim = points.shape

        estimated_gb = self._estimate_memory_gb(n_points, dim)
        max_memory = self.max_memory_gb or 4.0

        if estimated_gb <= max_memory:
            return self._forward_standard(points)

        raise RuntimeError(
            "memory budget is too small for exact persistent homology; "
            "increase max_memory_gb or use a core streaming backend"
        )

    def _forward_standard(self, points: Tensor) -> list[Tensor]:
        use_gpu = self._device.type == "cuda"

        return cast(
            list[Tensor],
            list(
                PersistentHomologyFunction.apply(  # pyright: ignore[reportArgumentType]
                    points,
                    self.max_dim,
                    self.max_radius,
                    self.metric,
                    self.reduction,
                    use_gpu,
                    "standard",
                )
            ),
        )

    def _estimate_memory_gb(self, n_points: int, dim: int) -> float:
        n_simplices = n_points * (n_points - 1) // 2
        bytes_per_entry = 8
        matrix_bytes = n_simplices * 10 * bytes_per_entry
        coordinate_bytes = n_points * dim * bytes_per_entry
        matrix_bytes += coordinate_bytes
        return matrix_bytes / (1024**3)

    def to(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> PersistentHomology:
        """Move module to a different device and/or dtype.

        :param args: Positional arguments forwarded to
            :meth:`torch.nn.Module.to`. Accepts a device string,
            :class:`torch.device`, :class:`torch.dtype`, or
            :class:`torch.Tensor` (to match its device/dtype).
        :param kwargs: Keyword arguments; ``device`` and ``dtype``
            are recognised explicitly.
        :returns: ``self`` so that calls can be chained.
        """
        device: torch.device | None = kwargs.get("device")
        dtype: torch.dtype | None = kwargs.get("dtype")
        if not device and args:
            first = args[0]
            if isinstance(first, torch.Tensor):
                if dtype is None:
                    dtype = first.dtype
                self._device = first.device
            elif isinstance(first, (str, torch.device)):
                self._device = torch.device(first)
            else:
                self._device = torch.device(first)
        elif device is not None:
            if isinstance(device, torch.Tensor):
                if dtype is None:
                    dtype = device.dtype
                self._device = device.device
            else:
                self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        return self

    def cuda(self, device: int | torch.device | None = None) -> PersistentHomology:
        """Move module to a CUDA device.

        :param device: CUDA device index or :class:`torch.device`
            (default: ``None``, uses ``cuda:0`` -- the current CUDA
            device).
        :returns: ``self`` so that calls can be chained.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        if device is None:
            self._device = torch.device("cuda")
        elif isinstance(device, int):
            self._device = torch.device(f"cuda:{device}")
        else:
            self._device = device
        return self

    def cpu(self) -> PersistentHomology:
        """Move module to CPU."""
        self._device = torch.device("cpu")
        return self

    @property
    def device(self) -> torch.device:
        """:returns: The :class:`torch.device` the module resides on."""
        return self._device

    def half(self) -> PersistentHomology:
        """Set floating-point dtype to ``float16``.

        :returns: ``self`` so that calls can be chained.
        """
        self._dtype = torch.float16
        return self

    def float(self) -> PersistentHomology:
        """Set floating-point dtype to ``float32``.

        :returns: ``self`` so that calls can be chained.
        """
        self._dtype = torch.float32
        return self

    def double(self) -> PersistentHomology:
        """Set floating-point dtype to ``float64``.

        :returns: ``self`` so that calls can be chained.
        """
        self._dtype = torch.float64
        return self

    @property
    def dtype(self) -> torch.dtype:
        """:returns: The floating-point :class:`torch.dtype` used
        by the module."""
        return self._dtype

    def train(self, mode: bool = True) -> PersistentHomology:
        """Set training mode.

        :param mode: ``True`` for training, ``False`` for evaluation
            (default: ``True``).
        :returns: ``self`` so that calls can be chained.
        """
        super().train(mode)
        return self

    def eval(self) -> PersistentHomology:
        """Set evaluation mode."""
        super().eval()
        return self

    def extra_repr(self) -> str:
        """Return a human-readable description of the module.

        :returns: String summarising the module configuration.
        """
        return (
            f"max_dim={self.max_dim}, "
            f"max_radius={self.max_radius:.3f}, "
            f"metric={self.metric!r}, "
            f"reduction={self.reduction!r}, "
            f"memory_mode={self.memory_mode!r}, "
            f"device={self._device}, "
            f"dtype={self._dtype}"
        )
