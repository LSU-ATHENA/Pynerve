"""TensorBoard integration for persistence-diagram summaries."""

from __future__ import annotations

from typing import Any, cast

import torch

from . import viz
from ._diagram import PersistenceDiagram


def log_diagram(
    writer: Any,
    diagram: PersistenceDiagram | torch.Tensor,
    step: int,
    tag: str = "diagram",
    method: str = "image",
    dim: int | None = None,
) -> None:
    """Log a persistence diagram to TensorBoard."""
    if method not in {"image", "heatmap", "scatter"}:
        raise ValueError("method must be one of: image, heatmap, scatter")

    if method == "image":
        img: Any = viz.diagram_to_image_data(diagram, resolution=(50, 50))
        if img.dim() == 2:
            img = img.unsqueeze(0)

        writer.add_image(tag, img, step)

    elif method == "heatmap":
        heatmap_data: Any = viz.diagram_to_heatmap_data(diagram, grid_size=30)
        grid = torch.as_tensor(heatmap_data["grid"], dtype=torch.float32)
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)

        writer.add_image(f"{tag}/heatmap", grid, step)

    elif method == "scatter":
        scatter_data: Any = viz.diagram_to_scatter_data(diagram, dim=dim if dim is not None else 0)
        for name in ("births", "deaths", "persistence"):
            values = torch.as_tensor(scatter_data[name], dtype=torch.float32)
            if values.numel() > 0:
                writer.add_histogram(f"{tag}/{name}", values, step)
        writer.add_scalar(f"{tag}/num_features", len(scatter_data["births"]), step)


def log_landscape(
    writer: Any,
    diagram: PersistenceDiagram | torch.Tensor,
    step: int,
    tag: str = "landscape",
    k: int = 3,
) -> None:
    """Log persistence landscape samples as scalar series."""
    if k <= 0:
        raise ValueError("k must be positive")

    data: Any = viz.diagram_to_landscape_data(diagram, k=k, num_samples=100)
    landscapes = data["landscapes"]  # [k, num_samples]
    x_values = data["x_values"]

    for i in range(min(k, landscapes.shape[0])):
        for j, (_, y) in enumerate(zip(x_values, landscapes[i], strict=True)):
            writer.add_scalar(f"{tag}/lambda_{i}", float(y), step * 100 + j)


def log_betti_curve(
    writer: Any,
    diagram: PersistenceDiagram | torch.Tensor,
    step: int,
    tag: str = "betti",
    num_samples: int = 50,
    dim: int | None = None,
) -> None:
    """Log a Betti curve as scalar samples."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    data: Any = viz.diagram_to_betti_data(
        diagram, num_samples=num_samples, dim=dim if dim is not None else 0
    )
    thresholds = data["thresholds"]
    betti_numbers = data["betti_numbers"]

    dim_tag = f"_dim{dim}" if dim is not None else ""
    for i, (_t, b) in enumerate(zip(thresholds, betti_numbers, strict=True)):
        writer.add_scalar(f"{tag}{dim_tag}", float(b), step * num_samples + i)


def log_statistics(
    writer: Any,
    diagram: PersistenceDiagram | torch.Tensor,
    step: int,
    tag: str = "diagram_stats",
    dims: list[int] | None = None,
) -> None:
    """Log scalar persistence-diagram statistics."""
    from . import statistics  # noqa: PLC0415

    if dims is None:
        dims = [-1]  # Aggregate

    tensor_diagram = cast(torch.Tensor, diagram)

    for dim in dims:
        dim_str = f"_dim{dim}" if dim >= 0 else ""

        total = statistics.total_persistence(tensor_diagram, dim=dim if dim >= 0 else None, p=1.0)
        writer.add_scalar(f"{tag}{dim_str}/total_persistence", total.item(), step)

        mean = statistics.mean_persistence(tensor_diagram, dim=dim if dim >= 0 else None)
        writer.add_scalar(f"{tag}{dim_str}/mean_persistence", mean.item(), step)

        max_p = statistics.max_persistence(tensor_diagram, dim=dim if dim >= 0 else None)
        writer.add_scalar(f"{tag}{dim_str}/max_persistence", max_p.item(), step)

        num = statistics.number_of_features(tensor_diagram, dim=dim if dim >= 0 else None)
        writer.add_scalar(f"{tag}{dim_str}/num_features", num.item(), step)

        entropy = statistics.persistence_entropy(tensor_diagram, dim=dim if dim >= 0 else None)
        writer.add_scalar(f"{tag}{dim_str}/entropy", entropy.item(), step)


class DiagramSummaryWriter:
    """SummaryWriter wrapper with persistence-diagram logging methods."""

    def __init__(self, log_dir: str | None = None, comment: str = "", **kwargs: Any) -> None:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        self._writer: Any = SummaryWriter(log_dir=log_dir, comment=comment, **kwargs)

    def add_diagram(
        self,
        diagram: PersistenceDiagram | torch.Tensor,
        global_step: int | None = None,
        tag: str = "diagram",
        method: str = "image",
    ) -> None:
        """Log persistence diagram."""
        log_diagram(
            self._writer, diagram, global_step if global_step is not None else 0, tag, method
        )

    def add_landscape(
        self,
        diagram: PersistenceDiagram | torch.Tensor,
        global_step: int | None = None,
        tag: str = "landscape",
        k: int = 3,
    ) -> None:
        """Log persistence landscape."""
        log_landscape(self._writer, diagram, global_step if global_step is not None else 0, tag, k)

    def add_betti_curve(
        self,
        diagram: PersistenceDiagram | torch.Tensor,
        global_step: int | None = None,
        tag: str = "betti",
        num_samples: int = 50,
        dim: int | None = None,
    ) -> None:
        """Log Betti curve."""
        log_betti_curve(
            self._writer,
            diagram,
            global_step if global_step is not None else 0,
            tag,
            num_samples,
            dim,
        )

    def add_diagram_stats(
        self,
        diagram: PersistenceDiagram | torch.Tensor,
        global_step: int | None = None,
        tag: str = "diagram_stats",
        dims: list[int] | None = None,
    ) -> None:
        """Log diagram statistics."""
        log_statistics(
            self._writer, diagram, global_step if global_step is not None else 0, tag, dims
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)

    def __enter__(self) -> DiagramSummaryWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._writer.close()


__all__ = [
    "log_diagram",
    "log_landscape",
    "log_betti_curve",
    "log_statistics",
    "DiagramSummaryWriter",
]
