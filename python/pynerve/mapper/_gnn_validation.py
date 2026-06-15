"""Validation helpers for Mapper GNN layers."""

from __future__ import annotations

from math import isfinite

import torch


def _validate_probability(value: float, name: str) -> float:
    result = float(value)
    if result <= 0 or result > 1 or not isfinite(result):
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return result


def _validate_floating_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating-point dtype")
    if tensor.numel() > 0 and not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _validated_edges(edges: torch.Tensor, n_nodes: int, device: torch.device) -> torch.Tensor:
    if not isinstance(edges, torch.Tensor):
        raise TypeError("edges must be a tensor")
    if edges.dim() != 2 or edges.shape[0] != 2:
        raise ValueError("edges must have shape (2, n_edges)")
    if torch.is_floating_point(edges) or edges.dtype == torch.bool:
        raise TypeError("edges must contain integer node indices")
    edges = edges.to(device=device, dtype=torch.long)
    if edges.numel() > 0 and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError("edges contain node indices outside node_features")
    return edges
