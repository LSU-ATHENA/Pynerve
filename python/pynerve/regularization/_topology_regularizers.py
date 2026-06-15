"""Topology-aware soft regularizers and constraints."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._torch_diagrams import birth_death as _birth_death
from .._torch_diagrams import validate_diagram as _validate_diagram
from .._validation import validate_finite_scalar as _finite_scalar


def _validate_birth_death_diagram(diagram: torch.Tensor, name: str = "diagram") -> None:
    _validate_diagram(diagram, min_cols=2, name=name)
    if diagram.shape[0] == 0:
        return
    if not torch.isfinite(diagram[:, :2]).all().item():
        raise ValueError(f"{name} birth/death coordinates must be finite")
    if not (diagram[:, 1] >= diagram[:, 0]).all().item():
        raise ValueError(f"{name} deaths must be greater than or equal to births")


def _validate_dimensional_diagram(diagram: torch.Tensor, name: str = "diagram") -> None:
    _validate_birth_death_diagram(diagram, name)
    if diagram.shape[0] == 0:
        return
    dimensions = diagram[:, 2]
    if not torch.isfinite(dimensions).all().item() or (dimensions < 0).any().item():
        raise ValueError(f"{name} dimensions must be finite and non-negative")
    if not torch.allclose(dimensions, dimensions.round()):
        raise ValueError(f"{name} dimensions must be integer-valued")


class MorseRegularizer(nn.Module):
    """Regularization encouraging Morse function properties."""

    def __init__(
        self,
        lambda_critical: float = 0.1,
        lambda_morse: float = 0.05,
        critical_threshold: float = 0.01,
    ):
        super().__init__()
        lambda_critical = _finite_scalar(lambda_critical, "lambda_critical")
        lambda_morse = _finite_scalar(lambda_morse, "lambda_morse")
        critical_threshold = _finite_scalar(critical_threshold, "critical_threshold")
        if lambda_critical < 0 or lambda_morse < 0:
            raise ValueError("regularization weights must be non-negative")
        if critical_threshold <= 0:
            raise ValueError("critical_threshold must be positive")

        self.lambda_critical = lambda_critical
        self.lambda_morse = lambda_morse
        self.threshold = critical_threshold

    def forward(
        self,
        function_values: torch.Tensor,
        gradient_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Morse regularization."""
        if not isinstance(function_values, torch.Tensor):
            raise TypeError("function_values must be a tensor")
        if function_values.numel() == 0:
            raise ValueError("function_values must be non-empty")
        if not torch.isfinite(function_values).all().item():
            raise ValueError("function_values must contain only finite values")

        loss = function_values.new_zeros(())

        if gradient_values is not None:
            if gradient_values.shape[:-1] != function_values.shape:
                raise ValueError("gradient_values must match function_values shape")
            if not torch.isfinite(gradient_values).all().item():
                raise ValueError("gradient_values must contain only finite values")
            grad_norms = gradient_values.norm(dim=-1)
            near_critical = grad_norms < self.threshold

            if near_critical.any():
                critical_count = near_critical.float().sum()
                loss += self.lambda_critical * critical_count

            loss += self.lambda_morse * F.relu(self.threshold - grad_norms).mean()

        return loss


class BettiConstraintLayer(nn.Module):
    """Layer that returns a soft Betti-number constraint loss."""

    def __init__(
        self,
        target_betti: list[int],
        persistence_fn: Callable[..., Any],
        lambda_constraint: float = 0.1,
    ):
        super().__init__()
        if not target_betti:
            raise ValueError("target_betti must be non-empty")
        if any(not math.isfinite(float(value)) or value < 0 for value in target_betti):
            raise ValueError("target_betti values must be non-negative")
        if not callable(persistence_fn):
            raise TypeError("persistence_fn must be callable")
        lambda_constraint = _finite_scalar(lambda_constraint, "lambda_constraint")
        if lambda_constraint < 0:
            raise ValueError("lambda_constraint must be non-negative")

        self.register_buffer("target_betti", torch.tensor(target_betti, dtype=torch.float32))
        self.target_betti: torch.Tensor
        self.persistence_fn = persistence_fn
        self.lambda_constraint = lambda_constraint

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input and its Betti constraint loss."""
        if isinstance(x, torch.Tensor) and x.numel() > 0 and not torch.isfinite(x).all().item():
            raise ValueError("x must contain only finite values")
        diagram = self.persistence_fn(x)
        pred_betti = self._compute_betti_numbers(diagram)
        loss = F.mse_loss(pred_betti, self.target_betti.to(pred_betti.device))

        return x, self.lambda_constraint * loss

    def _compute_betti_numbers(self, diagram: torch.Tensor) -> torch.Tensor:
        """Compute Betti numbers from diagram."""
        _validate_dimensional_diagram(diagram)

        if diagram.shape[0] == 0:
            return torch.zeros(
                len(self.target_betti),
                device=diagram.device,
                dtype=self.target_betti.dtype,
            )

        persistence = diagram[:, 1] - diagram[:, 0]
        significant = persistence > 0.01
        dimensions = diagram[:, 2].long()

        betti = torch.zeros(
            len(self.target_betti), device=diagram.device, dtype=self.target_betti.dtype
        )
        for d in range(len(self.target_betti)):
            mask = (dimensions == d) & significant
            betti[d] = mask.float().sum()

        return betti


class TopologicalSmoothness(nn.Module):
    """Regularization encouraging nearby diagrams to have nearby features."""

    def __init__(self, lambda_smooth: float = 0.1, neighborhood_size: int = 5):
        super().__init__()
        lambda_smooth = _finite_scalar(lambda_smooth, "lambda_smooth")
        if lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative")
        if neighborhood_size <= 0:
            raise ValueError("neighborhood_size must be positive")

        self.lambda_smooth = lambda_smooth
        self.neighborhood_size = neighborhood_size

    def forward(
        self, features: torch.Tensor, persistence_diagrams: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute topological smoothness regularization."""
        if features.dim() != 2:
            raise ValueError("features must have shape (n_samples, feature_dim)")
        if not torch.isfinite(features).all().item():
            raise ValueError("features must contain only finite values")
        n_samples = features.shape[0]
        if len(persistence_diagrams) != n_samples:
            raise ValueError("persistence_diagrams length must match feature rows")
        for diagram in persistence_diagrams:
            _validate_birth_death_diagram(diagram)
        if n_samples <= 1:
            return features.new_zeros(())

        diagram_dists = torch.zeros(n_samples, n_samples, device=features.device)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self._diagram_distance(persistence_diagrams[i], persistence_diagrams[j])
                dist = dist.to(device=features.device, dtype=features.dtype)
                diagram_dists[i, j] = dist
                diagram_dists[j, i] = dist

        smoothness_loss = features.new_zeros(())
        neighbor_count = min(self.neighborhood_size, n_samples - 1)

        for i in range(n_samples):
            row = diagram_dists[i].clone()
            row[i] = torch.inf
            _, neighbors = torch.topk(row, neighbor_count, largest=False)

            for j in neighbors:
                feature_diff = (features[i] - features[j]).norm()
                weight = torch.exp(-diagram_dists[i, j])
                smoothness_loss += weight * feature_diff

        return self.lambda_smooth * smoothness_loss / (n_samples * neighbor_count)

    def _diagram_distance(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """Compute a symmetric birth-death diagram distance."""
        _validate_birth_death_diagram(d1, "d1")
        _validate_birth_death_diagram(d2, "d2")

        if d1.shape[0] == 0 and d2.shape[0] == 0:
            return d1.new_zeros(())
        if d1.shape[0] == 0 or d2.shape[0] == 0:
            non_empty = d1 if d2.shape[0] == 0 else d2
            non_empty = non_empty.to(device=d1.device, dtype=d1.dtype)
            return (non_empty[:, 1] - non_empty[:, 0]).abs().sum() / 2**0.5

        d1_points = _birth_death(d1)
        d2_points = _birth_death(d2).to(device=d1_points.device, dtype=d1_points.dtype)
        pairwise = torch.cdist(d1_points, d2_points)
        diag1 = (d1[:, 1] - d1[:, 0]).abs() / 2**0.5
        diag2 = (d2[:, 1] - d2[:, 0]).abs().to(
            device=d1_points.device, dtype=d1_points.dtype
        ) / 2**0.5

        d1_cost = torch.minimum(pairwise.min(dim=1).values, diag1).mean()
        d2_cost = torch.minimum(pairwise.min(dim=0).values, diag2).mean()
        return 0.5 * (d1_cost + d2_cost)


class HomotopyRegularizer(nn.Module):
    """Regularizer encouraging homotopy-equivalent outputs."""

    def __init__(self, lambda_homotopy: float = 0.01):
        super().__init__()
        lambda_homotopy = _finite_scalar(lambda_homotopy, "lambda_homotopy")
        if lambda_homotopy < 0:
            raise ValueError("lambda_homotopy must be non-negative")
        self.lambda_homotopy = lambda_homotopy

    def forward(self, current_output: torch.Tensor, target_topology: torch.Tensor) -> torch.Tensor:
        """Compute homotopy preservation loss."""
        if current_output.shape != target_topology.shape:
            raise ValueError("current_output and target_topology must have same shape")
        if current_output.numel() == 0:
            raise ValueError("outputs must be non-empty")
        if (
            not torch.isfinite(current_output).all().item()
            or not torch.isfinite(target_topology).all().item()
        ):
            raise ValueError("outputs must contain only finite values")
        persistence_diff = (current_output - target_topology).abs().mean()
        return self.lambda_homotopy * persistence_diff
