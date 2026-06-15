"""Topology-aware dropout layers."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._constants import EPS
from .._validation import validate_finite_scalar as _finite_scalar
from .._validation import validate_finite_tensor as _validate_finite_tensor


def _validate_probability(name: str, value: float) -> float:
    parsed = _finite_scalar(value, name)
    if not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1")
    return parsed


class PersistentDropout(nn.Module):
    """Dropout that keeps high-persistence features with higher probability."""

    def __init__(self, p: float = 0.5, persistence_aware: bool = True, temperature: float = 0.1):
        super().__init__()
        p = _validate_probability("p", p)
        temperature = _finite_scalar(temperature, "temperature")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.p = p
        self.persistence_aware = persistence_aware
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        persistence_scores: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> torch.Tensor:
        """Apply persistence-aware dropout."""
        active = self.training if training is None else training
        if not active or self.p == 0:
            return x
        _validate_finite_tensor(x, "x")

        if self.persistence_aware and persistence_scores is not None:
            if x.dim() != 2:
                raise ValueError("persistence-aware dropout expects a 2D tensor")
            _validate_finite_tensor(persistence_scores, "persistence_scores")
            if persistence_scores.dim() == 1:
                if persistence_scores.shape[0] != x.shape[1]:
                    raise ValueError("persistence_scores must match feature count")
                scores = persistence_scores.unsqueeze(0).expand(x.shape[0], -1)
            else:
                if persistence_scores.shape != x.shape:
                    raise ValueError("persistence_scores must match x shape")
                scores = persistence_scores

            importance = torch.sigmoid(scores.to(x.device, x.dtype) / self.temperature)
            keep_prob = (1 - self.p) + self.p * importance

            mask = torch.bernoulli(keep_prob)
            scale = keep_prob.clamp_min(EPS).reciprocal()

            return x * mask * scale

        return F.dropout(x, p=self.p, training=active)


class TopologyPreservingDropout(nn.Module):
    """Dropout that can preserve graph connectivity proxies."""

    def __init__(
        self,
        p: float = 0.3,
        betti_preserve: bool = True,
        connectivity_preserve: bool = True,
    ):
        super().__init__()
        p = _validate_probability("p", p)

        self.p = p
        self.betti_preserve = betti_preserve
        self.connectivity_preserve = connectivity_preserve

    def forward(
        self,
        activations: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> torch.Tensor:
        """Apply topology-preserving dropout."""
        active = self.training if training is None else training
        if not active or self.p == 0:
            return activations
        _validate_finite_tensor(activations, "activations")

        if adjacency is not None and self.connectivity_preserve:
            return self._connectivity_preserving_dropout(activations, adjacency)

        return F.dropout(activations, p=self.p, training=active)

    def _connectivity_preserving_dropout(
        self, activations: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Dropout that preserves graph connectivity."""
        if activations.dim() < 2:
            raise ValueError("activations must have shape (n_nodes, features)")
        n_nodes = activations.shape[0]
        if adjacency.shape != (n_nodes, n_nodes):
            raise ValueError("adjacency must have shape (n_nodes, n_nodes)")
        _validate_finite_tensor(adjacency, "adjacency")
        if adjacency.numel() > 0 and (adjacency < 0).any().item():
            raise ValueError("adjacency must contain only non-negative values")

        n_drop = int(n_nodes * self.p)
        if n_drop == 0:
            return activations

        adjacency = adjacency.to(device=activations.device, dtype=activations.dtype)
        degrees = adjacency.sum(dim=1)
        drop_probs = 1.0 / (degrees + 1)
        drop_probs = drop_probs / drop_probs.sum()
        drop_indices = torch.multinomial(drop_probs, n_drop, replacement=False)
        mask = torch.ones(n_nodes, device=activations.device, dtype=activations.dtype)
        mask[drop_indices] = 0

        return activations * mask.unsqueeze(-1)
