"""Persistence-aware normalization layers."""

from __future__ import annotations

import torch
from torch import nn

from .._constants import EPS_1e_5
from .._validation import validate_finite_scalar as _finite_scalar
from .._validation import validate_finite_tensor as _validate_finite_tensor
from .._validation import validate_positive_int as _validate_positive_int


class PersistentBatchNorm(nn.Module):
    """Batch normalization with optional persistence-weighted statistics."""

    def __init__(self, num_features: int, persistence_weighted: bool = True, eps: float = EPS_1e_5):
        super().__init__()
        num_features = _validate_positive_int(num_features, "num_features")
        eps = _finite_scalar(eps, "eps")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_features = num_features
        self.persistence_weighted = persistence_weighted
        self.eps = eps

        self.gamma: torch.Tensor = nn.Parameter(torch.ones(num_features))
        self.beta: torch.Tensor = nn.Parameter(torch.zeros(num_features))
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(
        self,
        x: torch.Tensor,
        persistence_scores: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> torch.Tensor:
        """Apply persistence-aware batch normalization."""
        _validate_finite_tensor(x, "x")
        if x.dim() != 2 or x.shape[1] != self.num_features:
            raise ValueError("x must have shape (batch, num_features)")
        _validate_finite_tensor(self.gamma, "gamma")
        _validate_finite_tensor(self.beta, "beta")
        _validate_finite_tensor(self.running_mean, "running_mean")
        _validate_finite_tensor(self.running_var, "running_var")
        if (self.running_var < 0).any().item():
            raise ValueError("running_var must contain only non-negative values")

        active = self.training if training is None else training
        if active:
            if self.persistence_weighted and persistence_scores is not None:
                _validate_finite_tensor(persistence_scores, "persistence_scores")
                weights = persistence_scores
                if weights.dim() == 1:
                    if weights.shape[0] != x.shape[0]:
                        raise ValueError("persistence_scores must match batch size")
                    weights = weights.unsqueeze(1)
                elif weights.shape != x.shape:
                    raise ValueError("persistence_scores must match x shape")

                weights = weights.to(device=x.device, dtype=x.dtype)
                if (weights < 0).any().item():
                    raise ValueError("persistence_scores must contain only non-negative values")
                weights_sum = weights.sum(dim=0).clamp_min(self.eps)
                mean = (x * weights).sum(dim=0) / weights_sum

                var = ((x - mean) ** 2 * weights).sum(dim=0) / weights_sum
            else:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)

            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean.detach(), alpha=0.1)
                self.running_var.mul_(0.9).add_(var.detach(), alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
