"""BYOL-style self-supervised learning for persistence diagrams."""

from __future__ import annotations

import copy
from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._torch_diagrams import (
    encode_diagram_embedding as _encode_diagram_embedding,
)
from .._torch_diagrams import (
    encoder_output_dim as _encoder_output_dim,
)
from .._validation import validate_finite_scalar as _finite_scalar
from ._augmentation import TopologyAugmentation


class BYOLTopology(nn.Module):
    """BYOL-style self-supervised learning for persistence diagrams.

    Uses a momentum encoder (target network) alongside an online
    network with an extra predictor.  The online network is trained
    to predict target-network representations of augmented views.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        tau: float = 0.99,
    ):
        """Initialise BYOL topology module.

        :param encoder: Encoder network mapping persistence diagrams
            to fixed-size embeddings.
        :param projection_dim: Output dimension of projector and
            predictor.  Must be positive.
        :param hidden_dim: Hidden dimension of projector/predictor
            MLPs.  Must be positive.
        :param tau: Exponential moving average decay for the target
            network, in ``[0, 1]``.
        :raises ValueError: If ``projection_dim`` or ``hidden_dim``
            are non-positive, or ``tau`` is not in ``[0, 1]``.
        """
        super().__init__()
        tau = _finite_scalar(tau, "tau")
        if projection_dim <= 0 or hidden_dim <= 0:
            raise ValueError("projection_dim and hidden_dim must be positive")
        if not 0 <= tau <= 1:
            raise ValueError("tau must be in [0, 1]")

        encoder_dim = _encoder_output_dim(encoder)
        self.online_encoder = encoder
        self.online_projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.augmentation = TopologyAugmentation()
        self.tau = tau

        for module in (self.target_encoder, self.target_projector):
            for param in module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self) -> None:
        """Update target network via exponential moving average.

        :returns: ``None`` (in-place parameter update).
        """
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            target_params.data.mul_(self.tau).add_(online_params.data, alpha=1 - self.tau)

        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters(), strict=True
        ):
            target_params.data.mul_(self.tau).add_(online_params.data, alpha=1 - self.tau)

    def forward(self, diagrams: list[torch.Tensor]) -> torch.Tensor:
        """Compute symmetric regression loss over two augmented views.

        :param diagrams: Non-empty list of persistence diagram
            tensors.
        :returns: Scalar BYOL regression loss tensor.
        :raises ValueError: If ``diagrams`` is empty.
        """
        if not diagrams:
            raise ValueError("diagrams must be non-empty")

        views1 = [self.augmentation(d) for d in diagrams]
        views2 = [self.augmentation(d) for d in diagrams]

        online_q1 = self.online_predictor(self._encode_online(views1))
        with torch.no_grad():
            target_z2 = self._encode_target(views2)
        loss1 = self.regression_loss(online_q1, target_z2)

        online_q2 = self.online_predictor(self._encode_online(views2))
        with torch.no_grad():
            target_z1 = self._encode_target(views1)
        loss2 = self.regression_loss(online_q2, target_z1)
        return (loss1 + loss2) / 2

    def _encode_online(self, diagrams: list[torch.Tensor]) -> torch.Tensor:
        embeddings = [_encode_diagram_embedding(self.online_encoder, d) for d in diagrams]
        return cast(torch.Tensor, self.online_projector(torch.stack(embeddings)))

    @torch.no_grad()
    def _encode_target(self, diagrams: list[torch.Tensor]) -> torch.Tensor:
        embeddings = [_encode_diagram_embedding(self.target_encoder, d) for d in diagrams]
        return cast(torch.Tensor, self.target_projector(torch.stack(embeddings)))

    def regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cosine-similarity regression loss for BYOL.

        :param pred: Online-network predictions, shape ``(B, D)``.
        :param target: Target-network projections, shape ``(B, D)``.
        :returns: Scalar loss in ``[0, 4]``.  Value 0 means perfect
            alignment.
        :raises ValueError: If ``pred`` and ``target`` have
            different shapes.
        """
        if pred.shape != target.shape:
            raise ValueError("pred and target must have matching shapes")
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        return 2 - 2 * (pred_norm * target_norm).sum(dim=-1).mean()
