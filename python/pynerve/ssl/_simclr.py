"""SimCLR-style contrastive learning for persistence diagrams."""

from __future__ import annotations

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


class SimCLRTopology(nn.Module):
    """SimCLR-style contrastive learning for persistence diagrams.

    Produces two augmented views of each diagram through
    :class:`TopologyAugmentation`, passes them through a shared
    encoder + projection MLP, and minimises the NT-Xent loss.
    """

    def __init__(self, encoder: nn.Module, projection_dim: int = 128, temperature: float = 0.5):
        """Initialise SimCLR topology module.

        :param encoder: Encoder network that maps a persistence
            diagram to a fixed-size embedding.
        :param projection_dim: Output dimension of the projection
            head.  Must be positive.
        :param temperature: Temperature parameter for NT-Xent loss.
            Must be positive.
        :raises ValueError: If ``projection_dim`` or ``temperature``
            are non-positive.
        """
        super().__init__()
        temperature = _finite_scalar(temperature, "temperature")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.encoder = encoder
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(_encoder_output_dim(encoder), 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )
        self.augmentation = TopologyAugmentation()

    def forward(self, diagrams: list[torch.Tensor]) -> torch.Tensor:
        """Compute NT-Xent loss over two augmented views.

        :param diagrams: Non-empty list of persistence diagram
            tensors.
        :returns: Scalar NT-Xent loss tensor.
        :raises ValueError: If ``diagrams`` is empty.
        """
        if not diagrams:
            raise ValueError("diagrams must be non-empty")

        z1 = self._project_diagrams([self.augmentation(d) for d in diagrams])
        z2 = self._project_diagrams([self.augmentation(d) for d in diagrams])
        return self.nt_xent_loss(F.normalize(z1, dim=1), F.normalize(z2, dim=1))

    def _project_diagrams(self, diagrams: list[torch.Tensor]) -> torch.Tensor:
        embeddings = [_encode_diagram_embedding(self.encoder, d) for d in diagrams]
        return cast(torch.Tensor, self.projection_head(torch.stack(embeddings)))

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Normalised temperature-scaled cross-entropy loss.

        :param z1: Normalised projection of the first view, shape
            ``(B, D)``.
        :param z2: Normalised projection of the second view, shape
            ``(B, D)``.
        :returns: Scalar loss tensor averaged over the batch.
        :raises ValueError: If ``z1`` and ``z2`` have different
            shapes or are not 2-D.
        """
        if z1.shape != z2.shape or z1.dim() != 2:
            raise ValueError("z1 and z2 must be 2D tensors with matching shapes")

        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, torch.finfo(sim_matrix.dtype).min)

        pos_sim = torch.cat(
            [torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)],
            dim=0,
        )
        return (-pos_sim + torch.logsumexp(sim_matrix, dim=1)).mean()
