"""Masked prediction task for persistence diagrams."""

from __future__ import annotations

from typing import cast

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._torch_diagrams import (
    encode_diagram_rows as _encode_diagram_rows,
)
from .._torch_diagrams import (
    encoder_output_dim as _encoder_output_dim,
)
from .._validation import validate_finite_scalar as _finite_scalar
from ._validation import _validate_ssl_diagram


class PersistencePredictionTask(nn.Module):
    """Masked prediction task for persistence diagrams.

    Randomly masks a subset of birth/death pairs and trains a
    prediction head to reconstruct the masked coordinates.
    """

    def __init__(self, encoder: nn.Module, mask_ratio: float = 0.15, prediction_dim: int = 2):
        """Initialise the masked prediction task.

        :param encoder: Encoder network mapping persistence diagrams
            to row-wise embeddings.
        :param mask_ratio: Fraction of pairs to mask, in ``(0, 1]``.
        :param prediction_dim: Dimensionality of predicted values
            (default 2 for birth, death).  Must be positive.
        :raises ValueError: If ``mask_ratio`` is not in ``(0, 1]``
            or ``prediction_dim`` is non-positive.
        """
        super().__init__()
        mask_ratio = _finite_scalar(mask_ratio, "mask_ratio")
        if not 0 < mask_ratio <= 1:
            raise ValueError("mask_ratio must be in (0, 1]")
        if prediction_dim <= 0:
            raise ValueError("prediction_dim must be positive")

        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.prediction_head = nn.Sequential(
            nn.Linear(_encoder_output_dim(encoder), 128),
            nn.ReLU(),
            nn.Linear(128, prediction_dim),
        )

    def forward(self, diagram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask pairs and predict their birth/death coordinates.

        :param diagram: Persistence diagram tensor of shape
            ``(N, >=2)``.
        :returns: A tuple ``(loss, predictions, targets)`` where
            ``loss`` is a scalar MSE, ``predictions`` has shape
            ``(M, prediction_dim)``, and ``targets`` has shape
            ``(M, 2)`` with the masked birth/death values.
        :raises ValueError: If ``diagram`` fails validation.
        """
        _validate_ssl_diagram(diagram)
        n_pairs = diagram.shape[0]
        if n_pairs == 0:
            pred_dim: int = cast(nn.Linear, self.prediction_head[-1]).out_features
            empty = diagram.new_empty((0, pred_dim))
            return diagram.new_zeros(()), empty, diagram.new_empty((0, 2))

        n_mask = max(1, int(n_pairs * self.mask_ratio))
        mask_indices = torch.randperm(n_pairs, device=diagram.device)[:n_mask]

        masked_diagram = diagram.clone()
        masked_diagram[mask_indices, :2] = 0
        row_embeddings = _encode_diagram_rows(self.encoder, masked_diagram)
        predictions = self.prediction_head(row_embeddings)[mask_indices]
        targets = diagram[mask_indices, :2]
        loss = F.mse_loss(predictions, targets)
        return loss, predictions, targets
