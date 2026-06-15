"""Betti number prediction from persistence diagrams."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._torch_diagrams import (
    encode_diagram_embedding as _encode_embedding,
)
from .._torch_diagrams import (
    encoder_output_dim as _encoder_output_dim,
)
from ._validation import _validate_ssl_diagram


class BettiNumberPrediction(nn.Module):
    """Predicts Betti numbers from a persistence-diagram embedding.

    A lightweight prediction head that regresses the vector of Betti
    numbers (bet_0 through bet_{max_dim}) from the diagram's latent
    representation.  The output is passed through softplus to ensure
    non-negativity.
    """

    def __init__(self, encoder: nn.Module, max_dim: int = 3):
        """Initialise the Betti-number predictor.

        :param encoder: Module that encodes a persistence diagram into a
            fixed-size embedding vector.
        :param max_dim: Maximum homology dimension to predict (each
            dimension ``d`` in ``[0, max_dim]`` gets one output).
        :raises ValueError: If *max_dim* is negative.
        """
        super().__init__()
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")

        self.encoder = encoder
        self.max_dim = max_dim
        self.betti_predictor = nn.Sequential(
            nn.Linear(_encoder_output_dim(encoder), 128),
            nn.ReLU(),
            nn.Linear(128, max_dim + 1),
        )

    def forward(self, diagram: torch.Tensor) -> torch.Tensor:
        """Predict Betti numbers for a persistence diagram.

        :param diagram: Persistence-diagram tensor of shape
            ``(n_pairs, cols)``.
        :returns: Non-negative tensor of shape ``(max_dim + 1,)``
            containing predicted Betti numbers.
        """
        _validate_ssl_diagram(diagram)
        encoding = _encode_embedding(self.encoder, diagram)
        return F.softplus(self.betti_predictor(encoding))

    def compute_loss(self, diagram: torch.Tensor, target_betti: torch.Tensor) -> torch.Tensor:
        """Compute MSE between predicted and true Betti numbers.

        :param diagram: Persistence-diagram tensor of shape
            ``(n_pairs, cols)``.
        :param target_betti: Ground-truth Betti-number vector whose shape
            must match the prediction.
        :returns: Scalar MSE loss.
        :raises ValueError: If *target_betti* shape does not match the
            predicted shape.
        """
        pred_betti = self.forward(diagram)
        if pred_betti.shape != target_betti.shape:
            raise ValueError("target_betti must match predicted Betti shape")
        return F.mse_loss(pred_betti, target_betti.to(pred_betti))
